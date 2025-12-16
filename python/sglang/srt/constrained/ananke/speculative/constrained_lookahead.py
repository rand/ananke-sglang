# Copyright Rand Arete @ Ananke 2025
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""CDSL-style constrained decoding with speculative lookaheads.

This module implements the CDSL (Constrained Decoding with Speculative
Lookaheads) algorithm for efficient constrained generation.

Key Algorithm (from CDSL paper, NAACL 2025):
1. Generate K draft tokens using relaxed constraints (syntax-only)
2. Verify draft sequence against full domain stack
3. Accept longest valid prefix
4. Repeat from the next position

Adaptive Lookahead:
    The lookahead length is adjusted based on acceptance rate:
    - High acceptance rate -> increase lookahead (more speculation)
    - Low acceptance rate -> decrease lookahead (less waste)

Expected Speedup: 2-4x for typical code generation
- Draft generation: ~1ms (small model or greedy)
- Verification: ~2-5ms (batched constraint checking)
- vs Sequential: ~10-50ms per token

References:
    - CDSL: Constrained Decoding with Speculative Lookaheads (NAACL 2025)
    - Test-Time Compute Scaling (DeepMind, 2024)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

import torch

from .draft_model import DraftModel, DraftContext, DraftResult, NullDraftModel

logger = logging.getLogger(__name__)


@dataclass
class LookaheadConfig:
    """Configuration for constrained lookahead.

    Attributes:
        initial_lookahead: Initial number of tokens to look ahead
        min_lookahead: Minimum lookahead length
        max_lookahead: Maximum lookahead length
        acceptance_threshold_high: Increase lookahead if acceptance rate above this
        acceptance_threshold_low: Decrease lookahead if acceptance rate below this
        lookahead_step: Amount to increase/decrease lookahead
        adaptive: Whether to adapt lookahead based on acceptance rate
    """

    initial_lookahead: int = 5
    min_lookahead: int = 2
    max_lookahead: int = 15
    acceptance_threshold_high: float = 0.8
    acceptance_threshold_low: float = 0.3
    lookahead_step: int = 2
    adaptive: bool = True


@dataclass
class VerificationResult:
    """Result of verifying a draft sequence.

    Attributes:
        num_valid: Number of valid tokens from the draft
        total_draft: Total number of draft tokens
        rejection_reason: Reason for rejection (if any)
        latency_ms: Verification latency
    """

    num_valid: int
    total_draft: int
    rejection_reason: Optional[str] = None
    latency_ms: float = 0.0

    @property
    def acceptance_rate(self) -> float:
        """Calculate acceptance rate."""
        if self.total_draft == 0:
            return 0.0
        return self.num_valid / self.total_draft

    @property
    def fully_accepted(self) -> bool:
        """Check if all draft tokens were accepted."""
        return self.num_valid == self.total_draft and self.total_draft > 0


@dataclass
class LookaheadStats:
    """Statistics for constrained lookahead.

    Attributes:
        total_drafts: Total number of draft sequences generated
        total_tokens_drafted: Total tokens generated in drafts
        total_tokens_accepted: Total tokens accepted from drafts
        current_lookahead: Current lookahead length
        avg_acceptance_rate: Rolling average acceptance rate
        total_draft_latency_ms: Total time spent generating drafts
        total_verify_latency_ms: Total time spent verifying drafts
    """

    total_drafts: int = 0
    total_tokens_drafted: int = 0
    total_tokens_accepted: int = 0
    current_lookahead: int = 5
    avg_acceptance_rate: float = 0.0
    total_draft_latency_ms: float = 0.0
    total_verify_latency_ms: float = 0.0

    @property
    def overall_acceptance_rate(self) -> float:
        """Calculate overall acceptance rate."""
        if self.total_tokens_drafted == 0:
            return 0.0
        return self.total_tokens_accepted / self.total_tokens_drafted

    @property
    def speedup_estimate(self) -> float:
        """Estimate speedup from speculative decoding.

        Assumes:
        - Sequential token generation takes ~20ms/token
        - Draft generation + verification takes ~5ms total
        """
        if self.total_tokens_accepted == 0:
            return 1.0

        # Time if sequential (20ms per token)
        sequential_time = self.total_tokens_accepted * 20.0

        # Actual time spent
        actual_time = self.total_draft_latency_ms + self.total_verify_latency_ms

        if actual_time == 0:
            return 1.0

        return sequential_time / actual_time


class ConstraintVerifier(Protocol):
    """Protocol for constraint verification.

    Verifiers check if a sequence of draft tokens is valid under
    the full constraint system.
    """

    def verify_draft_tokens(
        self,
        draft_tokens: List[int],
    ) -> Tuple[int, Optional[Any]]:
        """Verify draft tokens against constraints.

        Args:
            draft_tokens: List of draft token IDs

        Returns:
            (num_valid, rejection_constraint) tuple
        """
        ...


class ConstrainedLookahead:
    """CDSL-style constrained decoding with speculative lookaheads.

    This class implements the core CDSL algorithm for efficient constrained
    generation. It combines a draft model for fast token generation with
    full constraint verification.

    Algorithm:
    1. Generate K draft tokens using draft model (fast, relaxed constraints)
    2. Verify draft against full constraint system
    3. Accept longest valid prefix
    4. Adapt lookahead based on acceptance rate

    Attributes:
        draft_model: Model for generating draft tokens
        verifier: Constraint verifier for checking drafts
        config: Lookahead configuration
        stats: Statistics about lookahead performance

    Example:
        >>> lookahead = ConstrainedLookahead(
        ...     draft_model=my_draft_model,
        ...     verifier=grammar,
        ...     config=LookaheadConfig(initial_lookahead=5),
        ... )
        >>> tokens = lookahead.generate_next(context)
        >>> print(f"Generated {len(tokens)} tokens")
    """

    def __init__(
        self,
        draft_model: DraftModel,
        verifier: ConstraintVerifier,
        config: Optional[LookaheadConfig] = None,
    ):
        """Initialize constrained lookahead.

        Args:
            draft_model: Model for generating drafts
            verifier: Constraint verifier
            config: Lookahead configuration (uses defaults if None)
        """
        self.draft_model = draft_model
        self.verifier = verifier
        self.config = config or LookaheadConfig()
        self.stats = LookaheadStats(current_lookahead=self.config.initial_lookahead)

        # Track recent acceptance rates for adaptation
        self._recent_rates: List[float] = []
        self._rate_window = 10

    def generate_next(
        self,
        context: DraftContext,
    ) -> List[int]:
        """Generate next batch of tokens using speculative lookahead.

        Generates a draft sequence and verifies it, returning the longest
        valid prefix.

        Args:
            context: Current generation context

        Returns:
            List of accepted tokens (may be empty if draft rejected)
        """
        # Generate draft
        draft_start = time.perf_counter()
        draft_result = self.draft_model.generate_draft(
            context,
            lookahead_length=self.stats.current_lookahead,
        )
        draft_latency = (time.perf_counter() - draft_start) * 1000
        self.stats.total_draft_latency_ms += draft_latency

        if not draft_result.tokens:
            return []

        # Verify draft
        verify_start = time.perf_counter()
        num_valid, rejection = self.verifier.verify_draft_tokens(draft_result.tokens)
        verify_latency = (time.perf_counter() - verify_start) * 1000
        self.stats.total_verify_latency_ms += verify_latency

        # Update statistics
        self.stats.total_drafts += 1
        self.stats.total_tokens_drafted += len(draft_result.tokens)
        self.stats.total_tokens_accepted += num_valid

        # Track acceptance rate
        acceptance_rate = num_valid / len(draft_result.tokens) if draft_result.tokens else 0.0
        self._update_acceptance_rate(acceptance_rate)

        # Adapt lookahead if configured
        if self.config.adaptive:
            self._adapt_lookahead()

        # Return accepted prefix
        return draft_result.tokens[:num_valid]

    def generate_batch(
        self,
        contexts: List[DraftContext],
    ) -> List[List[int]]:
        """Generate tokens for multiple contexts in batch.

        Generates drafts for all contexts and verifies them in parallel
        (where possible) for improved throughput.

        Args:
            contexts: List of generation contexts

        Returns:
            List of accepted token lists, one per context
        """
        results: List[List[int]] = []

        # Generate all drafts first
        draft_results: List[DraftResult] = []
        draft_start = time.perf_counter()
        for context in contexts:
            draft_result = self.draft_model.generate_draft(
                context,
                lookahead_length=self.stats.current_lookahead,
            )
            draft_results.append(draft_result)
        draft_latency = (time.perf_counter() - draft_start) * 1000
        self.stats.total_draft_latency_ms += draft_latency

        # Verify all drafts
        verify_start = time.perf_counter()
        for draft_result in draft_results:
            if not draft_result.tokens:
                results.append([])
                continue

            num_valid, _ = self.verifier.verify_draft_tokens(draft_result.tokens)

            # Update statistics
            self.stats.total_drafts += 1
            self.stats.total_tokens_drafted += len(draft_result.tokens)
            self.stats.total_tokens_accepted += num_valid

            results.append(draft_result.tokens[:num_valid])

        verify_latency = (time.perf_counter() - verify_start) * 1000
        self.stats.total_verify_latency_ms += verify_latency

        # Update acceptance rate
        total_drafted = sum(len(dr.tokens) for dr in draft_results if dr.tokens)
        total_accepted = sum(len(r) for r in results)
        if total_drafted > 0:
            self._update_acceptance_rate(total_accepted / total_drafted)

        if self.config.adaptive:
            self._adapt_lookahead()

        return results

    def _update_acceptance_rate(self, rate: float) -> None:
        """Update rolling acceptance rate."""
        self._recent_rates.append(rate)
        if len(self._recent_rates) > self._rate_window:
            self._recent_rates.pop(0)

        self.stats.avg_acceptance_rate = (
            sum(self._recent_rates) / len(self._recent_rates)
            if self._recent_rates else 0.0
        )

    def _adapt_lookahead(self) -> None:
        """Adapt lookahead length based on acceptance rate."""
        if len(self._recent_rates) < 3:
            return  # Need some history

        avg_rate = self.stats.avg_acceptance_rate

        if avg_rate > self.config.acceptance_threshold_high:
            # High acceptance rate -> increase lookahead
            new_lookahead = min(
                self.stats.current_lookahead + self.config.lookahead_step,
                self.config.max_lookahead,
            )
            if new_lookahead != self.stats.current_lookahead:
                logger.debug(
                    f"Increasing lookahead {self.stats.current_lookahead} -> {new_lookahead} "
                    f"(acceptance rate: {avg_rate:.2f})"
                )
                self.stats.current_lookahead = new_lookahead

        elif avg_rate < self.config.acceptance_threshold_low:
            # Low acceptance rate -> decrease lookahead
            new_lookahead = max(
                self.stats.current_lookahead - self.config.lookahead_step,
                self.config.min_lookahead,
            )
            if new_lookahead != self.stats.current_lookahead:
                logger.debug(
                    f"Decreasing lookahead {self.stats.current_lookahead} -> {new_lookahead} "
                    f"(acceptance rate: {avg_rate:.2f})"
                )
                self.stats.current_lookahead = new_lookahead

    def reset_stats(self) -> None:
        """Reset statistics."""
        self.stats = LookaheadStats(current_lookahead=self.config.initial_lookahead)
        self._recent_rates.clear()

    def get_stats(self) -> LookaheadStats:
        """Get current statistics."""
        return self.stats


class ParallelConstrainedLookahead:
    """Parallel constrained lookahead for multiple draft sequences.

    Generates multiple draft sequences in parallel and selects the best
    one based on verification results. This is useful when:
    - The draft model is probabilistic (sampling)
    - We want to explore multiple paths

    Attributes:
        draft_model: Model for generating drafts
        verifier: Constraint verifier
        num_sequences: Number of parallel sequences to generate
        config: Lookahead configuration
    """

    def __init__(
        self,
        draft_model: DraftModel,
        verifier: ConstraintVerifier,
        num_sequences: int = 3,
        config: Optional[LookaheadConfig] = None,
    ):
        """Initialize parallel constrained lookahead.

        Args:
            draft_model: Model for generating drafts
            verifier: Constraint verifier
            num_sequences: Number of parallel sequences
            config: Lookahead configuration
        """
        self.draft_model = draft_model
        self.verifier = verifier
        self.num_sequences = num_sequences
        self.config = config or LookaheadConfig()
        self._lookahead = ConstrainedLookahead(draft_model, verifier, config)

    def generate_next(
        self,
        context: DraftContext,
    ) -> List[int]:
        """Generate next batch of tokens using parallel speculation.

        Generates multiple draft sequences and returns the one with
        the longest valid prefix.

        Args:
            context: Current generation context

        Returns:
            List of accepted tokens from best sequence
        """
        # Generate multiple drafts
        drafts: List[DraftResult] = []
        for _ in range(self.num_sequences):
            draft = self.draft_model.generate_draft(
                context,
                lookahead_length=self._lookahead.stats.current_lookahead,
            )
            drafts.append(draft)

        # Verify all drafts and select best
        best_tokens: List[int] = []
        best_valid = 0

        for draft in drafts:
            if not draft.tokens:
                continue

            num_valid, _ = self.verifier.verify_draft_tokens(draft.tokens)

            if num_valid > best_valid:
                best_valid = num_valid
                best_tokens = draft.tokens[:num_valid]

        # Update stats (using the best result)
        self._lookahead.stats.total_drafts += 1
        self._lookahead.stats.total_tokens_drafted += max(len(d.tokens) for d in drafts) if drafts else 0
        self._lookahead.stats.total_tokens_accepted += best_valid

        return best_tokens

    def get_stats(self) -> LookaheadStats:
        """Get statistics from underlying lookahead."""
        return self._lookahead.get_stats()


def create_constrained_lookahead(
    draft_model: DraftModel,
    verifier: ConstraintVerifier,
    parallel: bool = False,
    num_sequences: int = 3,
    **config_kwargs: Any,
) -> ConstrainedLookahead:
    """Factory function to create constrained lookahead.

    Args:
        draft_model: Model for generating drafts
        verifier: Constraint verifier
        parallel: Whether to use parallel lookahead
        num_sequences: Number of parallel sequences (if parallel=True)
        **config_kwargs: Configuration parameters

    Returns:
        ConstrainedLookahead or ParallelConstrainedLookahead instance
    """
    config = LookaheadConfig(**config_kwargs) if config_kwargs else None

    if parallel:
        return ParallelConstrainedLookahead(
            draft_model,
            verifier,
            num_sequences=num_sequences,
            config=config,
        )  # type: ignore
    else:
        return ConstrainedLookahead(
            draft_model,
            verifier,
            config=config,
        )
