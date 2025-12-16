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
"""Beam search for constrained code generation.

This module implements beam search for code generation with Ananke constraints.
Beam search explores multiple constraint-satisfying paths simultaneously,
selecting the best candidates based on constraint verification scores.

Key Features:
1. Maintains K candidate sequences (beams)
2. Scores candidates via constraint verification
3. Prunes to top-K after each token
4. Diversity penalty to prevent beam collapse
5. Integration with speculative decoding

Design Principles:
1. Quality over speed: Explore multiple paths for better output
2. Constraint-aware scoring: Use domain verification for ranking
3. Diversity: Prevent identical beams
4. Efficient pruning: Keep only promising candidates

References:
    - Test-Time Compute Scaling (DeepMind, 2024)
    - Beam Search with Constraints (NeurIPS 2020)
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from heapq import heappush, heappop, nlargest
from typing import Any, Callable, Dict, Generic, List, Optional, Protocol, Tuple, TypeVar

import torch

logger = logging.getLogger(__name__)

# Type variable for state
S = TypeVar("S")


@dataclass(order=True)
class BeamCandidate(Generic[S]):
    """A candidate sequence in beam search.

    Candidates are ordered by score (negated for min-heap).

    Attributes:
        score: Combined score (constraint + language model)
        tokens: Token sequence
        state: Associated state (e.g., grammar state)
        log_prob: Log probability from language model
        constraint_score: Score from constraint verification
        diversity_penalty: Penalty for similarity to other beams
        finished: Whether generation is complete
        metadata: Additional candidate information
    """

    score: float
    tokens: List[int] = field(compare=False)
    state: Optional[S] = field(compare=False, default=None)
    log_prob: float = field(compare=False, default=0.0)
    constraint_score: float = field(compare=False, default=1.0)
    diversity_penalty: float = field(compare=False, default=0.0)
    finished: bool = field(compare=False, default=False)
    metadata: Dict[str, Any] = field(compare=False, default_factory=dict)

    @property
    def length(self) -> int:
        """Number of tokens in sequence."""
        return len(self.tokens)

    def extend(
        self,
        token: int,
        log_prob: float,
        constraint_score: float,
        state: Optional[S] = None,
        finished: bool = False,
    ) -> BeamCandidate[S]:
        """Create extended candidate with new token.

        Args:
            token: Token to append
            log_prob: Log probability of token
            constraint_score: Constraint score after token
            state: New state after token
            finished: Whether this completes generation

        Returns:
            New BeamCandidate with token appended
        """
        new_log_prob = self.log_prob + log_prob
        new_constraint = (self.constraint_score + constraint_score) / 2

        # Combined score (weighted geometric mean)
        combined_score = (
            0.7 * new_constraint +
            0.3 * math.exp(new_log_prob / max(len(self.tokens) + 1, 1))
        ) - self.diversity_penalty

        return BeamCandidate(
            score=-combined_score,  # Negated for min-heap
            tokens=self.tokens + [token],
            state=state,
            log_prob=new_log_prob,
            constraint_score=new_constraint,
            diversity_penalty=self.diversity_penalty,
            finished=finished,
        )


@dataclass
class BeamSearchConfig:
    """Configuration for beam search.

    Attributes:
        beam_width: Number of candidates to maintain
        max_length: Maximum sequence length
        length_penalty: Penalty/bonus for sequence length
        diversity_penalty: Penalty for similar sequences
        early_stopping: Stop when best beam is finished
        constraint_weight: Weight for constraint scores vs LM scores
        temperature: Sampling temperature for candidate expansion
    """

    beam_width: int = 5
    max_length: int = 512
    length_penalty: float = 0.6
    diversity_penalty: float = 0.5
    early_stopping: bool = True
    constraint_weight: float = 0.7
    temperature: float = 1.0


@dataclass
class BeamSearchStats:
    """Statistics for beam search.

    Attributes:
        total_steps: Number of expansion steps
        candidates_evaluated: Total candidates considered
        candidates_pruned: Candidates removed due to pruning
        finished_beams: Number of beams that completed
        avg_constraint_score: Average constraint score of final beams
        best_score: Best final score
        latency_ms: Total search time
    """

    total_steps: int = 0
    candidates_evaluated: int = 0
    candidates_pruned: int = 0
    finished_beams: int = 0
    avg_constraint_score: float = 0.0
    best_score: float = 0.0
    latency_ms: float = 0.0


class TokenScorer(Protocol):
    """Protocol for scoring next tokens.

    Implementations provide log probabilities and constraint scores
    for candidate next tokens.
    """

    def score_tokens(
        self,
        tokens: List[int],
        state: Any,
        top_k: int = 50,
    ) -> List[Tuple[int, float, float]]:
        """Score candidate next tokens.

        Args:
            tokens: Current token sequence
            state: Current state
            top_k: Number of top candidates to return

        Returns:
            List of (token_id, log_prob, constraint_score) tuples
        """
        ...


class ConstraintScorer(Protocol):
    """Protocol for constraint verification scoring."""

    def score(self, tokens: List[int], state: Any) -> float:
        """Score sequence against constraints.

        Args:
            tokens: Token sequence
            state: Current state

        Returns:
            Score from 0.0 to 1.0
        """
        ...


class BeamSearch(Generic[S]):
    """Beam search for constrained code generation.

    Maintains multiple candidate sequences and prunes to top-K after
    each expansion step. Uses constraint verification for scoring.

    Algorithm:
    1. Initialize beam with start token
    2. For each step:
       a. Expand each beam with top-K next tokens
       b. Score expansions via constraint verification
       c. Apply diversity penalty
       d. Prune to top beam_width candidates
    3. Return best finished sequence (or best partial if none finished)

    Attributes:
        config: Search configuration
        token_scorer: Scorer for next tokens
        constraint_scorer: Optional additional constraint scorer
        stats: Search statistics

    Example:
        >>> search = BeamSearch(
        ...     config=BeamSearchConfig(beam_width=5),
        ...     token_scorer=my_scorer,
        ... )
        >>> result = search.search(start_tokens=[0], initial_state=state)
        >>> print(f"Best sequence: {result.tokens}")
    """

    def __init__(
        self,
        config: Optional[BeamSearchConfig] = None,
        token_scorer: Optional[TokenScorer] = None,
        constraint_scorer: Optional[ConstraintScorer] = None,
    ):
        """Initialize beam search.

        Args:
            config: Search configuration
            token_scorer: Scorer for next tokens
            constraint_scorer: Optional constraint scorer
        """
        self.config = config or BeamSearchConfig()
        self.token_scorer = token_scorer
        self.constraint_scorer = constraint_scorer
        self.stats = BeamSearchStats()
        self._last_all_candidates: List[BeamCandidate[S]] = []  # Track all candidates from last search

    def search(
        self,
        start_tokens: List[int],
        initial_state: Optional[S] = None,
        end_token: Optional[int] = None,
    ) -> BeamCandidate[S]:
        """Run beam search.

        Args:
            start_tokens: Initial token sequence
            initial_state: Initial state
            end_token: Token ID that marks end of generation

        Returns:
            Best BeamCandidate found
        """
        start_time = time.perf_counter()
        self.stats = BeamSearchStats()

        if self.token_scorer is None:
            raise ValueError("token_scorer is required")

        # Initialize beam
        initial = BeamCandidate(
            score=0.0,
            tokens=start_tokens.copy(),
            state=initial_state,
            log_prob=0.0,
            constraint_score=1.0,
        )
        beams: List[BeamCandidate[S]] = [initial]
        finished: List[BeamCandidate[S]] = []

        # Search loop
        for step in range(self.config.max_length):
            self.stats.total_steps += 1

            if not beams:
                break

            # Expand all beams
            candidates: List[BeamCandidate[S]] = []

            for beam in beams:
                if beam.finished:
                    finished.append(beam)
                    continue

                # Get scored next tokens
                scored_tokens = self.token_scorer.score_tokens(
                    beam.tokens,
                    beam.state,
                    top_k=self.config.beam_width * 2,  # Extra candidates for diversity
                )
                self.stats.candidates_evaluated += len(scored_tokens)

                for token_id, log_prob, constraint_score in scored_tokens:
                    # Apply temperature
                    if self.config.temperature != 1.0:
                        log_prob = log_prob / self.config.temperature

                    # Check if finished
                    is_finished = (end_token is not None and token_id == end_token)

                    # Create extended candidate
                    new_candidate = beam.extend(
                        token=token_id,
                        log_prob=log_prob,
                        constraint_score=constraint_score,
                        state=beam.state,  # State updated by scorer
                        finished=is_finished,
                    )

                    # Apply additional constraint scoring
                    if self.constraint_scorer is not None:
                        extra_score = self.constraint_scorer.score(
                            new_candidate.tokens,
                            new_candidate.state,
                        )
                        new_candidate.constraint_score = (
                            new_candidate.constraint_score + extra_score
                        ) / 2

                    candidates.append(new_candidate)

            # Apply diversity penalty
            candidates = self._apply_diversity_penalty(candidates)

            # Prune to top-K
            original_count = len(candidates)
            candidates = self._prune_candidates(candidates)
            self.stats.candidates_pruned += original_count - len(candidates)

            # Separate finished and active beams
            beams = [c for c in candidates if not c.finished]
            finished.extend(c for c in candidates if c.finished)

            # Early stopping
            if self.config.early_stopping and finished:
                best_finished = min(finished, key=lambda c: c.score)
                best_active = min(beams, key=lambda c: c.score) if beams else None

                if best_active is None or best_finished.score <= best_active.score:
                    break

        # Select best result
        all_candidates = finished + beams
        if not all_candidates:
            # Fallback to initial
            result = initial
            all_candidates = [initial]
        else:
            result = min(all_candidates, key=lambda c: c.score)

        # Store all candidates for multi-result access
        self._last_all_candidates = sorted(all_candidates, key=lambda c: c.score)

        # Update stats
        self.stats.finished_beams = len(finished)
        self.stats.best_score = -result.score  # Un-negate
        if all_candidates:
            self.stats.avg_constraint_score = sum(
                c.constraint_score for c in all_candidates
            ) / len(all_candidates)
        self.stats.latency_ms = (time.perf_counter() - start_time) * 1000

        return result

    def _apply_diversity_penalty(
        self,
        candidates: List[BeamCandidate[S]],
    ) -> List[BeamCandidate[S]]:
        """Apply diversity penalty to prevent beam collapse.

        Penalizes candidates that are too similar to higher-ranked candidates.

        Args:
            candidates: List of candidates

        Returns:
            Candidates with diversity penalty applied
        """
        if self.config.diversity_penalty == 0:
            return candidates

        # Sort by score
        sorted_candidates = sorted(candidates, key=lambda c: c.score)

        # Apply penalty based on similarity to previous candidates
        for i, candidate in enumerate(sorted_candidates):
            penalty = 0.0
            for j in range(i):
                prior = sorted_candidates[j]
                # Simple similarity: shared prefix length
                shared = 0
                for a, b in zip(candidate.tokens, prior.tokens):
                    if a == b:
                        shared += 1
                    else:
                        break

                # Penalty proportional to shared prefix
                if candidate.length > 0:
                    similarity = shared / candidate.length
                    penalty += similarity * self.config.diversity_penalty

            candidate.diversity_penalty = penalty
            # Update score with penalty
            candidate.score += penalty

        return sorted_candidates

    def _prune_candidates(
        self,
        candidates: List[BeamCandidate[S]],
    ) -> List[BeamCandidate[S]]:
        """Prune candidates to beam width.

        Args:
            candidates: List of candidates

        Returns:
            Top beam_width candidates
        """
        if len(candidates) <= self.config.beam_width:
            return candidates

        # Use nlargest with key to get top candidates (smallest scores since negated)
        return nlargest(
            self.config.beam_width,
            candidates,
            key=lambda c: -c.score,  # Negate back for largest
        )

    def search_with_constraint(
        self,
        start_tokens: List[int],
        initial_state: Optional[S] = None,
        constraint_fn: Optional[Callable[[List[int]], float]] = None,
        end_token: Optional[int] = None,
    ) -> List[BeamCandidate[S]]:
        """Run beam search with custom constraint function.

        Returns multiple candidates sorted by score.

        Args:
            start_tokens: Initial token sequence
            initial_state: Initial state
            constraint_fn: Custom constraint scoring function
            end_token: End token ID

        Returns:
            List of top candidates
        """
        # Create a simple scorer wrapper
        class ConstraintWrapper:
            def __init__(self, fn: Callable[[List[int]], float]):
                self.fn = fn

            def score(self, tokens: List[int], state: Any) -> float:
                return self.fn(tokens)

        if constraint_fn is not None:
            self.constraint_scorer = ConstraintWrapper(constraint_fn)  # type: ignore

        # Run search
        self.search(start_tokens, initial_state, end_token)

        # Return all candidates sorted by score (best first)
        return self._last_all_candidates[:self.config.beam_width]

    def get_stats(self) -> BeamSearchStats:
        """Get search statistics."""
        return self.stats

    def get_all_candidates(self) -> List[BeamCandidate[S]]:
        """Get all candidates from the last search.

        Returns candidates sorted by score (best first).

        Returns:
            List of all candidates from last search
        """
        return self._last_all_candidates.copy()


class SimpleTokenScorer:
    """Simple token scorer using logits and mask.

    Scores tokens based on softmax probabilities and a constraint mask.

    Attributes:
        logits_fn: Function that returns logits for next token
        mask_fn: Function that returns constraint mask
        vocab_size: Vocabulary size
    """

    def __init__(
        self,
        logits_fn: Callable[[List[int]], torch.Tensor],
        mask_fn: Callable[[List[int], Any], torch.Tensor],
        vocab_size: int = 32000,
    ):
        """Initialize token scorer.

        Args:
            logits_fn: Function(tokens) -> logits tensor
            mask_fn: Function(tokens, state) -> mask tensor
            vocab_size: Vocabulary size
        """
        self._logits_fn = logits_fn
        self._mask_fn = mask_fn
        self._vocab_size = vocab_size

    def score_tokens(
        self,
        tokens: List[int],
        state: Any,
        top_k: int = 50,
    ) -> List[Tuple[int, float, float]]:
        """Score candidate next tokens.

        Args:
            tokens: Current token sequence
            state: Current state
            top_k: Number of candidates to return

        Returns:
            List of (token_id, log_prob, constraint_score) tuples
        """
        # Get logits
        logits = self._logits_fn(tokens)

        # Get constraint mask
        mask = self._mask_fn(tokens, state)

        # Apply mask
        logits = logits.clone()
        logits[~mask] = float("-inf")

        # Compute log probabilities
        log_probs = torch.log_softmax(logits, dim=-1)

        # Get top-k
        top_values, top_indices = torch.topk(log_probs, min(top_k, mask.sum().item()))

        results: List[Tuple[int, float, float]] = []
        for idx, log_prob in zip(top_indices.tolist(), top_values.tolist()):
            # Constraint score based on whether token is in mask
            constraint_score = 1.0 if mask[idx] else 0.0
            results.append((idx, log_prob, constraint_score))

        return results


def create_beam_search(
    token_scorer: Optional[TokenScorer] = None,
    constraint_scorer: Optional[ConstraintScorer] = None,
    **config_kwargs: Any,
) -> BeamSearch:
    """Factory function to create beam search.

    Args:
        token_scorer: Token scorer
        constraint_scorer: Constraint scorer
        **config_kwargs: Configuration parameters

    Returns:
        BeamSearch instance
    """
    config = BeamSearchConfig(**config_kwargs) if config_kwargs else None
    return BeamSearch(
        config=config,
        token_scorer=token_scorer,
        constraint_scorer=constraint_scorer,
    )
