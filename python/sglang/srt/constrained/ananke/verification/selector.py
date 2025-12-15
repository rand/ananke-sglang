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
"""Best-of-N selector using constraint verification.

This module implements the BestOfNSelector which generates multiple candidate
code completions and selects the best one based on Ananke constraint verification.

Key Insight (from Test-Time Compute Research):
    Instead of constraining every token, we can:
    1. Generate N candidates unconstrained (faster per-token)
    2. Verify each candidate against constraints (batch verification)
    3. Select best by score (quality improvement)

This trades increased total compute for:
- Zero per-token constraint overhead during generation
- Parallelizable candidate generation
- Better worst-case quality (multiple chances to get it right)
- Graceful degradation (return best of imperfect options)

Selection Strategies:
- BEST_SCORE: Select candidate with highest overall_score
- FIRST_VALID: Select first candidate that passes all constraints
- THRESHOLD: Select first candidate above a score threshold
- WEIGHTED: Use domain-weighted scoring with custom weights

References:
- BoNBoN Alignment: Best-of-N Sampling (NeurIPS 2024)
- Inference-Aware Fine-Tuning (December 2024)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

from .verifier import ConstraintVerifier, VerificationResult

logger = logging.getLogger(__name__)


class SelectionStrategy(Enum):
    """Strategy for selecting the best candidate.

    BEST_SCORE: Select the candidate with the highest overall_score.
        Always runs full verification on all candidates.

    FIRST_VALID: Select the first candidate that passes all constraints.
        Stops early when a valid candidate is found (faster).

    THRESHOLD: Select the first candidate with score >= threshold.
        Configurable quality threshold for early stopping.

    WEIGHTED: Use custom domain weights for scoring.
        Allows prioritizing specific domains (e.g., types over imports).

    ENSEMBLE: Combine multiple candidates via voting or merging.
        Future enhancement for complex generation tasks.
    """

    BEST_SCORE = auto()
    FIRST_VALID = auto()
    THRESHOLD = auto()
    WEIGHTED = auto()
    ENSEMBLE = auto()


@dataclass
class SelectionResult:
    """Result of Best-of-N selection.

    Attributes:
        selected: The selected candidate code string
        selected_index: Index of selected candidate in original list
        selected_result: Full verification result for selected candidate
        all_results: Verification results for all candidates
        num_candidates: Total number of candidates evaluated
        num_valid: Number of candidates that passed all constraints
        strategy: Selection strategy used
        latency_ns: Total selection time (nanoseconds)

    Example:
        >>> result = selector.select_best(candidates)
        >>> print(f"Selected candidate {result.selected_index} with score {result.selected_result.overall_score}")
    """

    selected: str
    selected_index: int
    selected_result: VerificationResult
    all_results: List[VerificationResult] = field(default_factory=list)
    num_candidates: int = 0
    num_valid: int = 0
    strategy: SelectionStrategy = SelectionStrategy.BEST_SCORE
    latency_ns: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "selected": self.selected,
            "selected_index": self.selected_index,
            "selected_result": self.selected_result.to_dict(),
            "num_candidates": self.num_candidates,
            "num_valid": self.num_valid,
            "strategy": self.strategy.name,
            "latency_ns": self.latency_ns,
        }


class BestOfNSelector:
    """Selects the best candidate from N generated completions.

    The selector uses Ananke constraint verification to evaluate and
    rank multiple candidate completions, selecting the best one based
    on the configured selection strategy.

    This implements a key test-time compute technique: trading increased
    total compute for improved quality by generating multiple candidates
    and selecting the best.

    Attributes:
        verifier: ConstraintVerifier instance for verification
        strategy: Selection strategy to use
        threshold: Score threshold for THRESHOLD strategy
        domain_weights: Custom weights for WEIGHTED strategy
        return_all_results: Whether to include all results in output

    Example:
        >>> selector = BestOfNSelector(language="python")
        >>> candidates = [
        ...     "def add(a, b): return a + b",
        ...     "def add(a, b) return a + b",  # Invalid syntax
        ...     "def add(a: int, b: int) -> int: return a + b",  # Best
        ... ]
        >>> result = selector.select_best(candidates)
        >>> result.selected_index
        2
    """

    def __init__(
        self,
        language: str = "python",
        verifier: Optional[ConstraintVerifier] = None,
        strategy: SelectionStrategy = SelectionStrategy.BEST_SCORE,
        threshold: float = 0.8,
        domain_weights: Optional[Dict[str, float]] = None,
        enabled_domains: Optional[set] = None,
        return_all_results: bool = True,
        type_context: Optional[Dict[str, str]] = None,
        import_context: Optional[List[str]] = None,
    ):
        """Initialize BestOfNSelector.

        Args:
            language: Target programming language
            verifier: Custom verifier (created if not provided)
            strategy: Selection strategy
            threshold: Score threshold for THRESHOLD strategy
            domain_weights: Custom domain weights
            enabled_domains: Domains to verify against
            return_all_results: Include all verification results in output
            type_context: Type bindings for verification
            import_context: Available imports for verification
        """
        self.language = language
        self.strategy = strategy
        self.threshold = threshold
        self.domain_weights = domain_weights
        self.return_all_results = return_all_results

        # Initialize verifier
        self.verifier = verifier or ConstraintVerifier(
            language=language,
            enabled_domains=enabled_domains,
            domain_weights=domain_weights,
            type_context=type_context,
            import_context=import_context,
        )

        # Statistics
        self._stats = {
            "selections": 0,
            "total_candidates": 0,
            "avg_valid_rate": 0.0,
            "avg_best_score": 0.0,
        }

    def select_best(
        self,
        candidates: List[str],
        strategy: Optional[SelectionStrategy] = None,
    ) -> SelectionResult:
        """Select the best candidate from the list.

        Verifies all candidates against constraints and selects the best
        based on the configured or provided selection strategy.

        Args:
            candidates: List of candidate code strings
            strategy: Override selection strategy for this call

        Returns:
            SelectionResult with selected candidate and metadata

        Raises:
            ValueError: If candidates list is empty

        Example:
            >>> result = selector.select_best(["def f():", "def f(): pass"])
            >>> result.selected
            'def f(): pass'
        """
        import time

        if not candidates:
            raise ValueError("Cannot select from empty candidate list")

        start_time = time.perf_counter_ns()
        strategy = strategy or self.strategy

        # Dispatch to strategy-specific selection
        if strategy == SelectionStrategy.FIRST_VALID:
            result = self._select_first_valid(candidates)
        elif strategy == SelectionStrategy.THRESHOLD:
            result = self._select_threshold(candidates)
        elif strategy == SelectionStrategy.WEIGHTED:
            result = self._select_weighted(candidates)
        else:  # BEST_SCORE (default)
            result = self._select_best_score(candidates)

        end_time = time.perf_counter_ns()
        result.latency_ns = end_time - start_time
        result.strategy = strategy

        # Update statistics
        self._update_stats(result)

        return result

    def _select_best_score(self, candidates: List[str]) -> SelectionResult:
        """Select candidate with highest overall_score.

        Verifies all candidates and returns the one with the best score.
        """
        # Verify all candidates
        all_results = self.verifier.verify_batch(candidates)

        # Find best by score
        best_idx = 0
        best_score = -1.0
        num_valid = 0

        for i, result in enumerate(all_results):
            if result.valid:
                num_valid += 1
            if result.overall_score > best_score:
                best_score = result.overall_score
                best_idx = i

        return SelectionResult(
            selected=candidates[best_idx],
            selected_index=best_idx,
            selected_result=all_results[best_idx],
            all_results=all_results if self.return_all_results else [],
            num_candidates=len(candidates),
            num_valid=num_valid,
        )

    def _select_first_valid(self, candidates: List[str]) -> SelectionResult:
        """Select first candidate that passes all constraints.

        Stops verification early when a valid candidate is found.
        Falls back to best score if no candidate is valid.
        """
        all_results: List[VerificationResult] = []
        first_valid_idx: Optional[int] = None

        for i, code in enumerate(candidates):
            result = self.verifier.verify(code)
            all_results.append(result)

            if result.valid and first_valid_idx is None:
                first_valid_idx = i
                if not self.return_all_results:
                    # Early termination
                    break

        if first_valid_idx is not None:
            selected_idx = first_valid_idx
        else:
            # No valid candidate - fall back to best score
            selected_idx = max(
                range(len(all_results)),
                key=lambda i: all_results[i].overall_score,
            )

        num_valid = sum(1 for r in all_results if r.valid)

        return SelectionResult(
            selected=candidates[selected_idx],
            selected_index=selected_idx,
            selected_result=all_results[selected_idx],
            all_results=all_results if self.return_all_results else [],
            num_candidates=len(candidates),
            num_valid=num_valid,
        )

    def _select_threshold(self, candidates: List[str]) -> SelectionResult:
        """Select first candidate with score >= threshold.

        Stops verification early when a sufficiently good candidate is found.
        Falls back to best score if no candidate meets threshold.
        """
        all_results: List[VerificationResult] = []
        threshold_idx: Optional[int] = None

        for i, code in enumerate(candidates):
            result = self.verifier.verify(code)
            all_results.append(result)

            if result.overall_score >= self.threshold and threshold_idx is None:
                threshold_idx = i
                if not self.return_all_results:
                    break

        if threshold_idx is not None:
            selected_idx = threshold_idx
        else:
            # No candidate meets threshold - select best
            selected_idx = max(
                range(len(all_results)),
                key=lambda i: all_results[i].overall_score,
            )

        num_valid = sum(1 for r in all_results if r.valid)

        return SelectionResult(
            selected=candidates[selected_idx],
            selected_index=selected_idx,
            selected_result=all_results[selected_idx],
            all_results=all_results if self.return_all_results else [],
            num_candidates=len(candidates),
            num_valid=num_valid,
        )

    def _select_weighted(self, candidates: List[str]) -> SelectionResult:
        """Select candidate using custom domain weights.

        Uses domain_weights to compute a custom-weighted score for each
        candidate, then selects the one with highest weighted score.
        """
        # Verify all candidates
        all_results = self.verifier.verify_batch(candidates)

        # Compute custom weighted scores
        def weighted_score(result: VerificationResult) -> float:
            if not self.domain_weights:
                return result.overall_score

            total_weight = 0.0
            weighted_sum = 0.0
            for name, domain_score in result.domain_scores.items():
                weight = self.domain_weights.get(name, 0.5)
                weighted_sum += domain_score.score * weight
                total_weight += weight

            return weighted_sum / total_weight if total_weight > 0 else 0.0

        # Find best by weighted score
        best_idx = max(range(len(all_results)), key=lambda i: weighted_score(all_results[i]))
        num_valid = sum(1 for r in all_results if r.valid)

        return SelectionResult(
            selected=candidates[best_idx],
            selected_index=best_idx,
            selected_result=all_results[best_idx],
            all_results=all_results if self.return_all_results else [],
            num_candidates=len(candidates),
            num_valid=num_valid,
        )

    def _update_stats(self, result: SelectionResult) -> None:
        """Update selection statistics."""
        self._stats["selections"] += 1
        self._stats["total_candidates"] += result.num_candidates

        # Rolling average valid rate
        n = self._stats["selections"]
        valid_rate = result.num_valid / result.num_candidates if result.num_candidates > 0 else 0
        self._stats["avg_valid_rate"] = (
            (self._stats["avg_valid_rate"] * (n - 1) + valid_rate) / n
        )

        # Rolling average best score
        self._stats["avg_best_score"] = (
            (self._stats["avg_best_score"] * (n - 1) + result.selected_result.overall_score) / n
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get selection statistics."""
        return self._stats.copy()

    def reset_stats(self) -> None:
        """Reset selection statistics."""
        self._stats = {
            "selections": 0,
            "total_candidates": 0,
            "avg_valid_rate": 0.0,
            "avg_best_score": 0.0,
        }


def select_best_of_n(
    candidates: List[str],
    language: str = "python",
    strategy: SelectionStrategy = SelectionStrategy.BEST_SCORE,
    **kwargs: Any,
) -> SelectionResult:
    """Convenience function to select best candidate.

    Creates a temporary BestOfNSelector and selects the best candidate.

    Args:
        candidates: List of candidate code strings
        language: Target programming language
        strategy: Selection strategy
        **kwargs: Additional arguments passed to BestOfNSelector

    Returns:
        SelectionResult with selected candidate

    Example:
        >>> result = select_best_of_n([
        ...     "x = 1",
        ...     "x: int = 1",
        ... ])
        >>> result.selected
        'x: int = 1'
    """
    selector = BestOfNSelector(language=language, strategy=strategy, **kwargs)
    return selector.select_best(candidates)
