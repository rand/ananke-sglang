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
"""Lazy constraint evaluation.

Defers constraint evaluation until needed, enabling:
- Skip expensive domains if already constrained enough
- Budget-limited evaluation
- Priority-based evaluation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar
import time

import torch

# Type variable for constraint types
C = TypeVar("C")


class EvaluationPriority(Enum):
    """Priority for lazy evaluation.

    Priorities:
    - CRITICAL: Always evaluate
    - HIGH: Evaluate unless already very constrained
    - NORMAL: Evaluate if budget allows
    - LOW: Only evaluate if nothing else to do
    - SKIP: Skip evaluation
    """

    CRITICAL = auto()
    HIGH = auto()
    NORMAL = auto()
    LOW = auto()
    SKIP = auto()


@dataclass
class LazyMask:
    """A lazily-evaluated mask.

    The mask computation is deferred until force() is called.

    Attributes:
        domain: Domain name
        constraint: The constraint to evaluate
        context: Generation context
        compute_fn: Function to compute the mask
        priority: Evaluation priority
        cached_mask: Cached result (if computed)
        compute_time_ns: Time taken to compute
    """

    domain: str
    constraint: Any
    context: Any
    compute_fn: Callable[[Any, Any], torch.Tensor]
    priority: EvaluationPriority = EvaluationPriority.NORMAL
    cached_mask: Optional[torch.Tensor] = None
    compute_time_ns: int = 0

    @property
    def is_evaluated(self) -> bool:
        """Check if mask has been computed."""
        return self.cached_mask is not None

    def force(self) -> torch.Tensor:
        """Force evaluation of the mask.

        Returns:
            The computed mask
        """
        if self.cached_mask is not None:
            return self.cached_mask

        start_ns = time.perf_counter_ns()
        self.cached_mask = self.compute_fn(self.constraint, self.context)
        end_ns = time.perf_counter_ns()
        self.compute_time_ns = end_ns - start_ns

        return self.cached_mask

    def invalidate(self) -> None:
        """Invalidate the cached mask."""
        self.cached_mask = None
        self.compute_time_ns = 0


@dataclass
class EvaluationBudget:
    """Budget for lazy evaluation.

    Attributes:
        max_time_ns: Maximum time in nanoseconds
        max_domains: Maximum domains to evaluate
        min_selectivity: Stop if selectivity exceeds this
    """

    max_time_ns: int = 5_000_000  # 5ms default
    max_domains: int = 5
    min_selectivity: float = 0.99

    def is_exceeded(
        self,
        elapsed_ns: int,
        domains_evaluated: int,
        current_selectivity: float,
    ) -> bool:
        """Check if budget is exceeded.

        Args:
            elapsed_ns: Time elapsed so far
            domains_evaluated: Number of domains evaluated
            current_selectivity: Current mask selectivity

        Returns:
            True if budget is exceeded
        """
        if elapsed_ns >= self.max_time_ns:
            return True
        if domains_evaluated >= self.max_domains:
            return True
        if current_selectivity >= self.min_selectivity:
            return True
        return False


@dataclass
class LazyEvaluationResult:
    """Result of lazy evaluation.

    Attributes:
        fused_mask: The final fused mask
        evaluated_domains: Domains that were evaluated
        skipped_domains: Domains that were skipped
        total_time_ns: Total evaluation time
        budget_exceeded: Whether budget was exceeded
    """

    fused_mask: torch.Tensor
    evaluated_domains: List[str] = field(default_factory=list)
    skipped_domains: List[str] = field(default_factory=list)
    total_time_ns: int = 0
    budget_exceeded: bool = False


class LazyConstraintEvaluator:
    """Evaluates constraints lazily with budget control.

    Key features:
    - Priority-based evaluation order
    - Budget-limited computation
    - Early stopping when sufficiently constrained

    Example:
        >>> evaluator = LazyConstraintEvaluator()
        >>> evaluator.register("syntax", syntax_mask_fn, EvaluationPriority.CRITICAL)
        >>> evaluator.register("types", type_mask_fn, EvaluationPriority.HIGH)
        >>> evaluator.register("semantics", semantic_mask_fn, EvaluationPriority.LOW)
        >>> result = evaluator.evaluate(constraints, context, budget)
    """

    def __init__(self) -> None:
        """Initialize the evaluator."""
        self._domains: Dict[str, Callable[[Any, Any], torch.Tensor]] = {}
        self._priorities: Dict[str, EvaluationPriority] = {}
        self._estimated_times: Dict[str, int] = {}  # nanoseconds

    def register(
        self,
        domain: str,
        compute_fn: Callable[[Any, Any], torch.Tensor],
        priority: EvaluationPriority = EvaluationPriority.NORMAL,
        estimated_time_ns: int = 1_000_000,  # 1ms default
    ) -> None:
        """Register a domain for lazy evaluation.

        Args:
            domain: Domain name
            compute_fn: Function to compute mask
            priority: Evaluation priority
            estimated_time_ns: Estimated computation time
        """
        self._domains[domain] = compute_fn
        self._priorities[domain] = priority
        self._estimated_times[domain] = estimated_time_ns

    def unregister(self, domain: str) -> None:
        """Unregister a domain.

        Args:
            domain: Domain name to remove
        """
        self._domains.pop(domain, None)
        self._priorities.pop(domain, None)
        self._estimated_times.pop(domain, None)

    def create_lazy_masks(
        self,
        constraints: Dict[str, Any],
        context: Any,
    ) -> Dict[str, LazyMask]:
        """Create lazy masks for constraints.

        Args:
            constraints: Map from domain to constraint
            context: Generation context

        Returns:
            Map from domain to LazyMask
        """
        lazy_masks: Dict[str, LazyMask] = {}

        for domain, constraint in constraints.items():
            if domain not in self._domains:
                continue

            lazy_masks[domain] = LazyMask(
                domain=domain,
                constraint=constraint,
                context=context,
                compute_fn=self._domains[domain],
                priority=self._priorities.get(domain, EvaluationPriority.NORMAL),
            )

        return lazy_masks

    def evaluate(
        self,
        constraints: Dict[str, Any],
        context: Any,
        budget: Optional[EvaluationBudget] = None,
    ) -> LazyEvaluationResult:
        """Evaluate constraints lazily within budget.

        Args:
            constraints: Map from domain to constraint
            context: Generation context
            budget: Optional evaluation budget

        Returns:
            LazyEvaluationResult with fused mask
        """
        budget = budget or EvaluationBudget()
        start_ns = time.perf_counter_ns()

        lazy_masks = self.create_lazy_masks(constraints, context)

        # Sort by priority
        sorted_domains = sorted(
            lazy_masks.keys(),
            key=lambda d: self._priorities.get(d, EvaluationPriority.NORMAL).value,
        )

        evaluated: List[str] = []
        skipped: List[str] = []

        # Initialize fused mask
        vocab_size = getattr(context, "vocab_size", 32000)
        device = getattr(context, "device", "cpu")
        fused_mask = torch.ones(vocab_size, dtype=torch.bool, device=device)

        budget_exceeded = False

        for domain in sorted_domains:
            lazy = lazy_masks[domain]

            # Check priority
            if lazy.priority == EvaluationPriority.SKIP:
                skipped.append(domain)
                continue

            # Check budget
            elapsed_ns = time.perf_counter_ns() - start_ns
            current_selectivity = (vocab_size - fused_mask.sum().item()) / vocab_size

            if lazy.priority not in (
                EvaluationPriority.CRITICAL,
                EvaluationPriority.HIGH,
            ):
                # Check if should skip based on budget
                if budget.is_exceeded(elapsed_ns, len(evaluated), current_selectivity):
                    budget_exceeded = True
                    skipped.append(domain)
                    continue

                # Check if estimated time would exceed budget
                remaining_ns = budget.max_time_ns - elapsed_ns
                if self._estimated_times.get(domain, 0) > remaining_ns:
                    skipped.append(domain)
                    continue

            # Evaluate
            mask = lazy.force()
            evaluated.append(domain)

            # Update estimated time
            self._estimated_times[domain] = lazy.compute_time_ns

            # Fuse
            fused_mask = fused_mask & mask

            # Short-circuit if all blocked
            if not fused_mask.any():
                break

        end_ns = time.perf_counter_ns()

        return LazyEvaluationResult(
            fused_mask=fused_mask,
            evaluated_domains=evaluated,
            skipped_domains=skipped,
            total_time_ns=end_ns - start_ns,
            budget_exceeded=budget_exceeded,
        )

    def update_estimated_time(self, domain: str, time_ns: int) -> None:
        """Update estimated computation time for a domain.

        Args:
            domain: Domain name
            time_ns: Observed computation time
        """
        if domain in self._estimated_times:
            # Exponential moving average
            old = self._estimated_times[domain]
            self._estimated_times[domain] = int(0.7 * time_ns + 0.3 * old)

    def set_priority(self, domain: str, priority: EvaluationPriority) -> None:
        """Set priority for a domain.

        Args:
            domain: Domain name
            priority: New priority
        """
        if domain in self._domains:
            self._priorities[domain] = priority


class AdaptiveLazyEvaluator:
    """Lazy evaluator that adapts priorities based on history.

    Automatically adjusts priorities based on:
    - Historical selectivity
    - Computation time
    - Usefulness (whether domain actually constrained anything)
    """

    def __init__(self, base_evaluator: LazyConstraintEvaluator) -> None:
        """Initialize adaptive evaluator.

        Args:
            base_evaluator: Base lazy evaluator
        """
        self._evaluator = base_evaluator
        self._selectivity_history: Dict[str, List[float]] = {}
        self._usefulness_history: Dict[str, List[bool]] = {}
        self._max_history = 100

    def evaluate(
        self,
        constraints: Dict[str, Any],
        context: Any,
        budget: Optional[EvaluationBudget] = None,
    ) -> LazyEvaluationResult:
        """Evaluate with adaptive priorities.

        Args:
            constraints: Map from domain to constraint
            context: Generation context
            budget: Optional evaluation budget

        Returns:
            LazyEvaluationResult
        """
        # Adapt priorities before evaluation
        self._adapt_priorities()

        # Evaluate
        result = self._evaluator.evaluate(constraints, context, budget)

        # Update history
        self._update_history(result, context)

        return result

    def _adapt_priorities(self) -> None:
        """Adapt priorities based on history."""
        for domain in self._evaluator._domains:
            selectivity_hist = self._selectivity_history.get(domain, [])
            usefulness_hist = self._usefulness_history.get(domain, [])

            if not selectivity_hist:
                continue

            avg_selectivity = sum(selectivity_hist) / len(selectivity_hist)
            usefulness_rate = sum(usefulness_hist) / len(usefulness_hist) if usefulness_hist else 0.5

            # High selectivity and useful -> HIGH priority
            if avg_selectivity > 0.5 and usefulness_rate > 0.7:
                self._evaluator.set_priority(domain, EvaluationPriority.HIGH)
            # Low usefulness -> LOW priority
            elif usefulness_rate < 0.3:
                self._evaluator.set_priority(domain, EvaluationPriority.LOW)
            else:
                self._evaluator.set_priority(domain, EvaluationPriority.NORMAL)

    def _update_history(
        self,
        result: LazyEvaluationResult,
        context: Any,
    ) -> None:
        """Update history from evaluation result."""
        vocab_size = getattr(context, "vocab_size", 32000)

        # Track selectivity and usefulness for evaluated domains
        for domain in result.evaluated_domains:
            # Would need individual domain masks to properly track
            # For now, just track that it was evaluated
            if domain not in self._selectivity_history:
                self._selectivity_history[domain] = []
                self._usefulness_history[domain] = []

            # Approximate: mark as useful if final mask is constrained
            final_selectivity = (vocab_size - result.fused_mask.sum().item()) / vocab_size
            self._usefulness_history[domain].append(final_selectivity > 0.01)

            # Trim history
            if len(self._usefulness_history[domain]) > self._max_history:
                self._usefulness_history[domain] = self._usefulness_history[domain][-self._max_history:]


def create_lazy_evaluator() -> LazyConstraintEvaluator:
    """Factory function to create a LazyConstraintEvaluator.

    Returns:
        New LazyConstraintEvaluator instance
    """
    return LazyConstraintEvaluator()
