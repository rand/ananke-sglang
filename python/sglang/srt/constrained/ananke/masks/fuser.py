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
"""Token mask fusion across constraint domains.

Fuses token masks from multiple domains using bitwise AND.
Optimizations:
- Selectivity-ordered evaluation (most restrictive first)
- Short-circuit on all-zeros mask
- Lazy evaluation of expensive domains
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar

import torch

# Type variable for constraint types
C = TypeVar("C")


class FusionStrategy(Enum):
    """Strategy for fusing masks from multiple domains.

    Strategies:
    - BITWISE_AND: Simple AND of all masks
    - SELECTIVITY_ORDERED: Evaluate most selective first
    - LAZY: Only evaluate domains when needed
    - CACHED: Use cached masks when available
    """

    BITWISE_AND = auto()
    SELECTIVITY_ORDERED = auto()
    LAZY = auto()
    CACHED = auto()


@dataclass
class DomainMaskInfo:
    """Information about a domain's mask.

    Attributes:
        domain_name: Name of the domain
        mask: The computed mask
        selectivity: Fraction of tokens blocked (0-1)
        compute_time_ns: Time to compute in nanoseconds
        from_cache: Whether mask was from cache
    """

    domain_name: str
    mask: torch.Tensor
    selectivity: float = 0.0
    compute_time_ns: int = 0
    from_cache: bool = False

    def __post_init__(self) -> None:
        if self.selectivity == 0.0 and self.mask is not None:
            # Compute selectivity as fraction blocked
            total = self.mask.numel()
            blocked = (total - self.mask.sum().item()) / total
            self.selectivity = blocked


@dataclass
class FusionResult:
    """Result of mask fusion.

    Attributes:
        fused_mask: The final fused mask
        domain_masks: Per-domain mask information
        total_time_ns: Total fusion time in nanoseconds
        short_circuited: Whether fusion short-circuited
    """

    fused_mask: torch.Tensor
    domain_masks: List[DomainMaskInfo] = field(default_factory=list)
    total_time_ns: int = 0
    short_circuited: bool = False

    @property
    def selectivity(self) -> float:
        """Get overall selectivity of fused mask."""
        total = self.fused_mask.numel()
        blocked = (total - self.fused_mask.sum().item()) / total
        return blocked

    @property
    def num_valid_tokens(self) -> int:
        """Get number of valid (unblocked) tokens."""
        return int(self.fused_mask.sum().item())


class TokenMaskFuser:
    """Fuses token masks from multiple constraint domains.

    Key optimizations:
    1. Selectivity ordering: Evaluate most restrictive domains first
    2. Short-circuit: Stop as soon as mask becomes all-zeros
    3. Lazy evaluation: Skip expensive domains if already restricted

    Example:
        >>> fuser = TokenMaskFuser()
        >>> fuser.register_domain("syntax", syntax_domain)
        >>> fuser.register_domain("types", type_domain)
        >>> result = fuser.fuse(constraint, context)
        >>> # Apply fused_mask to logits
    """

    def __init__(
        self,
        strategy: FusionStrategy = FusionStrategy.SELECTIVITY_ORDERED,
        short_circuit_threshold: float = 0.0,
    ) -> None:
        """Initialize the fuser.

        Args:
            strategy: Fusion strategy to use
            short_circuit_threshold: Stop if selectivity exceeds this
        """
        self._strategy = strategy
        self._short_circuit_threshold = short_circuit_threshold
        self._domains: Dict[str, Any] = {}
        self._selectivity_history: Dict[str, float] = {}
        self._priority_overrides: Dict[str, int] = {}

    def register_domain(
        self,
        name: str,
        domain: Any,
        priority: Optional[int] = None,
    ) -> None:
        """Register a constraint domain.

        Args:
            name: Domain name
            domain: ConstraintDomain instance
            priority: Optional priority override (higher = evaluate first)
        """
        self._domains[name] = domain
        if priority is not None:
            self._priority_overrides[name] = priority

    def unregister_domain(self, name: str) -> None:
        """Unregister a domain.

        Args:
            name: Domain name to remove
        """
        self._domains.pop(name, None)
        self._selectivity_history.pop(name, None)
        self._priority_overrides.pop(name, None)

    @property
    def domain_names(self) -> List[str]:
        """Get registered domain names."""
        return list(self._domains.keys())

    def fuse(
        self,
        constraints: Dict[str, Any],
        context: Any,
        exclude: Optional[List[str]] = None,
    ) -> FusionResult:
        """Fuse masks from all domains.

        Args:
            constraints: Map from domain name to constraint
            context: Generation context
            exclude: Optional list of domains to exclude

        Returns:
            FusionResult with fused mask and statistics
        """
        import time

        start_ns = time.perf_counter_ns()
        exclude_set = set(exclude or [])

        # Get ordered list of domains to evaluate
        ordered_domains = self._get_evaluation_order(exclude_set)

        # Initialize result mask
        vocab_size = getattr(context, "vocab_size", 32000)
        device = getattr(context, "device", "cpu")
        fused_mask = torch.ones(vocab_size, dtype=torch.bool, device=device)

        domain_masks: List[DomainMaskInfo] = []
        short_circuited = False

        for domain_name in ordered_domains:
            domain = self._domains[domain_name]
            constraint = constraints.get(domain_name)

            if constraint is None:
                continue

            # Compute mask for this domain
            mask_start_ns = time.perf_counter_ns()
            domain_mask = domain.token_mask(constraint, context)
            mask_end_ns = time.perf_counter_ns()

            # Record mask info
            info = DomainMaskInfo(
                domain_name=domain_name,
                mask=domain_mask,
                compute_time_ns=mask_end_ns - mask_start_ns,
            )
            domain_masks.append(info)

            # Update selectivity history
            self._selectivity_history[domain_name] = info.selectivity

            # Fuse with AND
            fused_mask = fused_mask & domain_mask

            # Short-circuit check
            if not fused_mask.any():
                short_circuited = True
                break

            # Optional early stop based on selectivity
            if self._short_circuit_threshold > 0:
                current_selectivity = (
                    vocab_size - fused_mask.sum().item()
                ) / vocab_size
                if current_selectivity >= self._short_circuit_threshold:
                    short_circuited = True
                    break

        end_ns = time.perf_counter_ns()

        return FusionResult(
            fused_mask=fused_mask,
            domain_masks=domain_masks,
            total_time_ns=end_ns - start_ns,
            short_circuited=short_circuited,
        )

    def fuse_masks(
        self,
        masks: List[torch.Tensor],
        short_circuit: bool = True,
    ) -> torch.Tensor:
        """Fuse a list of pre-computed masks.

        Args:
            masks: List of boolean masks
            short_circuit: Whether to stop on all-zeros

        Returns:
            Fused mask
        """
        if not masks:
            raise ValueError("No masks to fuse")

        result = masks[0].clone()

        for mask in masks[1:]:
            result = result & mask
            if short_circuit and not result.any():
                break

        return result

    def _get_evaluation_order(self, exclude: set[str]) -> List[str]:
        """Get ordered list of domains for evaluation.

        Orders by:
        1. Priority overrides (higher first)
        2. Historical selectivity (more selective first)
        3. Registration order

        Args:
            exclude: Set of domain names to exclude

        Returns:
            Ordered list of domain names
        """
        domains = [n for n in self._domains.keys() if n not in exclude]

        if self._strategy == FusionStrategy.BITWISE_AND:
            # No ordering
            return domains

        def sort_key(name: str) -> tuple:
            priority = self._priority_overrides.get(name, 0)
            selectivity = self._selectivity_history.get(name, 0.0)
            return (-priority, -selectivity)

        return sorted(domains, key=sort_key)

    def update_selectivity(self, domain_name: str, selectivity: float) -> None:
        """Update selectivity history for a domain.

        Args:
            domain_name: Domain name
            selectivity: New selectivity value
        """
        if domain_name in self._domains:
            # Exponential moving average
            old = self._selectivity_history.get(domain_name, selectivity)
            self._selectivity_history[domain_name] = 0.7 * selectivity + 0.3 * old

    def get_selectivity(self, domain_name: str) -> float:
        """Get historical selectivity for a domain.

        Args:
            domain_name: Domain name

        Returns:
            Selectivity value (0-1)
        """
        return self._selectivity_history.get(domain_name, 0.0)

    def reset_statistics(self) -> None:
        """Reset selectivity history and statistics."""
        self._selectivity_history.clear()


class MultiDomainMaskFuser:
    """Fuser that handles domain dependencies.

    Some domains may depend on others (e.g., types depends on syntax).
    This fuser respects dependency ordering.
    """

    def __init__(self) -> None:
        """Initialize the fuser."""
        self._domains: Dict[str, Any] = {}
        self._dependencies: Dict[str, List[str]] = {}
        self._base_fuser = TokenMaskFuser()

    def register_domain(
        self,
        name: str,
        domain: Any,
        dependencies: Optional[List[str]] = None,
    ) -> None:
        """Register a domain with dependencies.

        Args:
            name: Domain name
            domain: ConstraintDomain instance
            dependencies: Names of domains this depends on
        """
        self._domains[name] = domain
        self._dependencies[name] = dependencies or []
        self._base_fuser.register_domain(name, domain)

    def fuse(
        self,
        constraints: Dict[str, Any],
        context: Any,
    ) -> FusionResult:
        """Fuse masks respecting dependencies.

        Args:
            constraints: Map from domain name to constraint
            context: Generation context

        Returns:
            FusionResult with fused mask
        """
        # Topological sort of domains
        ordered = self._topological_sort()

        # Set priorities based on topological order
        for i, name in enumerate(ordered):
            self._base_fuser._priority_overrides[name] = len(ordered) - i

        return self._base_fuser.fuse(constraints, context)

    def _topological_sort(self) -> List[str]:
        """Topologically sort domains by dependencies.

        Returns:
            Sorted list of domain names
        """
        visited: set[str] = set()
        result: List[str] = []

        def visit(name: str) -> None:
            if name in visited:
                return
            visited.add(name)
            for dep in self._dependencies.get(name, []):
                if dep in self._domains:
                    visit(dep)
            result.append(name)

        for name in self._domains:
            visit(name)

        return result


def create_fuser(
    strategy: FusionStrategy = FusionStrategy.SELECTIVITY_ORDERED,
) -> TokenMaskFuser:
    """Factory function to create a TokenMaskFuser.

    Args:
        strategy: Fusion strategy to use

    Returns:
        New TokenMaskFuser instance
    """
    return TokenMaskFuser(strategy=strategy)
