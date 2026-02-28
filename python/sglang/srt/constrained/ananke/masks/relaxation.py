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
"""Mask relaxation protocol for constrained generation.

When domain masks AND together to produce a mask with very low popcount,
generation quality degrades (unicode artifacts, garbage output). This module
implements progressive relaxation: domains are tried in priority order, and
those that would cause popcount to drop below threshold are skipped.

Relaxation Order (most dispensable first):
    semantics   - Expensive semantic validation, often redundant
    controlflow - Control flow constraints, can be violated gracefully
    imports     - Import validation, least critical
    types       - Type constraints, usually important

Syntax constraints are NEVER relaxed as they ensure syntactic validity.

Example:
    >>> policy = RelaxationPolicy(threshold=10)
    >>> result = compute_mask_with_relaxation(
    ...     syntax_mask=syntax_mask,
    ...     domain_masks={"types": types_mask, "imports": imports_mask},
    ...     context=context,
    ...     policy=policy,
    ... )
    >>> if result.domains_relaxed:
    ...     logger.info(f"Relaxed domains: {result.domains_relaxed}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


class MaskRelaxation(Enum):
    """Level of mask relaxation that occurred.

    NONE: All domain constraints were applied successfully
    PARTIAL: Some domains were relaxed (skipped) to maintain popcount
    SYNTAX_ONLY: Only syntax constraints applied, all domains relaxed
    FULL: Even syntax relaxed (emergency fallback, should be rare)
    """

    NONE = auto()
    PARTIAL = auto()
    SYNTAX_ONLY = auto()
    FULL = auto()


# Domain relaxation order: most dispensable first, never relax syntax
# Domains listed here can be relaxed in this order when mask becomes too tight
RELAXATION_ORDER: List[str] = [
    "semantics",     # Most dispensable - expensive, often redundant
    "controlflow",   # Control flow - can violate gracefully
    "imports",       # Import validation - least critical
    "types",         # Type constraints - usually important but can relax
]

# Domains that should NEVER be relaxed
NEVER_RELAX: frozenset[str] = frozenset({"syntax"})


@dataclass
class RelaxationPolicy:
    """Configuration for mask relaxation behavior.

    Attributes:
        enabled: Whether relaxation is enabled at all
        threshold: Minimum popcount before relaxation triggers
        min_threshold: Absolute minimum popcount (even after relaxation)
        max_relaxation_level: Maximum relaxation level to allow
        domains_to_relax: Optional subset of domains that can be relaxed
        log_relaxation: Whether to log relaxation events
    """

    enabled: bool = True
    threshold: int = 10
    min_threshold: int = 1
    max_relaxation_level: MaskRelaxation = MaskRelaxation.SYNTAX_ONLY
    domains_to_relax: Optional[List[str]] = None
    log_relaxation: bool = True

    def can_relax_domain(self, domain: str) -> bool:
        """Check if a specific domain can be relaxed.

        Args:
            domain: Domain name to check

        Returns:
            True if domain can be relaxed under this policy
        """
        if domain in NEVER_RELAX:
            return False
        if self.domains_to_relax is not None:
            return domain in self.domains_to_relax
        return domain in RELAXATION_ORDER

    def get_relaxation_order(self) -> List[str]:
        """Get ordered list of domains that can be relaxed.

        Returns:
            Domains in relaxation order (most dispensable first)
        """
        if self.domains_to_relax is not None:
            # Maintain RELAXATION_ORDER for specified domains
            return [d for d in RELAXATION_ORDER if d in self.domains_to_relax]
        return list(RELAXATION_ORDER)


@dataclass
class RelaxationResult:
    """Result of mask computation with relaxation.

    Attributes:
        fused_mask: Final fused mask after relaxation
        relaxation_level: Level of relaxation that occurred
        domains_applied: Domains whose masks were applied
        domains_relaxed: Domains that were relaxed (skipped)
        final_popcount: Number of allowed tokens in final mask
        initial_popcount: Popcount before any domain masks applied
        popcount_history: Popcount after each domain was tried
    """

    fused_mask: torch.Tensor
    relaxation_level: MaskRelaxation = MaskRelaxation.NONE
    domains_applied: List[str] = field(default_factory=list)
    domains_relaxed: List[str] = field(default_factory=list)
    final_popcount: int = 0
    initial_popcount: int = 0
    popcount_history: Dict[str, int] = field(default_factory=dict)


def compute_mask_with_relaxation(
    syntax_mask: torch.Tensor,
    domain_masks: Dict[str, torch.Tensor],
    context: Any,
    policy: RelaxationPolicy,
) -> RelaxationResult:
    """Compute fused mask with progressive domain relaxation.

    This function implements the relaxation protocol:
    1. Start with syntax mask (never relaxed)
    2. For each domain in priority order (types, imports, controlflow, semantics):
       - Try applying the domain mask via AND
       - If popcount would drop below threshold, skip (relax) that domain
    3. Return fused mask with relaxation metadata

    Args:
        syntax_mask: Base syntax mask (always applied)
        domain_masks: Dictionary of domain name -> domain mask
        context: Generation context (for metadata)
        policy: Relaxation policy configuration

    Returns:
        RelaxationResult with fused mask and relaxation info
    """
    # Start with syntax mask (always applied, never relaxed)
    fused_mask = syntax_mask.clone()
    initial_popcount = int(fused_mask.sum().item())

    domains_applied: List[str] = []
    domains_relaxed: List[str] = []
    popcount_history: Dict[str, int] = {"syntax": initial_popcount}

    # If relaxation disabled, just AND all masks
    if not policy.enabled:
        for domain_name, domain_mask in domain_masks.items():
            fused_mask &= domain_mask
            domains_applied.append(domain_name)

        final_popcount = int(fused_mask.sum().item())
        return RelaxationResult(
            fused_mask=fused_mask,
            relaxation_level=MaskRelaxation.NONE,
            domains_applied=domains_applied,
            domains_relaxed=[],
            final_popcount=final_popcount,
            initial_popcount=initial_popcount,
            popcount_history=popcount_history,
        )

    # Apply domains in reverse relaxation order (most important first)
    # This means: types -> imports -> controlflow -> semantics
    application_order = list(reversed(policy.get_relaxation_order()))

    for domain_name in application_order:
        if domain_name not in domain_masks:
            continue

        domain_mask = domain_masks[domain_name]

        # Compute candidate mask
        candidate_mask = fused_mask & domain_mask
        candidate_popcount = int(candidate_mask.sum().item())

        # Check if applying this domain would drop below threshold
        if candidate_popcount >= policy.threshold:
            # Safe to apply
            fused_mask = candidate_mask
            domains_applied.append(domain_name)
            popcount_history[domain_name] = candidate_popcount
        else:
            # Would drop below threshold - relax this domain
            domains_relaxed.append(domain_name)
            popcount_history[f"{domain_name}_relaxed"] = candidate_popcount

            if policy.log_relaxation:
                logger.info(
                    f"Relaxing domain '{domain_name}': popcount would drop "
                    f"from {int(fused_mask.sum().item())} to {candidate_popcount} "
                    f"(threshold={policy.threshold})"
                )

    # Determine relaxation level
    final_popcount = int(fused_mask.sum().item())

    if not domains_relaxed:
        relaxation_level = MaskRelaxation.NONE
    elif len(domains_applied) == 0:
        relaxation_level = MaskRelaxation.SYNTAX_ONLY
    else:
        relaxation_level = MaskRelaxation.PARTIAL

    # Check against max relaxation level
    if relaxation_level.value > policy.max_relaxation_level.value:
        logger.warning(
            f"Relaxation level {relaxation_level.name} exceeds max "
            f"{policy.max_relaxation_level.name}"
        )

    return RelaxationResult(
        fused_mask=fused_mask,
        relaxation_level=relaxation_level,
        domains_applied=domains_applied,
        domains_relaxed=domains_relaxed,
        final_popcount=final_popcount,
        initial_popcount=initial_popcount,
        popcount_history=popcount_history,
    )


class RelaxationAwareEvaluator:
    """Evaluator that applies domain constraints with relaxation support.

    This evaluator wraps domain evaluation functions and applies them
    progressively with relaxation support. It maintains statistics about
    relaxation frequency to inform adaptive behavior.

    Example:
        >>> evaluator = RelaxationAwareEvaluator(policy=RelaxationPolicy(threshold=10))
        >>> evaluator.register("types", types_domain.token_mask)
        >>> evaluator.register("imports", imports_domain.token_mask)
        >>> result = evaluator.evaluate(constraints, context, syntax_mask)
    """

    def __init__(self, policy: Optional[RelaxationPolicy] = None) -> None:
        """Initialize the relaxation-aware evaluator.

        Args:
            policy: Relaxation policy (default: enabled with threshold=10)
        """
        self._policy = policy or RelaxationPolicy()
        self._domains: Dict[str, Callable[[Any, Any], torch.Tensor]] = {}

        # Statistics
        self._total_evaluations = 0
        self._relaxation_count = 0
        self._domain_relaxation_counts: Dict[str, int] = {}

    @property
    def policy(self) -> RelaxationPolicy:
        """Get the relaxation policy."""
        return self._policy

    @policy.setter
    def policy(self, value: RelaxationPolicy) -> None:
        """Set the relaxation policy."""
        self._policy = value

    def register(
        self,
        domain: str,
        evaluator: Callable[[Any, Any], torch.Tensor],
    ) -> None:
        """Register a domain evaluator function.

        Args:
            domain: Domain name (e.g., "types", "imports")
            evaluator: Function (constraint, context) -> mask
        """
        self._domains[domain] = evaluator
        if domain not in self._domain_relaxation_counts:
            self._domain_relaxation_counts[domain] = 0

    def unregister(self, domain: str) -> None:
        """Unregister a domain evaluator.

        Args:
            domain: Domain name to remove
        """
        self._domains.pop(domain, None)

    def evaluate(
        self,
        constraints: Dict[str, Any],
        context: Any,
        syntax_mask: torch.Tensor,
    ) -> RelaxationResult:
        """Evaluate constraints with relaxation support.

        Args:
            constraints: Dictionary of domain name -> constraint
            context: Generation context
            syntax_mask: Base syntax mask (already computed)

        Returns:
            RelaxationResult with fused mask and metadata
        """
        self._total_evaluations += 1

        # Compute domain masks
        domain_masks: Dict[str, torch.Tensor] = {}
        for domain_name, constraint in constraints.items():
            if domain_name in self._domains:
                evaluator = self._domains[domain_name]
                mask = evaluator(constraint, context)
                domain_masks[domain_name] = mask

        # Apply with relaxation
        result = compute_mask_with_relaxation(
            syntax_mask=syntax_mask,
            domain_masks=domain_masks,
            context=context,
            policy=self._policy,
        )

        # Update statistics
        if result.domains_relaxed:
            self._relaxation_count += 1
            for domain in result.domains_relaxed:
                self._domain_relaxation_counts[domain] = (
                    self._domain_relaxation_counts.get(domain, 0) + 1
                )

        return result

    def get_statistics(self) -> Dict[str, Any]:
        """Get relaxation statistics.

        Returns:
            Dictionary with evaluation and relaxation statistics
        """
        relaxation_rate = (
            self._relaxation_count / self._total_evaluations
            if self._total_evaluations > 0
            else 0.0
        )

        return {
            "total_evaluations": self._total_evaluations,
            "relaxation_count": self._relaxation_count,
            "relaxation_rate": relaxation_rate,
            "domain_relaxation_counts": dict(self._domain_relaxation_counts),
        }

    def reset_statistics(self) -> None:
        """Reset all statistics to zero."""
        self._total_evaluations = 0
        self._relaxation_count = 0
        self._domain_relaxation_counts = {d: 0 for d in self._domains}


def create_relaxation_policy(
    threshold: int = 10,
    enabled: bool = True,
    domains: Optional[List[str]] = None,
) -> RelaxationPolicy:
    """Factory function to create a RelaxationPolicy.

    Args:
        threshold: Minimum popcount before relaxation triggers
        enabled: Whether relaxation is enabled
        domains: Optional list of domains that can be relaxed

    Returns:
        Configured RelaxationPolicy
    """
    return RelaxationPolicy(
        enabled=enabled,
        threshold=threshold,
        domains_to_relax=domains,
    )
