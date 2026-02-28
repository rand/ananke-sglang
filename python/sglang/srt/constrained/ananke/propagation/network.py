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
"""Propagation network for cross-domain constraint flow.

The PropagationNetwork manages constraint propagation between different
domains (syntax, types, imports, control flow, semantics). When a constraint
changes in one domain, it may affect constraints in other domains.

The network uses a worklist algorithm to propagate changes until a fixed
point is reached or an iteration limit is hit.

Key properties:
- Monotonic: Constraints only become more restrictive
- Terminating: Limited iterations guarantee termination
- Compositional: Domains can be added/removed independently

References:
    - Hazel: "Statically Contextualizing LLMs with Typed Holes" (OOPSLA 2024)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

from core.constraint import Constraint, Satisfiability
from core.domain import ConstraintDomain, GenerationContext

if TYPE_CHECKING:
    from .edges import PropagationEdge


@dataclass
class PropagationResult:
    """Result of a propagation operation.

    Attributes:
        converged: Whether the network reached a fixed point
        iterations: Number of iterations performed
        changed_domains: Domains whose constraints changed
        errors: Any errors encountered during propagation
    """

    converged: bool
    iterations: int
    changed_domains: Set[str] = field(default_factory=set)
    errors: List[str] = field(default_factory=list)

    @property
    def is_success(self) -> bool:
        """True if propagation succeeded (converged without errors)."""
        return self.converged and len(self.errors) == 0


@dataclass
class DomainState:
    """State of a single domain in the network.

    Attributes:
        domain: The domain instance
        constraint: Current constraint for this domain
        dirty: Whether this domain needs propagation
    """

    domain: ConstraintDomain
    constraint: Constraint
    dirty: bool = False

    def mark_dirty(self) -> None:
        """Mark this domain as needing propagation."""
        self.dirty = True

    def mark_clean(self) -> None:
        """Mark this domain as up-to-date."""
        self.dirty = False


class PropagationNetwork:
    """Network for cross-domain constraint propagation.

    The network maintains:
    - A set of registered domains
    - Edges defining how constraints flow between domains
    - A worklist of domains needing propagation

    Propagation uses a worklist algorithm:
    1. Pick a dirty domain from the worklist
    2. Propagate its constraint to all outgoing edges
    3. If target domain's constraint changes, mark it dirty
    4. Repeat until no dirty domains or iteration limit reached

    Example:
        >>> network = PropagationNetwork()
        >>> network.register_domain(syntax_domain, syntax_constraint)
        >>> network.register_domain(type_domain, type_constraint)
        >>> network.add_edge(syntax_to_types_edge)
        >>> result = network.propagate(context)
    """

    # Default iteration limit to prevent infinite loops
    DEFAULT_MAX_ITERATIONS = 100

    def __init__(self, max_iterations: int = DEFAULT_MAX_ITERATIONS):
        """Initialize the propagation network.

        Args:
            max_iterations: Maximum propagation iterations before stopping
        """
        self._domains: Dict[str, DomainState] = {}
        self._edges: List[PropagationEdge] = []
        self._max_iterations = max_iterations

    @property
    def domain_names(self) -> List[str]:
        """Get list of registered domain names."""
        return list(self._domains.keys())

    @property
    def domains(self) -> Dict[str, ConstraintDomain]:
        """Get mapping of domain names to domain instances."""
        return {name: state.domain for name, state in self._domains.items()}

    def register_domain(
        self,
        domain: ConstraintDomain,
        initial_constraint: Optional[Constraint] = None,
    ) -> None:
        """Register a domain with the network.

        Args:
            domain: The domain to register
            initial_constraint: Initial constraint (defaults to TOP)
        """
        name = domain.name
        constraint = initial_constraint if initial_constraint is not None else domain.top
        self._domains[name] = DomainState(
            domain=domain,
            constraint=constraint,
            dirty=False,
        )

    def unregister_domain(self, domain_name: str) -> None:
        """Remove a domain from the network.

        Also removes any edges involving this domain.

        Args:
            domain_name: Name of the domain to remove
        """
        if domain_name in self._domains:
            del self._domains[domain_name]
            # Remove edges involving this domain
            self._edges = [
                e for e in self._edges
                if e.source != domain_name and e.target != domain_name
            ]

    def get_domain(self, name: str) -> Optional[ConstraintDomain]:
        """Get a domain by name.

        Args:
            name: Domain name

        Returns:
            The domain, or None if not registered
        """
        state = self._domains.get(name)
        return state.domain if state else None

    def get_constraint(self, domain_name: str) -> Optional[Constraint]:
        """Get the current constraint for a domain.

        Args:
            domain_name: Name of the domain

        Returns:
            The constraint, or None if domain not registered
        """
        state = self._domains.get(domain_name)
        return state.constraint if state else None

    def set_constraint(self, domain_name: str, constraint: Constraint) -> bool:
        """Set the constraint for a domain.

        Marks the domain dirty if the constraint changed.

        Args:
            domain_name: Name of the domain
            constraint: New constraint

        Returns:
            True if the constraint changed
        """
        state = self._domains.get(domain_name)
        if state is None:
            return False

        if state.constraint != constraint:
            state.constraint = constraint
            state.mark_dirty()
            return True
        return False

    def add_edge(self, edge: PropagationEdge) -> None:
        """Add a propagation edge to the network.

        Args:
            edge: The edge to add
        """
        self._edges.append(edge)

    def remove_edge(self, source: str, target: str) -> bool:
        """Remove edges between two domains.

        Args:
            source: Source domain name
            target: Target domain name

        Returns:
            True if any edges were removed
        """
        original_count = len(self._edges)
        self._edges = [
            e for e in self._edges
            if not (e.source == source and e.target == target)
        ]
        return len(self._edges) < original_count

    def get_edges_from(self, domain_name: str) -> List[PropagationEdge]:
        """Get all edges originating from a domain.

        Args:
            domain_name: Source domain name

        Returns:
            List of outgoing edges
        """
        return [e for e in self._edges if e.source == domain_name]

    def get_edges_to(self, domain_name: str) -> List[PropagationEdge]:
        """Get all edges targeting a domain.

        Args:
            domain_name: Target domain name

        Returns:
            List of incoming edges
        """
        return [e for e in self._edges if e.target == domain_name]

    def mark_dirty(self, domain_name: str) -> None:
        """Mark a domain as needing propagation.

        Args:
            domain_name: Name of the domain
        """
        state = self._domains.get(domain_name)
        if state:
            state.mark_dirty()

    def mark_all_dirty(self) -> None:
        """Mark all domains as needing propagation."""
        for state in self._domains.values():
            state.mark_dirty()

    def propagate(self, context: GenerationContext) -> PropagationResult:
        """Run constraint propagation until fixed point or limit.

        Uses a worklist algorithm to propagate constraints between
        domains according to registered edges.

        Args:
            context: The generation context

        Returns:
            PropagationResult indicating success/failure
        """
        changed_domains: Set[str] = set()
        errors: List[str] = []
        iterations = 0

        while iterations < self._max_iterations:
            # Find dirty domains
            dirty_domains = [
                name for name, state in self._domains.items()
                if state.dirty
            ]

            if not dirty_domains:
                # Fixed point reached
                return PropagationResult(
                    converged=True,
                    iterations=iterations,
                    changed_domains=changed_domains,
                    errors=errors,
                )

            # Process one dirty domain
            source_name = dirty_domains[0]
            source_state = self._domains[source_name]
            source_state.mark_clean()

            # Propagate along all outgoing edges
            for edge in self.get_edges_from(source_name):
                target_name = edge.target
                target_state = self._domains.get(target_name)

                if target_state is None:
                    errors.append(f"Target domain not found: {target_name}")
                    continue

                try:
                    # Compute new constraint via edge propagation
                    new_constraint = edge.propagate(
                        source_constraint=source_state.constraint,
                        target_constraint=target_state.constraint,
                        context=context,
                    )

                    # Check if constraint changed
                    if new_constraint != target_state.constraint:
                        # Verify monotonicity (new should be more restrictive)
                        if self._check_monotonicity(target_state.constraint, new_constraint):
                            target_state.constraint = new_constraint
                            target_state.mark_dirty()
                            changed_domains.add(target_name)
                        else:
                            errors.append(
                                f"Non-monotonic update on edge {source_name} -> {target_name}"
                            )
                except Exception as e:
                    errors.append(f"Propagation error on {source_name} -> {target_name}: {e}")

            iterations += 1

        # Iteration limit reached
        return PropagationResult(
            converged=False,
            iterations=iterations,
            changed_domains=changed_domains,
            errors=errors + ["Iteration limit reached"],
        )

    def _check_monotonicity(
        self,
        old_constraint: Constraint,
        new_constraint: Constraint,
    ) -> bool:
        """Check that constraint update is monotonic.

        Monotonicity means the new constraint is at least as restrictive
        as the old one. This is approximated by checking that
        old.meet(new) == new.

        Args:
            old_constraint: Previous constraint
            new_constraint: New constraint

        Returns:
            True if update is monotonic
        """
        # If either is TOP/BOTTOM, monotonicity holds in these cases:
        # - BOTTOM -> anything: invalid (not monotonic)
        # - anything -> BOTTOM: valid (most restrictive)
        # - TOP -> anything: valid (any refinement)
        # - anything -> TOP: invalid (relaxation)
        if old_constraint.is_bottom():
            return False  # Can't refine BOTTOM
        if new_constraint.is_top() and not old_constraint.is_top():
            return False  # Relaxation to TOP

        # General check: new should be the meet of old and new
        # (i.e., new is at least as restrictive)
        try:
            meet = old_constraint.meet(new_constraint)
            return meet == new_constraint
        except Exception:
            # If meet fails, assume monotonic (conservative)
            return True

    def propagate_single(
        self,
        source_domain: str,
        context: GenerationContext,
    ) -> PropagationResult:
        """Propagate from a single domain.

        Only propagates along edges originating from the specified domain.

        Args:
            source_domain: Domain to propagate from
            context: The generation context

        Returns:
            PropagationResult
        """
        source_state = self._domains.get(source_domain)
        if source_state is None:
            return PropagationResult(
                converged=True,
                iterations=0,
                errors=[f"Domain not found: {source_domain}"],
            )

        changed_domains: Set[str] = set()
        errors: List[str] = []

        for edge in self.get_edges_from(source_domain):
            target_name = edge.target
            target_state = self._domains.get(target_name)

            if target_state is None:
                errors.append(f"Target domain not found: {target_name}")
                continue

            try:
                new_constraint = edge.propagate(
                    source_constraint=source_state.constraint,
                    target_constraint=target_state.constraint,
                    context=context,
                )

                if new_constraint != target_state.constraint:
                    target_state.constraint = new_constraint
                    changed_domains.add(target_name)
            except Exception as e:
                errors.append(f"Propagation error: {e}")

        return PropagationResult(
            converged=True,
            iterations=1,
            changed_domains=changed_domains,
            errors=errors,
        )

    def observe_token(
        self,
        token_id: int,
        context: GenerationContext,
    ) -> None:
        """Update all domains after observing a token.

        Calls observe_token on each domain and triggers propagation.

        Args:
            token_id: The observed token ID
            context: The generation context
        """
        # Update each domain
        for name, state in self._domains.items():
            new_constraint = state.domain.observe_token(
                state.constraint,
                token_id,
                context,
            )
            if new_constraint != state.constraint:
                state.constraint = new_constraint
                state.mark_dirty()

        # Run propagation
        self.propagate(context)

    def checkpoint(self) -> Dict[str, Any]:
        """Create a checkpoint of the network state.

        Returns:
            Dictionary containing all state needed for restore
        """
        return {
            "constraints": {
                name: state.constraint
                for name, state in self._domains.items()
            },
            "dirty": {
                name: state.dirty
                for name, state in self._domains.items()
            },
            "domain_checkpoints": {
                name: state.domain.checkpoint()
                for name, state in self._domains.items()
            },
        }

    def restore(self, checkpoint: Dict[str, Any]) -> None:
        """Restore network state from a checkpoint.

        Args:
            checkpoint: Previously created checkpoint
        """
        constraints = checkpoint.get("constraints", {})
        dirty = checkpoint.get("dirty", {})
        domain_checkpoints = checkpoint.get("domain_checkpoints", {})

        for name, state in self._domains.items():
            if name in constraints:
                state.constraint = constraints[name]
            if name in dirty:
                state.dirty = dirty[name]
            if name in domain_checkpoints:
                state.domain.restore(domain_checkpoints[name])

    def is_satisfiable(self) -> bool:
        """Check if all domain constraints are satisfiable.

        Returns:
            True if no domain has an UNSAT constraint
        """
        for state in self._domains.values():
            if state.constraint.satisfiability() == Satisfiability.UNSAT:
                return False
        return True

    def get_unsatisfiable_domains(self) -> List[str]:
        """Get list of domains with unsatisfiable constraints.

        Returns:
            List of domain names
        """
        return [
            name for name, state in self._domains.items()
            if state.constraint.satisfiability() == Satisfiability.UNSAT
        ]
