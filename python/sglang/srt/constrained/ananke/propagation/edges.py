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
"""Propagation edges for cross-domain constraint flow.

Each edge defines how constraints flow from a source domain to a target domain.
Edges have:
- Source and target domain names
- A propagation function that computes the new target constraint
- A priority (lower = higher priority)

Standard edges implement common propagation patterns:
- syntax_to_types: Syntactic structure informs type expectations
- types_to_syntax: Type expectations restrict valid syntax
- types_to_imports: Type usage implies required imports
- imports_to_types: Available imports affect type environment
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Optional

from core.constraint import Constraint
from core.domain import GenerationContext


@dataclass
class PropagationEdge(ABC):
    """Base class for propagation edges.

    An edge connects a source domain to a target domain and defines
    how constraints propagate between them.

    Attributes:
        source: Name of the source domain
        target: Name of the target domain
        priority: Priority (lower = higher priority, default 100)
        enabled: Whether this edge is active
    """

    source: str
    target: str
    priority: int = 100
    enabled: bool = True

    @abstractmethod
    def propagate(
        self,
        source_constraint: Constraint,
        target_constraint: Constraint,
        context: GenerationContext,
    ) -> Constraint:
        """Compute the new target constraint after propagation.

        Given the source domain's constraint and the target's current
        constraint, compute what the target's new constraint should be.

        The result should be at least as restrictive as target_constraint
        (monotonicity requirement).

        Args:
            source_constraint: Constraint from the source domain
            target_constraint: Current constraint of the target domain
            context: The generation context

        Returns:
            New constraint for the target domain
        """
        pass

    def __lt__(self, other: PropagationEdge) -> bool:
        """Compare by priority for sorting."""
        return self.priority < other.priority


@dataclass
class FunctionEdge(PropagationEdge):
    """Edge using a custom propagation function.

    This allows defining edge behavior with a simple callable.

    Attributes:
        propagate_fn: Function to compute new target constraint
    """

    propagate_fn: Callable[
        [Constraint, Constraint, GenerationContext],
        Constraint
    ] = field(default=lambda s, t, c: t)

    def propagate(
        self,
        source_constraint: Constraint,
        target_constraint: Constraint,
        context: GenerationContext,
    ) -> Constraint:
        """Apply the custom propagation function."""
        return self.propagate_fn(source_constraint, target_constraint, context)


@dataclass
class IdentityEdge(PropagationEdge):
    """Edge that passes constraint unchanged.

    Useful for simple dependency tracking without constraint modification.
    """

    def propagate(
        self,
        source_constraint: Constraint,
        target_constraint: Constraint,
        context: GenerationContext,
    ) -> Constraint:
        """Return target constraint unchanged."""
        return target_constraint


@dataclass
class MeetEdge(PropagationEdge):
    """Edge that meets source and target constraints.

    The result is the meet (conjunction) of both constraints,
    making the target at least as restrictive as both.
    """

    def propagate(
        self,
        source_constraint: Constraint,
        target_constraint: Constraint,
        context: GenerationContext,
    ) -> Constraint:
        """Return meet of source and target."""
        return target_constraint.meet(source_constraint)


class SyntaxToTypesEdge(PropagationEdge):
    """Edge from syntax domain to types domain.

    Syntactic structure provides type expectations. For example:
    - Function call implies callable type at function position
    - Assignment implies type compatibility between LHS and RHS
    - Return statement implies function return type

    This edge reads syntactic context and refines type expectations.
    """

    def __init__(self, priority: int = 50):
        """Initialize syntax-to-types edge.

        Args:
            priority: Edge priority (default 50)
        """
        super().__init__(
            source="syntax",
            target="types",
            priority=priority,
        )

    def propagate(
        self,
        source_constraint: Constraint,
        target_constraint: Constraint,
        context: GenerationContext,
    ) -> Constraint:
        """Propagate syntactic context to type expectations.

        Currently returns target unchanged - full implementation would
        analyze syntax constraint to derive type expectations.
        """
        # Handle TOP/BOTTOM
        if source_constraint.is_bottom():
            return target_constraint
        if target_constraint.is_bottom():
            return target_constraint

        # Full implementation would analyze syntax structure
        # For now, return target unchanged
        return target_constraint


class TypesToSyntaxEdge(PropagationEdge):
    """Edge from types domain to syntax domain.

    Type expectations can restrict valid syntax. For example:
    - If expecting int, string literals may be invalid
    - If expecting callable, literals are invalid
    - If expecting List[T], only list syntax valid

    This edge reads type context and refines syntax grammar.
    """

    def __init__(self, priority: int = 50):
        """Initialize types-to-syntax edge.

        Args:
            priority: Edge priority (default 50)
        """
        super().__init__(
            source="types",
            target="syntax",
            priority=priority,
        )

    def propagate(
        self,
        source_constraint: Constraint,
        target_constraint: Constraint,
        context: GenerationContext,
    ) -> Constraint:
        """Propagate type expectations to syntax restrictions.

        Currently returns target unchanged - full implementation would
        analyze type constraint to derive syntax restrictions.
        """
        # Handle TOP/BOTTOM
        if source_constraint.is_bottom():
            return target_constraint
        if target_constraint.is_bottom():
            return target_constraint

        # Full implementation would restrict grammar based on types
        return target_constraint


class TypesToImportsEdge(PropagationEdge):
    """Edge from types domain to imports domain.

    Type usage implies required imports. For example:
    - Using List[T] requires 'from typing import List'
    - Using numpy.ndarray requires 'import numpy'
    - Using custom class requires its import

    This edge reads type usage and derives import requirements.
    """

    def __init__(self, priority: int = 75):
        """Initialize types-to-imports edge.

        Args:
            priority: Edge priority (default 75)
        """
        super().__init__(
            source="types",
            target="imports",
            priority=priority,
        )

    def propagate(
        self,
        source_constraint: Constraint,
        target_constraint: Constraint,
        context: GenerationContext,
    ) -> Constraint:
        """Propagate type usage to import requirements.

        Currently returns target unchanged - full implementation would
        analyze types and derive required imports.
        """
        if source_constraint.is_bottom():
            return target_constraint
        if target_constraint.is_bottom():
            return target_constraint

        return target_constraint


class ImportsToTypesEdge(PropagationEdge):
    """Edge from imports domain to types domain.

    Available imports affect the type environment. For example:
    - Imported modules provide type bindings
    - Import errors can make types unavailable
    - Version constraints affect available types

    This edge reads import state and updates type environment.
    """

    def __init__(self, priority: int = 25):
        """Initialize imports-to-types edge.

        Args:
            priority: Edge priority (default 25)
        """
        super().__init__(
            source="imports",
            target="types",
            priority=priority,
        )

    def propagate(
        self,
        source_constraint: Constraint,
        target_constraint: Constraint,
        context: GenerationContext,
    ) -> Constraint:
        """Propagate import availability to type environment.

        Currently returns target unchanged - full implementation would
        update type environment based on imports.
        """
        if source_constraint.is_bottom():
            return target_constraint
        if target_constraint.is_bottom():
            return target_constraint

        return target_constraint


class ControlFlowToSemanticsEdge(PropagationEdge):
    """Edge from control flow domain to semantics domain.

    Control flow affects semantic constraints. For example:
    - Unreachable code has no semantic effect
    - Loop invariants must hold on each iteration
    - Branches create conditional semantic constraints

    This edge reads CFG and derives semantic conditions.
    """

    def __init__(self, priority: int = 100):
        """Initialize control-flow-to-semantics edge.

        Args:
            priority: Edge priority (default 100)
        """
        super().__init__(
            source="controlflow",
            target="semantics",
            priority=priority,
        )

    def propagate(
        self,
        source_constraint: Constraint,
        target_constraint: Constraint,
        context: GenerationContext,
    ) -> Constraint:
        """Propagate control flow to semantic constraints."""
        if source_constraint.is_bottom():
            return target_constraint
        if target_constraint.is_bottom():
            return target_constraint

        return target_constraint


def create_standard_edges() -> list[PropagationEdge]:
    """Create the standard set of propagation edges.

    Returns a list of edges implementing common propagation patterns:
    - syntax <-> types
    - types <-> imports
    - controlflow -> semantics

    Returns:
        List of standard propagation edges
    """
    return [
        SyntaxToTypesEdge(),
        TypesToSyntaxEdge(),
        TypesToImportsEdge(),
        ImportsToTypesEdge(),
        ControlFlowToSemanticsEdge(),
    ]
