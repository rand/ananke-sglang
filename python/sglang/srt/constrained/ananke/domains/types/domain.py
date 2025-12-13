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
"""Type domain for Ananke's constraint system.

The TypeDomain implements incremental bidirectional type checking based on
the Hazel research (POPL 2024, OOPSLA 2025). It provides:

- TypeConstraint: The constraint type with expected types and equations
- Type checking that works on partial programs (marked lambda calculus)
- Incremental updates as tokens are generated
- Token masks based on type compatibility

References:
    - POPL 2024: "Total Type Error Localization and Recovery with Holes"
    - OOPSLA 2025: "Incremental Bidirectional Typing via Order Maintenance"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch

# Support both relative imports (when used as subpackage) and absolute imports (standalone testing)
try:
    from ...core.constraint import Satisfiability
    from ...core.domain import ConstraintDomain, GenerationContext
    from .constraint import (
        TYPE_BOTTOM,
        TYPE_TOP,
        TypeConstraint,
        Type,
        ANY,
        type_expecting,
    )
    from .environment import TypeEnvironment, EMPTY_ENVIRONMENT
    from .unification import Substitution, EMPTY_SUBSTITUTION
except ImportError:
    from core.constraint import Satisfiability
    from core.domain import ConstraintDomain, GenerationContext
    from domains.types.constraint import (
        TYPE_BOTTOM,
        TYPE_TOP,
        TypeConstraint,
        Type,
        ANY,
        type_expecting,
    )
    from domains.types.environment import TypeEnvironment, EMPTY_ENVIRONMENT
    from domains.types.unification import Substitution, EMPTY_SUBSTITUTION


@dataclass
class TypeDomainCheckpoint:
    """Checkpoint for TypeDomain state.

    Captures all mutable state needed to restore the type domain
    to a previous point. This is not frozen because Substitution
    contains a mutable mapping (could be made frozen with FrozenDict).

    Attributes:
        environment: Snapshot of the type environment
        substitution: Current unification substitution
        state_counter: The state counter value
        current_expected: The current expected type
    """

    environment: TypeEnvironment
    substitution: Substitution
    state_counter: int
    current_expected: Optional[Type]


class TypeDomain(ConstraintDomain[TypeConstraint]):
    """Type domain implementing incremental bidirectional type checking.

    The type domain tracks:
    - Type environment (variable -> type mappings)
    - Current expected type (from context)
    - Accumulated type equations
    - Substitution from unification

    Token masks are computed by checking which tokens could produce
    values of the expected type.

    Attributes:
        name: Domain identifier ("types")
        environment: Current type environment
        substitution: Current type substitution from unification
    """

    def __init__(
        self,
        environment: Optional[TypeEnvironment] = None,
        expected_type: Optional[Type] = None,
    ):
        """Initialize the type domain.

        Args:
            environment: Initial type environment (defaults to empty)
            expected_type: Initial expected type (defaults to Any)
        """
        self._environment = environment if environment is not None else EMPTY_ENVIRONMENT
        self._expected_type = expected_type if expected_type is not None else ANY
        self._substitution = EMPTY_SUBSTITUTION
        self._state_counter = 0

    @property
    def name(self) -> str:
        """Return the domain name."""
        return "types"

    @property
    def top(self) -> TypeConstraint:
        """Return the TOP constraint (no type restriction)."""
        return TYPE_TOP

    @property
    def bottom(self) -> TypeConstraint:
        """Return the BOTTOM constraint (unsatisfiable)."""
        return TYPE_BOTTOM

    @property
    def environment(self) -> TypeEnvironment:
        """Return the current type environment."""
        return self._environment

    @property
    def expected_type(self) -> Type:
        """Return the current expected type."""
        return self._expected_type

    def token_mask(
        self,
        constraint: TypeConstraint,
        context: GenerationContext,
    ) -> torch.Tensor:
        """Compute a token mask based on type constraints.

        Returns a boolean tensor indicating which tokens could produce
        values compatible with the expected type.

        For now, this returns all True - actual type-based filtering
        requires integration with the incremental parser to predict
        the type of each possible token.

        Args:
            constraint: The current type constraint
            context: Generation context with vocab size

        Returns:
            Boolean tensor of shape (vocab_size,)
        """
        # Return all True for now - type masking requires parser integration
        # A full implementation would:
        # 1. For each candidate token, predict resulting AST change
        # 2. Check if the change is type-compatible
        # 3. Use budget-limited checking for performance
        return torch.ones(context.vocab_size, dtype=torch.bool, device=context.device)

    def observe_token(
        self,
        constraint: TypeConstraint,
        token_id: int,
        context: GenerationContext,
    ) -> TypeConstraint:
        """Update the type constraint after observing a token.

        This is called after each token is generated. It:
        1. Updates the internal state counter
        2. Potentially updates the expected type based on parse progress
        3. Returns an updated constraint

        Args:
            constraint: Current type constraint
            token_id: The token that was generated
            context: Generation context

        Returns:
            Updated type constraint
        """
        # Handle TOP and BOTTOM
        if constraint.is_top() or constraint.is_bottom():
            return constraint

        # Update state counter
        self._state_counter += 1

        # Create updated constraint with new environment hash
        return constraint.with_environment_hash(self._state_counter)

    def checkpoint(self) -> TypeDomainCheckpoint:
        """Create a checkpoint of the current state.

        Returns:
            Checkpoint that can restore this state
        """
        return TypeDomainCheckpoint(
            environment=self._environment,
            substitution=self._substitution,
            state_counter=self._state_counter,
            current_expected=self._expected_type,
        )

    def restore(self, checkpoint: Checkpoint) -> None:
        """Restore state from a checkpoint.

        Args:
            checkpoint: Previously created checkpoint
        """
        if not isinstance(checkpoint, TypeDomainCheckpoint):
            raise TypeError(
                f"Expected TypeDomainCheckpoint, got {type(checkpoint).__name__}"
            )
        self._environment = checkpoint.environment
        self._substitution = checkpoint.substitution
        self._state_counter = checkpoint.state_counter
        self._expected_type = checkpoint.current_expected

    def satisfiability(self, constraint: TypeConstraint) -> Satisfiability:
        """Check the satisfiability of a type constraint.

        Args:
            constraint: The constraint to check

        Returns:
            SAT, UNSAT, or UNKNOWN
        """
        return constraint.satisfiability()

    def propagate_from(
        self,
        constraint: TypeConstraint,
        source_domain: str,
        source_constraint: Any,
    ) -> TypeConstraint:
        """Handle constraint propagation from another domain.

        Types can receive information from:
        - Syntax domain: syntactic structure provides type expectations
        - Imports domain: available imports affect type environment

        Args:
            constraint: Current type constraint
            source_domain: Name of the source domain
            source_constraint: The constraint from that domain

        Returns:
            Potentially refined type constraint
        """
        # For now, no cross-domain propagation
        return constraint

    def create_constraint(
        self,
        expected_type: Optional[Type] = None,
    ) -> TypeConstraint:
        """Create a new type constraint.

        Args:
            expected_type: The expected type (defaults to Any)

        Returns:
            A new TypeConstraint
        """
        if expected_type is None:
            return TYPE_TOP

        return type_expecting(expected_type)

    def bind_variable(self, name: str, ty: Type) -> None:
        """Bind a variable in the type environment.

        Args:
            name: Variable name
            ty: The type to bind
        """
        self._environment = self._environment.bind(name, ty)

    def lookup_variable(self, name: str) -> Optional[Type]:
        """Look up a variable's type.

        Args:
            name: Variable name

        Returns:
            The type if bound, None otherwise
        """
        return self._environment.lookup(name)

    def set_expected_type(self, ty: Type) -> None:
        """Set the current expected type.

        Args:
            ty: The expected type
        """
        self._expected_type = ty

    def push_scope(self) -> None:
        """Enter a new scope."""
        self._environment = self._environment.push_scope()

    def pop_scope(self) -> None:
        """Exit the current scope."""
        self._environment = self._environment.pop_scope()
