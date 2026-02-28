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
"""Base constraint algebra for the Ananke compositional constraint system.

This module defines the foundational constraint abstraction based on bounded
meet-semilattice theory. All constraint domains in Ananke must satisfy the
semilattice laws to ensure correct constraint composition and propagation.

Theoretical Foundation:
    Constraints form a bounded meet-semilattice ⟨C, ⊓, ⊤, ⊥⟩ where:
    - C is the set of constraints
    - ⊓ (meet) is constraint conjunction
    - ⊤ (top) is the trivial constraint (always satisfied)
    - ⊥ (bottom) is the absurd constraint (never satisfied)

Required Properties (enforced by property-based tests):
    c ⊓ ⊤ = c                        (identity)
    c ⊓ ⊥ = ⊥                        (annihilation)
    c ⊓ c = c                        (idempotence)
    c₁ ⊓ c₂ = c₂ ⊓ c₁               (commutativity)
    (c₁ ⊓ c₂) ⊓ c₃ = c₁ ⊓ (c₂ ⊓ c₃) (associativity)

References:
    - Hazel: "Statically Contextualizing LLMs with Typed Holes" (OOPSLA 2024)
    - Hazel: "Total Type Error Localization and Recovery with Holes" (POPL 2024)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Generic, TypeVar


class Satisfiability(Enum):
    """Represents the satisfiability status of a constraint.

    SAT: The constraint is satisfiable (at least one solution exists)
    UNSAT: The constraint is unsatisfiable (no solution exists)
    UNKNOWN: Satisfiability cannot be determined (e.g., undecidable or timeout)
    """

    SAT = auto()
    UNSAT = auto()
    UNKNOWN = auto()

    def __and__(self, other: Satisfiability) -> Satisfiability:
        """Combine satisfiability results conservatively.

        SAT ∧ SAT = SAT
        UNSAT ∧ _ = UNSAT
        _ ∧ UNSAT = UNSAT
        UNKNOWN ∧ SAT = UNKNOWN
        UNKNOWN ∧ UNKNOWN = UNKNOWN
        """
        if self == Satisfiability.UNSAT or other == Satisfiability.UNSAT:
            return Satisfiability.UNSAT
        if self == Satisfiability.UNKNOWN or other == Satisfiability.UNKNOWN:
            return Satisfiability.UNKNOWN
        return Satisfiability.SAT

    def __or__(self, other: Satisfiability) -> Satisfiability:
        """Disjunction of satisfiability (for alternative branches).

        SAT ∨ _ = SAT
        _ ∨ SAT = SAT
        UNSAT ∨ UNSAT = UNSAT
        UNKNOWN ∨ UNSAT = UNKNOWN
        """
        if self == Satisfiability.SAT or other == Satisfiability.SAT:
            return Satisfiability.SAT
        if self == Satisfiability.UNKNOWN or other == Satisfiability.UNKNOWN:
            return Satisfiability.UNKNOWN
        return Satisfiability.UNSAT


# Type variable for concrete constraint types
C = TypeVar("C", bound="Constraint")


class Constraint(ABC, Generic[C]):
    """Abstract base class for all constraints in the Ananke system.

    All constraint domains must implement this interface to participate in
    the compositional constraint system. Constraints form a bounded meet-
    semilattice, enabling systematic constraint composition and propagation.

    The key insight from Hazel research is that constraints should support
    "totality" - every partial program state should have a well-defined
    constraint, even if that constraint is BOTTOM (unsatisfiable).

    Type Parameters:
        C: The concrete constraint type (for proper typing of meet operation)

    Example:
        >>> class TypeConstraint(Constraint['TypeConstraint']):
        ...     def meet(self, other: TypeConstraint) -> TypeConstraint:
        ...         # Combine type constraints
        ...         ...
    """

    __slots__ = ()

    @abstractmethod
    def meet(self, other: C) -> C:
        """Compute the meet (greatest lower bound) of two constraints.

        The meet operation represents constraint conjunction - the result
        is a constraint that is satisfied only when BOTH input constraints
        are satisfied.

        Must satisfy semilattice laws:
            - Idempotent: c.meet(c) == c
            - Commutative: c1.meet(c2) == c2.meet(c1)
            - Associative: c1.meet(c2.meet(c3)) == c1.meet(c2).meet(c3)
            - Identity: c.meet(TOP) == c
            - Annihilation: c.meet(BOTTOM) == BOTTOM

        Args:
            other: The constraint to combine with this one

        Returns:
            The conjunction of both constraints
        """
        raise NotImplementedError

    @abstractmethod
    def is_top(self) -> bool:
        """Check if this is the top (trivial/unconstrained) element.

        The top element ⊤ represents no constraint at all - it is always
        satisfied. Every constraint c satisfies: c.meet(TOP) == c

        Returns:
            True if this constraint is equivalent to TOP
        """
        raise NotImplementedError

    @abstractmethod
    def is_bottom(self) -> bool:
        """Check if this is the bottom (absurd/unsatisfiable) element.

        The bottom element ⊥ represents an impossible constraint - it can
        never be satisfied. Every constraint c satisfies: c.meet(BOTTOM) == BOTTOM

        When a constraint becomes BOTTOM during generation, it indicates
        that the current generation path cannot produce valid output.

        Returns:
            True if this constraint is equivalent to BOTTOM
        """
        raise NotImplementedError

    @abstractmethod
    def satisfiability(self) -> Satisfiability:
        """Determine the satisfiability status of this constraint.

        This enables early termination when constraints become unsatisfiable,
        and guides the sampler toward satisfiable paths.

        Returns:
            SAT if the constraint can be satisfied
            UNSAT if the constraint cannot be satisfied (equivalent to BOTTOM)
            UNKNOWN if satisfiability cannot be determined
        """
        raise NotImplementedError

    def __and__(self, other: C) -> C:
        """Operator alias for meet: c1 & c2 == c1.meet(c2)"""
        return self.meet(other)

    def is_satisfiable(self) -> bool:
        """Convenience method to check if constraint is definitely satisfiable."""
        return self.satisfiability() == Satisfiability.SAT

    def is_unsatisfiable(self) -> bool:
        """Convenience method to check if constraint is definitely unsatisfiable."""
        return self.satisfiability() == Satisfiability.UNSAT

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        """Check structural equality of constraints.

        Two constraints are equal if they represent the same set of
        valid solutions. This is used for fixpoint detection in
        constraint propagation.
        """
        raise NotImplementedError

    @abstractmethod
    def __hash__(self) -> int:
        """Hash for use in sets and dicts.

        Must be consistent with __eq__: if c1 == c2, then hash(c1) == hash(c2)
        """
        raise NotImplementedError


class TopConstraint(Constraint["TopConstraint"]):
    """The trivial constraint that is always satisfied.

    TOP represents the absence of any constraint - all values satisfy it.
    It serves as the identity element for the meet operation.

    This is a singleton class - use TOP instead of instantiating directly.
    """

    __slots__ = ()

    _instance: TopConstraint | None = None

    def __new__(cls) -> TopConstraint:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def meet(self, other: Constraint) -> Constraint:
        """Meet with TOP returns the other constraint (identity law)."""
        return other

    def is_top(self) -> bool:
        return True

    def is_bottom(self) -> bool:
        return False

    def satisfiability(self) -> Satisfiability:
        return Satisfiability.SAT

    def __eq__(self, other: object) -> bool:
        return isinstance(other, TopConstraint)

    def __hash__(self) -> int:
        return hash("TOP")

    def __repr__(self) -> str:
        return "TOP"


class BottomConstraint(Constraint["BottomConstraint"]):
    """The absurd constraint that can never be satisfied.

    BOTTOM represents an impossible constraint - no values satisfy it.
    It serves as the absorbing element for the meet operation.

    This is a singleton class - use BOTTOM instead of instantiating directly.
    """

    __slots__ = ()

    _instance: BottomConstraint | None = None

    def __new__(cls) -> BottomConstraint:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def meet(self, other: Constraint) -> BottomConstraint:
        """Meet with BOTTOM returns BOTTOM (annihilation law)."""
        return self

    def is_top(self) -> bool:
        return False

    def is_bottom(self) -> bool:
        return True

    def satisfiability(self) -> Satisfiability:
        return Satisfiability.UNSAT

    def __eq__(self, other: object) -> bool:
        return isinstance(other, BottomConstraint)

    def __hash__(self) -> int:
        return hash("BOTTOM")

    def __repr__(self) -> str:
        return "BOTTOM"


# Singleton instances for use throughout the system
TOP: TopConstraint = TopConstraint()
BOTTOM: BottomConstraint = BottomConstraint()


def verify_semilattice_laws(c1: Constraint, c2: Constraint, c3: Constraint) -> bool:
    """Verify that constraints satisfy the semilattice laws.

    This function is used in property-based tests to ensure all constraint
    implementations correctly satisfy the required algebraic properties.

    Args:
        c1, c2, c3: Three arbitrary constraints to test

    Returns:
        True if all semilattice laws hold

    Raises:
        AssertionError: If any law is violated, with descriptive message
    """
    # Identity law: c ⊓ ⊤ = c
    assert c1.meet(TOP) == c1, f"Identity law violated: {c1}.meet(TOP) != {c1}"

    # Annihilation law: c ⊓ ⊥ = ⊥
    assert c1.meet(BOTTOM) == BOTTOM, f"Annihilation law violated: {c1}.meet(BOTTOM) != BOTTOM"

    # Idempotence: c ⊓ c = c
    assert c1.meet(c1) == c1, f"Idempotence violated: {c1}.meet({c1}) != {c1}"

    # Commutativity: c1 ⊓ c2 = c2 ⊓ c1
    meet_12 = c1.meet(c2)
    meet_21 = c2.meet(c1)
    assert meet_12 == meet_21, f"Commutativity violated: {c1}.meet({c2}) != {c2}.meet({c1})"

    # Associativity: (c1 ⊓ c2) ⊓ c3 = c1 ⊓ (c2 ⊓ c3)
    left_assoc = c1.meet(c2).meet(c3)
    right_assoc = c1.meet(c2.meet(c3))
    assert left_assoc == right_assoc, (
        f"Associativity violated: ({c1}.meet({c2})).meet({c3}) != {c1}.meet({c2}.meet({c3}))"
    )

    return True
