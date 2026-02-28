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
"""Semantic constraint for SMT-based property verification.

The SemanticConstraint tracks:
- Logical assertions that must hold
- Pre/post conditions from contracts
- Invariants that must be maintained

Semantic constraints form a semilattice where:
- TOP: No semantic restrictions
- BOTTOM: Contradictory assertions (UNSAT)
- meet(): Conjunction of formulas
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import FrozenSet, Optional, Tuple

try:
    from ...core.constraint import Constraint, Satisfiability
except ImportError:
    from core.constraint import Constraint, Satisfiability


class FormulaKind(Enum):
    """Kind of semantic formula."""

    ASSERTION = auto()  # Runtime assertion
    PRECONDITION = auto()  # Function precondition
    POSTCONDITION = auto()  # Function postcondition
    INVARIANT = auto()  # Loop or class invariant
    ASSUMPTION = auto()  # Assumed to be true
    GUARANTEE = auto()  # Must be proven true


@dataclass(frozen=True, slots=True)
class SMTFormula:
    """A semantic formula for SMT checking.

    Attributes:
        expression: String representation of the formula
        kind: Type of formula
        source: Optional source location
        name: Optional name for the formula
    """

    expression: str
    kind: FormulaKind = FormulaKind.ASSERTION
    source: Optional[str] = None
    name: Optional[str] = None

    def __repr__(self) -> str:
        name_part = f" ({self.name})" if self.name else ""
        return f"SMTFormula({self.expression!r}, {self.kind.name}{name_part})"

    def negate(self) -> SMTFormula:
        """Create the negation of this formula."""
        return SMTFormula(
            expression=f"not ({self.expression})",
            kind=self.kind,
            source=self.source,
            name=f"not_{self.name}" if self.name else None,
        )


@dataclass(frozen=True)
class SemanticConstraint(Constraint["SemanticConstraint"]):
    """Constraint on semantic properties.

    Tracks a set of formulas that must be satisfiable together.
    Uses SMT solving (via Z3 when available) to check satisfiability.

    Attributes:
        formulas: Set of formulas that must hold
        assumptions: Set of assumed formulas (context)
        _is_top: True if this is TOP (no restrictions)
        _is_bottom: True if this is BOTTOM (contradictory)
        _sat_cache: Cached satisfiability result
    """

    formulas: FrozenSet[SMTFormula] = field(default_factory=frozenset)
    assumptions: FrozenSet[SMTFormula] = field(default_factory=frozenset)
    _is_top: bool = False
    _is_bottom: bool = False

    def meet(self, other: SemanticConstraint) -> SemanticConstraint:
        """Compute the meet (conjunction) of two semantic constraints.

        Meet combines formulas:
        - formulas = union of both formula sets
        - assumptions = union of both assumption sets

        Satisfiability is not checked immediately (lazy evaluation).

        Args:
            other: The constraint to meet with

        Returns:
            The combined constraint
        """
        if self._is_bottom or other._is_bottom:
            return SEMANTIC_BOTTOM

        if self._is_top:
            return other
        if other._is_top:
            return self

        # Combine formula sets
        combined_formulas = self.formulas | other.formulas
        combined_assumptions = self.assumptions | other.assumptions

        return SemanticConstraint(
            formulas=combined_formulas,
            assumptions=combined_assumptions,
        )

    def is_top(self) -> bool:
        """Check if this is the TOP constraint (no restrictions)."""
        return self._is_top

    def is_bottom(self) -> bool:
        """Check if this is the BOTTOM constraint (contradictory)."""
        return self._is_bottom

    def satisfiability(self) -> Satisfiability:
        """Check satisfiability of the semantic constraint.

        Returns:
            SAT, UNSAT, or UNKNOWN based on formula analysis
        """
        if self._is_bottom:
            return Satisfiability.UNSAT
        if self._is_top:
            return Satisfiability.SAT
        if not self.formulas:
            return Satisfiability.SAT

        # For now, return UNKNOWN for non-trivial formulas
        # Full implementation would use Z3
        return Satisfiability.UNKNOWN

    def add_assertion(self, expression: str, name: Optional[str] = None) -> SemanticConstraint:
        """Add an assertion formula.

        Args:
            expression: The assertion expression
            name: Optional name for the assertion

        Returns:
            New constraint with assertion added
        """
        if self._is_bottom:
            return self

        formula = SMTFormula(
            expression=expression,
            kind=FormulaKind.ASSERTION,
            name=name,
        )
        return SemanticConstraint(
            formulas=self.formulas | {formula},
            assumptions=self.assumptions,
        )

    def add_precondition(self, expression: str, name: Optional[str] = None) -> SemanticConstraint:
        """Add a precondition formula.

        Args:
            expression: The precondition expression
            name: Optional name for the precondition

        Returns:
            New constraint with precondition added
        """
        if self._is_bottom:
            return self

        formula = SMTFormula(
            expression=expression,
            kind=FormulaKind.PRECONDITION,
            name=name,
        )
        return SemanticConstraint(
            formulas=self.formulas | {formula},
            assumptions=self.assumptions,
        )

    def add_postcondition(self, expression: str, name: Optional[str] = None) -> SemanticConstraint:
        """Add a postcondition formula.

        Args:
            expression: The postcondition expression
            name: Optional name for the postcondition

        Returns:
            New constraint with postcondition added
        """
        if self._is_bottom:
            return self

        formula = SMTFormula(
            expression=expression,
            kind=FormulaKind.POSTCONDITION,
            name=name,
        )
        return SemanticConstraint(
            formulas=self.formulas | {formula},
            assumptions=self.assumptions,
        )

    def add_invariant(self, expression: str, name: Optional[str] = None) -> SemanticConstraint:
        """Add an invariant formula.

        Args:
            expression: The invariant expression
            name: Optional name for the invariant

        Returns:
            New constraint with invariant added
        """
        if self._is_bottom:
            return self

        formula = SMTFormula(
            expression=expression,
            kind=FormulaKind.INVARIANT,
            name=name,
        )
        return SemanticConstraint(
            formulas=self.formulas | {formula},
            assumptions=self.assumptions,
        )

    def add_assumption(self, expression: str, name: Optional[str] = None) -> SemanticConstraint:
        """Add an assumption (context formula).

        Args:
            expression: The assumption expression
            name: Optional name for the assumption

        Returns:
            New constraint with assumption added
        """
        if self._is_bottom:
            return self

        formula = SMTFormula(
            expression=expression,
            kind=FormulaKind.ASSUMPTION,
            name=name,
        )
        return SemanticConstraint(
            formulas=self.formulas,
            assumptions=self.assumptions | {formula},
        )

    def get_formulas_by_kind(self, kind: FormulaKind) -> FrozenSet[SMTFormula]:
        """Get formulas of a specific kind.

        Args:
            kind: The formula kind to filter by

        Returns:
            FrozenSet of matching formulas
        """
        return frozenset(f for f in self.formulas if f.kind == kind)

    def get_assertions(self) -> FrozenSet[SMTFormula]:
        """Get all assertion formulas."""
        return self.get_formulas_by_kind(FormulaKind.ASSERTION)

    def get_preconditions(self) -> FrozenSet[SMTFormula]:
        """Get all precondition formulas."""
        return self.get_formulas_by_kind(FormulaKind.PRECONDITION)

    def get_postconditions(self) -> FrozenSet[SMTFormula]:
        """Get all postcondition formulas."""
        return self.get_formulas_by_kind(FormulaKind.POSTCONDITION)

    def get_invariants(self) -> FrozenSet[SMTFormula]:
        """Get all invariant formulas."""
        return self.get_formulas_by_kind(FormulaKind.INVARIANT)

    def formula_count(self) -> int:
        """Get the number of formulas."""
        return len(self.formulas)

    def assumption_count(self) -> int:
        """Get the number of assumptions."""
        return len(self.assumptions)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SemanticConstraint):
            return NotImplemented
        return (
            self._is_top == other._is_top
            and self._is_bottom == other._is_bottom
            and self.formulas == other.formulas
            and self.assumptions == other.assumptions
        )

    def __hash__(self) -> int:
        if self._is_top:
            return hash("SEMANTIC_TOP")
        if self._is_bottom:
            return hash("SEMANTIC_BOTTOM")
        return hash((self.formulas, self.assumptions))

    def __repr__(self) -> str:
        if self._is_top:
            return "SEMANTIC_TOP"
        if self._is_bottom:
            return "SEMANTIC_BOTTOM"
        parts = []
        if self.formulas:
            parts.append(f"formulas={len(self.formulas)}")
        if self.assumptions:
            parts.append(f"assumptions={len(self.assumptions)}")
        return f"SemanticConstraint({', '.join(parts)})"


# Singleton instances
SEMANTIC_TOP = SemanticConstraint(_is_top=True)
SEMANTIC_BOTTOM = SemanticConstraint(_is_bottom=True)


# Factory functions
def semantic_assertion(expression: str, name: Optional[str] = None) -> SemanticConstraint:
    """Create a constraint with a single assertion.

    Args:
        expression: The assertion expression
        name: Optional name

    Returns:
        SemanticConstraint with the assertion
    """
    formula = SMTFormula(expression=expression, kind=FormulaKind.ASSERTION, name=name)
    return SemanticConstraint(formulas=frozenset({formula}))


def semantic_precondition(expression: str, name: Optional[str] = None) -> SemanticConstraint:
    """Create a constraint with a single precondition.

    Args:
        expression: The precondition expression
        name: Optional name

    Returns:
        SemanticConstraint with the precondition
    """
    formula = SMTFormula(expression=expression, kind=FormulaKind.PRECONDITION, name=name)
    return SemanticConstraint(formulas=frozenset({formula}))


def semantic_postcondition(expression: str, name: Optional[str] = None) -> SemanticConstraint:
    """Create a constraint with a single postcondition.

    Args:
        expression: The postcondition expression
        name: Optional name

    Returns:
        SemanticConstraint with the postcondition
    """
    formula = SMTFormula(expression=expression, kind=FormulaKind.POSTCONDITION, name=name)
    return SemanticConstraint(formulas=frozenset({formula}))
