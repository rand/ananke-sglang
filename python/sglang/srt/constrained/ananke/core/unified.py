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
"""Unified constraint as the product of all constraint domains.

This module defines UnifiedConstraint, which combines constraints from all
five domains (syntax, types, imports, control flow, semantics) into a single
product type. The meet operation is component-wise, preserving the semilattice
structure across the entire constraint space.

The unified constraint enables:
1. Single point of truth for all active constraints
2. Component-wise meet for constraint conjunction
3. Efficient satisfiability checking via early termination
4. Cross-domain constraint propagation coordination

References:
    - Hazel: "Total Type Error Localization and Recovery with Holes" (POPL 2024)
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

from .constraint import (
    BOTTOM,
    TOP,
    BottomConstraint,
    Constraint,
    Satisfiability,
    TopConstraint,
)


@dataclass(frozen=True, slots=True)
class UnifiedConstraint(Constraint["UnifiedConstraint"]):
    """Product type of all five constraint domains.

    UnifiedConstraint combines constraints from:
    - Syntax: Grammar-based structural constraints (via llguidance)
    - Types: Type system constraints with incremental checking
    - Imports: Module/package availability constraints
    - Control Flow: CFG-based reachability constraints
    - Semantics: SMT-based semantic constraints

    The meet operation is component-wise:
        (s1, t1, i1, c1, m1) ⊓ (s2, t2, i2, c2, m2) =
        (s1 ⊓ s2, t1 ⊓ t2, i1 ⊓ i2, c1 ⊓ c2, m1 ⊓ m2)

    If any component becomes BOTTOM, the entire constraint is BOTTOM.

    Attributes:
        syntax: Syntax/grammar constraint
        types: Type system constraint
        imports: Import/module constraint
        controlflow: Control flow constraint
        semantics: Semantic/SMT constraint
    """

    syntax: Constraint = TOP
    types: Constraint = TOP
    imports: Constraint = TOP
    controlflow: Constraint = TOP
    semantics: Constraint = TOP

    def meet(self, other: UnifiedConstraint) -> UnifiedConstraint:
        """Component-wise meet of unified constraints.

        Each domain's constraint is combined using that domain's meet
        operation. If any domain becomes BOTTOM, the result represents
        an unsatisfiable state.

        Args:
            other: The unified constraint to combine with

        Returns:
            A new unified constraint with component-wise meets
        """
        return UnifiedConstraint(
            syntax=self.syntax.meet(other.syntax),
            types=self.types.meet(other.types),
            imports=self.imports.meet(other.imports),
            controlflow=self.controlflow.meet(other.controlflow),
            semantics=self.semantics.meet(other.semantics),
        )

    def is_top(self) -> bool:
        """Check if all components are TOP (no constraints)."""
        return (
            self.syntax.is_top()
            and self.types.is_top()
            and self.imports.is_top()
            and self.controlflow.is_top()
            and self.semantics.is_top()
        )

    def is_bottom(self) -> bool:
        """Check if any component is BOTTOM (unsatisfiable).

        A unified constraint is BOTTOM if ANY component is BOTTOM,
        since all constraints must be satisfied simultaneously.
        """
        return (
            self.syntax.is_bottom()
            or self.types.is_bottom()
            or self.imports.is_bottom()
            or self.controlflow.is_bottom()
            or self.semantics.is_bottom()
        )

    def satisfiability(self) -> Satisfiability:
        """Determine satisfiability with early termination.

        Returns UNSAT as soon as any component is unsatisfiable.
        Returns UNKNOWN if any component is unknown and none are unsat.
        Returns SAT only if all components are satisfiable.
        """
        has_unknown = False

        for constraint in [
            self.syntax,
            self.types,
            self.imports,
            self.controlflow,
            self.semantics,
        ]:
            sat = constraint.satisfiability()
            if sat == Satisfiability.UNSAT:
                return Satisfiability.UNSAT
            if sat == Satisfiability.UNKNOWN:
                has_unknown = True

        return Satisfiability.UNKNOWN if has_unknown else Satisfiability.SAT

    def with_syntax(self, syntax: Constraint) -> UnifiedConstraint:
        """Return a new constraint with updated syntax component."""
        return replace(self, syntax=syntax)

    def with_types(self, types: Constraint) -> UnifiedConstraint:
        """Return a new constraint with updated types component."""
        return replace(self, types=types)

    def with_imports(self, imports: Constraint) -> UnifiedConstraint:
        """Return a new constraint with updated imports component."""
        return replace(self, imports=imports)

    def with_controlflow(self, controlflow: Constraint) -> UnifiedConstraint:
        """Return a new constraint with updated controlflow component."""
        return replace(self, controlflow=controlflow)

    def with_semantics(self, semantics: Constraint) -> UnifiedConstraint:
        """Return a new constraint with updated semantics component."""
        return replace(self, semantics=semantics)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, UnifiedConstraint):
            return NotImplemented
        return (
            self.syntax == other.syntax
            and self.types == other.types
            and self.imports == other.imports
            and self.controlflow == other.controlflow
            and self.semantics == other.semantics
        )

    def __hash__(self) -> int:
        return hash(
            (
                hash(self.syntax),
                hash(self.types),
                hash(self.imports),
                hash(self.controlflow),
                hash(self.semantics),
            )
        )

    def __repr__(self) -> str:
        parts = []
        if not self.syntax.is_top():
            parts.append(f"syntax={self.syntax}")
        if not self.types.is_top():
            parts.append(f"types={self.types}")
        if not self.imports.is_top():
            parts.append(f"imports={self.imports}")
        if not self.controlflow.is_top():
            parts.append(f"controlflow={self.controlflow}")
        if not self.semantics.is_top():
            parts.append(f"semantics={self.semantics}")

        if not parts:
            return "UnifiedConstraint(TOP)"
        return f"UnifiedConstraint({', '.join(parts)})"


# Singleton instances for common cases
UNIFIED_TOP = UnifiedConstraint()
UNIFIED_BOTTOM = UnifiedConstraint(
    syntax=BOTTOM,
    types=BOTTOM,
    imports=BOTTOM,
    controlflow=BOTTOM,
    semantics=BOTTOM,
)


def unified_from_syntax(syntax: Constraint) -> UnifiedConstraint:
    """Create a unified constraint with only syntax constraint active."""
    return UnifiedConstraint(syntax=syntax)


def unified_from_types(types: Constraint) -> UnifiedConstraint:
    """Create a unified constraint with only types constraint active."""
    return UnifiedConstraint(types=types)


def unified_from_imports(imports: Constraint) -> UnifiedConstraint:
    """Create a unified constraint with only imports constraint active."""
    return UnifiedConstraint(imports=imports)


def unified_from_controlflow(controlflow: Constraint) -> UnifiedConstraint:
    """Create a unified constraint with only controlflow constraint active."""
    return UnifiedConstraint(controlflow=controlflow)


def unified_from_semantics(semantics: Constraint) -> UnifiedConstraint:
    """Create a unified constraint with only semantics constraint active."""
    return UnifiedConstraint(semantics=semantics)
