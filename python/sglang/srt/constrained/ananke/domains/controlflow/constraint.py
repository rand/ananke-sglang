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
"""Control flow constraint for CFG-based reachability.

The ControlFlowConstraint tracks:
- Must-reach points: Code locations that must be reachable
- Must-not-reach points: Code locations that must be unreachable
- Termination requirements: Whether functions must terminate

Control flow constraints form a semilattice where:
- TOP: No control flow restrictions
- BOTTOM: Conflicting requirements (same point must-reach and must-not-reach)
- meet(): Union of requirements
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import FrozenSet, Optional, Set

try:
    from ...core.constraint import Constraint, Satisfiability
except ImportError:
    from core.constraint import Constraint, Satisfiability


class TerminationRequirement(Enum):
    """Termination requirement for a code region."""

    MUST_TERMINATE = auto()
    MAY_NOT_TERMINATE = auto()
    UNKNOWN = auto()


class ReachabilityKind(Enum):
    """Kind of reachability requirement."""

    MUST_REACH = auto()
    MUST_NOT_REACH = auto()


@dataclass(frozen=True, slots=True)
class CodePoint:
    """A point in the control flow graph.

    Attributes:
        label: Unique identifier for this point
        kind: Optional description (e.g., "function_entry", "loop_exit")
        line: Optional source line number
    """

    label: str
    kind: Optional[str] = None
    line: Optional[int] = None

    def __repr__(self) -> str:
        if self.line is not None:
            return f"CodePoint({self.label}@L{self.line})"
        return f"CodePoint({self.label})"


@dataclass(frozen=True, slots=True)
class ReachabilityConstraintItem:
    """A single reachability requirement.

    Attributes:
        point: The code point
        kind: Whether it must or must not be reachable
        from_point: Optional source point (for path-specific constraints)
    """

    point: CodePoint
    kind: ReachabilityKind
    from_point: Optional[CodePoint] = None

    def conflicts_with(self, other: "ReachabilityConstraintItem") -> bool:
        """Check if this constraint conflicts with another."""
        if self.point != other.point:
            return False
        if self.from_point != other.from_point:
            return False
        return self.kind != other.kind


@dataclass(frozen=True)
class ControlFlowConstraint(Constraint["ControlFlowConstraint"]):
    """Constraint on control flow properties.

    Tracks:
    - must_reach: Points that must be reachable
    - must_not_reach: Points that must be unreachable
    - termination: Termination requirement for the current scope

    The constraint is satisfiable iff must_reach ∩ must_not_reach = ∅.

    Attributes:
        must_reach: Set of points that must be reachable
        must_not_reach: Set of points that must be unreachable
        termination: Termination requirement
        _is_top: True if this is TOP (no restrictions)
        _is_bottom: True if this is BOTTOM (unsatisfiable)
    """

    must_reach: FrozenSet[CodePoint] = field(default_factory=frozenset)
    must_not_reach: FrozenSet[CodePoint] = field(default_factory=frozenset)
    termination: TerminationRequirement = TerminationRequirement.UNKNOWN
    _is_top: bool = False
    _is_bottom: bool = False

    def meet(self, other: ControlFlowConstraint) -> ControlFlowConstraint:
        """Compute the meet (conjunction) of two control flow constraints.

        Meet combines constraints:
        - must_reach = union of both must_reach sets
        - must_not_reach = union of both must_not_reach sets
        - termination = stricter of the two requirements

        If must_reach ∩ must_not_reach ≠ ∅ after meet, return BOTTOM.

        Args:
            other: The constraint to meet with

        Returns:
            The combined constraint
        """
        if self._is_bottom or other._is_bottom:
            return CONTROLFLOW_BOTTOM

        if self._is_top:
            return other
        if other._is_top:
            return self

        # Combine must_reach sets
        combined_must_reach = self.must_reach | other.must_reach
        # Combine must_not_reach sets
        combined_must_not_reach = self.must_not_reach | other.must_not_reach

        # Check for conflict
        if combined_must_reach & combined_must_not_reach:
            return CONTROLFLOW_BOTTOM

        # Combine termination requirements
        combined_termination = self._combine_termination(
            self.termination, other.termination
        )

        return ControlFlowConstraint(
            must_reach=combined_must_reach,
            must_not_reach=combined_must_not_reach,
            termination=combined_termination,
        )

    @staticmethod
    def _combine_termination(
        a: TerminationRequirement, b: TerminationRequirement
    ) -> TerminationRequirement:
        """Combine two termination requirements."""
        if a == TerminationRequirement.MUST_TERMINATE:
            return a
        if b == TerminationRequirement.MUST_TERMINATE:
            return b
        if a == TerminationRequirement.MAY_NOT_TERMINATE:
            return a
        if b == TerminationRequirement.MAY_NOT_TERMINATE:
            return b
        return TerminationRequirement.UNKNOWN

    def is_top(self) -> bool:
        """Check if this is the TOP constraint (no restrictions)."""
        return self._is_top

    def is_bottom(self) -> bool:
        """Check if this is the BOTTOM constraint (unsatisfiable)."""
        return self._is_bottom

    def satisfiability(self) -> Satisfiability:
        """Check satisfiability of the control flow constraint.

        Returns:
            UNSAT if must_reach ∩ must_not_reach ≠ ∅
            SAT otherwise
        """
        if self._is_bottom:
            return Satisfiability.UNSAT

        # Check for conflict
        if self.must_reach & self.must_not_reach:
            return Satisfiability.UNSAT

        return Satisfiability.SAT

    def require_reach(self, point: CodePoint) -> ControlFlowConstraint:
        """Create a new constraint requiring a point to be reachable.

        Args:
            point: The code point that must be reachable

        Returns:
            New constraint with point required to be reachable
        """
        if self._is_bottom:
            return self
        if point in self.must_not_reach:
            return CONTROLFLOW_BOTTOM

        return ControlFlowConstraint(
            must_reach=self.must_reach | {point},
            must_not_reach=self.must_not_reach,
            termination=self.termination,
        )

    def require_not_reach(self, point: CodePoint) -> ControlFlowConstraint:
        """Create a new constraint requiring a point to be unreachable.

        Args:
            point: The code point that must be unreachable

        Returns:
            New constraint with point required to be unreachable
        """
        if self._is_bottom:
            return self
        if point in self.must_reach:
            return CONTROLFLOW_BOTTOM

        return ControlFlowConstraint(
            must_reach=self.must_reach,
            must_not_reach=self.must_not_reach | {point},
            termination=self.termination,
        )

    def require_termination(self) -> ControlFlowConstraint:
        """Create a new constraint requiring termination.

        Returns:
            New constraint with termination required
        """
        if self._is_bottom:
            return self

        return ControlFlowConstraint(
            must_reach=self.must_reach,
            must_not_reach=self.must_not_reach,
            termination=TerminationRequirement.MUST_TERMINATE,
        )

    def allow_non_termination(self) -> ControlFlowConstraint:
        """Create a new constraint allowing non-termination.

        Returns:
            New constraint allowing non-termination
        """
        if self._is_bottom:
            return self

        return ControlFlowConstraint(
            must_reach=self.must_reach,
            must_not_reach=self.must_not_reach,
            termination=TerminationRequirement.MAY_NOT_TERMINATE,
        )

    def is_must_reach(self, point: CodePoint) -> bool:
        """Check if a point is required to be reachable.

        Args:
            point: The code point to check

        Returns:
            True if the point must be reachable
        """
        return point in self.must_reach

    def is_must_not_reach(self, point: CodePoint) -> bool:
        """Check if a point is required to be unreachable.

        Args:
            point: The code point to check

        Returns:
            True if the point must be unreachable
        """
        return point in self.must_not_reach

    def requires_termination(self) -> bool:
        """Check if termination is required.

        Returns:
            True if termination is required
        """
        return self.termination == TerminationRequirement.MUST_TERMINATE

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ControlFlowConstraint):
            return NotImplemented
        return (
            self._is_top == other._is_top
            and self._is_bottom == other._is_bottom
            and self.must_reach == other.must_reach
            and self.must_not_reach == other.must_not_reach
            and self.termination == other.termination
        )

    def __hash__(self) -> int:
        if self._is_top:
            return hash("CONTROLFLOW_TOP")
        if self._is_bottom:
            return hash("CONTROLFLOW_BOTTOM")
        return hash((self.must_reach, self.must_not_reach, self.termination))

    def __repr__(self) -> str:
        if self._is_top:
            return "CONTROLFLOW_TOP"
        if self._is_bottom:
            return "CONTROLFLOW_BOTTOM"
        parts = []
        if self.must_reach:
            parts.append(f"must_reach={len(self.must_reach)}")
        if self.must_not_reach:
            parts.append(f"must_not_reach={len(self.must_not_reach)}")
        if self.termination != TerminationRequirement.UNKNOWN:
            parts.append(f"termination={self.termination.name}")
        return f"ControlFlowConstraint({', '.join(parts)})"


# Singleton instances
CONTROLFLOW_TOP = ControlFlowConstraint(_is_top=True)
CONTROLFLOW_BOTTOM = ControlFlowConstraint(_is_bottom=True)


# Factory functions
def controlflow_requiring_reach(*labels: str) -> ControlFlowConstraint:
    """Create a constraint requiring the given points to be reachable.

    Args:
        *labels: Labels of code points that must be reachable

    Returns:
        ControlFlowConstraint requiring those points
    """
    points = frozenset(CodePoint(label=lbl) for lbl in labels)
    return ControlFlowConstraint(must_reach=points)


def controlflow_forbidding_reach(*labels: str) -> ControlFlowConstraint:
    """Create a constraint forbidding the given points from being reachable.

    Args:
        *labels: Labels of code points that must be unreachable

    Returns:
        ControlFlowConstraint forbidding those points
    """
    points = frozenset(CodePoint(label=lbl) for lbl in labels)
    return ControlFlowConstraint(must_not_reach=points)


def controlflow_requiring_termination() -> ControlFlowConstraint:
    """Create a constraint requiring termination.

    Returns:
        ControlFlowConstraint requiring termination
    """
    return ControlFlowConstraint(termination=TerminationRequirement.MUST_TERMINATE)
