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
"""Mark types for the marked lambda calculus (POPL 2024).

Marks indicate type inconsistencies and holes in partial programs.
Every AST node can have an optional mark indicating its status.

Mark Types:
- HoleMark: Empty hole awaiting a term
- InconsistentMark: Type mismatch between synthesized and expected
- NonEmptyHoleMark: Hole with partial content

The key insight is that marks localize errors - instead of the entire
program being "broken", specific locations are marked as problematic
while the rest of the program remains well-typed.

References:
    - POPL 2024: "Total Type Error Localization and Recovery with Holes"
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

# Support both relative imports (when used as subpackage) and absolute imports (standalone testing)
try:
    from ..constraint import Type
    from .provenance import Provenance
except ImportError:
    from domains.types.constraint import Type
    from domains.types.marking.provenance import Provenance


class Mark(ABC):
    """Abstract base class for marks.

    A mark indicates a type inconsistency location in a program.
    Marks allow type checking to continue even when errors exist,
    localizing problems to specific AST nodes.
    """

    @abstractmethod
    def is_error(self) -> bool:
        """Return True if this mark represents an error."""
        pass

    @abstractmethod
    def is_hole(self) -> bool:
        """Return True if this mark represents a hole."""
        pass

    @abstractmethod
    def synthesized_type(self) -> Optional[Type]:
        """Return the synthesized type at this mark, if any."""
        pass


@dataclass(frozen=True, slots=True)
class HoleMark(Mark):
    """An empty hole awaiting a term.

    HoleMarks represent gaps in the program that need to be filled.
    They have an expected type from context that constrains what
    can be placed there.

    During generation, new tokens fill holes. The expected_type
    guides which tokens are valid.

    Attributes:
        hole_id: Unique identifier for this hole
        expected_type: The type expected from context (may be None)
        provenance: Where this hole was introduced
    """

    hole_id: str
    expected_type: Optional[Type] = None
    provenance: Optional[Provenance] = None

    def is_error(self) -> bool:
        """Holes are not errors - they're valid incomplete states."""
        return False

    def is_hole(self) -> bool:
        return True

    def synthesized_type(self) -> Optional[Type]:
        """Holes don't have a synthesized type yet."""
        return None

    def with_expected_type(self, expected: Type) -> HoleMark:
        """Create a copy with updated expected type."""
        return HoleMark(
            hole_id=self.hole_id,
            expected_type=expected,
            provenance=self.provenance,
        )

    def __repr__(self) -> str:
        if self.expected_type:
            return f"HoleMark({self.hole_id!r}, expected={self.expected_type})"
        return f"HoleMark({self.hole_id!r})"


@dataclass(frozen=True, slots=True)
class InconsistentMark(Mark):
    """A type inconsistency between synthesized and expected types.

    InconsistentMarks represent actual type errors - places where
    the inferred type doesn't match what was expected from context.

    Unlike traditional type checkers that halt on errors, marked
    type checking continues by inserting these marks and propagating
    error information.

    Attributes:
        synthesized: The type that was inferred (synthesized)
        expected: The type that was expected from context (analyzed)
        provenance: Trace of where/why this inconsistency arose
    """

    synthesized: Type
    expected: Type
    provenance: Provenance

    def is_error(self) -> bool:
        """Inconsistencies are errors."""
        return True

    def is_hole(self) -> bool:
        return False

    def synthesized_type(self) -> Optional[Type]:
        return self.synthesized

    def error_message(self) -> str:
        """Generate a human-readable error message."""
        return (
            f"Type mismatch: expected {self.expected}, got {self.synthesized}"
            f" ({self.provenance.context})"
        )

    def __repr__(self) -> str:
        return (
            f"InconsistentMark(got={self.synthesized}, "
            f"expected={self.expected}, "
            f"at={self.provenance.location})"
        )


@dataclass(frozen=True, slots=True)
class NonEmptyHoleMark(Mark):
    """A hole with partial content.

    NonEmptyHoleMarks represent holes that have been partially filled
    but still contain sub-holes. The inner field holds the partial
    content with its own marks.

    This enables incremental filling - a function hole might be
    partially filled with a lambda that still has holes in its body.

    Attributes:
        hole_id: Unique identifier for this hole
        inner_type: The type of the partial content
        expected_type: The type expected from context
        provenance: Where this hole was introduced
    """

    hole_id: str
    inner_type: Type
    expected_type: Optional[Type] = None
    provenance: Optional[Provenance] = None

    def is_error(self) -> bool:
        """Non-empty holes are not errors."""
        return False

    def is_hole(self) -> bool:
        return True

    def synthesized_type(self) -> Optional[Type]:
        return self.inner_type

    def __repr__(self) -> str:
        if self.expected_type:
            return (
                f"NonEmptyHoleMark({self.hole_id!r}, "
                f"inner={self.inner_type}, expected={self.expected_type})"
            )
        return f"NonEmptyHoleMark({self.hole_id!r}, inner={self.inner_type})"


def is_marked_with_error(mark: Optional[Mark]) -> bool:
    """Check if a mark represents an error.

    Args:
        mark: The mark to check (may be None)

    Returns:
        True if the mark is an error mark
    """
    return mark is not None and mark.is_error()


def is_marked_with_hole(mark: Optional[Mark]) -> bool:
    """Check if a mark represents a hole.

    Args:
        mark: The mark to check (may be None)

    Returns:
        True if the mark is a hole mark
    """
    return mark is not None and mark.is_hole()


def create_hole_mark(
    hole_id: str,
    expected_type: Optional[Type] = None,
    provenance: Optional[Provenance] = None,
) -> HoleMark:
    """Create a hole mark.

    Args:
        hole_id: Unique identifier
        expected_type: Expected type from context
        provenance: Where the hole was introduced

    Returns:
        A new HoleMark
    """
    return HoleMark(
        hole_id=hole_id,
        expected_type=expected_type,
        provenance=provenance,
    )


def create_inconsistent_mark(
    synthesized: Type,
    expected: Type,
    provenance: Provenance,
) -> InconsistentMark:
    """Create an inconsistency mark.

    Args:
        synthesized: The inferred type
        expected: The expected type
        provenance: Error trace

    Returns:
        A new InconsistentMark
    """
    return InconsistentMark(
        synthesized=synthesized,
        expected=expected,
        provenance=provenance,
    )
