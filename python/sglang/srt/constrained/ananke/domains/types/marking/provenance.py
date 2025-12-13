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
"""Provenance tracking for error localization (POPL 2024).

Provenance traces type errors back to their source locations, enabling
precise error messages and incremental updates.

A provenance chain links each error to:
- The source location where it occurred
- The context (e.g., "function argument", "return type")
- Parent provenances for nested errors

References:
    - POPL 2024: "Total Type Error Localization and Recovery with Holes"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True, slots=True)
class SourceSpan:
    """A span of source code.

    Represents a contiguous region in source code, used to locate
    type errors precisely.

    Attributes:
        start: Starting byte offset (inclusive)
        end: Ending byte offset (exclusive)
        file: Optional filename or identifier
        start_line: Optional starting line number (1-indexed)
        start_col: Optional starting column number (1-indexed)
        end_line: Optional ending line number
        end_col: Optional ending column number
    """

    start: int
    end: int
    file: Optional[str] = None
    start_line: Optional[int] = None
    start_col: Optional[int] = None
    end_line: Optional[int] = None
    end_col: Optional[int] = None

    @property
    def length(self) -> int:
        """Return the length of this span in bytes."""
        return self.end - self.start

    def contains(self, offset: int) -> bool:
        """Check if this span contains a byte offset."""
        return self.start <= offset < self.end

    def overlaps(self, other: SourceSpan) -> bool:
        """Check if this span overlaps with another."""
        return self.start < other.end and other.start < self.end

    def merge(self, other: SourceSpan) -> SourceSpan:
        """Merge two spans into one covering both.

        Args:
            other: The span to merge with

        Returns:
            A new span covering both input spans
        """
        return SourceSpan(
            start=min(self.start, other.start),
            end=max(self.end, other.end),
            file=self.file if self.file == other.file else None,
        )

    def __repr__(self) -> str:
        if self.file:
            if self.start_line and self.start_col:
                return f"{self.file}:{self.start_line}:{self.start_col}"
            return f"{self.file}[{self.start}:{self.end}]"
        return f"[{self.start}:{self.end}]"


# Sentinel for unknown spans
UNKNOWN_SPAN = SourceSpan(start=0, end=0)


@dataclass(frozen=True, slots=True)
class Provenance:
    """Traces a type error back to its source.

    Provenance forms a chain linking errors to their locations and causes.
    Each provenance has:
    - A source location where the error occurred
    - A context describing what kind of construct was being checked
    - An optional parent for nested errors

    This enables:
    - Precise error messages ("in argument 2 of call to foo")
    - Error hierarchies (inner errors linked to outer context)
    - Incremental invalidation (which errors depend on which spans)

    Attributes:
        location: The source span where the error occurred
        context: Description of the syntactic context
        parent: Parent provenance for nested errors
        message: Optional additional error message
    """

    location: SourceSpan
    context: str
    parent: Optional[Provenance] = None
    message: Optional[str] = None

    def chain(self, child_location: SourceSpan, child_context: str) -> Provenance:
        """Create a child provenance linked to this one.

        Args:
            child_location: Location of the child error
            child_context: Context description for the child

        Returns:
            A new Provenance with this as parent
        """
        return Provenance(
            location=child_location,
            context=child_context,
            parent=self,
        )

    def with_message(self, message: str) -> Provenance:
        """Create a copy with an additional message.

        Args:
            message: The message to add

        Returns:
            A new Provenance with the message
        """
        return Provenance(
            location=self.location,
            context=self.context,
            parent=self.parent,
            message=message,
        )

    def root(self) -> Provenance:
        """Get the root provenance in the chain.

        Returns:
            The topmost provenance (with no parent)
        """
        current = self
        while current.parent is not None:
            current = current.parent
        return current

    def chain_length(self) -> int:
        """Get the length of the provenance chain.

        Returns:
            Number of provenances from this to root
        """
        length = 1
        current = self
        while current.parent is not None:
            length += 1
            current = current.parent
        return length

    def full_context(self) -> str:
        """Get the full context string from root to this.

        Returns:
            Context descriptions joined with " > "
        """
        contexts: List[str] = []
        current: Optional[Provenance] = self
        while current is not None:
            contexts.append(current.context)
            current = current.parent
        return " > ".join(reversed(contexts))

    def all_locations(self) -> List[SourceSpan]:
        """Get all source locations in the provenance chain.

        Returns:
            List of spans from root to this
        """
        locations: List[SourceSpan] = []
        current: Optional[Provenance] = self
        while current is not None:
            locations.append(current.location)
            current = current.parent
        return list(reversed(locations))

    def __repr__(self) -> str:
        base = f"Provenance({self.location}, {self.context!r}"
        if self.message:
            base += f", message={self.message!r}"
        if self.parent:
            base += f", parent=..."
        return base + ")"


# Common context strings
CONTEXT_FUNCTION_ARGUMENT = "function argument"
CONTEXT_FUNCTION_RETURN = "function return"
CONTEXT_VARIABLE_BINDING = "variable binding"
CONTEXT_IF_CONDITION = "if condition"
CONTEXT_IF_BRANCH = "if branch"
CONTEXT_ELSE_BRANCH = "else branch"
CONTEXT_LIST_ELEMENT = "list element"
CONTEXT_DICT_KEY = "dictionary key"
CONTEXT_DICT_VALUE = "dictionary value"
CONTEXT_TUPLE_ELEMENT = "tuple element"
CONTEXT_ASSIGNMENT = "assignment"
CONTEXT_BINARY_OPERATOR = "binary operator"
CONTEXT_UNARY_OPERATOR = "unary operator"
CONTEXT_ATTRIBUTE_ACCESS = "attribute access"
CONTEXT_SUBSCRIPT = "subscript"
CONTEXT_EXPRESSION = "expression"


def create_provenance(
    start: int,
    end: int,
    context: str,
    parent: Optional[Provenance] = None,
    file: Optional[str] = None,
) -> Provenance:
    """Convenience function to create a provenance.

    Args:
        start: Start offset
        end: End offset
        context: Context description
        parent: Optional parent provenance
        file: Optional filename

    Returns:
        A new Provenance
    """
    return Provenance(
        location=SourceSpan(start=start, end=end, file=file),
        context=context,
        parent=parent,
    )
