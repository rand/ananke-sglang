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
"""Base classes for incremental parsing.

This module defines the abstract base class for incremental parsers that
convert token streams into partial ASTs suitable for type checking.

Each language-specific parser implements IncrementalParser to handle:
- Token-by-token AST extension
- Hole detection in partial programs
- Source span tracking
- Recovery from parse errors

References:
    - tree-sitter: https://tree-sitter.github.io/tree-sitter/
    - OOPSLA 2025: "Incremental Bidirectional Typing via Order Maintenance"
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Callable, Iterator

from domains.types.marking.provenance import SourceSpan, UNKNOWN_SPAN
from domains.types.marking.marked_ast import (
    MarkedASTNode,
    ASTNodeKind,
    create_hole_node,
)


class ParseState(Enum):
    """State of the incremental parser."""

    VALID = auto()       # Current parse is valid
    PARTIAL = auto()     # Incomplete but recoverable
    ERROR = auto()       # Parse error encountered
    COMPLETE = auto()    # Program is complete


@dataclass(frozen=True, slots=True)
class ParseError:
    """Information about a parse error.

    Attributes:
        message: Human-readable error description
        span: Source location of the error
        expected: What was expected at this position
        found: What was actually found
    """

    message: str
    span: SourceSpan
    expected: Optional[List[str]] = None
    found: Optional[str] = None

    def __str__(self) -> str:
        result = f"ParseError at {self.span}: {self.message}"
        if self.expected:
            result += f" (expected: {', '.join(self.expected)})"
        if self.found:
            result += f" (found: {self.found})"
        return result


@dataclass
class ParseResult:
    """Result of parsing operations.

    Attributes:
        state: Current parse state
        ast: The partial or complete AST
        errors: List of parse errors encountered
        holes: List of hole locations in the AST
        position: Current position in the source
    """

    state: ParseState
    ast: Optional[MarkedASTNode]
    errors: List[ParseError] = field(default_factory=list)
    holes: List[str] = field(default_factory=list)  # Hole IDs
    position: int = 0

    @property
    def is_valid(self) -> bool:
        """Whether the parse result represents a valid partial program."""
        return self.state in (ParseState.VALID, ParseState.PARTIAL, ParseState.COMPLETE)

    @property
    def is_complete(self) -> bool:
        """Whether the program is syntactically complete."""
        return self.state == ParseState.COMPLETE

    @property
    def has_errors(self) -> bool:
        """Whether there are parse errors."""
        return len(self.errors) > 0


@dataclass
class TokenInfo:
    """Information about a token for parsing.

    Attributes:
        token_id: The token ID from the tokenizer
        text: The decoded text of the token
        position: Position in the source text
        length: Length of the token in characters
    """

    token_id: int
    text: str
    position: int
    length: int


class IncrementalParser(ABC):
    """Abstract base class for incremental parsers.

    An incremental parser maintains a partial AST that is extended
    token-by-token during generation. It supports:

    1. Token extension: Adding tokens and updating the AST
    2. Hole detection: Finding incomplete parts of the program
    3. Rollback: Restoring to previous states
    4. Error recovery: Continuing after parse errors

    Subclasses implement language-specific parsing using tree-sitter
    or other incremental parsing libraries.

    Example usage:
        >>> parser = PythonIncrementalParser()
        >>> result = parser.parse_initial("def foo(")
        >>> result = parser.extend_with_token(TokenInfo(42, "x", 8, 1))
        >>> holes = parser.find_holes()
    """

    @property
    @abstractmethod
    def language(self) -> str:
        """Return the language name (e.g., 'python', 'typescript')."""
        pass

    @property
    @abstractmethod
    def current_source(self) -> str:
        """Return the current source text."""
        pass

    @property
    @abstractmethod
    def current_position(self) -> int:
        """Return the current position in the source."""
        pass

    @abstractmethod
    def parse_initial(self, source: str) -> ParseResult:
        """Parse an initial source string.

        Args:
            source: The source code to parse

        Returns:
            ParseResult with the partial AST
        """
        pass

    @abstractmethod
    def extend_with_token(self, token: TokenInfo) -> ParseResult:
        """Extend the current parse with a new token.

        This is the core incremental operation. It should:
        1. Append the token text to the source
        2. Incrementally update the parse tree
        3. Update hole positions
        4. Return the new parse result

        Args:
            token: Information about the token to add

        Returns:
            ParseResult with the updated AST
        """
        pass

    @abstractmethod
    def extend_with_text(self, text: str) -> ParseResult:
        """Extend the current parse with raw text.

        Args:
            text: The text to append

        Returns:
            ParseResult with the updated AST
        """
        pass

    @abstractmethod
    def find_holes(self) -> List[Tuple[str, SourceSpan]]:
        """Find all holes in the current partial AST.

        A "hole" is a position where more code is expected.
        This includes:
        - Incomplete expressions (e.g., "x + ")
        - Missing function bodies
        - Unclosed delimiters
        - Placeholder positions

        Returns:
            List of (hole_id, span) tuples
        """
        pass

    @abstractmethod
    def get_expected_tokens(self) -> List[str]:
        """Get tokens that would be valid at the current position.

        This is used for constraint guidance during generation.

        Returns:
            List of expected token types or values
        """
        pass

    @abstractmethod
    def checkpoint(self) -> Any:
        """Create a checkpoint of the current parser state.

        Returns:
            Opaque checkpoint object that can be passed to restore()
        """
        pass

    @abstractmethod
    def restore(self, checkpoint: Any) -> None:
        """Restore parser state from a checkpoint.

        Args:
            checkpoint: A checkpoint created by checkpoint()
        """
        pass

    @abstractmethod
    def get_ast(self) -> Optional[MarkedASTNode]:
        """Get the current AST as a MarkedASTNode.

        Converts the internal parse tree to the marked AST
        representation used by the type system.

        Returns:
            The current AST, or None if parsing failed
        """
        pass

    def copy(self) -> 'IncrementalParser':
        """Create an independent copy of this parser.

        Default implementation uses checkpoint/restore.
        Subclasses may override for efficiency.

        Returns:
            A new parser with the same state
        """
        # Default implementation - subclasses should override
        raise NotImplementedError("Subclasses must implement copy()")

    def reset(self) -> None:
        """Reset the parser to its initial state."""
        self.parse_initial("")


class HoleDetector:
    """Utility class for detecting holes in ASTs.

    Analyzes partial ASTs to find positions where code is missing
    or incomplete. Used by incremental parsers to identify what
    needs to be generated.
    """

    @staticmethod
    def find_holes_in_ast(node: MarkedASTNode) -> List[Tuple[str, SourceSpan]]:
        """Find all holes in a MarkedASTNode tree.

        Args:
            node: The root node to search

        Returns:
            List of (hole_id, span) for each hole found
        """
        holes: List[Tuple[str, SourceSpan]] = []
        HoleDetector._find_holes_recursive(node, holes)
        return holes

    @staticmethod
    def _find_holes_recursive(
        node: MarkedASTNode,
        holes: List[Tuple[str, SourceSpan]]
    ) -> None:
        """Recursively find holes in the AST."""
        if node.kind == ASTNodeKind.HOLE:
            hole_id = node.node_id or f"hole_{len(holes)}"
            holes.append((hole_id, node.span))

        for child in node.children:
            HoleDetector._find_holes_recursive(child, holes)

    @staticmethod
    def find_first_hole(node: MarkedASTNode) -> Optional[Tuple[str, SourceSpan]]:
        """Find the first hole in the AST (depth-first).

        Args:
            node: The root node to search

        Returns:
            (hole_id, span) of the first hole, or None
        """
        holes = HoleDetector.find_holes_in_ast(node)
        return holes[0] if holes else None

    @staticmethod
    def count_holes(node: MarkedASTNode) -> int:
        """Count the number of holes in the AST.

        Args:
            node: The root node to search

        Returns:
            Number of holes
        """
        return len(HoleDetector.find_holes_in_ast(node))


class SourceTracker:
    """Tracks source positions during incremental parsing.

    Maintains a mapping from token positions to source spans,
    enabling accurate error reporting and type annotation.
    """

    def __init__(self) -> None:
        """Initialize the source tracker."""
        self._source: str = ""
        self._tokens: List[TokenInfo] = []
        self._line_offsets: List[int] = [0]

    @property
    def source(self) -> str:
        """Get the current source text."""
        return self._source

    @property
    def position(self) -> int:
        """Get the current position."""
        return len(self._source)

    def append(self, text: str) -> SourceSpan:
        """Append text and return its span.

        Args:
            text: Text to append

        Returns:
            SourceSpan covering the new text
        """
        start = len(self._source)
        self._source += text
        end = len(self._source)

        # Update line offsets
        for i, char in enumerate(text):
            if char == '\n':
                self._line_offsets.append(start + i + 1)

        # Calculate line/column
        start_line = self._find_line(start)
        start_col = start - self._line_offsets[start_line]
        end_line = self._find_line(end)
        end_col = end - self._line_offsets[end_line]

        return SourceSpan(
            start=start,
            end=end,
            start_line=start_line + 1,  # 1-indexed
            start_col=start_col + 1,
            end_line=end_line + 1,
            end_col=end_col + 1,
        )

    def append_token(self, token: TokenInfo) -> SourceSpan:
        """Append a token and track it.

        Args:
            token: Token to append

        Returns:
            SourceSpan covering the token
        """
        span = self.append(token.text)
        self._tokens.append(token)
        return span

    def span_at(self, start: int, end: int) -> SourceSpan:
        """Create a span for a given range.

        Args:
            start: Start offset
            end: End offset

        Returns:
            SourceSpan for the range
        """
        start_line = self._find_line(start)
        start_col = start - self._line_offsets[start_line]
        end_line = self._find_line(end)
        end_col = end - self._line_offsets[end_line]

        return SourceSpan(
            start=start,
            end=end,
            start_line=start_line + 1,
            start_col=start_col + 1,
            end_line=end_line + 1,
            end_col=end_col + 1,
        )

    def _find_line(self, offset: int) -> int:
        """Find the line number for an offset.

        Uses binary search for efficiency.
        """
        left, right = 0, len(self._line_offsets) - 1
        while left < right:
            mid = (left + right + 1) // 2
            if self._line_offsets[mid] <= offset:
                left = mid
            else:
                right = mid - 1
        return left

    def checkpoint(self) -> Tuple[str, List[TokenInfo], List[int]]:
        """Create a checkpoint of the current state."""
        return (
            self._source,
            self._tokens.copy(),
            self._line_offsets.copy(),
        )

    def restore(self, checkpoint: Tuple[str, List[TokenInfo], List[int]]) -> None:
        """Restore from a checkpoint."""
        self._source, self._tokens, self._line_offsets = checkpoint
        self._tokens = self._tokens.copy()
        self._line_offsets = self._line_offsets.copy()

    def reset(self) -> None:
        """Reset to initial state."""
        self._source = ""
        self._tokens = []
        self._line_offsets = [0]
