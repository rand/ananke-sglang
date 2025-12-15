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
"""Go incremental parser.

This module provides incremental parsing for Go code, supporting
hole detection and expected token computation for constrained decoding.

Features:
- Incremental parsing with checkpoint/restore
- Hole detection for incomplete code
- Expected token suggestions
- Bracket matching and context tracking
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

from parsing.base import (
    IncrementalParser,
    ParseResult,
    ParseState,
)
from parsing.partial_ast import HoleKind, HoleInfo


# =============================================================================
# Go Syntax Elements
# =============================================================================

GO_EXPRESSION_STARTERS: FrozenSet[str] = frozenset({
    # Literals
    "true", "false", "nil", "iota",
    # Unary operators
    "!", "&", "*", "+", "-", "^", "<-",
    # Keywords that start expressions
    "func", "map", "chan", "struct", "interface",
    # Delimiters that start expressions
    "(", "[", "{",
})

GO_STATEMENT_STARTERS: FrozenSet[str] = frozenset({
    "if", "for", "switch", "select",
    "go", "defer", "return", "break", "continue", "goto", "fallthrough",
    "var", "const", "type",
})

GO_DECLARATION_STARTERS: FrozenSet[str] = frozenset({
    "package", "import", "func", "type", "var", "const",
})

GO_BINARY_OPERATORS: FrozenSet[str] = frozenset({
    "+", "-", "*", "/", "%",
    "&", "|", "^", "&^", "<<", ">>",
    "==", "!=", "<", ">", "<=", ">=",
    "&&", "||",
    "=", ":=", "+=", "-=", "*=", "/=", "%=",
    "&=", "|=", "^=", "&^=", "<<=", ">>=",
})

GO_BRACKETS = {
    "(": ")",
    "[": "]",
    "{": "}",
}

GO_CLOSING_BRACKETS: FrozenSet[str] = frozenset({")", "]", "}"})


@dataclass
class GoParserState:
    """Mutable state for Go parser."""
    source: str = ""
    position: int = 0
    bracket_stack: List[str] = field(default_factory=list)
    context_stack: List[str] = field(default_factory=list)
    in_type_context: bool = False
    last_token: str = ""
    errors: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class GoParserCheckpoint:
    """Checkpoint for Go parser state."""
    source: str
    position: int
    bracket_stack: Tuple[str, ...]
    context_stack: Tuple[str, ...]
    in_type_context: bool
    last_token: str


class GoIncrementalParser(IncrementalParser):
    """Incremental parser for Go code."""

    def __init__(self):
        self._state = GoParserState()
        self._ast: Optional[Any] = None

    @property
    def language(self) -> str:
        return "go"

    @property
    def current_source(self) -> str:
        return self._state.source

    @property
    def current_position(self) -> int:
        return len(self._state.source)

    def parse_initial(self, source: str) -> ParseResult:
        """Parse initial source code."""
        self._state = GoParserState(source=source)
        self._update_context(source)
        return self._create_result()

    def extend_with_text(self, text: str) -> ParseResult:
        """Extend current source with new text."""
        self._state.source += text
        self._update_context(text)
        return self._create_result()

    def extend_with_token(self, token_id: int, token_text: str) -> ParseResult:
        """Extend with a single token."""
        return self.extend_with_text(token_text)

    def _update_context(self, new_text: str) -> None:
        """Update parser context based on new text."""
        for char in new_text:
            if char in GO_BRACKETS:
                self._state.bracket_stack.append(char)
            elif char in GO_CLOSING_BRACKETS:
                if self._state.bracket_stack:
                    open_bracket = self._state.bracket_stack[-1]
                    if GO_BRACKETS.get(open_bracket) == char:
                        self._state.bracket_stack.pop()

        # Track last meaningful token
        tokens = self._tokenize_simple(new_text)
        if tokens:
            self._state.last_token = tokens[-1]

        # Update type context
        self._update_type_context()

    def _update_type_context(self) -> None:
        """Update whether we're in a type context."""
        source = self._state.source.strip()

        # In type context after certain patterns
        type_context_patterns = [
            r":\s*$",           # After colon in declaration
            r"\[\s*$",         # In array/slice type
            r"map\[\s*$",      # In map key type
            r"\]\s*$",         # After map key, before value
            r"func\s*\(\s*$",  # In function params
            r"\)\s*$",         # After params, before return type
            r"chan\s*$",       # After chan keyword
            r"<-chan\s*$",     # After receive channel
            r"chan<-\s*$",     # After send channel
            r"\*\s*$",         # After pointer star
        ]

        self._state.in_type_context = any(
            re.search(pattern, source) for pattern in type_context_patterns
        )

    def _tokenize_simple(self, text: str) -> List[str]:
        """Simple tokenization for context tracking."""
        tokens = []
        current = []

        for char in text:
            if char.isspace():
                if current:
                    tokens.append("".join(current))
                    current = []
            elif char in "()[]{}.,;:":
                if current:
                    tokens.append("".join(current))
                    current = []
                tokens.append(char)
            elif char in "+-*/%&|^<>=!":
                if current:
                    tokens.append("".join(current))
                    current = []
                current.append(char)
            else:
                current.append(char)

        if current:
            tokens.append("".join(current))

        return tokens

    def _create_result(self) -> ParseResult:
        """Create parse result from current state."""
        has_unclosed_brackets = len(self._state.bracket_stack) > 0
        has_incomplete = self._has_incomplete_construct()

        if has_unclosed_brackets or has_incomplete:
            state = ParseState.PARTIAL
        elif self._state.errors:
            state = ParseState.ERROR
        else:
            # Check if we have a complete program
            source = self._state.source.strip()
            if source and not has_incomplete:
                state = ParseState.VALID
            else:
                state = ParseState.VALID

        return ParseResult(
            state=state,
            ast=self._ast,
            errors=[],
            holes=[],
            position=len(self._state.source),
        )

    def _has_incomplete_construct(self) -> bool:
        """Check if source has incomplete constructs."""
        source = self._state.source.strip()
        if not source:
            return False

        # Trailing binary operators indicate incomplete expression
        for op in GO_BINARY_OPERATORS:
            if source.endswith(op):
                return True

        # Trailing keywords that expect something
        trailing_keywords = [
            "func", "var", "const", "type", "import", "package",
            "if", "for", "switch", "select", "case", "default",
            "go", "defer", "return", "chan", "map", "struct", "interface",
        ]
        for kw in trailing_keywords:
            if source.endswith(kw) or source.endswith(kw + " "):
                return True

        return False

    def find_holes(self) -> List[Tuple[HoleKind, Any]]:
        """Find holes (incomplete parts) in the parsed code."""
        holes = []

        # Check for unclosed brackets
        for bracket in self._state.bracket_stack:
            if bracket == "(":
                holes.append((HoleKind.EXPRESSION, {"bracket": bracket}))
            elif bracket == "[":
                holes.append((HoleKind.EXPRESSION, {"bracket": bracket}))
            elif bracket == "{":
                holes.append((HoleKind.BODY, {"bracket": bracket}))

        # Check for incomplete declarations
        source = self._state.source.strip()

        # Incomplete function declaration
        if re.search(r"func\s+\w+\s*\([^)]*\)\s*(?:\([^)]*\)|[^{]*)?\s*$", source):
            if "{" not in source.split("func")[-1]:
                holes.append((HoleKind.BODY, {"kind": "function"}))

        # Incomplete type declaration
        if re.search(r"type\s+\w+\s+(?:struct|interface)\s*$", source):
            holes.append((HoleKind.BODY, {"kind": "type"}))

        # Incomplete variable declaration
        if re.search(r"(?:var|const)\s+\w+\s*(?:,\s*\w+)*\s*$", source):
            holes.append((HoleKind.TYPE, {"kind": "declaration"}))

        if re.search(r"(?:var|const)\s+\w+\s+\w+\s*=\s*$", source):
            holes.append((HoleKind.EXPRESSION, {"kind": "initializer"}))

        # Short declaration without value
        if re.search(r"\w+\s*:=\s*$", source):
            holes.append((HoleKind.EXPRESSION, {"kind": "short_declaration"}))

        # Trailing operators
        for op in GO_BINARY_OPERATORS:
            if source.endswith(op) or source.endswith(op + " "):
                holes.append((HoleKind.EXPRESSION, {"operator": op}))
                break

        return holes

    def get_expected_tokens(self) -> List[str]:
        """Get list of expected tokens at current position."""
        expected = []

        # Based on bracket stack
        if self._state.bracket_stack:
            last_bracket = self._state.bracket_stack[-1]
            expected.append(GO_BRACKETS[last_bracket])

        source = self._state.source.strip()

        # After package keyword
        if source.endswith("package"):
            expected.append("identifier")
            return expected

        # After import keyword
        if source.endswith("import"):
            expected.extend(["(", "string"])
            return expected

        # After func keyword
        if source.endswith("func"):
            expected.extend(["identifier", "("])
            return expected

        # In type context
        if self._state.in_type_context:
            expected.extend([
                "identifier", "*", "[]", "map", "chan", "func",
                "struct", "interface", "(", "[",
            ])
            return expected

        # After opening brace (in block)
        if source.endswith("{"):
            expected.extend(list(GO_STATEMENT_STARTERS))
            expected.extend(list(GO_EXPRESSION_STARTERS))
            expected.append("}")
            return expected

        # After binary operator
        for op in GO_BINARY_OPERATORS:
            if source.endswith(op):
                expected.extend(list(GO_EXPRESSION_STARTERS))
                expected.append("identifier")
                return expected

        # Default: statement or expression starters
        expected.extend(list(GO_STATEMENT_STARTERS))
        expected.extend(list(GO_EXPRESSION_STARTERS))
        expected.append("identifier")

        return list(set(expected))

    def checkpoint(self) -> GoParserCheckpoint:
        """Create a checkpoint of current state."""
        return GoParserCheckpoint(
            source=self._state.source,
            position=len(self._state.source),
            bracket_stack=tuple(self._state.bracket_stack),
            context_stack=tuple(self._state.context_stack),
            in_type_context=self._state.in_type_context,
            last_token=self._state.last_token,
        )

    def restore(self, checkpoint: GoParserCheckpoint) -> None:
        """Restore parser state from checkpoint."""
        self._state = GoParserState(
            source=checkpoint.source,
            position=checkpoint.position,
            bracket_stack=list(checkpoint.bracket_stack),
            context_stack=list(checkpoint.context_stack),
            in_type_context=checkpoint.in_type_context,
            last_token=checkpoint.last_token,
        )

    def copy(self) -> "GoIncrementalParser":
        """Create a copy of this parser."""
        new_parser = GoIncrementalParser()
        new_parser._state = GoParserState(
            source=self._state.source,
            position=self._state.position,
            bracket_stack=self._state.bracket_stack.copy(),
            context_stack=self._state.context_stack.copy(),
            in_type_context=self._state.in_type_context,
            last_token=self._state.last_token,
            errors=self._state.errors.copy(),
        )
        return new_parser

    def get_ast(self) -> Optional[Any]:
        """Get the current AST."""
        return self._ast

    def get_source(self) -> str:
        """Get the current source code."""
        return self._state.source


def create_go_parser() -> GoIncrementalParser:
    """Factory function to create a Go parser."""
    return GoIncrementalParser()
