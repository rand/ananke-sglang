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
"""Swift incremental parser.

This module provides incremental parsing for Swift code, supporting
hole detection and expected token computation for constrained decoding.

Features:
- Incremental parsing with checkpoint/restore
- Hole detection for incomplete code
- Expected token suggestions
- Bracket matching and context tracking

References:
    - Swift Grammar: https://docs.swift.org/swift-book/ReferenceManual/
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
from parsing.partial_ast import HoleKind


# =============================================================================
# Swift Syntax Elements
# =============================================================================

SWIFT_EXPRESSION_STARTERS: FrozenSet[str] = frozenset({
    # Literals
    "true", "false", "nil",
    # Keywords that start expressions
    "self", "super", "try", "throw",
    # Unary operators
    "!", "-", "+", "~",
    # Delimiters that start expressions
    "(", "[", "{",
    # Closures
    "#",  # #selector, #keyPath
})

SWIFT_STATEMENT_STARTERS: FrozenSet[str] = frozenset({
    "if", "guard", "switch", "for", "while", "repeat",
    "do", "try", "throw", "return", "break", "continue", "fallthrough",
    "defer", "where",
})

SWIFT_DECLARATION_STARTERS: FrozenSet[str] = frozenset({
    "import", "let", "var", "func", "class", "struct", "enum",
    "protocol", "extension", "typealias", "associatedtype",
    "init", "deinit", "subscript", "operator", "precedencegroup",
    "actor",
})

SWIFT_MODIFIERS: FrozenSet[str] = frozenset({
    "public", "private", "fileprivate", "internal", "open",
    "static", "class", "final", "override", "required", "convenience",
    "lazy", "weak", "unowned", "mutating", "nonmutating",
    "dynamic", "optional", "indirect",
    "async", "await", "throws", "rethrows",
    "inout", "some", "any",
})

SWIFT_BINARY_OPERATORS: FrozenSet[str] = frozenset({
    "+", "-", "*", "/", "%",
    "&+", "&-", "&*",
    "==", "!=", "===", "!==",
    "<", ">", "<=", ">=",
    "&&", "||",
    "&", "|", "^", "~",
    "<<", ">>",
    "=", "+=", "-=", "*=", "/=", "%=",
    "&=", "|=", "^=", "<<=", ">>=",
    "??",
    "...", "..<",
    "~=",
    "->",
})

SWIFT_BRACKETS = {
    "(": ")",
    "[": "]",
    "{": "}",
    "<": ">",
}

SWIFT_CLOSING_BRACKETS: FrozenSet[str] = frozenset({")", "]", "}", ">"})


@dataclass
class SwiftParserState:
    """Mutable state for Swift parser."""
    source: str = ""
    position: int = 0
    bracket_stack: List[str] = field(default_factory=list)
    context_stack: List[str] = field(default_factory=list)
    in_type_context: bool = False
    in_closure_context: bool = False
    last_token: str = ""
    errors: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class SwiftParserCheckpoint:
    """Checkpoint for Swift parser state."""
    source: str
    position: int
    bracket_stack: Tuple[str, ...]
    context_stack: Tuple[str, ...]
    in_type_context: bool
    in_closure_context: bool
    last_token: str


class SwiftIncrementalParser(IncrementalParser):
    """Incremental parser for Swift code."""

    def __init__(self):
        self._state = SwiftParserState()
        self._ast: Optional[Any] = None

    @property
    def language(self) -> str:
        return "swift"

    @property
    def current_source(self) -> str:
        return self._state.source

    @property
    def current_position(self) -> int:
        return len(self._state.source)

    def parse_initial(self, source: str) -> ParseResult:
        """Parse initial source code."""
        self._state = SwiftParserState(source=source)
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
            if char in SWIFT_BRACKETS:
                self._state.bracket_stack.append(char)
                if char == "{":
                    self._state.in_closure_context = True
            elif char in SWIFT_CLOSING_BRACKETS:
                if self._state.bracket_stack:
                    open_bracket = self._state.bracket_stack[-1]
                    if SWIFT_BRACKETS.get(open_bracket) == char:
                        self._state.bracket_stack.pop()
                        if char == "}":
                            self._state.in_closure_context = False

        # Track last meaningful token
        tokens = self._tokenize_simple(new_text)
        if tokens:
            self._state.last_token = tokens[-1]

        # Update type context
        self._update_type_context()

    def _update_type_context(self) -> None:
        """Update whether we're in a type context."""
        source = self._state.source.strip()

        type_context_patterns = [
            r":\s*$",           # After colon (type annotation)
            r"->\s*$",          # After arrow (return type)
            r"<\s*$",           # In generic type params
            r",\s*$",           # After comma in generics
            r"as\??\s*$",       # After cast
            r"is\s*$",          # After type check
            r"where\s*$",       # In generic constraints
        ]

        self._state.in_type_context = any(
            re.search(pattern, source) for pattern in type_context_patterns
        )

    def _tokenize_simple(self, text: str) -> List[str]:
        """Simple tokenization for context tracking."""
        tokens = []
        current: List[str] = []

        for char in text:
            if char.isspace():
                if current:
                    tokens.append("".join(current))
                    current = []
            elif char in "()[]{}.,;:<>@#":
                if current:
                    tokens.append("".join(current))
                    current = []
                tokens.append(char)
            elif char in "+-*/%&|^=!?~":
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

        # Trailing binary operators
        for op in SWIFT_BINARY_OPERATORS:
            if source.endswith(op) and op not in (")", "]", "}"):
                return True

        # Trailing keywords
        trailing_keywords = [
            "func", "let", "var", "class", "struct", "enum", "protocol",
            "extension", "import", "typealias",
            "if", "guard", "switch", "for", "while", "do",
            "return", "throw", "try", "catch",
            "async", "await", "throws",
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
            elif bracket == "<":
                holes.append((HoleKind.TYPE, {"bracket": bracket}))

        source = self._state.source.strip()

        # Incomplete function declaration
        if re.search(r"func\s+\w+\s*\([^)]*\)\s*(?:->\s*\w+)?\s*$", source):
            if "{" not in source.split("func")[-1]:
                holes.append((HoleKind.BODY, {"kind": "function"}))

        # Incomplete class/struct/enum declaration
        if re.search(r"(?:class|struct|enum)\s+\w+(?:\s*:\s*[^{]*)?\s*$", source):
            holes.append((HoleKind.BODY, {"kind": "type"}))

        # Incomplete variable declaration
        if re.search(r"(?:let|var)\s+\w+\s*(?::\s*\w+)?\s*=\s*$", source):
            holes.append((HoleKind.EXPRESSION, {"kind": "initializer"}))

        # Incomplete type annotation
        if re.search(r"(?:let|var)\s+\w+\s*:\s*$", source):
            holes.append((HoleKind.TYPE, {"kind": "declaration"}))

        # Trailing operators
        for op in SWIFT_BINARY_OPERATORS:
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
            expected.append(SWIFT_BRACKETS[last_bracket])

        source = self._state.source.strip()

        # After import keyword
        if source.endswith("import"):
            expected.append("identifier")
            return expected

        # After func keyword
        if source.endswith("func"):
            expected.append("identifier")
            return expected

        # After class/struct/enum
        if source.endswith(("class", "struct", "enum", "protocol")):
            expected.append("identifier")
            return expected

        # After let/var
        if source.endswith(("let", "var")):
            expected.append("identifier")
            return expected

        # In type context
        if self._state.in_type_context:
            expected.extend([
                "identifier", "?", "!", "[", "(", "some", "any",
                "Int", "String", "Bool", "Double", "Float",
                "Array", "Dictionary", "Set", "Optional",
            ])
            return expected

        # After opening brace
        if source.endswith("{"):
            expected.extend(list(SWIFT_STATEMENT_STARTERS))
            expected.extend(list(SWIFT_DECLARATION_STARTERS))
            expected.extend(list(SWIFT_EXPRESSION_STARTERS))
            expected.append("}")
            return expected

        # After binary operator
        for op in SWIFT_BINARY_OPERATORS:
            if source.endswith(op):
                expected.extend(list(SWIFT_EXPRESSION_STARTERS))
                expected.append("identifier")
                return expected

        # Default
        expected.extend(list(SWIFT_STATEMENT_STARTERS))
        expected.extend(list(SWIFT_DECLARATION_STARTERS))
        expected.extend(list(SWIFT_EXPRESSION_STARTERS))
        expected.append("identifier")

        return list(set(expected))

    def checkpoint(self) -> SwiftParserCheckpoint:
        """Create a checkpoint of current state."""
        return SwiftParserCheckpoint(
            source=self._state.source,
            position=len(self._state.source),
            bracket_stack=tuple(self._state.bracket_stack),
            context_stack=tuple(self._state.context_stack),
            in_type_context=self._state.in_type_context,
            in_closure_context=self._state.in_closure_context,
            last_token=self._state.last_token,
        )

    def restore(self, checkpoint: SwiftParserCheckpoint) -> None:
        """Restore parser state from checkpoint."""
        self._state = SwiftParserState(
            source=checkpoint.source,
            position=checkpoint.position,
            bracket_stack=list(checkpoint.bracket_stack),
            context_stack=list(checkpoint.context_stack),
            in_type_context=checkpoint.in_type_context,
            in_closure_context=checkpoint.in_closure_context,
            last_token=checkpoint.last_token,
        )

    def copy(self) -> "SwiftIncrementalParser":
        """Create a copy of this parser."""
        new_parser = SwiftIncrementalParser()
        new_parser._state = SwiftParserState(
            source=self._state.source,
            position=self._state.position,
            bracket_stack=self._state.bracket_stack.copy(),
            context_stack=self._state.context_stack.copy(),
            in_type_context=self._state.in_type_context,
            in_closure_context=self._state.in_closure_context,
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


def create_swift_parser() -> SwiftIncrementalParser:
    """Factory function to create a Swift parser."""
    return SwiftIncrementalParser()
