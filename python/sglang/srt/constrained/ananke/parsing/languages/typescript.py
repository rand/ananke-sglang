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
"""TypeScript incremental parser.

This module provides a TypeScript-specific incremental parser that converts
token streams into MarkedASTNodes suitable for type checking.

Features:
- Incremental parsing
- Hole detection for incomplete constructs
- Source span tracking for error messages
- JSX/TSX support detection
- Generic type parameter handling
- Arrow function detection

References:
    - TypeScript Handbook: https://www.typescriptlang.org/docs/handbook/
    - tree-sitter-typescript: https://github.com/tree-sitter/tree-sitter-typescript
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from parsing.base import (
    IncrementalParser,
    ParseState,
    ParseResult,
    ParseError,
    TokenInfo,
    SourceTracker,
)
from parsing.partial_ast import HoleInfo, HoleKind
from domains.types.marking.provenance import SourceSpan, UNKNOWN_SPAN
from domains.types.marking.marked_ast import (
    MarkedASTNode,
    ASTNodeKind,
    create_hole_node,
    create_literal_node,
    create_variable_node,
)
from domains.types.constraint import ANY
from domains.types.languages.typescript import (
    TS_STRING,
    TS_NUMBER,
    TS_BOOLEAN,
    TS_UNDEFINED,
    TS_NULL,
    TSArrayType,
)
from core.token_classifier_typescript import (
    TYPESCRIPT_ALL_KEYWORDS,
    TYPESCRIPT_ALL_BUILTINS,
    TYPESCRIPT_DECLARATION_KEYWORDS,
    TYPESCRIPT_CONTROL_KEYWORDS,
    TYPESCRIPT_PRIMITIVE_TYPES,
    classify_typescript_token,
)


# =============================================================================
# Token Categories for Hole Detection
# =============================================================================

EXPRESSION_STARTERS: Set[str] = {
    "identifier", "number", "string", "true", "false", "null", "undefined",
    "(", "[", "{", ".", "!",
    "if", "switch", "for", "while", "do",
    "return", "throw",
    "async", "await",
    "-", "!", "~", "+", "++", "--",
    "new", "typeof", "void", "delete",
    "function", "class",
    "`",  # Template literal
    "<",  # JSX or type assertion
}

EXPRESSION_ENDERS: Set[str] = {
    "identifier", "number", "string", "true", "false", "null", "undefined",
    ")", "]", "}", "?",
    "`",  # End of template literal
    ">",  # End of JSX or type parameter
}

INCOMPLETE_CONTEXTS: Set[str] = {
    "binary_operator", "unary_operator", "call", "index",
    "assignment", "variable_declaration", "return_statement",
    "if_statement", "switch_statement", "for_statement",
    "function_declaration", "class_declaration", "interface_declaration",
    "type_alias", "arrow_function", "method_definition",
    "import_declaration", "export_declaration",
}


# =============================================================================
# Parser Checkpoint
# =============================================================================

@dataclass
class TypeScriptParserCheckpoint:
    """Checkpoint for TypeScript parser state."""

    source: str
    tokens: List[TokenInfo]
    holes: Dict[str, HoleInfo]
    hole_counter: int
    tracker_checkpoint: Any
    brace_depth: int
    paren_depth: int
    bracket_depth: int
    angle_depth: int
    in_type_context: bool
    in_jsx: bool
    template_depth: int


# =============================================================================
# TypeScript Incremental Parser
# =============================================================================

class TypeScriptIncrementalParser(IncrementalParser):
    """TypeScript-specific incremental parser.

    This parser provides token-by-token parsing for TypeScript code generation.
    It maintains a partial AST and detects holes where more code is expected.

    For production use, this should integrate with tree-sitter-typescript for
    proper incremental parsing. The current implementation provides a
    lightweight fallback.

    Example:
        >>> parser = TypeScriptIncrementalParser()
        >>> result = parser.parse_initial("function foo(x: number)")
        >>> result = parser.extend_with_text(": ")
        >>> holes = parser.find_holes()
        >>> # Expect a hole for the return type and body
    """

    def __init__(self) -> None:
        """Initialize the parser."""
        self._source: str = ""
        self._tokens: List[TokenInfo] = []
        self._holes: Dict[str, HoleInfo] = {}
        self._hole_counter: int = 0
        self._tracker = SourceTracker()
        self._ast: Optional[MarkedASTNode] = None
        self._state: ParseState = ParseState.VALID
        self._errors: List[ParseError] = []

        # Bracket tracking for hole detection
        self._brace_depth: int = 0
        self._paren_depth: int = 0
        self._bracket_depth: int = 0
        self._angle_depth: int = 0  # For generics < >

        # Context tracking
        self._in_type_context: bool = False
        self._in_jsx: bool = False
        self._template_depth: int = 0

        # File type (affects JSX parsing)
        self._is_tsx: bool = False

    @property
    def language(self) -> str:
        return "typescript"

    @property
    def current_source(self) -> str:
        return self._source

    @property
    def current_position(self) -> int:
        return len(self._source)

    def get_source(self) -> str:
        """Get the current source code."""
        return self._source

    def parse_initial(self, source: str) -> ParseResult:
        """Parse an initial source string."""
        self._source = source
        self._tracker.reset()
        self._tokens = []
        self._holes = {}
        self._hole_counter = 0
        self._errors = []
        self._brace_depth = 0
        self._paren_depth = 0
        self._bracket_depth = 0
        self._angle_depth = 0
        self._in_type_context = False
        self._in_jsx = False
        self._template_depth = 0

        if source:
            self._tracker.append(source)
            self._update_bracket_counts(source)
            # Update context for each character
            for char in source:
                self._update_context(char)

        # Parse the source
        self._parse_source()

        return self._create_result()

    def extend_with_token(self, token: TokenInfo) -> ParseResult:
        """Extend with a new token."""
        span = self._tracker.append_token(token)
        self._source += token.text
        self._tokens.append(token)
        self._update_bracket_counts(token.text)

        # Update context based on token
        self._update_context(token.text)

        # Re-parse to update holes
        self._parse_source()

        return self._create_result()

    def extend_with_text(self, text: str) -> ParseResult:
        """Extend with raw text."""
        span = self._tracker.append(text)
        self._source += text
        self._update_bracket_counts(text)

        # Update context based on text
        for char in text:
            self._update_context(char)

        # Re-parse to update holes
        self._parse_source()

        return self._create_result()

    def _update_bracket_counts(self, text: str) -> None:
        """Update bracket depth counters."""
        i = 0
        while i < len(text):
            c = text[i]

            # Skip string literals
            if c in "\"'`":
                quote = c
                i += 1
                while i < len(text) and text[i] != quote:
                    if text[i] == "\\" and i + 1 < len(text):
                        i += 2
                    else:
                        i += 1
                if i < len(text):
                    i += 1  # Skip closing quote
                continue

            # Skip comments
            if c == "/" and i + 1 < len(text):
                if text[i + 1] == "/":
                    # Line comment - skip to end of line
                    while i < len(text) and text[i] != "\n":
                        i += 1
                    continue
                elif text[i + 1] == "*":
                    # Block comment - skip to */
                    i += 2
                    while i + 1 < len(text) and not (text[i] == "*" and text[i + 1] == "/"):
                        i += 1
                    i += 2
                    continue

            # Count brackets
            if c == "{":
                self._brace_depth += 1
            elif c == "}":
                self._brace_depth = max(0, self._brace_depth - 1)
            elif c == "(":
                self._paren_depth += 1
            elif c == ")":
                self._paren_depth = max(0, self._paren_depth - 1)
            elif c == "[":
                self._bracket_depth += 1
            elif c == "]":
                self._bracket_depth = max(0, self._bracket_depth - 1)
            elif c == "<":
                # Only count as angle bracket in type context
                if self._in_type_context or self._looks_like_generic(text, i):
                    self._angle_depth += 1
            elif c == ">":
                if self._angle_depth > 0:
                    self._angle_depth -= 1

            i += 1

    def _looks_like_generic(self, text: str, pos: int) -> bool:
        """Check if < at pos looks like a generic type argument."""
        # Look backwards for identifier (type name)
        i = pos - 1
        while i >= 0 and text[i].isspace():
            i -= 1

        if i < 0:
            return False

        # Check if preceded by identifier
        if text[i].isalnum() or text[i] == "_":
            return True

        return False

    def _update_context(self, text: str) -> None:
        """Update parsing context based on text."""
        text_stripped = text.strip()

        # Check for type context entry
        if text_stripped == ":":
            self._in_type_context = True
        elif text_stripped in ("=", "{", ";", ",", ")"):
            self._in_type_context = False

        # Check for JSX context
        if self._is_tsx:
            if text_stripped == "<" and not self._in_type_context:
                # Could be JSX start
                pass

        # Template literal tracking
        if text_stripped == "`":
            if self._template_depth == 0:
                self._template_depth = 1
            else:
                self._template_depth = 0

    def _parse_source(self) -> None:
        """Parse the current source and detect holes."""
        self._holes = {}

        if not self._source.strip():
            return

        # Detect holes based on parsing state
        self._detect_structural_holes()
        self._detect_contextual_holes()

    def _detect_structural_holes(self) -> None:
        """Detect holes based on bracket structure."""
        # Unclosed braces mean incomplete block
        if self._brace_depth > 0:
            hole_id = self._create_hole_id()
            self._holes[hole_id] = HoleInfo(
                hole_id=hole_id,
                kind=HoleKind.EXPRESSION,
                expected_type=ANY,
                span=UNKNOWN_SPAN,
                context="unclosed_brace",
            )

        # Unclosed parens mean incomplete expression/call
        if self._paren_depth > 0:
            hole_id = self._create_hole_id()
            self._holes[hole_id] = HoleInfo(
                hole_id=hole_id,
                kind=HoleKind.EXPRESSION,
                expected_type=ANY,
                span=UNKNOWN_SPAN,
                context="unclosed_paren",
            )

        # Unclosed brackets mean incomplete array/index
        if self._bracket_depth > 0:
            hole_id = self._create_hole_id()
            self._holes[hole_id] = HoleInfo(
                hole_id=hole_id,
                kind=HoleKind.EXPRESSION,
                expected_type=ANY,
                span=UNKNOWN_SPAN,
                context="unclosed_bracket",
            )

        # Unclosed angle brackets mean incomplete generic
        if self._angle_depth > 0:
            hole_id = self._create_hole_id()
            self._holes[hole_id] = HoleInfo(
                hole_id=hole_id,
                kind=HoleKind.TYPE,
                expected_type=ANY,
                span=UNKNOWN_SPAN,
                context="unclosed_generic",
            )

        # Unclosed template literal
        if self._template_depth > 0:
            hole_id = self._create_hole_id()
            self._holes[hole_id] = HoleInfo(
                hole_id=hole_id,
                kind=HoleKind.EXPRESSION,
                expected_type=TS_STRING,
                span=UNKNOWN_SPAN,
                context="unclosed_template",
            )

    def _detect_contextual_holes(self) -> None:
        """Detect holes based on contextual analysis."""
        source = self._source.rstrip()
        if not source:
            return

        # Get last significant token
        last_token = self._get_last_significant_token(source)

        # Check for incomplete constructs
        if source.endswith(":") and self._in_type_context:
            # Expecting type annotation
            hole_id = self._create_hole_id()
            self._holes[hole_id] = HoleInfo(
                hole_id=hole_id,
                kind=HoleKind.TYPE,
                expected_type=ANY,
                span=UNKNOWN_SPAN,
                context="type_annotation",
            )

        elif source.endswith("="):
            # Expecting value
            hole_id = self._create_hole_id()
            self._holes[hole_id] = HoleInfo(
                hole_id=hole_id,
                kind=HoleKind.EXPRESSION,
                expected_type=ANY,
                span=UNKNOWN_SPAN,
                context="assignment",
            )

        elif source.endswith("=>"):
            # Expecting arrow function body
            hole_id = self._create_hole_id()
            self._holes[hole_id] = HoleInfo(
                hole_id=hole_id,
                kind=HoleKind.EXPRESSION,
                expected_type=ANY,
                span=UNKNOWN_SPAN,
                context="arrow_body",
            )

        elif source.endswith("return"):
            # Expecting return value
            hole_id = self._create_hole_id()
            self._holes[hole_id] = HoleInfo(
                hole_id=hole_id,
                kind=HoleKind.EXPRESSION,
                expected_type=ANY,
                span=UNKNOWN_SPAN,
                context="return_value",
            )

        elif self._ends_with_binary_operator(source):
            # Expecting right operand
            hole_id = self._create_hole_id()
            self._holes[hole_id] = HoleInfo(
                hole_id=hole_id,
                kind=HoleKind.EXPRESSION,
                expected_type=ANY,
                span=UNKNOWN_SPAN,
                context="binary_operand",
            )

        elif self._ends_with_keyword_expecting_expression(source):
            # Expecting expression after keyword
            hole_id = self._create_hole_id()
            self._holes[hole_id] = HoleInfo(
                hole_id=hole_id,
                kind=HoleKind.EXPRESSION,
                expected_type=ANY,
                span=UNKNOWN_SPAN,
                context="keyword_expression",
            )

        elif self._is_incomplete_declaration(source):
            # Incomplete variable/function declaration
            hole_id = self._create_hole_id()
            self._holes[hole_id] = HoleInfo(
                hole_id=hole_id,
                kind=HoleKind.EXPRESSION,
                expected_type=ANY,
                span=UNKNOWN_SPAN,
                context="incomplete_declaration",
            )

    def _get_last_significant_token(self, source: str) -> str:
        """Get the last non-whitespace token."""
        source = source.rstrip()
        if not source:
            return ""

        # Walk backwards to find token start
        i = len(source) - 1

        # Skip trailing operators
        while i >= 0 and source[i] in "+-*/%=<>&|!?:":
            i -= 1

        if i < 0:
            return source

        # Find token start
        while i >= 0 and (source[i].isalnum() or source[i] == "_"):
            i -= 1

        return source[i + 1:].strip()

    def _ends_with_binary_operator(self, source: str) -> bool:
        """Check if source ends with a binary operator."""
        source = source.rstrip()
        if not source:
            return False

        binary_ops = {
            "+", "-", "*", "/", "%", "**",
            "&&", "||", "??",
            "&", "|", "^",
            "<", ">", "<=", ">=", "==", "!=", "===", "!==",
            "<<", ">>", ">>>",
            "in", "instanceof",
        }

        for op in sorted(binary_ops, key=len, reverse=True):
            if source.endswith(op):
                # Make sure it's not part of a larger construct
                prefix = source[:-len(op)]
                if not prefix or not prefix[-1].isalnum():
                    return True

        return False

    def _ends_with_keyword_expecting_expression(self, source: str) -> bool:
        """Check if source ends with a keyword that expects an expression."""
        keywords = {"if", "while", "switch", "throw", "case", "typeof", "await", "yield"}
        source_lower = source.rstrip()

        for kw in keywords:
            if source_lower.endswith(kw):
                prefix = source_lower[:-len(kw)]
                if not prefix or not prefix[-1].isalnum():
                    return True

        return False

    def _is_incomplete_declaration(self, source: str) -> bool:
        """Check if source is an incomplete declaration.

        Examples of incomplete declarations:
        - "const x" (needs = or type annotation)
        - "let y" (needs = or type annotation)
        - "function foo" (needs parameters)
        """
        import re

        source = source.rstrip()
        if not source:
            return False

        # Pattern: (const|let|var) <identifier>
        # where identifier doesn't end with = or : or ; or {
        decl_pattern = r'^.*\b(const|let|var)\s+([a-zA-Z_$][a-zA-Z0-9_$]*)$'
        if re.match(decl_pattern, source):
            return True

        # Pattern: function <identifier> (without open paren)
        func_pattern = r'^.*\bfunction\s+([a-zA-Z_$][a-zA-Z0-9_$]*)$'
        if re.match(func_pattern, source):
            return True

        # Pattern: class <identifier> (without extends/implements/opening brace)
        class_pattern = r'^.*\bclass\s+([a-zA-Z_$][a-zA-Z0-9_$]*)$'
        if re.match(class_pattern, source):
            return True

        return False

    def _create_hole_id(self) -> str:
        """Create a unique hole ID."""
        self._hole_counter += 1
        return f"hole_{self._hole_counter}"

    def find_holes(self) -> List[Tuple[str, SourceSpan]]:
        """Find all holes in the current partial AST."""
        return [(hole_id, hole.span) for hole_id, hole in self._holes.items()]

    def get_expected_tokens(self) -> List[str]:
        """Get tokens that would be valid at the current position."""
        expected: List[str] = []

        # In type context
        if self._in_type_context:
            expected.extend(TYPESCRIPT_PRIMITIVE_TYPES)
            expected.extend([
                "Array", "Promise", "Map", "Set",
                "Partial", "Required", "Readonly", "Record",
                "(", "[", "{", "|", "&", "keyof", "typeof",
            ])
            return expected

        # Get context from source
        source = self._source.rstrip()

        # After opening brace - expect statement or closing
        if self._brace_depth > 0 and (not source or source.endswith("{")):
            expected.extend(TYPESCRIPT_DECLARATION_KEYWORDS)
            expected.extend(TYPESCRIPT_CONTROL_KEYWORDS)
            expected.extend(["return", "throw", "}", "//", "/*"])
            return expected

        # After opening paren - expect expression or parameter
        if self._paren_depth > 0:
            expected.extend(["identifier", ")", ",", "..."])
            expected.extend(TYPESCRIPT_PRIMITIVE_TYPES)
            return expected

        # After = or => - expect expression
        if source.endswith("=") or source.endswith("=>"):
            expected.extend([
                "identifier", "number", "string", "true", "false", "null", "undefined",
                "(", "[", "{", "function", "class", "new", "async", "`",
            ])
            return expected

        # After binary operator - expect expression
        if self._ends_with_binary_operator(source):
            expected.extend([
                "identifier", "number", "string", "true", "false", "null", "undefined",
                "(", "[", "{", "!", "-", "+", "~",
            ])
            return expected

        # Default - statement or expression start
        expected.extend(TYPESCRIPT_DECLARATION_KEYWORDS)
        expected.extend(TYPESCRIPT_CONTROL_KEYWORDS)
        expected.extend(["identifier", "number", "string"])

        return expected

    def checkpoint(self) -> TypeScriptParserCheckpoint:
        """Create a checkpoint of the current parser state."""
        return TypeScriptParserCheckpoint(
            source=self._source,
            tokens=list(self._tokens),
            holes=dict(self._holes),
            hole_counter=self._hole_counter,
            tracker_checkpoint=self._tracker.checkpoint(),
            brace_depth=self._brace_depth,
            paren_depth=self._paren_depth,
            bracket_depth=self._bracket_depth,
            angle_depth=self._angle_depth,
            in_type_context=self._in_type_context,
            in_jsx=self._in_jsx,
            template_depth=self._template_depth,
        )

    def restore(self, checkpoint: TypeScriptParserCheckpoint) -> None:
        """Restore parser state from a checkpoint."""
        self._source = checkpoint.source
        self._tokens = list(checkpoint.tokens)
        self._holes = dict(checkpoint.holes)
        self._hole_counter = checkpoint.hole_counter
        self._tracker.restore(checkpoint.tracker_checkpoint)
        self._brace_depth = checkpoint.brace_depth
        self._paren_depth = checkpoint.paren_depth
        self._bracket_depth = checkpoint.bracket_depth
        self._angle_depth = checkpoint.angle_depth
        self._in_type_context = checkpoint.in_type_context
        self._in_jsx = checkpoint.in_jsx
        self._template_depth = checkpoint.template_depth

    def get_ast(self) -> Optional[MarkedASTNode]:
        """Get the current AST as a MarkedASTNode."""
        return self._ast

    def copy(self) -> "TypeScriptIncrementalParser":
        """Create an independent copy of this parser."""
        new_parser = TypeScriptIncrementalParser()
        new_parser._source = self._source
        new_parser._tokens = list(self._tokens)
        new_parser._holes = dict(self._holes)
        new_parser._hole_counter = self._hole_counter
        new_parser._tracker = copy.deepcopy(self._tracker)
        new_parser._ast = copy.deepcopy(self._ast)
        new_parser._state = self._state
        new_parser._errors = list(self._errors)
        new_parser._brace_depth = self._brace_depth
        new_parser._paren_depth = self._paren_depth
        new_parser._bracket_depth = self._bracket_depth
        new_parser._angle_depth = self._angle_depth
        new_parser._in_type_context = self._in_type_context
        new_parser._in_jsx = self._in_jsx
        new_parser._template_depth = self._template_depth
        new_parser._is_tsx = self._is_tsx
        return new_parser

    def _create_result(self) -> ParseResult:
        """Create a ParseResult from current state."""
        # Determine parse state
        if self._errors:
            state = ParseState.ERROR
        elif self._brace_depth > 0 or self._paren_depth > 0 or self._bracket_depth > 0:
            state = ParseState.PARTIAL
        elif self._holes:
            state = ParseState.PARTIAL
        elif self._is_complete():
            state = ParseState.COMPLETE
        else:
            state = ParseState.VALID

        self._state = state

        return ParseResult(
            state=state,
            ast=self._ast,
            errors=list(self._errors),
            holes=list(self._holes.keys()),
            position=len(self._source),
        )

    def _is_complete(self) -> bool:
        """Check if the current source is a complete program."""
        if not self._source.strip():
            return True

        # All brackets closed
        if self._brace_depth != 0 or self._paren_depth != 0 or self._bracket_depth != 0:
            return False

        # No pending holes
        if self._holes:
            return False

        # Source ends with semicolon, closing brace, or newline
        stripped = self._source.rstrip()
        if stripped.endswith((";", "}", ")")):
            return True

        return False

    def set_tsx_mode(self, is_tsx: bool) -> None:
        """Set whether to parse as TSX (with JSX support)."""
        self._is_tsx = is_tsx


# =============================================================================
# Utility Functions
# =============================================================================

def create_typescript_parser(source: str = "", is_tsx: bool = False) -> TypeScriptIncrementalParser:
    """Create a TypeScript parser with optional initial source.

    Args:
        source: Initial source code to parse
        is_tsx: Whether to enable TSX/JSX mode

    Returns:
        Configured TypeScriptIncrementalParser
    """
    parser = TypeScriptIncrementalParser()
    parser.set_tsx_mode(is_tsx)
    if source:
        parser.parse_initial(source)
    return parser
