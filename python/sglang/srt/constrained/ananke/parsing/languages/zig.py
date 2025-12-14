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
"""Zig incremental parser.

This module provides a Zig-specific incremental parser that converts
token streams into MarkedASTNodes suitable for type checking.

Features:
- Incremental parsing (ideally via tree-sitter-zig)
- Hole detection for incomplete constructs
- Source span tracking for error messages
- Comptime block awareness
- Error handling construct detection

References:
    - tree-sitter-zig: https://github.com/tree-sitter-grammars/tree-sitter-zig
    - Zig Grammar: https://ziglang.org/documentation/master/#Grammar
"""

from __future__ import annotations

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
from domains.types.languages.zig import (
    ZIG_BOOL,
    ZIG_COMPTIME_INT,
    ZIG_COMPTIME_FLOAT,
    ZIG_U8,
    ZIG_VOID,
    ZigSliceType,
    ZigArrayType,
    ZigPointerType,
)
from core.token_classifier_zig import (
    ZIG_ALL_KEYWORDS,
    ZIG_BUILTINS,
    ZIG_CONTROL_KEYWORDS,
    ZIG_DEFINITION_KEYWORDS,
    ZIG_ERROR_KEYWORDS,
    classify_zig_token,
)


# =============================================================================
# Token Categories for Hole Detection
# =============================================================================

EXPRESSION_STARTERS: Set[str] = {
    "identifier", "number", "string", "true", "false", "null", "undefined",
    "(", "[", "{", ".", "@",
    "if", "switch", "for", "while",
    "try", "catch",
    "comptime", "async",
    "-", "!", "~", "&", "*", "%",
}

EXPRESSION_ENDERS: Set[str] = {
    "identifier", "number", "string", "true", "false", "null", "undefined",
    ")", "]", "}",
}

INCOMPLETE_CONTEXTS: Set[str] = {
    "binary_operator", "unary_operator", "call", "subscript",
    "assignment", "return_statement", "const_declaration", "var_declaration",
    "if_expression", "while_expression", "for_expression", "switch_expression",
    "function_declaration", "struct_declaration", "enum_declaration",
    "comptime_block", "error_union",
}


# =============================================================================
# Parser Checkpoint
# =============================================================================

@dataclass
class ZigParserCheckpoint:
    """Checkpoint for Zig parser state."""

    source: str
    tokens: List[TokenInfo]
    holes: Dict[str, HoleInfo]
    hole_counter: int
    tracker_checkpoint: Any
    brace_depth: int
    paren_depth: int
    bracket_depth: int


# =============================================================================
# Zig Incremental Parser
# =============================================================================

class ZigIncrementalParser(IncrementalParser):
    """Zig-specific incremental parser.

    This parser provides token-by-token parsing for Zig code generation.
    It maintains a partial AST and detects holes where more code is expected.

    For production use, this should integrate with tree-sitter-zig for
    proper incremental parsing. The current implementation provides a
    lightweight fallback.

    Example:
        >>> parser = ZigIncrementalParser()
        >>> result = parser.parse_initial("fn foo(x: i32)")
        >>> result = parser.extend_with_text(" ")
        >>> holes = parser.find_holes()
        >>> # Expect a hole for the return type
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

    @property
    def language(self) -> str:
        return "zig"

    @property
    def current_source(self) -> str:
        return self._source

    @property
    def current_position(self) -> int:
        return len(self._source)

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

        if source:
            self._tracker.append(source)
            self._update_bracket_counts(source)

        # Parse the source
        self._parse_source()

        return self._create_result()

    def extend_with_token(self, token: TokenInfo) -> ParseResult:
        """Extend with a new token."""
        span = self._tracker.append_token(token)
        self._source += token.text
        self._tokens.append(token)
        self._update_bracket_counts(token.text)

        # Re-parse with new content
        self._parse_source()

        return self._create_result()

    def extend_with_text(self, text: str) -> ParseResult:
        """Extend with raw text."""
        self._tracker.append(text)
        self._source += text
        self._update_bracket_counts(text)

        # Re-parse with new content
        self._parse_source()

        return self._create_result()

    def find_holes(self) -> List[Tuple[str, SourceSpan]]:
        """Find all holes in the current parse."""
        return [(h.hole_id, h.span) for h in self._holes.values()]

    def get_expected_tokens(self) -> List[str]:
        """Get expected tokens at current position."""
        source = self._source.rstrip()

        if not source:
            return list(EXPRESSION_STARTERS) + ["fn", "const", "var", "pub", "test"]

        last_char = source[-1]

        # After opening brackets
        if last_char == "(":
            return ["identifier", "comptime", "@", ")", "*", "type"]
        if last_char == "[":
            return ["number", "*", ":", "_", "]"]
        if last_char == "{":
            return [".", "identifier", "}", "if", "while", "for", "return"]

        # After operators
        if last_char in "+-*/%&|^":
            return list(EXPRESSION_STARTERS)
        if last_char == "=":
            if len(source) >= 2 and source[-2] in "!<>=":
                return list(EXPRESSION_STARTERS)
            return list(EXPRESSION_STARTERS)

        # After comma
        if last_char == ",":
            return ["identifier", "comptime", "@", ".", "*"]

        # After colon (type annotation or struct field)
        if last_char == ":":
            return ["identifier", "*", "[", "?", "!", "@", "type", "anytype"]

        # After function declaration
        if source.endswith("fn "):
            return ["identifier"]
        if source.endswith(") "):
            return ["identifier", "!", "void", "anytype", "callconv", "*", "?", "["]

        # After control flow keywords
        if source.endswith("if ") or source.endswith("if("):
            return ["(", "identifier", "@"]
        if source.endswith("while ") or source.endswith("while("):
            return ["(", "identifier", "@"]
        if source.endswith("for ") or source.endswith("for("):
            return ["(", "identifier", "@"]
        if source.endswith("switch ") or source.endswith("switch("):
            return ["(", "identifier", "@"]

        # After try/catch
        if source.endswith("try "):
            return ["identifier", "@", "("]
        if source.endswith("catch "):
            return ["|", "{", "identifier"]

        # After comptime
        if source.endswith("comptime "):
            return ["{", "identifier", "var", "const", "@"]

        # After pub
        if source.endswith("pub "):
            return ["fn", "const", "var", "struct", "enum", "union"]

        # After const/var
        if source.endswith("const ") or source.endswith("var "):
            return ["identifier"]

        # After return
        if source.endswith("return"):
            return list(EXPRESSION_STARTERS) + [";"]

        # After struct/enum/union
        if source.endswith("struct ") or source.endswith("struct{"):
            return ["{", "identifier", "pub", "const", "}"]
        if source.endswith("enum ") or source.endswith("enum{") or source.endswith("enum("):
            return ["{", "(", "identifier", "}"]
        if source.endswith("union ") or source.endswith("union{") or source.endswith("union("):
            return ["{", "(", "identifier", "enum", "}"]

        # After error keyword
        if source.endswith("error{"):
            return ["identifier", "}"]

        # Default
        return list(EXPRESSION_STARTERS) + list(EXPRESSION_ENDERS)

    def checkpoint(self) -> ZigParserCheckpoint:
        """Create a checkpoint."""
        return ZigParserCheckpoint(
            source=self._source,
            tokens=self._tokens.copy(),
            holes=self._holes.copy(),
            hole_counter=self._hole_counter,
            tracker_checkpoint=self._tracker.checkpoint(),
            brace_depth=self._brace_depth,
            paren_depth=self._paren_depth,
            bracket_depth=self._bracket_depth,
        )

    def restore(self, checkpoint: Any) -> None:
        """Restore from a checkpoint."""
        if not isinstance(checkpoint, ZigParserCheckpoint):
            raise TypeError("Invalid checkpoint type")

        self._source = checkpoint.source
        self._tokens = checkpoint.tokens.copy()
        self._holes = checkpoint.holes.copy()
        self._hole_counter = checkpoint.hole_counter
        self._tracker.restore(checkpoint.tracker_checkpoint)
        self._brace_depth = checkpoint.brace_depth
        self._paren_depth = checkpoint.paren_depth
        self._bracket_depth = checkpoint.bracket_depth

        # Re-parse to restore AST
        self._parse_source()

    def get_ast(self) -> Optional[MarkedASTNode]:
        """Get the current AST."""
        return self._ast

    def copy(self) -> 'ZigIncrementalParser':
        """Create an independent copy."""
        new_parser = ZigIncrementalParser()
        checkpoint = self.checkpoint()
        new_parser.restore(checkpoint)
        return new_parser

    def _update_bracket_counts(self, text: str) -> None:
        """Update bracket depth counters."""
        for char in text:
            if char == "{":
                self._brace_depth += 1
            elif char == "}":
                self._brace_depth = max(0, self._brace_depth - 1)
            elif char == "(":
                self._paren_depth += 1
            elif char == ")":
                self._paren_depth = max(0, self._paren_depth - 1)
            elif char == "[":
                self._bracket_depth += 1
            elif char == "]":
                self._bracket_depth = max(0, self._bracket_depth - 1)

    def _parse_source(self) -> None:
        """Parse the current source into an AST."""
        self._errors = []
        self._holes = {}

        if not self._source.strip():
            # Empty source - create a hole
            self._ast = self._create_module_hole()
            self._state = ParseState.PARTIAL
            return

        try:
            # Build AST from source
            self._ast = self._build_ast_from_source()
            self._detect_holes()

            if self._holes:
                self._state = ParseState.PARTIAL
            else:
                self._state = ParseState.VALID

        except Exception as e:
            # Parse error - mark as partial with hole
            self._ast = self._create_error_recovery_ast(str(e))
            self._state = ParseState.PARTIAL
            self._errors.append(ParseError(
                message=str(e),
                span=UNKNOWN_SPAN,
            ))

    def _build_ast_from_source(self) -> MarkedASTNode:
        """Build a MarkedASTNode from Zig source.

        This is a simplified implementation. For production use,
        integrate with tree-sitter-zig.
        """
        source = self._source.strip()

        # Try to classify the top-level construct
        if source.startswith("fn ") or source.startswith("pub fn "):
            return self._parse_function_def(source)
        elif source.startswith("const ") or source.startswith("pub const "):
            return self._parse_const_decl(source)
        elif source.startswith("var ") or source.startswith("pub var "):
            return self._parse_var_decl(source)
        elif source.startswith("struct") or source.startswith("pub struct") or source.startswith("packed struct") or source.startswith("extern struct"):
            return self._parse_struct_def(source)
        elif source.startswith("enum") or source.startswith("pub enum"):
            return self._parse_enum_def(source)
        elif source.startswith("union") or source.startswith("pub union"):
            return self._parse_union_def(source)
        elif source.startswith("test "):
            return self._parse_test_decl(source)
        elif source.startswith("comptime ") or source.startswith("comptime{"):
            return self._parse_comptime_block(source)
        elif source.startswith("@import") or source.startswith("@embedFile"):
            return self._parse_import(source)
        else:
            return self._parse_expression(source)

    def _parse_function_def(self, source: str) -> MarkedASTNode:
        """Parse a function definition."""
        span = self._tracker.span_at(0, len(source))

        # Check for incomplete function
        has_body = "{" in source and "}" in source
        has_return_type = ")" in source and (
            source.split(")")[-1].strip().startswith(("void", "!", "anytype", "*", "[", "?", "type")) or
            any(c.isalpha() for c in source.split(")")[-1].strip()[:10])
        )

        if not has_body:
            hole_id = self._new_hole_id()
            self._holes[hole_id] = HoleInfo(
                hole_id=hole_id,
                kind=HoleKind.BODY,
                span=span,
                context="function body",
            )

        if ")" in source and not has_return_type and not has_body:
            hole_id = self._new_hole_id()
            self._holes[hole_id] = HoleInfo(
                hole_id=hole_id,
                kind=HoleKind.TYPE,
                span=span,
                context="return type",
            )

        return MarkedASTNode(
            kind=ASTNodeKind.FUNCTION_DEF,
            span=span,
            data={"source": source},
        )

    def _parse_const_decl(self, source: str) -> MarkedASTNode:
        """Parse a const declaration."""
        span = self._tracker.span_at(0, len(source))

        # Check for incomplete declaration
        if "=" in source:
            rhs = source.split("=", 1)[-1].strip()
            if not rhs or rhs.endswith(("=", "+", "-", "*", "/", "(", "[", "{", ",")):
                hole_id = self._new_hole_id()
                self._holes[hole_id] = HoleInfo(
                    hole_id=hole_id,
                    kind=HoleKind.EXPRESSION,
                    span=span,
                    context="const value",
                )
        elif ":" in source and "=" not in source:
            # Has type but no value
            hole_id = self._new_hole_id()
            self._holes[hole_id] = HoleInfo(
                hole_id=hole_id,
                kind=HoleKind.EXPRESSION,
                span=span,
                context="const value",
            )

        return MarkedASTNode(
            kind=ASTNodeKind.ASSIGNMENT,
            span=span,
            data={"source": source, "is_const": True},
        )

    def _parse_var_decl(self, source: str) -> MarkedASTNode:
        """Parse a var declaration."""
        span = self._tracker.span_at(0, len(source))

        # Check for incomplete declaration
        if "=" in source:
            rhs = source.split("=", 1)[-1].strip()
            if not rhs or rhs.endswith(("=", "+", "-", "*", "/", "(", "[", "{", ",")):
                hole_id = self._new_hole_id()
                self._holes[hole_id] = HoleInfo(
                    hole_id=hole_id,
                    kind=HoleKind.EXPRESSION,
                    span=span,
                    context="var value",
                )

        return MarkedASTNode(
            kind=ASTNodeKind.ASSIGNMENT,
            span=span,
            data={"source": source, "is_const": False},
        )

    def _parse_struct_def(self, source: str) -> MarkedASTNode:
        """Parse a struct definition."""
        span = self._tracker.span_at(0, len(source))

        # Check for incomplete struct
        if self._brace_depth > 0 or (source.count("{") > source.count("}")):
            hole_id = self._new_hole_id()
            self._holes[hole_id] = HoleInfo(
                hole_id=hole_id,
                kind=HoleKind.BODY,
                span=span,
                context="struct body",
            )

        return MarkedASTNode(
            kind=ASTNodeKind.CLASS_DEF,  # Using CLASS_DEF for struct
            span=span,
            data={"source": source, "kind": "struct"},
        )

    def _parse_enum_def(self, source: str) -> MarkedASTNode:
        """Parse an enum definition."""
        span = self._tracker.span_at(0, len(source))

        if self._brace_depth > 0 or (source.count("{") > source.count("}")):
            hole_id = self._new_hole_id()
            self._holes[hole_id] = HoleInfo(
                hole_id=hole_id,
                kind=HoleKind.BODY,
                span=span,
                context="enum body",
            )

        return MarkedASTNode(
            kind=ASTNodeKind.CLASS_DEF,
            span=span,
            data={"source": source, "kind": "enum"},
        )

    def _parse_union_def(self, source: str) -> MarkedASTNode:
        """Parse a union definition."""
        span = self._tracker.span_at(0, len(source))

        if self._brace_depth > 0 or (source.count("{") > source.count("}")):
            hole_id = self._new_hole_id()
            self._holes[hole_id] = HoleInfo(
                hole_id=hole_id,
                kind=HoleKind.BODY,
                span=span,
                context="union body",
            )

        return MarkedASTNode(
            kind=ASTNodeKind.CLASS_DEF,
            span=span,
            data={"source": source, "kind": "union"},
        )

    def _parse_test_decl(self, source: str) -> MarkedASTNode:
        """Parse a test declaration."""
        span = self._tracker.span_at(0, len(source))

        if self._brace_depth > 0 or (source.count("{") > source.count("}")):
            hole_id = self._new_hole_id()
            self._holes[hole_id] = HoleInfo(
                hole_id=hole_id,
                kind=HoleKind.BODY,
                span=span,
                context="test body",
            )

        return MarkedASTNode(
            kind=ASTNodeKind.FUNCTION_DEF,
            span=span,
            data={"source": source, "is_test": True},
        )

    def _parse_comptime_block(self, source: str) -> MarkedASTNode:
        """Parse a comptime block."""
        span = self._tracker.span_at(0, len(source))

        if self._brace_depth > 0 or (source.count("{") > source.count("}")):
            hole_id = self._new_hole_id()
            self._holes[hole_id] = HoleInfo(
                hole_id=hole_id,
                kind=HoleKind.BODY,
                span=span,
                context="comptime block",
            )

        return MarkedASTNode(
            kind=ASTNodeKind.EXPRESSION,
            span=span,
            data={"source": source, "is_comptime": True},
        )

    def _parse_import(self, source: str) -> MarkedASTNode:
        """Parse an @import or @embedFile."""
        span = self._tracker.span_at(0, len(source))

        return MarkedASTNode(
            kind=ASTNodeKind.IMPORT,
            span=span,
            data={"source": source},
        )

    def _parse_expression(self, source: str) -> MarkedASTNode:
        """Parse an expression."""
        span = self._tracker.span_at(0, len(source))

        # Check for specific expression types
        if source.startswith("if ") or source.startswith("if("):
            return self._parse_if_expr(source, span)
        elif source.startswith("switch ") or source.startswith("switch("):
            return self._parse_switch_expr(source, span)
        elif source.startswith("for ") or source.startswith("for("):
            return self._parse_for_expr(source, span)
        elif source.startswith("while ") or source.startswith("while("):
            return self._parse_while_expr(source, span)

        # Try to parse as literal
        category, keyword, value = classify_zig_token(source)

        if category.name == "INT_LITERAL":
            return create_literal_node(value=value, ty=ZIG_COMPTIME_INT, span=span)
        elif category.name == "FLOAT_LITERAL":
            return create_literal_node(value=value, ty=ZIG_COMPTIME_FLOAT, span=span)
        elif category.name == "BOOL_LITERAL":
            return create_literal_node(value=value, ty=ZIG_BOOL, span=span)
        elif category.name == "STRING_LITERAL":
            return create_literal_node(
                value=value,
                ty=ZigPointerType(
                    pointee=ZigArrayType(element=ZIG_U8, length=len(value) if value else 0, sentinel=0),
                    is_const=True,
                ),
                span=span,
            )
        elif category.name == "IDENTIFIER":
            return create_variable_node(name=source, ty=ANY, span=span)

        # Generic expression
        return MarkedASTNode(
            kind=ASTNodeKind.EXPRESSION,
            span=span,
            data={"source": source},
        )

    def _parse_if_expr(self, source: str, span: SourceSpan) -> MarkedASTNode:
        """Parse an if expression."""
        # Check for incomplete if
        if self._brace_depth > 0 or self._paren_depth > 0:
            hole_id = self._new_hole_id()
            self._holes[hole_id] = HoleInfo(
                hole_id=hole_id,
                kind=HoleKind.EXPRESSION,
                span=span,
                context="if expression",
            )

        return MarkedASTNode(
            kind=ASTNodeKind.EXPRESSION,
            span=span,
            data={"source": source, "expr_kind": "if"},
        )

    def _parse_switch_expr(self, source: str, span: SourceSpan) -> MarkedASTNode:
        """Parse a switch expression."""
        if self._brace_depth > 0 or self._paren_depth > 0:
            hole_id = self._new_hole_id()
            self._holes[hole_id] = HoleInfo(
                hole_id=hole_id,
                kind=HoleKind.EXPRESSION,
                span=span,
                context="switch expression",
            )

        return MarkedASTNode(
            kind=ASTNodeKind.EXPRESSION,
            span=span,
            data={"source": source, "expr_kind": "switch"},
        )

    def _parse_for_expr(self, source: str, span: SourceSpan) -> MarkedASTNode:
        """Parse a for expression."""
        if self._brace_depth > 0 or self._paren_depth > 0:
            hole_id = self._new_hole_id()
            self._holes[hole_id] = HoleInfo(
                hole_id=hole_id,
                kind=HoleKind.EXPRESSION,
                span=span,
                context="for expression",
            )

        return MarkedASTNode(
            kind=ASTNodeKind.EXPRESSION,
            span=span,
            data={"source": source, "expr_kind": "for"},
        )

    def _parse_while_expr(self, source: str, span: SourceSpan) -> MarkedASTNode:
        """Parse a while expression."""
        if self._brace_depth > 0 or self._paren_depth > 0:
            hole_id = self._new_hole_id()
            self._holes[hole_id] = HoleInfo(
                hole_id=hole_id,
                kind=HoleKind.EXPRESSION,
                span=span,
                context="while expression",
            )

        return MarkedASTNode(
            kind=ASTNodeKind.EXPRESSION,
            span=span,
            data={"source": source, "expr_kind": "while"},
        )

    def _detect_holes(self) -> None:
        """Detect incomplete constructs that represent holes."""
        source = self._source.rstrip()

        if not source:
            return

        last_char = source[-1]

        # Check for trailing operators
        if last_char in "+-*/%&|^<>=!":
            hole_id = self._new_hole_id()
            self._holes[hole_id] = HoleInfo(
                hole_id=hole_id,
                kind=HoleKind.EXPRESSION,
                span=self._tracker.span_at(len(source), len(source)),
                context="operand",
            )

        # Check for unclosed brackets
        if self._brace_depth > 0:
            hole_id = self._new_hole_id()
            self._holes[hole_id] = HoleInfo(
                hole_id=hole_id,
                kind=HoleKind.BODY,
                span=self._tracker.span_at(len(source), len(source)),
                context="closing brace",
            )

        if self._paren_depth > 0:
            hole_id = self._new_hole_id()
            self._holes[hole_id] = HoleInfo(
                hole_id=hole_id,
                kind=HoleKind.EXPRESSION,
                span=self._tracker.span_at(len(source), len(source)),
                context="closing parenthesis",
            )

        if self._bracket_depth > 0:
            hole_id = self._new_hole_id()
            self._holes[hole_id] = HoleInfo(
                hole_id=hole_id,
                kind=HoleKind.EXPRESSION,
                span=self._tracker.span_at(len(source), len(source)),
                context="closing bracket",
            )

        # Check for trailing comma
        if last_char == ",":
            hole_id = self._new_hole_id()
            self._holes[hole_id] = HoleInfo(
                hole_id=hole_id,
                kind=HoleKind.EXPRESSION,
                span=self._tracker.span_at(len(source), len(source)),
                context="sequence element",
            )

        # Check for trailing colon (type annotation expected)
        if last_char == ":":
            hole_id = self._new_hole_id()
            self._holes[hole_id] = HoleInfo(
                hole_id=hole_id,
                kind=HoleKind.TYPE,
                span=self._tracker.span_at(len(source), len(source)),
                context="type annotation",
            )

        # Check for try without error handling
        if "try " in source and "catch" not in source and "orelse" not in source:
            # Might need error handling
            pass  # This is actually valid Zig - error propagates

        # Check for incomplete error union: E!
        if source.endswith("!"):
            hole_id = self._new_hole_id()
            self._holes[hole_id] = HoleInfo(
                hole_id=hole_id,
                kind=HoleKind.TYPE,
                span=self._tracker.span_at(len(source), len(source)),
                context="error union payload type",
            )

        # Check for incomplete optional: ?
        if source.endswith("?") and not source.endswith(".?"):
            hole_id = self._new_hole_id()
            self._holes[hole_id] = HoleInfo(
                hole_id=hole_id,
                kind=HoleKind.TYPE,
                span=self._tracker.span_at(len(source), len(source)),
                context="optional inner type",
            )

    def _create_module_hole(self) -> MarkedASTNode:
        """Create a module-level hole."""
        hole_id = self._new_hole_id()
        span = UNKNOWN_SPAN

        self._holes[hole_id] = HoleInfo(
            hole_id=hole_id,
            kind=HoleKind.STATEMENT,
            span=span,
            context="module",
        )

        return create_hole_node(
            hole_id=hole_id,
            span=span,
            expected_type=ANY,
        )

    def _create_error_recovery_ast(self, error: str) -> MarkedASTNode:
        """Create an AST node for error recovery."""
        hole_id = self._new_hole_id()
        span = self._tracker.span_at(0, len(self._source))

        self._holes[hole_id] = HoleInfo(
            hole_id=hole_id,
            kind=HoleKind.EXPRESSION,
            span=span,
            context=f"error recovery: {error}",
        )

        return MarkedASTNode(
            kind=ASTNodeKind.ERROR,
            span=span,
            children=[
                create_hole_node(
                    hole_id=hole_id,
                    span=span,
                    expected_type=ANY,
                )
            ],
            data={"error": error, "source": self._source},
        )

    def _new_hole_id(self) -> str:
        """Generate a new hole ID."""
        hole_id = f"hole_{self._hole_counter}"
        self._hole_counter += 1
        return hole_id

    def _create_result(self) -> ParseResult:
        """Create a ParseResult from current state."""
        return ParseResult(
            state=self._state,
            ast=self._ast,
            errors=self._errors.copy(),
            holes=list(self._holes.keys()),
            position=len(self._source),
        )


# Factory function
def create_zig_parser() -> ZigIncrementalParser:
    """Create a new Zig incremental parser.

    Returns:
        A fresh ZigIncrementalParser instance
    """
    return ZigIncrementalParser()
