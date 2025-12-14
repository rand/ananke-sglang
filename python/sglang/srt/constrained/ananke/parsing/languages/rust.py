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
"""Rust incremental parser.

This module provides a Rust-specific incremental parser that converts
token streams into MarkedASTNodes suitable for type checking.

Features:
- Incremental parsing (ideally via tree-sitter-rust)
- Hole detection for incomplete constructs
- Source span tracking for error messages
- Lifetime annotation awareness
- Pattern matching support
- Attribute handling

References:
    - tree-sitter-rust: https://github.com/tree-sitter/tree-sitter-rust
    - Rust Reference: https://doc.rust-lang.org/reference/
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
from domains.types.languages.rust import (
    RUST_BOOL,
    RUST_I32,
    RUST_F64,
    RUST_CHAR,
    RUST_UNIT,
    RustReferenceType,
    RustSliceType,
    RustStringType,
)
from core.token_classifier_rust import (
    RUST_ALL_KEYWORDS,
    RUST_STD_MACROS,
    RUST_CONTROL_KEYWORDS,
    RUST_DEFINITION_KEYWORDS,
    classify_rust_token,
)


# =============================================================================
# Token Categories for Hole Detection
# =============================================================================

EXPRESSION_STARTERS: Set[str] = {
    "identifier", "number", "string", "true", "false",
    "(", "[", "{", ".", "&", "*", "!",
    "if", "match", "loop", "while", "for",
    "return", "break", "continue",
    "async", "await", "move",
    "-", "!", "&", "*", "|",
    "unsafe", "box",
}

EXPRESSION_ENDERS: Set[str] = {
    "identifier", "number", "string", "true", "false",
    ")", "]", "}", "?",
}

INCOMPLETE_CONTEXTS: Set[str] = {
    "binary_operator", "unary_operator", "call", "index",
    "assignment", "let_statement", "return_expression",
    "if_expression", "match_expression", "loop_expression",
    "function_definition", "struct_definition", "enum_definition",
    "impl_block", "trait_definition", "lifetime",
}


# =============================================================================
# Parser Checkpoint
# =============================================================================

@dataclass
class RustParserCheckpoint:
    """Checkpoint for Rust parser state."""

    source: str
    tokens: List[TokenInfo]
    holes: Dict[str, HoleInfo]
    hole_counter: int
    tracker_checkpoint: Any
    brace_depth: int
    paren_depth: int
    bracket_depth: int
    angle_depth: int


# =============================================================================
# Rust Incremental Parser
# =============================================================================

class RustIncrementalParser(IncrementalParser):
    """Rust-specific incremental parser.

    This parser provides token-by-token parsing for Rust code generation.
    It maintains a partial AST and detects holes where more code is expected.

    For production use, this should integrate with tree-sitter-rust for
    proper incremental parsing. The current implementation provides a
    lightweight fallback.

    Example:
        >>> parser = RustIncrementalParser()
        >>> result = parser.parse_initial("fn foo(x: i32)")
        >>> result = parser.extend_with_text(" ")
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

    @property
    def language(self) -> str:
        return "rust"

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
        self._angle_depth = 0

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
            return list(EXPRESSION_STARTERS) + ["fn", "struct", "enum", "trait", "impl", "mod", "use", "pub"]

        last_char = source[-1]

        # After opening brackets
        if last_char == "(":
            return ["identifier", "self", "&", "*", "mut", ")", "_", "ref"]
        if last_char == "[":
            return ["identifier", "number", "]", "_", "&"]
        if last_char == "{":
            return ["identifier", "let", "if", "match", "loop", "while", "for", "return", "}", "unsafe"]
        if last_char == "<":
            return ["identifier", "'", "&", "*", "dyn", "impl", ">"]

        # After operators
        if last_char in "+-*/%&|^":
            return list(EXPRESSION_STARTERS)
        if last_char == "=":
            if len(source) >= 2 and source[-2] in "!<>=":
                return list(EXPRESSION_STARTERS)
            return list(EXPRESSION_STARTERS)

        # After comma
        if last_char == ",":
            return ["identifier", "&", "*", "'", "mut", "self"]

        # After colon
        if last_char == ":":
            if len(source) >= 2 and source[-2] == ":":
                # Path separator ::
                return ["identifier", "self", "super", "crate", "<"]
            # Type annotation
            return ["identifier", "&", "*", "[", "(", "!", "dyn", "impl", "'"]

        # After arrow
        if source.endswith("->"):
            return ["identifier", "&", "*", "[", "(", "!", "Self", "dyn", "impl"]

        # After function declaration
        if source.endswith("fn "):
            return ["identifier"]
        if source.endswith(") "):
            return ["->", "{", "where"]

        # After control flow keywords
        if source.endswith("if "):
            return ["let", "identifier", "(", "!"]
        if source.endswith("match "):
            return ["identifier", "(", "&", "*"]
        if source.endswith("while ") or source.endswith("loop "):
            return ["{", "identifier"]
        if source.endswith("for "):
            return ["identifier", "(", "_", "mut", "ref"]

        # After let
        if source.endswith("let "):
            return ["identifier", "mut", "_", "(", "ref"]
        if source.endswith("let mut "):
            return ["identifier", "_", "("]

        # After pub
        if source.endswith("pub "):
            return ["fn", "struct", "enum", "trait", "mod", "use", "const", "static", "type", "("]

        # After struct/enum/trait
        if source.endswith("struct "):
            return ["identifier"]
        if source.endswith("enum "):
            return ["identifier"]
        if source.endswith("trait "):
            return ["identifier"]
        if source.endswith("impl "):
            return ["identifier", "<", "!"]

        # After mod
        if source.endswith("mod "):
            return ["identifier"]

        # After use
        if source.endswith("use "):
            return ["identifier", "crate", "self", "super", "{"]

        # After return
        if source.endswith("return"):
            return list(EXPRESSION_STARTERS) + [";"]

        # After where
        if source.endswith("where "):
            return ["identifier", "Self"]

        # After lifetime '
        if last_char == "'":
            return ["identifier", "static", "_"]

        # After reference &
        if source.endswith("& ") or source.endswith("&"):
            return ["identifier", "mut", "'", "self"]

        # After unsafe
        if source.endswith("unsafe "):
            return ["{", "fn", "impl", "trait"]

        # After async
        if source.endswith("async "):
            return ["fn", "move", "{"]

        # Default
        return list(EXPRESSION_STARTERS) + list(EXPRESSION_ENDERS)

    def checkpoint(self) -> RustParserCheckpoint:
        """Create a checkpoint."""
        return RustParserCheckpoint(
            source=self._source,
            tokens=self._tokens.copy(),
            holes=self._holes.copy(),
            hole_counter=self._hole_counter,
            tracker_checkpoint=self._tracker.checkpoint(),
            brace_depth=self._brace_depth,
            paren_depth=self._paren_depth,
            bracket_depth=self._bracket_depth,
            angle_depth=self._angle_depth,
        )

    def restore(self, checkpoint: Any) -> None:
        """Restore from a checkpoint."""
        if not isinstance(checkpoint, RustParserCheckpoint):
            raise TypeError("Invalid checkpoint type")

        self._source = checkpoint.source
        self._tokens = checkpoint.tokens.copy()
        self._holes = checkpoint.holes.copy()
        self._hole_counter = checkpoint.hole_counter
        self._tracker.restore(checkpoint.tracker_checkpoint)
        self._brace_depth = checkpoint.brace_depth
        self._paren_depth = checkpoint.paren_depth
        self._bracket_depth = checkpoint.bracket_depth
        self._angle_depth = checkpoint.angle_depth

        # Re-parse to restore AST
        self._parse_source()

    def get_ast(self) -> Optional[MarkedASTNode]:
        """Get the current AST."""
        return self._ast

    def copy(self) -> 'RustIncrementalParser':
        """Create an independent copy."""
        new_parser = RustIncrementalParser()
        checkpoint = self.checkpoint()
        new_parser.restore(checkpoint)
        return new_parser

    def _update_bracket_counts(self, text: str) -> None:
        """Update bracket depth counters.

        Note: Angle brackets for generics are context-sensitive in Rust.
        This is a simplified heuristic.
        """
        i = 0
        while i < len(text):
            char = text[i]
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
            elif char == "<":
                # Heuristic: likely generics if preceded by identifier
                if i > 0 and (text[i-1].isalnum() or text[i-1] in "_>"):
                    self._angle_depth += 1
            elif char == ">":
                if self._angle_depth > 0:
                    self._angle_depth -= 1
            i += 1

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
        """Build a MarkedASTNode from Rust source.

        This is a simplified implementation. For production use,
        integrate with tree-sitter-rust.
        """
        source = self._source.strip()

        # Handle attributes
        if source.startswith("#[") or source.startswith("#!["):
            return self._parse_attribute(source)

        # Try to classify the top-level construct
        if source.startswith("fn ") or source.startswith("pub fn ") or source.startswith("async fn ") or source.startswith("pub async fn ") or source.startswith("const fn ") or source.startswith("unsafe fn "):
            return self._parse_function_def(source)
        elif source.startswith("let "):
            return self._parse_let_stmt(source)
        elif source.startswith("const ") or source.startswith("pub const "):
            return self._parse_const_decl(source)
        elif source.startswith("static ") or source.startswith("pub static "):
            return self._parse_static_decl(source)
        elif source.startswith("struct ") or source.startswith("pub struct "):
            return self._parse_struct_def(source)
        elif source.startswith("enum ") or source.startswith("pub enum "):
            return self._parse_enum_def(source)
        elif source.startswith("trait ") or source.startswith("pub trait "):
            return self._parse_trait_def(source)
        elif source.startswith("impl ") or source.startswith("impl<"):
            return self._parse_impl_block(source)
        elif source.startswith("mod ") or source.startswith("pub mod "):
            return self._parse_mod_decl(source)
        elif source.startswith("use ") or source.startswith("pub use "):
            return self._parse_use_decl(source)
        elif source.startswith("type ") or source.startswith("pub type "):
            return self._parse_type_alias(source)
        elif source.startswith("macro_rules!"):
            return self._parse_macro_def(source)
        else:
            return self._parse_expression(source)

    def _parse_function_def(self, source: str) -> MarkedASTNode:
        """Parse a function definition."""
        span = self._tracker.span_at(0, len(source))

        # Check for incomplete function
        has_body = "{" in source and source.count("{") <= source.count("}")
        has_return_type = "->" in source or (
            ")" in source and "{" in source
        )

        if not has_body:
            hole_id = self._new_hole_id()
            self._holes[hole_id] = HoleInfo(
                hole_id=hole_id,
                kind=HoleKind.BODY,
                span=span,
                context="function body",
            )

        # Check for incomplete parameter list
        if "(" in source and source.count("(") > source.count(")"):
            hole_id = self._new_hole_id()
            self._holes[hole_id] = HoleInfo(
                hole_id=hole_id,
                kind=HoleKind.EXPRESSION,
                span=span,
                context="function parameters",
            )

        return MarkedASTNode(
            kind=ASTNodeKind.FUNCTION_DEF,
            span=span,
            data={"source": source},
        )

    def _parse_let_stmt(self, source: str) -> MarkedASTNode:
        """Parse a let statement."""
        span = self._tracker.span_at(0, len(source))

        # Check for incomplete let
        if "=" in source:
            rhs = source.split("=", 1)[-1].strip()
            if not rhs or rhs.endswith(("=", "+", "-", "*", "/", "(", "[", "{", ",", "|")):
                hole_id = self._new_hole_id()
                self._holes[hole_id] = HoleInfo(
                    hole_id=hole_id,
                    kind=HoleKind.EXPRESSION,
                    span=span,
                    context="let value",
                )
        elif ":" in source and "=" not in source:
            # Has type but no value
            hole_id = self._new_hole_id()
            self._holes[hole_id] = HoleInfo(
                hole_id=hole_id,
                kind=HoleKind.EXPRESSION,
                span=span,
                context="let value",
            )
        elif "let " in source and "=" not in source and ":" not in source:
            # Just "let x" - might need value
            if not source.endswith(";"):
                hole_id = self._new_hole_id()
                self._holes[hole_id] = HoleInfo(
                    hole_id=hole_id,
                    kind=HoleKind.EXPRESSION,
                    span=span,
                    context="let binding",
                )

        return MarkedASTNode(
            kind=ASTNodeKind.ASSIGNMENT,
            span=span,
            data={"source": source, "is_let": True},
        )

    def _parse_const_decl(self, source: str) -> MarkedASTNode:
        """Parse a const declaration."""
        span = self._tracker.span_at(0, len(source))

        # Const requires type and value
        if "=" not in source:
            hole_id = self._new_hole_id()
            self._holes[hole_id] = HoleInfo(
                hole_id=hole_id,
                kind=HoleKind.EXPRESSION,
                span=span,
                context="const value",
            )
        if ":" not in source:
            hole_id = self._new_hole_id()
            self._holes[hole_id] = HoleInfo(
                hole_id=hole_id,
                kind=HoleKind.TYPE,
                span=span,
                context="const type",
            )

        return MarkedASTNode(
            kind=ASTNodeKind.ASSIGNMENT,
            span=span,
            data={"source": source, "is_const": True},
        )

    def _parse_static_decl(self, source: str) -> MarkedASTNode:
        """Parse a static declaration."""
        span = self._tracker.span_at(0, len(source))

        # Static requires type and value
        if "=" not in source:
            hole_id = self._new_hole_id()
            self._holes[hole_id] = HoleInfo(
                hole_id=hole_id,
                kind=HoleKind.EXPRESSION,
                span=span,
                context="static value",
            )
        if ":" not in source:
            hole_id = self._new_hole_id()
            self._holes[hole_id] = HoleInfo(
                hole_id=hole_id,
                kind=HoleKind.TYPE,
                span=span,
                context="static type",
            )

        return MarkedASTNode(
            kind=ASTNodeKind.ASSIGNMENT,
            span=span,
            data={"source": source, "is_static": True},
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

        # Tuple struct with incomplete parens
        if "(" in source and source.count("(") > source.count(")"):
            hole_id = self._new_hole_id()
            self._holes[hole_id] = HoleInfo(
                hole_id=hole_id,
                kind=HoleKind.TYPE,
                span=span,
                context="tuple struct fields",
            )

        return MarkedASTNode(
            kind=ASTNodeKind.CLASS_DEF,
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
                context="enum variants",
            )

        return MarkedASTNode(
            kind=ASTNodeKind.CLASS_DEF,
            span=span,
            data={"source": source, "kind": "enum"},
        )

    def _parse_trait_def(self, source: str) -> MarkedASTNode:
        """Parse a trait definition."""
        span = self._tracker.span_at(0, len(source))

        if self._brace_depth > 0 or (source.count("{") > source.count("}")):
            hole_id = self._new_hole_id()
            self._holes[hole_id] = HoleInfo(
                hole_id=hole_id,
                kind=HoleKind.BODY,
                span=span,
                context="trait body",
            )

        return MarkedASTNode(
            kind=ASTNodeKind.CLASS_DEF,
            span=span,
            data={"source": source, "kind": "trait"},
        )

    def _parse_impl_block(self, source: str) -> MarkedASTNode:
        """Parse an impl block."""
        span = self._tracker.span_at(0, len(source))

        if self._brace_depth > 0 or (source.count("{") > source.count("}")):
            hole_id = self._new_hole_id()
            self._holes[hole_id] = HoleInfo(
                hole_id=hole_id,
                kind=HoleKind.BODY,
                span=span,
                context="impl body",
            )

        # Check for incomplete generics
        if self._angle_depth > 0:
            hole_id = self._new_hole_id()
            self._holes[hole_id] = HoleInfo(
                hole_id=hole_id,
                kind=HoleKind.TYPE,
                span=span,
                context="generic parameters",
            )

        return MarkedASTNode(
            kind=ASTNodeKind.CLASS_DEF,
            span=span,
            data={"source": source, "kind": "impl"},
        )

    def _parse_mod_decl(self, source: str) -> MarkedASTNode:
        """Parse a module declaration."""
        span = self._tracker.span_at(0, len(source))

        # Module with body
        if "{" in source and source.count("{") > source.count("}"):
            hole_id = self._new_hole_id()
            self._holes[hole_id] = HoleInfo(
                hole_id=hole_id,
                kind=HoleKind.BODY,
                span=span,
                context="module body",
            )

        return MarkedASTNode(
            kind=ASTNodeKind.MODULE,
            span=span,
            data={"source": source},
        )

    def _parse_use_decl(self, source: str) -> MarkedASTNode:
        """Parse a use declaration."""
        span = self._tracker.span_at(0, len(source))

        # Check for incomplete use tree
        if "{" in source and source.count("{") > source.count("}"):
            hole_id = self._new_hole_id()
            self._holes[hole_id] = HoleInfo(
                hole_id=hole_id,
                kind=HoleKind.EXPRESSION,
                span=span,
                context="use items",
            )

        return MarkedASTNode(
            kind=ASTNodeKind.IMPORT,
            span=span,
            data={"source": source},
        )

    def _parse_type_alias(self, source: str) -> MarkedASTNode:
        """Parse a type alias."""
        span = self._tracker.span_at(0, len(source))

        # Type alias needs = and type
        if "=" not in source:
            hole_id = self._new_hole_id()
            self._holes[hole_id] = HoleInfo(
                hole_id=hole_id,
                kind=HoleKind.TYPE,
                span=span,
                context="aliased type",
            )

        return MarkedASTNode(
            kind=ASTNodeKind.EXPRESSION,
            span=span,
            data={"source": source, "kind": "type_alias"},
        )

    def _parse_macro_def(self, source: str) -> MarkedASTNode:
        """Parse a macro definition."""
        span = self._tracker.span_at(0, len(source))

        if self._brace_depth > 0 or (source.count("{") > source.count("}")):
            hole_id = self._new_hole_id()
            self._holes[hole_id] = HoleInfo(
                hole_id=hole_id,
                kind=HoleKind.BODY,
                span=span,
                context="macro body",
            )

        return MarkedASTNode(
            kind=ASTNodeKind.EXPRESSION,
            span=span,
            data={"source": source, "kind": "macro_def"},
        )

    def _parse_attribute(self, source: str) -> MarkedASTNode:
        """Parse an attribute."""
        span = self._tracker.span_at(0, len(source))

        # Check for incomplete attribute
        if source.count("[") > source.count("]"):
            hole_id = self._new_hole_id()
            self._holes[hole_id] = HoleInfo(
                hole_id=hole_id,
                kind=HoleKind.EXPRESSION,
                span=span,
                context="attribute content",
            )

        return MarkedASTNode(
            kind=ASTNodeKind.EXPRESSION,
            span=span,
            data={"source": source, "kind": "attribute"},
        )

    def _parse_expression(self, source: str) -> MarkedASTNode:
        """Parse an expression."""
        span = self._tracker.span_at(0, len(source))

        # Check for specific expression types
        if source.startswith("if ") or source.startswith("if("):
            return self._parse_if_expr(source, span)
        elif source.startswith("match "):
            return self._parse_match_expr(source, span)
        elif source.startswith("loop ") or source.startswith("loop{"):
            return self._parse_loop_expr(source, span)
        elif source.startswith("while ") or source.startswith("while("):
            return self._parse_while_expr(source, span)
        elif source.startswith("for ") or source.startswith("for("):
            return self._parse_for_expr(source, span)

        # Try to parse as literal
        category, keyword, value = classify_rust_token(source)

        if category.name == "INT_LITERAL":
            return create_literal_node(value=value, ty=RUST_I32, span=span)
        elif category.name == "FLOAT_LITERAL":
            return create_literal_node(value=value, ty=RUST_F64, span=span)
        elif category.name == "BOOL_LITERAL":
            return create_literal_node(value=value, ty=RUST_BOOL, span=span)
        elif category.name == "STRING_LITERAL":
            return create_literal_node(
                value=value,
                ty=RustReferenceType(
                    referent=RustStringType(),
                    is_mutable=False,
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

    def _parse_match_expr(self, source: str, span: SourceSpan) -> MarkedASTNode:
        """Parse a match expression."""
        if self._brace_depth > 0:
            hole_id = self._new_hole_id()
            self._holes[hole_id] = HoleInfo(
                hole_id=hole_id,
                kind=HoleKind.EXPRESSION,
                span=span,
                context="match arms",
            )

        return MarkedASTNode(
            kind=ASTNodeKind.EXPRESSION,
            span=span,
            data={"source": source, "expr_kind": "match"},
        )

    def _parse_loop_expr(self, source: str, span: SourceSpan) -> MarkedASTNode:
        """Parse a loop expression."""
        if self._brace_depth > 0:
            hole_id = self._new_hole_id()
            self._holes[hole_id] = HoleInfo(
                hole_id=hole_id,
                kind=HoleKind.BODY,
                span=span,
                context="loop body",
            )

        return MarkedASTNode(
            kind=ASTNodeKind.EXPRESSION,
            span=span,
            data={"source": source, "expr_kind": "loop"},
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

    def _detect_holes(self) -> None:
        """Detect incomplete constructs that represent holes."""
        source = self._source.rstrip()

        if not source:
            return

        last_char = source[-1]

        # Check for trailing operators
        if last_char in "+-*/%&|^<>=!":
            # Not all are operators in all contexts, but good heuristic
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

        if self._angle_depth > 0:
            hole_id = self._new_hole_id()
            self._holes[hole_id] = HoleInfo(
                hole_id=hole_id,
                kind=HoleKind.TYPE,
                span=self._tracker.span_at(len(source), len(source)),
                context="generic parameter",
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

        # Check for trailing colon
        if last_char == ":":
            if not source.endswith("::"):
                hole_id = self._new_hole_id()
                self._holes[hole_id] = HoleInfo(
                    hole_id=hole_id,
                    kind=HoleKind.TYPE,
                    span=self._tracker.span_at(len(source), len(source)),
                    context="type annotation",
                )

        # Check for trailing arrow
        if source.endswith("->"):
            hole_id = self._new_hole_id()
            self._holes[hole_id] = HoleInfo(
                hole_id=hole_id,
                kind=HoleKind.TYPE,
                span=self._tracker.span_at(len(source), len(source)),
                context="return type",
            )

        # Check for trailing path separator
        if source.endswith("::"):
            hole_id = self._new_hole_id()
            self._holes[hole_id] = HoleInfo(
                hole_id=hole_id,
                kind=HoleKind.EXPRESSION,
                span=self._tracker.span_at(len(source), len(source)),
                context="path segment",
            )

        # Check for incomplete match arm
        if source.endswith("=>"):
            hole_id = self._new_hole_id()
            self._holes[hole_id] = HoleInfo(
                hole_id=hole_id,
                kind=HoleKind.EXPRESSION,
                span=self._tracker.span_at(len(source), len(source)),
                context="match arm expression",
            )

        # Check for incomplete closure
        if source.endswith("|"):
            # Could be end of closure params
            hole_id = self._new_hole_id()
            self._holes[hole_id] = HoleInfo(
                hole_id=hole_id,
                kind=HoleKind.EXPRESSION,
                span=self._tracker.span_at(len(source), len(source)),
                context="closure body",
            )

        # Check for incomplete reference
        if source.endswith("&") or source.endswith("&mut"):
            hole_id = self._new_hole_id()
            self._holes[hole_id] = HoleInfo(
                hole_id=hole_id,
                kind=HoleKind.EXPRESSION,
                span=self._tracker.span_at(len(source), len(source)),
                context="referenced expression",
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
def create_rust_parser() -> RustIncrementalParser:
    """Create a new Rust incremental parser.

    Returns:
        A fresh RustIncrementalParser instance
    """
    return RustIncrementalParser()
