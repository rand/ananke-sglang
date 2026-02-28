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
"""Unit tests for Rust incremental parser.

Tests for the RustIncrementalParser including:
- Basic parsing functionality
- Hole detection
- Expected token prediction
- Checkpoint and restore
- Various Rust constructs
"""

import pytest

from parsing.base import ParseState, TokenInfo
from parsing.languages.rust import (
    RustIncrementalParser,
    RustParserCheckpoint,
    create_rust_parser,
    EXPRESSION_STARTERS,
    EXPRESSION_ENDERS,
)
from parsing.partial_ast import HoleKind
from domains.types.marking.marked_ast import ASTNodeKind


# ===========================================================================
# Parser Creation Tests
# ===========================================================================


class TestRustParserCreation:
    """Tests for parser creation."""

    def test_create_parser(self):
        """Should create parser via factory function."""
        parser = create_rust_parser()
        assert isinstance(parser, RustIncrementalParser)

    def test_parser_language(self):
        """Parser should report 'rust' language."""
        parser = RustIncrementalParser()
        assert parser.language == "rust"

    def test_initial_state(self):
        """Parser should start with empty state."""
        parser = RustIncrementalParser()
        assert parser.current_source == ""
        assert parser.current_position == 0


# ===========================================================================
# Basic Parsing Tests
# ===========================================================================


class TestRustBasicParsing:
    """Tests for basic parsing functionality."""

    @pytest.fixture
    def parser(self):
        return RustIncrementalParser()

    def test_parse_empty(self, parser):
        """Parsing empty source should create hole."""
        result = parser.parse_initial("")
        assert result.state == ParseState.PARTIAL
        assert len(result.holes) > 0

    def test_parse_simple_let(self, parser):
        """Should parse simple let binding."""
        result = parser.parse_initial("let x = 42;")
        assert result.ast is not None

    def test_parse_let_mut(self, parser):
        """Should parse mutable let binding."""
        result = parser.parse_initial("let mut y: i32 = 0;")
        assert result.ast is not None

    def test_parse_function(self, parser):
        """Should parse complete function."""
        source = """fn add(a: i32, b: i32) -> i32 {
    a + b
}"""
        result = parser.parse_initial(source)
        assert result.ast is not None
        assert result.ast.kind == ASTNodeKind.FUNCTION_DEF

    def test_parse_struct(self, parser):
        """Should parse struct definition."""
        source = """struct Point {
    x: f32,
    y: f32,
}"""
        result = parser.parse_initial(source)
        assert result.ast is not None

    def test_parse_enum(self, parser):
        """Should parse enum definition."""
        source = """enum Color {
    Red,
    Green,
    Blue,
}"""
        result = parser.parse_initial(source)
        assert result.ast is not None


# ===========================================================================
# Hole Detection Tests
# ===========================================================================


class TestRustHoleDetection:
    """Tests for hole detection in incomplete code."""

    @pytest.fixture
    def parser(self):
        return RustIncrementalParser()

    def test_detect_missing_function_body(self, parser):
        """Should detect missing function body."""
        result = parser.parse_initial("fn foo(x: i32) -> i32")
        holes = parser.find_holes()
        assert len(holes) > 0

    def test_detect_incomplete_let(self, parser):
        """Should detect incomplete let assignment."""
        result = parser.parse_initial("let x =")
        holes = parser.find_holes()
        assert len(holes) > 0

    def test_detect_unclosed_brace(self, parser):
        """Should detect unclosed brace."""
        result = parser.parse_initial("fn foo() {")
        holes = parser.find_holes()
        assert len(holes) > 0

    def test_detect_unclosed_paren(self, parser):
        """Should detect unclosed parenthesis."""
        result = parser.parse_initial("fn foo(x: i32")
        holes = parser.find_holes()
        assert len(holes) > 0

    def test_detect_trailing_operator(self, parser):
        """Should detect trailing operator."""
        result = parser.parse_initial("let x = a +")
        holes = parser.find_holes()
        assert len(holes) > 0

    def test_detect_incomplete_if(self, parser):
        """Should detect incomplete if expression."""
        result = parser.parse_initial("if x > 0 {")
        holes = parser.find_holes()
        assert len(holes) > 0

    def test_detect_missing_type(self, parser):
        """Should detect missing type annotation."""
        result = parser.parse_initial("let x:")
        holes = parser.find_holes()
        assert len(holes) > 0

    def test_detect_incomplete_match(self, parser):
        """Should detect incomplete match expression."""
        result = parser.parse_initial("match x {")
        holes = parser.find_holes()
        assert len(holes) > 0

    def test_detect_incomplete_generic(self, parser):
        """Should detect incomplete generic parameter."""
        result = parser.parse_initial("Vec<")
        holes = parser.find_holes()
        assert len(holes) > 0

    def test_detect_trailing_arrow(self, parser):
        """Should detect trailing arrow."""
        result = parser.parse_initial("fn foo() ->")
        holes = parser.find_holes()
        assert len(holes) > 0

    def test_detect_trailing_path_sep(self, parser):
        """Should detect trailing path separator."""
        result = parser.parse_initial("std::")
        holes = parser.find_holes()
        assert len(holes) > 0


# ===========================================================================
# Expected Token Tests
# ===========================================================================


class TestRustExpectedTokens:
    """Tests for expected token prediction."""

    @pytest.fixture
    def parser(self):
        return RustIncrementalParser()

    def test_expected_after_fn(self, parser):
        """After 'fn ', should expect identifier."""
        parser.parse_initial("fn ")
        expected = parser.get_expected_tokens()
        assert "identifier" in expected

    def test_expected_after_open_paren(self, parser):
        """After '(', should expect parameter-related tokens."""
        parser.parse_initial("fn foo(")
        expected = parser.get_expected_tokens()
        assert "identifier" in expected or ")" in expected or "self" in expected

    def test_expected_after_colon(self, parser):
        """After ':', should expect type tokens."""
        parser.parse_initial("let x:")
        expected = parser.get_expected_tokens()
        assert "identifier" in expected or "&" in expected

    def test_expected_after_arrow(self, parser):
        """After '->', should expect return type."""
        parser.parse_initial("fn foo() ->")
        expected = parser.get_expected_tokens()
        assert "identifier" in expected or "&" in expected

    def test_expected_after_open_brace(self, parser):
        """After '{', should expect statement/expression tokens."""
        parser.parse_initial("fn foo() {")
        expected = parser.get_expected_tokens()
        assert "let" in expected or "identifier" in expected

    def test_expected_after_pub(self, parser):
        """After 'pub ', should expect tokens for continuing the declaration."""
        parser.parse_initial("pub ")
        expected = parser.get_expected_tokens()
        # Parser should return some expected tokens (may include 'fn', 'struct', 'identifier', etc.)
        assert len(expected) > 0

    def test_expected_after_let(self, parser):
        """After 'let ', should expect pattern."""
        parser.parse_initial("let ")
        expected = parser.get_expected_tokens()
        assert "identifier" in expected or "mut" in expected

    def test_expected_after_let_mut(self, parser):
        """After 'let mut ', should expect identifier."""
        parser.parse_initial("let mut ")
        expected = parser.get_expected_tokens()
        assert "identifier" in expected

    def test_expected_after_impl(self, parser):
        """After 'impl ', should expect type or generic."""
        parser.parse_initial("impl ")
        expected = parser.get_expected_tokens()
        assert "identifier" in expected or "<" in expected

    def test_expected_after_match(self, parser):
        """After 'match ', should expect expression."""
        parser.parse_initial("match ")
        expected = parser.get_expected_tokens()
        assert "identifier" in expected


# ===========================================================================
# Incremental Parsing Tests
# ===========================================================================


class TestRustIncrementalParsing:
    """Tests for incremental parsing."""

    @pytest.fixture
    def parser(self):
        return RustIncrementalParser()

    def test_extend_with_text(self, parser):
        """Should extend source with text."""
        parser.parse_initial("let ")
        result = parser.extend_with_text("x")
        assert parser.current_source == "let x"

    def test_extend_builds_incrementally(self, parser):
        """Should build source incrementally."""
        parser.parse_initial("fn")
        parser.extend_with_text(" foo")
        parser.extend_with_text("()")
        parser.extend_with_text(" -> i32")
        parser.extend_with_text(" { 42 }")
        assert "fn foo() -> i32 { 42 }" in parser.current_source

    def test_extend_with_token(self, parser):
        """Should extend with token info."""
        parser.parse_initial("let ")
        token = TokenInfo(token_id=0, text="x", position=4, length=1)
        result = parser.extend_with_token(token)
        assert parser.current_source == "let x"

    def test_holes_update_on_extend(self, parser):
        """Holes should update as code is extended."""
        parser.parse_initial("fn foo(x: i32)")
        holes_before = len(parser.find_holes())

        parser.extend_with_text(" -> i32 { x }")
        holes_after = len(parser.find_holes())

        # Should have fewer holes after completing the function
        assert holes_after <= holes_before


# ===========================================================================
# Checkpoint and Restore Tests
# ===========================================================================


class TestRustParserCheckpoint:
    """Tests for checkpoint and restore functionality."""

    @pytest.fixture
    def parser(self):
        return RustIncrementalParser()

    def test_create_checkpoint(self, parser):
        """Should create checkpoint."""
        parser.parse_initial("let x = ")
        checkpoint = parser.checkpoint()
        assert isinstance(checkpoint, RustParserCheckpoint)
        assert checkpoint.source == "let x = "

    def test_restore_checkpoint(self, parser):
        """Should restore from checkpoint."""
        parser.parse_initial("let x = ")
        checkpoint = parser.checkpoint()

        # Extend parser
        parser.extend_with_text("42;")
        assert parser.current_source == "let x = 42;"

        # Restore
        parser.restore(checkpoint)
        assert parser.current_source == "let x = "

    def test_checkpoint_preserves_holes(self, parser):
        """Checkpoint should preserve hole state."""
        parser.parse_initial("fn foo(")
        checkpoint = parser.checkpoint()
        holes_at_checkpoint = len(parser.find_holes())

        parser.extend_with_text("x: i32) -> i32 { x }")
        assert len(parser.find_holes()) != holes_at_checkpoint

        parser.restore(checkpoint)
        assert len(parser.find_holes()) == holes_at_checkpoint

    def test_checkpoint_preserves_bracket_depth(self, parser):
        """Checkpoint should preserve bracket depth."""
        parser.parse_initial("fn foo() {")
        checkpoint = parser.checkpoint()

        parser.extend_with_text(" if x { y(); }")
        parser.restore(checkpoint)

        assert parser._brace_depth == checkpoint.brace_depth


# ===========================================================================
# Copy Tests
# ===========================================================================


class TestRustParserCopy:
    """Tests for parser copy functionality."""

    @pytest.fixture
    def parser(self):
        return RustIncrementalParser()

    def test_copy_creates_independent_parser(self, parser):
        """Copy should create independent parser."""
        parser.parse_initial("let x = ")
        copy = parser.copy()

        # Extend original
        parser.extend_with_text("42;")

        # Copy should be unaffected
        assert copy.current_source == "let x = "

    def test_copy_preserves_state(self, parser):
        """Copy should preserve parser state."""
        parser.parse_initial("fn foo(x: i32)")
        copy = parser.copy()

        assert copy.current_source == parser.current_source
        assert copy.language == parser.language


# ===========================================================================
# Construct-Specific Parsing Tests
# ===========================================================================


class TestRustConstructParsing:
    """Tests for parsing specific Rust constructs."""

    @pytest.fixture
    def parser(self):
        return RustIncrementalParser()

    def test_parse_trait(self, parser):
        """Should parse trait definition."""
        source = """trait Processor {
    fn process(&self);
}"""
        result = parser.parse_initial(source)
        assert result.ast is not None

    def test_parse_impl_block(self, parser):
        """Should parse impl block."""
        source = """impl Point {
    fn new(x: f32, y: f32) -> Self {
        Point { x, y }
    }
}"""
        result = parser.parse_initial(source)
        assert result.ast is not None

    def test_parse_trait_impl(self, parser):
        """Should parse trait implementation."""
        source = """impl Clone for Point {
    fn clone(&self) -> Self {
        Point { x: self.x, y: self.y }
    }
}"""
        result = parser.parse_initial(source)
        assert result.ast is not None

    def test_parse_use_declaration(self, parser):
        """Should parse use declaration."""
        result = parser.parse_initial("use std::collections::HashMap;")
        assert result.ast is not None
        assert result.ast.kind == ASTNodeKind.IMPORT

    def test_parse_mod_declaration(self, parser):
        """Should parse mod declaration."""
        result = parser.parse_initial("mod tests { }")
        assert result.ast is not None

    def test_parse_if_expression(self, parser):
        """Should parse if expression."""
        result = parser.parse_initial("if x > 0 { x } else { -x }")
        assert result.ast is not None

    def test_parse_match_expression(self, parser):
        """Should parse match expression."""
        source = """match x {
    Some(v) => v,
    None => 0,
}"""
        result = parser.parse_initial(source)
        assert result.ast is not None

    def test_parse_loop_expression(self, parser):
        """Should parse loop expression."""
        result = parser.parse_initial("loop { break 42; }")
        assert result.ast is not None

    def test_parse_while_expression(self, parser):
        """Should parse while expression."""
        result = parser.parse_initial("while condition { body(); }")
        assert result.ast is not None

    def test_parse_for_expression(self, parser):
        """Should parse for expression."""
        result = parser.parse_initial("for item in items { process(item); }")
        assert result.ast is not None

    def test_parse_async_fn(self, parser):
        """Should parse async function."""
        result = parser.parse_initial("async fn fetch() -> Result<(), Error> { Ok(()) }")
        assert result.ast is not None

    def test_parse_attribute(self, parser):
        """Should parse attribute."""
        result = parser.parse_initial("#[derive(Debug)]")
        assert result.ast is not None

    def test_parse_const_declaration(self, parser):
        """Should parse const declaration."""
        result = parser.parse_initial("const MAX: usize = 100;")
        assert result.ast is not None

    def test_parse_static_declaration(self, parser):
        """Should parse static declaration."""
        result = parser.parse_initial("static COUNTER: AtomicU32 = AtomicU32::new(0);")
        assert result.ast is not None

    def test_parse_type_alias(self, parser):
        """Should parse type alias."""
        result = parser.parse_initial("type Result<T> = std::result::Result<T, Error>;")
        assert result.ast is not None


# ===========================================================================
# Literal Parsing Tests
# ===========================================================================


class TestRustLiteralParsing:
    """Tests for literal parsing."""

    @pytest.fixture
    def parser(self):
        return RustIncrementalParser()

    def test_parse_integer_literal(self, parser):
        """Should parse integer literals."""
        result = parser.parse_initial("42")
        assert result.ast is not None

    def test_parse_float_literal(self, parser):
        """Should parse float literals."""
        result = parser.parse_initial("3.14")
        assert result.ast is not None

    def test_parse_bool_literal(self, parser):
        """Should parse boolean literals."""
        result = parser.parse_initial("true")
        assert result.ast is not None

    def test_parse_string_literal(self, parser):
        """Should parse string literals."""
        result = parser.parse_initial('"hello"')
        assert result.ast is not None

    def test_parse_char_literal(self, parser):
        """Should parse character literals."""
        result = parser.parse_initial("'a'")
        assert result.ast is not None


# ===========================================================================
# AST Node Tests
# ===========================================================================


class TestRustASTNodes:
    """Tests for AST node creation."""

    @pytest.fixture
    def parser(self):
        return RustIncrementalParser()

    def test_function_creates_function_def_node(self, parser):
        """Function should create FUNCTION_DEF node."""
        result = parser.parse_initial("fn foo() {}")
        assert result.ast.kind == ASTNodeKind.FUNCTION_DEF

    def test_let_creates_assignment_node(self, parser):
        """Let should create ASSIGNMENT node."""
        result = parser.parse_initial("let x = 42;")
        assert result.ast.kind == ASTNodeKind.ASSIGNMENT
        assert result.ast.data.get("is_let") is True

    def test_const_creates_assignment_node(self, parser):
        """Const should create ASSIGNMENT node."""
        result = parser.parse_initial("const X: i32 = 42;")
        assert result.ast.kind == ASTNodeKind.ASSIGNMENT
        assert result.ast.data.get("is_const") is True

    def test_struct_creates_class_def_node(self, parser):
        """Struct should create CLASS_DEF node."""
        result = parser.parse_initial("struct Point { x: f32, y: f32 }")
        assert result.ast.kind == ASTNodeKind.CLASS_DEF
        assert result.ast.data.get("kind") == "struct"

    def test_enum_creates_class_def_node(self, parser):
        """Enum should create CLASS_DEF node."""
        result = parser.parse_initial("enum Color { Red, Green, Blue }")
        assert result.ast.kind == ASTNodeKind.CLASS_DEF
        assert result.ast.data.get("kind") == "enum"

    def test_use_creates_import_node(self, parser):
        """Use should create IMPORT node."""
        result = parser.parse_initial("use std::io;")
        assert result.ast.kind == ASTNodeKind.IMPORT

    def test_mod_creates_module_node(self, parser):
        """Mod should create MODULE node."""
        result = parser.parse_initial("mod tests;")
        assert result.ast.kind == ASTNodeKind.MODULE


# ===========================================================================
# Error Recovery Tests
# ===========================================================================


class TestRustErrorRecovery:
    """Tests for error recovery during parsing."""

    @pytest.fixture
    def parser(self):
        return RustIncrementalParser()

    def test_recover_from_syntax_error(self, parser):
        """Should recover from syntax errors."""
        result = parser.parse_initial("fn ()")  # Missing function name
        assert result.ast is not None
        assert result.state == ParseState.PARTIAL

    def test_continue_after_error(self, parser):
        """Should allow continuing after error."""
        parser.parse_initial("fn foo(")
        result = parser.extend_with_text("x: i32) -> i32 { x }")
        assert parser.current_source == "fn foo(x: i32) -> i32 { x }"


# ===========================================================================
# Edge Cases
# ===========================================================================


class TestRustParserEdgeCases:
    """Tests for edge cases."""

    @pytest.fixture
    def parser(self):
        return RustIncrementalParser()

    def test_whitespace_only(self, parser):
        """Should handle whitespace-only source."""
        result = parser.parse_initial("   \n\t  ")
        assert result.state == ParseState.PARTIAL

    def test_nested_brackets(self, parser):
        """Should track nested brackets correctly."""
        parser.parse_initial("fn foo() { if x { if y { z(); } } }")
        assert parser._brace_depth == 0
        assert parser._paren_depth == 0

    def test_nested_generics(self, parser):
        """Should track nested generics correctly."""
        parser.parse_initial("HashMap<String, Vec<i32>>")
        assert parser._angle_depth == 0

    def test_multiple_extends(self, parser):
        """Should handle many sequential extends."""
        parser.parse_initial("let ")
        for char in "x = 42;":
            parser.extend_with_text(char)
        assert parser.current_source == "let x = 42;"

    def test_get_ast(self, parser):
        """get_ast should return current AST."""
        parser.parse_initial("let x = 1;")
        ast = parser.get_ast()
        assert ast is not None
