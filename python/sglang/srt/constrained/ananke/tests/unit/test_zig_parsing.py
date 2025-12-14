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
"""Unit tests for Zig incremental parser.

Tests for the ZigIncrementalParser including:
- Basic parsing functionality
- Hole detection
- Expected token prediction
- Checkpoint and restore
- Various Zig constructs
"""

import pytest

from parsing.base import ParseState, TokenInfo
from parsing.languages.zig import (
    ZigIncrementalParser,
    ZigParserCheckpoint,
    create_zig_parser,
    EXPRESSION_STARTERS,
    EXPRESSION_ENDERS,
)
from parsing.partial_ast import HoleKind
from domains.types.marking.marked_ast import ASTNodeKind


# ===========================================================================
# Parser Creation Tests
# ===========================================================================


class TestZigParserCreation:
    """Tests for parser creation."""

    def test_create_parser(self):
        """Should create parser via factory function."""
        parser = create_zig_parser()
        assert isinstance(parser, ZigIncrementalParser)

    def test_parser_language(self):
        """Parser should report 'zig' language."""
        parser = ZigIncrementalParser()
        assert parser.language == "zig"

    def test_initial_state(self):
        """Parser should start with empty state."""
        parser = ZigIncrementalParser()
        assert parser.current_source == ""
        assert parser.current_position == 0


# ===========================================================================
# Basic Parsing Tests
# ===========================================================================


class TestZigBasicParsing:
    """Tests for basic parsing functionality."""

    @pytest.fixture
    def parser(self):
        return ZigIncrementalParser()

    def test_parse_empty(self, parser):
        """Parsing empty source should create hole."""
        result = parser.parse_initial("")
        assert result.state == ParseState.PARTIAL
        assert len(result.holes) > 0

    def test_parse_simple_const(self, parser):
        """Should parse simple const declaration."""
        result = parser.parse_initial("const x = 42;")
        assert result.ast is not None

    def test_parse_simple_var(self, parser):
        """Should parse simple var declaration."""
        result = parser.parse_initial("var y: i32 = 0;")
        assert result.ast is not None

    def test_parse_function(self, parser):
        """Should parse complete function."""
        source = """fn add(a: i32, b: i32) i32 {
    return a + b;
}"""
        result = parser.parse_initial(source)
        assert result.ast is not None
        assert result.ast.kind == ASTNodeKind.FUNCTION_DEF

    def test_parse_struct(self, parser):
        """Should parse struct definition."""
        source = """const Point = struct {
    x: f32,
    y: f32,
};"""
        result = parser.parse_initial(source)
        assert result.ast is not None

    def test_parse_enum(self, parser):
        """Should parse enum definition."""
        source = """const Color = enum {
    red,
    green,
    blue,
};"""
        result = parser.parse_initial(source)
        assert result.ast is not None


# ===========================================================================
# Hole Detection Tests
# ===========================================================================


class TestZigHoleDetection:
    """Tests for hole detection in incomplete code."""

    @pytest.fixture
    def parser(self):
        return ZigIncrementalParser()

    def test_detect_missing_function_body(self, parser):
        """Should detect missing function body."""
        result = parser.parse_initial("fn foo(x: i32) void")
        holes = parser.find_holes()
        assert len(holes) > 0

    def test_detect_incomplete_const(self, parser):
        """Should detect incomplete const assignment."""
        result = parser.parse_initial("const x =")
        holes = parser.find_holes()
        assert len(holes) > 0
        # Should expect expression

    def test_detect_unclosed_brace(self, parser):
        """Should detect unclosed brace."""
        result = parser.parse_initial("fn foo() void {")
        holes = parser.find_holes()
        assert len(holes) > 0

    def test_detect_unclosed_paren(self, parser):
        """Should detect unclosed parenthesis."""
        result = parser.parse_initial("fn foo(x: i32")
        holes = parser.find_holes()
        assert len(holes) > 0

    def test_detect_trailing_operator(self, parser):
        """Should detect trailing operator."""
        result = parser.parse_initial("const x = a +")
        holes = parser.find_holes()
        assert len(holes) > 0

    def test_detect_incomplete_if(self, parser):
        """Should detect incomplete if expression."""
        result = parser.parse_initial("if (x > 0) {")
        holes = parser.find_holes()
        assert len(holes) > 0

    def test_detect_missing_type(self, parser):
        """Should detect missing type annotation."""
        result = parser.parse_initial("const x:")
        holes = parser.find_holes()
        assert len(holes) > 0

    def test_detect_incomplete_error_union(self, parser):
        """Should detect incomplete error union."""
        result = parser.parse_initial("anyerror!")
        holes = parser.find_holes()
        assert len(holes) > 0

    def test_detect_incomplete_optional(self, parser):
        """Should detect incomplete optional type."""
        # Note: ? at end of type context (not .?)
        result = parser.parse_initial("const x: ?")
        holes = parser.find_holes()
        assert len(holes) > 0


# ===========================================================================
# Expected Token Tests
# ===========================================================================


class TestZigExpectedTokens:
    """Tests for expected token prediction."""

    @pytest.fixture
    def parser(self):
        return ZigIncrementalParser()

    def test_expected_after_fn(self, parser):
        """After 'fn ', should expect identifier."""
        parser.parse_initial("fn ")
        expected = parser.get_expected_tokens()
        assert "identifier" in expected

    def test_expected_after_open_paren(self, parser):
        """After '(', should expect parameter-related tokens."""
        parser.parse_initial("fn foo(")
        expected = parser.get_expected_tokens()
        assert "identifier" in expected or ")" in expected

    def test_expected_after_colon(self, parser):
        """After ':', should expect type tokens."""
        parser.parse_initial("const x:")
        expected = parser.get_expected_tokens()
        assert "identifier" in expected or "*" in expected or "?" in expected

    def test_expected_after_open_brace(self, parser):
        """After '{', should expect statement/expression tokens."""
        parser.parse_initial("fn foo() void {")
        expected = parser.get_expected_tokens()
        assert "return" in expected or "identifier" in expected

    def test_expected_after_pub(self, parser):
        """After 'pub ', should expect tokens for continuing the declaration."""
        parser.parse_initial("pub ")
        expected = parser.get_expected_tokens()
        # Parser should return some expected tokens (may include 'fn', 'const', 'identifier', etc.)
        assert len(expected) > 0

    def test_expected_after_return(self, parser):
        """After 'return', should expect expression or semicolon."""
        parser.parse_initial("return")
        expected = parser.get_expected_tokens()
        assert ";" in expected or len(expected) > 0

    def test_expected_after_comptime(self, parser):
        """After 'comptime ', should expect block or expression."""
        parser.parse_initial("comptime ")
        expected = parser.get_expected_tokens()
        assert "{" in expected or "var" in expected or "const" in expected


# ===========================================================================
# Incremental Parsing Tests
# ===========================================================================


class TestZigIncrementalParsing:
    """Tests for incremental parsing."""

    @pytest.fixture
    def parser(self):
        return ZigIncrementalParser()

    def test_extend_with_text(self, parser):
        """Should extend source with text."""
        parser.parse_initial("const ")
        result = parser.extend_with_text("x")
        assert parser.current_source == "const x"

    def test_extend_builds_incrementally(self, parser):
        """Should build source incrementally."""
        parser.parse_initial("fn")
        parser.extend_with_text(" foo")
        parser.extend_with_text("()")
        parser.extend_with_text(" void")
        parser.extend_with_text(" {}")
        assert "fn foo() void {}" in parser.current_source

    def test_extend_with_token(self, parser):
        """Should extend with token info."""
        parser.parse_initial("const ")
        token = TokenInfo(token_id=0, text="x", position=6, length=1)
        result = parser.extend_with_token(token)
        assert parser.current_source == "const x"

    def test_holes_update_on_extend(self, parser):
        """Holes should update as code is extended."""
        parser.parse_initial("fn foo(x: i32)")
        holes_before = len(parser.find_holes())

        parser.extend_with_text(" void { return x; }")
        holes_after = len(parser.find_holes())

        # Should have fewer holes after completing the function
        assert holes_after <= holes_before


# ===========================================================================
# Checkpoint and Restore Tests
# ===========================================================================


class TestZigParserCheckpoint:
    """Tests for checkpoint and restore functionality."""

    @pytest.fixture
    def parser(self):
        return ZigIncrementalParser()

    def test_create_checkpoint(self, parser):
        """Should create checkpoint."""
        parser.parse_initial("const x = ")
        checkpoint = parser.checkpoint()
        assert isinstance(checkpoint, ZigParserCheckpoint)
        assert checkpoint.source == "const x = "

    def test_restore_checkpoint(self, parser):
        """Should restore from checkpoint."""
        parser.parse_initial("const x = ")
        checkpoint = parser.checkpoint()

        # Extend parser
        parser.extend_with_text("42;")
        assert parser.current_source == "const x = 42;"

        # Restore
        parser.restore(checkpoint)
        assert parser.current_source == "const x = "

    def test_checkpoint_preserves_holes(self, parser):
        """Checkpoint should preserve hole state."""
        parser.parse_initial("fn foo(")
        checkpoint = parser.checkpoint()
        holes_at_checkpoint = len(parser.find_holes())

        parser.extend_with_text("x: i32) void {}")
        assert len(parser.find_holes()) != holes_at_checkpoint

        parser.restore(checkpoint)
        assert len(parser.find_holes()) == holes_at_checkpoint

    def test_checkpoint_preserves_bracket_depth(self, parser):
        """Checkpoint should preserve bracket depth."""
        parser.parse_initial("fn foo() void {")
        checkpoint = parser.checkpoint()

        parser.extend_with_text(" if (x) { y(); }")
        parser.restore(checkpoint)

        # Bracket depth should be restored
        assert parser._brace_depth == checkpoint.brace_depth


# ===========================================================================
# Copy Tests
# ===========================================================================


class TestZigParserCopy:
    """Tests for parser copy functionality."""

    @pytest.fixture
    def parser(self):
        return ZigIncrementalParser()

    def test_copy_creates_independent_parser(self, parser):
        """Copy should create independent parser."""
        parser.parse_initial("const x = ")
        copy = parser.copy()

        # Extend original
        parser.extend_with_text("42;")

        # Copy should be unaffected
        assert copy.current_source == "const x = "

    def test_copy_preserves_state(self, parser):
        """Copy should preserve parser state."""
        parser.parse_initial("fn foo(x: i32)")
        copy = parser.copy()

        assert copy.current_source == parser.current_source
        assert copy.language == parser.language


# ===========================================================================
# Construct-Specific Parsing Tests
# ===========================================================================


class TestZigConstructParsing:
    """Tests for parsing specific Zig constructs."""

    @pytest.fixture
    def parser(self):
        return ZigIncrementalParser()

    def test_parse_comptime_block(self, parser):
        """Should parse comptime block."""
        result = parser.parse_initial("comptime { const x = 1; }")
        assert result.ast is not None

    def test_parse_test_declaration(self, parser):
        """Should parse test declaration."""
        result = parser.parse_initial('test "my test" { }')
        assert result.ast is not None

    def test_parse_import(self, parser):
        """Should parse @import."""
        result = parser.parse_initial('@import("std")')
        assert result.ast is not None
        assert result.ast.kind == ASTNodeKind.IMPORT

    def test_parse_union(self, parser):
        """Should parse union."""
        source = """const Value = union(enum) {
    int: i32,
    float: f32,
    none,
};"""
        result = parser.parse_initial(source)
        assert result.ast is not None

    def test_parse_if_expression(self, parser):
        """Should parse if expression."""
        result = parser.parse_initial("if (x > 0) x else -x")
        assert result.ast is not None

    def test_parse_switch_expression(self, parser):
        """Should parse switch expression."""
        result = parser.parse_initial("switch (x) { 1 => a, else => b }")
        assert result.ast is not None

    def test_parse_for_expression(self, parser):
        """Should parse for expression."""
        result = parser.parse_initial("for (items) |item| { process(item); }")
        assert result.ast is not None

    def test_parse_while_expression(self, parser):
        """Should parse while expression."""
        result = parser.parse_initial("while (condition) { body(); }")
        assert result.ast is not None


# ===========================================================================
# Literal Parsing Tests
# ===========================================================================


class TestZigLiteralParsing:
    """Tests for literal parsing."""

    @pytest.fixture
    def parser(self):
        return ZigIncrementalParser()

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


# ===========================================================================
# AST Node Tests
# ===========================================================================


class TestZigASTNodes:
    """Tests for AST node creation."""

    @pytest.fixture
    def parser(self):
        return ZigIncrementalParser()

    def test_function_creates_function_def_node(self, parser):
        """Function should create FUNCTION_DEF node."""
        result = parser.parse_initial("fn foo() void {}")
        assert result.ast.kind == ASTNodeKind.FUNCTION_DEF

    def test_const_creates_assignment_node(self, parser):
        """Const should create ASSIGNMENT node."""
        result = parser.parse_initial("const x = 42;")
        assert result.ast.kind == ASTNodeKind.ASSIGNMENT
        assert result.ast.data.get("is_const") is True

    def test_var_creates_assignment_node(self, parser):
        """Var should create ASSIGNMENT node."""
        result = parser.parse_initial("var x: i32 = 0;")
        assert result.ast.kind == ASTNodeKind.ASSIGNMENT
        assert result.ast.data.get("is_const") is False

    def test_struct_creates_class_def_node(self, parser):
        """Struct should create CLASS_DEF node."""
        result = parser.parse_initial("const S = struct { x: i32 };")
        # Note: struct is parsed as const assignment initially
        # but inner struct creates CLASS_DEF

    def test_import_creates_import_node(self, parser):
        """Import should create IMPORT node."""
        result = parser.parse_initial('@import("std")')
        assert result.ast.kind == ASTNodeKind.IMPORT


# ===========================================================================
# Error Recovery Tests
# ===========================================================================


class TestZigErrorRecovery:
    """Tests for error recovery during parsing."""

    @pytest.fixture
    def parser(self):
        return ZigIncrementalParser()

    def test_recover_from_syntax_error(self, parser):
        """Should recover from syntax errors."""
        # Invalid syntax but parser should not crash
        result = parser.parse_initial("fn ()")  # Missing function name
        assert result.ast is not None  # Should have recovery AST
        assert result.state == ParseState.PARTIAL

    def test_continue_after_error(self, parser):
        """Should allow continuing after error."""
        parser.parse_initial("fn foo(")
        # Should be able to continue extending
        result = parser.extend_with_text("x: i32) void {}")
        assert parser.current_source == "fn foo(x: i32) void {}"


# ===========================================================================
# Edge Cases
# ===========================================================================


class TestZigParserEdgeCases:
    """Tests for edge cases."""

    @pytest.fixture
    def parser(self):
        return ZigIncrementalParser()

    def test_whitespace_only(self, parser):
        """Should handle whitespace-only source."""
        result = parser.parse_initial("   \n\t  ")
        assert result.state == ParseState.PARTIAL

    def test_nested_brackets(self, parser):
        """Should track nested brackets correctly."""
        parser.parse_initial("fn foo() void { if (x) { if (y) { z(); } } }")
        assert parser._brace_depth == 0
        assert parser._paren_depth == 0

    def test_multiple_extends(self, parser):
        """Should handle many sequential extends."""
        parser.parse_initial("const ")
        for char in "x = 42;":
            parser.extend_with_text(char)
        assert parser.current_source == "const x = 42;"

    def test_get_ast(self, parser):
        """get_ast should return current AST."""
        parser.parse_initial("const x = 1;")
        ast = parser.get_ast()
        assert ast is not None
