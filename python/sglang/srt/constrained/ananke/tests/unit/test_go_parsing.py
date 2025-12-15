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
"""Tests for Go incremental parser."""

import pytest

from parsing.languages.go import (
    GoIncrementalParser,
    GoParserCheckpoint,
    create_go_parser,
    GO_EXPRESSION_STARTERS,
    GO_STATEMENT_STARTERS,
    GO_DECLARATION_STARTERS,
    GO_BINARY_OPERATORS,
)
from parsing.base import ParseState
from parsing.partial_ast import HoleKind


class TestGoParserCreation:
    """Tests for Go parser creation."""

    def test_create_parser(self):
        """Should create a Go parser."""
        parser = GoIncrementalParser()
        assert parser.language == "go"
        assert parser.current_source == ""

    def test_factory_function(self):
        """Should create parser via factory function."""
        parser = create_go_parser()
        assert isinstance(parser, GoIncrementalParser)


class TestGoParserInitialParse:
    """Tests for Go initial parsing."""

    @pytest.fixture
    def parser(self):
        return GoIncrementalParser()

    def test_parse_empty(self, parser):
        """Should parse empty source."""
        result = parser.parse_initial("")
        assert result.is_valid
        assert result.errors == []

    def test_parse_package_declaration(self, parser):
        """Should parse package declaration."""
        result = parser.parse_initial("package main")
        assert result.is_valid

    def test_parse_import(self, parser):
        """Should parse import statement."""
        result = parser.parse_initial('import "fmt"')
        assert result.is_valid

    def test_parse_import_block(self, parser):
        """Should parse import block."""
        source = '''import (
    "fmt"
    "os"
)'''
        result = parser.parse_initial(source)
        assert result.is_valid

    def test_parse_function(self, parser):
        """Should parse function declaration."""
        source = '''func main() {
    fmt.Println("Hello")
}'''
        result = parser.parse_initial(source)
        assert result.is_valid

    def test_parse_incomplete_function(self, parser):
        """Should detect incomplete function."""
        result = parser.parse_initial("func main() {")
        assert result.state == ParseState.PARTIAL  # Unclosed brace


class TestGoParserExtend:
    """Tests for Go parser extension."""

    @pytest.fixture
    def parser(self):
        return GoIncrementalParser()

    def test_extend_empty_source(self, parser):
        """Should extend empty source."""
        parser.parse_initial("")
        result = parser.extend_with_text("package main")
        assert result.is_valid
        assert parser.current_source == "package main"

    def test_extend_multiple_times(self, parser):
        """Should extend multiple times."""
        parser.parse_initial("package main\n")
        parser.extend_with_text('import "fmt"\n')
        result = parser.extend_with_text("func main() {}")
        assert result.is_valid

    def test_extend_with_token(self, parser):
        """Should extend with token."""
        parser.parse_initial("package ")
        result = parser.extend_with_token(0, "main")
        assert "package main" in parser.current_source


class TestGoParserBracketTracking:
    """Tests for Go parser bracket tracking."""

    @pytest.fixture
    def parser(self):
        return GoIncrementalParser()

    def test_track_parentheses(self, parser):
        """Should track parentheses."""
        parser.parse_initial("func foo(")
        result = parser.parse_initial("func foo()")
        assert result.is_valid

    def test_track_braces(self, parser):
        """Should track braces."""
        parser.parse_initial("func foo() {")
        result = parser._create_result()
        assert result.state == ParseState.PARTIAL  # Unclosed brace

        parser.extend_with_text("}")
        result = parser._create_result()
        assert result.is_valid

    def test_track_brackets(self, parser):
        """Should track brackets."""
        parser.parse_initial("var x []int")
        result = parser._create_result()
        assert result.is_valid

    def test_nested_brackets(self, parser):
        """Should track nested brackets."""
        parser.parse_initial("func foo() { if true { } }")
        result = parser._create_result()
        assert result.is_valid


class TestGoParserHoleDetection:
    """Tests for Go parser hole detection."""

    @pytest.fixture
    def parser(self):
        return GoIncrementalParser()

    def test_detect_unclosed_paren(self, parser):
        """Should detect unclosed parenthesis."""
        parser.parse_initial("func foo(x int")
        holes = parser.find_holes()
        assert any(h[0] == HoleKind.EXPRESSION for h in holes)

    def test_detect_unclosed_brace(self, parser):
        """Should detect unclosed brace."""
        parser.parse_initial("func foo() {")
        holes = parser.find_holes()
        assert any(h[0] == HoleKind.BODY for h in holes)

    def test_detect_unclosed_bracket(self, parser):
        """Should detect unclosed bracket."""
        parser.parse_initial("var x = arr[")
        holes = parser.find_holes()
        assert any(h[0] == HoleKind.EXPRESSION for h in holes)

    def test_detect_trailing_operator(self, parser):
        """Should detect trailing operator."""
        parser.parse_initial("x = y +")
        holes = parser.find_holes()
        assert any(h[0] == HoleKind.EXPRESSION for h in holes)

    def test_detect_incomplete_var_decl(self, parser):
        """Should detect incomplete var declaration."""
        parser.parse_initial("var x")
        holes = parser.find_holes()
        assert any(h[0] == HoleKind.TYPE for h in holes)

    def test_detect_incomplete_short_decl(self, parser):
        """Should detect incomplete short declaration."""
        parser.parse_initial("x :=")
        holes = parser.find_holes()
        assert any(h[0] == HoleKind.EXPRESSION for h in holes)

    def test_no_holes_in_complete_code(self, parser):
        """Should find no holes in complete code."""
        parser.parse_initial("package main\n\nfunc main() {\n    x := 1\n}")
        holes = parser.find_holes()
        assert len(holes) == 0


class TestGoParserExpectedTokens:
    """Tests for Go parser expected token suggestions."""

    @pytest.fixture
    def parser(self):
        return GoIncrementalParser()

    def test_expect_after_package(self, parser):
        """Should expect identifier after package."""
        parser.parse_initial("package")
        expected = parser.get_expected_tokens()
        assert "identifier" in expected

    def test_expect_after_import(self, parser):
        """Should expect string or paren after import."""
        parser.parse_initial("import")
        expected = parser.get_expected_tokens()
        assert "(" in expected or "string" in expected

    def test_expect_after_func(self, parser):
        """Should expect identifier or paren after func."""
        parser.parse_initial("func")
        expected = parser.get_expected_tokens()
        assert "identifier" in expected or "(" in expected

    def test_expect_in_type_context(self, parser):
        """Should expect type tokens in type context."""
        parser.parse_initial("var x:")
        parser._update_type_context()
        if parser._state.in_type_context:
            expected = parser.get_expected_tokens()
            assert "identifier" in expected or "*" in expected

    def test_expect_after_opening_brace(self, parser):
        """Should expect statements after opening brace."""
        parser.parse_initial("func foo() {")
        expected = parser.get_expected_tokens()
        assert "}" in expected
        # Should also suggest statements
        assert any(s in expected for s in GO_STATEMENT_STARTERS)

    def test_expect_after_operator(self, parser):
        """Should expect expression after operator."""
        parser.parse_initial("x = y +")
        expected = parser.get_expected_tokens()
        assert "identifier" in expected


class TestGoParserCheckpoint:
    """Tests for Go parser checkpoint/restore."""

    @pytest.fixture
    def parser(self):
        return GoIncrementalParser()

    def test_checkpoint_creation(self, parser):
        """Should create checkpoint."""
        parser.parse_initial("package main")
        checkpoint = parser.checkpoint()
        assert isinstance(checkpoint, GoParserCheckpoint)
        assert checkpoint.source == "package main"

    def test_restore_checkpoint(self, parser):
        """Should restore from checkpoint."""
        parser.parse_initial("package main\n")
        checkpoint = parser.checkpoint()

        # Extend with more code
        parser.extend_with_text("import \"fmt\"")
        assert "import" in parser.current_source

        # Restore
        parser.restore(checkpoint)
        assert parser.current_source == "package main\n"
        assert "import" not in parser.current_source

    def test_checkpoint_preserves_bracket_stack(self, parser):
        """Should preserve bracket stack in checkpoint."""
        parser.parse_initial("func foo(x int) {")
        checkpoint = parser.checkpoint()
        assert len(checkpoint.bracket_stack) == 1
        assert checkpoint.bracket_stack[0] == "{"


class TestGoParserCopy:
    """Tests for Go parser copying."""

    @pytest.fixture
    def parser(self):
        return GoIncrementalParser()

    def test_copy_parser(self, parser):
        """Should copy parser state."""
        parser.parse_initial("package main")
        copy = parser.copy()

        assert copy.current_source == parser.current_source
        assert copy is not parser

    def test_copy_is_independent(self, parser):
        """Copied parser should be independent."""
        parser.parse_initial("package main\n")
        copy = parser.copy()

        # Modify original
        parser.extend_with_text("import \"fmt\"")

        # Copy should be unchanged
        assert "import" not in copy.current_source


class TestGoParserContextTracking:
    """Tests for Go parser context tracking."""

    @pytest.fixture
    def parser(self):
        return GoIncrementalParser()

    def test_track_type_context_after_colon(self, parser):
        """Should detect type context after colon."""
        parser.parse_initial("var x:")
        parser._update_type_context()
        assert parser._state.in_type_context

    def test_track_type_context_in_slice(self, parser):
        """Should detect type context in slice type."""
        parser.parse_initial("var x []")
        parser._update_type_context()
        # May or may not be in type context depending on implementation

    def test_track_type_context_in_map(self, parser):
        """Should detect type context in map type."""
        parser.parse_initial("var x map[")
        parser._update_type_context()
        assert parser._state.in_type_context

    def test_track_type_context_after_chan(self, parser):
        """Should detect type context after chan."""
        parser.parse_initial("var x chan ")
        parser._update_type_context()
        assert parser._state.in_type_context


class TestGoSyntaxElements:
    """Tests for Go syntax element constants."""

    def test_expression_starters(self):
        """Should have expression starters."""
        assert "true" in GO_EXPRESSION_STARTERS
        assert "false" in GO_EXPRESSION_STARTERS
        assert "nil" in GO_EXPRESSION_STARTERS
        assert "(" in GO_EXPRESSION_STARTERS
        assert "func" in GO_EXPRESSION_STARTERS

    def test_statement_starters(self):
        """Should have statement starters."""
        assert "if" in GO_STATEMENT_STARTERS
        assert "for" in GO_STATEMENT_STARTERS
        assert "switch" in GO_STATEMENT_STARTERS
        assert "return" in GO_STATEMENT_STARTERS
        assert "go" in GO_STATEMENT_STARTERS
        assert "defer" in GO_STATEMENT_STARTERS

    def test_declaration_starters(self):
        """Should have declaration starters."""
        assert "package" in GO_DECLARATION_STARTERS
        assert "import" in GO_DECLARATION_STARTERS
        assert "func" in GO_DECLARATION_STARTERS
        assert "type" in GO_DECLARATION_STARTERS
        assert "var" in GO_DECLARATION_STARTERS
        assert "const" in GO_DECLARATION_STARTERS

    def test_binary_operators(self):
        """Should have binary operators."""
        # Arithmetic
        assert "+" in GO_BINARY_OPERATORS
        assert "-" in GO_BINARY_OPERATORS
        assert "*" in GO_BINARY_OPERATORS
        assert "/" in GO_BINARY_OPERATORS
        # Comparison
        assert "==" in GO_BINARY_OPERATORS
        assert "!=" in GO_BINARY_OPERATORS
        assert "<" in GO_BINARY_OPERATORS
        assert ">" in GO_BINARY_OPERATORS
        # Assignment
        assert "=" in GO_BINARY_OPERATORS
        assert ":=" in GO_BINARY_OPERATORS
        assert "+=" in GO_BINARY_OPERATORS
        # Logical
        assert "&&" in GO_BINARY_OPERATORS
        assert "||" in GO_BINARY_OPERATORS
        # Bitwise
        assert "&" in GO_BINARY_OPERATORS
        assert "|" in GO_BINARY_OPERATORS
        assert "^" in GO_BINARY_OPERATORS
        assert "<<" in GO_BINARY_OPERATORS
        assert ">>" in GO_BINARY_OPERATORS


class TestGoParserIncompleteConstructs:
    """Tests for Go parser incomplete construct detection."""

    @pytest.fixture
    def parser(self):
        return GoIncrementalParser()

    def test_detect_trailing_func(self, parser):
        """Should detect trailing func keyword."""
        parser.parse_initial("func")
        assert parser._has_incomplete_construct()

    def test_detect_trailing_var(self, parser):
        """Should detect trailing var keyword."""
        parser.parse_initial("var")
        assert parser._has_incomplete_construct()

    def test_detect_trailing_if(self, parser):
        """Should detect trailing if keyword."""
        parser.parse_initial("if")
        assert parser._has_incomplete_construct()

    def test_detect_trailing_for(self, parser):
        """Should detect trailing for keyword."""
        parser.parse_initial("for")
        assert parser._has_incomplete_construct()

    def test_detect_trailing_plus(self, parser):
        """Should detect trailing plus operator."""
        parser.parse_initial("x +")
        assert parser._has_incomplete_construct()

    def test_detect_trailing_assignment(self, parser):
        """Should detect trailing assignment."""
        parser.parse_initial("x =")
        assert parser._has_incomplete_construct()

    def test_complete_construct(self, parser):
        """Should not flag complete constructs."""
        parser.parse_initial("x := 1")
        assert not parser._has_incomplete_construct()


class TestGoParserRealWorldExamples:
    """Tests for Go parser with real-world code patterns."""

    @pytest.fixture
    def parser(self):
        return GoIncrementalParser()

    def test_parse_hello_world(self, parser):
        """Should parse hello world program."""
        source = '''package main

import "fmt"

func main() {
    fmt.Println("Hello, World!")
}'''
        result = parser.parse_initial(source)
        assert result.is_valid

    def test_parse_http_server(self, parser):
        """Should parse HTTP server snippet."""
        source = '''package main

import (
    "fmt"
    "net/http"
)

func handler(w http.ResponseWriter, r *http.Request) {
    fmt.Fprintf(w, "Hello")
}'''
        result = parser.parse_initial(source)
        assert result.is_valid

    def test_parse_struct_definition(self, parser):
        """Should parse struct definition."""
        source = '''type User struct {
    ID   int
    Name string
    Age  int
}'''
        result = parser.parse_initial(source)
        assert result.is_valid

    def test_parse_interface(self, parser):
        """Should parse interface definition."""
        source = '''type Reader interface {
    Read(p []byte) (n int, err error)
}'''
        result = parser.parse_initial(source)
        assert result.is_valid

    def test_parse_goroutine(self, parser):
        """Should parse goroutine."""
        source = '''func main() {
    go func() {
        fmt.Println("async")
    }()
}'''
        result = parser.parse_initial(source)
        assert result.is_valid

    def test_parse_channel_operations(self, parser):
        """Should parse channel operations."""
        source = '''func main() {
    ch := make(chan int)
    ch <- 1
    x := <-ch
}'''
        result = parser.parse_initial(source)
        assert result.is_valid
