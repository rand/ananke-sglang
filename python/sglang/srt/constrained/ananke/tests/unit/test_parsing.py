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
"""Tests for the parsing module.

Tests incremental parsing, partial AST construction, and hole detection.
"""

import pytest

from parsing.base import (
    IncrementalParser,
    ParseState,
    ParseResult,
    ParseError,
    TokenInfo,
    HoleDetector,
    SourceTracker,
)
from parsing.partial_ast import (
    PartialAST,
    HoleInfo,
    HoleKind,
    ASTDiff,
    PartialASTBuilder,
)
from parsing.languages.python import (
    PythonIncrementalParser,
    create_python_parser,
)
from parsing import get_parser

from domains.types.constraint import INT, STR, BOOL, FLOAT, ANY
from domains.types.marking.provenance import UNKNOWN_SPAN
from domains.types.marking.marked_ast import (
    MarkedASTNode,
    ASTNodeKind,
    create_hole_node,
    create_literal_node,
)


class TestSourceTracker:
    """Tests for SourceTracker."""

    def test_empty_tracker(self):
        """Empty tracker has correct initial state."""
        tracker = SourceTracker()
        assert tracker.source == ""
        assert tracker.position == 0

    def test_append_text(self):
        """Appending text updates source and returns span."""
        tracker = SourceTracker()
        span = tracker.append("hello")

        assert tracker.source == "hello"
        assert tracker.position == 5
        assert span.start == 0
        assert span.end == 5

    def test_append_multiline(self):
        """Multiline text tracks line numbers."""
        tracker = SourceTracker()
        span1 = tracker.append("line1\n")
        span2 = tracker.append("line2")

        assert span1.start_line == 1
        assert span2.start_line == 2

    def test_append_token(self):
        """Token appending tracks token info."""
        tracker = SourceTracker()
        token = TokenInfo(token_id=42, text="def", position=0, length=3)
        span = tracker.append_token(token)

        assert tracker.source == "def"
        assert span.start == 0
        assert span.end == 3

    def test_checkpoint_restore(self):
        """Checkpoint and restore preserves state."""
        tracker = SourceTracker()
        tracker.append("hello")
        cp = tracker.checkpoint()

        tracker.append(" world")
        assert tracker.source == "hello world"

        tracker.restore(cp)
        assert tracker.source == "hello"

    def test_span_at(self):
        """span_at creates correct spans."""
        tracker = SourceTracker()
        tracker.append("hello world")

        span = tracker.span_at(0, 5)
        assert span.start == 0
        assert span.end == 5


class TestHoleDetector:
    """Tests for HoleDetector."""

    def test_find_holes_empty(self):
        """No holes in literal node."""
        node = create_literal_node(value=42, ty=INT, span=UNKNOWN_SPAN)
        holes = HoleDetector.find_holes_in_ast(node)
        assert holes == []

    def test_find_single_hole(self):
        """Find single hole."""
        node = create_hole_node(hole_id="h0", span=UNKNOWN_SPAN)
        holes = HoleDetector.find_holes_in_ast(node)

        assert len(holes) == 1
        assert holes[0][0] == "h0"

    def test_find_nested_holes(self):
        """Find holes in nested structure."""
        hole1 = create_hole_node(hole_id="h1", span=UNKNOWN_SPAN)
        hole2 = create_hole_node(hole_id="h2", span=UNKNOWN_SPAN)
        parent = MarkedASTNode(
            kind=ASTNodeKind.LIST,
            span=UNKNOWN_SPAN,
            children=[hole1, hole2],
        )

        holes = HoleDetector.find_holes_in_ast(parent)
        assert len(holes) == 2
        hole_ids = {h[0] for h in holes}
        assert hole_ids == {"h1", "h2"}

    def test_find_first_hole(self):
        """find_first_hole returns first hole."""
        hole = create_hole_node(hole_id="first", span=UNKNOWN_SPAN)
        result = HoleDetector.find_first_hole(hole)

        assert result is not None
        assert result[0] == "first"

    def test_find_first_hole_empty(self):
        """find_first_hole returns None when no holes."""
        node = create_literal_node(value=42, ty=INT, span=UNKNOWN_SPAN)
        result = HoleDetector.find_first_hole(node)
        assert result is None

    def test_count_holes(self):
        """count_holes returns correct count."""
        hole1 = create_hole_node(hole_id="h1", span=UNKNOWN_SPAN)
        hole2 = create_hole_node(hole_id="h2", span=UNKNOWN_SPAN)
        literal = create_literal_node(value=1, ty=INT, span=UNKNOWN_SPAN)
        parent = MarkedASTNode(
            kind=ASTNodeKind.LIST,
            span=UNKNOWN_SPAN,
            children=[hole1, literal, hole2],
        )

        assert HoleDetector.count_holes(parent) == 2


class TestPartialAST:
    """Tests for PartialAST."""

    def test_empty_partial_ast(self):
        """Empty partial AST has single hole."""
        ast = PartialAST.empty()

        assert not ast.is_complete
        assert ast.hole_count == 1
        assert ast.root.kind == ASTNodeKind.HOLE

    def test_from_node(self):
        """Create PartialAST from node."""
        node = create_literal_node(value=42, ty=INT, span=UNKNOWN_SPAN)
        ast = PartialAST.from_node(node)

        assert ast.is_complete  # No holes
        assert ast.root == node

    def test_from_node_with_holes(self):
        """Create PartialAST from node with holes."""
        hole = create_hole_node(hole_id="h0", span=UNKNOWN_SPAN)
        ast = PartialAST.from_node(hole)

        assert not ast.is_complete
        assert "h0" in ast.holes

    def test_first_unfilled_hole(self):
        """first_unfilled_hole returns first hole."""
        ast = PartialAST.empty()
        hole = ast.first_unfilled_hole()

        assert hole is not None
        assert hole.hole_id == "root"

    def test_add_hole(self):
        """Adding holes updates hole dict."""
        node = create_literal_node(value=42, ty=INT, span=UNKNOWN_SPAN)
        ast = PartialAST.from_node(node)

        info = HoleInfo(
            hole_id="new_hole",
            kind=HoleKind.EXPRESSION,
            span=UNKNOWN_SPAN,
        )
        ast.add_hole(info)

        assert "new_hole" in ast.holes
        assert not ast.is_complete

    def test_copy(self):
        """copy creates independent copy."""
        ast1 = PartialAST.empty()
        ast2 = ast1.copy()

        # Verify independence
        info = HoleInfo(
            hole_id="extra",
            kind=HoleKind.EXPRESSION,
            span=UNKNOWN_SPAN,
        )
        ast2.add_hole(info)

        assert "extra" in ast2.holes
        assert "extra" not in ast1.holes


class TestPartialASTBuilder:
    """Tests for PartialASTBuilder."""

    def test_build_literal(self):
        """Build literal node."""
        builder = PartialASTBuilder()
        ast = builder.literal(42, INT).build()

        assert ast.is_complete
        assert ast.root.kind == ASTNodeKind.LITERAL
        assert ast.root.data.get("value") == 42

    def test_build_variable(self):
        """Build variable node."""
        builder = PartialASTBuilder()
        ast = builder.variable("x", INT).build()

        assert ast.is_complete
        assert ast.root.kind == ASTNodeKind.VARIABLE
        assert ast.root.data.get("name") == "x"

    def test_build_hole(self):
        """Build hole node."""
        builder = PartialASTBuilder()
        ast = builder.hole(expected_type=INT).build()

        assert not ast.is_complete
        assert ast.root.kind == ASTNodeKind.HOLE
        assert len(ast.holes) == 1

    def test_build_list(self):
        """Build list node."""
        builder = PartialASTBuilder()
        ast = (
            builder
            .literal(1, INT)
            .literal(2, INT)
            .list_node(2)
            .build()
        )

        assert ast.is_complete
        assert ast.root.kind == ASTNodeKind.LIST
        assert len(ast.root.children) == 2

    def test_build_tuple(self):
        """Build tuple node."""
        builder = PartialASTBuilder()
        ast = (
            builder
            .literal("a", STR)
            .literal("b", STR)
            .tuple_node(2)
            .build()
        )

        assert ast.is_complete
        assert ast.root.kind == ASTNodeKind.TUPLE

    def test_reset(self):
        """Reset clears builder state."""
        builder = PartialASTBuilder()
        builder.literal(1, INT)
        builder.reset()

        # Should raise since stack is empty
        with pytest.raises(ValueError):
            builder.build()


class TestASTDiff:
    """Tests for ASTDiff."""

    def test_compute_no_changes(self):
        """No changes between identical ASTs."""
        node = create_literal_node(value=42, ty=INT, span=UNKNOWN_SPAN)
        ast1 = PartialAST.from_node(node)
        ast2 = PartialAST.from_node(node)

        diff = ASTDiff.compute(ast1, ast2)
        # Both have no holes, so no hole changes
        assert diff.added_holes == []
        assert diff.removed_holes == []

    def test_compute_added_hole(self):
        """Detect added holes."""
        node = create_literal_node(value=42, ty=INT, span=UNKNOWN_SPAN)
        ast1 = PartialAST.from_node(node)
        ast2 = PartialAST.from_node(node)
        ast2.add_hole(HoleInfo("new", HoleKind.EXPRESSION, UNKNOWN_SPAN))

        diff = ASTDiff.compute(ast1, ast2)
        assert "new" in diff.added_holes

    def test_compute_removed_hole(self):
        """Detect removed holes."""
        ast1 = PartialAST.empty()  # Has "root" hole
        node = create_literal_node(value=42, ty=INT, span=UNKNOWN_SPAN)
        ast2 = PartialAST.from_node(node)  # No holes

        diff = ASTDiff.compute(ast1, ast2)
        assert "root" in diff.removed_holes


class TestPythonIncrementalParser:
    """Tests for PythonIncrementalParser."""

    def test_create_parser(self):
        """Create parser instance."""
        parser = PythonIncrementalParser()
        assert parser.language == "python"
        assert parser.current_source == ""

    def test_parse_initial_empty(self):
        """Parse empty source."""
        parser = PythonIncrementalParser()
        result = parser.parse_initial("")

        assert result.state == ParseState.PARTIAL
        assert len(result.holes) > 0  # Should have a hole

    def test_parse_initial_literal(self):
        """Parse literal expression."""
        parser = PythonIncrementalParser()
        result = parser.parse_initial("42")

        assert result.is_valid
        assert result.ast is not None

    def test_parse_initial_function(self):
        """Parse incomplete function definition."""
        parser = PythonIncrementalParser()
        result = parser.parse_initial("def foo(x: int)")

        # Missing body, so should be partial
        assert result.state == ParseState.PARTIAL

    def test_extend_with_text(self):
        """Extend parse with text."""
        parser = PythonIncrementalParser()
        parser.parse_initial("x = ")
        result = parser.extend_with_text("42")

        assert result.is_valid
        assert "42" in parser.current_source

    def test_extend_with_token(self):
        """Extend parse with token."""
        parser = PythonIncrementalParser()
        parser.parse_initial("x = ")
        token = TokenInfo(token_id=1, text="1", position=4, length=1)
        result = parser.extend_with_token(token)

        assert result.is_valid
        assert parser.current_source == "x = 1"

    def test_find_holes_incomplete(self):
        """Find holes in incomplete code."""
        parser = PythonIncrementalParser()
        parser.parse_initial("x + ")  # Missing RHS

        holes = parser.find_holes()
        assert len(holes) > 0

    def test_checkpoint_restore(self):
        """Checkpoint and restore works."""
        parser = PythonIncrementalParser()
        parser.parse_initial("x = 1")
        cp = parser.checkpoint()

        parser.extend_with_text(" + 2")
        assert "2" in parser.current_source

        parser.restore(cp)
        assert parser.current_source == "x = 1"

    def test_copy(self):
        """Copy creates independent parser."""
        parser1 = PythonIncrementalParser()
        parser1.parse_initial("x = 1")
        parser2 = parser1.copy()

        parser2.extend_with_text(" + 2")

        assert parser1.current_source == "x = 1"
        assert parser2.current_source == "x = 1 + 2"

    def test_get_ast(self):
        """get_ast returns MarkedASTNode."""
        parser = PythonIncrementalParser()
        parser.parse_initial("42")

        ast = parser.get_ast()
        assert ast is not None
        assert isinstance(ast, MarkedASTNode)

    def test_expected_tokens_empty(self):
        """Expected tokens for empty source."""
        parser = PythonIncrementalParser()
        parser.parse_initial("")

        expected = parser.get_expected_tokens()
        assert len(expected) > 0

    def test_expected_tokens_after_operator(self):
        """Expected tokens after operator."""
        parser = PythonIncrementalParser()
        parser.parse_initial("x +")

        expected = parser.get_expected_tokens()
        assert "identifier" in expected or len(expected) > 0


class TestGetParser:
    """Tests for get_parser factory function."""

    def test_get_python_parser(self):
        """Get Python parser."""
        parser = get_parser("python")
        assert parser.language == "python"

    def test_get_python_parser_alias(self):
        """Get Python parser via alias."""
        parser = get_parser("py")
        assert parser.language == "python"

    def test_unsupported_language(self):
        """Unsupported language raises ValueError."""
        with pytest.raises(ValueError) as exc:
            get_parser("brainfuck")
        assert "Unsupported" in str(exc.value)


class TestCreatePythonParser:
    """Tests for create_python_parser factory."""

    def test_creates_parser(self):
        """Factory creates parser instance."""
        parser = create_python_parser()
        assert isinstance(parser, PythonIncrementalParser)
        assert parser.language == "python"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
