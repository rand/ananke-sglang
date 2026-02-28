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
"""Unit tests for marked lambda calculus implementation (POPL 2024).

Tests marks, provenance, marked AST, and totalization.
"""

import pytest

from domains.types.constraint import INT, STR, BOOL, ANY, FunctionType
from domains.types.marking.provenance import (
    SourceSpan,
    Provenance,
    UNKNOWN_SPAN,
    CONTEXT_FUNCTION_ARGUMENT,
    CONTEXT_FUNCTION_RETURN,
    CONTEXT_VARIABLE_BINDING,
    CONTEXT_IF_CONDITION,
    CONTEXT_EXPRESSION,
    create_provenance,
)
from domains.types.marking.marks import (
    Mark,
    HoleMark,
    InconsistentMark,
    NonEmptyHoleMark,
    is_marked_with_error,
    is_marked_with_hole,
    create_hole_mark,
    create_inconsistent_mark,
)
from domains.types.marking.marked_ast import (
    MarkedASTNode,
    MarkedAST,
    ASTNodeKind,
)
from domains.types.marking.totalization import (
    totalize,
    TotalizationResult,
)
from domains.types.environment import EMPTY_ENVIRONMENT


class TestSourceSpan:
    """Tests for SourceSpan class."""

    def test_create_span(self):
        """Should create span with start and end."""
        span = SourceSpan(start=10, end=20)
        assert span.start == 10
        assert span.end == 20
        assert span.length == 10

    def test_span_with_file(self):
        """Should create span with file info."""
        span = SourceSpan(start=0, end=100, file="test.py")
        assert span.file == "test.py"

    def test_span_with_line_col(self):
        """Should create span with line/column info."""
        span = SourceSpan(
            start=0, end=10,
            start_line=1, start_col=1,
            end_line=1, end_col=11,
        )
        assert span.start_line == 1
        assert span.start_col == 1

    def test_contains(self):
        """Should check if offset is in span."""
        span = SourceSpan(start=10, end=20)
        assert span.contains(10)  # Start inclusive
        assert span.contains(15)
        assert not span.contains(20)  # End exclusive
        assert not span.contains(5)

    def test_overlaps(self):
        """Should check if spans overlap."""
        span1 = SourceSpan(start=10, end=20)
        span2 = SourceSpan(start=15, end=25)
        span3 = SourceSpan(start=20, end=30)
        span4 = SourceSpan(start=0, end=10)

        assert span1.overlaps(span2)
        assert span2.overlaps(span1)
        assert not span1.overlaps(span3)  # Adjacent, not overlapping
        assert not span1.overlaps(span4)

    def test_merge(self):
        """Should merge two spans."""
        span1 = SourceSpan(start=10, end=20)
        span2 = SourceSpan(start=15, end=25)
        merged = span1.merge(span2)
        assert merged.start == 10
        assert merged.end == 25

    def test_repr_with_file(self):
        """repr should include file info."""
        span = SourceSpan(
            start=0, end=10, file="test.py",
            start_line=5, start_col=3,
        )
        repr_str = repr(span)
        assert "test.py" in repr_str
        assert "5" in repr_str
        assert "3" in repr_str

    def test_repr_without_file(self):
        """repr should show byte range without file."""
        span = SourceSpan(start=10, end=20)
        repr_str = repr(span)
        assert "10" in repr_str
        assert "20" in repr_str


class TestProvenance:
    """Tests for Provenance class."""

    def test_create_provenance(self):
        """Should create provenance with location and context."""
        span = SourceSpan(start=0, end=10)
        prov = Provenance(location=span, context="test context")
        assert prov.location == span
        assert prov.context == "test context"
        assert prov.parent is None

    def test_chain_provenance(self):
        """Should chain provenances."""
        span1 = SourceSpan(start=0, end=10)
        span2 = SourceSpan(start=5, end=8)
        parent = Provenance(location=span1, context="outer")
        child = parent.chain(span2, "inner")
        assert child.parent == parent
        assert child.context == "inner"

    def test_with_message(self):
        """Should add message to provenance."""
        span = SourceSpan(start=0, end=10)
        prov = Provenance(location=span, context="test")
        with_msg = prov.with_message("error message")
        assert with_msg.message == "error message"
        assert with_msg.context == "test"

    def test_root(self):
        """Should get root provenance."""
        span = SourceSpan(start=0, end=10)
        root = Provenance(location=span, context="root")
        child = root.chain(span, "child")
        grandchild = child.chain(span, "grandchild")
        assert grandchild.root() == root

    def test_chain_length(self):
        """Should count chain length."""
        span = SourceSpan(start=0, end=10)
        root = Provenance(location=span, context="root")
        assert root.chain_length() == 1
        child = root.chain(span, "child")
        assert child.chain_length() == 2

    def test_full_context(self):
        """Should get full context string."""
        span = SourceSpan(start=0, end=10)
        root = Provenance(location=span, context="function")
        child = root.chain(span, "argument 1")
        full = child.full_context()
        assert "function" in full
        assert "argument 1" in full

    def test_all_locations(self):
        """Should get all locations in chain."""
        span1 = SourceSpan(start=0, end=10)
        span2 = SourceSpan(start=5, end=8)
        root = Provenance(location=span1, context="outer")
        child = root.chain(span2, "inner")
        locations = child.all_locations()
        assert len(locations) == 2
        assert span1 in locations
        assert span2 in locations

    def test_create_provenance_helper(self):
        """create_provenance helper should work."""
        prov = create_provenance(0, 10, "test", file="test.py")
        assert prov.location.start == 0
        assert prov.location.end == 10
        assert prov.context == "test"
        assert prov.location.file == "test.py"


class TestHoleMark:
    """Tests for HoleMark class."""

    def test_create_hole_mark(self):
        """Should create hole mark."""
        mark = HoleMark(hole_id="h1")
        assert mark.hole_id == "h1"
        assert mark.expected_type is None
        assert not mark.is_error()
        assert mark.is_hole()

    def test_hole_mark_with_expected(self):
        """Should create hole mark with expected type."""
        mark = HoleMark(hole_id="h1", expected_type=INT)
        assert mark.expected_type == INT

    def test_hole_mark_with_provenance(self):
        """Should create hole mark with provenance."""
        prov = create_provenance(0, 10, "test")
        mark = HoleMark(hole_id="h1", provenance=prov)
        assert mark.provenance == prov

    def test_synthesized_type_none(self):
        """Hole should have no synthesized type."""
        mark = HoleMark(hole_id="h1")
        assert mark.synthesized_type() is None

    def test_with_expected_type(self):
        """Should create copy with updated expected type."""
        mark = HoleMark(hole_id="h1", expected_type=INT)
        updated = mark.with_expected_type(STR)
        assert updated.expected_type == STR
        assert updated.hole_id == "h1"
        assert mark.expected_type == INT  # Original unchanged

    def test_create_hole_mark_helper(self):
        """create_hole_mark helper should work."""
        mark = create_hole_mark("h1", INT)
        assert mark.hole_id == "h1"
        assert mark.expected_type == INT


class TestInconsistentMark:
    """Tests for InconsistentMark class."""

    def test_create_inconsistent_mark(self):
        """Should create inconsistent mark."""
        prov = create_provenance(0, 10, "test")
        mark = InconsistentMark(
            synthesized=INT,
            expected=STR,
            provenance=prov,
        )
        assert mark.synthesized == INT
        assert mark.expected == STR
        assert mark.is_error()
        assert not mark.is_hole()

    def test_synthesized_type(self):
        """Should return synthesized type."""
        prov = create_provenance(0, 10, "test")
        mark = InconsistentMark(synthesized=INT, expected=STR, provenance=prov)
        assert mark.synthesized_type() == INT

    def test_error_message(self):
        """Should generate error message."""
        prov = create_provenance(0, 10, "expression")
        mark = InconsistentMark(synthesized=INT, expected=STR, provenance=prov)
        msg = mark.error_message()
        assert "INT" in msg or "int" in msg.lower()
        assert "STR" in msg or "str" in msg.lower()
        assert "expression" in msg

    def test_create_inconsistent_mark_helper(self):
        """create_inconsistent_mark helper should work."""
        prov = create_provenance(0, 10, "test")
        mark = create_inconsistent_mark(INT, STR, prov)
        assert mark.synthesized == INT
        assert mark.expected == STR


class TestNonEmptyHoleMark:
    """Tests for NonEmptyHoleMark class."""

    def test_create_non_empty_hole(self):
        """Should create non-empty hole mark."""
        mark = NonEmptyHoleMark(hole_id="h1", inner_type=INT)
        assert mark.hole_id == "h1"
        assert mark.inner_type == INT
        assert not mark.is_error()
        assert mark.is_hole()

    def test_synthesized_type(self):
        """Should return inner type."""
        mark = NonEmptyHoleMark(hole_id="h1", inner_type=INT)
        assert mark.synthesized_type() == INT

    def test_with_expected_type(self):
        """Should create with expected type."""
        mark = NonEmptyHoleMark(
            hole_id="h1",
            inner_type=INT,
            expected_type=STR,
        )
        assert mark.expected_type == STR


class TestMarkHelpers:
    """Tests for mark helper functions."""

    def test_is_marked_with_error_true(self):
        """Should detect error marks."""
        prov = create_provenance(0, 10, "test")
        mark = InconsistentMark(synthesized=INT, expected=STR, provenance=prov)
        assert is_marked_with_error(mark)

    def test_is_marked_with_error_false_hole(self):
        """Hole should not be error."""
        mark = HoleMark(hole_id="h1")
        assert not is_marked_with_error(mark)

    def test_is_marked_with_error_none(self):
        """None should not be error."""
        assert not is_marked_with_error(None)

    def test_is_marked_with_hole_true(self):
        """Should detect hole marks."""
        mark = HoleMark(hole_id="h1")
        assert is_marked_with_hole(mark)

    def test_is_marked_with_hole_non_empty(self):
        """Non-empty hole should also be hole."""
        mark = NonEmptyHoleMark(hole_id="h1", inner_type=INT)
        assert is_marked_with_hole(mark)

    def test_is_marked_with_hole_false(self):
        """Error should not be hole."""
        prov = create_provenance(0, 10, "test")
        mark = InconsistentMark(synthesized=INT, expected=STR, provenance=prov)
        assert not is_marked_with_hole(mark)


class TestMarkedASTNode:
    """Tests for MarkedASTNode class."""

    def test_create_node(self):
        """Should create AST node."""
        node = MarkedASTNode(
            kind=ASTNodeKind.LITERAL,
            span=SourceSpan(0, 10),
            data={"value": 42},
        )
        assert node.kind == ASTNodeKind.LITERAL
        assert node.data["value"] == 42

    def test_node_with_type(self):
        """Should set synthesized type."""
        node = MarkedASTNode(kind=ASTNodeKind.LITERAL, span=SourceSpan(0, 10))
        typed_node = node.with_type(INT)
        assert typed_node.synthesized_type == INT
        assert node.synthesized_type is None  # Original unchanged

    def test_node_with_mark(self):
        """Should set mark."""
        node = MarkedASTNode(kind=ASTNodeKind.HOLE, span=SourceSpan(0, 10))
        mark = HoleMark(hole_id="h1", expected_type=INT)
        marked_node = node.with_mark(mark)
        assert marked_node.mark == mark

    def test_node_with_children(self):
        """Should set children."""
        parent = MarkedASTNode(kind=ASTNodeKind.LIST, span=SourceSpan(0, 20))
        child1 = MarkedASTNode(kind=ASTNodeKind.LITERAL, span=SourceSpan(1, 5))
        child2 = MarkedASTNode(kind=ASTNodeKind.LITERAL, span=SourceSpan(7, 11))
        with_children = parent.with_children([child1, child2])
        assert len(with_children.children) == 2

    def test_get_type(self):
        """Should get synthesized type."""
        node = MarkedASTNode(kind=ASTNodeKind.LITERAL, span=SourceSpan(0, 10))
        node = node.with_type(INT)
        assert node.get_type() == INT

    def test_is_hole(self):
        """Should detect hole nodes."""
        node = MarkedASTNode(kind=ASTNodeKind.HOLE, span=SourceSpan(0, 10))
        assert node.is_hole()
        literal = MarkedASTNode(kind=ASTNodeKind.LITERAL, span=SourceSpan(0, 10))
        assert not literal.is_hole()


class TestMarkedAST:
    """Tests for MarkedAST class."""

    def test_create_ast(self):
        """Should create marked AST."""
        root = MarkedASTNode(kind=ASTNodeKind.LIST, span=SourceSpan(0, 20))
        ast = MarkedAST(root=root)
        assert ast.root == root

    def test_collect_errors_empty(self):
        """Should return empty list when no errors."""
        root = MarkedASTNode(kind=ASTNodeKind.LITERAL, span=SourceSpan(0, 10))
        root = root.with_type(INT)
        ast = MarkedAST(root=root)
        errors = ast.collect_errors()
        assert errors == []

    def test_collect_errors_with_error(self):
        """Should collect error marks."""
        prov = create_provenance(0, 10, "test")
        mark = InconsistentMark(synthesized=INT, expected=STR, provenance=prov)
        root = MarkedASTNode(kind=ASTNodeKind.LITERAL, span=SourceSpan(0, 10))
        root = root.with_mark(mark)
        ast = MarkedAST(root=root)
        errors = ast.collect_errors()
        assert len(errors) == 1
        assert errors[0][0] == mark

    def test_collect_errors_nested(self):
        """Should collect errors from nested nodes."""
        prov = create_provenance(0, 10, "test")
        mark = InconsistentMark(synthesized=INT, expected=STR, provenance=prov)
        child = MarkedASTNode(kind=ASTNodeKind.LITERAL, span=SourceSpan(1, 5))
        child = child.with_mark(mark)
        root = MarkedASTNode(kind=ASTNodeKind.LIST, span=SourceSpan(0, 20))
        root = root.with_children([child])
        ast = MarkedAST(root=root)
        errors = ast.collect_errors()
        assert len(errors) == 1

    def test_find_holes(self):
        """Should find all holes in AST."""
        hole_mark = HoleMark(hole_id="h1", expected_type=INT)
        hole = MarkedASTNode(kind=ASTNodeKind.HOLE, span=SourceSpan(1, 5))
        hole = hole.with_mark(hole_mark)
        root = MarkedASTNode(kind=ASTNodeKind.LIST, span=SourceSpan(0, 20))
        root = root.with_children([hole])
        ast = MarkedAST(root=root)
        holes = ast.all_holes()
        assert len(holes) == 1
        assert holes[0].mark.hole_id == "h1"

    def test_has_errors(self):
        """Should detect if AST has errors."""
        prov = create_provenance(0, 10, "test")
        mark = InconsistentMark(synthesized=INT, expected=STR, provenance=prov)
        root = MarkedASTNode(kind=ASTNodeKind.LITERAL, span=SourceSpan(0, 10))
        root = root.with_mark(mark)
        ast = MarkedAST(root=root)
        assert ast.has_errors()

    def test_has_holes(self):
        """Should detect if AST has holes."""
        hole_mark = HoleMark(hole_id="h1")
        root = MarkedASTNode(kind=ASTNodeKind.HOLE, span=SourceSpan(0, 10))
        root = root.with_mark(hole_mark)
        ast = MarkedAST(root=root)
        assert ast.has_holes()


class TestTotalization:
    """Tests for totalization - assigning types to all partial programs."""

    def test_totalize_literal(self):
        """Should totalize literal node."""
        node = MarkedASTNode(
            kind=ASTNodeKind.LITERAL,
            span=SourceSpan(0, 5),
            data={"value": 42},
            synthesized_type=INT,
        )
        result = totalize(node, INT, EMPTY_ENVIRONMENT)
        assert result.ast is not None
        assert result.ast.root.synthesized_type == INT

    def test_totalize_hole(self):
        """Should totalize hole with expected type."""
        node = MarkedASTNode(
            kind=ASTNodeKind.HOLE,
            span=SourceSpan(0, 5),
        )
        result = totalize(node, INT, EMPTY_ENVIRONMENT)
        assert result.has_holes
        assert isinstance(result.ast.root.mark, HoleMark)
        assert result.ast.root.mark.expected_type == INT

    def test_totalize_variable(self):
        """Should totalize variable lookup."""
        env = EMPTY_ENVIRONMENT.bind("x", INT)
        node = MarkedASTNode(
            kind=ASTNodeKind.VARIABLE,
            span=SourceSpan(0, 1),
            data={"name": "x"},
            synthesized_type=INT,
        )
        result = totalize(node, INT, env)
        assert not result.has_errors

    def test_totalize_type_mismatch(self):
        """Should mark type mismatch."""
        node = MarkedASTNode(
            kind=ASTNodeKind.LITERAL,
            span=SourceSpan(0, 5),
            data={"value": 42},
            synthesized_type=INT,  # Synthesized as int
        )
        result = totalize(node, STR, EMPTY_ENVIRONMENT)  # Expecting string
        # Should have error (type mismatch)
        assert result.ast is not None
        assert result.has_errors

    def test_totalization_result_properties(self):
        """TotalizationResult should have expected properties."""
        node = MarkedASTNode(
            kind=ASTNodeKind.LITERAL,
            span=SourceSpan(0, 5),
            data={"value": 42},
            synthesized_type=INT,
        )
        result = totalize(node, INT, EMPTY_ENVIRONMENT)
        assert result.ast is not None
        assert isinstance(result.has_errors, bool)
        assert isinstance(result.has_holes, bool)


class TestASTNodeKind:
    """Tests for ASTNodeKind enum."""

    def test_all_kinds_exist(self):
        """Should have all expected node kinds."""
        kinds = [
            ASTNodeKind.LITERAL,
            ASTNodeKind.VARIABLE,
            ASTNodeKind.LAMBDA,
            ASTNodeKind.APPLICATION,
            ASTNodeKind.LET,
            ASTNodeKind.IF,
            ASTNodeKind.LIST,
            ASTNodeKind.TUPLE,
            ASTNodeKind.DICT,
            ASTNodeKind.HOLE,
            ASTNodeKind.BINARY_OP,
            ASTNodeKind.UNARY_OP,
            ASTNodeKind.ATTRIBUTE,
        ]
        for kind in kinds:
            assert kind is not None
