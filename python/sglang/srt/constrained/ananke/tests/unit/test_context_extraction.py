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
"""Tests for the ChatLSP-style context extraction module.

Tests the 5 ChatLSP methods for extracting type context from holes.
"""

import pytest

from core.context_extraction import (
    ContextExtractor,
    ExpectedTypeInfo,
    RelevantBinding,
    FunctionSignature,
    TypeErrorReport,
    extract_hole_context,
)
from domains.types.constraint import (
    INT,
    STR,
    BOOL,
    FLOAT,
    NONE,
    ANY,
    NEVER,
    FunctionType,
    ListType,
    DictType,
    TupleType,
    UnionType,
)
from domains.types.environment import TypeEnvironment, EMPTY_ENVIRONMENT
from domains.types.marking.marks import HoleMark, InconsistentMark
from domains.types.marking.provenance import SourceSpan, Provenance, UNKNOWN_SPAN
from domains.types.marking.marked_ast import (
    MarkedAST,
    MarkedASTNode,
    ASTNodeKind,
    create_hole_node,
    create_literal_node,
)


class TestExpectedTypeInfo:
    """Tests for ExpectedTypeInfo."""

    def test_unknown_type(self):
        """Unknown type info has ANY type."""
        info = ExpectedTypeInfo.unknown()
        assert info.type == ANY
        assert info.type_string == "Any"

    def test_with_constraints(self):
        """Type info can have constraints."""
        info = ExpectedTypeInfo(
            type=INT,
            type_string="int",
            constraints=["Must be positive"],
        )
        assert len(info.constraints) == 1


class TestContextExtractor:
    """Tests for ContextExtractor."""

    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = ContextExtractor()

    def test_expected_type_primitive(self):
        """Expected type for primitive."""
        hole = HoleMark("h0", INT, None)
        ast = MarkedAST(root=create_hole_node("h0", UNKNOWN_SPAN, INT))

        info = self.extractor.expected_type(hole, ast)

        assert info.type == INT
        assert info.type_string == "int"
        assert not info.is_callable
        assert not info.is_iterable

    def test_expected_type_function(self):
        """Expected type for function."""
        fn_type = FunctionType((INT, STR), BOOL)
        hole = HoleMark("h0", fn_type, None)
        ast = MarkedAST(root=create_hole_node("h0", UNKNOWN_SPAN, fn_type))

        info = self.extractor.expected_type(hole, ast)

        assert info.is_callable
        assert "callable" in info.type_string.lower() or "Callable" in info.type_string

    def test_expected_type_list(self):
        """Expected type for list."""
        list_type = ListType(INT)
        hole = HoleMark("h0", list_type, None)
        ast = MarkedAST(root=create_hole_node("h0", UNKNOWN_SPAN, list_type))

        info = self.extractor.expected_type(hole, ast)

        assert info.is_iterable
        assert "list" in info.type_string

    def test_expected_type_none(self):
        """Expected type when None."""
        hole = HoleMark("h0", None, None)
        ast = MarkedAST(root=create_hole_node("h0", UNKNOWN_SPAN))

        info = self.extractor.expected_type(hole, ast)

        assert info.type == ANY

    def test_relevant_types_empty_env(self):
        """Relevant types with empty environment."""
        hole = HoleMark("h0", INT, None)

        types = self.extractor.relevant_types(hole, EMPTY_ENVIRONMENT)

        assert types == []

    def test_relevant_types_with_bindings(self):
        """Relevant types with bindings."""
        hole = HoleMark("h0", INT, None)
        env = EMPTY_ENVIRONMENT.bind("x", INT).bind("y", STR)

        types = self.extractor.relevant_types(hole, env)

        # x: INT should be highly relevant to INT expected
        assert len(types) > 0
        names = {t.name for t in types}
        assert "x" in names

    def test_relevant_types_ranking(self):
        """Relevant types are ranked by relevance."""
        hole = HoleMark("h0", INT, None)
        env = EMPTY_ENVIRONMENT.bind("exact", INT).bind("wider", FLOAT).bind("unrelated", STR)

        types = self.extractor.relevant_types(hole, env)

        # Exact match should be first
        if len(types) > 0:
            assert types[0].name == "exact"

    def test_relevant_headers_empty(self):
        """Relevant headers with no functions."""
        hole = HoleMark("h0", INT, None)

        headers = self.extractor.relevant_headers(hole, EMPTY_ENVIRONMENT)

        assert headers == []

    def test_relevant_headers_with_functions(self):
        """Relevant headers with function bindings."""
        fn_type = FunctionType((STR,), INT)
        hole = HoleMark("h0", INT, None)
        env = EMPTY_ENVIRONMENT.bind("parse_int", fn_type)

        headers = self.extractor.relevant_headers(hole, env)

        assert len(headers) > 0
        assert headers[0].name == "parse_int"

    def test_error_report_no_errors(self):
        """Error report with no errors."""
        ast = MarkedAST(root=create_literal_node(42, INT, UNKNOWN_SPAN))

        errors = self.extractor.error_report(ast)

        assert errors == []

    def test_error_report_with_inconsistency(self):
        """Error report with inconsistent mark."""
        provenance = Provenance(UNKNOWN_SPAN, "test")
        mark = InconsistentMark(INT, STR, provenance)
        node = MarkedASTNode(
            kind=ASTNodeKind.LITERAL,
            span=UNKNOWN_SPAN,
            synthesized_type=INT,
            mark=mark,
        )
        ast = MarkedAST(root=node)

        errors = self.extractor.error_report(ast)

        assert len(errors) == 1
        assert errors[0].expected == STR
        assert errors[0].got == INT

    def test_ai_tutorial_int(self):
        """AI tutorial for int type."""
        hole = HoleMark("h0", INT, None)

        tutorial = self.extractor.ai_tutorial(hole, EMPTY_ENVIRONMENT)

        assert "int" in tutorial
        assert "Fill" in tutorial

    def test_ai_tutorial_function(self):
        """AI tutorial for function type."""
        fn_type = FunctionType((INT, STR), BOOL)
        hole = HoleMark("h0", fn_type, None)

        tutorial = self.extractor.ai_tutorial(hole, EMPTY_ENVIRONMENT)

        assert "function" in tutorial.lower()
        assert "arguments" in tutorial.lower() or "takes" in tutorial.lower()

    def test_ai_tutorial_list(self):
        """AI tutorial for list type."""
        list_type = ListType(INT)
        hole = HoleMark("h0", list_type, None)

        tutorial = self.extractor.ai_tutorial(hole, EMPTY_ENVIRONMENT)

        assert "list" in tutorial.lower()

    def test_ai_tutorial_with_bindings(self):
        """AI tutorial includes relevant bindings."""
        hole = HoleMark("h0", INT, None)
        env = EMPTY_ENVIRONMENT.bind("count", INT)

        tutorial = self.extractor.ai_tutorial(hole, env)

        assert "count" in tutorial


class TestRelevantBinding:
    """Tests for RelevantBinding dataclass."""

    def test_creation(self):
        """Create relevant binding."""
        binding = RelevantBinding(
            name="x",
            type=INT,
            relevance=0.9,
            source="local",
        )

        assert binding.name == "x"
        assert binding.type == INT
        assert binding.relevance == 0.9


class TestFunctionSignature:
    """Tests for FunctionSignature dataclass."""

    def test_format_no_params(self):
        """Format function with no parameters."""
        sig = FunctionSignature(
            name="get_value",
            params=(),
            return_type=INT,
        )

        formatted = sig.format()
        assert "get_value()" in formatted
        assert "int" in formatted

    def test_format_with_params(self):
        """Format function with parameters."""
        sig = FunctionSignature(
            name="add",
            params=(("x", INT), ("y", INT)),
            return_type=INT,
        )

        formatted = sig.format()
        assert "add(" in formatted
        assert "x" in formatted


class TestTypeErrorReport:
    """Tests for TypeErrorReport dataclass."""

    def test_creation(self):
        """Create error report."""
        report = TypeErrorReport(
            location=UNKNOWN_SPAN,
            message="Type mismatch",
            expected=STR,
            got=INT,
            suggestions=["Use str(...)"],
        )

        assert report.expected == STR
        assert report.got == INT
        assert len(report.suggestions) == 1


class TestExtractHoleContext:
    """Tests for extract_hole_context convenience function."""

    def test_extract_existing_hole(self):
        """Extract context for existing hole."""
        hole = create_hole_node("h0", UNKNOWN_SPAN, INT)
        ast = MarkedAST(root=hole)

        context = extract_hole_context("h0", ast, EMPTY_ENVIRONMENT)

        assert "expected_type" in context
        assert "relevant_types" in context
        assert "tutorial" in context

    def test_extract_missing_hole(self):
        """Extract context for missing hole."""
        node = create_literal_node(42, INT, UNKNOWN_SPAN)
        ast = MarkedAST(root=node)

        context = extract_hole_context("nonexistent", ast, EMPTY_ENVIRONMENT)

        assert "error" in context


class TestContextExtractorExamples:
    """Tests for example generation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = ContextExtractor()

    def test_int_examples(self):
        """Generate examples for int."""
        hole = HoleMark("h0", INT, None)
        ast = MarkedAST(root=create_hole_node("h0", UNKNOWN_SPAN, INT))

        info = self.extractor.expected_type(hole, ast)

        assert len(info.examples) > 0
        # Should have numeric examples
        assert any(ex.lstrip("-").isdigit() for ex in info.examples)

    def test_str_examples(self):
        """Generate examples for str."""
        hole = HoleMark("h0", STR, None)
        ast = MarkedAST(root=create_hole_node("h0", UNKNOWN_SPAN, STR))

        info = self.extractor.expected_type(hole, ast)

        assert len(info.examples) > 0
        # Should have string examples with quotes
        assert any('"' in ex or "'" in ex for ex in info.examples)

    def test_bool_examples(self):
        """Generate examples for bool."""
        hole = HoleMark("h0", BOOL, None)
        ast = MarkedAST(root=create_hole_node("h0", UNKNOWN_SPAN, BOOL))

        info = self.extractor.expected_type(hole, ast)

        assert len(info.examples) > 0
        assert any("True" in ex or "False" in ex for ex in info.examples)


class TestContextExtractorSubtyping:
    """Tests for subtype-based relevance scoring."""

    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = ContextExtractor()

    def test_int_relevant_to_float(self):
        """Int is relevant when float expected (int <: float)."""
        hole = HoleMark("h0", FLOAT, None)
        env = EMPTY_ENVIRONMENT.bind("count", INT)

        types = self.extractor.relevant_types(hole, env)

        # Int should be relevant since int <: float
        assert len(types) > 0
        names = {t.name for t in types}
        assert "count" in names

    def test_union_membership(self):
        """Type is relevant if member of expected union."""
        union_type = UnionType(frozenset([INT, STR]))
        hole = HoleMark("h0", union_type, None)
        env = EMPTY_ENVIRONMENT.bind("value", INT)

        types = self.extractor.relevant_types(hole, env)

        assert len(types) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
