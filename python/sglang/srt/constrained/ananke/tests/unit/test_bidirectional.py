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
"""Unit tests for bidirectional type checking.

Tests synthesis, analysis, and subsumption for the bidirectional type system.
"""

import pytest

from domains.types.constraint import (
    ANY,
    BOOL,
    FLOAT,
    INT,
    NEVER,
    NONE,
    STR,
    AnyType,
    DictType,
    FunctionType,
    HoleType,
    ListType,
    NeverType,
    SetType,
    TupleType,
    TypeVar,
    UnionType,
    ClassType,
)
from domains.types.environment import EMPTY_ENVIRONMENT, TypeEnvironment
from domains.types.marking.marks import HoleMark, InconsistentMark
from domains.types.marking.marked_ast import MarkedASTNode, ASTNodeKind
from domains.types.marking.provenance import SourceSpan
from domains.types.bidirectional.subsumption import (
    subsumes,
    check_subsumption,
    SubsumptionResult,
    is_assignable,
    join,
    meet,
)
from domains.types.bidirectional.synthesis import (
    synthesize,
    SynthesisResult,
)
from domains.types.bidirectional.analysis import (
    analyze,
    AnalysisResult,
    analyze_against_expected,
)


class TestSubsumes:
    """Tests for subsumption (subtype checking)."""

    def test_reflexivity(self):
        """Same type should subsume itself."""
        assert subsumes(INT, INT)
        assert subsumes(STR, STR)
        assert subsumes(BOOL, BOOL)

    def test_any_is_supertype(self):
        """Any should be supertype of everything."""
        assert subsumes(INT, ANY)
        assert subsumes(STR, ANY)
        assert subsumes(ListType(INT), ANY)
        assert subsumes(FunctionType((INT,), STR), ANY)

    def test_never_is_subtype(self):
        """Never should be subtype of everything."""
        assert subsumes(NEVER, INT)
        assert subsumes(NEVER, STR)
        assert subsumes(NEVER, ANY)
        assert subsumes(NEVER, ListType(INT))

    def test_hole_type_subsumes(self):
        """Hole types should satisfy any constraint."""
        hole = HoleType("h1")
        assert subsumes(hole, INT)
        assert subsumes(INT, hole)
        assert subsumes(hole, hole)

    def test_list_covariance(self):
        """List should be covariant in element type."""
        # List[Never] <: List[Int] because Never <: Int
        assert subsumes(ListType(NEVER), ListType(INT))
        # List[Int] <: List[Any] because Int <: Any
        assert subsumes(ListType(INT), ListType(ANY))

    def test_list_not_subtype(self):
        """List[Int] should not be subtype of List[Str]."""
        assert not subsumes(ListType(INT), ListType(STR))

    def test_set_covariance(self):
        """Set should be covariant in element type."""
        assert subsumes(SetType(NEVER), SetType(INT))
        assert subsumes(SetType(INT), SetType(ANY))

    def test_tuple_covariance(self):
        """Tuple should be covariant in each element."""
        assert subsumes(
            TupleType((INT, STR)),
            TupleType((INT, STR)),
        )
        assert subsumes(
            TupleType((NEVER, NEVER)),
            TupleType((INT, STR)),
        )

    def test_tuple_length_mismatch(self):
        """Tuples with different lengths should not be subtypes."""
        assert not subsumes(
            TupleType((INT,)),
            TupleType((INT, STR)),
        )

    def test_function_contravariant_params(self):
        """Function params should be contravariant."""
        # (Any) -> Int <: (Int) -> Int because Int <: Any (contravariant)
        assert subsumes(
            FunctionType((ANY,), INT),
            FunctionType((INT,), INT),
        )

    def test_function_covariant_return(self):
        """Function return should be covariant."""
        # (Int) -> Int <: (Int) -> Any because Int <: Any
        assert subsumes(
            FunctionType((INT,), INT),
            FunctionType((INT,), ANY),
        )

    def test_function_arity_mismatch(self):
        """Functions with different arity should not be subtypes."""
        assert not subsumes(
            FunctionType((INT,), STR),
            FunctionType((INT, INT), STR),
        )

    def test_union_subtype_of_member(self):
        """Type should be subtype of union containing it."""
        union = UnionType(frozenset({INT, STR}))
        assert subsumes(INT, union)
        assert subsumes(STR, union)

    def test_union_must_subsume_all(self):
        """Union subtype must have all members subsume target."""
        union = UnionType(frozenset({INT, STR}))
        # Union[Int, Str] <: Any because both Int <: Any and Str <: Any
        assert subsumes(union, ANY)
        # Union[Int, Str] not <: Int because Str not <: Int
        assert not subsumes(union, INT)

    def test_class_type_same_name(self):
        """Class types with same name should check type args."""
        assert subsumes(
            ClassType("Container", (INT,)),
            ClassType("Container", (INT,)),
        )

    def test_class_type_different_name(self):
        """Class types with different names should not be subtypes."""
        assert not subsumes(
            ClassType("Foo", ()),
            ClassType("Bar", ()),
        )

    def test_dict_type(self):
        """Dict type should check key and value."""
        assert subsumes(
            DictType(STR, INT),
            DictType(STR, INT),
        )
        # Key type mismatch
        assert not subsumes(
            DictType(INT, INT),
            DictType(STR, INT),
        )


class TestCheckSubsumption:
    """Tests for check_subsumption with detailed results."""

    def test_success_result(self):
        """Should return success for valid subtyping."""
        result = check_subsumption(INT, ANY)
        assert result.success
        assert result.reason is None

    def test_failure_with_reason(self):
        """Should return failure with reason."""
        result = check_subsumption(INT, STR)
        assert not result.success
        assert result.reason is not None

    def test_type_var_subsumption(self):
        """Same type variable should subsume itself."""
        var = TypeVar("T")
        result = check_subsumption(var, var)
        assert result.success

    def test_type_var_mismatch(self):
        """Different type variables should not subsume."""
        result = check_subsumption(TypeVar("T"), TypeVar("U"))
        assert not result.success


class TestSubsumptionResult:
    """Tests for SubsumptionResult class."""

    def test_ok_result(self):
        """ok() should create success result."""
        result = SubsumptionResult.ok()
        assert result.success
        assert result.reason is None

    def test_fail_result(self):
        """fail() should create failure result with reason."""
        result = SubsumptionResult.fail("test reason")
        assert not result.success
        assert result.reason == "test reason"


class TestIsAssignable:
    """Tests for is_assignable function."""

    def test_assignable(self):
        """Should check assignment compatibility."""
        assert is_assignable(INT, INT)
        assert is_assignable(INT, ANY)
        assert not is_assignable(INT, STR)


class TestJoin:
    """Tests for join (least upper bound)."""

    def test_join_same(self):
        """Join of same type should be that type."""
        assert join(INT, INT) == INT

    def test_join_with_any(self):
        """Join with Any should be Any."""
        assert join(INT, ANY) == ANY
        assert join(ANY, INT) == ANY

    def test_join_with_never(self):
        """Join with Never should be other type."""
        assert join(INT, NEVER) == INT
        assert join(NEVER, INT) == INT

    def test_join_subtype(self):
        """Join should return supertype if one subsumes other."""
        # Int <: Any, so join is Any
        result = join(INT, ANY)
        assert result == ANY

    def test_join_incompatible(self):
        """Join of incompatible types should be union."""
        result = join(INT, STR)
        assert isinstance(result, UnionType)
        assert INT in result.members
        assert STR in result.members


class TestMeet:
    """Tests for meet (greatest lower bound)."""

    def test_meet_same(self):
        """Meet of same type should be that type."""
        assert meet(INT, INT) == INT

    def test_meet_with_any(self):
        """Meet with Any should be other type."""
        assert meet(INT, ANY) == INT
        assert meet(ANY, INT) == INT

    def test_meet_with_never(self):
        """Meet with Never should be Never."""
        assert meet(INT, NEVER) == NEVER
        assert meet(NEVER, INT) == NEVER

    def test_meet_subtype(self):
        """Meet should return subtype if one subsumes other."""
        # Int <: Any, so meet is Int
        assert meet(INT, ANY) == INT

    def test_meet_incompatible(self):
        """Meet of incompatible types should be Never."""
        result = meet(INT, STR)
        assert result == NEVER


class TestSynthesis:
    """Tests for type synthesis."""

    def test_synthesize_int_literal(self):
        """Should synthesize int for int literal."""
        node = MarkedASTNode(
            kind=ASTNodeKind.LITERAL,
            span=SourceSpan(0, 5),
            data={"value": 42},
        )
        result = synthesize(node, EMPTY_ENVIRONMENT)
        assert result.synthesized_type == INT

    def test_synthesize_str_literal(self):
        """Should synthesize str for string literal."""
        node = MarkedASTNode(
            kind=ASTNodeKind.LITERAL,
            span=SourceSpan(0, 10),
            data={"value": "hello"},
        )
        result = synthesize(node, EMPTY_ENVIRONMENT)
        assert result.synthesized_type == STR

    def test_synthesize_bool_literal(self):
        """Should synthesize bool for boolean literal."""
        node = MarkedASTNode(
            kind=ASTNodeKind.LITERAL,
            span=SourceSpan(0, 4),
            data={"value": True},
        )
        result = synthesize(node, EMPTY_ENVIRONMENT)
        assert result.synthesized_type == BOOL

    def test_synthesize_float_literal(self):
        """Should synthesize float for float literal."""
        node = MarkedASTNode(
            kind=ASTNodeKind.LITERAL,
            span=SourceSpan(0, 5),
            data={"value": 3.14},
        )
        result = synthesize(node, EMPTY_ENVIRONMENT)
        assert result.synthesized_type == FLOAT

    def test_synthesize_none_literal(self):
        """Should synthesize None for None literal."""
        node = MarkedASTNode(
            kind=ASTNodeKind.LITERAL,
            span=SourceSpan(0, 4),
            data={"value": None},
        )
        result = synthesize(node, EMPTY_ENVIRONMENT)
        assert result.synthesized_type == NONE

    def test_synthesize_variable(self):
        """Should synthesize variable type from environment."""
        env = EMPTY_ENVIRONMENT.bind("x", INT)
        node = MarkedASTNode(
            kind=ASTNodeKind.VARIABLE,
            span=SourceSpan(0, 1),
            data={"name": "x"},
        )
        result = synthesize(node, env)
        assert result.synthesized_type == INT

    def test_synthesize_unbound_variable(self):
        """Should synthesize Any for unbound variable."""
        node = MarkedASTNode(
            kind=ASTNodeKind.VARIABLE,
            span=SourceSpan(0, 1),
            data={"name": "unknown"},
        )
        result = synthesize(node, EMPTY_ENVIRONMENT)
        assert result.synthesized_type == ANY
        assert result.has_errors

    def test_synthesize_application(self):
        """Should synthesize function return type for application."""
        func_type = FunctionType((INT,), STR)
        env = EMPTY_ENVIRONMENT.bind("f", func_type)

        func_node = MarkedASTNode(
            kind=ASTNodeKind.VARIABLE,
            span=SourceSpan(0, 1),
            data={"name": "f"},
        )
        arg_node = MarkedASTNode(
            kind=ASTNodeKind.LITERAL,
            span=SourceSpan(2, 4),
            data={"value": 42},
        )
        app_node = MarkedASTNode(
            kind=ASTNodeKind.APPLICATION,
            span=SourceSpan(0, 5),
            children=[func_node, arg_node],
        )

        result = synthesize(app_node, env)
        assert result.synthesized_type == STR

    def test_synthesize_empty_list(self):
        """Should synthesize List[Any] for empty list."""
        node = MarkedASTNode(
            kind=ASTNodeKind.LIST,
            span=SourceSpan(0, 2),
            children=[],
        )
        result = synthesize(node, EMPTY_ENVIRONMENT)
        assert isinstance(result.synthesized_type, ListType)
        assert result.synthesized_type.element == ANY

    def test_synthesize_homogeneous_list(self):
        """Should synthesize List[Int] for list of ints."""
        children = [
            MarkedASTNode(
                kind=ASTNodeKind.LITERAL,
                span=SourceSpan(i, i+1),
                data={"value": i},
            )
            for i in range(3)
        ]
        node = MarkedASTNode(
            kind=ASTNodeKind.LIST,
            span=SourceSpan(0, 10),
            children=children,
        )
        result = synthesize(node, EMPTY_ENVIRONMENT)
        assert isinstance(result.synthesized_type, ListType)
        assert result.synthesized_type.element == INT

    def test_synthesize_tuple(self):
        """Should synthesize tuple with element types."""
        children = [
            MarkedASTNode(
                kind=ASTNodeKind.LITERAL,
                span=SourceSpan(0, 2),
                data={"value": 42},
            ),
            MarkedASTNode(
                kind=ASTNodeKind.LITERAL,
                span=SourceSpan(4, 10),
                data={"value": "hello"},
            ),
        ]
        node = MarkedASTNode(
            kind=ASTNodeKind.TUPLE,
            span=SourceSpan(0, 11),
            children=children,
        )
        result = synthesize(node, EMPTY_ENVIRONMENT)
        assert isinstance(result.synthesized_type, TupleType)
        assert result.synthesized_type.elements == (INT, STR)

    def test_synthesize_binary_arithmetic(self):
        """Should synthesize numeric type for arithmetic."""
        left = MarkedASTNode(
            kind=ASTNodeKind.LITERAL,
            span=SourceSpan(0, 1),
            data={"value": 1},
        )
        right = MarkedASTNode(
            kind=ASTNodeKind.LITERAL,
            span=SourceSpan(4, 5),
            data={"value": 2},
        )
        node = MarkedASTNode(
            kind=ASTNodeKind.BINARY_OP,
            span=SourceSpan(0, 5),
            children=[left, right],
            data={"operator": "+"},
        )
        result = synthesize(node, EMPTY_ENVIRONMENT)
        assert result.synthesized_type == INT

    def test_synthesize_binary_comparison(self):
        """Should synthesize bool for comparison."""
        left = MarkedASTNode(
            kind=ASTNodeKind.LITERAL,
            span=SourceSpan(0, 1),
            data={"value": 1},
        )
        right = MarkedASTNode(
            kind=ASTNodeKind.LITERAL,
            span=SourceSpan(4, 5),
            data={"value": 2},
        )
        node = MarkedASTNode(
            kind=ASTNodeKind.BINARY_OP,
            span=SourceSpan(0, 5),
            children=[left, right],
            data={"operator": "<"},
        )
        result = synthesize(node, EMPTY_ENVIRONMENT)
        assert result.synthesized_type == BOOL

    def test_synthesize_hole_with_expected(self):
        """Should synthesize expected type for hole if marked."""
        mark = HoleMark(hole_id="h1", expected_type=INT)
        node = MarkedASTNode(
            kind=ASTNodeKind.HOLE,
            span=SourceSpan(0, 1),
            mark=mark,
        )
        result = synthesize(node, EMPTY_ENVIRONMENT)
        assert result.synthesized_type == INT


class TestSynthesisResult:
    """Tests for SynthesisResult class."""

    def test_has_errors_false(self):
        """Should report no errors when empty."""
        node = MarkedASTNode(
            kind=ASTNodeKind.LITERAL,
            span=SourceSpan(0, 5),
        )
        result = SynthesisResult(synthesized_type=INT, node=node)
        assert not result.has_errors

    def test_has_errors_true(self):
        """Should report errors when present."""
        node = MarkedASTNode(
            kind=ASTNodeKind.LITERAL,
            span=SourceSpan(0, 5),
        )
        result = SynthesisResult(
            synthesized_type=INT,
            node=node,
            errors=["test error"],
        )
        assert result.has_errors


class TestAnalysis:
    """Tests for type analysis."""

    def test_analyze_literal_matching(self):
        """Should succeed when literal matches expected type."""
        node = MarkedASTNode(
            kind=ASTNodeKind.LITERAL,
            span=SourceSpan(0, 5),
            data={"value": 42},
        )
        result = analyze(node, INT, EMPTY_ENVIRONMENT)
        assert result.success

    def test_analyze_literal_any(self):
        """Should succeed when expected is Any."""
        node = MarkedASTNode(
            kind=ASTNodeKind.LITERAL,
            span=SourceSpan(0, 5),
            data={"value": 42},
        )
        result = analyze(node, ANY, EMPTY_ENVIRONMENT)
        assert result.success

    def test_analyze_literal_mismatch(self):
        """Should fail when literal doesn't match expected."""
        node = MarkedASTNode(
            kind=ASTNodeKind.LITERAL,
            span=SourceSpan(0, 5),
            data={"value": 42},  # int
        )
        result = analyze(node, STR, EMPTY_ENVIRONMENT)  # expected string
        assert not result.success
        assert isinstance(result.node.mark, InconsistentMark)

    def test_analyze_hole(self):
        """Should succeed for hole and mark with expected type."""
        node = MarkedASTNode(
            kind=ASTNodeKind.HOLE,
            span=SourceSpan(0, 1),
        )
        result = analyze(node, INT, EMPTY_ENVIRONMENT)
        assert result.success
        assert isinstance(result.node.mark, HoleMark)
        assert result.node.mark.expected_type == INT

    def test_analyze_lambda(self):
        """Should analyze lambda against function type."""
        body = MarkedASTNode(
            kind=ASTNodeKind.VARIABLE,
            span=SourceSpan(10, 11),
            data={"name": "x"},
        )
        node = MarkedASTNode(
            kind=ASTNodeKind.LAMBDA,
            span=SourceSpan(0, 20),
            data={"params": ["x"]},
            children=[body],
        )
        expected = FunctionType((INT,), INT)
        result = analyze(node, expected, EMPTY_ENVIRONMENT)
        assert result.success
        assert result.node.synthesized_type == expected

    def test_analyze_lambda_wrong_arity(self):
        """Should fail lambda with wrong parameter count."""
        body = MarkedASTNode(
            kind=ASTNodeKind.LITERAL,
            span=SourceSpan(10, 11),
            data={"value": 42},
        )
        node = MarkedASTNode(
            kind=ASTNodeKind.LAMBDA,
            span=SourceSpan(0, 20),
            data={"params": ["x"]},  # 1 param
            children=[body],
        )
        expected = FunctionType((INT, STR), INT)  # expects 2 params
        result = analyze(node, expected, EMPTY_ENVIRONMENT)
        assert not result.success

    def test_analyze_lambda_not_function_type(self):
        """Should fail lambda against non-function type."""
        body = MarkedASTNode(
            kind=ASTNodeKind.LITERAL,
            span=SourceSpan(10, 11),
            data={"value": 42},
        )
        node = MarkedASTNode(
            kind=ASTNodeKind.LAMBDA,
            span=SourceSpan(0, 20),
            data={"params": ["x"]},
            children=[body],
        )
        result = analyze(node, INT, EMPTY_ENVIRONMENT)  # Not a function type
        assert not result.success

    def test_analyze_if(self):
        """Should analyze if expression."""
        cond = MarkedASTNode(
            kind=ASTNodeKind.LITERAL,
            span=SourceSpan(3, 7),
            data={"value": True},
        )
        then_branch = MarkedASTNode(
            kind=ASTNodeKind.LITERAL,
            span=SourceSpan(13, 14),
            data={"value": 1},
        )
        else_branch = MarkedASTNode(
            kind=ASTNodeKind.LITERAL,
            span=SourceSpan(20, 21),
            data={"value": 2},
        )
        node = MarkedASTNode(
            kind=ASTNodeKind.IF,
            span=SourceSpan(0, 25),
            children=[cond, then_branch, else_branch],
        )
        result = analyze(node, INT, EMPTY_ENVIRONMENT)
        assert result.success


class TestAnalysisResult:
    """Tests for AnalysisResult class."""

    def test_has_errors_success(self):
        """Should not have errors on success."""
        node = MarkedASTNode(
            kind=ASTNodeKind.LITERAL,
            span=SourceSpan(0, 5),
        )
        result = AnalysisResult(success=True, node=node)
        assert not result.has_errors

    def test_has_errors_failure(self):
        """Should have errors on failure."""
        node = MarkedASTNode(
            kind=ASTNodeKind.LITERAL,
            span=SourceSpan(0, 5),
        )
        result = AnalysisResult(
            success=False,
            node=node,
            errors=["type mismatch"],
        )
        assert result.has_errors


class TestAnalyzeAgainstExpected:
    """Tests for analyze_against_expected convenience function."""

    def test_with_expected(self):
        """Should call analyze when expected is provided."""
        node = MarkedASTNode(
            kind=ASTNodeKind.LITERAL,
            span=SourceSpan(0, 5),
            data={"value": 42},
        )
        result = analyze_against_expected(node, INT, EMPTY_ENVIRONMENT)
        assert result.success

    def test_without_expected(self):
        """Should call synthesize when expected is None."""
        node = MarkedASTNode(
            kind=ASTNodeKind.LITERAL,
            span=SourceSpan(0, 5),
            data={"value": 42},
        )
        result = analyze_against_expected(node, None, EMPTY_ENVIRONMENT)
        assert result.success
        assert result.node.synthesized_type == INT
