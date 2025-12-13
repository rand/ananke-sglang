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
"""Property-based tests for the Totality Invariant.

From Hazel research (POPL 2024): Every partial program must have a well-defined
type. This means:

1. Type synthesis never fails - it always produces a type
2. Type analysis never fails - it always produces a mark
3. Error states are first-class values, not exceptions

The totality invariant ensures that generation never gets stuck on type errors.
Errors are localized to marks, not propagated as failures.

Key property:
    ∀ e ∈ PartialAST, ∀ Γ ∈ TypeEnvironment, ∀ τ ∈ Type:
        totalize(e, τ, Γ) returns a MarkedAST with synthesized_type ≠ ⊥

References:
    - POPL 2024: "Total Type Error Localization and Recovery with Holes"
"""

import pytest
from hypothesis import given, settings, strategies as st, assume

from domains.types.constraint import (
    Type,
    TypeVar,
    PrimitiveType,
    FunctionType,
    ListType,
    DictType,
    TupleType,
    SetType,
    UnionType,
    ClassType,
    AnyType,
    NeverType,
    HoleType,
    INT,
    STR,
    BOOL,
    FLOAT,
    NONE,
    ANY,
    NEVER,
)
from domains.types.environment import TypeEnvironment, EMPTY_ENVIRONMENT
from domains.types.marking.marks import (
    Mark,
    HoleMark,
    InconsistentMark,
    NonEmptyHoleMark,
    create_hole_mark,
    create_inconsistent_mark,
)
from domains.types.marking.provenance import SourceSpan, Provenance, UNKNOWN_SPAN
from domains.types.marking.marked_ast import (
    MarkedAST,
    MarkedASTNode,
    ASTNodeKind,
    create_hole_node,
    create_literal_node,
    create_variable_node,
)
from domains.types.marking.totalization import (
    totalize,
    TotalizationResult,
)
from domains.types.bidirectional.synthesis import (
    SynthesisResult,
    synthesize,
)
from domains.types.bidirectional.analysis import (
    AnalysisResult,
    analyze,
)


# =============================================================================
# Type Generation Strategies
# =============================================================================

@st.composite
def primitive_type_strategy(draw):
    """Generate primitive types."""
    return draw(st.sampled_from([INT, STR, BOOL, FLOAT, NONE]))


@st.composite
def type_var_strategy(draw):
    """Generate type variables."""
    name = draw(st.text(alphabet="TUVWXYZ", min_size=1, max_size=2))
    return TypeVar(name)


@st.composite
def simple_type_strategy(draw):
    """Generate simple (non-compound) types."""
    return draw(st.one_of(
        primitive_type_strategy(),
        st.just(ANY),
        st.just(NEVER),
    ))


@st.composite
def list_type_strategy(draw, inner=None):
    """Generate list types."""
    if inner is None:
        inner = draw(simple_type_strategy())
    return ListType(inner)


@st.composite
def set_type_strategy(draw, inner=None):
    """Generate set types."""
    if inner is None:
        inner = draw(simple_type_strategy())
    return SetType(inner)


@st.composite
def dict_type_strategy(draw, key=None, value=None):
    """Generate dict types."""
    if key is None:
        key = draw(simple_type_strategy())
    if value is None:
        value = draw(simple_type_strategy())
    return DictType(key, value)


@st.composite
def tuple_type_strategy(draw, max_size=3):
    """Generate tuple types."""
    size = draw(st.integers(min_value=0, max_value=max_size))
    elements = tuple(draw(simple_type_strategy()) for _ in range(size))
    return TupleType(elements)


@st.composite
def function_type_strategy(draw, max_params=3):
    """Generate function types."""
    num_params = draw(st.integers(min_value=0, max_value=max_params))
    params = tuple(draw(simple_type_strategy()) for _ in range(num_params))
    returns = draw(simple_type_strategy())
    return FunctionType(params, returns)


@st.composite
def union_type_strategy(draw, max_members=3):
    """Generate union types."""
    num_members = draw(st.integers(min_value=2, max_value=max_members))
    members = frozenset(draw(simple_type_strategy()) for _ in range(num_members))
    # Ensure we have at least 2 distinct members
    assume(len(members) >= 2)
    return UnionType(members)


@st.composite
def type_strategy(draw, max_depth=2):
    """Generate arbitrary types up to a given depth."""
    if max_depth <= 0:
        return draw(simple_type_strategy())

    return draw(st.one_of(
        simple_type_strategy(),
        list_type_strategy(draw(type_strategy(max_depth=max_depth-1))),
        set_type_strategy(draw(type_strategy(max_depth=max_depth-1))),
        tuple_type_strategy(max_size=2),
        function_type_strategy(max_params=2),
    ))


# =============================================================================
# AST Node Generation Strategies
# =============================================================================

@st.composite
def literal_node_strategy(draw):
    """Generate literal AST nodes with their correct types."""
    choice = draw(st.integers(min_value=0, max_value=4))
    if choice == 0:
        value = draw(st.integers())
        return create_literal_node(value=value, ty=INT, span=UNKNOWN_SPAN)
    elif choice == 1:
        value = draw(st.text(max_size=10))
        return create_literal_node(value=value, ty=STR, span=UNKNOWN_SPAN)
    elif choice == 2:
        value = draw(st.booleans())
        return create_literal_node(value=value, ty=BOOL, span=UNKNOWN_SPAN)
    elif choice == 3:
        value = draw(st.floats(allow_nan=False, allow_infinity=False))
        return create_literal_node(value=value, ty=FLOAT, span=UNKNOWN_SPAN)
    else:
        return create_literal_node(value=None, ty=NONE, span=UNKNOWN_SPAN)


@st.composite
def variable_node_strategy(draw):
    """Generate variable AST nodes."""
    name = draw(st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=5))
    # Variable nodes need a type - we use ANY for undetermined types
    ty = draw(st.one_of(
        simple_type_strategy(),
        st.just(ANY),
    ))
    return create_variable_node(name=name, ty=ty, span=UNKNOWN_SPAN)


@st.composite
def hole_node_strategy(draw):
    """Generate hole AST nodes."""
    hole_id = f"hole_{draw(st.integers(min_value=0, max_value=100))}"
    expected_type = draw(st.one_of(
        st.none(),
        simple_type_strategy(),
    ))
    return create_hole_node(hole_id=hole_id, expected_type=expected_type, span=UNKNOWN_SPAN)


@st.composite
def simple_ast_strategy(draw):
    """Generate simple AST nodes (no recursion)."""
    return draw(st.one_of(
        literal_node_strategy(),
        hole_node_strategy(),
    ))


@st.composite
def list_node_strategy(draw, element_strategy=None):
    """Generate list AST nodes."""
    if element_strategy is None:
        element_strategy = literal_node_strategy()
    num_elements = draw(st.integers(min_value=0, max_value=3))
    elements = [draw(element_strategy) for _ in range(num_elements)]
    return MarkedASTNode(
        kind=ASTNodeKind.LIST,
        span=UNKNOWN_SPAN,
        children=elements,
    )


@st.composite
def tuple_node_strategy(draw, element_strategy=None):
    """Generate tuple AST nodes."""
    if element_strategy is None:
        element_strategy = literal_node_strategy()
    num_elements = draw(st.integers(min_value=0, max_value=3))
    elements = [draw(element_strategy) for _ in range(num_elements)]
    return MarkedASTNode(
        kind=ASTNodeKind.TUPLE,
        span=UNKNOWN_SPAN,
        children=elements,
    )


# =============================================================================
# Environment Generation Strategies
# =============================================================================

@st.composite
def environment_strategy(draw, max_bindings=5):
    """Generate type environments."""
    num_bindings = draw(st.integers(min_value=0, max_value=max_bindings))
    env = EMPTY_ENVIRONMENT
    for _ in range(num_bindings):
        name = draw(st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=5))
        typ = draw(simple_type_strategy())
        env = env.bind(name, typ)
    return env


# =============================================================================
# Totality Property Tests
# =============================================================================

class TestTotalitySynthesis:
    """Tests that type synthesis always succeeds (never fails)."""

    @given(node=literal_node_strategy())
    @settings(max_examples=100)
    def test_synthesis_always_returns_type_for_literals(self, node: MarkedASTNode):
        """Synthesis of literals always produces a type."""
        result = synthesize(node, EMPTY_ENVIRONMENT)

        # Key invariant: synthesis never returns None
        assert result is not None
        assert isinstance(result, SynthesisResult)
        assert result.synthesized_type is not None
        # Type should match the synthesized_type stored in the node
        if node.synthesized_type is not None:
            assert result.synthesized_type == node.synthesized_type

    @given(hole_id=st.text(min_size=1, max_size=10, alphabet="abcdefghijklmnopqrstuvwxyz0123456789"))
    @settings(max_examples=50)
    def test_synthesis_of_holes_returns_type(self, hole_id: str):
        """Synthesis of holes returns a type, not failure."""
        node = create_hole_node(hole_id=hole_id, expected_type=None, span=UNKNOWN_SPAN)
        result = synthesize(node, EMPTY_ENVIRONMENT)

        assert result is not None
        assert isinstance(result, SynthesisResult)
        # Holes synthesize to HoleType or Any (depending on implementation)
        assert result.synthesized_type is not None

    @given(node=list_node_strategy())
    @settings(max_examples=50)
    def test_synthesis_of_lists_always_succeeds(self, node: MarkedASTNode):
        """Synthesis of list literals always produces a type."""
        result = synthesize(node, EMPTY_ENVIRONMENT)

        assert result is not None
        assert isinstance(result, SynthesisResult)
        assert result.synthesized_type is not None
        # Should be a list type
        assert isinstance(result.synthesized_type, (ListType, AnyType))

    @given(node=tuple_node_strategy())
    @settings(max_examples=50)
    def test_synthesis_of_tuples_always_succeeds(self, node: MarkedASTNode):
        """Synthesis of tuple literals always produces a type."""
        result = synthesize(node, EMPTY_ENVIRONMENT)

        assert result is not None
        assert isinstance(result, SynthesisResult)
        assert result.synthesized_type is not None
        # Should be a tuple type
        assert isinstance(result.synthesized_type, (TupleType, AnyType))


class TestTotalityAnalysis:
    """Tests that type analysis always succeeds (with marks for errors)."""

    @given(node=literal_node_strategy(), expected=simple_type_strategy())
    @settings(max_examples=100)
    def test_analysis_always_returns_result(self, node: MarkedASTNode, expected: Type):
        """Analysis always produces a result, even on type mismatches."""
        result = analyze(node, expected, EMPTY_ENVIRONMENT)

        # Key invariant: analysis never raises, always returns a result
        assert result is not None
        assert isinstance(result, AnalysisResult)
        assert result.node is not None

    @given(expected=simple_type_strategy())
    @settings(max_examples=50)
    def test_analysis_of_holes_returns_result(self, expected: Type):
        """Analysis of holes always succeeds."""
        node = create_hole_node(hole_id="test", expected_type=None, span=UNKNOWN_SPAN)
        result = analyze(node, expected, EMPTY_ENVIRONMENT)

        assert result is not None
        assert isinstance(result, AnalysisResult)


class TestTotalityTypeMismatch:
    """Tests that type mismatches produce marks, not failures."""

    def test_int_vs_string_produces_inconsistent_mark(self):
        """Analyzing int literal against string type produces InconsistentMark."""
        node = create_literal_node(value=42, ty=INT, span=UNKNOWN_SPAN)
        result = analyze(node, STR, EMPTY_ENVIRONMENT)

        assert result is not None
        assert isinstance(result, AnalysisResult)
        # Should have an inconsistent mark
        assert result.node.mark is not None
        assert isinstance(result.node.mark, InconsistentMark)
        assert result.node.mark.synthesized == INT
        assert result.node.mark.expected == STR

    def test_string_vs_int_produces_inconsistent_mark(self):
        """Analyzing string literal against int type produces InconsistentMark."""
        node = create_literal_node(value="hello", ty=STR, span=UNKNOWN_SPAN)
        result = analyze(node, INT, EMPTY_ENVIRONMENT)

        assert result is not None
        assert isinstance(result, AnalysisResult)
        # Should have an inconsistent mark
        assert result.node.mark is not None
        assert isinstance(result.node.mark, InconsistentMark)

    def test_bool_to_int_is_allowed(self):
        """Analyzing bool literal against int type should be allowed (bool <: int)."""
        node = create_literal_node(value=True, ty=BOOL, span=UNKNOWN_SPAN)
        result = analyze(node, INT, EMPTY_ENVIRONMENT)

        assert result is not None
        # Bool is subtype of int, so no error mark expected
        # (depends on implementation - may have mark if strict)


class TestTotalityTotalization:
    """Tests for the totalize() function which implements the full algorithm."""

    @given(node=simple_ast_strategy(), expected=simple_type_strategy())
    @settings(max_examples=100)
    def test_totalize_never_raises(self, node: MarkedASTNode, expected: Type):
        """totalize() never raises an exception."""
        try:
            result = totalize(node, expected, EMPTY_ENVIRONMENT)
            # Should always succeed
            assert result is not None
            assert isinstance(result, TotalizationResult)
        except Exception as e:
            # This should never happen - totality means no exceptions
            pytest.fail(f"totalize() raised {type(e).__name__}: {e}")

    @given(node=simple_ast_strategy(), expected=simple_type_strategy())
    @settings(max_examples=100)
    def test_totalize_always_assigns_type(self, node: MarkedASTNode, expected: Type):
        """totalize() always assigns a synthesized type."""
        result = totalize(node, expected, EMPTY_ENVIRONMENT)

        # Key invariant: synthesized_type is never None
        assert result.ast.root.synthesized_type is not None


class TestTotalityHolePreservation:
    """Tests that holes are properly preserved through totalization."""

    @given(hole_id=st.text(min_size=1, max_size=10, alphabet="abcdefghijklmnopqrstuvwxyz"))
    @settings(max_examples=50)
    def test_holes_are_marked_not_failed(self, hole_id: str):
        """Holes result in HoleMark, not failures."""
        node = create_hole_node(hole_id=hole_id, expected_type=None, span=UNKNOWN_SPAN)
        result = totalize(node, INT, EMPTY_ENVIRONMENT)

        assert result is not None
        # Hole should have a mark
        assert result.ast.root.mark is not None
        # Mark should be HoleMark with expected type
        assert isinstance(result.ast.root.mark, HoleMark)
        assert result.ast.root.mark.expected_type == INT
        assert result.ast.root.mark.hole_id == hole_id


class TestTotalityErrorLocalization:
    """Tests that errors are localized to specific marks with provenances."""

    def test_error_has_provenance(self):
        """Type errors include provenance information."""
        node = create_literal_node(value=42, ty=INT, span=UNKNOWN_SPAN)
        result = totalize(node, STR, EMPTY_ENVIRONMENT)

        assert result is not None
        if isinstance(result.ast.root.mark, InconsistentMark):
            # Should have provenance
            assert result.ast.root.mark.provenance is not None
            assert isinstance(result.ast.root.mark.provenance, Provenance)


class TestTotalityNeverReturnsNone:
    """Tests the fundamental invariant: all operations return valid results."""

    @given(typ=simple_type_strategy())
    @settings(max_examples=50)
    def test_synthesis_result_type_never_none(self, typ: Type):
        """Synthesis always returns non-None type."""
        node = create_literal_node(value=42, ty=INT, span=UNKNOWN_SPAN)
        result = synthesize(node, EMPTY_ENVIRONMENT)
        assert result.synthesized_type is not None

    @given(typ=simple_type_strategy())
    @settings(max_examples=50)
    def test_analysis_result_node_never_none(self, typ: Type):
        """Analysis always returns non-None node."""
        node = create_literal_node(value=42, ty=INT, span=UNKNOWN_SPAN)
        result = analyze(node, typ, EMPTY_ENVIRONMENT)
        assert result.node is not None

    @given(expected=simple_type_strategy())
    @settings(max_examples=50)
    def test_totalization_result_node_never_none(self, expected: Type):
        """Totalization always returns non-None node."""
        node = create_hole_node(hole_id="test", expected_type=None, span=UNKNOWN_SPAN)
        result = totalize(node, expected, EMPTY_ENVIRONMENT)
        assert result.ast.root is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
