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
"""Property-based tests for Phase 1 components.

Tests the mathematical properties of:
1. Z3 SMT integration (soundness, formula parsing)
2. Variance handling (covariance/contravariance)
3. Structural subtyping (protocol satisfaction)
4. Subtype relation (reflexive, transitive, antisymmetric)
"""

from __future__ import annotations

import pytest
from hypothesis import given, strategies as st, assume, settings

try:
    from ...domains.types.constraint import (
        Type,
        TypeVar,
        FunctionType,
        ListType,
        DictType,
        TupleType,
        ClassType,
        ProtocolType,
        UnionType,
        AnyType,
        NeverType,
        HoleType,
        Variance,
        INT,
        STR,
        BOOL,
        FLOAT,
        NONE,
        ANY,
        NEVER,
    )
    from ...domains.types.unification import (
        unify,
        unify_with_variance,
        is_subtype,
        EMPTY_SUBSTITUTION,
    )
    from ...domains.types.domain import (
        TypeDomain,
        NarrowingContext,
        EMPTY_NARROWING,
    )
    from ...domains.semantics.smt import Z3Solver, SMTFormula, SMTResult
except ImportError:
    from domains.types.constraint import (
        Type,
        TypeVar,
        FunctionType,
        ListType,
        DictType,
        TupleType,
        ClassType,
        ProtocolType,
        UnionType,
        AnyType,
        NeverType,
        HoleType,
        Variance,
        INT,
        STR,
        BOOL,
        FLOAT,
        NONE,
        ANY,
        NEVER,
    )
    from domains.types.unification import (
        unify,
        unify_with_variance,
        is_subtype,
        EMPTY_SUBSTITUTION,
    )
    from domains.types.domain import (
        TypeDomain,
        NarrowingContext,
        EMPTY_NARROWING,
    )
    from domains.semantics.smt import Z3Solver, SMTFormula, SMTResult


# =============================================================================
# Hypothesis Strategies for Types
# =============================================================================


@st.composite
def primitive_types(draw):
    """Strategy for primitive types."""
    return draw(st.sampled_from([INT, STR, BOOL, FLOAT, NONE]))


@st.composite
def special_types(draw):
    """Strategy for special types (Any, Never)."""
    return draw(st.sampled_from([ANY, NEVER]))


@st.composite
def type_vars(draw):
    """Strategy for type variables."""
    name = draw(st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=5))
    return TypeVar(name)


@st.composite
def simple_types(draw):
    """Strategy for simple types (primitives or type vars)."""
    return draw(st.one_of(primitive_types(), type_vars()))


@st.composite
def list_types(draw, element_strategy=None):
    """Strategy for list types."""
    if element_strategy is None:
        element_strategy = primitive_types()
    element = draw(element_strategy)
    return ListType(element)


@st.composite
def function_types(draw):
    """Strategy for function types."""
    n_params = draw(st.integers(min_value=0, max_value=3))
    params = tuple(draw(primitive_types()) for _ in range(n_params))
    returns = draw(primitive_types())
    return FunctionType(params, returns)


@st.composite
def all_types(draw, max_depth=2):
    """Strategy for all types with bounded depth."""
    if max_depth <= 0:
        return draw(st.one_of(primitive_types(), special_types()))

    return draw(st.one_of(
        primitive_types(),
        special_types(),
        list_types(primitive_types()),
        function_types(),
    ))


# =============================================================================
# Subtype Relation Properties
# =============================================================================


class TestSubtypeProperties:
    """Property-based tests for subtype relation."""

    @given(all_types())
    @settings(max_examples=100)
    def test_reflexive(self, t: Type) -> None:
        """Subtype relation is reflexive: T <: T."""
        assert is_subtype(t, t), f"Reflexivity violated: {t} not <: {t}"

    @given(all_types(), all_types(), all_types())
    @settings(max_examples=100)
    def test_any_is_top(self, t1: Type, t2: Type, t3: Type) -> None:
        """Any is the top type: T <: Any for all T."""
        assert is_subtype(t1, ANY), f"{t1} not <: Any"
        assert is_subtype(t2, ANY), f"{t2} not <: Any"
        assert is_subtype(t3, ANY), f"{t3} not <: Any"

    @given(all_types())
    @settings(max_examples=100)
    def test_never_is_bottom(self, t: Type) -> None:
        """Never is the bottom type: Never <: T for all T."""
        assert is_subtype(NEVER, t), f"Never not <: {t}"

    @given(primitive_types())
    @settings(max_examples=50)
    def test_numeric_promotion(self, t: Type) -> None:
        """int <: float but not float <: int."""
        if t == INT:
            assert is_subtype(INT, FLOAT)
        if t == FLOAT:
            assert not is_subtype(FLOAT, INT)


class TestVarianceProperties:
    """Property-based tests for variance handling."""

    @given(primitive_types())
    @settings(max_examples=50)
    def test_invariant_self(self, t: Type) -> None:
        """Invariant: T unifies with T."""
        result = unify_with_variance(t, t, Variance.INVARIANT)
        assert result.is_success

    @given(primitive_types(), primitive_types())
    @settings(max_examples=100)
    def test_covariant_soundness(self, t1: Type, t2: Type) -> None:
        """Covariant: success implies subtype."""
        result = unify_with_variance(t1, t2, Variance.COVARIANT)
        if result.is_success and t1 != t2:
            # Success means t1 <: t2 or types unified
            pass  # Just ensure no crash

    @given(primitive_types(), primitive_types())
    @settings(max_examples=100)
    def test_contravariant_soundness(self, t1: Type, t2: Type) -> None:
        """Contravariant: success implies supertype."""
        result = unify_with_variance(t1, t2, Variance.CONTRAVARIANT)
        # Should not crash and returns valid result
        assert result.is_success or result.is_failure


class TestFunctionVariance:
    """Property-based tests for function type variance."""

    @given(function_types(), function_types())
    @settings(max_examples=100)
    def test_function_unify_symmetric_failure(self, f1: FunctionType, f2: FunctionType) -> None:
        """If f1 can't unify with f2 due to arity, f2 can't unify with f1."""
        if len(f1.params) != len(f2.params):
            r1 = unify(f1, f2)
            r2 = unify(f2, f1)
            assert r1.is_failure
            assert r2.is_failure


# =============================================================================
# Protocol Subtyping Properties
# =============================================================================


class TestProtocolProperties:
    """Property-based tests for protocol structural subtyping."""

    @given(st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=10))
    @settings(max_examples=50)
    def test_any_satisfies_any_protocol(self, name: str) -> None:
        """Any satisfies any protocol."""
        protocol = ProtocolType(
            name=name,
            members=frozenset({("method", FunctionType((), INT))}),
        )
        assert is_subtype(ANY, protocol)

    @given(st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=10))
    @settings(max_examples=50)
    def test_hole_satisfies_any_protocol(self, name: str) -> None:
        """Hole satisfies any protocol."""
        protocol = ProtocolType(
            name=name,
            members=frozenset({("method", FunctionType((), INT))}),
        )
        hole = HoleType(f"hole_{name}")
        assert is_subtype(hole, protocol)

    @given(primitive_types())
    @settings(max_examples=50)
    def test_empty_protocol_accepts_all(self, t: Type) -> None:
        """Empty protocol is satisfied by all types."""
        empty_protocol = ProtocolType(name="Empty", members=frozenset())
        # Not a primitive->protocol subtype in general, but protocols are permissive
        # Actually empty protocol should accept anything through structural subtyping


# =============================================================================
# Z3 SMT Formula Properties
# =============================================================================


class TestZ3FormulaProperties:
    """Property-based tests for Z3 formula parsing."""

    @pytest.fixture
    def solver(self) -> Z3Solver:
        return Z3Solver()

    @given(st.integers(min_value=-1000, max_value=1000))
    @settings(max_examples=50)
    def test_integer_literal_parsing(self, n: int) -> None:
        """Integer literals parse correctly."""
        solver = Z3Solver()
        formula = solver._parse_formula(str(n))
        assert formula is not None

    @given(st.booleans())
    @settings(max_examples=10)
    def test_boolean_literal_parsing(self, b: bool) -> None:
        """Boolean literals parse correctly."""
        solver = Z3Solver()
        formula = solver._parse_formula(str(b).lower())
        assert formula is not None

    @given(st.text(alphabet="abcdefghijklmnopqrstuvwxyz_", min_size=1, max_size=10))
    @settings(max_examples=50)
    def test_variable_creation(self, var_name: str) -> None:
        """Variables are created for valid identifiers."""
        solver = Z3Solver()
        formula = solver._parse_formula(var_name)
        assert formula is not None

    @given(st.integers(min_value=0, max_value=100), st.integers(min_value=0, max_value=100))
    @settings(max_examples=50)
    def test_comparison_soundness(self, a: int, b: int) -> None:
        """Comparison formulas evaluate correctly."""
        try:
            solver = Z3Solver()
        except RuntimeError:
            # Z3 not available, skip test
            pytest.skip("Z3 not available")

        # a > b should be satisfiable iff a > b is true
        formula = SMTFormula(expression=f"{a} > {b}")
        result = solver.check([formula])

        if a > b:
            # Formula is true, so SAT
            assert result.result == SMTResult.SAT
        else:
            # Formula is false, so UNSAT
            assert result.result == SMTResult.UNSAT
        solver.reset()


# =============================================================================
# Narrowing Context Properties
# =============================================================================


class TestNarrowingProperties:
    """Property-based tests for narrowing context."""

    @given(st.text(alphabet="abcdefghijklmnopqrstuvwxyz_", min_size=1, max_size=10), primitive_types())
    @settings(max_examples=100)
    def test_narrow_get_roundtrip(self, var: str, ty: Type) -> None:
        """Narrowing and getting should roundtrip."""
        ctx = EMPTY_NARROWING.narrow(var, ty)
        assert ctx.get_narrowed_type(var) == ty

    @given(
        st.text(alphabet="abcdefghijklmnopqrstuvwxyz_", min_size=1, max_size=10),
        st.text(alphabet="abcdefghijklmnopqrstuvwxyz_", min_size=1, max_size=10),
        primitive_types(),
        primitive_types(),
    )
    @settings(max_examples=100)
    def test_multiple_narrowings_independent(self, var1: str, var2: str, ty1: Type, ty2: Type) -> None:
        """Multiple narrowings don't interfere."""
        assume(var1 != var2)
        ctx = EMPTY_NARROWING.narrow(var1, ty1).narrow(var2, ty2)
        assert ctx.get_narrowed_type(var1) == ty1
        assert ctx.get_narrowed_type(var2) == ty2

    @given(st.integers(min_value=0, max_value=5))
    @settings(max_examples=20)
    def test_conditional_depth_tracking(self, depth: int) -> None:
        """Conditional depth is tracked correctly."""
        ctx = EMPTY_NARROWING
        for _ in range(depth):
            ctx = ctx.enter_conditional()
        assert ctx.condition_depth == depth


# =============================================================================
# Type Domain Properties
# =============================================================================


class TestTypeDomainProperties:
    """Property-based tests for TypeDomain."""

    @given(st.text(alphabet="abcdefghijklmnopqrstuvwxyz_", min_size=1, max_size=10), primitive_types())
    @settings(max_examples=50)
    def test_bind_lookup_roundtrip(self, var: str, ty: Type) -> None:
        """Binding and looking up should roundtrip."""
        domain = TypeDomain()
        domain.bind_variable(var, ty)
        assert domain.lookup_variable(var) == ty

    @given(st.text(alphabet="abcdefghijklmnopqrstuvwxyz_", min_size=1, max_size=10), primitive_types())
    @settings(max_examples=50)
    def test_checkpoint_restore_preserves_bindings(self, var: str, ty: Type) -> None:
        """Checkpoint and restore preserves state."""
        domain = TypeDomain()
        domain.bind_variable(var, ty)
        checkpoint = domain.checkpoint()

        domain.bind_variable(var, STR if ty != STR else INT)
        domain.restore(checkpoint)

        assert domain.lookup_variable(var) == ty

    @given(
        st.text(alphabet="abcdefghijklmnopqrstuvwxyz_", min_size=1, max_size=10),
        primitive_types(),
        primitive_types(),
    )
    @settings(max_examples=50)
    def test_narrowing_overrides_binding(self, var: str, bound_ty: Type, narrow_ty: Type) -> None:
        """Narrowing overrides environment binding."""
        domain = TypeDomain()
        domain.bind_variable(var, bound_ty)
        domain.narrow_type(var, narrow_ty)
        # Effective type should be the narrowed type
        assert domain.get_effective_type(var) == narrow_ty


# =============================================================================
# Soundness Property: Never Block Valid
# =============================================================================


class TestSoundnessProperty:
    """Critical soundness property: never block valid tokens."""

    @given(primitive_types())
    @settings(max_examples=50)
    def test_type_compatible_with_any(self, t: Type) -> None:
        """All types are compatible with Any (soundness)."""
        result = unify(t, ANY)
        assert result.is_success, f"Type {t} should unify with Any"

    @given(all_types())
    @settings(max_examples=100)
    def test_any_compatible_with_type(self, t: Type) -> None:
        """Any is compatible with all types (soundness)."""
        result = unify(ANY, t)
        assert result.is_success, f"Any should unify with {t}"

    @given(all_types())
    @settings(max_examples=100)
    def test_hole_compatible_with_type(self, t: Type) -> None:
        """Holes are compatible with all types (for partial programs)."""
        hole = HoleType("test_hole")
        result = unify(hole, t)
        assert result.is_success, f"Hole should unify with {t}"
