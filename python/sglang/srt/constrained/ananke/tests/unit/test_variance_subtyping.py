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
"""Tests for variance handling, structural subtyping, and flow-sensitive narrowing.

Phase 1.2 enhancements to the type domain:
- Variance (covariance/contravariance) for generics
- ProtocolType for structural subtyping (duck typing)
- Flow-sensitive type narrowing
"""

from __future__ import annotations

import pytest

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
        Substitution,
        EMPTY_SUBSTITUTION,
    )
    from ...domains.types.domain import (
        TypeDomain,
        NarrowingContext,
        EMPTY_NARROWING,
    )
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
        Substitution,
        EMPTY_SUBSTITUTION,
    )
    from domains.types.domain import (
        TypeDomain,
        NarrowingContext,
        EMPTY_NARROWING,
    )


# =============================================================================
# Variance Tests
# =============================================================================


class TestVarianceEnum:
    """Tests for the Variance enum."""

    def test_variance_values_exist(self) -> None:
        """Test all variance values exist."""
        assert Variance.COVARIANT is not None
        assert Variance.CONTRAVARIANT is not None
        assert Variance.INVARIANT is not None
        assert Variance.BIVARIANT is not None


class TestSubtypeChecking:
    """Tests for is_subtype function."""

    def test_same_type_is_subtype(self) -> None:
        """Same type is subtype of itself."""
        assert is_subtype(INT, INT)
        assert is_subtype(STR, STR)
        assert is_subtype(BOOL, BOOL)

    def test_any_is_universal(self) -> None:
        """Any is compatible with everything."""
        assert is_subtype(INT, ANY)
        assert is_subtype(ANY, INT)
        assert is_subtype(STR, ANY)

    def test_never_is_bottom(self) -> None:
        """Never is subtype of everything."""
        assert is_subtype(NEVER, INT)
        assert is_subtype(NEVER, STR)
        assert is_subtype(NEVER, FunctionType((INT,), STR))

    def test_int_subtype_float(self) -> None:
        """int is subtype of float (numeric promotion)."""
        assert is_subtype(INT, FLOAT)
        assert not is_subtype(FLOAT, INT)

    def test_list_covariance(self) -> None:
        """List[int] <: List[int]."""
        list_int = ListType(INT)
        assert is_subtype(list_int, list_int)

    def test_tuple_covariance(self) -> None:
        """Tuple covariance element-wise."""
        tuple_int_str = TupleType((INT, STR))
        tuple_int_str_2 = TupleType((INT, STR))
        assert is_subtype(tuple_int_str, tuple_int_str_2)

    def test_function_variance(self) -> None:
        """Function contravariant in params, covariant in return."""
        # (Animal) -> Cat <: (Cat) -> Animal
        # But without inheritance hierarchy, we test with Any
        fn_any_int = FunctionType((ANY,), INT)
        fn_int_int = FunctionType((INT,), INT)
        # (Any) -> int <: (int) -> int because Any >: int (contravariant in param)
        assert is_subtype(fn_any_int, fn_int_int)

    def test_union_subtype(self) -> None:
        """int <: int | str."""
        union_int_str = UnionType(frozenset({INT, STR}))
        assert is_subtype(INT, union_int_str)
        assert is_subtype(STR, union_int_str)
        assert not is_subtype(BOOL, union_int_str)

    def test_class_nominal_subtyping(self) -> None:
        """Classes use nominal subtyping."""
        class_a = ClassType("MyClass")
        class_b = ClassType("MyClass")
        class_c = ClassType("OtherClass")
        assert is_subtype(class_a, class_b)
        assert not is_subtype(class_a, class_c)


class TestVarianceAwareUnification:
    """Tests for unify_with_variance function."""

    def test_invariant_requires_exact(self) -> None:
        """Invariant variance requires exact type match."""
        result = unify_with_variance(INT, INT, Variance.INVARIANT)
        assert result.is_success

        result = unify_with_variance(INT, STR, Variance.INVARIANT)
        assert result.is_failure

    def test_covariant_accepts_subtype(self) -> None:
        """Covariant accepts subtypes."""
        result = unify_with_variance(INT, FLOAT, Variance.COVARIANT)
        assert result.is_success  # int <: float

    def test_contravariant_accepts_supertype(self) -> None:
        """Contravariant accepts supertypes."""
        result = unify_with_variance(FLOAT, INT, Variance.CONTRAVARIANT)
        # Contravariant: INT <: FLOAT means accept FLOAT when expecting INT
        assert result.is_success

    def test_bivariant_accepts_both(self) -> None:
        """Bivariant accepts both directions."""
        result = unify_with_variance(INT, FLOAT, Variance.BIVARIANT)
        assert result.is_success
        result = unify_with_variance(FLOAT, INT, Variance.BIVARIANT)
        assert result.is_success

    def test_variance_preserves_substitution(self) -> None:
        """Type variables should still be resolved."""
        a = TypeVar("a")
        result = unify_with_variance(a, INT, Variance.COVARIANT)
        assert result.is_success
        assert result.substitution.mapping.get("a") == INT


# =============================================================================
# Protocol Type (Structural Subtyping) Tests
# =============================================================================


class TestProtocolType:
    """Tests for ProtocolType structural subtyping."""

    def test_protocol_creation(self) -> None:
        """Test creating a protocol type."""
        protocol = ProtocolType(
            name="Printable",
            members=frozenset({("__str__", FunctionType((), STR))}),
        )
        assert protocol.name == "Printable"
        assert len(protocol.members) == 1

    def test_protocol_get_member(self) -> None:
        """Test getting protocol members."""
        str_fn = FunctionType((), STR)
        protocol = ProtocolType(
            name="Printable",
            members=frozenset({("__str__", str_fn)}),
        )
        assert protocol.get_member("__str__") == str_fn
        assert protocol.get_member("nonexistent") is None

    def test_any_satisfies_protocol(self) -> None:
        """Any type satisfies any protocol."""
        protocol = ProtocolType(
            name="MyProtocol",
            members=frozenset({("method", FunctionType((), INT))}),
        )
        assert is_subtype(ANY, protocol)

    def test_hole_satisfies_protocol(self) -> None:
        """Hole type satisfies any protocol."""
        protocol = ProtocolType(
            name="MyProtocol",
            members=frozenset({("method", FunctionType((), INT))}),
        )
        hole = HoleType("h1")
        assert is_subtype(hole, protocol)

    def test_protocol_satisfies_protocol_if_superset(self) -> None:
        """Protocol satisfies another if it has all required members."""
        base_protocol = ProtocolType(
            name="Base",
            members=frozenset({("method_a", FunctionType((), INT))}),
        )
        derived_protocol = ProtocolType(
            name="Derived",
            members=frozenset({
                ("method_a", FunctionType((), INT)),
                ("method_b", FunctionType((), STR)),
            }),
        )
        # Derived has all members of Base, so it satisfies Base
        assert is_subtype(derived_protocol, base_protocol)

    def test_protocol_unification(self) -> None:
        """Test unifying protocol types."""
        protocol_a = ProtocolType(
            name="Proto",
            members=frozenset({("x", INT)}),
        )
        protocol_b = ProtocolType(
            name="Proto",
            members=frozenset({("x", INT)}),
        )
        result = unify(protocol_a, protocol_b)
        assert result.is_success

    def test_callable_satisfies_single_call_protocol(self) -> None:
        """Function satisfies protocol with only __call__."""
        callable_protocol = ProtocolType(
            name="Callable",
            members=frozenset({("__call__", FunctionType((INT,), STR))}),
        )
        fn = FunctionType((INT,), STR)
        assert is_subtype(fn, callable_protocol)


# =============================================================================
# Flow-Sensitive Narrowing Tests
# =============================================================================


class TestNarrowingContext:
    """Tests for NarrowingContext."""

    def test_empty_narrowing(self) -> None:
        """Test empty narrowing context."""
        assert EMPTY_NARROWING.condition_depth == 0
        assert len(EMPTY_NARROWING.narrowings) == 0

    def test_narrow_type(self) -> None:
        """Test narrowing a type."""
        ctx = EMPTY_NARROWING.narrow("x", INT)
        assert ctx.get_narrowed_type("x") == INT
        assert ctx.get_narrowed_type("y") is None

    def test_multiple_narrowings(self) -> None:
        """Test multiple narrowings."""
        ctx = EMPTY_NARROWING.narrow("x", INT).narrow("y", STR)
        assert ctx.get_narrowed_type("x") == INT
        assert ctx.get_narrowed_type("y") == STR

    def test_enter_conditional(self) -> None:
        """Test entering a conditional branch."""
        ctx = EMPTY_NARROWING.enter_conditional()
        assert ctx.condition_depth == 1

    def test_exit_conditional_clears_narrowings(self) -> None:
        """Test exiting conditional clears narrowings."""
        ctx = EMPTY_NARROWING.narrow("x", INT).enter_conditional()
        ctx = ctx.exit_conditional()
        # Narrowings cleared on exit
        assert len(ctx.narrowings) == 0
        assert ctx.condition_depth == 0


class TestTypeDomainNarrowing:
    """Tests for TypeDomain narrowing methods."""

    def test_narrow_type_method(self) -> None:
        """Test TypeDomain.narrow_type."""
        domain = TypeDomain()
        domain.narrow_type("x", INT)
        assert domain.get_effective_type("x") == INT

    def test_narrowing_overrides_environment(self) -> None:
        """Test that narrowing overrides environment type."""
        domain = TypeDomain()
        domain.bind_variable("x", ANY)
        assert domain.get_effective_type("x") == ANY

        domain.narrow_type("x", INT)
        assert domain.get_effective_type("x") == INT

    def test_enter_exit_conditional(self) -> None:
        """Test entering and exiting conditional branches."""
        domain = TypeDomain()
        domain.narrow_type("x", INT)
        domain.enter_conditional_branch()
        # Narrowing still applies in branch
        assert domain.get_effective_type("x") == INT

        domain.exit_conditional_branch()
        # Narrowing cleared after exit
        assert domain.get_effective_type("x") is None

    def test_checkpoint_preserves_narrowing(self) -> None:
        """Test that checkpoint preserves narrowing context."""
        domain = TypeDomain()
        domain.narrow_type("x", INT)

        checkpoint = domain.checkpoint()

        domain.narrow_type("x", STR)
        assert domain.get_effective_type("x") == STR

        domain.restore(checkpoint)
        assert domain.get_effective_type("x") == INT


# =============================================================================
# Integration Tests
# =============================================================================


class TestVarianceIntegration:
    """Integration tests for variance with type domain."""

    def test_function_type_unification_with_variance(self) -> None:
        """Test that function types unify with proper variance."""
        # (int) -> 'a unifies with (int) -> str
        a = TypeVar("a")
        fn1 = FunctionType((INT,), a)
        fn2 = FunctionType((INT,), STR)
        result = unify(fn1, fn2)
        assert result.is_success
        assert result.substitution.mapping.get("a") == STR

    def test_function_arity_mismatch_fails(self) -> None:
        """Test that different arities don't unify."""
        fn1 = FunctionType((INT,), BOOL)
        fn2 = FunctionType((INT, STR), BOOL)
        result = unify(fn1, fn2)
        assert result.is_failure

    def test_list_element_unification(self) -> None:
        """Test list element type unification."""
        a = TypeVar("a")
        list_a = ListType(a)
        list_int = ListType(INT)
        result = unify(list_a, list_int)
        assert result.is_success
        assert result.substitution.mapping.get("a") == INT


class TestProtocolIntegration:
    """Integration tests for protocols with type domain."""

    def test_protocol_as_expected_type(self) -> None:
        """Test using protocol as expected type."""
        protocol = ProtocolType(
            name="Sized",
            members=frozenset({("__len__", FunctionType((), INT))}),
        )
        # Any should be assignable to protocol
        result = unify(ANY, protocol)
        assert result.is_success

    def test_class_satisfies_protocol_permissive(self) -> None:
        """Classes are permissively assumed to satisfy protocols."""
        protocol = ProtocolType(
            name="MyProtocol",
            members=frozenset({("method", FunctionType((), INT))}),
        )
        my_class = ClassType("MyClass")
        # Without explicit member info, be permissive
        assert is_subtype(my_class, protocol)
