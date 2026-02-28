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
"""Unit tests for type unification algorithm.

Tests Robinson's unification algorithm with occurs check for all type forms.
"""

import pytest

from domains.types.constraint import (
    ANY,
    BOOL,
    FLOAT,
    INT,
    NEVER,
    STR,
    AnyType,
    ClassType,
    DictType,
    FunctionType,
    HoleType,
    ListType,
    NeverType,
    SetType,
    TupleType,
    Type,
    TypeEquation,
    TypeVar,
    UnionType,
)
from domains.types.unification import (
    EMPTY_SUBSTITUTION,
    Substitution,
    UnificationError,
    UnificationResult,
    occurs_check,
    solve_equations,
    unify,
    unify_types,
)


class TestSubstitution:
    """Tests for the Substitution class."""

    def test_empty_substitution(self):
        """Empty substitution should have no mappings."""
        assert EMPTY_SUBSTITUTION.mapping == {}

    def test_apply_to_type_var(self):
        """Applying substitution should replace type variables."""
        subst = Substitution({"T": INT})
        var = TypeVar("T")
        result = subst.apply(var)
        assert result == INT

    def test_apply_to_unknown_var(self):
        """Applying substitution to unknown var should return the var."""
        subst = Substitution({"T": INT})
        var = TypeVar("U")
        result = subst.apply(var)
        assert result == var

    def test_apply_to_primitive(self):
        """Applying substitution to primitive should return unchanged."""
        subst = Substitution({"T": INT})
        result = subst.apply(STR)
        assert result == STR

    def test_apply_to_list_type(self):
        """Applying substitution should work recursively on ListType."""
        subst = Substitution({"T": INT})
        list_t = ListType(TypeVar("T"))
        result = subst.apply(list_t)
        assert result == ListType(INT)

    def test_apply_to_function_type(self):
        """Applying substitution should work on FunctionType."""
        subst = Substitution({"T": INT, "U": STR})
        func_t = FunctionType((TypeVar("T"),), TypeVar("U"))
        result = subst.apply(func_t)
        assert result == FunctionType((INT,), STR)

    def test_compose_substitutions(self):
        """Composing substitutions should work correctly."""
        subst1 = Substitution({"T": INT})
        subst2 = Substitution({"U": TypeVar("T")})
        composed = subst1.compose(subst2)
        # Applying composed should give U -> INT
        var_u = TypeVar("U")
        result = composed.apply(var_u)
        assert result == INT

    def test_extend_substitution(self):
        """Extending substitution should add new binding."""
        subst = Substitution({"T": INT})
        extended = subst.extend("U", STR)
        assert extended.mapping == {"T": INT, "U": STR}

    def test_apply_to_equation(self):
        """Applying substitution to equation should work on both sides."""
        subst = Substitution({"T": INT})
        eq = TypeEquation(TypeVar("T"), ListType(TypeVar("T")))
        result = subst.apply_to_equation(eq)
        assert result.lhs == INT
        assert result.rhs == ListType(INT)


class TestOccursCheck:
    """Tests for the occurs check function."""

    def test_occurs_in_same_var(self):
        """Variable occurs in itself."""
        var = TypeVar("T")
        assert occurs_check(var, var)

    def test_not_occurs_in_different_var(self):
        """Variable doesn't occur in different variable."""
        var_t = TypeVar("T")
        var_u = TypeVar("U")
        assert not occurs_check(var_t, var_u)

    def test_occurs_in_list(self):
        """Variable occurs in List containing it."""
        var = TypeVar("T")
        list_t = ListType(var)
        assert occurs_check(var, list_t)

    def test_not_occurs_in_list(self):
        """Variable doesn't occur in List not containing it."""
        var_t = TypeVar("T")
        list_u = ListType(TypeVar("U"))
        assert not occurs_check(var_t, list_u)

    def test_occurs_in_function_param(self):
        """Variable occurs in function parameter."""
        var = TypeVar("T")
        func = FunctionType((var,), INT)
        assert occurs_check(var, func)

    def test_occurs_in_function_return(self):
        """Variable occurs in function return type."""
        var = TypeVar("T")
        func = FunctionType((INT,), var)
        assert occurs_check(var, func)

    def test_occurs_in_dict_key(self):
        """Variable occurs in dict key type."""
        var = TypeVar("T")
        dict_t = DictType(var, INT)
        assert occurs_check(var, dict_t)

    def test_occurs_in_dict_value(self):
        """Variable occurs in dict value type."""
        var = TypeVar("T")
        dict_t = DictType(STR, var)
        assert occurs_check(var, dict_t)

    def test_occurs_in_tuple(self):
        """Variable occurs in tuple element."""
        var = TypeVar("T")
        tuple_t = TupleType((INT, var, STR))
        assert occurs_check(var, tuple_t)

    def test_occurs_in_union(self):
        """Variable occurs in union member."""
        var = TypeVar("T")
        union_t = UnionType(frozenset({INT, var}))
        assert occurs_check(var, union_t)

    def test_occurs_in_class_type_arg(self):
        """Variable occurs in class type argument."""
        var = TypeVar("T")
        class_t = ClassType("MyClass", (var,))
        assert occurs_check(var, class_t)

    def test_not_occurs_in_primitive(self):
        """Variable doesn't occur in primitive types."""
        var = TypeVar("T")
        assert not occurs_check(var, INT)
        assert not occurs_check(var, STR)
        assert not occurs_check(var, BOOL)

    def test_not_occurs_in_any(self):
        """Variable doesn't occur in Any."""
        var = TypeVar("T")
        assert not occurs_check(var, ANY)

    def test_not_occurs_in_never(self):
        """Variable doesn't occur in Never."""
        var = TypeVar("T")
        assert not occurs_check(var, NEVER)


class TestUnify:
    """Tests for the unify function."""

    def test_unify_same_types(self):
        """Same types should unify trivially."""
        result = unify(INT, INT)
        assert result.is_success
        assert result.substitution.mapping == {}

    def test_unify_type_var_with_concrete(self):
        """Type variable should unify with concrete type."""
        result = unify(TypeVar("T"), INT)
        assert result.is_success
        assert result.substitution.mapping == {"T": INT}

    def test_unify_concrete_with_type_var(self):
        """Concrete type should unify with type variable."""
        result = unify(INT, TypeVar("T"))
        assert result.is_success
        assert result.substitution.mapping == {"T": INT}

    def test_unify_any_with_anything(self):
        """Any should unify with anything."""
        result = unify(ANY, INT)
        assert result.is_success
        result = unify(INT, ANY)
        assert result.is_success
        result = unify(ANY, ListType(STR))
        assert result.is_success

    def test_unify_never_with_anything(self):
        """Never should unify with anything."""
        result = unify(NEVER, INT)
        assert result.is_success
        result = unify(INT, NEVER)
        assert result.is_success

    def test_unify_hole_with_anything(self):
        """HoleType should unify with anything."""
        hole = HoleType("h1")
        result = unify(hole, INT)
        assert result.is_success
        result = unify(INT, hole)
        assert result.is_success

    def test_unify_list_types(self):
        """List types should unify element-wise."""
        result = unify(ListType(TypeVar("T")), ListType(INT))
        assert result.is_success
        assert result.substitution.mapping == {"T": INT}

    def test_unify_set_types(self):
        """Set types should unify element-wise."""
        result = unify(SetType(TypeVar("T")), SetType(STR))
        assert result.is_success
        assert result.substitution.mapping == {"T": STR}

    def test_unify_dict_types(self):
        """Dict types should unify key and value."""
        result = unify(
            DictType(TypeVar("K"), TypeVar("V")),
            DictType(STR, INT),
        )
        assert result.is_success
        assert result.substitution.mapping == {"K": STR, "V": INT}

    def test_unify_tuple_types(self):
        """Tuple types should unify element-wise."""
        result = unify(
            TupleType((TypeVar("T"), TypeVar("U"))),
            TupleType((INT, STR)),
        )
        assert result.is_success
        assert result.substitution.mapping == {"T": INT, "U": STR}

    def test_unify_function_types(self):
        """Function types should unify params and return."""
        result = unify(
            FunctionType((TypeVar("A"),), TypeVar("B")),
            FunctionType((INT,), STR),
        )
        assert result.is_success
        assert result.substitution.mapping == {"A": INT, "B": STR}

    def test_unify_class_types_same_name(self):
        """Class types with same name should unify type args."""
        result = unify(
            ClassType("Container", (TypeVar("T"),)),
            ClassType("Container", (INT,)),
        )
        assert result.is_success
        assert result.substitution.mapping == {"T": INT}

    def test_unify_union_types_same_members(self):
        """Union types with same members should unify."""
        result = unify(
            UnionType(frozenset({INT, STR})),
            UnionType(frozenset({INT, STR})),
        )
        assert result.is_success

    def test_fail_unify_different_primitives(self):
        """Different primitives should not unify."""
        result = unify(INT, STR)
        assert result.is_failure
        assert "incompatible types" in result.error.reason

    def test_fail_unify_function_arity_mismatch(self):
        """Functions with different arity should not unify."""
        result = unify(
            FunctionType((INT,), STR),
            FunctionType((INT, INT), STR),
        )
        assert result.is_failure
        assert "arity" in result.error.reason.lower()

    def test_fail_unify_tuple_length_mismatch(self):
        """Tuples with different lengths should not unify."""
        result = unify(
            TupleType((INT, STR)),
            TupleType((INT, STR, BOOL)),
        )
        assert result.is_failure
        assert "length" in result.error.reason.lower()

    def test_fail_unify_class_name_mismatch(self):
        """Classes with different names should not unify."""
        result = unify(
            ClassType("Foo", ()),
            ClassType("Bar", ()),
        )
        assert result.is_failure
        assert "name" in result.error.reason.lower()

    def test_fail_occurs_check(self):
        """Infinite type should fail occurs check."""
        var = TypeVar("T")
        result = unify(var, ListType(var))
        assert result.is_failure
        assert "infinite" in result.error.reason.lower()

    def test_unify_nested_types(self):
        """Nested types should unify correctly."""
        result = unify(
            ListType(DictType(TypeVar("K"), TypeVar("V"))),
            ListType(DictType(STR, INT)),
        )
        assert result.is_success
        assert result.substitution.mapping == {"K": STR, "V": INT}


class TestSolveEquations:
    """Tests for solving systems of type equations."""

    def test_solve_empty(self):
        """Empty equations should produce empty substitution."""
        result = solve_equations([])
        assert result.is_success
        assert result.substitution.mapping == {}

    def test_solve_single_equation(self):
        """Single equation should be solved."""
        eq = TypeEquation(TypeVar("T"), INT)
        result = solve_equations([eq])
        assert result.is_success
        assert result.substitution.mapping == {"T": INT}

    def test_solve_multiple_equations(self):
        """Multiple equations should be solved together."""
        eqs = [
            TypeEquation(TypeVar("T"), INT),
            TypeEquation(TypeVar("U"), STR),
        ]
        result = solve_equations(eqs)
        assert result.is_success
        assert result.substitution.mapping == {"T": INT, "U": STR}

    def test_solve_dependent_equations(self):
        """Equations that depend on each other should work."""
        eqs = [
            TypeEquation(TypeVar("T"), TypeVar("U")),
            TypeEquation(TypeVar("U"), INT),
        ]
        result = solve_equations(eqs)
        assert result.is_success
        # T should resolve to INT through U
        assert result.substitution.apply(TypeVar("T")) == INT

    def test_solve_contradictory_equations(self):
        """Contradictory equations should fail."""
        eqs = [
            TypeEquation(TypeVar("T"), INT),
            TypeEquation(TypeVar("T"), STR),
        ]
        result = solve_equations(eqs)
        assert result.is_failure


class TestUnifyTypes:
    """Tests for unifying a list of types."""

    def test_unify_empty_list(self):
        """Empty list should produce empty substitution."""
        result = unify_types([])
        assert result.is_success

    def test_unify_single_type(self):
        """Single type should produce empty substitution."""
        result = unify_types([INT])
        assert result.is_success

    def test_unify_same_types(self):
        """List of same types should unify."""
        result = unify_types([INT, INT, INT])
        assert result.is_success

    def test_unify_with_type_vars(self):
        """Type variables should unify with concrete types."""
        result = unify_types([TypeVar("T"), INT, TypeVar("T")])
        assert result.is_success
        assert result.substitution.apply(TypeVar("T")) == INT

    def test_unify_incompatible_types(self):
        """Incompatible types should fail."""
        result = unify_types([INT, STR])
        assert result.is_failure


class TestUnificationResult:
    """Tests for UnificationResult dataclass."""

    def test_success_result(self):
        """Success result should have substitution."""
        result = UnificationResult.success(Substitution({"T": INT}))
        assert result.is_success
        assert not result.is_failure
        assert result.substitution.mapping == {"T": INT}

    def test_failure_result(self):
        """Failure result should have error."""
        result = UnificationResult.failure(INT, STR, "mismatch")
        assert result.is_failure
        assert not result.is_success
        assert result.error.lhs == INT
        assert result.error.rhs == STR
        assert result.error.reason == "mismatch"


class TestUnificationError:
    """Tests for UnificationError dataclass."""

    def test_error_repr(self):
        """Error should have readable repr."""
        error = UnificationError(INT, STR, "type mismatch")
        repr_str = repr(error)
        assert "INT" in repr_str or "int" in repr_str.lower()
        assert "STR" in repr_str or "str" in repr_str.lower()
        assert "type mismatch" in repr_str
