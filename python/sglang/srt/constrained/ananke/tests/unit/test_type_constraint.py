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
"""Unit tests for type domain constraint and types."""

import pytest
import torch

from core.constraint import Satisfiability
from core.domain import GenerationContext
from domains.types.constraint import (
    # Type hierarchy
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
    # Singletons
    INT,
    STR,
    BOOL,
    FLOAT,
    NONE,
    ANY,
    NEVER,
    # Constraint
    TypeConstraint,
    TYPE_TOP,
    TYPE_BOTTOM,
    TypeEquation,
    type_expecting,
    type_with_equation,
    OptionalType,
)
from domains.types.domain import TypeDomain, TypeDomainCheckpoint
from domains.types.unification import (
    Substitution,
    EMPTY_SUBSTITUTION,
    unify,
    occurs_check,
    solve_equations,
    unify_types,
    UnificationResult,
)
from domains.types.environment import (
    TypeEnvironment,
    EMPTY_ENVIRONMENT,
    create_environment,
    merge_environments,
)


class TestTypeHierarchy:
    """Tests for the type hierarchy."""

    def test_primitive_types(self):
        """Primitive types have no free type vars."""
        assert INT.free_type_vars() == frozenset()
        assert STR.free_type_vars() == frozenset()
        assert BOOL.free_type_vars() == frozenset()

    def test_type_var(self):
        """TypeVar tracks its name."""
        a = TypeVar("a")
        assert a.name == "a"
        assert a.free_type_vars() == frozenset({"a"})

    def test_function_type(self):
        """FunctionType tracks params and returns."""
        fn = FunctionType((INT, STR), BOOL)
        assert fn.params == (INT, STR)
        assert fn.returns == BOOL
        assert fn.free_type_vars() == frozenset()

    def test_function_type_with_vars(self):
        """FunctionType with type variables."""
        a = TypeVar("a")
        b = TypeVar("b")
        fn = FunctionType((a, b), a)
        assert fn.free_type_vars() == frozenset({"a", "b"})

    def test_list_type(self):
        """ListType wraps element type."""
        lst = ListType(INT)
        assert lst.element == INT
        assert repr(lst) == "List[int]"

    def test_dict_type(self):
        """DictType has key and value types."""
        d = DictType(STR, INT)
        assert d.key == STR
        assert d.value == INT
        assert repr(d) == "Dict[str, int]"

    def test_tuple_type(self):
        """TupleType has fixed element types."""
        t = TupleType((INT, STR, BOOL))
        assert t.elements == (INT, STR, BOOL)
        assert repr(t) == "Tuple[int, str, bool]"

    def test_union_type(self):
        """UnionType holds multiple types."""
        u = UnionType(frozenset({INT, STR}))
        assert INT in u.members
        assert STR in u.members

    def test_optional_type(self):
        """OptionalType is Union with None."""
        opt = OptionalType(INT)
        assert INT in opt.members
        assert NONE in opt.members

    def test_class_type(self):
        """ClassType with generic params."""
        cls = ClassType("MyClass", (INT, STR))
        assert cls.name == "MyClass"
        assert cls.type_args == (INT, STR)
        assert repr(cls) == "MyClass[int, str]"

    def test_any_type(self):
        """AnyType is the top type."""
        assert repr(ANY) == "Any"
        assert ANY.free_type_vars() == frozenset()

    def test_never_type(self):
        """NeverType is the bottom type."""
        assert repr(NEVER) == "Never"
        assert NEVER.free_type_vars() == frozenset()

    def test_hole_type(self):
        """HoleType with expected type."""
        h = HoleType("h1", INT)
        assert h.hole_id == "h1"
        assert h.expected == INT
        assert repr(h) == "?h1:int"


class TestTypeSubstitution:
    """Tests for type substitution."""

    def test_substitute_primitive(self):
        """Substitution on primitives does nothing."""
        subst = {"a": INT}
        assert INT.substitute(subst) == INT

    def test_substitute_type_var(self):
        """Substitution replaces type variable."""
        a = TypeVar("a")
        subst = {"a": INT}
        assert a.substitute(subst) == INT

    def test_substitute_unbound_var(self):
        """Unbound variable unchanged."""
        a = TypeVar("a")
        subst = {"b": INT}
        assert a.substitute(subst) == a

    def test_substitute_function_type(self):
        """Substitution applies recursively to functions."""
        a = TypeVar("a")
        fn = FunctionType((a,), a)
        subst = {"a": INT}
        result = fn.substitute(subst)
        assert result == FunctionType((INT,), INT)

    def test_substitute_list_type(self):
        """Substitution applies to list element."""
        a = TypeVar("a")
        lst = ListType(a)
        subst = {"a": INT}
        assert lst.substitute(subst) == ListType(INT)


class TestUnification:
    """Tests for type unification."""

    def test_unify_same_type(self):
        """Same types unify trivially."""
        result = unify(INT, INT)
        assert result.is_success
        assert result.substitution.mapping == {}

    def test_unify_type_var(self):
        """Type variable unifies with concrete type."""
        a = TypeVar("a")
        result = unify(a, INT)
        assert result.is_success
        assert result.substitution.mapping == {"a": INT}

    def test_unify_type_var_reverse(self):
        """Type variable on right side."""
        a = TypeVar("a")
        result = unify(INT, a)
        assert result.is_success
        assert result.substitution.mapping == {"a": INT}

    def test_unify_any(self):
        """Any unifies with anything."""
        result = unify(ANY, INT)
        assert result.is_success
        result2 = unify(INT, ANY)
        assert result2.is_success

    def test_unify_never(self):
        """Never unifies with anything."""
        result = unify(NEVER, INT)
        assert result.is_success

    def test_unify_hole(self):
        """Hole unifies with anything."""
        h = HoleType("h1")
        result = unify(h, INT)
        assert result.is_success

    def test_unify_function_types(self):
        """Function types unify structurally."""
        a = TypeVar("a")
        fn1 = FunctionType((INT,), a)
        fn2 = FunctionType((INT,), STR)
        result = unify(fn1, fn2)
        assert result.is_success
        assert result.substitution.mapping == {"a": STR}

    def test_unify_function_arity_mismatch(self):
        """Different arities don't unify."""
        fn1 = FunctionType((INT,), BOOL)
        fn2 = FunctionType((INT, STR), BOOL)
        result = unify(fn1, fn2)
        assert result.is_failure

    def test_unify_list_types(self):
        """List types unify element-wise."""
        a = TypeVar("a")
        lst1 = ListType(a)
        lst2 = ListType(INT)
        result = unify(lst1, lst2)
        assert result.is_success
        assert result.substitution.mapping == {"a": INT}

    def test_unify_incompatible(self):
        """Incompatible types fail."""
        result = unify(INT, STR)
        assert result.is_failure

    def test_occurs_check(self):
        """Occurs check prevents infinite types."""
        a = TypeVar("a")
        lst = ListType(a)
        result = unify(a, lst)
        assert result.is_failure
        assert "infinite type" in result.error.reason

    def test_solve_equations(self):
        """Solve multiple equations."""
        a = TypeVar("a")
        b = TypeVar("b")
        eqs = [
            TypeEquation(a, INT),
            TypeEquation(b, STR),
        ]
        result = solve_equations(eqs)
        assert result.is_success
        assert result.substitution.mapping["a"] == INT
        assert result.substitution.mapping["b"] == STR

    def test_unify_types_list(self):
        """Unify a list of types."""
        result = unify_types([INT, INT, INT])
        assert result.is_success


class TestTypeEnvironment:
    """Tests for type environment."""

    def test_empty_environment(self):
        """Empty environment has no bindings."""
        env = EMPTY_ENVIRONMENT
        assert env.lookup("x") is None
        assert len(env) == 0

    def test_bind_lookup(self):
        """Can bind and lookup variables."""
        env = EMPTY_ENVIRONMENT.bind("x", INT)
        assert env.lookup("x") == INT

    def test_bind_many(self):
        """Can bind multiple variables."""
        env = EMPTY_ENVIRONMENT.bind_many({"x": INT, "y": STR})
        assert env.lookup("x") == INT
        assert env.lookup("y") == STR

    def test_push_pop_scope(self):
        """Nested scopes work correctly."""
        env = EMPTY_ENVIRONMENT.bind("x", INT)
        inner = env.push_scope().bind("y", STR)

        assert inner.lookup("x") == INT
        assert inner.lookup("y") == STR

        outer = inner.pop_scope()
        assert outer.lookup("x") == INT
        assert outer.lookup("y") is None

    def test_shadowing(self):
        """Inner bindings shadow outer."""
        env = EMPTY_ENVIRONMENT.bind("x", INT)
        inner = env.push_scope().bind("x", STR)

        assert inner.lookup("x") == STR
        assert env.lookup("x") == INT

    def test_all_bindings(self):
        """Get all bindings including parent scopes."""
        env = EMPTY_ENVIRONMENT.bind("x", INT).push_scope().bind("y", STR)
        all_bindings = env.all_bindings()
        assert all_bindings == {"x": INT, "y": STR}

    def test_names(self):
        """Get all bound names."""
        env = EMPTY_ENVIRONMENT.bind("x", INT).bind("y", STR)
        assert env.names() == frozenset({"x", "y"})

    def test_snapshot_restore(self):
        """Can snapshot and restore environment."""
        env = EMPTY_ENVIRONMENT.bind("x", INT)
        snapshot = env.snapshot()

        restored = TypeEnvironment.from_snapshot(snapshot)
        assert restored.lookup("x") == INT

    def test_create_environment(self):
        """Factory function creates environment."""
        env = create_environment({"x": INT, "y": STR})
        assert env.lookup("x") == INT
        assert env.lookup("y") == STR

    def test_merge_environments(self):
        """Merge two environments."""
        env1 = create_environment({"x": INT})
        env2 = create_environment({"y": STR})
        merged = merge_environments(env1, env2)
        assert merged.lookup("x") == INT
        assert merged.lookup("y") == STR


class TestTypeConstraint:
    """Tests for TypeConstraint."""

    def test_type_top_is_top(self):
        """TYPE_TOP is the top element."""
        assert TYPE_TOP.is_top()
        assert not TYPE_TOP.is_bottom()

    def test_type_bottom_is_bottom(self):
        """TYPE_BOTTOM is the bottom element."""
        assert TYPE_BOTTOM.is_bottom()
        assert not TYPE_BOTTOM.is_top()

    def test_type_top_satisfiability(self):
        """TYPE_TOP is satisfiable."""
        assert TYPE_TOP.satisfiability() == Satisfiability.SAT

    def test_type_bottom_satisfiability(self):
        """TYPE_BOTTOM is unsatisfiable."""
        assert TYPE_BOTTOM.satisfiability() == Satisfiability.UNSAT

    def test_type_expecting(self):
        """Create constraint expecting a type."""
        c = type_expecting(INT)
        assert c.expected_type == INT
        assert not c.is_top()
        assert not c.is_bottom()

    def test_type_with_equation(self):
        """Create constraint with equation."""
        a = TypeVar("a")
        c = type_with_equation(a, INT)
        assert len(c.equations) == 1

    def test_meet_identity_top(self):
        """c ⊓ ⊤ = c (identity law)."""
        c = type_expecting(INT)
        result = c.meet(TYPE_TOP)
        assert result.expected_type == INT

    def test_meet_identity_top_reverse(self):
        """⊤ ⊓ c = c (identity law, reversed)."""
        c = type_expecting(INT)
        result = TYPE_TOP.meet(c)
        assert result.expected_type == INT

    def test_meet_annihilation_bottom(self):
        """c ⊓ ⊥ = ⊥ (annihilation law)."""
        c = type_expecting(INT)
        result = c.meet(TYPE_BOTTOM)
        assert result == TYPE_BOTTOM

    def test_meet_annihilation_bottom_reverse(self):
        """⊥ ⊓ c = ⊥ (annihilation law, reversed)."""
        c = type_expecting(INT)
        result = TYPE_BOTTOM.meet(c)
        assert result == TYPE_BOTTOM

    def test_meet_idempotence(self):
        """c ⊓ c = c (idempotence)."""
        c = type_expecting(INT)
        result = c.meet(c)
        assert result.expected_type == INT

    def test_meet_same_expected(self):
        """Meet with same expected type."""
        c1 = type_expecting(INT)
        c2 = type_expecting(INT)
        result = c1.meet(c2)
        assert result.expected_type == INT

    def test_meet_different_expected(self):
        """Meet with incompatible expected types returns BOTTOM."""
        c1 = type_expecting(INT)
        c2 = type_expecting(STR)
        result = c1.meet(c2)
        assert result == TYPE_BOTTOM

    def test_meet_combines_equations(self):
        """Meet combines equations."""
        a = TypeVar("a")
        b = TypeVar("b")
        c1 = type_with_equation(a, INT)
        c2 = type_with_equation(b, STR)
        result = c1.meet(c2)
        assert len(result.equations) == 2

    def test_with_expected_type(self):
        """Can update expected type."""
        c = type_expecting(INT)
        c2 = c.with_expected_type(STR)
        assert c2.expected_type == STR

    def test_with_equation(self):
        """Can add equation."""
        c = type_expecting(INT)
        a = TypeVar("a")
        c2 = c.with_equation(TypeEquation(a, INT))
        assert len(c2.equations) == 1

    def test_with_environment_hash(self):
        """Can update environment hash."""
        c = type_expecting(INT)
        c2 = c.with_environment_hash(42)
        assert c2.environment_hash == 42


class TestTypeDomain:
    """Tests for TypeDomain."""

    def test_domain_name(self):
        """Domain name is 'types'."""
        domain = TypeDomain()
        assert domain.name == "types"

    def test_domain_top_bottom(self):
        """Domain has correct TOP and BOTTOM."""
        domain = TypeDomain()
        assert domain.top == TYPE_TOP
        assert domain.bottom == TYPE_BOTTOM

    def test_token_mask_all_true(self):
        """Token mask returns all-True (placeholder)."""
        domain = TypeDomain()
        ctx = GenerationContext(vocab_size=100, device="cpu")
        constraint = type_expecting(INT)

        mask = domain.token_mask(constraint, ctx)

        assert mask.shape == (100,)
        assert mask.dtype == torch.bool
        assert mask.all()

    def test_observe_token_updates_state(self):
        """observe_token updates environment hash."""
        domain = TypeDomain()
        ctx = GenerationContext(vocab_size=100)
        constraint = type_expecting(INT)
        initial_hash = constraint.environment_hash

        updated = domain.observe_token(constraint, token_id=42, context=ctx)

        assert updated.environment_hash != initial_hash

    def test_observe_token_bottom_stays_bottom(self):
        """observe_token on BOTTOM returns BOTTOM."""
        domain = TypeDomain()
        ctx = GenerationContext(vocab_size=100)

        result = domain.observe_token(TYPE_BOTTOM, token_id=42, context=ctx)

        assert result == TYPE_BOTTOM

    def test_observe_token_top_stays_top(self):
        """observe_token on TOP returns TOP."""
        domain = TypeDomain()
        ctx = GenerationContext(vocab_size=100)

        result = domain.observe_token(TYPE_TOP, token_id=42, context=ctx)

        assert result == TYPE_TOP

    def test_checkpoint_restore(self):
        """Can checkpoint and restore state."""
        domain = TypeDomain()
        domain.bind_variable("x", INT)
        domain.set_expected_type(STR)

        cp = domain.checkpoint()

        # Modify state
        domain.bind_variable("y", BOOL)
        domain.set_expected_type(FLOAT)

        # Restore
        domain.restore(cp)

        assert domain.lookup_variable("x") == INT
        assert domain.lookup_variable("y") is None
        assert domain.expected_type == STR

    def test_satisfiability_delegates(self):
        """satisfiability delegates to constraint."""
        domain = TypeDomain()

        assert domain.satisfiability(TYPE_TOP) == Satisfiability.SAT
        assert domain.satisfiability(TYPE_BOTTOM) == Satisfiability.UNSAT

    def test_bind_lookup_variable(self):
        """Can bind and lookup variables."""
        domain = TypeDomain()
        domain.bind_variable("x", INT)

        assert domain.lookup_variable("x") == INT
        assert domain.lookup_variable("y") is None

    def test_push_pop_scope(self):
        """Can push and pop scopes."""
        domain = TypeDomain()
        domain.bind_variable("x", INT)
        domain.push_scope()
        domain.bind_variable("y", STR)

        assert domain.lookup_variable("x") == INT
        assert domain.lookup_variable("y") == STR

        domain.pop_scope()

        assert domain.lookup_variable("x") == INT
        assert domain.lookup_variable("y") is None

    def test_create_constraint(self):
        """Can create constraints."""
        domain = TypeDomain()

        c1 = domain.create_constraint()
        assert c1 == TYPE_TOP

        c2 = domain.create_constraint(INT)
        assert c2.expected_type == INT


class TestTypeConstraintRepr:
    """Tests for string representation."""

    def test_repr_top(self):
        """TYPE_TOP has correct repr."""
        assert repr(TYPE_TOP) == "TYPE_TOP"

    def test_repr_bottom(self):
        """TYPE_BOTTOM has correct repr."""
        assert repr(TYPE_BOTTOM) == "TYPE_BOTTOM"

    def test_repr_constraint(self):
        """Regular constraint has informative repr."""
        c = type_expecting(INT)
        r = repr(c)
        assert "TypeConstraint" in r
        assert "expected" in r


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
