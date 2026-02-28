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
"""Unit tests for type environment.

Tests for TypeEnvironment immutable mapping with scope support.
"""

import pytest

from domains.types.constraint import INT, STR, BOOL, FLOAT, FunctionType
from domains.types.environment import (
    EMPTY_ENVIRONMENT,
    TypeEnvironment,
    TypeEnvironmentSnapshot,
    create_environment,
    merge_environments,
)


class TestTypeEnvironment:
    """Tests for TypeEnvironment class."""

    def test_empty_environment(self):
        """Empty environment should have no bindings."""
        env = EMPTY_ENVIRONMENT
        assert len(env) == 0
        assert env.lookup("x") is None

    def test_bind_single(self):
        """Binding a variable should create new environment."""
        env = EMPTY_ENVIRONMENT.bind("x", INT)
        assert env.lookup("x") == INT
        # Original should be unchanged
        assert EMPTY_ENVIRONMENT.lookup("x") is None

    def test_bind_multiple(self):
        """Multiple bindings should accumulate."""
        env = EMPTY_ENVIRONMENT
        env = env.bind("x", INT)
        env = env.bind("y", STR)
        env = env.bind("z", BOOL)
        assert env.lookup("x") == INT
        assert env.lookup("y") == STR
        assert env.lookup("z") == BOOL

    def test_bind_shadow(self):
        """Later binding should shadow earlier."""
        env = EMPTY_ENVIRONMENT.bind("x", INT)
        env = env.bind("x", STR)
        assert env.lookup("x") == STR

    def test_bind_many(self):
        """bind_many should add multiple bindings at once."""
        env = EMPTY_ENVIRONMENT.bind_many({
            "x": INT,
            "y": STR,
            "z": BOOL,
        })
        assert env.lookup("x") == INT
        assert env.lookup("y") == STR
        assert env.lookup("z") == BOOL

    def test_contains(self):
        """contains should check if variable is bound."""
        env = EMPTY_ENVIRONMENT.bind("x", INT)
        assert env.contains("x")
        assert not env.contains("y")

    def test_push_scope(self):
        """push_scope should create nested scope."""
        env = EMPTY_ENVIRONMENT.bind("x", INT)
        inner = env.push_scope()
        # Inner should still see outer bindings
        assert inner.lookup("x") == INT
        # Inner has no local bindings
        assert len(inner) == 0

    def test_nested_scope_shadow(self):
        """Inner scope can shadow outer scope."""
        env = EMPTY_ENVIRONMENT.bind("x", INT)
        inner = env.push_scope().bind("x", STR)
        assert inner.lookup("x") == STR
        # Outer unchanged
        assert env.lookup("x") == INT

    def test_pop_scope(self):
        """pop_scope should return to parent scope."""
        env = EMPTY_ENVIRONMENT.bind("x", INT)
        inner = env.push_scope().bind("y", STR)
        outer = inner.pop_scope()
        assert outer.lookup("x") == INT
        assert outer.lookup("y") is None

    def test_pop_scope_at_root(self):
        """pop_scope at root should return self."""
        env = EMPTY_ENVIRONMENT.bind("x", INT)
        result = env.pop_scope()
        assert result == env

    def test_all_bindings(self):
        """all_bindings should return all bindings including parents."""
        env = EMPTY_ENVIRONMENT.bind("x", INT)
        inner = env.push_scope().bind("y", STR)
        all_bindings = inner.all_bindings()
        assert all_bindings == {"x": INT, "y": STR}

    def test_all_bindings_shadowing(self):
        """all_bindings should show shadowed values from inner scope."""
        env = EMPTY_ENVIRONMENT.bind("x", INT)
        inner = env.push_scope().bind("x", STR)
        all_bindings = inner.all_bindings()
        assert all_bindings == {"x": STR}

    def test_local_bindings(self):
        """local_bindings should return only current scope."""
        env = EMPTY_ENVIRONMENT.bind("x", INT)
        inner = env.push_scope().bind("y", STR)
        local = inner.local_bindings()
        assert local == {"y": STR}

    def test_names(self):
        """names should return all bound names."""
        env = EMPTY_ENVIRONMENT.bind("x", INT).bind("y", STR)
        names = env.names()
        assert names == frozenset({"x", "y"})

    def test_names_with_scope(self):
        """names should include parent scope names."""
        env = EMPTY_ENVIRONMENT.bind("x", INT)
        inner = env.push_scope().bind("y", STR)
        names = inner.names()
        assert names == frozenset({"x", "y"})

    def test_len(self):
        """len should return local binding count."""
        env = EMPTY_ENVIRONMENT.bind("x", INT).bind("y", STR)
        assert len(env) == 2

    def test_iter(self):
        """iter should iterate over local binding names."""
        env = EMPTY_ENVIRONMENT.bind("x", INT).bind("y", STR)
        names = set(env)
        assert names == {"x", "y"}

    def test_equality(self):
        """Environments with same bindings should be equal."""
        env1 = EMPTY_ENVIRONMENT.bind("x", INT)
        env2 = EMPTY_ENVIRONMENT.bind("x", INT)
        assert env1 == env2

    def test_equality_different_bindings(self):
        """Environments with different bindings should not be equal."""
        env1 = EMPTY_ENVIRONMENT.bind("x", INT)
        env2 = EMPTY_ENVIRONMENT.bind("x", STR)
        assert env1 != env2

    def test_hash(self):
        """Environments should be hashable."""
        env = EMPTY_ENVIRONMENT.bind("x", INT)
        h = hash(env)
        assert isinstance(h, int)

    def test_hash_consistency(self):
        """Equal environments should have equal hashes."""
        env1 = EMPTY_ENVIRONMENT.bind("x", INT)
        env2 = EMPTY_ENVIRONMENT.bind("x", INT)
        assert hash(env1) == hash(env2)

    def test_repr(self):
        """repr should be readable."""
        env = EMPTY_ENVIRONMENT.bind("x", INT)
        repr_str = repr(env)
        assert "TypeEnvironment" in repr_str
        assert "x" in repr_str


class TestTypeEnvironmentSnapshot:
    """Tests for snapshot/restore functionality."""

    def test_snapshot_empty(self):
        """Empty environment snapshot should work."""
        snapshot = EMPTY_ENVIRONMENT.snapshot()
        assert snapshot.bindings == {}
        assert snapshot.parent_snapshot is None

    def test_snapshot_with_bindings(self):
        """Snapshot should capture bindings."""
        env = EMPTY_ENVIRONMENT.bind("x", INT)
        snapshot = env.snapshot()
        assert snapshot.bindings == {"x": INT}

    def test_snapshot_with_parent(self):
        """Snapshot should capture parent chain."""
        env = EMPTY_ENVIRONMENT.bind("x", INT)
        inner = env.push_scope().bind("y", STR)
        snapshot = inner.snapshot()
        assert snapshot.bindings == {"y": STR}
        assert snapshot.parent_snapshot is not None
        assert snapshot.parent_snapshot.bindings == {"x": INT}

    def test_restore_from_snapshot(self):
        """Should restore environment from snapshot."""
        original = EMPTY_ENVIRONMENT.bind("x", INT).bind("y", STR)
        snapshot = original.snapshot()
        restored = TypeEnvironment.from_snapshot(snapshot)
        assert restored.lookup("x") == INT
        assert restored.lookup("y") == STR

    def test_restore_with_parent(self):
        """Should restore parent chain from snapshot."""
        env = EMPTY_ENVIRONMENT.bind("x", INT)
        inner = env.push_scope().bind("y", STR)
        snapshot = inner.snapshot()
        restored = TypeEnvironment.from_snapshot(snapshot)
        assert restored.lookup("x") == INT
        assert restored.lookup("y") == STR
        # Should be able to pop scope
        outer = restored.pop_scope()
        assert outer.lookup("x") == INT
        assert outer.lookup("y") is None

    def test_snapshot_hashable(self):
        """Snapshot should be hashable."""
        snapshot = EMPTY_ENVIRONMENT.bind("x", INT).snapshot()
        h = hash(snapshot)
        assert isinstance(h, int)


class TestCreateEnvironment:
    """Tests for create_environment factory function."""

    def test_create_empty(self):
        """create_environment with None should return empty."""
        env = create_environment()
        assert env == EMPTY_ENVIRONMENT

    def test_create_with_bindings(self):
        """create_environment with bindings should create populated env."""
        env = create_environment({"x": INT, "y": STR})
        assert env.lookup("x") == INT
        assert env.lookup("y") == STR


class TestMergeEnvironments:
    """Tests for merge_environments function."""

    def test_merge_disjoint(self):
        """Merging disjoint environments should combine bindings."""
        env1 = EMPTY_ENVIRONMENT.bind("x", INT)
        env2 = EMPTY_ENVIRONMENT.bind("y", STR)
        merged = merge_environments(env1, env2)
        assert merged.lookup("x") == INT
        assert merged.lookup("y") == STR

    def test_merge_overlapping(self):
        """Second environment should take precedence on overlap."""
        env1 = EMPTY_ENVIRONMENT.bind("x", INT)
        env2 = EMPTY_ENVIRONMENT.bind("x", STR)
        merged = merge_environments(env1, env2)
        assert merged.lookup("x") == STR

    def test_merge_with_nested_scopes(self):
        """Merge should flatten nested scopes."""
        env1 = EMPTY_ENVIRONMENT.bind("x", INT)
        inner1 = env1.push_scope().bind("y", STR)
        env2 = EMPTY_ENVIRONMENT.bind("z", BOOL)
        merged = merge_environments(inner1, env2)
        assert merged.lookup("x") == INT
        assert merged.lookup("y") == STR
        assert merged.lookup("z") == BOOL


class TestEnvironmentWithComplexTypes:
    """Tests with complex type bindings."""

    def test_bind_function_type(self):
        """Should bind function types correctly."""
        func_type = FunctionType((INT, STR), BOOL)
        env = EMPTY_ENVIRONMENT.bind("f", func_type)
        assert env.lookup("f") == func_type

    def test_snapshot_with_complex_types(self):
        """Snapshot should preserve complex types."""
        func_type = FunctionType((INT,), STR)
        env = EMPTY_ENVIRONMENT.bind("f", func_type)
        snapshot = env.snapshot()
        restored = TypeEnvironment.from_snapshot(snapshot)
        assert restored.lookup("f") == func_type
