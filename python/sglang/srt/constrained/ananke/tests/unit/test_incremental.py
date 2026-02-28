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
"""Unit tests for incremental bidirectional typing.

Tests for order maintenance, dependency graphs, delta typing, and invalidation.
"""

import pytest
from typing import Optional, Tuple

from domains.types.constraint import INT, STR, BOOL, FLOAT, ANY, Type
from domains.types.environment import EMPTY_ENVIRONMENT
from domains.types.marking.marks import Mark, HoleMark, InconsistentMark
from domains.types.marking.provenance import SourceSpan

from domains.types.incremental.order_maintenance import (
    OrderMaintenanceList,
    OrderedElement,
    create_order_list,
    MAX_LABEL,
)
from domains.types.incremental.dependency_graph import (
    DependencyGraph,
    DependencyKind,
    NodeId,
    create_dependency_graph,
    snapshot_graph,
    restore_graph,
)
from domains.types.incremental.delta_typing import (
    TypeCache,
    CheckMode,
    DeltaTypingEngine,
    create_delta_engine,
)
from domains.types.incremental.invalidation import (
    InvalidationEngine,
    StabilityKind,
    create_invalidation_engine,
    is_type_concrete,
)


# ===========================================================================
# Order Maintenance Tests
# ===========================================================================


class TestOrderMaintenanceList:
    """Tests for OrderMaintenanceList data structure."""

    def test_empty_list(self):
        """Empty list should have no elements."""
        order = create_order_list()
        assert order.is_empty()
        assert len(order) == 0
        assert order.head is None
        assert order.tail is None

    def test_insert_first(self):
        """insert_first should add element at beginning."""
        order = create_order_list()
        elem = order.insert_first("a")
        assert not order.is_empty()
        assert len(order) == 1
        assert order.head == elem
        assert order.tail == elem
        assert elem.value == "a"

    def test_insert_last(self):
        """insert_last should add element at end."""
        order = create_order_list()
        elem = order.insert_last("a")
        assert len(order) == 1
        assert order.head == elem
        assert order.tail == elem

    def test_insert_first_multiple(self):
        """Multiple insert_first should maintain order."""
        order = create_order_list()
        c = order.insert_first("c")
        b = order.insert_first("b")
        a = order.insert_first("a")

        assert order.head == a
        assert order.tail == c
        assert order.comes_before(a, b)
        assert order.comes_before(b, c)
        assert order.comes_before(a, c)

    def test_insert_last_multiple(self):
        """Multiple insert_last should maintain order."""
        order = create_order_list()
        a = order.insert_last("a")
        b = order.insert_last("b")
        c = order.insert_last("c")

        assert order.head == a
        assert order.tail == c
        assert order.comes_before(a, b)
        assert order.comes_before(b, c)

    def test_insert_after(self):
        """insert_after should insert in correct position."""
        order = create_order_list()
        a = order.insert_first("a")
        c = order.insert_after(a, "c")
        b = order.insert_after(a, "b")

        # Order should be: a < b < c
        assert order.comes_before(a, b)
        assert order.comes_before(b, c)
        assert order.comes_before(a, c)

    def test_insert_before(self):
        """insert_before should insert in correct position."""
        order = create_order_list()
        c = order.insert_first("c")
        a = order.insert_before(c, "a")
        b = order.insert_before(c, "b")

        # Order should be: a < b < c
        assert order.comes_before(a, b)
        assert order.comes_before(b, c)

    def test_order_query(self):
        """order() should return correct comparison."""
        order = create_order_list()
        a = order.insert_first("a")
        b = order.insert_after(a, "b")

        assert order.order(a, b) == -1  # a < b
        assert order.order(b, a) == 1   # b > a
        assert order.order(a, a) == 0   # a == a

    def test_delete(self):
        """delete should remove element from list."""
        order = create_order_list()
        a = order.insert_first("a")
        b = order.insert_after(a, "b")
        c = order.insert_after(b, "c")

        order.delete(b)

        assert len(order) == 2
        assert order.comes_before(a, c)
        assert b.prev is None
        assert b.next is None

    def test_delete_head(self):
        """Deleting head should update head pointer."""
        order = create_order_list()
        a = order.insert_first("a")
        b = order.insert_after(a, "b")

        order.delete(a)

        assert order.head == b
        assert len(order) == 1

    def test_delete_tail(self):
        """Deleting tail should update tail pointer."""
        order = create_order_list()
        a = order.insert_first("a")
        b = order.insert_after(a, "b")

        order.delete(b)

        assert order.tail == a
        assert len(order) == 1

    def test_find(self):
        """find should locate element by value."""
        order = create_order_list()
        a = order.insert_first("a")
        b = order.insert_after(a, "b")

        found = order.find("a")
        assert found == a

        not_found = order.find("c")
        assert not_found is None

    def test_iteration(self):
        """Iteration should yield values in order."""
        order = create_order_list()
        order.insert_last("a")
        order.insert_last("b")
        order.insert_last("c")

        values = list(order)
        assert values == ["a", "b", "c"]

    def test_many_insertions(self):
        """Many insertions should maintain correct ordering."""
        order = create_order_list()
        elements = []

        # Insert 100 elements
        for i in range(100):
            if i == 0:
                elem = order.insert_first(i)
            else:
                elem = order.insert_after(elements[-1], i)
            elements.append(elem)

        # Verify all orderings
        for i in range(len(elements) - 1):
            assert order.comes_before(elements[i], elements[i + 1])

    def test_interleaved_insertions(self):
        """Interleaved insertions should trigger relabeling."""
        order = create_order_list()

        # Insert first element
        a = order.insert_first("a")

        # Insert many elements between a and end
        current = a
        for i in range(50):
            current = order.insert_after(current, f"x{i}")

        # Insert elements at the beginning to force relabeling
        for i in range(50):
            order.insert_first(f"y{i}")

        # Verify list is still valid
        values = list(order)
        assert len(values) == 101
        assert "a" in values

    def test_ordered_element_comparison(self):
        """OrderedElement comparison operators should work."""
        order = create_order_list()
        a = order.insert_first("a")
        b = order.insert_after(a, "b")

        assert a < b
        assert a <= b
        assert b > a
        assert b >= a
        assert a <= a
        assert a >= a


# ===========================================================================
# Dependency Graph Tests
# ===========================================================================


class TestDependencyGraph:
    """Tests for DependencyGraph class."""

    def make_node(self, start: int, end: int, kind: str = "expr") -> NodeId:
        """Helper to create a NodeId."""
        return NodeId(span=SourceSpan(start, end), kind=kind)

    def test_empty_graph(self):
        """Empty graph should have no nodes or edges."""
        graph = create_dependency_graph()
        assert graph.node_count() == 0
        assert graph.edge_count() == 0

    def test_add_dependency(self):
        """Adding dependency should create edge."""
        graph = create_dependency_graph()
        n1 = self.make_node(0, 5)
        n2 = self.make_node(10, 15)

        dep = graph.add_dependency(n1, n2, DependencyKind.SYNTHESIS)

        assert dep.source == n1
        assert dep.target == n2
        assert dep.kind == DependencyKind.SYNTHESIS
        assert graph.edge_count() == 1

    def test_add_synthesis_dependency(self):
        """Synthesis dependency helper should work."""
        graph = create_dependency_graph()
        parent = self.make_node(0, 20)
        child = self.make_node(5, 10)

        dep = graph.add_synthesis_dependency(parent, child)

        assert dep.kind == DependencyKind.SYNTHESIS
        assert dep.source == parent
        assert dep.target == child

    def test_add_analysis_dependency(self):
        """Analysis dependency helper should work."""
        graph = create_dependency_graph()
        parent = self.make_node(0, 20)
        child = self.make_node(5, 10)

        dep = graph.add_analysis_dependency(child, parent)

        assert dep.kind == DependencyKind.ANALYSIS
        assert dep.source == child
        assert dep.target == parent

    def test_add_environment_dependency(self):
        """Environment dependency should be indexed by binding name."""
        graph = create_dependency_graph()
        usage = self.make_node(10, 15, "var")
        binding = self.make_node(0, 5, "let")

        dep = graph.add_environment_dependency(usage, "x", binding)

        assert dep.kind == DependencyKind.ENVIRONMENT
        assert dep.binding_name == "x"

        # Should be findable by binding name
        nodes = graph.nodes_depending_on_binding("x")
        assert usage in nodes

    def test_dependencies_of(self):
        """dependencies_of should return forward edges."""
        graph = create_dependency_graph()
        n1 = self.make_node(0, 5)
        n2 = self.make_node(10, 15)
        n3 = self.make_node(20, 25)

        graph.add_dependency(n1, n2, DependencyKind.SYNTHESIS)
        graph.add_dependency(n1, n3, DependencyKind.ANALYSIS)

        deps = graph.dependencies_of(n1)
        assert len(deps) == 2
        targets = {d.target for d in deps}
        assert n2 in targets
        assert n3 in targets

    def test_dependents_of(self):
        """dependents_of should return backward edges."""
        graph = create_dependency_graph()
        n1 = self.make_node(0, 5)
        n2 = self.make_node(10, 15)
        n3 = self.make_node(20, 25)

        graph.add_dependency(n2, n1, DependencyKind.SYNTHESIS)
        graph.add_dependency(n3, n1, DependencyKind.ANALYSIS)

        deps = graph.dependents_of(n1)
        assert len(deps) == 2
        sources = {d.source for d in deps}
        assert n2 in sources
        assert n3 in sources

    def test_remove_node(self):
        """Removing node should remove all its edges."""
        graph = create_dependency_graph()
        n1 = self.make_node(0, 5)
        n2 = self.make_node(10, 15)
        n3 = self.make_node(20, 25)

        graph.add_dependency(n1, n2, DependencyKind.SYNTHESIS)
        graph.add_dependency(n2, n3, DependencyKind.SYNTHESIS)

        graph.remove_node(n2)

        # n1 should have no dependencies now
        assert len(graph.dependencies_of(n1)) == 0
        # n3 should have no dependents now
        assert len(graph.dependents_of(n3)) == 0

    def test_affected_by_span(self):
        """affected_by should find transitively affected nodes."""
        graph = create_dependency_graph()

        # Create a dependency chain: n1 -> n2 -> n3
        n1 = self.make_node(0, 5)
        n2 = self.make_node(10, 15)
        n3 = self.make_node(20, 25)

        graph.add_dependency(n2, n1, DependencyKind.SYNTHESIS)  # n2 depends on n1
        graph.add_dependency(n3, n2, DependencyKind.SYNTHESIS)  # n3 depends on n2

        # Edit at n1's location
        affected = graph.affected_by(SourceSpan(0, 5))

        # All three should be affected (n1 directly, n2 and n3 transitively)
        assert n1 in affected
        assert n2 in affected
        assert n3 in affected

    def test_affected_by_binding_change(self):
        """affected_by_binding_change should find nodes using the binding."""
        graph = create_dependency_graph()

        binding = self.make_node(0, 5, "let")
        usage1 = self.make_node(10, 15, "var")
        usage2 = self.make_node(20, 25, "var")
        dependent = self.make_node(30, 35, "call")

        graph.add_environment_dependency(usage1, "x", binding)
        graph.add_environment_dependency(usage2, "x", binding)
        graph.add_dependency(dependent, usage1, DependencyKind.SYNTHESIS)

        affected = graph.affected_by_binding_change("x")

        assert usage1 in affected
        assert usage2 in affected
        assert dependent in affected

    def test_topological_order(self):
        """topological_order should respect dependencies."""
        graph = create_dependency_graph()

        n1 = self.make_node(0, 5)
        n2 = self.make_node(10, 15)
        n3 = self.make_node(20, 25)

        # n3 depends on n2, n2 depends on n1
        graph.add_dependency(n2, n1, DependencyKind.SYNTHESIS)
        graph.add_dependency(n3, n2, DependencyKind.SYNTHESIS)

        nodes = frozenset({n1, n2, n3})
        order = graph.topological_order(nodes)

        # n1 must come before n2, n2 must come before n3
        assert order.index(n1) < order.index(n2)
        assert order.index(n2) < order.index(n3)

    def test_topological_order_cycle_detection(self):
        """topological_order should detect cycles."""
        graph = create_dependency_graph()

        n1 = self.make_node(0, 5)
        n2 = self.make_node(10, 15)

        # Create a cycle: n1 -> n2 -> n1
        graph.add_dependency(n1, n2, DependencyKind.SYNTHESIS)
        graph.add_dependency(n2, n1, DependencyKind.SYNTHESIS)

        nodes = frozenset({n1, n2})
        with pytest.raises(ValueError, match="[Cc]ycle"):
            graph.topological_order(nodes)

    def test_snapshot_and_restore(self):
        """Snapshot and restore should preserve graph state."""
        graph = create_dependency_graph()

        n1 = self.make_node(0, 5)
        n2 = self.make_node(10, 15)
        graph.add_dependency(n1, n2, DependencyKind.SYNTHESIS)

        # Take snapshot
        snap = snapshot_graph(graph)

        # Modify graph
        n3 = self.make_node(20, 25)
        graph.add_dependency(n2, n3, DependencyKind.ANALYSIS)

        # Restore from snapshot
        restored = restore_graph(snap)

        # Restored should have original state
        assert restored.edge_count() == 1
        deps = restored.dependencies_of(n1)
        assert len(deps) == 1

    def test_clear(self):
        """clear should remove all nodes and edges."""
        graph = create_dependency_graph()

        n1 = self.make_node(0, 5)
        n2 = self.make_node(10, 15)
        graph.add_dependency(n1, n2, DependencyKind.SYNTHESIS)

        graph.clear()

        assert graph.node_count() == 0
        assert graph.edge_count() == 0


# ===========================================================================
# Delta Typing Engine Tests
# ===========================================================================


class TestTypeCache:
    """Tests for TypeCache class."""

    def make_node(self, start: int, kind: str = "expr") -> NodeId:
        """Helper to create a NodeId."""
        return NodeId(span=SourceSpan(start, start + 5), kind=kind)

    def test_empty_cache(self):
        """Empty cache should have no entries."""
        cache = TypeCache()
        assert len(cache) == 0
        assert cache.get(self.make_node(0)) is None

    def test_set_and_get(self):
        """Setting entry should be retrievable."""
        cache = TypeCache()
        node = self.make_node(0)

        entry = cache.set(node, CheckMode.SYNTHESIS, synthesized_type=INT)

        retrieved = cache.get(node)
        assert retrieved == entry
        assert retrieved.synthesized_type == INT

    def test_get_type(self):
        """get_type should return synthesized type."""
        cache = TypeCache()
        node = self.make_node(0)

        cache.set(node, CheckMode.SYNTHESIS, synthesized_type=STR)

        assert cache.get_type(node) == STR

    def test_invalidate(self):
        """Invalidating should mark entry as dirty."""
        cache = TypeCache()
        node = self.make_node(0)

        cache.set(node, CheckMode.SYNTHESIS, synthesized_type=INT)
        assert cache.is_valid(node)

        cache.invalidate(node)
        assert not cache.is_valid(node)

    def test_remove(self):
        """Removing should delete entry."""
        cache = TypeCache()
        node = self.make_node(0)

        cache.set(node, CheckMode.SYNTHESIS, synthesized_type=INT)
        cache.remove(node)

        assert cache.get(node) is None
        assert len(cache) == 0

    def test_dirty_nodes(self):
        """dirty_nodes should return invalidated entries in order."""
        cache = TypeCache()
        n1 = self.make_node(0)
        n2 = self.make_node(10)
        n3 = self.make_node(20)

        cache.set(n1, CheckMode.SYNTHESIS, synthesized_type=INT)
        cache.set(n2, CheckMode.SYNTHESIS, synthesized_type=STR)
        cache.set(n3, CheckMode.SYNTHESIS, synthesized_type=BOOL)

        cache.invalidate(n1)
        cache.invalidate(n3)

        dirty = cache.dirty_nodes()
        assert n1 in dirty
        assert n2 not in dirty
        assert n3 in dirty


class TestDeltaTypingEngine:
    """Tests for DeltaTypingEngine class."""

    def make_node(self, start: int, kind: str = "expr") -> NodeId:
        """Helper to create a NodeId."""
        return NodeId(span=SourceSpan(start, start + 5), kind=kind)

    def test_create_engine(self):
        """Creating engine should work."""
        def checker(node, env, expected):
            return INT, None

        engine = create_delta_engine(checker)
        assert engine is not None
        assert len(engine.cache) == 0

    def test_check_node_caches_result(self):
        """Checking a node should cache the result."""
        call_count = 0

        def checker(node, env, expected):
            nonlocal call_count
            call_count += 1
            return INT, None

        engine = create_delta_engine(checker)
        node = self.make_node(0)

        # First check
        typ1, mark1 = engine.check_node(node, EMPTY_ENVIRONMENT)
        assert call_count == 1
        assert typ1 == INT

        # Second check should use cache
        typ2, mark2 = engine.check_node(node, EMPTY_ENVIRONMENT)
        assert call_count == 1  # Not called again
        assert typ2 == INT

    def test_invalidate_forces_recheck(self):
        """Invalidating should force recheck."""
        call_count = 0

        def checker(node, env, expected):
            nonlocal call_count
            call_count += 1
            return INT, None

        engine = create_delta_engine(checker)
        node = self.make_node(0)

        # Initial check
        engine.check_node(node, EMPTY_ENVIRONMENT)
        assert call_count == 1

        # Invalidate
        engine.invalidate_span(SourceSpan(0, 5))

        # Should force recheck
        engine.check_node(node, EMPTY_ENVIRONMENT)
        assert call_count == 2

    def test_apply_edit(self):
        """apply_edit should invalidate and recheck."""
        types = {0: INT, 10: STR}

        def checker(node, env, expected):
            return types.get(node.span.start, ANY), None

        engine = create_delta_engine(checker)
        n1 = self.make_node(0)
        n2 = self.make_node(10)

        # Initial setup
        engine.cache.set(n1, CheckMode.SYNTHESIS, synthesized_type=INT)
        engine.cache.set(n2, CheckMode.SYNTHESIS, synthesized_type=STR)

        # Add dependency: n2 depends on n1
        engine.cache.add_dependency(n2, n1, DependencyKind.SYNTHESIS)

        # Apply edit at n1
        result = engine.apply_edit(SourceSpan(0, 5), EMPTY_ENVIRONMENT)

        assert result.rechecked_count >= 1

    def test_full_check(self):
        """full_check should check all nodes."""
        def checker(node, env, expected):
            return INT, None

        engine = create_delta_engine(checker)
        nodes = [self.make_node(i * 10) for i in range(5)]

        result = engine.full_check(nodes, EMPTY_ENVIRONMENT)

        assert result.rechecked_count == 5

    def test_tracks_new_errors(self):
        """Engine should track new type errors."""
        from domains.types.marking.provenance import Provenance

        error_node = self.make_node(10)

        def checker(node, env, expected):
            if node == error_node:
                mark = InconsistentMark(
                    synthesized=INT,
                    expected=STR,
                    provenance=Provenance(node.span, "test"),
                )
                return INT, mark
            return INT, None

        engine = create_delta_engine(checker)

        # Set up initial state (no error)
        engine.cache.set(error_node, CheckMode.SYNTHESIS, synthesized_type=INT, mark=None)
        engine.cache.invalidate(error_node)

        # Recheck should detect new error
        result = engine.recheck_dirty(EMPTY_ENVIRONMENT)

        assert len(result.new_errors) == 1
        assert result.new_errors[0][0] == error_node


# ===========================================================================
# Invalidation Engine Tests
# ===========================================================================


class TestInvalidationEngine:
    """Tests for InvalidationEngine class."""

    def make_node(self, start: int, kind: str = "expr") -> NodeId:
        """Helper to create a NodeId."""
        return NodeId(span=SourceSpan(start, start + 5), kind=kind)

    def test_invalidate_direct(self):
        """Invalidation should find directly affected nodes."""
        graph = create_dependency_graph()
        engine = create_invalidation_engine(graph)

        n1 = self.make_node(0)
        n2 = self.make_node(10)

        # n2 depends on n1
        graph.add_dependency(n2, n1, DependencyKind.SYNTHESIS)

        # Edit at n1
        result = engine.invalidate(SourceSpan(0, 5))

        assert n1 in result.directly_affected
        assert n1 in result.invalidated

    def test_invalidate_transitive(self):
        """Invalidation should propagate transitively."""
        graph = create_dependency_graph()
        engine = create_invalidation_engine(graph)

        n1 = self.make_node(0)
        n2 = self.make_node(10)
        n3 = self.make_node(20)

        # n2 depends on n1, n3 depends on n2
        graph.add_dependency(n2, n1, DependencyKind.SYNTHESIS)
        graph.add_dependency(n3, n2, DependencyKind.SYNTHESIS)

        result = engine.invalidate(SourceSpan(0, 5))

        assert n1 in result.invalidated
        assert n2 in result.invalidated
        assert n3 in result.invalidated

    def test_invalidate_stops_at_annotated(self):
        """Invalidation should stop at annotated nodes."""
        graph = create_dependency_graph()
        engine = create_invalidation_engine(graph)

        n1 = self.make_node(0)
        n2 = self.make_node(10)
        n3 = self.make_node(20)

        graph.add_dependency(n2, n1, DependencyKind.SYNTHESIS)
        graph.add_dependency(n3, n2, DependencyKind.SYNTHESIS)

        # Mark n2 as annotated
        engine.set_annotated(n2)

        result = engine.invalidate(SourceSpan(0, 5))

        # n1 directly affected, so invalidated
        assert n1 in result.invalidated
        # n2 is stable boundary
        assert (n2, StabilityKind.ANNOTATED) in result.stable_boundaries
        # n3 should not be invalidated
        assert n3 not in result.invalidated

    def test_invalidate_stops_at_concrete(self):
        """Invalidation should stop at concrete type nodes."""
        graph = create_dependency_graph()
        engine = create_invalidation_engine(graph)

        n1 = self.make_node(0)
        n2 = self.make_node(10)
        n3 = self.make_node(20)

        graph.add_dependency(n2, n1, DependencyKind.SYNTHESIS)
        graph.add_dependency(n3, n2, DependencyKind.SYNTHESIS)

        # Set concrete type for n2
        engine.set_cached_type(n2, INT)  # INT is concrete

        result = engine.invalidate(SourceSpan(0, 5))

        # n1 directly affected
        assert n1 in result.invalidated
        # n2 is stable (concrete type)
        assert (n2, StabilityKind.CONCRETE) in result.stable_boundaries
        # n3 should not be invalidated
        assert n3 not in result.invalidated

    def test_invalidate_binding(self):
        """invalidate_binding should find nodes using the binding."""
        graph = create_dependency_graph()
        engine = create_invalidation_engine(graph)

        binding = self.make_node(0, "let")
        usage1 = self.make_node(10, "var")
        usage2 = self.make_node(20, "var")

        graph.add_environment_dependency(usage1, "x", binding)
        graph.add_environment_dependency(usage2, "x", binding)

        result = engine.invalidate_binding("x")

        assert usage1 in result.invalidated
        assert usage2 in result.invalidated

    def test_invalidate_node(self):
        """invalidate_node should invalidate specific node and dependents."""
        graph = create_dependency_graph()
        engine = create_invalidation_engine(graph)

        n1 = self.make_node(0)
        n2 = self.make_node(10)
        n3 = self.make_node(20)

        graph.add_dependency(n2, n1, DependencyKind.SYNTHESIS)
        graph.add_dependency(n3, n2, DependencyKind.SYNTHESIS)

        result = engine.invalidate_node(n2)

        # n1 should not be invalidated (n2 doesn't depend on anything)
        assert n1 not in result.invalidated
        # n2 directly targeted
        assert n2 in result.invalidated
        # n3 depends on n2
        assert n3 in result.invalidated


class TestIsTypeConcrete:
    """Tests for is_type_concrete function."""

    def test_primitive_types_are_concrete(self):
        """Primitive types should be concrete."""
        from domains.types.constraint import PrimitiveType

        assert is_type_concrete(INT)
        assert is_type_concrete(STR)
        assert is_type_concrete(BOOL)
        assert is_type_concrete(FLOAT)

    def test_type_var_is_not_concrete(self):
        """TypeVar should not be concrete."""
        from domains.types.constraint import TypeVar

        var = TypeVar("T")
        assert not is_type_concrete(var)

    def test_any_is_concrete(self):
        """Any type is concrete (no variables)."""
        assert is_type_concrete(ANY)

    def test_list_with_concrete_element_is_concrete(self):
        """List[int] should be concrete."""
        from domains.types.constraint import ListType

        list_int = ListType(INT)
        assert is_type_concrete(list_int)

    def test_list_with_type_var_is_not_concrete(self):
        """List[T] should not be concrete."""
        from domains.types.constraint import ListType, TypeVar

        list_t = ListType(TypeVar("T"))
        assert not is_type_concrete(list_t)

    def test_function_with_concrete_types_is_concrete(self):
        """(int) -> str should be concrete."""
        from domains.types.constraint import FunctionType

        func = FunctionType((INT,), STR)
        assert is_type_concrete(func)

    def test_function_with_type_var_is_not_concrete(self):
        """(T) -> int should not be concrete."""
        from domains.types.constraint import FunctionType, TypeVar

        func = FunctionType((TypeVar("T"),), INT)
        assert not is_type_concrete(func)

    def test_dict_with_concrete_types_is_concrete(self):
        """Dict[str, int] should be concrete."""
        from domains.types.constraint import DictType

        dict_type = DictType(STR, INT)
        assert is_type_concrete(dict_type)

    def test_tuple_with_concrete_types_is_concrete(self):
        """Tuple[int, str] should be concrete."""
        from domains.types.constraint import TupleType

        tuple_type = TupleType((INT, STR))
        assert is_type_concrete(tuple_type)

    def test_nested_type_var_is_not_concrete(self):
        """List[Dict[str, T]] should not be concrete."""
        from domains.types.constraint import ListType, DictType, TypeVar

        nested = ListType(DictType(STR, TypeVar("T")))
        assert not is_type_concrete(nested)
