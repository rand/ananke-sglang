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
"""Tests for the holes module."""

import pytest

from holes import (
    # Core hole types
    Hole,
    HoleId,
    HoleGranularity,
    HoleState,
    SourceLocation,
    TypeEnvironment,
    create_hole,
    # Registry
    HoleRegistry,
    RegistryCheckpoint,
    create_registry,
    # Closures
    HoleClosure,
    FilledClosure,
    Continuation,
    ContinuationKind,
    ClosureManager,
    create_closure,
    # Environment capture
    CapturedBinding,
    ScopeInfo,
    EnvironmentCapture,
    EnvironmentCapturer,
    capture_environment,
    merge_captures,
    # Factory
    HoleSpec,
    HoleFactory,
    create_factory,
    # Fill-and-resume
    EvaluationState,
    EvaluationResult,
    FillResult,
    FillAndResumeEngine,
    create_fill_resume_engine,
    # Strategies
    HoleSelectionStrategy,
    DepthFirstStrategy,
    BreadthFirstStrategy,
    SourceOrderStrategy,
    TypeGuidedStrategy,
    PriorityStrategy,
    GranularityStrategy,
    CompositeStrategy,
    depth_first,
    breadth_first,
    source_order,
    type_guided,
    priority_based,
    granularity_based,
    default_strategy,
)


# =============================================================================
# HoleId Tests
# =============================================================================


class TestHoleId:
    """Tests for HoleId."""

    def test_create_default(self) -> None:
        """Test default HoleId creation."""
        hole_id = HoleId.create("test")
        assert hole_id.name == "test"
        assert hole_id.namespace == "default"
        assert hole_id.index == 0
        assert hole_id.depth == 0

    def test_create_with_params(self) -> None:
        """Test HoleId with all parameters."""
        hole_id = HoleId.create(
            "body",
            namespace="function",
            index=5,
            depth=2,
        )
        assert hole_id.name == "body"
        assert hole_id.namespace == "function"
        assert hole_id.index == 5
        assert hole_id.depth == 2

    def test_str_representation(self) -> None:
        """Test string representation."""
        hole_id = HoleId("ns", "name", 3, 0)
        assert str(hole_id) == "?ns:name[3]"

    def test_with_depth(self) -> None:
        """Test with_depth method."""
        hole_id = HoleId.create("test", depth=0)
        deeper = hole_id.with_depth(3)
        assert deeper.depth == 3
        assert deeper.name == hole_id.name
        assert deeper.namespace == hole_id.namespace
        assert deeper.index == hole_id.index

    def test_child(self) -> None:
        """Test child method."""
        parent = HoleId.create("parent", depth=1)
        child = parent.child("child", 0)
        assert child.name == "child"
        assert child.depth == 2
        assert child.namespace == parent.namespace

    def test_hashable(self) -> None:
        """Test HoleId is hashable."""
        hole_id = HoleId.create("test")
        d = {hole_id: "value"}
        assert d[hole_id] == "value"


# =============================================================================
# SourceLocation Tests
# =============================================================================


class TestSourceLocation:
    """Tests for SourceLocation."""

    def test_create(self) -> None:
        """Test SourceLocation creation."""
        loc = SourceLocation(line=10, column=5, offset=100, length=20)
        assert loc.line == 10
        assert loc.column == 5
        assert loc.offset == 100
        assert loc.length == 20

    def test_str_with_file(self) -> None:
        """Test string with file."""
        loc = SourceLocation(line=10, column=5, file="test.py")
        assert str(loc) == "test.py:10:5"

    def test_str_without_file(self) -> None:
        """Test string without file."""
        loc = SourceLocation(line=10, column=5)
        assert str(loc) == "10:5"

    def test_contains(self) -> None:
        """Test contains method."""
        loc = SourceLocation(offset=10, length=20)
        assert loc.contains(10)
        assert loc.contains(25)
        assert not loc.contains(5)
        assert not loc.contains(30)

    def test_overlaps(self) -> None:
        """Test overlaps method."""
        loc1 = SourceLocation(offset=10, length=20)
        loc2 = SourceLocation(offset=25, length=10)
        loc3 = SourceLocation(offset=50, length=10)
        assert loc1.overlaps(loc2)
        assert not loc1.overlaps(loc3)


# =============================================================================
# TypeEnvironment Tests
# =============================================================================


class TestTypeEnvironment:
    """Tests for TypeEnvironment."""

    def test_empty(self) -> None:
        """Test empty environment."""
        env = TypeEnvironment.empty()
        assert env.lookup("x") is None
        assert len(env.names()) == 0

    def test_from_dict(self) -> None:
        """Test from_dict factory."""
        env = TypeEnvironment.from_dict({"x": "int", "y": "str"})
        assert env.lookup("x") == "int"
        assert env.lookup("y") == "str"

    def test_bind(self) -> None:
        """Test bind method."""
        env = TypeEnvironment.empty()
        env2 = env.bind("x", "int")
        assert env.lookup("x") is None  # Original unchanged
        assert env2.lookup("x") == "int"

    def test_parent_lookup(self) -> None:
        """Test lookup with parent environment."""
        parent = TypeEnvironment.from_dict({"x": "int"})
        child = TypeEnvironment(bindings=frozenset([("y", "str")]), parent=parent)
        assert child.lookup("y") == "str"
        assert child.lookup("x") == "int"

    def test_all_bindings(self) -> None:
        """Test all_bindings method."""
        parent = TypeEnvironment.from_dict({"x": "int"})
        child = TypeEnvironment(bindings=frozenset([("y", "str")]), parent=parent)
        all_bindings = child.all_bindings()
        assert all_bindings["x"] == "int"
        assert all_bindings["y"] == "str"

    def test_names(self) -> None:
        """Test names method."""
        env = TypeEnvironment.from_dict({"x": "int", "y": "str"})
        assert env.names() == frozenset(["x", "y"])


# =============================================================================
# Hole Tests
# =============================================================================


class TestHole:
    """Tests for Hole."""

    def test_create_basic(self) -> None:
        """Test basic hole creation."""
        hole_id = HoleId.create("test")
        hole = Hole(id=hole_id)
        assert hole.id == hole_id
        assert hole.is_empty
        assert not hole.is_filled
        assert hole.state == HoleState.EMPTY

    def test_create_with_type(self) -> None:
        """Test hole with expected type."""
        hole = create_hole("test", expected_type="int")
        assert hole.expected_type == "int"

    def test_fill(self) -> None:
        """Test fill method."""
        hole = create_hole("test")
        filled = hole.fill("return 42")
        assert not hole.is_filled  # Original unchanged
        assert filled.is_filled
        assert filled.content == "return 42"
        assert filled.state == HoleState.FILLED

    def test_unfill(self) -> None:
        """Test unfill method."""
        hole = create_hole("test")
        filled = hole.fill("return 42")
        unfilled = filled.unfill()
        assert unfilled.is_empty
        assert unfilled.content is None

    def test_with_constraint(self) -> None:
        """Test with_constraint method."""
        hole = create_hole("test")
        updated = hole.with_constraint("constraint")
        assert updated.constraint == "constraint"
        assert hole.constraint is None  # Original unchanged

    def test_with_type(self) -> None:
        """Test with_type method."""
        hole = create_hole("test")
        updated = hole.with_type("float")
        assert updated.expected_type == "float"

    def test_validate(self) -> None:
        """Test validate method."""
        hole = create_hole("test")
        filled = hole.fill("return 42")
        validated = filled.validate()
        assert validated.state == HoleState.VALIDATED
        assert validated.is_valid

    def test_invalidate(self) -> None:
        """Test invalidate method."""
        hole = create_hole("test")
        filled = hole.fill("return 42")
        invalid = filled.invalidate()
        assert invalid.state == HoleState.INVALID
        assert not invalid.is_valid

    def test_child_hole(self) -> None:
        """Test child_hole method."""
        parent = create_hole("parent")
        child = parent.child_hole("child")
        assert child.parent == parent.id
        assert child.depth == parent.depth + 1

    def test_is_nested(self) -> None:
        """Test is_nested property."""
        parent = create_hole("parent")
        child = parent.child_hole("child")
        assert not parent.is_nested
        assert child.is_nested


# =============================================================================
# HoleRegistry Tests
# =============================================================================


class TestHoleRegistry:
    """Tests for HoleRegistry."""

    def test_create_empty(self) -> None:
        """Test empty registry."""
        registry = create_registry()
        assert registry.count == 0
        assert registry.empty_count == 0

    def test_create_hole(self) -> None:
        """Test creating a hole in registry."""
        registry = HoleRegistry()
        hole = registry.create("test", expected_type="int")
        assert registry.count == 1
        assert hole.expected_type == "int"

    def test_lookup(self) -> None:
        """Test lookup method."""
        registry = HoleRegistry()
        hole = registry.create("test")
        found = registry.lookup(hole.id)
        assert found == hole

    def test_lookup_missing(self) -> None:
        """Test lookup for missing hole."""
        registry = HoleRegistry()
        missing_id = HoleId.create("missing")
        assert registry.lookup(missing_id) is None

    def test_get(self) -> None:
        """Test get method."""
        registry = HoleRegistry()
        hole = registry.create("test")
        found = registry.get(hole.id)
        assert found == hole

    def test_get_missing_raises(self) -> None:
        """Test get raises for missing hole."""
        registry = HoleRegistry()
        missing_id = HoleId.create("missing")
        with pytest.raises(KeyError):
            registry.get(missing_id)

    def test_fill(self) -> None:
        """Test fill method."""
        registry = HoleRegistry()
        hole = registry.create("test")
        filled = registry.fill(hole.id, "return 42")
        assert filled.is_filled
        assert registry.filled_count == 1

    def test_unfill(self) -> None:
        """Test unfill method."""
        registry = HoleRegistry()
        hole = registry.create("test")
        registry.fill(hole.id, "return 42")
        unfilled = registry.unfill(hole.id)
        assert unfilled.is_empty
        assert registry.empty_count == 1

    def test_children(self) -> None:
        """Test children method."""
        registry = HoleRegistry()
        parent = registry.create("parent")
        child1 = registry.create("child1", parent=parent.id)
        child2 = registry.create("child2", parent=parent.id)
        children = registry.children(parent.id)
        assert child1.id in children
        assert child2.id in children

    def test_descendants(self) -> None:
        """Test descendants method."""
        registry = HoleRegistry()
        grandparent = registry.create("grandparent")
        parent = registry.create("parent", parent=grandparent.id)
        child = registry.create("child", parent=parent.id)
        descendants = registry.descendants(grandparent.id)
        assert parent.id in descendants
        assert child.id in descendants

    def test_ancestors(self) -> None:
        """Test ancestors method."""
        registry = HoleRegistry()
        grandparent = registry.create("grandparent")
        parent = registry.create("parent", parent=grandparent.id)
        child = registry.create("child", parent=parent.id)
        ancestors = registry.ancestors(child.id)
        assert ancestors == [parent.id, grandparent.id]

    def test_checkpoint_restore(self) -> None:
        """Test checkpoint and restore."""
        registry = HoleRegistry()
        hole1 = registry.create("hole1")
        checkpoint = registry.checkpoint()
        hole2 = registry.create("hole2")
        assert registry.count == 2
        registry.restore(checkpoint)
        assert registry.count == 1
        assert registry.lookup(hole1.id) is not None
        assert registry.lookup(hole2.id) is None

    def test_root_holes(self) -> None:
        """Test root_holes method."""
        registry = HoleRegistry()
        root1 = registry.create("root1")
        root2 = registry.create("root2")
        child = registry.create("child", parent=root1.id)
        roots = registry.root_holes()
        assert root1.id in roots
        assert root2.id in roots
        assert child.id not in roots

    def test_next_hole(self) -> None:
        """Test next_hole method."""
        registry = HoleRegistry()
        hole1 = registry.create("hole1")
        hole2 = registry.create("hole2")
        next_h = registry.next_hole()
        assert next_h is not None
        assert next_h.id == hole1.id  # First by index

    def test_next_hole_with_predicate(self) -> None:
        """Test next_hole with predicate."""
        registry = HoleRegistry()
        hole1 = registry.create("hole1", expected_type="int")
        hole2 = registry.create("hole2", expected_type="str")
        next_h = registry.next_hole(lambda h: h.expected_type == "str")
        assert next_h is not None
        assert next_h.id == hole2.id

    def test_iter(self) -> None:
        """Test iteration over registry."""
        registry = HoleRegistry()
        hole1 = registry.create("hole1")
        hole2 = registry.create("hole2")
        holes = list(registry)
        assert len(holes) == 2


# =============================================================================
# Closure Tests
# =============================================================================


class TestContinuation:
    """Tests for Continuation."""

    def test_identity(self) -> None:
        """Test identity continuation."""
        cont = Continuation.identity()
        assert cont.kind == ContinuationKind.IDENTITY
        result = cont.apply("value")
        assert result == "value"

    def test_compose(self) -> None:
        """Test continuation composition."""
        cont1 = Continuation.identity()
        cont2 = Continuation(kind=ContinuationKind.APPLICATION)
        composed = cont1.compose(cont2)
        assert composed.next == cont2


class TestHoleClosure:
    """Tests for HoleClosure."""

    def test_create(self) -> None:
        """Test closure creation."""
        hole = create_hole("test", expected_type="int")
        closure = HoleClosure.create(hole)
        assert closure.hole == hole
        assert closure.continuation.kind == ContinuationKind.IDENTITY

    def test_fill(self) -> None:
        """Test fill method."""
        hole = create_hole("test")
        closure = HoleClosure.create(hole)
        filled = closure.fill("return 42")
        assert isinstance(filled, FilledClosure)
        assert filled.content == "return 42"

    def test_with_continuation(self) -> None:
        """Test with_continuation method."""
        hole = create_hole("test")
        closure = HoleClosure.create(hole)
        new_cont = Continuation(kind=ContinuationKind.APPLICATION)
        updated = closure.with_continuation(new_cont)
        assert updated.continuation.kind == ContinuationKind.APPLICATION


class TestFilledClosure:
    """Tests for FilledClosure."""

    def test_unfill(self) -> None:
        """Test unfill method."""
        hole = create_hole("test")
        closure = HoleClosure.create(hole)
        filled = closure.fill("return 42")
        unfilled = filled.unfill()
        assert isinstance(unfilled, HoleClosure)
        assert unfilled.hole.is_empty

    def test_evaluate(self) -> None:
        """Test evaluate method."""
        hole = create_hole("test")
        closure = HoleClosure.create(hole)
        filled = closure.fill("return 42")
        result = filled.evaluate()
        assert result == "return 42"

    def test_evaluate_with_evaluator(self) -> None:
        """Test evaluate with custom evaluator."""
        hole = create_hole("test")
        closure = HoleClosure.create(hole)
        filled = closure.fill("return 42")

        def evaluator(code, env):
            return f"evaluated: {code}"

        result = filled.evaluate(evaluator)
        assert result == "evaluated: return 42"


class TestClosureManager:
    """Tests for ClosureManager."""

    def test_add_closure(self) -> None:
        """Test add_closure method."""
        manager = ClosureManager()
        hole = create_hole("test")
        closure = HoleClosure.create(hole)
        manager.add_closure(closure)
        assert manager.unfilled_count == 1

    def test_fill(self) -> None:
        """Test fill method."""
        manager = ClosureManager()
        hole = create_hole("test")
        closure = HoleClosure.create(hole)
        manager.add_closure(closure)
        filled = manager.fill(hole.id, "return 42")
        assert filled is not None
        assert manager.filled_count == 1
        assert manager.unfilled_count == 0

    def test_unfill(self) -> None:
        """Test unfill method."""
        manager = ClosureManager()
        hole = create_hole("test")
        closure = HoleClosure.create(hole)
        manager.add_closure(closure)
        manager.fill(hole.id, "return 42")
        unfilled = manager.unfill(hole.id)
        assert unfilled is not None
        assert manager.unfilled_count == 1

    def test_next_unfilled(self) -> None:
        """Test next_unfilled method."""
        manager = ClosureManager()
        hole1 = create_hole("test1")
        hole2 = create_hole("test2")
        manager.add_closure(HoleClosure.create(hole1))
        manager.add_closure(HoleClosure.create(hole2))
        next_closure = manager.next_unfilled()
        assert next_closure is not None


# =============================================================================
# Environment Capture Tests
# =============================================================================


class TestCapturedBinding:
    """Tests for CapturedBinding."""

    def test_create(self) -> None:
        """Test binding creation."""
        binding = CapturedBinding(name="x", typ="int")
        assert binding.name == "x"
        assert binding.typ == "int"
        assert not binding.is_parameter
        assert not binding.is_import

    def test_str(self) -> None:
        """Test string representation."""
        binding = CapturedBinding(name="x", typ="int")
        assert str(binding) == "x: int"


class TestEnvironmentCapture:
    """Tests for EnvironmentCapture."""

    def test_empty(self) -> None:
        """Test empty capture."""
        capture = EnvironmentCapture.empty()
        assert len(capture.names()) == 0

    def test_from_dict(self) -> None:
        """Test from_dict factory."""
        capture = EnvironmentCapture.from_dict({"x": "int", "y": "str"})
        assert capture.get_type("x") == "int"
        assert capture.get_type("y") == "str"

    def test_add_binding(self) -> None:
        """Test add_binding method."""
        capture = EnvironmentCapture()
        binding = CapturedBinding(name="x", typ="int")
        capture.add_binding(binding)
        assert capture.lookup("x") == binding

    def test_local_names(self) -> None:
        """Test local_names method."""
        capture = EnvironmentCapture()
        capture.add_binding(CapturedBinding(name="local", typ="int"))
        capture.add_binding(CapturedBinding(name="global_var", typ="str", is_global=True))
        assert "local" in capture.local_names()
        assert "global_var" not in capture.local_names()

    def test_to_type_environment(self) -> None:
        """Test to_type_environment method."""
        capture = EnvironmentCapture.from_dict({"x": "int"})
        env = capture.to_type_environment()
        assert env.lookup("x") == "int"


class TestMergeCaptures:
    """Tests for merge_captures function."""

    def test_merge_empty(self) -> None:
        """Test merging empty captures."""
        result = merge_captures()
        assert len(result.names()) == 0

    def test_merge_two(self) -> None:
        """Test merging two captures."""
        c1 = EnvironmentCapture.from_dict({"x": "int"})
        c2 = EnvironmentCapture.from_dict({"y": "str"})
        result = merge_captures(c1, c2)
        assert result.get_type("x") == "int"
        assert result.get_type("y") == "str"

    def test_merge_override(self) -> None:
        """Test later captures override earlier."""
        c1 = EnvironmentCapture.from_dict({"x": "int"})
        c2 = EnvironmentCapture.from_dict({"x": "str"})
        result = merge_captures(c1, c2)
        assert result.get_type("x") == "str"


# =============================================================================
# Factory Tests
# =============================================================================


class TestHoleFactory:
    """Tests for HoleFactory."""

    def test_from_spec(self) -> None:
        """Test from_spec method."""
        registry = HoleRegistry()
        factory = HoleFactory(registry)
        spec = HoleSpec(name="test", expected_type="int")
        hole = factory.from_spec(spec)
        assert hole.expected_type == "int"
        assert registry.count == 1

    def test_from_ast_gap(self) -> None:
        """Test from_ast_gap method."""
        registry = HoleRegistry()
        factory = HoleFactory(registry)
        location = SourceLocation(line=10, column=5)
        hole = factory.from_ast_gap(location, name="gap", expected_type="int")
        assert hole.location == location
        assert hole.metadata.get("source") == "ast_gap"

    def test_from_type_error(self) -> None:
        """Test from_type_error method."""
        registry = HoleRegistry()
        factory = HoleFactory(registry)
        location = SourceLocation(line=10, column=5)
        hole = factory.from_type_error(
            location,
            synthesized_type="str",
            expected_type="int",
        )
        assert hole.expected_type == "int"
        assert hole.metadata.get("synthesized_type") == "str"

    def test_from_placeholder(self) -> None:
        """Test from_placeholder method."""
        registry = HoleRegistry()
        factory = HoleFactory(registry)
        location = SourceLocation(line=10, column=5)
        hole = factory.from_placeholder("???", location)
        assert hole.metadata.get("source") == "placeholder"

    def test_create_nested(self) -> None:
        """Test create_nested method."""
        registry = HoleRegistry()
        factory = HoleFactory(registry)
        parent = factory.from_spec(HoleSpec(name="parent"))
        child = factory.create_nested(parent.id, "child")
        assert child.parent == parent.id

    def test_batch_create(self) -> None:
        """Test batch_create method."""
        registry = HoleRegistry()
        factory = HoleFactory(registry)
        specs = [
            HoleSpec(name="hole1"),
            HoleSpec(name="hole2"),
            HoleSpec(name="hole3"),
        ]
        holes = factory.batch_create(specs)
        assert len(holes) == 3
        assert registry.count == 3


# =============================================================================
# Fill-and-Resume Tests
# =============================================================================


class TestFillAndResumeEngine:
    """Tests for FillAndResumeEngine."""

    def test_create(self) -> None:
        """Test engine creation."""
        registry = HoleRegistry()
        engine = FillAndResumeEngine(registry)
        assert engine.state == EvaluationState.READY

    def test_start_no_holes(self) -> None:
        """Test start with no holes."""
        registry = HoleRegistry()
        engine = FillAndResumeEngine(registry)
        result = engine.start_evaluation("code")
        assert result.is_complete

    def test_start_with_hole(self) -> None:
        """Test start with a hole."""
        registry = HoleRegistry()
        hole = registry.create("test")
        engine = FillAndResumeEngine(registry)
        result = engine.start_evaluation("code")
        assert result.is_paused
        assert result.paused_at == hole.id

    def test_fill(self) -> None:
        """Test fill method."""
        registry = HoleRegistry()
        hole = registry.create("test")
        engine = FillAndResumeEngine(registry)
        engine.start_evaluation("code")
        result = engine.fill(hole.id, "return 42")
        assert result.success
        assert result.hole.is_filled

    def test_unfill(self) -> None:
        """Test unfill method."""
        registry = HoleRegistry()
        hole = registry.create("test")
        engine = FillAndResumeEngine(registry)
        engine.start_evaluation("code")
        engine.fill(hole.id, "return 42")
        unfilled = engine.unfill(hole.id)
        assert unfilled is not None
        assert unfilled.is_empty

    def test_resume(self) -> None:
        """Test resume method."""
        registry = HoleRegistry()
        hole = registry.create("test")
        engine = FillAndResumeEngine(registry)
        engine.start_evaluation("code")
        engine.fill(hole.id, "return 42")
        result = engine.resume()
        assert result.is_complete

    def test_checkpoint_restore(self) -> None:
        """Test checkpoint and restore."""
        registry = HoleRegistry()
        hole = registry.create("test")
        engine = FillAndResumeEngine(registry)
        engine.start_evaluation("code")
        checkpoint_idx = engine.checkpoint()
        engine.fill(hole.id, "return 42")
        assert engine.registry.filled_count == 1
        engine.restore(checkpoint_idx)
        assert engine.registry.empty_count == 1

    def test_rollback(self) -> None:
        """Test rollback method."""
        registry = HoleRegistry()
        hole = registry.create("test")
        engine = FillAndResumeEngine(registry)
        engine.start_evaluation("code")
        engine.checkpoint()
        engine.fill(hole.id, "return 42")
        assert engine.rollback()
        assert engine.registry.empty_count == 1

    def test_get_unfilled_holes(self) -> None:
        """Test get_unfilled_holes method."""
        registry = HoleRegistry()
        hole1 = registry.create("hole1")
        hole2 = registry.create("hole2")
        engine = FillAndResumeEngine(registry)
        unfilled = engine.get_unfilled_holes()
        assert len(unfilled) == 2


# =============================================================================
# Strategy Tests
# =============================================================================


class TestDepthFirstStrategy:
    """Tests for DepthFirstStrategy."""

    def test_select_deepest(self) -> None:
        """Test selecting deepest hole."""
        registry = HoleRegistry()
        parent = registry.create("parent")
        child = registry.create("child", parent=parent.id)
        strategy = depth_first()
        holes = list(registry.empty_holes())
        selected = strategy.select(holes)
        assert selected is not None
        assert selected.id == child.id

    def test_rank(self) -> None:
        """Test ranking by depth."""
        registry = HoleRegistry()
        shallow = registry.create("shallow")
        deep = registry.create("deep", parent=shallow.id)
        strategy = depth_first()
        holes = list(registry.empty_holes())
        ranked = strategy.rank(holes)
        assert ranked[0].id == deep.id


class TestBreadthFirstStrategy:
    """Tests for BreadthFirstStrategy."""

    def test_select_shallowest(self) -> None:
        """Test selecting shallowest hole."""
        registry = HoleRegistry()
        parent = registry.create("parent")
        child = registry.create("child", parent=parent.id)
        strategy = breadth_first()
        holes = list(registry.empty_holes())
        selected = strategy.select(holes)
        assert selected is not None
        assert selected.id == parent.id


class TestSourceOrderStrategy:
    """Tests for SourceOrderStrategy."""

    def test_select_by_location(self) -> None:
        """Test selecting by source location."""
        registry = HoleRegistry()
        factory = HoleFactory(registry)
        h1 = factory.from_ast_gap(SourceLocation(line=20, column=0))
        h2 = factory.from_ast_gap(SourceLocation(line=10, column=0))
        strategy = source_order()
        holes = list(registry.empty_holes())
        selected = strategy.select(holes)
        assert selected is not None
        assert selected.id == h2.id  # Line 10 comes first


class TestTypeGuidedStrategy:
    """Tests for TypeGuidedStrategy."""

    def test_select_most_constrained(self) -> None:
        """Test selecting most constrained hole."""
        registry = HoleRegistry()
        any_hole = registry.create("any", expected_type="Any")
        int_hole = registry.create("int", expected_type="int")
        strategy = type_guided()
        holes = list(registry.empty_holes())
        selected = strategy.select(holes)
        # int is more constrained than Any
        assert selected is not None
        assert selected.expected_type == "int"


class TestPriorityStrategy:
    """Tests for PriorityStrategy."""

    def test_select_highest_priority(self) -> None:
        """Test selecting highest priority."""
        registry = HoleRegistry()
        h1 = registry.create("low")
        h2 = registry.create("high")
        strategy = priority_based()
        strategy.set_priority(h1.id, 1)
        strategy.set_priority(h2.id, 10)
        holes = list(registry.empty_holes())
        selected = strategy.select(holes)
        assert selected is not None
        assert selected.id == h2.id


class TestGranularityStrategy:
    """Tests for GranularityStrategy."""

    def test_smaller_first(self) -> None:
        """Test smaller granularity first."""
        registry = HoleRegistry()
        big = registry.create("big", granularity=HoleGranularity.BLOCK)
        small = registry.create("small", granularity=HoleGranularity.TOKEN)
        strategy = granularity_based(larger_first=False)
        holes = list(registry.empty_holes())
        selected = strategy.select(holes)
        assert selected is not None
        assert selected.granularity == HoleGranularity.TOKEN

    def test_larger_first(self) -> None:
        """Test larger granularity first."""
        registry = HoleRegistry()
        big = registry.create("big", granularity=HoleGranularity.BLOCK)
        small = registry.create("small", granularity=HoleGranularity.TOKEN)
        strategy = granularity_based(larger_first=True)
        holes = list(registry.empty_holes())
        selected = strategy.select(holes)
        assert selected is not None
        assert selected.granularity == HoleGranularity.BLOCK


class TestCompositeStrategy:
    """Tests for CompositeStrategy."""

    def test_fallback(self) -> None:
        """Test fallback between strategies."""
        strategy = CompositeStrategy([
            type_guided(),
            source_order(),
        ])
        registry = HoleRegistry()
        hole = registry.create("test")
        holes = list(registry.empty_holes())
        selected = strategy.select(holes)
        assert selected is not None


class TestDefaultStrategy:
    """Tests for default_strategy."""

    def test_creates_composite(self) -> None:
        """Test default_strategy returns CompositeStrategy."""
        strategy = default_strategy()
        assert isinstance(strategy, CompositeStrategy)
