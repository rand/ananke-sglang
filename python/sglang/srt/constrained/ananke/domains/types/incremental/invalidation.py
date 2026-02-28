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
"""Invalidation engine for incremental type checking.

This module handles the invalidation of type cache entries when edits occur.
The key optimization is *selective invalidation* - we only invalidate nodes
whose types could actually change, stopping propagation at boundaries where
the type is guaranteed to remain the same.

The invalidation algorithm:
1. Find nodes directly affected by the edit (overlapping spans)
2. Propagate invalidation along dependency edges
3. Stop at "stable" boundaries where type won't change
4. Return the minimal set of nodes to recheck

Boundaries that stop propagation:
- Explicit type annotations (type is fixed)
- Nodes with fully-concrete types (no type variables)
- Nodes outside the dependency closure

References:
    - OOPSLA 2025: "Incremental Bidirectional Typing via Order Maintenance"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Callable,
    Dict,
    FrozenSet,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
)

from domains.types.constraint import Type, TypeVar as TypeVariable
from domains.types.marking.provenance import SourceSpan
from domains.types.incremental.dependency_graph import (
    DependencyGraph,
    DependencyKind,
    NodeId,
)


class StabilityKind(Enum):
    """Why a node's type is considered stable.

    Attributes:
        ANNOTATED: Type was explicitly annotated by user
        CONCRETE: Type is fully concrete (no variables)
        PRIMITIVE: Type is a primitive type
        UNCHANGED: Edit didn't affect this node's type
    """

    ANNOTATED = auto()
    CONCRETE = auto()
    PRIMITIVE = auto()
    UNCHANGED = auto()


@dataclass(frozen=True, slots=True)
class InvalidationResult:
    """Result of invalidation analysis.

    Attributes:
        invalidated: Nodes that were invalidated
        stable_boundaries: Nodes where propagation stopped
        directly_affected: Nodes directly overlapping the edit
    """

    invalidated: FrozenSet[NodeId]
    stable_boundaries: FrozenSet[Tuple[NodeId, StabilityKind]]
    directly_affected: FrozenSet[NodeId]

    @property
    def count(self) -> int:
        """Number of invalidated nodes."""
        return len(self.invalidated)


class StabilityChecker(Protocol):
    """Protocol for checking if a node's type is stable."""

    def is_stable(self, node_id: NodeId) -> Optional[StabilityKind]:
        """Check if a node's type is stable.

        Args:
            node_id: The node to check

        Returns:
            The kind of stability, or None if not stable
        """
        ...


def is_type_concrete(typ: Type) -> bool:
    """Check if a type contains no type variables.

    Args:
        typ: The type to check

    Returns:
        True if the type is fully concrete
    """
    # Use the type's free_type_vars method if available
    # This is the most reliable way to check for type variables
    if hasattr(typ, "free_type_vars"):
        return len(typ.free_type_vars()) == 0

    # Fallback for types without free_type_vars
    # Check for type variables by type name
    if isinstance(typ, TypeVariable):
        return False

    # Primitive types are concrete
    return True


@dataclass
class InvalidationEngine:
    """Engine for selective invalidation of type cache entries.

    This class implements the invalidation algorithm that determines
    which cached type results need to be recomputed after an edit.

    Attributes:
        _dependencies: The dependency graph
        _stability_checker: Optional checker for type stability
        _annotated_nodes: Set of nodes with explicit type annotations
        _cached_types: Map from node to its cached type
    """

    _dependencies: DependencyGraph
    _stability_checker: Optional[StabilityChecker] = None
    _annotated_nodes: Set[NodeId] = field(default_factory=set)
    _cached_types: Dict[NodeId, Type] = field(default_factory=dict)

    def set_annotated(self, node_id: NodeId) -> None:
        """Mark a node as having an explicit type annotation.

        Args:
            node_id: The annotated node
        """
        self._annotated_nodes.add(node_id)

    def set_cached_type(self, node_id: NodeId, typ: Type) -> None:
        """Update the cached type for a node.

        Args:
            node_id: The node
            typ: Its type
        """
        self._cached_types[node_id] = typ

    def remove_node(self, node_id: NodeId) -> None:
        """Remove a node from tracking.

        Args:
            node_id: The node to remove
        """
        self._annotated_nodes.discard(node_id)
        self._cached_types.pop(node_id, None)

    def check_stability(self, node_id: NodeId) -> Optional[StabilityKind]:
        """Check if a node's type is stable.

        A stable node's type won't change even if dependencies change.

        Args:
            node_id: The node to check

        Returns:
            The kind of stability, or None if not stable
        """
        # Check custom stability checker first
        if self._stability_checker:
            result = self._stability_checker.is_stable(node_id)
            if result:
                return result

        # Check for explicit annotation
        if node_id in self._annotated_nodes:
            return StabilityKind.ANNOTATED

        # Check for concrete cached type
        if node_id in self._cached_types:
            typ = self._cached_types[node_id]
            if is_type_concrete(typ):
                return StabilityKind.CONCRETE

        return None

    def invalidate(self, edit_span: SourceSpan) -> InvalidationResult:
        """Determine which nodes to invalidate after an edit.

        Args:
            edit_span: The source span where the edit occurred

        Returns:
            InvalidationResult with invalidated nodes and boundaries
        """
        # Phase 1: Find directly affected nodes (only those overlapping the edit span)
        directly_affected: Set[NodeId] = set()
        for node in self._all_known_nodes():
            if node.span.overlaps(edit_span):
                directly_affected.add(node)

        # Phase 2: Propagate with stability checking
        invalidated: Set[NodeId] = set()
        stable_boundaries: Set[Tuple[NodeId, StabilityKind]] = set()
        worklist: List[NodeId] = list(directly_affected)
        visited: Set[NodeId] = set()

        while worklist:
            current = worklist.pop()

            if current in visited:
                continue
            visited.add(current)

            # Check stability - only for non-directly-affected nodes
            # Directly affected nodes are always invalidated
            is_direct = current in directly_affected
            if not is_direct:
                stability = self.check_stability(current)
                if stability:
                    # Stable boundary - don't invalidate or propagate
                    stable_boundaries.add((current, stability))
                    continue

            # Invalidate this node
            invalidated.add(current)

            # Propagate to dependents
            for dep in self._dependencies.dependents_of(current):
                if dep.source not in visited:
                    worklist.append(dep.source)

        return InvalidationResult(
            invalidated=frozenset(invalidated),
            stable_boundaries=frozenset(stable_boundaries),
            directly_affected=frozenset(directly_affected),
        )

    def _all_known_nodes(self) -> Set[NodeId]:
        """Get all nodes known to the invalidation engine."""
        nodes: Set[NodeId] = set()
        nodes.update(self._annotated_nodes)
        nodes.update(self._cached_types.keys())
        # Also get nodes from dependency graph
        for node in self._dependencies._forward.keys():
            nodes.add(node)
        for node in self._dependencies._backward.keys():
            nodes.add(node)
        return nodes

    def invalidate_binding(self, binding_name: str) -> InvalidationResult:
        """Determine which nodes to invalidate when a binding changes.

        Args:
            binding_name: The name of the changed binding

        Returns:
            InvalidationResult with invalidated nodes
        """
        # Get nodes depending on this binding
        directly_affected = self._dependencies.nodes_depending_on_binding(binding_name)

        # Propagate with stability checking
        invalidated: Set[NodeId] = set()
        stable_boundaries: Set[Tuple[NodeId, StabilityKind]] = set()
        worklist: List[NodeId] = list(directly_affected)
        visited: Set[NodeId] = set()

        while worklist:
            current = worklist.pop()

            if current in visited:
                continue
            visited.add(current)

            # Check stability
            stability = self.check_stability(current)
            if stability and current not in directly_affected:
                stable_boundaries.add((current, stability))
                continue

            invalidated.add(current)

            for dep in self._dependencies.dependents_of(current):
                if dep.source not in visited:
                    worklist.append(dep.source)

        return InvalidationResult(
            invalidated=frozenset(invalidated),
            stable_boundaries=frozenset(stable_boundaries),
            directly_affected=frozenset(directly_affected),
        )

    def invalidate_node(self, node_id: NodeId) -> InvalidationResult:
        """Invalidate a specific node and its dependents.

        Args:
            node_id: The node to invalidate

        Returns:
            InvalidationResult with invalidated nodes
        """
        directly_affected = frozenset({node_id})

        invalidated: Set[NodeId] = set()
        stable_boundaries: Set[Tuple[NodeId, StabilityKind]] = set()
        worklist: List[NodeId] = [node_id]
        visited: Set[NodeId] = set()

        while worklist:
            current = worklist.pop()

            if current in visited:
                continue
            visited.add(current)

            stability = self.check_stability(current)
            if stability and current != node_id:
                stable_boundaries.add((current, stability))
                continue

            invalidated.add(current)

            for dep in self._dependencies.dependents_of(current):
                if dep.source not in visited:
                    worklist.append(dep.source)

        return InvalidationResult(
            invalidated=frozenset(invalidated),
            stable_boundaries=frozenset(stable_boundaries),
            directly_affected=directly_affected,
        )

    def clear(self) -> None:
        """Clear all invalidation state."""
        self._annotated_nodes.clear()
        self._cached_types.clear()


@dataclass(frozen=True)
class InvalidationSnapshot:
    """Snapshot of invalidation engine state.

    Attributes:
        annotated_nodes: Frozen set of annotated nodes
        cached_types: Frozen map of cached types
    """

    annotated_nodes: FrozenSet[NodeId]
    cached_types: Dict[NodeId, Type]


def snapshot_invalidation(engine: InvalidationEngine) -> InvalidationSnapshot:
    """Create a snapshot of invalidation engine state.

    Args:
        engine: The engine to snapshot

    Returns:
        An immutable snapshot
    """
    return InvalidationSnapshot(
        annotated_nodes=frozenset(engine._annotated_nodes),
        cached_types=dict(engine._cached_types),
    )


def restore_invalidation(
    snapshot: InvalidationSnapshot,
    dependencies: DependencyGraph,
) -> InvalidationEngine:
    """Restore an invalidation engine from a snapshot.

    Args:
        snapshot: The snapshot to restore
        dependencies: The dependency graph

    Returns:
        A restored InvalidationEngine
    """
    return InvalidationEngine(
        _dependencies=dependencies,
        _annotated_nodes=set(snapshot.annotated_nodes),
        _cached_types=dict(snapshot.cached_types),
    )


def create_invalidation_engine(
    dependencies: DependencyGraph,
    stability_checker: Optional[StabilityChecker] = None,
) -> InvalidationEngine:
    """Create a new invalidation engine.

    Args:
        dependencies: The dependency graph
        stability_checker: Optional custom stability checker

    Returns:
        A new InvalidationEngine
    """
    return InvalidationEngine(
        _dependencies=dependencies,
        _stability_checker=stability_checker,
    )
