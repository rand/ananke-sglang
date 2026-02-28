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
"""Dependency graph for incremental bidirectional typing.

This module tracks type dependencies between AST nodes, enabling minimal
recomputation when the AST changes (e.g., when a new token is generated).

The key insight from OOPSLA 2025 is that bidirectional typing creates
two kinds of dependencies:
1. Synthesis dependencies: A node's type depends on its children's types
2. Analysis dependencies: A node's type depends on context from ancestors

By tracking both directions, we can:
- Identify exactly which nodes need rechecking after an edit
- Stop propagation at unchanged type boundaries
- Achieve O(k) rechecking where k = affected nodes << n total nodes

References:
    - OOPSLA 2025: "Incremental Bidirectional Typing via Order Maintenance"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Dict,
    FrozenSet,
    Generic,
    Iterator,
    List,
    Optional,
    Set,
    TypeVar,
)

from domains.types.marking.provenance import SourceSpan


class DependencyKind(Enum):
    """The kind of type dependency between nodes.

    Attributes:
        SYNTHESIS: Child-to-parent dependency (type flows up)
        ANALYSIS: Parent-to-child dependency (type flows down)
        ENVIRONMENT: Dependency on type environment bindings
        SUBSUMPTION: Dependency from subsumption check
    """

    SYNTHESIS = auto()  # Type synthesized from children
    ANALYSIS = auto()  # Type analyzed from parent context
    ENVIRONMENT = auto()  # Depends on environment binding
    SUBSUMPTION = auto()  # Depends on subsumption relationship


@dataclass(frozen=True, slots=True)
class NodeId:
    """Unique identifier for an AST node.

    We use a combination of span and node kind to identify nodes,
    as this survives incremental parsing better than pointer identity.

    Attributes:
        span: Source location of the node
        kind: The syntactic kind of node (e.g., "call", "lambda")
        index: Disambiguator for nodes at same location
    """

    span: SourceSpan
    kind: str
    index: int = 0

    def __repr__(self) -> str:
        return f"NodeId({self.span}, {self.kind!r})"


@dataclass(frozen=True, slots=True)
class Dependency:
    """A dependency between two AST nodes in the type graph.

    Attributes:
        source: The node that depends on the target
        target: The node being depended upon
        kind: The kind of dependency
        binding_name: For ENVIRONMENT deps, the name being depended on
    """

    source: NodeId
    target: NodeId
    kind: DependencyKind
    binding_name: Optional[str] = None

    def __repr__(self) -> str:
        base = f"Dependency({self.source} -> {self.target}, {self.kind.name}"
        if self.binding_name:
            base += f", {self.binding_name!r}"
        return base + ")"


T = TypeVar("T")


@dataclass
class DependencyGraph:
    """Graph tracking type dependencies between AST nodes.

    Supports bidirectional dependency tracking for incremental typing:
    - Forward edges: What does this node depend on?
    - Backward edges: What nodes depend on this node?

    Both directions are needed for efficient invalidation:
    - When a node changes, find dependents via backward edges
    - When checking validity, traverse dependencies via forward edges

    Attributes:
        _forward: Map from node to its dependencies
        _backward: Map from node to its dependents
        _env_deps: Map from binding name to nodes depending on it
    """

    _forward: Dict[NodeId, Set[Dependency]] = field(default_factory=dict)
    _backward: Dict[NodeId, Set[Dependency]] = field(default_factory=dict)
    _env_deps: Dict[str, Set[NodeId]] = field(default_factory=dict)

    def add_dependency(
        self,
        source: NodeId,
        target: NodeId,
        kind: DependencyKind,
        binding_name: Optional[str] = None,
    ) -> Dependency:
        """Add a dependency from source to target.

        Args:
            source: The node that depends on target
            target: The node being depended upon
            kind: The kind of dependency
            binding_name: For ENVIRONMENT deps, the binding name

        Returns:
            The created Dependency
        """
        dep = Dependency(
            source=source,
            target=target,
            kind=kind,
            binding_name=binding_name,
        )

        # Forward edge: source -> target
        if source not in self._forward:
            self._forward[source] = set()
        self._forward[source].add(dep)

        # Backward edge: target <- source
        if target not in self._backward:
            self._backward[target] = set()
        self._backward[target].add(dep)

        # Environment index
        if kind == DependencyKind.ENVIRONMENT and binding_name:
            if binding_name not in self._env_deps:
                self._env_deps[binding_name] = set()
            self._env_deps[binding_name].add(source)

        return dep

    def add_synthesis_dependency(
        self, parent: NodeId, child: NodeId
    ) -> Dependency:
        """Add a synthesis (upward) dependency.

        The parent's type is synthesized from the child's type.

        Args:
            parent: The parent node (synthesizer)
            child: The child node (provides type info)

        Returns:
            The created Dependency
        """
        return self.add_dependency(parent, child, DependencyKind.SYNTHESIS)

    def add_analysis_dependency(
        self, child: NodeId, parent: NodeId
    ) -> Dependency:
        """Add an analysis (downward) dependency.

        The child's type is analyzed against context from the parent.

        Args:
            child: The child node (analyzee)
            parent: The parent node (provides context)

        Returns:
            The created Dependency
        """
        return self.add_dependency(child, parent, DependencyKind.ANALYSIS)

    def add_environment_dependency(
        self, node: NodeId, binding_name: str, binding_node: NodeId
    ) -> Dependency:
        """Add an environment binding dependency.

        The node depends on a binding in the type environment.

        Args:
            node: The node using the binding
            binding_name: The name of the binding
            binding_node: The node that defines the binding

        Returns:
            The created Dependency
        """
        return self.add_dependency(
            node,
            binding_node,
            DependencyKind.ENVIRONMENT,
            binding_name=binding_name,
        )

    def remove_node(self, node: NodeId) -> None:
        """Remove a node and all its dependencies.

        Args:
            node: The node to remove
        """
        # Remove forward edges from this node
        if node in self._forward:
            for dep in self._forward[node]:
                # Remove backward edge at target
                if dep.target in self._backward:
                    self._backward[dep.target].discard(dep)
                # Remove from env index
                if dep.binding_name and dep.binding_name in self._env_deps:
                    self._env_deps[dep.binding_name].discard(node)
            del self._forward[node]

        # Remove backward edges to this node
        if node in self._backward:
            for dep in self._backward[node]:
                # Remove forward edge at source
                if dep.source in self._forward:
                    self._forward[dep.source].discard(dep)
            del self._backward[node]

    def dependencies_of(self, node: NodeId) -> FrozenSet[Dependency]:
        """Get all dependencies of a node (what it depends on).

        Args:
            node: The node to query

        Returns:
            Set of dependencies from this node
        """
        return frozenset(self._forward.get(node, set()))

    def dependents_of(self, node: NodeId) -> FrozenSet[Dependency]:
        """Get all dependents of a node (what depends on it).

        Args:
            node: The node to query

        Returns:
            Set of dependencies to this node
        """
        return frozenset(self._backward.get(node, set()))

    def nodes_depending_on_binding(self, binding_name: str) -> FrozenSet[NodeId]:
        """Get all nodes that depend on a binding name.

        Args:
            binding_name: The binding name to query

        Returns:
            Set of nodes depending on this binding
        """
        return frozenset(self._env_deps.get(binding_name, set()))

    def affected_by(self, edit_span: SourceSpan) -> FrozenSet[NodeId]:
        """Find all nodes affected by an edit at a source span.

        This is the core incremental typing query. Given an edit location,
        find all nodes whose types may need rechecking.

        The algorithm:
        1. Find nodes overlapping the edit span (directly affected)
        2. Traverse backward edges to find dependents (transitively affected)
        3. Stop at nodes whose types won't change

        Args:
            edit_span: The source span where an edit occurred

        Returns:
            Set of node IDs that may need rechecking
        """
        # Phase 1: Find directly affected nodes
        directly_affected: Set[NodeId] = set()
        for node in self._all_nodes():
            if node.span.overlaps(edit_span):
                directly_affected.add(node)

        # Phase 2: Compute transitive closure via backward edges
        affected: Set[NodeId] = set(directly_affected)
        worklist: List[NodeId] = list(directly_affected)

        while worklist:
            current = worklist.pop()

            # Find all nodes that depend on current
            for dep in self._backward.get(current, set()):
                if dep.source not in affected:
                    affected.add(dep.source)
                    worklist.append(dep.source)

        return frozenset(affected)

    def affected_by_binding_change(self, binding_name: str) -> FrozenSet[NodeId]:
        """Find all nodes affected when a binding's type changes.

        Args:
            binding_name: The binding that changed

        Returns:
            Set of affected node IDs
        """
        # Start with nodes directly depending on the binding
        directly_affected = self._env_deps.get(binding_name, set())

        # Compute transitive closure
        affected: Set[NodeId] = set(directly_affected)
        worklist: List[NodeId] = list(directly_affected)

        while worklist:
            current = worklist.pop()
            for dep in self._backward.get(current, set()):
                if dep.source not in affected:
                    affected.add(dep.source)
                    worklist.append(dep.source)

        return frozenset(affected)

    def topological_order(
        self, nodes: FrozenSet[NodeId]
    ) -> List[NodeId]:
        """Sort nodes in topological order based on dependencies.

        Nodes are ordered so that a node comes after all its dependencies.
        This is important for incremental rechecking - we must recheck
        dependencies before dependents.

        Args:
            nodes: The nodes to sort

        Returns:
            Nodes in topological order

        Raises:
            ValueError: If there's a cycle in the dependencies
        """
        # Compute in-degree for each node (restricted to the subset)
        in_degree: Dict[NodeId, int] = {n: 0 for n in nodes}

        for node in nodes:
            for dep in self._forward.get(node, set()):
                if dep.target in nodes:
                    in_degree[node] += 1

        # Kahn's algorithm
        result: List[NodeId] = []
        ready: List[NodeId] = [n for n, d in in_degree.items() if d == 0]

        while ready:
            current = ready.pop()
            result.append(current)

            # For each dependent of current (in the subset)
            for dep in self._backward.get(current, set()):
                if dep.source in in_degree:
                    in_degree[dep.source] -= 1
                    if in_degree[dep.source] == 0:
                        ready.append(dep.source)

        if len(result) != len(nodes):
            raise ValueError("Cycle detected in dependency graph")

        return result

    def _all_nodes(self) -> Iterator[NodeId]:
        """Iterate over all known nodes."""
        seen: Set[NodeId] = set()
        for node in self._forward:
            if node not in seen:
                seen.add(node)
                yield node
        for node in self._backward:
            if node not in seen:
                seen.add(node)
                yield node

    def node_count(self) -> int:
        """Get the number of nodes in the graph."""
        return len(set(self._forward.keys()) | set(self._backward.keys()))

    def edge_count(self) -> int:
        """Get the number of edges in the graph."""
        return sum(len(deps) for deps in self._forward.values())

    def clear(self) -> None:
        """Remove all nodes and dependencies."""
        self._forward.clear()
        self._backward.clear()
        self._env_deps.clear()

    def __repr__(self) -> str:
        return f"DependencyGraph(nodes={self.node_count()}, edges={self.edge_count()})"


@dataclass
class DependencyGraphSnapshot:
    """A snapshot of a dependency graph for checkpointing.

    Enables save/restore of dependency state for rollback during generation.

    Attributes:
        forward: Frozen forward edges
        backward: Frozen backward edges
        env_deps: Frozen environment dependency index
    """

    forward: Dict[NodeId, FrozenSet[Dependency]]
    backward: Dict[NodeId, FrozenSet[Dependency]]
    env_deps: Dict[str, FrozenSet[NodeId]]


def snapshot_graph(graph: DependencyGraph) -> DependencyGraphSnapshot:
    """Create a snapshot of a dependency graph.

    Args:
        graph: The graph to snapshot

    Returns:
        An immutable snapshot
    """
    return DependencyGraphSnapshot(
        forward={k: frozenset(v) for k, v in graph._forward.items()},
        backward={k: frozenset(v) for k, v in graph._backward.items()},
        env_deps={k: frozenset(v) for k, v in graph._env_deps.items()},
    )


def restore_graph(snapshot: DependencyGraphSnapshot) -> DependencyGraph:
    """Restore a dependency graph from a snapshot.

    Args:
        snapshot: The snapshot to restore

    Returns:
        A new DependencyGraph with the snapshot's state
    """
    return DependencyGraph(
        _forward={k: set(v) for k, v in snapshot.forward.items()},
        _backward={k: set(v) for k, v in snapshot.backward.items()},
        _env_deps={k: set(v) for k, v in snapshot.env_deps.items()},
    )


def create_dependency_graph() -> DependencyGraph:
    """Create an empty dependency graph.

    Returns:
        A new empty DependencyGraph
    """
    return DependencyGraph()
