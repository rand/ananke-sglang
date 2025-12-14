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
"""Delta typing engine for incremental bidirectional type checking.

This module implements the core incremental typing algorithm from OOPSLA 2025.
When a token is generated, we:
1. Update the partial AST incrementally
2. Identify affected nodes via the dependency graph
3. Recheck only those nodes in dependency order
4. Update the marked AST with new types/marks

The key data structure is the TypeCache, which stores:
- Synthesized types at each node
- Analysis results (marks) at each node
- Dependencies discovered during checking

References:
    - OOPSLA 2025: "Incremental Bidirectional Typing via Order Maintenance"
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
)

from domains.types.constraint import Type, AnyType, NeverType, ANY, NEVER
from domains.types.environment import TypeEnvironment, EMPTY_ENVIRONMENT
from domains.types.marking.marks import Mark, HoleMark, InconsistentMark
from domains.types.marking.provenance import SourceSpan, UNKNOWN_SPAN
from domains.types.incremental.dependency_graph import (
    DependencyGraph,
    DependencyKind,
    NodeId,
    create_dependency_graph,
)
from domains.types.incremental.order_maintenance import (
    OrderMaintenanceList,
    OrderedElement,
    create_order_list,
)


class CheckMode(Enum):
    """The mode of type checking for a node.

    Attributes:
        SYNTHESIS: Type is synthesized from the node
        ANALYSIS: Type is analyzed against an expected type
    """

    SYNTHESIS = auto()
    ANALYSIS = auto()


@dataclass
class TypeCacheEntry:
    """Cached type information for an AST node.

    Attributes:
        node_id: Identifier for the AST node
        mode: Whether type was synthesized or analyzed
        synthesized_type: The synthesized type (if any)
        expected_type: The expected type for analysis (if any)
        mark: Any mark attached to this node
        environment: The type environment at this node
        is_dirty: Whether this entry needs rechecking
    """

    node_id: NodeId
    mode: CheckMode
    synthesized_type: Optional[Type] = None
    expected_type: Optional[Type] = None
    mark: Optional[Mark] = None
    environment: TypeEnvironment = field(default_factory=lambda: EMPTY_ENVIRONMENT)
    is_dirty: bool = True


@dataclass
class TypeCacheStats:
    """Statistics for type cache performance.

    Attributes:
        hits: Number of cache hits
        misses: Number of cache misses
        evictions: Number of entries evicted due to capacity
        re_checks: Number of entries that were rechecked after invalidation
    """

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    re_checks: int = 0

    @property
    def hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def reset(self) -> None:
        """Reset all statistics to zero."""
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.re_checks = 0


# Default maximum size for TypeCache (5000 entries ~ 3-4 MB)
DEFAULT_TYPE_CACHE_MAX_SIZE = 5000


class TypeCache:
    """Cache of type checking results for incremental reuse.

    The cache stores:
    - Type checking results at each node
    - An order maintenance list for dependency ordering
    - The dependency graph linking nodes

    Uses LRU eviction to bound memory growth during long generation sequences.
    Following the pattern from MaskCache, uses OrderedDict for O(1) access ordering.

    Attributes:
        _entries: Ordered map from node ID to cache entry (LRU order)
        _order: Order maintenance structure for topological ordering
        _order_elements: Map from node ID to order element
        _dependencies: The dependency graph
        _max_size: Maximum number of entries before eviction
        _stats: Cache statistics
    """

    def __init__(self, max_size: int = DEFAULT_TYPE_CACHE_MAX_SIZE) -> None:
        """Initialize the type cache.

        Args:
            max_size: Maximum number of entries (default 5000)
        """
        self._entries: OrderedDict[NodeId, TypeCacheEntry] = OrderedDict()
        self._order: OrderMaintenanceList[NodeId] = create_order_list()
        self._order_elements: Dict[NodeId, OrderedElement[NodeId]] = {}
        self._dependencies: DependencyGraph = create_dependency_graph()
        self._max_size = max_size
        self._stats = TypeCacheStats()

    @property
    def max_size(self) -> int:
        """Get maximum cache size."""
        return self._max_size

    @property
    def stats(self) -> TypeCacheStats:
        """Get cache statistics."""
        return self._stats

    def get(self, node_id: NodeId) -> Optional[TypeCacheEntry]:
        """Get the cache entry for a node.

        Updates LRU order on access (moves to end = most recently used).

        Args:
            node_id: The node to look up

        Returns:
            The cache entry, or None if not cached
        """
        entry = self._entries.get(node_id)
        if entry is None:
            self._stats.misses += 1
            return None

        # Move to end (most recently used) on access
        self._entries.move_to_end(node_id)
        self._stats.hits += 1
        return entry

    def get_type(self, node_id: NodeId) -> Optional[Type]:
        """Get the synthesized type for a node.

        Does not update LRU order (use get() for full entry access).

        Args:
            node_id: The node to look up

        Returns:
            The synthesized type, or None if not cached
        """
        entry = self._entries.get(node_id)
        if entry is not None:
            # Move to end on access
            self._entries.move_to_end(node_id)
        return entry.synthesized_type if entry else None

    def _evict_if_needed(self) -> None:
        """Evict least recently used entries if at capacity.

        Removes entries from the front of the OrderedDict (oldest access).
        Also cleans up order maintenance and dependency tracking.
        """
        while len(self._entries) >= self._max_size:
            # Remove least recently used (front of OrderedDict)
            evicted_node_id, _ = self._entries.popitem(last=False)
            self._stats.evictions += 1

            # Clean up order maintenance
            if evicted_node_id in self._order_elements:
                self._order.delete(self._order_elements[evicted_node_id])
                del self._order_elements[evicted_node_id]

            # Clean up dependency graph
            self._dependencies.remove_node(evicted_node_id)

    def set(
        self,
        node_id: NodeId,
        mode: CheckMode,
        synthesized_type: Optional[Type] = None,
        expected_type: Optional[Type] = None,
        mark: Optional[Mark] = None,
        environment: Optional[TypeEnvironment] = None,
    ) -> TypeCacheEntry:
        """Set or update a cache entry.

        Evicts LRU entries if at capacity before adding new entries.

        Args:
            node_id: The node to cache
            mode: Synthesis or analysis mode
            synthesized_type: The synthesized type
            expected_type: The expected type (for analysis)
            mark: Any mark on this node
            environment: The type environment

        Returns:
            The created or updated cache entry
        """
        # Evict if needed before adding new entry
        if node_id not in self._entries:
            self._evict_if_needed()

        entry = TypeCacheEntry(
            node_id=node_id,
            mode=mode,
            synthesized_type=synthesized_type,
            expected_type=expected_type,
            mark=mark,
            environment=environment or EMPTY_ENVIRONMENT,
            is_dirty=False,
        )
        self._entries[node_id] = entry

        # Move to end (most recently used)
        self._entries.move_to_end(node_id)

        # Add to order maintenance if new
        if node_id not in self._order_elements:
            if self._order.is_empty():
                elem = self._order.insert_first(node_id)
            else:
                # Insert at end (will be reordered as dependencies are discovered)
                elem = self._order.insert_last(node_id)
            self._order_elements[node_id] = elem

        return entry

    def invalidate(self, node_id: NodeId) -> None:
        """Mark a node as needing rechecking.

        Args:
            node_id: The node to invalidate
        """
        if node_id in self._entries:
            self._entries[node_id].is_dirty = True

    def remove(self, node_id: NodeId) -> None:
        """Remove a node from the cache.

        Args:
            node_id: The node to remove
        """
        self._entries.pop(node_id, None)
        if node_id in self._order_elements:
            self._order.delete(self._order_elements[node_id])
            del self._order_elements[node_id]
        self._dependencies.remove_node(node_id)

    def is_valid(self, node_id: NodeId) -> bool:
        """Check if a node's cache entry is still valid.

        Args:
            node_id: The node to check

        Returns:
            True if the entry exists and is not dirty
        """
        entry = self._entries.get(node_id)
        return entry is not None and not entry.is_dirty

    def add_dependency(
        self,
        source: NodeId,
        target: NodeId,
        kind: DependencyKind,
    ) -> None:
        """Add a dependency and update ordering.

        Args:
            source: The dependent node
            target: The dependency
            kind: The kind of dependency
        """
        self._dependencies.add_dependency(source, target, kind)

    def dirty_nodes(self) -> List[NodeId]:
        """Get all dirty nodes in dependency order.

        Returns:
            Dirty nodes ordered so dependencies come first
        """
        dirty = [
            node_id
            for node_id, entry in self._entries.items()
            if entry.is_dirty
        ]

        # Sort by order maintenance
        def get_order_key(node_id: NodeId) -> int:
            elem = self._order_elements.get(node_id)
            return elem.label if elem else 0

        return sorted(dirty, key=get_order_key)

    @property
    def dependencies(self) -> DependencyGraph:
        """Get the dependency graph."""
        return self._dependencies

    def clear(self) -> None:
        """Clear all cached entries (does not reset stats)."""
        self._entries.clear()
        self._order = create_order_list()
        self._order_elements.clear()
        self._dependencies.clear()

    def reset_stats(self) -> None:
        """Reset cache statistics."""
        self._stats.reset()

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        valid = sum(1 for e in self._entries.values() if not e.is_dirty)
        dirty = len(self._entries) - valid
        return (
            f"TypeCache(size={len(self._entries)}/{self._max_size}, "
            f"valid={valid}, dirty={dirty}, "
            f"hit_rate={self._stats.hit_rate:.2%})"
        )


@dataclass
class DeltaTypingResult:
    """Result of incremental type checking.

    Attributes:
        rechecked_count: Number of nodes rechecked
        new_errors: New type errors discovered
        resolved_errors: Errors that were resolved
        changed_types: Nodes whose types changed
    """

    rechecked_count: int
    new_errors: List[Tuple[NodeId, InconsistentMark]]
    resolved_errors: List[NodeId]
    changed_types: List[Tuple[NodeId, Type, Type]]  # (node, old_type, new_type)


# Type for the actual type checking function
TypeChecker = Callable[[NodeId, TypeEnvironment, Optional[Type]], Tuple[Type, Optional[Mark]]]


class DeltaTypingEngine:
    """Engine for incremental bidirectional type checking.

    This class coordinates incremental type checking by:
    1. Maintaining a type cache of previous results
    2. Using the dependency graph to identify affected nodes
    3. Rechecking only affected nodes in proper order
    4. Tracking what changed for downstream consumers

    The engine is parameterized by a TypeChecker function that
    performs actual type checking for a single node.
    """

    def __init__(
        self,
        checker: TypeChecker,
        cache_max_size: int = DEFAULT_TYPE_CACHE_MAX_SIZE,
    ):
        """Initialize the delta typing engine.

        Args:
            checker: Function to type-check a single node
            cache_max_size: Maximum size for the type cache (default 5000)
        """
        self._checker = checker
        self._cache = TypeCache(max_size=cache_max_size)

    @property
    def cache(self) -> TypeCache:
        """Get the type cache."""
        return self._cache

    def check_node(
        self,
        node_id: NodeId,
        environment: TypeEnvironment,
        expected: Optional[Type] = None,
    ) -> Tuple[Type, Optional[Mark]]:
        """Check a single node, using cache if valid.

        Args:
            node_id: The node to check
            environment: The type environment
            expected: Optional expected type for analysis

        Returns:
            Tuple of (synthesized_type, mark)
        """
        # Check cache first
        if self._cache.is_valid(node_id):
            entry = self._cache.get(node_id)
            if entry:
                return entry.synthesized_type or ANY, entry.mark

        # Actually check the node
        result_type, mark = self._checker(node_id, environment, expected)

        # Cache the result
        mode = CheckMode.ANALYSIS if expected else CheckMode.SYNTHESIS
        self._cache.set(
            node_id=node_id,
            mode=mode,
            synthesized_type=result_type,
            expected_type=expected,
            mark=mark,
            environment=environment,
        )

        return result_type, mark

    def invalidate_span(self, edit_span: SourceSpan) -> Set[NodeId]:
        """Invalidate nodes affected by an edit.

        Args:
            edit_span: The span where an edit occurred

        Returns:
            Set of invalidated node IDs
        """
        # Find affected nodes via dependency graph
        affected = set(self._cache.dependencies.affected_by(edit_span))

        # Also check cached nodes directly overlapping the edit span
        for node_id in self._cache._entries.keys():
            if node_id.span.overlaps(edit_span):
                affected.add(node_id)

        # Invalidate all affected nodes
        for node_id in affected:
            self._cache.invalidate(node_id)
        return affected

    def recheck_dirty(
        self,
        environment: TypeEnvironment,
    ) -> DeltaTypingResult:
        """Recheck all dirty nodes.

        Args:
            environment: The base type environment

        Returns:
            Results including what changed
        """
        dirty_nodes = self._cache.dirty_nodes()
        rechecked = 0
        new_errors: List[Tuple[NodeId, InconsistentMark]] = []
        resolved_errors: List[NodeId] = []
        changed_types: List[Tuple[NodeId, Type, Type]] = []

        for node_id in dirty_nodes:
            old_entry = self._cache.get(node_id)
            old_type = old_entry.synthesized_type if old_entry else None
            old_mark = old_entry.mark if old_entry else None
            old_env = old_entry.environment if old_entry else environment
            expected = old_entry.expected_type if old_entry else None

            # Recheck the node
            new_type, new_mark = self._checker(node_id, old_env, expected)
            rechecked += 1

            # Update cache
            mode = CheckMode.ANALYSIS if expected else CheckMode.SYNTHESIS
            self._cache.set(
                node_id=node_id,
                mode=mode,
                synthesized_type=new_type,
                expected_type=expected,
                mark=new_mark,
                environment=old_env,
            )

            # Track changes
            if old_type != new_type:
                changed_types.append((node_id, old_type or ANY, new_type))

            # Track error changes
            was_error = isinstance(old_mark, InconsistentMark)
            is_error = isinstance(new_mark, InconsistentMark)

            if is_error and not was_error:
                new_errors.append((node_id, new_mark))
            elif was_error and not is_error:
                resolved_errors.append(node_id)

        return DeltaTypingResult(
            rechecked_count=rechecked,
            new_errors=new_errors,
            resolved_errors=resolved_errors,
            changed_types=changed_types,
        )

    def apply_edit(
        self,
        edit_span: SourceSpan,
        environment: TypeEnvironment,
    ) -> DeltaTypingResult:
        """Apply an edit and recheck affected nodes.

        This is the main entry point for incremental typing.

        Args:
            edit_span: Where the edit occurred
            environment: The type environment

        Returns:
            Results of the incremental recheck
        """
        # Invalidate affected nodes
        self.invalidate_span(edit_span)

        # Recheck dirty nodes
        return self.recheck_dirty(environment)

    def full_check(
        self,
        node_ids: List[NodeId],
        environment: TypeEnvironment,
    ) -> DeltaTypingResult:
        """Perform a full check of all nodes.

        Used for initial checking before incremental updates begin.

        Args:
            node_ids: All nodes to check (in dependency order)
            environment: The type environment

        Returns:
            Results of the full check
        """
        # Mark all as dirty
        for node_id in node_ids:
            self._cache.set(
                node_id=node_id,
                mode=CheckMode.SYNTHESIS,
                environment=environment,
            )
            self._cache.invalidate(node_id)

        # Recheck all
        return self.recheck_dirty(environment)

    def clear(self) -> None:
        """Clear all cached state."""
        self._cache.clear()


@dataclass
class TypeCacheSnapshot:
    """Snapshot of type cache state for checkpointing.

    Attributes:
        entries: Map from node ID to cache entry data
    """

    entries: Dict[NodeId, Tuple[CheckMode, Optional[Type], Optional[Type], Optional[Mark]]]


def snapshot_cache(cache: TypeCache) -> TypeCacheSnapshot:
    """Create a snapshot of a type cache.

    Args:
        cache: The cache to snapshot

    Returns:
        An immutable snapshot
    """
    entries = {}
    for node_id, entry in cache._entries.items():
        entries[node_id] = (
            entry.mode,
            entry.synthesized_type,
            entry.expected_type,
            entry.mark,
        )
    return TypeCacheSnapshot(entries=entries)


def create_type_cache(max_size: int = DEFAULT_TYPE_CACHE_MAX_SIZE) -> TypeCache:
    """Factory function to create a TypeCache.

    Args:
        max_size: Maximum number of entries (default 5000)

    Returns:
        New TypeCache instance with LRU eviction
    """
    return TypeCache(max_size=max_size)


def create_delta_engine(
    checker: TypeChecker,
    cache_max_size: int = DEFAULT_TYPE_CACHE_MAX_SIZE,
) -> DeltaTypingEngine:
    """Create a new delta typing engine.

    Args:
        checker: The type checking function
        cache_max_size: Maximum size for the type cache

    Returns:
        A new DeltaTypingEngine
    """
    return DeltaTypingEngine(checker, cache_max_size=cache_max_size)
