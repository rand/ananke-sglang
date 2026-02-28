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
"""Incremental bidirectional typing via order maintenance (OOPSLA 2025).

Provides ~275x speedup over naive reanalysis for token-by-token updates.

This package implements the key algorithms from the OOPSLA 2025 paper
"Incremental Bidirectional Typing via Order Maintenance":

1. Order Maintenance (Dietz & Sleator algorithm):
   - O(1) amortized insert and query operations
   - Maintains total ordering over AST nodes
   - Enables efficient dependency-ordered rechecking

2. Dependency Graph:
   - Tracks synthesis and analysis dependencies
   - Bidirectional edges for efficient invalidation
   - Environment binding dependencies

3. Delta Typing Engine:
   - Caches type checking results
   - Incrementally rechecks only affected nodes
   - Tracks errors and type changes

4. Invalidation Engine:
   - Selective invalidation with stability checking
   - Stops at annotated or concrete type boundaries
   - Minimizes recomputation

Usage:
    >>> from domains.types.incremental import (
    ...     create_order_list,
    ...     create_dependency_graph,
    ...     create_delta_engine,
    ...     create_invalidation_engine,
    ... )
    >>>
    >>> # Create order list for node ordering
    >>> order = create_order_list()
    >>> elem1 = order.insert_first("node1")
    >>> elem2 = order.insert_after(elem1, "node2")
    >>> assert order.comes_before(elem1, elem2)
    >>>
    >>> # Create dependency graph
    >>> deps = create_dependency_graph()
    >>> deps.add_synthesis_dependency(parent_id, child_id)
    >>>
    >>> # Create delta typing engine
    >>> engine = create_delta_engine(my_checker)
    >>> result = engine.apply_edit(edit_span, environment)

References:
    - OOPSLA 2025: "Incremental Bidirectional Typing via Order Maintenance"
    - Dietz & Sleator (1987): "Two algorithms for maintaining order in a list"
"""

from domains.types.incremental.order_maintenance import (
    MAX_LABEL,
    OrderedElement,
    OrderMaintenanceList,
    create_order_list,
)

from domains.types.incremental.dependency_graph import (
    Dependency,
    DependencyGraph,
    DependencyGraphSnapshot,
    DependencyKind,
    NodeId,
    create_dependency_graph,
    restore_graph,
    snapshot_graph,
)

from domains.types.incremental.delta_typing import (
    CheckMode,
    DeltaTypingEngine,
    DeltaTypingResult,
    TypeCache,
    TypeCacheEntry,
    TypeCacheSnapshot,
    TypeChecker,
    create_delta_engine,
    snapshot_cache,
)

from domains.types.incremental.invalidation import (
    InvalidationEngine,
    InvalidationResult,
    InvalidationSnapshot,
    StabilityChecker,
    StabilityKind,
    create_invalidation_engine,
    is_type_concrete,
    restore_invalidation,
    snapshot_invalidation,
)


__all__ = [
    # Order Maintenance
    "MAX_LABEL",
    "OrderedElement",
    "OrderMaintenanceList",
    "create_order_list",
    # Dependency Graph
    "Dependency",
    "DependencyGraph",
    "DependencyGraphSnapshot",
    "DependencyKind",
    "NodeId",
    "create_dependency_graph",
    "restore_graph",
    "snapshot_graph",
    # Delta Typing
    "CheckMode",
    "DeltaTypingEngine",
    "DeltaTypingResult",
    "TypeCache",
    "TypeCacheEntry",
    "TypeCacheSnapshot",
    "TypeChecker",
    "create_delta_engine",
    "snapshot_cache",
    # Invalidation
    "InvalidationEngine",
    "InvalidationResult",
    "InvalidationSnapshot",
    "StabilityChecker",
    "StabilityKind",
    "create_invalidation_engine",
    "is_type_concrete",
    "restore_invalidation",
    "snapshot_invalidation",
]
