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
"""Unit tests for TypeCache LRU eviction.

Tests for the bounded TypeCache including:
- LRU eviction at capacity
- Statistics tracking (hits, misses, evictions)
- Access ordering behavior
- Invalidation with evicted entries
"""

import pytest

from domains.types.incremental.delta_typing import (
    TypeCache,
    TypeCacheStats,
    TypeCacheEntry,
    CheckMode,
    create_type_cache,
    DEFAULT_TYPE_CACHE_MAX_SIZE,
)
from domains.types.incremental.dependency_graph import NodeId
from domains.types.marking.provenance import SourceSpan
from domains.types.constraint import ANY


# ===========================================================================
# Helper functions
# ===========================================================================


def make_node_id(index: int) -> NodeId:
    """Create a test NodeId with the given index."""
    return NodeId(
        span=SourceSpan(start=index * 10, end=index * 10 + 5),
        kind="test",
        index=index,
    )


# ===========================================================================
# TypeCacheStats Tests
# ===========================================================================


class TestTypeCacheStats:
    """Tests for TypeCacheStats."""

    def test_initial_values(self):
        """Stats should start at zero."""
        stats = TypeCacheStats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.evictions == 0
        assert stats.re_checks == 0

    def test_hit_rate_empty(self):
        """Hit rate should be 0.0 with no accesses."""
        stats = TypeCacheStats()
        assert stats.hit_rate == 0.0

    def test_hit_rate_all_hits(self):
        """Hit rate should be 1.0 with all hits."""
        stats = TypeCacheStats(hits=10, misses=0)
        assert stats.hit_rate == 1.0

    def test_hit_rate_all_misses(self):
        """Hit rate should be 0.0 with all misses."""
        stats = TypeCacheStats(hits=0, misses=10)
        assert stats.hit_rate == 0.0

    def test_hit_rate_mixed(self):
        """Hit rate should be correct with mixed hits/misses."""
        stats = TypeCacheStats(hits=3, misses=7)
        assert stats.hit_rate == 0.3

    def test_reset(self):
        """Reset should clear all stats."""
        stats = TypeCacheStats(hits=10, misses=5, evictions=3, re_checks=2)
        stats.reset()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.evictions == 0
        assert stats.re_checks == 0


# ===========================================================================
# TypeCache Basic Tests
# ===========================================================================


class TestTypeCacheBasics:
    """Basic tests for TypeCache."""

    def test_default_max_size(self):
        """Default max size should be DEFAULT_TYPE_CACHE_MAX_SIZE."""
        cache = TypeCache()
        assert cache.max_size == DEFAULT_TYPE_CACHE_MAX_SIZE

    def test_custom_max_size(self):
        """Custom max size should be respected."""
        cache = TypeCache(max_size=100)
        assert cache.max_size == 100

    def test_empty_cache(self):
        """Empty cache should have length 0."""
        cache = TypeCache()
        assert len(cache) == 0

    def test_set_and_get(self):
        """Should be able to set and get entries."""
        cache = TypeCache(max_size=10)
        node_id = make_node_id(1)

        cache.set(node_id, CheckMode.SYNTHESIS, synthesized_type=ANY)
        entry = cache.get(node_id)

        assert entry is not None
        assert entry.node_id == node_id
        assert entry.synthesized_type == ANY

    def test_get_nonexistent(self):
        """Get on nonexistent key should return None and track miss."""
        cache = TypeCache(max_size=10)
        node_id = make_node_id(1)

        entry = cache.get(node_id)

        assert entry is None
        assert cache.stats.misses == 1

    def test_stats_property(self):
        """Stats property should return TypeCacheStats."""
        cache = TypeCache(max_size=10)
        assert isinstance(cache.stats, TypeCacheStats)


# ===========================================================================
# TypeCache LRU Eviction Tests
# ===========================================================================


class TestTypeCacheLRUEviction:
    """Tests for LRU eviction behavior."""

    def test_eviction_at_capacity(self):
        """Should evict oldest entry when at capacity."""
        cache = TypeCache(max_size=3)

        # Add 3 entries
        for i in range(3):
            cache.set(make_node_id(i), CheckMode.SYNTHESIS)

        assert len(cache) == 3

        # Add 4th entry - should evict node 0
        cache.set(make_node_id(3), CheckMode.SYNTHESIS)

        assert len(cache) == 3
        assert cache.get(make_node_id(0)) is None  # Evicted
        assert cache.get(make_node_id(1)) is not None
        assert cache.get(make_node_id(2)) is not None
        assert cache.get(make_node_id(3)) is not None

    def test_eviction_tracks_stats(self):
        """Eviction should be tracked in stats."""
        cache = TypeCache(max_size=2)

        cache.set(make_node_id(0), CheckMode.SYNTHESIS)
        cache.set(make_node_id(1), CheckMode.SYNTHESIS)
        assert cache.stats.evictions == 0

        cache.set(make_node_id(2), CheckMode.SYNTHESIS)
        assert cache.stats.evictions == 1

        cache.set(make_node_id(3), CheckMode.SYNTHESIS)
        assert cache.stats.evictions == 2

    def test_access_updates_lru_order(self):
        """Accessing an entry should move it to most recently used."""
        cache = TypeCache(max_size=3)

        # Add entries 0, 1, 2
        for i in range(3):
            cache.set(make_node_id(i), CheckMode.SYNTHESIS)

        # Access entry 0 - moves it to end (most recently used)
        cache.get(make_node_id(0))

        # Add entry 3 - should evict entry 1 (now oldest)
        cache.set(make_node_id(3), CheckMode.SYNTHESIS)

        assert cache.get(make_node_id(0)) is not None  # Still present (was accessed)
        assert cache.get(make_node_id(1)) is None  # Evicted (oldest after 0 was accessed)
        assert cache.get(make_node_id(2)) is not None
        assert cache.get(make_node_id(3)) is not None

    def test_set_existing_does_not_evict(self):
        """Updating existing entry should not trigger eviction."""
        cache = TypeCache(max_size=3)

        for i in range(3):
            cache.set(make_node_id(i), CheckMode.SYNTHESIS)

        # Update existing entry
        cache.set(make_node_id(0), CheckMode.ANALYSIS, synthesized_type=ANY)

        assert len(cache) == 3
        assert cache.stats.evictions == 0

    def test_get_type_updates_lru_order(self):
        """get_type should also update LRU order."""
        cache = TypeCache(max_size=3)

        for i in range(3):
            cache.set(make_node_id(i), CheckMode.SYNTHESIS, synthesized_type=ANY)

        # Access via get_type
        cache.get_type(make_node_id(0))

        # Add new entry - should evict entry 1
        cache.set(make_node_id(3), CheckMode.SYNTHESIS)

        assert cache.get(make_node_id(0)) is not None
        assert cache.get(make_node_id(1)) is None  # Evicted


# ===========================================================================
# TypeCache Statistics Tracking Tests
# ===========================================================================


class TestTypeCacheStatistics:
    """Tests for statistics tracking."""

    def test_hits_tracked(self):
        """Cache hits should be tracked."""
        cache = TypeCache(max_size=10)
        node_id = make_node_id(1)

        cache.set(node_id, CheckMode.SYNTHESIS)

        # First access - hit
        cache.get(node_id)
        assert cache.stats.hits == 1

        # Second access - another hit
        cache.get(node_id)
        assert cache.stats.hits == 2

    def test_misses_tracked(self):
        """Cache misses should be tracked."""
        cache = TypeCache(max_size=10)

        cache.get(make_node_id(1))
        assert cache.stats.misses == 1

        cache.get(make_node_id(2))
        assert cache.stats.misses == 2

    def test_miss_tracked_on_evicted_access(self):
        """Accessing evicted entry should count as miss."""
        cache = TypeCache(max_size=2)

        cache.set(make_node_id(0), CheckMode.SYNTHESIS)
        cache.set(make_node_id(1), CheckMode.SYNTHESIS)
        cache.set(make_node_id(2), CheckMode.SYNTHESIS)  # Evicts 0

        # Reset stats to isolate test
        cache.reset_stats()

        cache.get(make_node_id(0))  # Should be a miss
        assert cache.stats.misses == 1
        assert cache.stats.hits == 0

    def test_reset_stats(self):
        """reset_stats should clear statistics."""
        cache = TypeCache(max_size=10)
        node_id = make_node_id(1)

        cache.set(node_id, CheckMode.SYNTHESIS)
        cache.get(node_id)  # Hit
        cache.get(make_node_id(99))  # Miss

        cache.reset_stats()

        assert cache.stats.hits == 0
        assert cache.stats.misses == 0


# ===========================================================================
# TypeCache Integration with Order Maintenance
# ===========================================================================


class TestTypeCacheOrderMaintenance:
    """Tests for order maintenance integration with eviction."""

    def test_eviction_cleans_up_order_elements(self):
        """Evicted entries should be removed from order maintenance."""
        cache = TypeCache(max_size=2)

        node0 = make_node_id(0)
        node1 = make_node_id(1)
        node2 = make_node_id(2)

        cache.set(node0, CheckMode.SYNTHESIS)
        cache.set(node1, CheckMode.SYNTHESIS)

        assert node0 in cache._order_elements
        assert node1 in cache._order_elements

        # Evict node0
        cache.set(node2, CheckMode.SYNTHESIS)

        assert node0 not in cache._order_elements
        assert node1 in cache._order_elements
        assert node2 in cache._order_elements

    def test_dirty_nodes_excludes_evicted(self):
        """dirty_nodes should not include evicted entries."""
        cache = TypeCache(max_size=2)

        node0 = make_node_id(0)
        node1 = make_node_id(1)
        node2 = make_node_id(2)

        # Add and mark dirty
        cache.set(node0, CheckMode.SYNTHESIS)
        cache.invalidate(node0)
        cache.set(node1, CheckMode.SYNTHESIS)
        cache.invalidate(node1)

        dirty = cache.dirty_nodes()
        assert node0 in dirty
        assert node1 in dirty

        # Evict node0
        cache.set(node2, CheckMode.SYNTHESIS)
        cache.invalidate(node2)

        dirty = cache.dirty_nodes()
        assert node0 not in dirty  # Evicted
        assert node1 in dirty
        assert node2 in dirty


# ===========================================================================
# TypeCache Invalidation Tests
# ===========================================================================


class TestTypeCacheInvalidation:
    """Tests for cache invalidation."""

    def test_invalidate_marks_dirty(self):
        """Invalidate should mark entry as dirty."""
        cache = TypeCache(max_size=10)
        node_id = make_node_id(1)

        cache.set(node_id, CheckMode.SYNTHESIS)
        assert cache.is_valid(node_id)

        cache.invalidate(node_id)
        assert not cache.is_valid(node_id)

    def test_invalidate_evicted_is_noop(self):
        """Invalidating evicted entry should be no-op."""
        cache = TypeCache(max_size=2)

        node0 = make_node_id(0)
        cache.set(node0, CheckMode.SYNTHESIS)
        cache.set(make_node_id(1), CheckMode.SYNTHESIS)
        cache.set(make_node_id(2), CheckMode.SYNTHESIS)  # Evicts node0

        # Should not raise
        cache.invalidate(node0)


# ===========================================================================
# Factory Function Tests
# ===========================================================================


class TestCreateTypeCache:
    """Tests for create_type_cache factory."""

    def test_creates_cache(self):
        """Should create a TypeCache."""
        cache = create_type_cache()
        assert isinstance(cache, TypeCache)

    def test_default_size(self):
        """Should use default max size."""
        cache = create_type_cache()
        assert cache.max_size == DEFAULT_TYPE_CACHE_MAX_SIZE

    def test_custom_size(self):
        """Should accept custom max size."""
        cache = create_type_cache(max_size=500)
        assert cache.max_size == 500


# ===========================================================================
# TypeCache repr Tests
# ===========================================================================


class TestTypeCacheRepr:
    """Tests for TypeCache string representation."""

    def test_repr_empty(self):
        """Empty cache repr should show zeros."""
        cache = TypeCache(max_size=100)
        repr_str = repr(cache)

        assert "size=0/100" in repr_str
        assert "valid=0" in repr_str
        assert "dirty=0" in repr_str

    def test_repr_with_entries(self):
        """Cache with entries should show counts."""
        cache = TypeCache(max_size=100)

        cache.set(make_node_id(1), CheckMode.SYNTHESIS)
        cache.set(make_node_id(2), CheckMode.SYNTHESIS)
        cache.invalidate(make_node_id(2))

        repr_str = repr(cache)

        assert "size=2/100" in repr_str
        assert "valid=1" in repr_str
        assert "dirty=1" in repr_str
