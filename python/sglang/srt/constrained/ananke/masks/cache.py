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
"""LRU cache for token masks.

Caches computed token masks to avoid recomputation.
Cache keys include:
- Domain name
- Constraint hash
- Generation position
- Context hash
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, Generic, Optional, Tuple, TypeVar, Hashable
import time

import torch

# Type variable for constraint types
C = TypeVar("C")


@dataclass
class CacheKey:
    """Key for mask cache lookups.

    Attributes:
        domain: Domain name
        constraint_hash: Hash of the constraint
        position: Generation position
        context_hash: Hash of relevant context
    """

    domain: str
    constraint_hash: int
    position: int = 0
    context_hash: int = 0

    def __hash__(self) -> int:
        return hash((self.domain, self.constraint_hash, self.position, self.context_hash))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CacheKey):
            return False
        return (
            self.domain == other.domain and
            self.constraint_hash == other.constraint_hash and
            self.position == other.position and
            self.context_hash == other.context_hash
        )


@dataclass
class CacheEntry:
    """Entry in the mask cache.

    Attributes:
        mask: The cached mask tensor
        created_at: Timestamp when created
        hits: Number of cache hits
        compute_time_ns: Time to compute originally
    """

    mask: torch.Tensor
    created_at: float = field(default_factory=time.time)
    hits: int = 0
    compute_time_ns: int = 0

    @property
    def age_seconds(self) -> float:
        """Get age of entry in seconds."""
        return time.time() - self.created_at


@dataclass
class CacheStats:
    """Statistics for cache performance.

    Attributes:
        hits: Number of cache hits
        misses: Number of cache misses
        evictions: Number of entries evicted
        total_compute_time_saved_ns: Time saved by cache hits
    """

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_compute_time_saved_ns: int = 0

    @property
    def hit_rate(self) -> float:
        """Get cache hit rate as a fraction (0.0 to 1.0)."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def hit_rate_percent(self) -> float:
        """Get cache hit rate as a percentage."""
        return self.hit_rate * 100.0

    @property
    def total_requests(self) -> int:
        """Get total number of cache requests."""
        return self.hits + self.misses

    @property
    def time_saved_ms(self) -> float:
        """Get total compute time saved in milliseconds."""
        return self.total_compute_time_saved_ns / 1_000_000.0

    def reset(self) -> None:
        """Reset statistics."""
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.total_compute_time_saved_ns = 0

    def summary(self) -> str:
        """Get a human-readable summary of cache performance.

        Returns:
            Multi-line string with cache statistics
        """
        lines = [
            "Cache Statistics:",
            f"  Requests: {self.total_requests} (hits: {self.hits}, misses: {self.misses})",
            f"  Hit rate: {self.hit_rate_percent:.1f}%",
            f"  Evictions: {self.evictions}",
            f"  Time saved: {self.time_saved_ms:.2f}ms",
        ]
        return "\n".join(lines)


class MaskCache:
    """LRU cache for token masks.

    Caches computed masks to avoid recomputation when:
    - Same constraint is used multiple times
    - Position hasn't changed
    - Context is equivalent

    Example:
        >>> cache = MaskCache(max_size=1000)
        >>> key = cache.make_key("types", constraint, position)
        >>> cached = cache.get(key)
        >>> if cached is None:
        ...     mask = compute_mask(...)
        ...     cache.put(key, mask)
    """

    def __init__(
        self,
        max_size: int = 1000,
        max_age_seconds: float = 60.0,
    ) -> None:
        """Initialize the cache.

        Args:
            max_size: Maximum number of entries
            max_age_seconds: Maximum age for entries
        """
        self._max_size = max_size
        self._max_age_seconds = max_age_seconds
        self._cache: OrderedDict[CacheKey, CacheEntry] = OrderedDict()
        self._stats = CacheStats()

    @property
    def size(self) -> int:
        """Get current cache size."""
        return len(self._cache)

    @property
    def stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._stats

    def make_key(
        self,
        domain: str,
        constraint: Any,
        position: int = 0,
        context: Optional[Any] = None,
    ) -> CacheKey:
        """Create a cache key.

        Args:
            domain: Domain name
            constraint: The constraint
            position: Generation position
            context: Optional context

        Returns:
            CacheKey for lookups
        """
        # Try to get constraint hash
        try:
            constraint_hash = hash(constraint)
        except TypeError:
            # Unhashable constraint - use id
            constraint_hash = id(constraint)

        # Try to get context hash
        context_hash = 0
        if context is not None:
            try:
                context_hash = hash(context)
            except TypeError:
                context_hash = id(context)

        return CacheKey(
            domain=domain,
            constraint_hash=constraint_hash,
            position=position,
            context_hash=context_hash,
        )

    def get(self, key: CacheKey) -> Optional[torch.Tensor]:
        """Get a mask from cache.

        Args:
            key: Cache key

        Returns:
            Cached mask if found and valid, None otherwise
        """
        entry = self._cache.get(key)

        if entry is None:
            self._stats.misses += 1
            return None

        # Check age
        if entry.age_seconds > self._max_age_seconds:
            self._cache.pop(key)
            self._stats.misses += 1
            self._stats.evictions += 1
            return None

        # Move to end (most recently used)
        self._cache.move_to_end(key)

        # Update stats
        entry.hits += 1
        self._stats.hits += 1
        self._stats.total_compute_time_saved_ns += entry.compute_time_ns

        return entry.mask

    def put(
        self,
        key: CacheKey,
        mask: torch.Tensor,
        compute_time_ns: int = 0,
    ) -> None:
        """Put a mask in the cache.

        Args:
            key: Cache key
            mask: The mask tensor
            compute_time_ns: Time taken to compute
        """
        # Evict if at capacity
        while len(self._cache) >= self._max_size:
            # Remove least recently used
            self._cache.popitem(last=False)
            self._stats.evictions += 1

        self._cache[key] = CacheEntry(
            mask=mask,
            compute_time_ns=compute_time_ns,
        )

    def invalidate(self, key: CacheKey) -> bool:
        """Invalidate a specific cache entry.

        Args:
            key: Cache key to invalidate

        Returns:
            True if entry was found and removed
        """
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def invalidate_domain(self, domain: str) -> int:
        """Invalidate all entries for a domain.

        Args:
            domain: Domain name

        Returns:
            Number of entries invalidated
        """
        keys_to_remove = [k for k in self._cache.keys() if k.domain == domain]
        for key in keys_to_remove:
            del self._cache[key]
        return len(keys_to_remove)

    def invalidate_position(self, position: int) -> int:
        """Invalidate all entries at a position.

        Args:
            position: Generation position

        Returns:
            Number of entries invalidated
        """
        keys_to_remove = [k for k in self._cache.keys() if k.position == position]
        for key in keys_to_remove:
            del self._cache[key]
        return len(keys_to_remove)

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()

    def reset_stats(self) -> None:
        """Reset cache statistics."""
        self._stats.reset()

    def evict_stale(self) -> int:
        """Evict stale entries.

        Returns:
            Number of entries evicted
        """
        current_time = time.time()
        keys_to_remove = [
            key for key, entry in self._cache.items()
            if current_time - entry.created_at > self._max_age_seconds
        ]
        for key in keys_to_remove:
            del self._cache[key]
        self._stats.evictions += len(keys_to_remove)
        return len(keys_to_remove)


class DomainCache:
    """Cache for a single domain's masks.

    Specialized cache that understands domain-specific caching.
    """

    def __init__(
        self,
        domain_name: str,
        max_size: int = 100,
    ) -> None:
        """Initialize domain cache.

        Args:
            domain_name: Name of the domain
            max_size: Maximum entries
        """
        self._domain_name = domain_name
        self._cache = MaskCache(max_size=max_size)

    @property
    def domain_name(self) -> str:
        """Get the domain name."""
        return self._domain_name

    def get(
        self,
        constraint: Any,
        position: int = 0,
    ) -> Optional[torch.Tensor]:
        """Get cached mask for constraint.

        Args:
            constraint: The constraint
            position: Generation position

        Returns:
            Cached mask if found
        """
        key = self._cache.make_key(self._domain_name, constraint, position)
        return self._cache.get(key)

    def put(
        self,
        constraint: Any,
        mask: torch.Tensor,
        position: int = 0,
        compute_time_ns: int = 0,
    ) -> None:
        """Cache a mask for constraint.

        Args:
            constraint: The constraint
            mask: The computed mask
            position: Generation position
            compute_time_ns: Computation time
        """
        key = self._cache.make_key(self._domain_name, constraint, position)
        self._cache.put(key, mask, compute_time_ns)

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()

    @property
    def stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._cache.stats


class MultiDomainCache:
    """Cache manager for multiple domains.

    Manages separate caches per domain with global limits.
    """

    def __init__(
        self,
        max_total_size: int = 10000,
        max_per_domain: int = 1000,
    ) -> None:
        """Initialize multi-domain cache.

        Args:
            max_total_size: Maximum total entries
            max_per_domain: Maximum per domain
        """
        self._max_total_size = max_total_size
        self._max_per_domain = max_per_domain
        self._domain_caches: Dict[str, DomainCache] = {}

    def get_domain_cache(self, domain: str) -> DomainCache:
        """Get or create cache for a domain.

        Args:
            domain: Domain name

        Returns:
            DomainCache for the domain
        """
        if domain not in self._domain_caches:
            self._domain_caches[domain] = DomainCache(
                domain_name=domain,
                max_size=self._max_per_domain,
            )
        return self._domain_caches[domain]

    def get(
        self,
        domain: str,
        constraint: Any,
        position: int = 0,
    ) -> Optional[torch.Tensor]:
        """Get cached mask.

        Args:
            domain: Domain name
            constraint: The constraint
            position: Generation position

        Returns:
            Cached mask if found
        """
        cache = self.get_domain_cache(domain)
        return cache.get(constraint, position)

    def put(
        self,
        domain: str,
        constraint: Any,
        mask: torch.Tensor,
        position: int = 0,
        compute_time_ns: int = 0,
    ) -> None:
        """Cache a mask.

        Args:
            domain: Domain name
            constraint: The constraint
            mask: The computed mask
            position: Generation position
            compute_time_ns: Computation time
        """
        cache = self.get_domain_cache(domain)
        cache.put(constraint, mask, position, compute_time_ns)

    def clear(self, domain: Optional[str] = None) -> None:
        """Clear caches.

        Args:
            domain: Optional specific domain to clear
        """
        if domain:
            if domain in self._domain_caches:
                self._domain_caches[domain].clear()
        else:
            for cache in self._domain_caches.values():
                cache.clear()

    def total_stats(self) -> CacheStats:
        """Get combined statistics.

        Returns:
            Combined CacheStats from all domains
        """
        combined = CacheStats()
        for cache in self._domain_caches.values():
            stats = cache.stats
            combined.hits += stats.hits
            combined.misses += stats.misses
            combined.evictions += stats.evictions
            combined.total_compute_time_saved_ns += stats.total_compute_time_saved_ns
        return combined


def create_cache(
    max_size: int = 1000,
    max_age_seconds: float = 60.0,
) -> MaskCache:
    """Factory function to create a MaskCache.

    Args:
        max_size: Maximum cache size
        max_age_seconds: Maximum entry age

    Returns:
        New MaskCache instance
    """
    return MaskCache(max_size=max_size, max_age_seconds=max_age_seconds)
