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
"""Speculative mask precomputation for latency hiding.

While the model computes logits (GPU-bound, ~10-50ms), we precompute masks
for likely next tokens in parallel. When the model finishes, the mask for
the selected token is often already cached.

Key insight: Model inference takes ~10-50ms; mask computation takes ~2-5ms.
By overlapping them, we achieve near-zero effective mask latency for cached paths.
"""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import threading
import time

import torch


@dataclass
class SpeculativeCacheEntry:
    """Entry in the speculative mask cache.

    Attributes:
        mask: The precomputed mask (if ready)
        future: The future for ongoing computation (if computing)
        context_hash: Hash of the context used for computation
        created_at_ns: When this entry was created
        hit_count: Number of cache hits
    """

    mask: Optional[torch.Tensor] = None
    future: Optional[Future] = None
    context_hash: int = 0
    created_at_ns: int = 0
    hit_count: int = 0

    @property
    def is_ready(self) -> bool:
        """Check if the mask is ready."""
        return self.mask is not None

    @property
    def is_computing(self) -> bool:
        """Check if computation is in progress."""
        return self.future is not None and not self.future.done()


@dataclass
class SpeculativeCacheStats:
    """Statistics for the speculative mask cache.

    Attributes:
        hits: Number of cache hits
        misses: Number of cache misses
        precompute_requests: Number of precomputation requests
        precompute_completions: Number of successful precomputations
        precompute_evictions: Number of evicted precomputations (computed but never used)
        avg_hit_latency_ns: Average latency for cache hits
        avg_miss_latency_ns: Average latency for cache misses
    """

    hits: int = 0
    misses: int = 0
    precompute_requests: int = 0
    precompute_completions: int = 0
    precompute_evictions: int = 0
    avg_hit_latency_ns: float = 0.0
    avg_miss_latency_ns: float = 0.0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return self.hits / total

    @property
    def precompute_efficiency(self) -> float:
        """Ratio of precomputed masks that were actually used."""
        if self.precompute_completions == 0:
            return 0.0
        return (self.precompute_completions - self.precompute_evictions) / self.precompute_completions


def _safe_hash(obj: Any) -> int:
    """Safely compute hash of an object, falling back to id() if unhashable."""
    try:
        return hash(obj)
    except TypeError:
        return id(obj)


class SpeculativeMaskCache:
    """Precomputes masks for likely next tokens while model is computing.

    This cache speculatively precomputes masks for likely next states,
    overlapping mask computation with model inference to hide latency.

    Usage:
        cache = SpeculativeMaskCache(
            compute_fn=compute_mask,
            lookahead_depth=3,
            max_workers=4,
        )

        # After computing current mask, precompute for likely next tokens
        current_mask = cache.get_or_compute(current_state, context)
        cache.precompute(likely_tokens, current_state, context)

        # When model finishes, next mask might already be cached
        next_mask = cache.get_or_compute(next_state, new_context)

    Thread Safety:
        The cache is thread-safe for concurrent get/precompute operations.
    """

    def __init__(
        self,
        compute_fn: Callable[[Any, Any], torch.Tensor],
        lookahead_depth: int = 3,
        max_workers: int = 4,
        max_cache_size: int = 64,
        state_transition_fn: Optional[Callable[[Any, int], Any]] = None,
    ) -> None:
        """Initialize the speculative mask cache.

        Args:
            compute_fn: Function to compute mask (state, context) -> mask
            lookahead_depth: Number of tokens to precompute ahead
            max_workers: Maximum concurrent precomputations
            max_cache_size: Maximum entries to cache (LRU eviction)
            state_transition_fn: Optional function to compute next state
                                 (current_state, token_id) -> next_state
                                 If None, cache uses (state, token_id) tuple as key
        """
        self._compute_fn = compute_fn
        self._lookahead_depth = lookahead_depth
        self._max_cache_size = max_cache_size
        self._state_transition_fn = state_transition_fn

        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._cache: OrderedDict[Any, SpeculativeCacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._stats = SpeculativeCacheStats()

        # Track pending precomputations
        self._pending: Set[Any] = set()

    def get_or_compute(
        self,
        state: Any,
        context: Any,
        wait_for_pending: bool = True,
    ) -> torch.Tensor:
        """Get a mask from cache or compute it.

        If the mask is being precomputed and wait_for_pending is True,
        waits for the precomputation to complete.

        Args:
            state: Current constraint state
            context: Generation context
            wait_for_pending: If True, wait for pending precomputations

        Returns:
            The computed or cached mask
        """
        start_ns = time.perf_counter_ns()
        cache_key = self._make_key(state, context)

        with self._lock:
            if cache_key in self._cache:
                entry = self._cache[cache_key]

                # If ready, return cached mask
                if entry.is_ready:
                    self._stats.hits += 1
                    entry.hit_count += 1
                    # Move to end (most recently used)
                    self._cache.move_to_end(cache_key)
                    elapsed_ns = time.perf_counter_ns() - start_ns
                    self._update_avg_latency(True, elapsed_ns)
                    return entry.mask

                # If computing and we should wait
                if entry.is_computing and wait_for_pending:
                    future = entry.future

        # Wait for pending computation outside the lock
        if 'future' in locals() and future is not None:
            try:
                mask = future.result(timeout=5.0)  # 5 second timeout
                with self._lock:
                    if cache_key in self._cache:
                        entry = self._cache[cache_key]
                        entry.mask = mask
                        entry.future = None
                        self._stats.hits += 1
                        entry.hit_count += 1
                        elapsed_ns = time.perf_counter_ns() - start_ns
                        self._update_avg_latency(True, elapsed_ns)
                return mask
            except Exception:
                pass  # Fall through to compute

        # Cache miss - compute now
        self._stats.misses += 1
        mask = self._compute_fn(state, context)

        # Cache the result
        with self._lock:
            self._cache[cache_key] = SpeculativeCacheEntry(
                mask=mask,
                context_hash=_safe_hash(context),
                created_at_ns=time.perf_counter_ns(),
            )
            self._evict_if_needed()

        elapsed_ns = time.perf_counter_ns() - start_ns
        self._update_avg_latency(False, elapsed_ns)
        return mask

    def precompute(
        self,
        likely_tokens: List[int],
        current_state: Any,
        context: Any,
    ) -> None:
        """Precompute masks for likely next tokens.

        This should be called immediately after returning the current mask,
        while the model is computing logits. The precomputed masks will be
        ready when the model selects a token.

        Args:
            likely_tokens: List of likely next token IDs (sorted by probability)
            current_state: Current constraint state
            context: Current generation context
        """
        # Only precompute for top-k likely tokens
        tokens_to_precompute = likely_tokens[: self._lookahead_depth]

        for token_id in tokens_to_precompute:
            # Compute next state
            if self._state_transition_fn is not None:
                next_state = self._state_transition_fn(current_state, token_id)
            else:
                next_state = (current_state, token_id)

            # Get next context (simplified - in practice would extend context)
            next_context = context

            cache_key = self._make_key(next_state, next_context)

            with self._lock:
                # Skip if already cached or computing
                if cache_key in self._cache:
                    continue
                if cache_key in self._pending:
                    continue

                self._pending.add(cache_key)
                self._stats.precompute_requests += 1

            # Submit precomputation
            future = self._executor.submit(
                self._precompute_entry,
                cache_key,
                next_state,
                next_context,
            )

            # Store future in cache
            with self._lock:
                if cache_key not in self._cache:
                    self._cache[cache_key] = SpeculativeCacheEntry(
                        future=future,
                        context_hash=_safe_hash(context),
                        created_at_ns=time.perf_counter_ns(),
                    )
                    self._evict_if_needed()

    def _precompute_entry(
        self,
        cache_key: Any,
        state: Any,
        context: Any,
    ) -> torch.Tensor:
        """Precompute a single cache entry."""
        try:
            mask = self._compute_fn(state, context)

            with self._lock:
                self._pending.discard(cache_key)
                if cache_key in self._cache:
                    entry = self._cache[cache_key]
                    entry.mask = mask
                    entry.future = None
                    self._stats.precompute_completions += 1

            return mask
        except Exception:
            with self._lock:
                self._pending.discard(cache_key)
                self._cache.pop(cache_key, None)
            raise

    def _make_key(self, state: Any, context: Any) -> Tuple:
        """Create a cache key from state and context."""
        # Use hash of context to avoid storing large objects
        # Note: hasattr(__hash__) is always True in Python, but some objects
        # raise TypeError when hash() is called (e.g., mutable objects)
        try:
            context_hash = hash(context)
        except TypeError:
            context_hash = id(context)

        try:
            state_key = hash(state) if not isinstance(state, int) else state
        except TypeError:
            state_key = id(state)

        return (state_key, context_hash)

    def _evict_if_needed(self) -> None:
        """Evict oldest entries if cache is full."""
        while len(self._cache) > self._max_cache_size:
            # Evict oldest (first) entry
            oldest_key, oldest_entry = next(iter(self._cache.items()))

            # Track eviction of unused precomputed masks
            if oldest_entry.is_ready and oldest_entry.hit_count == 0:
                self._stats.precompute_evictions += 1

            # Cancel pending computation if any
            if oldest_entry.future is not None and not oldest_entry.future.done():
                oldest_entry.future.cancel()

            del self._cache[oldest_key]
            self._pending.discard(oldest_key)

    def _update_avg_latency(self, is_hit: bool, latency_ns: int) -> None:
        """Update average latency statistics."""
        # Exponential moving average with alpha=0.1
        alpha = 0.1
        if is_hit:
            self._stats.avg_hit_latency_ns = (
                alpha * latency_ns + (1 - alpha) * self._stats.avg_hit_latency_ns
            )
        else:
            self._stats.avg_miss_latency_ns = (
                alpha * latency_ns + (1 - alpha) * self._stats.avg_miss_latency_ns
            )

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            # Cancel all pending computations
            for entry in self._cache.values():
                if entry.future is not None and not entry.future.done():
                    entry.future.cancel()

            self._cache.clear()
            self._pending.clear()

    def get_stats(self) -> SpeculativeCacheStats:
        """Get cache statistics."""
        with self._lock:
            return SpeculativeCacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                precompute_requests=self._stats.precompute_requests,
                precompute_completions=self._stats.precompute_completions,
                precompute_evictions=self._stats.precompute_evictions,
                avg_hit_latency_ns=self._stats.avg_hit_latency_ns,
                avg_miss_latency_ns=self._stats.avg_miss_latency_ns,
            )

    def shutdown(self) -> None:
        """Shutdown the thread pool."""
        self.clear()
        self._executor.shutdown(wait=False, cancel_futures=True)

    def __enter__(self) -> "SpeculativeMaskCache":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.shutdown()


def create_speculative_cache(
    compute_fn: Callable[[Any, Any], torch.Tensor],
    lookahead_depth: int = 3,
    max_workers: int = 4,
    max_cache_size: int = 64,
) -> SpeculativeMaskCache:
    """Factory function to create a SpeculativeMaskCache.

    Args:
        compute_fn: Function to compute mask (state, context) -> mask
        lookahead_depth: Number of tokens to precompute ahead
        max_workers: Maximum concurrent precomputations
        max_cache_size: Maximum entries to cache

    Returns:
        New SpeculativeMaskCache instance
    """
    return SpeculativeMaskCache(
        compute_fn=compute_fn,
        lookahead_depth=lookahead_depth,
        max_workers=max_workers,
        max_cache_size=max_cache_size,
    )
