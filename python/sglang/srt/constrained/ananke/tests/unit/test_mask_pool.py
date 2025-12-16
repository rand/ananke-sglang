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
"""Tests for mask pool module.

Tests the MaskPool implementation for Phase 3.3 GPU-side mask computation.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List

import pytest
import torch

try:
    from ...masks.pool import (
        MaskPool,
        MaskPoolStats,
        MultiDeviceMaskPool,
        create_mask_pool,
        create_multi_device_pool,
    )
except ImportError:
    from masks.pool import (
        MaskPool,
        MaskPoolStats,
        MultiDeviceMaskPool,
        create_mask_pool,
        create_multi_device_pool,
    )


# =============================================================================
# MaskPool Tests
# =============================================================================


class TestMaskPool:
    """Tests for MaskPool."""

    def test_create_default(self) -> None:
        """Test default creation."""
        pool = MaskPool()
        assert pool.vocab_size == 32000
        assert pool.pool_size == 16
        assert pool.available_count == 16
        assert pool.borrowed_count == 0

    def test_create_custom(self) -> None:
        """Test custom creation."""
        pool = MaskPool(vocab_size=50000, pool_size=8)
        assert pool.vocab_size == 50000
        assert pool.pool_size == 8

    def test_borrow_returns_mask(self) -> None:
        """Test that borrow returns a valid mask."""
        pool = MaskPool(vocab_size=1000, pool_size=4)

        with pool.borrow() as mask:
            assert isinstance(mask, torch.Tensor)
            assert mask.shape == (1000,)
            assert mask.dtype == torch.bool
            # Should be all True (reset)
            assert mask.all().item()

    def test_borrow_decrements_available(self) -> None:
        """Test that borrowing decrements available count."""
        pool = MaskPool(vocab_size=1000, pool_size=4)
        assert pool.available_count == 4

        with pool.borrow():
            assert pool.available_count == 3
            assert pool.borrowed_count == 1

        # After context exit, should be returned
        assert pool.available_count == 4
        assert pool.borrowed_count == 0

    def test_multiple_borrows(self) -> None:
        """Test multiple simultaneous borrows."""
        pool = MaskPool(vocab_size=1000, pool_size=4)

        masks: List[torch.Tensor] = []
        with pool.borrow() as m1:
            with pool.borrow() as m2:
                with pool.borrow() as m3:
                    assert pool.available_count == 1
                    assert pool.borrowed_count == 3
                    masks = [m1, m2, m3]

                    # Each mask should be independent
                    m1.fill_(False)
                    assert m2.all().item()  # m2 should still be True

    def test_pool_miss(self) -> None:
        """Test behavior when pool is exhausted."""
        pool = MaskPool(vocab_size=1000, pool_size=2)

        with pool.borrow() as m1:
            with pool.borrow() as m2:
                # Pool is now empty
                with pool.borrow() as m3:
                    # Should still work (new allocation)
                    assert isinstance(m3, torch.Tensor)
                    stats = pool.get_stats()
                    assert stats.pool_misses == 1

    def test_stats_tracking(self) -> None:
        """Test statistics are tracked correctly."""
        pool = MaskPool(vocab_size=1000, pool_size=4)

        with pool.borrow():
            pass
        with pool.borrow():
            with pool.borrow():
                pass

        stats = pool.get_stats()
        assert stats.total_borrows == 3
        assert stats.total_returns == 3
        assert stats.pool_misses == 0
        assert stats.peak_borrowed == 2

    def test_borrow_raw(self) -> None:
        """Test raw borrow without context manager."""
        pool = MaskPool(vocab_size=1000, pool_size=4)

        mask, idx = pool.borrow_raw()
        assert isinstance(mask, torch.Tensor)
        assert idx is not None
        assert pool.borrowed_count == 1

        pool.return_mask(idx)
        assert pool.borrowed_count == 0

    def test_resize(self) -> None:
        """Test pool resize."""
        pool = MaskPool(vocab_size=1000, pool_size=4)
        pool.resize(2000)
        assert pool.vocab_size == 2000

        with pool.borrow() as mask:
            assert mask.shape == (2000,)

    def test_resize_while_borrowed_fails(self) -> None:
        """Test that resize fails with borrowed masks."""
        pool = MaskPool(vocab_size=1000, pool_size=4)

        with pool.borrow():
            with pytest.raises(RuntimeError, match="Cannot resize"):
                pool.resize(2000)

    def test_reset_stats(self) -> None:
        """Test statistics reset."""
        pool = MaskPool(vocab_size=1000, pool_size=4)

        with pool.borrow():
            pass

        stats = pool.get_stats()
        assert stats.total_borrows == 1

        pool.reset_stats()
        stats = pool.get_stats()
        assert stats.total_borrows == 0

    def test_thread_safety(self) -> None:
        """Test thread safety of pool operations."""
        pool = MaskPool(vocab_size=1000, pool_size=4)
        results: List[bool] = []
        lock = threading.Lock()

        def borrow_and_use() -> None:
            try:
                with pool.borrow() as mask:
                    # Simulate some work
                    mask.fill_(False)
                    time.sleep(0.001)
                    assert not mask.any()
                with lock:
                    results.append(True)
            except Exception:
                with lock:
                    results.append(False)

        threads = [
            threading.Thread(target=borrow_and_use)
            for _ in range(10)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(results)
        assert len(results) == 10

    def test_concurrent_borrows(self) -> None:
        """Test concurrent borrow operations."""
        pool = MaskPool(vocab_size=1000, pool_size=8)

        def worker() -> int:
            count = 0
            for _ in range(5):
                with pool.borrow() as mask:
                    mask[0] = False
                    count += 1
            return count

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(worker) for _ in range(4)]
            results = [f.result() for f in futures]

        assert sum(results) == 20


# =============================================================================
# MultiDeviceMaskPool Tests
# =============================================================================


class TestMultiDeviceMaskPool:
    """Tests for MultiDeviceMaskPool."""

    def test_create_default(self) -> None:
        """Test default creation."""
        pool = MultiDeviceMaskPool()
        assert pool._vocab_size == 32000
        assert pool._pool_size == 8

    def test_borrow_cpu(self) -> None:
        """Test borrowing from CPU pool."""
        pool = MultiDeviceMaskPool(vocab_size=1000)

        with pool.borrow(device=torch.device("cpu")) as mask:
            assert mask.device == torch.device("cpu")
            assert mask.shape == (1000,)

    def test_creates_pool_on_demand(self) -> None:
        """Test that pools are created on demand."""
        pool = MultiDeviceMaskPool(vocab_size=1000)
        assert len(pool._pools) == 0

        with pool.borrow(device=torch.device("cpu")):
            pass

        assert len(pool._pools) == 1
        assert torch.device("cpu") in pool._pools

    def test_get_stats(self) -> None:
        """Test getting stats from all pools."""
        pool = MultiDeviceMaskPool(vocab_size=1000)

        with pool.borrow(device=torch.device("cpu")):
            pass

        stats = pool.get_stats()
        assert "cpu" in stats
        assert stats["cpu"].total_borrows == 1


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_mask_pool(self) -> None:
        """Test create_mask_pool factory."""
        pool = create_mask_pool(vocab_size=5000, pool_size=4)
        assert pool.vocab_size == 5000
        assert pool.pool_size == 4

    def test_create_multi_device_pool(self) -> None:
        """Test create_multi_device_pool factory."""
        pool = create_multi_device_pool(vocab_size=5000, pool_size_per_device=4)
        assert pool._vocab_size == 5000
        assert pool._pool_size == 4


# =============================================================================
# Integration Tests
# =============================================================================


class TestMaskPoolIntegration:
    """Integration tests for mask pool."""

    def test_mask_fusion_with_pool(self) -> None:
        """Test using pool for mask fusion operations."""
        pool = MaskPool(vocab_size=1000, pool_size=4)

        # Simulate mask fusion workflow
        syntax_mask = torch.ones(1000, dtype=torch.bool)
        syntax_mask[500:] = False

        type_mask = torch.ones(1000, dtype=torch.bool)
        type_mask[:200] = False

        with pool.borrow() as result:
            result.copy_(syntax_mask)
            result &= type_mask
            # Only tokens 200-499 should be valid
            assert result[199] == False
            assert result[200] == True
            assert result[499] == True
            assert result[500] == False

    def test_sequential_fusions(self) -> None:
        """Test sequential fusion operations reusing masks."""
        pool = MaskPool(vocab_size=1000, pool_size=2)

        for _ in range(10):
            with pool.borrow() as mask:
                mask[::2] = False  # Block even indices
                assert mask[1].item()  # Odd should be True
                assert not mask[0].item()  # Even should be False

        stats = pool.get_stats()
        assert stats.total_borrows == 10
        assert stats.pool_misses == 0  # Should reuse

    def test_zero_allocation_path(self) -> None:
        """Test that zero allocations occur in hot path."""
        pool = MaskPool(vocab_size=1000, pool_size=4)

        # Warm up
        for _ in range(4):
            with pool.borrow():
                pass

        # Now all pool slots have been used once
        # Subsequent borrows should not allocate
        for _ in range(100):
            with pool.borrow() as mask:
                mask[0] = False

        stats = pool.get_stats()
        assert stats.pool_misses == 0
