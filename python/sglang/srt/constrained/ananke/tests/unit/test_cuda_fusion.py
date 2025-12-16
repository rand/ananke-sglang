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
"""Tests for CUDA mask fusion module.

Tests the CUDAMaskFuser implementation for Phase 3.3 GPU-side mask computation.
"""

from __future__ import annotations

from typing import List

import pytest
import torch

try:
    from ...masks.cuda_fusion import (
        CUDAMaskFuser,
        CUDAFusionConfig,
        CUDAFusionStats,
        FusionBenchmark,
        BatchedCUDAFuser,
        DeviceSelectionStrategy,
        create_cuda_fuser,
        create_batched_fuser,
        CUDA_AVAILABLE,
    )
except ImportError:
    from masks.cuda_fusion import (
        CUDAMaskFuser,
        CUDAFusionConfig,
        CUDAFusionStats,
        FusionBenchmark,
        BatchedCUDAFuser,
        DeviceSelectionStrategy,
        create_cuda_fuser,
        create_batched_fuser,
        CUDA_AVAILABLE,
    )


# =============================================================================
# CUDAFusionConfig Tests
# =============================================================================


class TestCUDAFusionConfig:
    """Tests for CUDAFusionConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = CUDAFusionConfig()
        assert config.device_strategy == DeviceSelectionStrategy.AUTO
        assert config.vocab_threshold == 50000
        assert config.min_masks_for_gpu == 2
        assert config.use_cuda_graphs is True

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = CUDAFusionConfig(
            device_strategy=DeviceSelectionStrategy.ALWAYS_CPU,
            vocab_threshold=100000,
            min_masks_for_gpu=5,
            use_cuda_graphs=False,
        )
        assert config.device_strategy == DeviceSelectionStrategy.ALWAYS_CPU
        assert config.vocab_threshold == 100000
        assert config.min_masks_for_gpu == 5
        assert config.use_cuda_graphs is False


# =============================================================================
# CUDAMaskFuser Tests (CPU)
# =============================================================================


class TestCUDAMaskFuserCPU:
    """Tests for CUDAMaskFuser on CPU."""

    def test_create_default(self) -> None:
        """Test default creation."""
        fuser = CUDAMaskFuser()
        assert fuser.vocab_size == 32000

    def test_create_custom(self) -> None:
        """Test custom creation."""
        config = CUDAFusionConfig(device_strategy=DeviceSelectionStrategy.ALWAYS_CPU)
        fuser = CUDAMaskFuser(vocab_size=50000, config=config)
        assert fuser.vocab_size == 50000

    def test_fuse_single_mask(self) -> None:
        """Test fusing a single mask."""
        fuser = CUDAMaskFuser(vocab_size=1000)
        mask = torch.ones(1000, dtype=torch.bool)
        mask[500:] = False

        result = fuser.fuse([mask], force_device="cpu")

        assert result.shape == (1000,)
        assert result[:500].all().item()
        assert not result[500:].any().item()

    def test_fuse_two_masks(self) -> None:
        """Test fusing two masks."""
        fuser = CUDAMaskFuser(vocab_size=1000)
        mask1 = torch.ones(1000, dtype=torch.bool)
        mask1[500:] = False  # Valid: 0-499

        mask2 = torch.ones(1000, dtype=torch.bool)
        mask2[:200] = False  # Valid: 200-999

        result = fuser.fuse([mask1, mask2], force_device="cpu")

        # Intersection: 200-499
        assert not result[:200].any().item()
        assert result[200:500].all().item()
        assert not result[500:].any().item()

    def test_fuse_three_masks(self) -> None:
        """Test fusing three masks."""
        fuser = CUDAMaskFuser(vocab_size=1000)
        mask1 = torch.ones(1000, dtype=torch.bool)
        mask1[600:] = False  # Valid: 0-599

        mask2 = torch.ones(1000, dtype=torch.bool)
        mask2[:200] = False  # Valid: 200-999

        mask3 = torch.ones(1000, dtype=torch.bool)
        mask3[400:500] = False  # Invalid: 400-499

        result = fuser.fuse([mask1, mask2, mask3], force_device="cpu")

        # Intersection: 200-399 and 500-599
        assert not result[:200].any().item()
        assert result[200:400].all().item()
        assert not result[400:500].any().item()
        assert result[500:600].all().item()
        assert not result[600:].any().item()

    def test_fuse_empty_list_raises(self) -> None:
        """Test that fusing empty list raises error."""
        fuser = CUDAMaskFuser(vocab_size=1000)
        with pytest.raises(ValueError, match="No masks"):
            fuser.fuse([])

    def test_fuse_all_zeros_short_circuit(self) -> None:
        """Test that fusion short-circuits on all zeros."""
        fuser = CUDAMaskFuser(vocab_size=1000)
        mask1 = torch.ones(1000, dtype=torch.bool)
        mask1[500:] = False  # Valid: 0-499

        mask2 = torch.zeros(1000, dtype=torch.bool)  # All invalid

        mask3 = torch.ones(1000, dtype=torch.bool)  # Won't matter

        result = fuser.fuse([mask1, mask2, mask3], force_device="cpu")

        assert not result.any().item()

    def test_stats_tracking(self) -> None:
        """Test statistics are tracked correctly."""
        fuser = CUDAMaskFuser(vocab_size=1000)
        mask = torch.ones(1000, dtype=torch.bool)

        fuser.fuse([mask], force_device="cpu")
        fuser.fuse([mask, mask], force_device="cpu")

        stats = fuser.get_stats()
        assert stats.total_fusions == 2
        assert stats.cpu_fusions == 2
        assert stats.gpu_fusions == 0

    def test_reset_stats(self) -> None:
        """Test statistics reset."""
        fuser = CUDAMaskFuser(vocab_size=1000)
        mask = torch.ones(1000, dtype=torch.bool)
        fuser.fuse([mask], force_device="cpu")

        fuser.reset_stats()
        stats = fuser.get_stats()
        assert stats.total_fusions == 0


# =============================================================================
# CUDAMaskFuser Device Selection Tests
# =============================================================================


class TestDeviceSelection:
    """Tests for device selection logic."""

    def test_always_cpu_strategy(self) -> None:
        """Test ALWAYS_CPU strategy."""
        config = CUDAFusionConfig(device_strategy=DeviceSelectionStrategy.ALWAYS_CPU)
        fuser = CUDAMaskFuser(vocab_size=100000, config=config)
        mask = torch.ones(100000, dtype=torch.bool)

        fuser.fuse([mask, mask, mask])
        stats = fuser.get_stats()
        assert stats.cpu_fusions == 1
        assert stats.gpu_fusions == 0

    def test_force_device_cpu(self) -> None:
        """Test forcing CPU device."""
        fuser = CUDAMaskFuser(vocab_size=1000)
        mask = torch.ones(1000, dtype=torch.bool)

        fuser.fuse([mask], force_device="cpu")
        stats = fuser.get_stats()
        assert stats.cpu_fusions == 1

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_force_device_gpu(self) -> None:
        """Test forcing GPU device."""
        fuser = CUDAMaskFuser(vocab_size=1000)
        mask = torch.ones(1000, dtype=torch.bool)

        fuser.fuse([mask, mask], force_device="gpu")
        stats = fuser.get_stats()
        assert stats.gpu_fusions == 1

    def test_threshold_strategy(self) -> None:
        """Test THRESHOLD strategy."""
        config = CUDAFusionConfig(
            device_strategy=DeviceSelectionStrategy.THRESHOLD,
            vocab_threshold=5000,
        )

        # Small vocab - should use CPU
        fuser_small = CUDAMaskFuser(vocab_size=1000, config=config)
        mask_small = torch.ones(1000, dtype=torch.bool)
        fuser_small.fuse([mask_small, mask_small])
        assert fuser_small.get_stats().cpu_fusions == 1

        # Large vocab - would use GPU if available
        fuser_large = CUDAMaskFuser(vocab_size=10000, config=config)
        mask_large = torch.ones(10000, dtype=torch.bool)
        fuser_large.fuse([mask_large, mask_large], force_device="cpu")
        # Force CPU to avoid GPU dependency in test

    def test_benchmark_device_selection(self) -> None:
        """Test device selection benchmark."""
        fuser = CUDAMaskFuser(vocab_size=1000)
        benchmark = fuser.benchmark_device_selection(num_masks=3)

        assert isinstance(benchmark, FusionBenchmark)
        assert benchmark.vocab_size == 1000
        assert benchmark.num_masks == 3
        assert benchmark.cpu_time_ns > 0
        assert benchmark.recommended_device in ("cpu", "gpu")


# =============================================================================
# BatchedCUDAFuser Tests
# =============================================================================


class TestBatchedCUDAFuser:
    """Tests for BatchedCUDAFuser."""

    def test_create_default(self) -> None:
        """Test default creation."""
        fuser = BatchedCUDAFuser()
        assert fuser._vocab_size == 32000

    def test_fuse_batch_single_request(self) -> None:
        """Test batch fusion with single request."""
        fuser = BatchedCUDAFuser(vocab_size=1000, device=torch.device("cpu"))
        mask1 = torch.ones(1000, dtype=torch.bool)
        mask1[500:] = False
        mask2 = torch.ones(1000, dtype=torch.bool)
        mask2[:200] = False

        results = fuser.fuse_batch([[mask1, mask2]])

        assert len(results) == 1
        assert results[0][200:500].all().item()
        assert not results[0][:200].any().item()

    def test_fuse_batch_multiple_requests(self) -> None:
        """Test batch fusion with multiple requests."""
        fuser = BatchedCUDAFuser(vocab_size=1000, device=torch.device("cpu"))

        # Request 1: Valid 200-499
        req1_m1 = torch.ones(1000, dtype=torch.bool)
        req1_m1[500:] = False
        req1_m2 = torch.ones(1000, dtype=torch.bool)
        req1_m2[:200] = False

        # Request 2: Valid 0-299
        req2_m1 = torch.ones(1000, dtype=torch.bool)
        req2_m1[300:] = False

        results = fuser.fuse_batch([
            [req1_m1, req1_m2],
            [req2_m1],
        ])

        assert len(results) == 2
        # Request 1: 200-499
        assert results[0][200:500].all().item()
        assert not results[0][500:].any().item()
        # Request 2: 0-299
        assert results[1][:300].all().item()
        assert not results[1][300:].any().item()

    def test_fuse_batch_empty(self) -> None:
        """Test batch fusion with empty batch."""
        fuser = BatchedCUDAFuser(vocab_size=1000, device=torch.device("cpu"))
        results = fuser.fuse_batch([])
        assert results == []

    def test_fuse_batch_empty_masks(self) -> None:
        """Test batch fusion with empty mask lists."""
        fuser = BatchedCUDAFuser(vocab_size=1000, device=torch.device("cpu"))
        results = fuser.fuse_batch([[], []])
        assert len(results) == 2
        # Empty mask list = all valid
        assert results[0].all().item()
        assert results[1].all().item()


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_cuda_fuser(self) -> None:
        """Test create_cuda_fuser factory."""
        fuser = create_cuda_fuser(vocab_size=5000)
        assert fuser.vocab_size == 5000

    def test_create_cuda_fuser_with_config(self) -> None:
        """Test create_cuda_fuser with config kwargs."""
        fuser = create_cuda_fuser(
            vocab_size=5000,
            device_strategy=DeviceSelectionStrategy.ALWAYS_CPU,
            vocab_threshold=10000,
        )
        assert fuser.vocab_size == 5000
        assert fuser._config.device_strategy == DeviceSelectionStrategy.ALWAYS_CPU

    def test_create_batched_fuser(self) -> None:
        """Test create_batched_fuser factory."""
        fuser = create_batched_fuser(vocab_size=5000)
        assert fuser._vocab_size == 5000


# =============================================================================
# Integration Tests
# =============================================================================


class TestCUDAFusionIntegration:
    """Integration tests for CUDA fusion."""

    def test_end_to_end_fusion_workflow(self) -> None:
        """Test complete fusion workflow."""
        fuser = CUDAMaskFuser(vocab_size=1000)

        # Simulate constraint domain masks
        syntax_mask = torch.ones(1000, dtype=torch.bool)
        syntax_mask[800:] = False  # Syntax: 0-799

        type_mask = torch.ones(1000, dtype=torch.bool)
        type_mask[:100] = False  # Types: 100-999
        type_mask[700:800] = False  # Also exclude 700-799

        import_mask = torch.ones(1000, dtype=torch.bool)
        import_mask[600:650] = False  # Imports: exclude 600-649

        result = fuser.fuse([syntax_mask, type_mask, import_mask], force_device="cpu")

        # Valid: 100-599, 650-699
        assert not result[:100].any().item()  # Blocked by types
        assert result[100:600].all().item()  # Valid
        assert not result[600:650].any().item()  # Blocked by imports
        assert result[650:700].all().item()  # Valid
        assert not result[700:].any().item()  # Blocked by types/syntax

    def test_repeated_fusion_patterns(self) -> None:
        """Test that repeated patterns work correctly."""
        fuser = CUDAMaskFuser(vocab_size=1000)

        for i in range(10):
            mask1 = torch.ones(1000, dtype=torch.bool)
            mask1[i * 100:(i + 1) * 100] = False
            mask2 = torch.ones(1000, dtype=torch.bool)

            result = fuser.fuse([mask1, mask2], force_device="cpu")

            # Check specific pattern for each iteration
            assert not result[i * 100:(i + 1) * 100].any().item()

        stats = fuser.get_stats()
        assert stats.total_fusions == 10

    def test_varying_selectivity(self) -> None:
        """Test fusion with varying mask selectivity."""
        fuser = CUDAMaskFuser(vocab_size=1000)

        # Very selective mask (few valid)
        selective_mask = torch.zeros(1000, dtype=torch.bool)
        selective_mask[400:410] = True  # Only 10 valid

        # Permissive mask (many valid)
        permissive_mask = torch.ones(1000, dtype=torch.bool)
        permissive_mask[900:] = False  # 900 valid

        result = fuser.fuse([selective_mask, permissive_mask], force_device="cpu")

        # Result should be intersection: 400-409
        assert result.sum().item() == 10
        assert result[400:410].all().item()


# =============================================================================
# CUDA-Specific Tests (only run if CUDA available)
# =============================================================================


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestCUDASpecific:
    """Tests that require CUDA."""

    def test_gpu_fusion(self) -> None:
        """Test GPU fusion."""
        config = CUDAFusionConfig(device_strategy=DeviceSelectionStrategy.ALWAYS_GPU)
        fuser = CUDAMaskFuser(vocab_size=10000, config=config)

        mask1 = torch.ones(10000, dtype=torch.bool)
        mask1[5000:] = False
        mask2 = torch.ones(10000, dtype=torch.bool)
        mask2[:2000] = False

        result = fuser.fuse([mask1, mask2])

        # Result should be on GPU
        assert result.is_cuda
        # Intersection: 2000-4999
        result_cpu = result.cpu()
        assert not result_cpu[:2000].any().item()
        assert result_cpu[2000:5000].all().item()
        assert not result_cpu[5000:].any().item()

    def test_cuda_graph_caching(self) -> None:
        """Test CUDA graph caching."""
        config = CUDAFusionConfig(
            device_strategy=DeviceSelectionStrategy.ALWAYS_GPU,
            use_cuda_graphs=True,
        )
        fuser = CUDAMaskFuser(vocab_size=10000, config=config)

        mask1 = torch.ones(10000, dtype=torch.bool)
        mask2 = torch.ones(10000, dtype=torch.bool)

        # First call captures graph
        fuser.fuse([mask1, mask2])

        # Subsequent calls should use cached graph
        for _ in range(5):
            fuser.fuse([mask1, mask2])

        stats = fuser.get_stats()
        # First was a miss, rest were hits
        assert stats.graph_cache_misses >= 1
        assert stats.graph_cache_hits >= 4

    def test_batched_gpu_fusion(self) -> None:
        """Test batched GPU fusion."""
        fuser = BatchedCUDAFuser(vocab_size=10000)

        batch = [
            [torch.ones(10000, dtype=torch.bool), torch.ones(10000, dtype=torch.bool)],
            [torch.zeros(10000, dtype=torch.bool)],
        ]

        results = fuser.fuse_batch(batch)

        assert len(results) == 2
        assert results[0].all().item()  # All ones AND all ones = all ones
        assert not results[1].any().item()  # All zeros
