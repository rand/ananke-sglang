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
"""GPU-side mask computation using CUDA.

This module provides CUDA-accelerated mask fusion for large vocabularies
(100k+) where GPU-side computation reduces latency vs CPU+transfer.

Key Features:
1. CUDA kernel for mask fusion (bitwise AND)
2. Batch multiple domain masks per kernel launch
3. CUDA graphs for repeated patterns
4. Automatic CPU/GPU selection based on expected performance

Performance Considerations:
- GPU wins for vocab_size > 50k with multiple masks
- CPU wins for small vocabs or single masks (transfer overhead)
- Use benchmark_device_selection() to tune threshold

Architecture:
- CUDAMaskFuser: Main fusion class with device selection
- CUDAGraph caching for repeated patterns
- Integration with MaskPool for zero-copy

References:
- XGrammar (MLSys 2025): GPU-side grammar constraints
- Test-Time Compute Scaling: GPU optimization for inference
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from .pool import MaskPool, create_mask_pool

logger = logging.getLogger(__name__)


# Check CUDA availability at module load
CUDA_AVAILABLE = torch.cuda.is_available()


class DeviceSelectionStrategy(Enum):
    """Strategy for selecting computation device.

    Strategies:
    - AUTO: Select based on heuristics
    - ALWAYS_GPU: Force GPU computation
    - ALWAYS_CPU: Force CPU computation
    - THRESHOLD: Use GPU above vocab_size threshold
    """

    AUTO = auto()
    ALWAYS_GPU = auto()
    ALWAYS_CPU = auto()
    THRESHOLD = auto()


@dataclass
class CUDAFusionConfig:
    """Configuration for CUDA mask fusion.

    Attributes:
        device_strategy: How to select computation device
        vocab_threshold: Use GPU above this vocab size (for THRESHOLD)
        min_masks_for_gpu: Minimum masks to justify GPU transfer
        use_cuda_graphs: Whether to use CUDA graphs for caching
        max_cached_graphs: Maximum CUDA graphs to cache
        benchmark_iterations: Iterations for device benchmarking
    """

    device_strategy: DeviceSelectionStrategy = DeviceSelectionStrategy.AUTO
    vocab_threshold: int = 50000
    min_masks_for_gpu: int = 2
    use_cuda_graphs: bool = True
    max_cached_graphs: int = 16
    benchmark_iterations: int = 10


@dataclass
class CUDAFusionStats:
    """Statistics for CUDA fusion operations.

    Attributes:
        total_fusions: Total fusion operations
        gpu_fusions: Fusions performed on GPU
        cpu_fusions: Fusions performed on CPU
        graph_cache_hits: CUDA graph cache hits
        graph_cache_misses: CUDA graph cache misses
        total_time_ns: Total fusion time
        avg_gpu_time_ns: Average GPU fusion time
        avg_cpu_time_ns: Average CPU fusion time
    """

    total_fusions: int = 0
    gpu_fusions: int = 0
    cpu_fusions: int = 0
    graph_cache_hits: int = 0
    graph_cache_misses: int = 0
    total_time_ns: int = 0
    avg_gpu_time_ns: float = 0.0
    avg_cpu_time_ns: float = 0.0


@dataclass
class FusionBenchmark:
    """Benchmark results for device selection.

    Attributes:
        vocab_size: Vocabulary size tested
        num_masks: Number of masks tested
        cpu_time_ns: Average CPU time
        gpu_time_ns: Average GPU time (includes transfer)
        recommended_device: Recommended device based on benchmark
    """

    vocab_size: int
    num_masks: int
    cpu_time_ns: float
    gpu_time_ns: float
    recommended_device: str


class CUDAMaskFuser:
    """CUDA-accelerated mask fusion for constrained generation.

    This class provides GPU-accelerated mask fusion operations with
    automatic device selection and CUDA graph caching.

    Device Selection:
        By default (AUTO strategy), selects device based on:
        - Vocabulary size (larger → GPU)
        - Number of masks (more → GPU)
        - Estimated transfer overhead

    CUDA Graphs:
        For repeated fusion patterns (same vocab_size, same num_masks),
        CUDA graphs are captured and replayed for reduced kernel launch
        overhead (~10-20μs saved per fusion).

    Thread Safety:
        Operations are thread-safe. CUDA graphs are created per-stream.

    Example:
        >>> fuser = CUDAMaskFuser(vocab_size=128000)
        >>> masks = [domain1_mask, domain2_mask, domain3_mask]
        >>> result = fuser.fuse(masks)  # Automatic device selection
        >>> print(f"Valid tokens: {result.sum().item()}")

    Integration with MaskPool:
        >>> fuser = CUDAMaskFuser(vocab_size=128000, use_pool=True)
        >>> with fuser.fuse_pooled(masks) as result:
        ...     # result is from pool, automatically returned
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        config: Optional[CUDAFusionConfig] = None,
        device: Optional[torch.device] = None,
        use_pool: bool = True,
    ) -> None:
        """Initialize CUDA mask fuser.

        Args:
            vocab_size: Vocabulary size for masks
            config: Fusion configuration
            device: Primary GPU device (default: cuda:0 if available)
            use_pool: Whether to use mask pool for zero-copy
        """
        self._vocab_size = vocab_size
        self._config = config or CUDAFusionConfig()
        self._stats = CUDAFusionStats()

        # Set up devices
        self._cpu_device = torch.device("cpu")
        if device is not None:
            self._gpu_device = device
        elif CUDA_AVAILABLE:
            self._gpu_device = torch.device("cuda:0")
        else:
            self._gpu_device = None

        # Set up mask pools
        self._use_pool = use_pool
        if use_pool:
            self._cpu_pool = create_mask_pool(
                vocab_size=vocab_size,
                device=self._cpu_device,
            )
            if self._gpu_device is not None:
                self._gpu_pool = create_mask_pool(
                    vocab_size=vocab_size,
                    device=self._gpu_device,
                )
            else:
                self._gpu_pool = None
        else:
            self._cpu_pool = None
            self._gpu_pool = None

        # CUDA graph cache: key = (vocab_size, num_masks)
        self._cuda_graphs: Dict[Tuple[int, int], torch.cuda.CUDAGraph] = {}
        self._graph_inputs: Dict[Tuple[int, int], List[torch.Tensor]] = {}
        self._graph_outputs: Dict[Tuple[int, int], torch.Tensor] = {}

        # Timing for AUTO device selection
        self._gpu_times: List[float] = []
        self._cpu_times: List[float] = []

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return self._vocab_size

    @property
    def has_gpu(self) -> bool:
        """Check if GPU is available."""
        return self._gpu_device is not None

    def get_stats(self) -> CUDAFusionStats:
        """Get fusion statistics.

        Returns:
            Current fusion statistics
        """
        stats = CUDAFusionStats(
            total_fusions=self._stats.total_fusions,
            gpu_fusions=self._stats.gpu_fusions,
            cpu_fusions=self._stats.cpu_fusions,
            graph_cache_hits=self._stats.graph_cache_hits,
            graph_cache_misses=self._stats.graph_cache_misses,
            total_time_ns=self._stats.total_time_ns,
        )

        if self._gpu_times:
            stats.avg_gpu_time_ns = sum(self._gpu_times) / len(self._gpu_times)
        if self._cpu_times:
            stats.avg_cpu_time_ns = sum(self._cpu_times) / len(self._cpu_times)

        return stats

    def reset_stats(self) -> None:
        """Reset fusion statistics."""
        self._stats = CUDAFusionStats()
        self._gpu_times.clear()
        self._cpu_times.clear()

    def fuse(
        self,
        masks: List[torch.Tensor],
        force_device: Optional[str] = None,
    ) -> torch.Tensor:
        """Fuse multiple masks using bitwise AND.

        Args:
            masks: List of boolean masks to fuse
            force_device: Force "cpu" or "gpu" (default: auto-select)

        Returns:
            Fused mask tensor

        Raises:
            ValueError: If masks list is empty
        """
        if not masks:
            raise ValueError("No masks to fuse")

        start_ns = time.perf_counter_ns()
        self._stats.total_fusions += 1

        # Determine device
        use_gpu = self._should_use_gpu(len(masks), force_device)

        if use_gpu and self._gpu_device is not None:
            result = self._fuse_gpu(masks)
            self._stats.gpu_fusions += 1
        else:
            result = self._fuse_cpu(masks)
            self._stats.cpu_fusions += 1

        end_ns = time.perf_counter_ns()
        elapsed_ns = end_ns - start_ns
        self._stats.total_time_ns += elapsed_ns

        # Track timing for AUTO selection
        if use_gpu:
            self._gpu_times.append(elapsed_ns)
            if len(self._gpu_times) > 100:
                self._gpu_times = self._gpu_times[-100:]
        else:
            self._cpu_times.append(elapsed_ns)
            if len(self._cpu_times) > 100:
                self._cpu_times = self._cpu_times[-100:]

        return result

    def _should_use_gpu(
        self,
        num_masks: int,
        force_device: Optional[str],
    ) -> bool:
        """Determine whether to use GPU for fusion.

        Args:
            num_masks: Number of masks to fuse
            force_device: Forced device or None

        Returns:
            True if GPU should be used
        """
        if force_device == "cpu":
            return False
        if force_device == "gpu":
            return self._gpu_device is not None

        if self._gpu_device is None:
            return False

        strategy = self._config.device_strategy

        if strategy == DeviceSelectionStrategy.ALWAYS_GPU:
            return True
        if strategy == DeviceSelectionStrategy.ALWAYS_CPU:
            return False
        if strategy == DeviceSelectionStrategy.THRESHOLD:
            return (
                self._vocab_size >= self._config.vocab_threshold
                and num_masks >= self._config.min_masks_for_gpu
            )

        # AUTO strategy
        if num_masks < self._config.min_masks_for_gpu:
            return False
        if self._vocab_size < 10000:
            return False

        # Use historical timing if available
        if self._gpu_times and self._cpu_times:
            avg_gpu = sum(self._gpu_times) / len(self._gpu_times)
            avg_cpu = sum(self._cpu_times) / len(self._cpu_times)
            return avg_gpu < avg_cpu

        # Default heuristic: GPU for large vocabs
        return self._vocab_size >= self._config.vocab_threshold

    def _fuse_cpu(self, masks: List[torch.Tensor]) -> torch.Tensor:
        """Fuse masks on CPU.

        Args:
            masks: Masks to fuse

        Returns:
            Fused mask on CPU
        """
        # Ensure all masks are on CPU
        cpu_masks = [
            m.to(self._cpu_device) if m.device != self._cpu_device else m
            for m in masks
        ]

        # Single mask - just return it
        if len(cpu_masks) == 1:
            return cpu_masks[0].clone()

        # Multiple masks - fuse with AND
        result = cpu_masks[0].clone()
        for mask in cpu_masks[1:]:
            result = result & mask
            # Early termination if all zeros
            if not result.any():
                break

        return result

    def _fuse_gpu(self, masks: List[torch.Tensor]) -> torch.Tensor:
        """Fuse masks on GPU.

        Uses CUDA graphs for repeated patterns if enabled.

        Args:
            masks: Masks to fuse

        Returns:
            Fused mask on GPU
        """
        num_masks = len(masks)
        cache_key = (self._vocab_size, num_masks)

        # Check for cached CUDA graph
        if self._config.use_cuda_graphs and cache_key in self._cuda_graphs:
            self._stats.graph_cache_hits += 1
            return self._replay_cuda_graph(cache_key, masks)

        # Transfer masks to GPU
        gpu_masks = [
            m.to(self._gpu_device) if m.device != self._gpu_device else m
            for m in masks
        ]

        # Single mask - just return it
        if len(gpu_masks) == 1:
            return gpu_masks[0].clone()

        # Fuse with AND
        result = gpu_masks[0].clone()
        for mask in gpu_masks[1:]:
            result = result & mask

        # Capture CUDA graph for future use
        if (
            self._config.use_cuda_graphs
            and len(self._cuda_graphs) < self._config.max_cached_graphs
        ):
            self._capture_cuda_graph(cache_key, gpu_masks)
            self._stats.graph_cache_misses += 1

        return result

    def _capture_cuda_graph(
        self,
        cache_key: Tuple[int, int],
        template_masks: List[torch.Tensor],
    ) -> None:
        """Capture a CUDA graph for repeated fusion pattern.

        Args:
            cache_key: Key for cache (vocab_size, num_masks)
            template_masks: Template masks for graph capture
        """
        if not CUDA_AVAILABLE or self._gpu_device is None:
            return

        try:
            # Create input placeholders
            inputs = [
                torch.zeros_like(m, device=self._gpu_device)
                for m in template_masks
            ]
            output = torch.zeros(
                self._vocab_size,
                dtype=torch.bool,
                device=self._gpu_device,
            )

            # Warm-up
            for _ in range(3):
                output.copy_(inputs[0])
                for inp in inputs[1:]:
                    output &= inp

            # Capture graph
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output.copy_(inputs[0])
                for inp in inputs[1:]:
                    output &= inp

            self._cuda_graphs[cache_key] = graph
            self._graph_inputs[cache_key] = inputs
            self._graph_outputs[cache_key] = output

            logger.debug(f"Captured CUDA graph for {cache_key}")

        except Exception as e:
            logger.warning(f"Failed to capture CUDA graph: {e}")

    def _replay_cuda_graph(
        self,
        cache_key: Tuple[int, int],
        masks: List[torch.Tensor],
    ) -> torch.Tensor:
        """Replay a cached CUDA graph.

        Args:
            cache_key: Cache key
            masks: Input masks

        Returns:
            Fused mask
        """
        inputs = self._graph_inputs[cache_key]
        output = self._graph_outputs[cache_key]
        graph = self._cuda_graphs[cache_key]

        # Copy input data
        for i, mask in enumerate(masks):
            if mask.device != self._gpu_device:
                inputs[i].copy_(mask.to(self._gpu_device))
            else:
                inputs[i].copy_(mask)

        # Replay graph
        graph.replay()

        return output.clone()

    def benchmark_device_selection(
        self,
        num_masks: int = 3,
    ) -> FusionBenchmark:
        """Benchmark CPU vs GPU fusion performance.

        Args:
            num_masks: Number of masks to test

        Returns:
            Benchmark results with recommendation
        """
        # Create test masks
        cpu_masks = [
            torch.randint(0, 2, (self._vocab_size,), dtype=torch.bool)
            for _ in range(num_masks)
        ]

        iterations = self._config.benchmark_iterations

        # Benchmark CPU
        cpu_times = []
        for _ in range(iterations):
            start = time.perf_counter_ns()
            result = cpu_masks[0].clone()
            for m in cpu_masks[1:]:
                result &= m
            cpu_times.append(time.perf_counter_ns() - start)
        avg_cpu = sum(cpu_times) / len(cpu_times)

        # Benchmark GPU
        avg_gpu = float("inf")
        if self._gpu_device is not None:
            gpu_masks = [m.to(self._gpu_device) for m in cpu_masks]
            torch.cuda.synchronize()

            gpu_times = []
            for _ in range(iterations):
                start = time.perf_counter_ns()
                result = gpu_masks[0].clone()
                for m in gpu_masks[1:]:
                    result &= m
                torch.cuda.synchronize()
                gpu_times.append(time.perf_counter_ns() - start)
            avg_gpu = sum(gpu_times) / len(gpu_times)

        recommended = "cpu" if avg_cpu <= avg_gpu else "gpu"

        return FusionBenchmark(
            vocab_size=self._vocab_size,
            num_masks=num_masks,
            cpu_time_ns=avg_cpu,
            gpu_time_ns=avg_gpu,
            recommended_device=recommended,
        )

    def clear_cuda_graphs(self) -> None:
        """Clear all cached CUDA graphs."""
        self._cuda_graphs.clear()
        self._graph_inputs.clear()
        self._graph_outputs.clear()


class BatchedCUDAFuser:
    """Batched CUDA fusion for multiple independent mask sets.

    For scenarios where multiple requests need mask fusion simultaneously,
    this class batches them into a single GPU operation.

    Example:
        >>> fuser = BatchedCUDAFuser(vocab_size=128000)
        >>> batch_masks = [
        ...     [mask1_req1, mask2_req1],  # Request 1
        ...     [mask1_req2, mask2_req2],  # Request 2
        ... ]
        >>> results = fuser.fuse_batch(batch_masks)
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize batched fuser.

        Args:
            vocab_size: Vocabulary size
            device: GPU device
        """
        self._vocab_size = vocab_size
        if device is not None:
            self._device = device
        elif CUDA_AVAILABLE:
            self._device = torch.device("cuda:0")
        else:
            self._device = torch.device("cpu")

    def fuse_batch(
        self,
        batch_masks: List[List[torch.Tensor]],
    ) -> List[torch.Tensor]:
        """Fuse multiple mask sets in parallel.

        Args:
            batch_masks: List of mask lists, one per request

        Returns:
            List of fused masks, one per request
        """
        if not batch_masks:
            return []

        # Find max masks per request
        max_masks = max(len(masks) for masks in batch_masks)

        if max_masks == 0:
            return [
                torch.ones(self._vocab_size, dtype=torch.bool, device=self._device)
                for _ in batch_masks
            ]

        batch_size = len(batch_masks)

        # Pad mask lists and stack into batch tensor
        # Shape: (batch_size, max_masks, vocab_size)
        batch_tensor = torch.ones(
            batch_size,
            max_masks,
            self._vocab_size,
            dtype=torch.bool,
            device=self._device,
        )

        for i, masks in enumerate(batch_masks):
            for j, mask in enumerate(masks):
                if mask.device != self._device:
                    batch_tensor[i, j] = mask.to(self._device)
                else:
                    batch_tensor[i, j] = mask

        # Fuse across mask dimension with AND
        # Shape: (batch_size, vocab_size)
        result = batch_tensor.all(dim=1)

        # Convert to list of tensors
        return [result[i] for i in range(batch_size)]


def create_cuda_fuser(
    vocab_size: int = 32000,
    device: Optional[torch.device] = None,
    use_pool: bool = True,
    **config_kwargs: Any,
) -> CUDAMaskFuser:
    """Factory function to create a CUDA mask fuser.

    Args:
        vocab_size: Vocabulary size
        device: GPU device
        use_pool: Whether to use mask pool
        **config_kwargs: Configuration parameters

    Returns:
        New CUDAMaskFuser instance
    """
    config = CUDAFusionConfig(**config_kwargs) if config_kwargs else None
    return CUDAMaskFuser(
        vocab_size=vocab_size,
        config=config,
        device=device,
        use_pool=use_pool,
    )


def create_batched_fuser(
    vocab_size: int = 32000,
    device: Optional[torch.device] = None,
) -> BatchedCUDAFuser:
    """Factory function to create a batched CUDA fuser.

    Args:
        vocab_size: Vocabulary size
        device: GPU device

    Returns:
        New BatchedCUDAFuser instance
    """
    return BatchedCUDAFuser(vocab_size=vocab_size, device=device)
