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
"""Token mask computation and fusion across constraint domains.

This module provides:
- TokenMaskFuser: Fuses masks from multiple domains
- MaskCache: LRU cache for computed masks
- IncrementalMaskComputer: Incremental mask computation
- LazyConstraintEvaluator: Budget-limited lazy evaluation
"""

from __future__ import annotations

from .fuser import (
    TokenMaskFuser,
    MultiDomainMaskFuser,
    FusionStrategy,
    FusionResult,
    DomainMaskInfo,
    create_fuser,
)
from .cache import (
    MaskCache,
    DomainCache,
    MultiDomainCache,
    CacheKey,
    CacheEntry,
    CacheStats,
    create_cache,
)
from .incremental import (
    IncrementalMaskComputer,
    PositionAwareMaskComputer,
    ChangeKind,
    ConstraintChange,
    ComputationResult,
    create_incremental_computer,
)
from .lazy import (
    LazyConstraintEvaluator,
    AdaptiveLazyEvaluator,
    LazyMask,
    EvaluationPriority,
    EvaluationBudget,
    LazyEvaluationResult,
    create_lazy_evaluator,
    # Tiered evaluation
    TieredConstraintEvaluator,
    EvaluationTier,
    TieredEvaluationResult,
    DEFAULT_DOMAIN_TIERS,
    create_tiered_evaluator,
    # Parallel evaluation
    ParallelDomainEvaluator,
    ParallelEvaluationResult,
    create_parallel_evaluator,
)
from .speculative import (
    SpeculativeMaskCache,
    SpeculativeCacheEntry,
    SpeculativeCacheStats,
    create_speculative_cache,
)
from .pool import (
    MaskPool,
    MaskPoolStats,
    MultiDeviceMaskPool,
    create_mask_pool,
    create_multi_device_pool,
)
from .cuda_fusion import (
    CUDAMaskFuser,
    CUDAFusionConfig,
    CUDAFusionStats,
    FusionBenchmark,
    BatchedCUDAFuser,
    DeviceSelectionStrategy,
    create_cuda_fuser,
    create_batched_fuser,
)

__all__ = [
    # Fuser
    "TokenMaskFuser",
    "MultiDomainMaskFuser",
    "FusionStrategy",
    "FusionResult",
    "DomainMaskInfo",
    "create_fuser",
    # Cache
    "MaskCache",
    "DomainCache",
    "MultiDomainCache",
    "CacheKey",
    "CacheEntry",
    "CacheStats",
    "create_cache",
    # Incremental
    "IncrementalMaskComputer",
    "PositionAwareMaskComputer",
    "ChangeKind",
    "ConstraintChange",
    "ComputationResult",
    "create_incremental_computer",
    # Lazy
    "LazyConstraintEvaluator",
    "AdaptiveLazyEvaluator",
    "LazyMask",
    "EvaluationPriority",
    "EvaluationBudget",
    "LazyEvaluationResult",
    "create_lazy_evaluator",
    # Tiered
    "TieredConstraintEvaluator",
    "EvaluationTier",
    "TieredEvaluationResult",
    "DEFAULT_DOMAIN_TIERS",
    "create_tiered_evaluator",
    # Parallel
    "ParallelDomainEvaluator",
    "ParallelEvaluationResult",
    "create_parallel_evaluator",
    # Speculative
    "SpeculativeMaskCache",
    "SpeculativeCacheEntry",
    "SpeculativeCacheStats",
    "create_speculative_cache",
    # Pool
    "MaskPool",
    "MaskPoolStats",
    "MultiDeviceMaskPool",
    "create_mask_pool",
    "create_multi_device_pool",
    # CUDA Fusion
    "CUDAMaskFuser",
    "CUDAFusionConfig",
    "CUDAFusionStats",
    "FusionBenchmark",
    "BatchedCUDAFuser",
    "DeviceSelectionStrategy",
    "create_cuda_fuser",
    "create_batched_fuser",
]
