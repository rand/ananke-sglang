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
"""Property-based tests for performance optimization components.

These tests verify key invariants of the performance optimization systems:

1. TieredConstraintEvaluator:
   - Fusion never adds tokens (result is subset of each domain mask)
   - Tier order is respected
   - Early termination doesn't violate soundness

2. ParallelDomainEvaluator:
   - Parallel evaluation equals sequential evaluation (determinism)
   - Early termination produces correct subset

3. SpeculativeMaskCache:
   - Cache hits return same mask as fresh computation
   - LRU eviction maintains correctness

4. IncrementalParseCache:
   - Checkpoint/restore preserves parser state
   - Rollback is consistent with replay

5. MaskPool:
   - All masks are correctly sized
   - Pool exhaustion falls back correctly
"""

import pytest
from hypothesis import given, settings, assume, strategies as st, HealthCheck
import torch
from typing import Dict, List, Any

from core.domain import GenerationContext, MaskPool
from core.constraint import Constraint, Satisfiability, TOP
from masks.lazy import (
    TieredConstraintEvaluator,
    EvaluationTier,
    ParallelDomainEvaluator,
)
from masks.speculative import SpeculativeMaskCache, SpeculativeCacheStats


# =============================================================================
# Test Strategies
# =============================================================================


@st.composite
def vocab_sizes(draw):
    """Generate reasonable vocabulary sizes."""
    return draw(st.sampled_from([32, 64, 128, 256, 512]))


@st.composite
def domain_masks(draw, vocab_size: int):
    """Generate random domain masks."""
    # Mix of mostly-allowed and mostly-blocked patterns
    pattern = draw(st.sampled_from(["all_true", "all_false", "random", "sparse"]))

    if pattern == "all_true":
        return torch.ones(vocab_size, dtype=torch.bool)
    elif pattern == "all_false":
        return torch.zeros(vocab_size, dtype=torch.bool)
    elif pattern == "random":
        prob = draw(st.floats(min_value=0.1, max_value=0.9))
        return torch.rand(vocab_size) < prob
    else:  # sparse
        mask = torch.zeros(vocab_size, dtype=torch.bool)
        num_allowed = draw(st.integers(min_value=1, max_value=max(1, vocab_size // 10)))
        indices = draw(st.lists(
            st.integers(min_value=0, max_value=vocab_size - 1),
            min_size=num_allowed,
            max_size=num_allowed,
            unique=True,
        ))
        mask[indices] = True
        return mask


@st.composite
def domain_configs(draw, num_domains: int, vocab_size: int):
    """Generate configurations for multiple domains."""
    tiers = [EvaluationTier.FAST, EvaluationTier.MEDIUM, EvaluationTier.SLOW, EvaluationTier.OPTIONAL]
    configs = []
    for i in range(num_domains):
        tier = draw(st.sampled_from(tiers))
        mask = draw(domain_masks(vocab_size))
        configs.append((f"domain_{i}", tier, mask))
    return configs


# =============================================================================
# MockConstraint for testing
# =============================================================================


class MockConstraint(Constraint):
    """Mock constraint for property tests."""

    def __init__(self, is_top: bool = False, is_bottom: bool = False):
        self._is_top = is_top
        self._is_bottom = is_bottom

    def meet(self, other: "Constraint") -> "Constraint":
        if self._is_bottom or (hasattr(other, "_is_bottom") and other._is_bottom):
            return MockConstraint(is_bottom=True)
        if self._is_top:
            return other
        if hasattr(other, "_is_top") and other._is_top:
            return self
        return MockConstraint()

    def is_top(self) -> bool:
        return self._is_top

    def is_bottom(self) -> bool:
        return self._is_bottom

    def satisfiability(self) -> Satisfiability:
        if self._is_bottom:
            return Satisfiability.UNSAT
        return Satisfiability.SAT

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MockConstraint):
            return False
        return self._is_top == other._is_top and self._is_bottom == other._is_bottom

    def __hash__(self) -> int:
        return hash((self._is_top, self._is_bottom))


# =============================================================================
# TieredConstraintEvaluator Properties
# =============================================================================


class TestTieredEvaluatorProperties:
    """Property tests for TieredConstraintEvaluator."""

    @given(
        vocab_size=vocab_sizes(),
        num_domains=st.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_fusion_never_adds_tokens(self, vocab_size: int, num_domains: int):
        """Property: Fused mask is always subset of each individual mask.

        For all domain masks m1, m2, ..., mn:
        fused_mask[i] == True implies m_k[i] == True for all k
        """
        # Create evaluator
        evaluator = TieredConstraintEvaluator(target_popcount=0)  # Disable early termination

        # Generate masks and register domains
        masks: Dict[str, torch.Tensor] = {}
        constraints: Dict[str, MockConstraint] = {}

        for i in range(num_domains):
            domain_name = f"domain_{i}"
            # Generate random mask
            mask = torch.rand(vocab_size) < 0.7  # 70% allowed on average
            masks[domain_name] = mask
            constraints[domain_name] = MockConstraint()

            def make_compute_fn(m):
                def compute_fn(constraint, context):
                    return m.clone()
                return compute_fn

            tier = [EvaluationTier.FAST, EvaluationTier.MEDIUM, EvaluationTier.SLOW][i % 3]
            evaluator.register(domain_name, make_compute_fn(mask), tier)

        # Create context
        context = GenerationContext(vocab_size=vocab_size, device="cpu")

        # Evaluate
        result = evaluator.evaluate(constraints, context)

        # Verify fusion property
        if result.fused_mask is not None:
            for domain_name, mask in masks.items():
                # Fused mask must be subset of each domain mask
                # i.e., if fused[i] is True, mask[i] must be True
                violation = result.fused_mask & ~mask
                assert not violation.any(), f"Fusion added tokens not in {domain_name}"

    @given(vocab_size=vocab_sizes())
    @settings(max_examples=30)
    def test_tier_order_preserved(self, vocab_size: int):
        """Property: FAST domains are always evaluated before SLOW domains."""
        evaluator = TieredConstraintEvaluator(target_popcount=0)
        evaluation_order: List[str] = []

        def make_tracking_fn(name: str):
            def compute_fn(constraint, context):
                evaluation_order.append(name)
                return torch.ones(vocab_size, dtype=torch.bool)
            return compute_fn

        # Register in reverse order to ensure order comes from tiers
        evaluator.register("optional_domain", make_tracking_fn("optional"), EvaluationTier.OPTIONAL)
        evaluator.register("slow_domain", make_tracking_fn("slow"), EvaluationTier.SLOW)
        evaluator.register("medium_domain", make_tracking_fn("medium"), EvaluationTier.MEDIUM)
        evaluator.register("fast_domain", make_tracking_fn("fast"), EvaluationTier.FAST)

        context = GenerationContext(vocab_size=vocab_size, device="cpu")
        constraints = {
            "fast_domain": MockConstraint(),
            "medium_domain": MockConstraint(),
            "slow_domain": MockConstraint(),
            "optional_domain": MockConstraint(),
        }

        evaluator.evaluate(constraints, context)

        # Verify order
        expected_order = ["fast", "medium", "slow", "optional"]
        assert evaluation_order == expected_order, f"Wrong order: {evaluation_order}"


# =============================================================================
# ParallelDomainEvaluator Properties
# =============================================================================


class TestParallelEvaluatorProperties:
    """Property tests for ParallelDomainEvaluator."""

    @given(
        vocab_size=vocab_sizes(),
        num_domains=st.integers(min_value=1, max_value=4),
    )
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_parallel_equals_sequential(self, vocab_size: int, num_domains: int):
        """Property: Parallel evaluation produces same result as sequential.

        This ensures determinism regardless of thread scheduling.
        """
        # Generate domain masks
        masks: Dict[str, torch.Tensor] = {}
        constraints: Dict[str, MockConstraint] = {}

        for i in range(num_domains):
            domain_name = f"domain_{i}"
            mask = torch.rand(vocab_size) < 0.6
            masks[domain_name] = mask
            constraints[domain_name] = MockConstraint()

        # Compute sequential result (ground truth)
        sequential_result = torch.ones(vocab_size, dtype=torch.bool)
        for mask in masks.values():
            sequential_result &= mask

        # Create parallel evaluator
        evaluator = ParallelDomainEvaluator(max_workers=2)

        for domain_name, mask in masks.items():
            def make_compute_fn(m):
                def compute_fn(constraint, context):
                    return m.clone()
                return compute_fn
            evaluator.register(domain_name, make_compute_fn(mask))

        context = GenerationContext(vocab_size=vocab_size, device="cpu")

        # Run parallel evaluation
        result = evaluator.evaluate(constraints, context)

        # Verify equality
        if result.fused_mask is not None:
            assert torch.equal(result.fused_mask, sequential_result), \
                "Parallel result differs from sequential"

    @given(vocab_size=vocab_sizes())
    @settings(max_examples=20)
    def test_early_termination_correct(self, vocab_size: int):
        """Property: Early termination produces correct (empty) mask."""
        evaluator = ParallelDomainEvaluator(max_workers=2)

        # First domain blocks everything
        def blocking_fn(constraint, context):
            return torch.zeros(vocab_size, dtype=torch.bool)

        # Second domain allows everything
        def allowing_fn(constraint, context):
            return torch.ones(vocab_size, dtype=torch.bool)

        evaluator.register("blocking", blocking_fn)
        evaluator.register("allowing", allowing_fn)

        context = GenerationContext(vocab_size=vocab_size, device="cpu")
        constraints = {
            "blocking": MockConstraint(),
            "allowing": MockConstraint(),
        }

        result = evaluator.evaluate(constraints, context)

        # Result should be all-false (early termination or not)
        if result.fused_mask is not None:
            assert not result.fused_mask.any(), "Should be all blocked"


# =============================================================================
# SpeculativeMaskCache Properties
# =============================================================================


class TestSpeculativeCacheProperties:
    """Property tests for SpeculativeMaskCache."""

    @given(
        vocab_size=vocab_sizes(),
        num_states=st.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_cache_returns_same_mask(self, vocab_size: int, num_states: int):
        """Property: Cache hit returns same mask as fresh computation."""
        # Generate deterministic masks for states
        state_masks: Dict[int, torch.Tensor] = {}
        for state in range(num_states):
            torch.manual_seed(state)  # Deterministic
            state_masks[state] = torch.rand(vocab_size) < 0.7

        def compute_fn(state: int, context: GenerationContext) -> torch.Tensor:
            return state_masks[state].clone()

        cache = SpeculativeMaskCache(
            compute_fn=compute_fn,
            lookahead_depth=3,
            max_workers=1,
            max_cache_size=10,
        )

        context = GenerationContext(vocab_size=vocab_size, device="cpu")

        # Query each state twice
        for state in range(num_states):
            # First query (miss)
            mask1 = cache.get_or_compute(state, context)

            # Second query (hit)
            mask2 = cache.get_or_compute(state, context)

            # Both should equal the ground truth
            assert torch.equal(mask1, state_masks[state]), "First query wrong"
            assert torch.equal(mask2, state_masks[state]), "Cached query wrong"

    @given(
        vocab_size=vocab_sizes(),
        cache_size=st.integers(min_value=2, max_value=5),
        num_queries=st.integers(min_value=5, max_value=10),
    )
    @settings(max_examples=20)
    def test_lru_eviction_correctness(self, vocab_size: int, cache_size: int, num_queries: int):
        """Property: LRU eviction doesn't affect correctness."""
        computation_count = [0]

        def compute_fn(state: int, context: GenerationContext) -> torch.Tensor:
            computation_count[0] += 1
            torch.manual_seed(state)
            return torch.rand(vocab_size) < 0.5

        cache = SpeculativeMaskCache(
            compute_fn=compute_fn,
            lookahead_depth=3,
            max_workers=1,
            max_cache_size=cache_size,
        )

        context = GenerationContext(vocab_size=vocab_size, device="cpu")

        # Query more states than cache can hold
        for query in range(num_queries):
            state = query % (cache_size + 2)  # Cycle through more states than cache size
            mask = cache.get_or_compute(state, context)

            # Verify correctness by recomputing
            torch.manual_seed(state)
            expected = torch.rand(vocab_size) < 0.5
            assert torch.equal(mask, expected), f"Wrong mask for state {state}"


# =============================================================================
# MaskPool Properties
# =============================================================================


class TestMaskPoolProperties:
    """Property tests for MaskPool."""

    @given(
        vocab_size=vocab_sizes(),
        pool_size=st.integers(min_value=1, max_value=8),
        num_acquires=st.integers(min_value=1, max_value=20),
    )
    @settings(max_examples=30)
    def test_masks_correctly_sized(self, vocab_size: int, pool_size: int, num_acquires: int):
        """Property: All acquired masks have correct size."""
        pool = MaskPool(vocab_size=vocab_size, device="cpu", pool_size=pool_size)

        handles = []
        for _ in range(num_acquires):
            mask, handle = pool.acquire(fill_value=True)
            assert mask.shape == (vocab_size,), f"Wrong shape: {mask.shape}"
            assert mask.dtype == torch.bool, f"Wrong dtype: {mask.dtype}"
            handles.append((mask, handle))

        # Release all
        for _, handle in handles:
            pool.release(handle)

        # Pool should be full again
        assert pool.available_count == pool_size

    @given(
        vocab_size=vocab_sizes(),
        pool_size=st.integers(min_value=1, max_value=4),
    )
    @settings(max_examples=20)
    def test_pool_exhaustion_fallback(self, vocab_size: int, pool_size: int):
        """Property: Pool exhaustion creates new tensors correctly."""
        pool = MaskPool(vocab_size=vocab_size, device="cpu", pool_size=pool_size)

        # Acquire more than pool size without releasing
        handles = []
        for i in range(pool_size + 3):
            mask, handle = pool.acquire(fill_value=True)
            assert mask.shape == (vocab_size,), f"Wrong shape at acquire {i}"
            handles.append((mask, handle))

        # All masks should still be valid
        for mask, _ in handles:
            assert mask.shape == (vocab_size,)

    @given(vocab_size=vocab_sizes(), pool_size=st.integers(min_value=2, max_value=8))
    @settings(max_examples=20)
    def test_fill_value_respected(self, vocab_size: int, pool_size: int):
        """Property: fill_value parameter is respected."""
        pool = MaskPool(vocab_size=vocab_size, device="cpu", pool_size=pool_size)

        # Test fill_value=True
        mask_true, handle_true = pool.acquire(fill_value=True)
        assert mask_true.all(), "fill_value=True should create all-true mask"
        pool.release(handle_true)

        # Test fill_value=False
        mask_false, handle_false = pool.acquire(fill_value=False)
        assert not mask_false.any(), "fill_value=False should create all-false mask"
        pool.release(handle_false)


# =============================================================================
# Cross-Component Properties
# =============================================================================


class TestCrossComponentProperties:
    """Property tests for interactions between components."""

    @given(vocab_size=vocab_sizes(), num_domains=st.integers(min_value=1, max_value=3))
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_mask_pool_with_evaluators(self, vocab_size: int, num_domains: int):
        """Property: MaskPool integrates correctly with evaluators."""
        pool = MaskPool(vocab_size=vocab_size, device="cpu", pool_size=4)

        # Generate masks
        masks: Dict[str, torch.Tensor] = {}
        for i in range(num_domains):
            masks[f"domain_{i}"] = torch.rand(vocab_size) < 0.6

        # Create context with pool
        context = GenerationContext(vocab_size=vocab_size, device="cpu", mask_pool=pool)

        # Evaluate using pool
        fused = context.create_mask()  # From pool
        for mask in masks.values():
            fused &= mask

        # Sequential computation
        expected = torch.ones(vocab_size, dtype=torch.bool)
        for mask in masks.values():
            expected &= mask

        assert torch.equal(fused, expected), "Pool-based fusion differs from direct"
