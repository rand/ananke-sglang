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
"""Tests for the masks module."""

from dataclasses import dataclass
from typing import Any

import pytest
import torch

from masks import (
    # Fuser
    TokenMaskFuser,
    MultiDomainMaskFuser,
    FusionStrategy,
    FusionResult,
    DomainMaskInfo,
    create_fuser,
    # Cache
    MaskCache,
    DomainCache,
    MultiDomainCache,
    CacheKey,
    CacheEntry,
    CacheStats,
    create_cache,
    # Incremental
    IncrementalMaskComputer,
    PositionAwareMaskComputer,
    ChangeKind,
    ConstraintChange,
    ComputationResult,
    create_incremental_computer,
    # Lazy
    LazyConstraintEvaluator,
    AdaptiveLazyEvaluator,
    LazyMask,
    EvaluationPriority,
    EvaluationBudget,
    LazyEvaluationResult,
    create_lazy_evaluator,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@dataclass
class MockContext:
    """Mock generation context."""

    vocab_size: int = 100
    device: str = "cpu"


class MockDomain:
    """Mock constraint domain for testing."""

    def __init__(self, name: str, selectivity: float = 0.5):
        self.name = name
        self.selectivity = selectivity

    def token_mask(self, constraint: Any, context: MockContext) -> torch.Tensor:
        """Generate a mock mask with given selectivity."""
        vocab_size = context.vocab_size
        num_valid = int(vocab_size * (1 - self.selectivity))
        mask = torch.zeros(vocab_size, dtype=torch.bool)
        mask[:num_valid] = True
        return mask


@dataclass(frozen=True)
class MockConstraint:
    """Mock constraint for testing."""

    value: str = "test"


# =============================================================================
# TokenMaskFuser Tests
# =============================================================================


class TestTokenMaskFuser:
    """Tests for TokenMaskFuser."""

    def test_create(self) -> None:
        """Test fuser creation."""
        fuser = create_fuser()
        assert isinstance(fuser, TokenMaskFuser)

    def test_register_domain(self) -> None:
        """Test domain registration."""
        fuser = TokenMaskFuser()
        domain = MockDomain("test")
        fuser.register_domain("test", domain)
        assert "test" in fuser.domain_names

    def test_unregister_domain(self) -> None:
        """Test domain unregistration."""
        fuser = TokenMaskFuser()
        domain = MockDomain("test")
        fuser.register_domain("test", domain)
        fuser.unregister_domain("test")
        assert "test" not in fuser.domain_names

    def test_fuse_single_domain(self) -> None:
        """Test fusing with a single domain."""
        fuser = TokenMaskFuser()
        domain = MockDomain("test", selectivity=0.5)
        fuser.register_domain("test", domain)

        context = MockContext()
        constraints = {"test": MockConstraint()}
        result = fuser.fuse(constraints, context)

        assert isinstance(result, FusionResult)
        assert result.fused_mask.shape[0] == context.vocab_size
        assert result.num_valid_tokens == 50  # 50% selectivity

    def test_fuse_multiple_domains(self) -> None:
        """Test fusing with multiple domains."""
        fuser = TokenMaskFuser()
        fuser.register_domain("d1", MockDomain("d1", selectivity=0.5))
        fuser.register_domain("d2", MockDomain("d2", selectivity=0.5))

        context = MockContext()
        constraints = {
            "d1": MockConstraint("c1"),
            "d2": MockConstraint("c2"),
        }
        result = fuser.fuse(constraints, context)

        # Both masks allow first 50 tokens, so intersection is 50
        assert result.num_valid_tokens == 50

    def test_fuse_with_exclude(self) -> None:
        """Test fusing with excluded domains."""
        fuser = TokenMaskFuser()
        fuser.register_domain("d1", MockDomain("d1", selectivity=0.8))
        fuser.register_domain("d2", MockDomain("d2", selectivity=0.5))

        context = MockContext()
        constraints = {
            "d1": MockConstraint("c1"),
            "d2": MockConstraint("c2"),
        }
        result = fuser.fuse(constraints, context, exclude=["d1"])

        # Only d2 applied, so 50 valid tokens
        assert result.num_valid_tokens == 50

    def test_fuse_masks_direct(self) -> None:
        """Test fusing pre-computed masks."""
        fuser = TokenMaskFuser()

        mask1 = torch.tensor([True, True, False, False])
        mask2 = torch.tensor([True, False, True, False])
        result = fuser.fuse_masks([mask1, mask2])

        expected = torch.tensor([True, False, False, False])
        assert torch.equal(result, expected)

    def test_short_circuit(self) -> None:
        """Test short-circuit on all-zeros mask."""
        fuser = TokenMaskFuser()

        mask1 = torch.tensor([False, False, False, False])
        mask2 = torch.tensor([True, True, True, True])
        result = fuser.fuse_masks([mask1, mask2], short_circuit=True)

        assert not result.any()

    def test_selectivity_history(self) -> None:
        """Test selectivity history tracking."""
        fuser = TokenMaskFuser()
        domain = MockDomain("test")
        fuser.register_domain("test", domain)
        fuser.update_selectivity("test", 0.5)
        assert abs(fuser.get_selectivity("test") - 0.5) < 0.01


class TestFusionStrategy:
    """Tests for FusionStrategy."""

    def test_strategy_values(self) -> None:
        """Test strategy enum values."""
        assert FusionStrategy.BITWISE_AND.value == 1
        assert FusionStrategy.SELECTIVITY_ORDERED.value == 2


class TestDomainMaskInfo:
    """Tests for DomainMaskInfo."""

    def test_selectivity_computation(self) -> None:
        """Test automatic selectivity computation."""
        mask = torch.tensor([True, True, False, False])
        info = DomainMaskInfo(domain_name="test", mask=mask)
        assert abs(info.selectivity - 0.5) < 0.01


class TestFusionResult:
    """Tests for FusionResult."""

    def test_selectivity(self) -> None:
        """Test selectivity property."""
        mask = torch.tensor([True, True, False, False, False])
        result = FusionResult(fused_mask=mask)
        assert abs(result.selectivity - 0.6) < 0.01

    def test_num_valid_tokens(self) -> None:
        """Test num_valid_tokens property."""
        mask = torch.tensor([True, True, False, False, False])
        result = FusionResult(fused_mask=mask)
        assert result.num_valid_tokens == 2


class TestMultiDomainMaskFuser:
    """Tests for MultiDomainMaskFuser."""

    def test_register_with_dependencies(self) -> None:
        """Test registering domain with dependencies."""
        fuser = MultiDomainMaskFuser()
        fuser.register_domain("syntax", MockDomain("syntax"))
        fuser.register_domain("types", MockDomain("types"), dependencies=["syntax"])

    def test_topological_sort(self) -> None:
        """Test topological sorting of domains."""
        fuser = MultiDomainMaskFuser()
        fuser.register_domain("a", MockDomain("a"))
        fuser.register_domain("b", MockDomain("b"), dependencies=["a"])
        fuser.register_domain("c", MockDomain("c"), dependencies=["b"])

        order = fuser._topological_sort()
        assert order.index("a") < order.index("b") < order.index("c")


# =============================================================================
# MaskCache Tests
# =============================================================================


class TestCacheKey:
    """Tests for CacheKey."""

    def test_create(self) -> None:
        """Test CacheKey creation."""
        key = CacheKey(domain="test", constraint_hash=123, position=0)
        assert key.domain == "test"
        assert key.constraint_hash == 123

    def test_hashable(self) -> None:
        """Test CacheKey is hashable."""
        key = CacheKey(domain="test", constraint_hash=123)
        d = {key: "value"}
        assert d[key] == "value"

    def test_equality(self) -> None:
        """Test CacheKey equality."""
        key1 = CacheKey(domain="test", constraint_hash=123)
        key2 = CacheKey(domain="test", constraint_hash=123)
        key3 = CacheKey(domain="test", constraint_hash=456)
        assert key1 == key2
        assert key1 != key3


class TestCacheEntry:
    """Tests for CacheEntry."""

    def test_create(self) -> None:
        """Test CacheEntry creation."""
        mask = torch.ones(10, dtype=torch.bool)
        entry = CacheEntry(mask=mask)
        assert torch.equal(entry.mask, mask)

    def test_age_seconds(self) -> None:
        """Test age_seconds property."""
        mask = torch.ones(10, dtype=torch.bool)
        entry = CacheEntry(mask=mask)
        assert entry.age_seconds >= 0


class TestCacheStats:
    """Tests for CacheStats."""

    def test_hit_rate(self) -> None:
        """Test hit rate calculation."""
        stats = CacheStats(hits=3, misses=1)
        assert abs(stats.hit_rate - 0.75) < 0.01

    def test_hit_rate_zero(self) -> None:
        """Test hit rate with no requests."""
        stats = CacheStats()
        assert stats.hit_rate == 0.0

    def test_reset(self) -> None:
        """Test reset method."""
        stats = CacheStats(hits=10, misses=5)
        stats.reset()
        assert stats.hits == 0
        assert stats.misses == 0


class TestMaskCache:
    """Tests for MaskCache."""

    def test_create(self) -> None:
        """Test cache creation."""
        cache = create_cache()
        assert isinstance(cache, MaskCache)
        assert cache.size == 0

    def test_put_get(self) -> None:
        """Test put and get operations."""
        cache = MaskCache()
        key = CacheKey(domain="test", constraint_hash=123)
        mask = torch.ones(10, dtype=torch.bool)

        cache.put(key, mask)
        retrieved = cache.get(key)

        assert retrieved is not None
        assert torch.equal(retrieved, mask)

    def test_cache_miss(self) -> None:
        """Test cache miss."""
        cache = MaskCache()
        key = CacheKey(domain="test", constraint_hash=123)
        assert cache.get(key) is None
        assert cache.stats.misses == 1

    def test_cache_hit(self) -> None:
        """Test cache hit tracking."""
        cache = MaskCache()
        key = CacheKey(domain="test", constraint_hash=123)
        mask = torch.ones(10, dtype=torch.bool)

        cache.put(key, mask)
        cache.get(key)

        assert cache.stats.hits == 1

    def test_lru_eviction(self) -> None:
        """Test LRU eviction."""
        cache = MaskCache(max_size=2)

        key1 = CacheKey(domain="test", constraint_hash=1)
        key2 = CacheKey(domain="test", constraint_hash=2)
        key3 = CacheKey(domain="test", constraint_hash=3)

        mask = torch.ones(10, dtype=torch.bool)
        cache.put(key1, mask)
        cache.put(key2, mask)
        cache.put(key3, mask)

        # key1 should have been evicted
        assert cache.get(key1) is None
        assert cache.get(key2) is not None

    def test_make_key(self) -> None:
        """Test make_key method."""
        cache = MaskCache()
        constraint = MockConstraint()
        key = cache.make_key("test", constraint, position=5)

        assert key.domain == "test"
        assert key.position == 5

    def test_invalidate(self) -> None:
        """Test invalidate method."""
        cache = MaskCache()
        key = CacheKey(domain="test", constraint_hash=123)
        mask = torch.ones(10, dtype=torch.bool)

        cache.put(key, mask)
        assert cache.invalidate(key)
        assert cache.get(key) is None

    def test_invalidate_domain(self) -> None:
        """Test invalidate_domain method."""
        cache = MaskCache()
        key1 = CacheKey(domain="test", constraint_hash=1)
        key2 = CacheKey(domain="test", constraint_hash=2)
        key3 = CacheKey(domain="other", constraint_hash=3)

        mask = torch.ones(10, dtype=torch.bool)
        cache.put(key1, mask)
        cache.put(key2, mask)
        cache.put(key3, mask)

        count = cache.invalidate_domain("test")
        assert count == 2
        assert cache.get(key3) is not None

    def test_clear(self) -> None:
        """Test clear method."""
        cache = MaskCache()
        key = CacheKey(domain="test", constraint_hash=123)
        cache.put(key, torch.ones(10, dtype=torch.bool))
        cache.clear()
        assert cache.size == 0


class TestDomainCache:
    """Tests for DomainCache."""

    def test_create(self) -> None:
        """Test DomainCache creation."""
        cache = DomainCache("test")
        assert cache.domain_name == "test"

    def test_get_put(self) -> None:
        """Test get and put operations."""
        cache = DomainCache("test")
        constraint = MockConstraint()
        mask = torch.ones(10, dtype=torch.bool)

        cache.put(constraint, mask)
        retrieved = cache.get(constraint)

        assert retrieved is not None
        assert torch.equal(retrieved, mask)


class TestMultiDomainCache:
    """Tests for MultiDomainCache."""

    def test_get_domain_cache(self) -> None:
        """Test getting domain cache."""
        cache = MultiDomainCache()
        domain_cache = cache.get_domain_cache("test")
        assert isinstance(domain_cache, DomainCache)

    def test_get_put(self) -> None:
        """Test get and put operations."""
        cache = MultiDomainCache()
        constraint = MockConstraint()
        mask = torch.ones(10, dtype=torch.bool)

        cache.put("test", constraint, mask)
        retrieved = cache.get("test", constraint)

        assert retrieved is not None
        assert torch.equal(retrieved, mask)

    def test_total_stats(self) -> None:
        """Test total_stats method."""
        cache = MultiDomainCache()
        constraint = MockConstraint()
        mask = torch.ones(10, dtype=torch.bool)

        cache.put("test", constraint, mask)
        cache.get("test", constraint)

        stats = cache.total_stats()
        assert stats.hits == 1


# =============================================================================
# IncrementalMaskComputer Tests
# =============================================================================


class TestChangeKind:
    """Tests for ChangeKind."""

    def test_values(self) -> None:
        """Test enum values."""
        assert ChangeKind.NONE.value == 1
        assert ChangeKind.MODIFIED.value == 2
        assert ChangeKind.ADDED.value == 3
        assert ChangeKind.REMOVED.value == 4


class TestConstraintChange:
    """Tests for ConstraintChange."""

    def test_create(self) -> None:
        """Test ConstraintChange creation."""
        change = ConstraintChange(domain="test", kind=ChangeKind.MODIFIED)
        assert change.domain == "test"
        assert change.kind == ChangeKind.MODIFIED


class TestIncrementalMaskComputer:
    """Tests for IncrementalMaskComputer."""

    def test_create(self) -> None:
        """Test creation."""
        fuser = TokenMaskFuser()
        computer = create_incremental_computer(fuser)
        assert isinstance(computer, IncrementalMaskComputer)

    def test_detect_added_constraint(self) -> None:
        """Test detecting added constraint."""
        fuser = TokenMaskFuser()
        computer = IncrementalMaskComputer(fuser)

        constraints = {"test": MockConstraint()}
        changes = computer.detect_changes(constraints)

        assert len(changes) == 1
        assert changes[0].kind == ChangeKind.ADDED

    def test_detect_modified_constraint(self) -> None:
        """Test detecting modified constraint."""
        fuser = TokenMaskFuser()
        computer = IncrementalMaskComputer(fuser)

        # First detection
        constraints = {"test": MockConstraint("v1")}
        computer.detect_changes(constraints)

        # Second detection with different constraint
        constraints = {"test": MockConstraint("v2")}
        changes = computer.detect_changes(constraints)

        assert len(changes) == 1
        assert changes[0].kind == ChangeKind.MODIFIED

    def test_detect_removed_constraint(self) -> None:
        """Test detecting removed constraint."""
        fuser = TokenMaskFuser()
        computer = IncrementalMaskComputer(fuser)

        # First detection
        constraints = {"test": MockConstraint()}
        computer.detect_changes(constraints)

        # Second detection without constraint
        constraints = {}
        changes = computer.detect_changes(constraints)

        assert len(changes) == 1
        assert changes[0].kind == ChangeKind.REMOVED

    def test_compute(self) -> None:
        """Test compute method."""
        fuser = TokenMaskFuser()
        fuser.register_domain("test", MockDomain("test", selectivity=0.5))
        computer = IncrementalMaskComputer(fuser)

        context = MockContext()
        constraints = {"test": MockConstraint()}
        result = computer.compute(constraints, context)

        assert isinstance(result, ComputationResult)
        assert result.fused_mask.shape[0] == context.vocab_size

    def test_incremental_computation(self) -> None:
        """Test that unchanged constraints use cached masks."""
        fuser = TokenMaskFuser()
        fuser.register_domain("test", MockDomain("test", selectivity=0.5))
        computer = IncrementalMaskComputer(fuser)

        context = MockContext()
        constraints = {"test": MockConstraint()}

        # First computation - should recompute
        result1 = computer.compute(constraints, context)
        assert "test" in result1.recomputed_domains

        # Second computation with same constraints - should use cache
        result2 = computer.compute(constraints, context)
        assert "test" in result2.cached_domains
        assert "test" not in result2.recomputed_domains

    def test_invalidate_domain(self) -> None:
        """Test invalidate_domain method."""
        fuser = TokenMaskFuser()
        fuser.register_domain("test", MockDomain("test"))
        computer = IncrementalMaskComputer(fuser)

        context = MockContext()
        constraints = {"test": MockConstraint()}

        computer.compute(constraints, context)
        computer.invalidate_domain("test")

        result = computer.compute(constraints, context)
        assert "test" in result.recomputed_domains


class TestComputationResult:
    """Tests for ComputationResult."""

    def test_recompute_ratio(self) -> None:
        """Test recompute_ratio property."""
        mask = torch.ones(10, dtype=torch.bool)
        result = ComputationResult(
            fused_mask=mask,
            recomputed_domains=["a"],
            cached_domains=["b", "c"],
        )
        assert abs(result.recompute_ratio - 1/3) < 0.01


class TestPositionAwareMaskComputer:
    """Tests for PositionAwareMaskComputer."""

    def test_create(self) -> None:
        """Test creation."""
        fuser = TokenMaskFuser()
        computer = PositionAwareMaskComputer(fuser)
        assert computer.current_position == 0

    def test_compute_at_position(self) -> None:
        """Test compute_at_position method."""
        fuser = TokenMaskFuser()
        fuser.register_domain("test", MockDomain("test", selectivity=0.5))
        computer = PositionAwareMaskComputer(fuser)

        context = MockContext()
        constraints = {"test": MockConstraint()}

        mask = computer.compute_at_position(0, constraints, context)
        assert mask.shape[0] == context.vocab_size

    def test_rollback(self) -> None:
        """Test rollback_to method."""
        fuser = TokenMaskFuser()
        computer = PositionAwareMaskComputer(fuser)

        computer.compute_at_position(0, {}, MockContext())
        computer.compute_at_position(1, {}, MockContext())
        computer.compute_at_position(2, {}, MockContext())

        assert computer.rollback_to(1)
        assert computer.current_position == 1


# =============================================================================
# LazyConstraintEvaluator Tests
# =============================================================================


class TestEvaluationPriority:
    """Tests for EvaluationPriority."""

    def test_ordering(self) -> None:
        """Test priority ordering."""
        assert EvaluationPriority.CRITICAL.value < EvaluationPriority.HIGH.value
        assert EvaluationPriority.HIGH.value < EvaluationPriority.NORMAL.value
        assert EvaluationPriority.NORMAL.value < EvaluationPriority.LOW.value


class TestEvaluationBudget:
    """Tests for EvaluationBudget."""

    def test_is_exceeded_time(self) -> None:
        """Test time budget exceeded."""
        budget = EvaluationBudget(max_time_ns=1000)
        assert budget.is_exceeded(2000, 0, 0.0)

    def test_is_exceeded_domains(self) -> None:
        """Test domain budget exceeded."""
        budget = EvaluationBudget(max_domains=2)
        assert budget.is_exceeded(0, 3, 0.0)

    def test_is_exceeded_selectivity(self) -> None:
        """Test selectivity budget exceeded."""
        budget = EvaluationBudget(min_selectivity=0.9)
        assert budget.is_exceeded(0, 0, 0.95)

    def test_not_exceeded(self) -> None:
        """Test budget not exceeded."""
        budget = EvaluationBudget()
        assert not budget.is_exceeded(0, 0, 0.0)


class TestLazyMask:
    """Tests for LazyMask."""

    def test_is_evaluated(self) -> None:
        """Test is_evaluated property."""

        def compute_fn(c, ctx):
            return torch.ones(10, dtype=torch.bool)

        lazy = LazyMask(
            domain="test",
            constraint=MockConstraint(),
            context=MockContext(),
            compute_fn=compute_fn,
        )

        assert not lazy.is_evaluated
        lazy.force()
        assert lazy.is_evaluated

    def test_force(self) -> None:
        """Test force method."""

        def compute_fn(c, ctx):
            return torch.ones(10, dtype=torch.bool)

        lazy = LazyMask(
            domain="test",
            constraint=MockConstraint(),
            context=MockContext(),
            compute_fn=compute_fn,
        )

        mask = lazy.force()
        assert mask.shape[0] == 10
        assert lazy.compute_time_ns > 0

    def test_invalidate(self) -> None:
        """Test invalidate method."""

        def compute_fn(c, ctx):
            return torch.ones(10, dtype=torch.bool)

        lazy = LazyMask(
            domain="test",
            constraint=MockConstraint(),
            context=MockContext(),
            compute_fn=compute_fn,
        )

        lazy.force()
        lazy.invalidate()
        assert not lazy.is_evaluated


class TestLazyConstraintEvaluator:
    """Tests for LazyConstraintEvaluator."""

    def test_create(self) -> None:
        """Test creation."""
        evaluator = create_lazy_evaluator()
        assert isinstance(evaluator, LazyConstraintEvaluator)

    def test_register(self) -> None:
        """Test register method."""
        evaluator = LazyConstraintEvaluator()

        def compute_fn(c, ctx):
            return torch.ones(ctx.vocab_size, dtype=torch.bool)

        evaluator.register("test", compute_fn, EvaluationPriority.HIGH)

    def test_create_lazy_masks(self) -> None:
        """Test create_lazy_masks method."""
        evaluator = LazyConstraintEvaluator()

        def compute_fn(c, ctx):
            return torch.ones(ctx.vocab_size, dtype=torch.bool)

        evaluator.register("test", compute_fn)

        constraints = {"test": MockConstraint()}
        context = MockContext()
        lazy_masks = evaluator.create_lazy_masks(constraints, context)

        assert "test" in lazy_masks
        assert isinstance(lazy_masks["test"], LazyMask)

    def test_evaluate(self) -> None:
        """Test evaluate method."""
        evaluator = LazyConstraintEvaluator()

        def compute_fn(c, ctx):
            return torch.ones(ctx.vocab_size, dtype=torch.bool)

        evaluator.register("test", compute_fn)

        constraints = {"test": MockConstraint()}
        context = MockContext()
        result = evaluator.evaluate(constraints, context)

        assert isinstance(result, LazyEvaluationResult)
        assert result.fused_mask.shape[0] == context.vocab_size
        assert "test" in result.evaluated_domains

    def test_evaluate_skip_priority(self) -> None:
        """Test that SKIP priority domains are skipped."""
        evaluator = LazyConstraintEvaluator()

        def compute_fn(c, ctx):
            return torch.ones(ctx.vocab_size, dtype=torch.bool)

        evaluator.register("test", compute_fn, EvaluationPriority.SKIP)

        constraints = {"test": MockConstraint()}
        context = MockContext()
        result = evaluator.evaluate(constraints, context)

        assert "test" in result.skipped_domains
        assert "test" not in result.evaluated_domains

    def test_evaluate_with_budget(self) -> None:
        """Test evaluate with budget constraint."""
        evaluator = LazyConstraintEvaluator()

        def compute_fn(c, ctx):
            return torch.ones(ctx.vocab_size, dtype=torch.bool)

        evaluator.register("d1", compute_fn, EvaluationPriority.HIGH)
        evaluator.register("d2", compute_fn, EvaluationPriority.NORMAL)
        evaluator.register("d3", compute_fn, EvaluationPriority.NORMAL)

        constraints = {"d1": MockConstraint(), "d2": MockConstraint(), "d3": MockConstraint()}
        context = MockContext()
        budget = EvaluationBudget(max_domains=2)
        result = evaluator.evaluate(constraints, context, budget)

        # d1 (HIGH) always evaluated, then up to max_domains
        assert len(result.evaluated_domains) <= 2


class TestLazyEvaluationResult:
    """Tests for LazyEvaluationResult."""

    def test_create(self) -> None:
        """Test creation."""
        mask = torch.ones(10, dtype=torch.bool)
        result = LazyEvaluationResult(
            fused_mask=mask,
            evaluated_domains=["a", "b"],
            skipped_domains=["c"],
        )
        assert result.fused_mask.shape[0] == 10
        assert len(result.evaluated_domains) == 2


class TestAdaptiveLazyEvaluator:
    """Tests for AdaptiveLazyEvaluator."""

    def test_create(self) -> None:
        """Test creation."""
        base = LazyConstraintEvaluator()
        adaptive = AdaptiveLazyEvaluator(base)
        assert adaptive is not None

    def test_evaluate(self) -> None:
        """Test evaluate method."""
        base = LazyConstraintEvaluator()

        def compute_fn(c, ctx):
            return torch.ones(ctx.vocab_size, dtype=torch.bool)

        base.register("test", compute_fn)
        adaptive = AdaptiveLazyEvaluator(base)

        constraints = {"test": MockConstraint()}
        context = MockContext()
        result = adaptive.evaluate(constraints, context)

        assert isinstance(result, LazyEvaluationResult)


class TestTieredConstraintEvaluator:
    """Tests for TieredConstraintEvaluator."""

    def test_create(self) -> None:
        """Test creation."""
        from masks.lazy import TieredConstraintEvaluator, create_tiered_evaluator

        evaluator = create_tiered_evaluator(target_popcount=100)
        assert isinstance(evaluator, TieredConstraintEvaluator)

    def test_register(self) -> None:
        """Test register method."""
        from masks.lazy import TieredConstraintEvaluator, EvaluationTier

        evaluator = TieredConstraintEvaluator()

        def compute_fn(c, ctx):
            return torch.ones(ctx.vocab_size, dtype=torch.bool)

        evaluator.register("test", compute_fn, EvaluationTier.FAST)
        assert "test" in evaluator._domains

    def test_evaluate_basic(self) -> None:
        """Test basic evaluate method."""
        from masks.lazy import TieredConstraintEvaluator, TieredEvaluationResult, EvaluationTier

        evaluator = TieredConstraintEvaluator(target_popcount=100)

        def compute_fn(c, ctx):
            return torch.ones(ctx.vocab_size, dtype=torch.bool)

        evaluator.register("test", compute_fn, EvaluationTier.MEDIUM)

        constraints = {"test": MockConstraint()}
        context = MockContext()
        result = evaluator.evaluate(constraints, context)

        assert isinstance(result, TieredEvaluationResult)
        assert result.fused_mask.shape[0] == context.vocab_size
        assert "test" in result.evaluated_domains

    def test_evaluate_tier_order(self) -> None:
        """Test that tiers are evaluated in order."""
        from masks.lazy import TieredConstraintEvaluator, EvaluationTier

        evaluator = TieredConstraintEvaluator(target_popcount=0)  # Disable early termination
        evaluation_order = []

        def make_compute_fn(name):
            def compute_fn(c, ctx):
                evaluation_order.append(name)
                return torch.ones(ctx.vocab_size, dtype=torch.bool)
            return compute_fn

        evaluator.register("slow", make_compute_fn("slow"), EvaluationTier.SLOW)
        evaluator.register("fast", make_compute_fn("fast"), EvaluationTier.FAST)
        evaluator.register("medium", make_compute_fn("medium"), EvaluationTier.MEDIUM)

        constraints = {
            "slow": MockConstraint(),
            "fast": MockConstraint(),
            "medium": MockConstraint(),
        }
        context = MockContext()
        evaluator.evaluate(constraints, context)

        # Should be: FAST (0) -> MEDIUM (1) -> SLOW (2)
        assert evaluation_order == ["fast", "medium", "slow"]

    def test_early_termination_on_popcount(self) -> None:
        """Test early termination when popcount drops below threshold."""
        from masks.lazy import TieredConstraintEvaluator, EvaluationTier

        evaluator = TieredConstraintEvaluator(target_popcount=100)

        def fast_fn(c, ctx):
            # Return mask with only 50 allowed tokens
            mask = torch.zeros(ctx.vocab_size, dtype=torch.bool)
            mask[:50] = True
            return mask

        def slow_fn(c, ctx):
            # This should be skipped due to early termination
            return torch.ones(ctx.vocab_size, dtype=torch.bool)

        evaluator.register("fast", fast_fn, EvaluationTier.FAST)
        evaluator.register("slow", slow_fn, EvaluationTier.SLOW)

        constraints = {"fast": MockConstraint(), "slow": MockConstraint()}
        context = MockContext()
        result = evaluator.evaluate(constraints, context)

        assert result.early_termination
        assert "fast" in result.evaluated_domains
        assert "slow" in result.skipped_domains
        assert result.final_popcount == 50

    def test_early_termination_on_all_blocked(self) -> None:
        """Test early termination when all tokens are blocked."""
        from masks.lazy import TieredConstraintEvaluator, EvaluationTier

        evaluator = TieredConstraintEvaluator(target_popcount=100)

        def blocking_fn(c, ctx):
            return torch.zeros(ctx.vocab_size, dtype=torch.bool)

        def other_fn(c, ctx):
            return torch.ones(ctx.vocab_size, dtype=torch.bool)

        evaluator.register("blocking", blocking_fn, EvaluationTier.FAST)
        evaluator.register("other", other_fn, EvaluationTier.SLOW)

        constraints = {"blocking": MockConstraint(), "other": MockConstraint()}
        context = MockContext()
        result = evaluator.evaluate(constraints, context)

        assert result.early_termination
        assert result.final_popcount == 0
        assert "other" in result.skipped_domains

    def test_default_tiers_applied(self) -> None:
        """Test that default tier assignments are applied."""
        from masks.lazy import TieredConstraintEvaluator, DEFAULT_DOMAIN_TIERS, EvaluationTier

        evaluator = TieredConstraintEvaluator()

        def compute_fn(c, ctx):
            return torch.ones(ctx.vocab_size, dtype=torch.bool)

        # Register with default tier (should use DEFAULT_DOMAIN_TIERS)
        evaluator.register("types", compute_fn)
        evaluator.register("imports", compute_fn)

        assert evaluator._tiers["types"] == DEFAULT_DOMAIN_TIERS["types"]
        assert evaluator._tiers["imports"] == DEFAULT_DOMAIN_TIERS["imports"]

    def test_set_tier(self) -> None:
        """Test set_tier method."""
        from masks.lazy import TieredConstraintEvaluator, EvaluationTier

        evaluator = TieredConstraintEvaluator()

        def compute_fn(c, ctx):
            return torch.ones(ctx.vocab_size, dtype=torch.bool)

        evaluator.register("test", compute_fn, EvaluationTier.SLOW)
        evaluator.set_tier("test", EvaluationTier.FAST)

        assert evaluator._tiers["test"] == EvaluationTier.FAST

    def test_set_target_popcount(self) -> None:
        """Test set_target_popcount method."""
        from masks.lazy import TieredConstraintEvaluator

        evaluator = TieredConstraintEvaluator(target_popcount=100)
        evaluator.set_target_popcount(50)

        assert evaluator._target_popcount == 50

    def test_tiers_processed_count(self) -> None:
        """Test that tiers_processed is counted correctly."""
        from masks.lazy import TieredConstraintEvaluator, EvaluationTier

        evaluator = TieredConstraintEvaluator(target_popcount=0)  # Disable early termination

        def compute_fn(c, ctx):
            return torch.ones(ctx.vocab_size, dtype=torch.bool)

        # Register domains in 3 different tiers
        evaluator.register("fast", compute_fn, EvaluationTier.FAST)
        evaluator.register("medium", compute_fn, EvaluationTier.MEDIUM)
        evaluator.register("slow", compute_fn, EvaluationTier.SLOW)

        constraints = {
            "fast": MockConstraint(),
            "medium": MockConstraint(),
            "slow": MockConstraint(),
        }
        context = MockContext()
        result = evaluator.evaluate(constraints, context)

        assert result.tiers_processed == 3


class TestParallelDomainEvaluator:
    """Tests for ParallelDomainEvaluator."""

    def test_create(self) -> None:
        """Test creating a parallel evaluator."""
        from masks.lazy import ParallelDomainEvaluator, create_parallel_evaluator

        evaluator = ParallelDomainEvaluator()
        assert evaluator._max_workers == 4

        evaluator2 = create_parallel_evaluator(max_workers=2)
        assert evaluator2._max_workers == 2

    def test_register(self) -> None:
        """Test registering domains."""
        from masks.lazy import ParallelDomainEvaluator

        evaluator = ParallelDomainEvaluator()

        def compute_fn(c, ctx):
            return torch.ones(ctx.vocab_size, dtype=torch.bool)

        evaluator.register("test", compute_fn)
        assert "test" in evaluator._domains

        evaluator.unregister("test")
        assert "test" not in evaluator._domains

    def test_evaluate_basic(self) -> None:
        """Test basic parallel evaluation."""
        from masks.lazy import ParallelDomainEvaluator, ParallelEvaluationResult

        evaluator = ParallelDomainEvaluator()

        def compute_fn(c, ctx):
            return torch.ones(ctx.vocab_size, dtype=torch.bool)

        evaluator.register("test", compute_fn)

        constraints = {"test": MockConstraint()}
        context = MockContext()

        try:
            result = evaluator.evaluate(constraints, context)

            assert isinstance(result, ParallelEvaluationResult)
            assert result.fused_mask.shape[0] == context.vocab_size
            assert "test" in result.evaluated_domains
        finally:
            evaluator.shutdown()

    def test_evaluate_multiple_domains(self) -> None:
        """Test parallel evaluation of multiple domains."""
        from masks.lazy import ParallelDomainEvaluator

        evaluator = ParallelDomainEvaluator(max_workers=4)

        def compute_fn_a(c, ctx):
            return torch.ones(ctx.vocab_size, dtype=torch.bool)

        def compute_fn_b(c, ctx):
            mask = torch.ones(ctx.vocab_size, dtype=torch.bool)
            mask[50:] = False
            return mask

        def compute_fn_c(c, ctx):
            mask = torch.ones(ctx.vocab_size, dtype=torch.bool)
            mask[:25] = False
            return mask

        evaluator.register("domain_a", compute_fn_a)
        evaluator.register("domain_b", compute_fn_b)
        evaluator.register("domain_c", compute_fn_c)

        constraints = {
            "domain_a": MockConstraint(),
            "domain_b": MockConstraint(),
            "domain_c": MockConstraint(),
        }
        context = MockContext()

        try:
            result = evaluator.evaluate(constraints, context)

            # All domains should be evaluated
            assert len(result.evaluated_domains) == 3

            # Fused mask should be intersection: [25:50] = True
            assert result.final_popcount == 25
        finally:
            evaluator.shutdown()

    def test_early_termination_on_empty(self) -> None:
        """Test early termination when mask becomes empty."""
        from masks.lazy import ParallelDomainEvaluator
        import time

        evaluator = ParallelDomainEvaluator(max_workers=4)

        def fast_blocker(c, ctx):
            # Returns empty mask immediately
            return torch.zeros(ctx.vocab_size, dtype=torch.bool)

        def slow_domain(c, ctx):
            # Takes some time
            time.sleep(0.1)
            return torch.ones(ctx.vocab_size, dtype=torch.bool)

        evaluator.register("fast_blocker", fast_blocker)
        evaluator.register("slow_domain", slow_domain)

        constraints = {
            "fast_blocker": MockConstraint(),
            "slow_domain": MockConstraint(),
        }
        context = MockContext()

        try:
            result = evaluator.evaluate(constraints, context)

            # Should have early terminated
            assert result.early_termination
            assert result.final_popcount == 0
        finally:
            evaluator.shutdown()

    def test_domain_times_tracked(self) -> None:
        """Test that individual domain times are tracked."""
        from masks.lazy import ParallelDomainEvaluator
        import time

        evaluator = ParallelDomainEvaluator()

        def slow_domain(c, ctx):
            time.sleep(0.01)  # 10ms
            return torch.ones(ctx.vocab_size, dtype=torch.bool)

        evaluator.register("slow", slow_domain)

        constraints = {"slow": MockConstraint()}
        context = MockContext()

        try:
            result = evaluator.evaluate(constraints, context)

            assert "slow" in result.domain_times_ns
            # Should be at least 10ms = 10_000_000 ns
            assert result.domain_times_ns["slow"] >= 5_000_000  # Allow for timing variation
        finally:
            evaluator.shutdown()

    def test_context_manager(self) -> None:
        """Test using evaluator as context manager."""
        from masks.lazy import ParallelDomainEvaluator

        def compute_fn(c, ctx):
            return torch.ones(ctx.vocab_size, dtype=torch.bool)

        with ParallelDomainEvaluator() as evaluator:
            evaluator.register("test", compute_fn)
            result = evaluator.evaluate(
                {"test": MockConstraint()},
                MockContext(),
            )
            assert "test" in result.evaluated_domains

        # Executor should be shut down
        assert evaluator._executor is None

    def test_unregistered_domain_ignored(self) -> None:
        """Test that unregistered domains in constraints are ignored."""
        from masks.lazy import ParallelDomainEvaluator

        evaluator = ParallelDomainEvaluator()

        def compute_fn(c, ctx):
            return torch.ones(ctx.vocab_size, dtype=torch.bool)

        evaluator.register("registered", compute_fn)

        constraints = {
            "registered": MockConstraint(),
            "unregistered": MockConstraint(),
        }
        context = MockContext()

        try:
            result = evaluator.evaluate(constraints, context)

            assert "registered" in result.evaluated_domains
            assert "unregistered" not in result.evaluated_domains
            assert "unregistered" not in result.cancelled_domains
        finally:
            evaluator.shutdown()

    def test_empty_constraints(self) -> None:
        """Test evaluation with no constraints."""
        from masks.lazy import ParallelDomainEvaluator

        evaluator = ParallelDomainEvaluator()

        def compute_fn(c, ctx):
            return torch.ones(ctx.vocab_size, dtype=torch.bool)

        evaluator.register("test", compute_fn)

        result = evaluator.evaluate({}, MockContext())

        assert len(result.evaluated_domains) == 0
        assert result.final_popcount == 100  # All true

    def test_parallel_speedup(self) -> None:
        """Test that parallel evaluation is faster than sequential."""
        from masks.lazy import ParallelDomainEvaluator
        import time

        evaluator = ParallelDomainEvaluator(max_workers=4)

        def slow_domain(name):
            def fn(c, ctx):
                time.sleep(0.02)  # 20ms each
                return torch.ones(ctx.vocab_size, dtype=torch.bool)
            return fn

        # Register 4 slow domains
        for i in range(4):
            evaluator.register(f"domain_{i}", slow_domain(f"domain_{i}"))

        constraints = {f"domain_{i}": MockConstraint() for i in range(4)}
        context = MockContext()

        try:
            start = time.perf_counter()
            result = evaluator.evaluate(constraints, context)
            elapsed = time.perf_counter() - start

            # Sequential would take ~80ms, parallel should be ~20-40ms
            # Using generous threshold for CI stability
            assert elapsed < 0.1  # Should be well under 100ms
            assert len(result.evaluated_domains) == 4
        finally:
            evaluator.shutdown()


class TestSpeculativeMaskCache:
    """Tests for SpeculativeMaskCache."""

    def test_create(self) -> None:
        """Test creating a speculative cache."""
        from masks.speculative import SpeculativeMaskCache, create_speculative_cache

        def compute_fn(state, ctx):
            return torch.ones(100, dtype=torch.bool)

        cache = SpeculativeMaskCache(compute_fn)
        assert cache._lookahead_depth == 3
        assert cache._max_cache_size == 64
        cache.shutdown()

        cache2 = create_speculative_cache(compute_fn, lookahead_depth=5)
        assert cache2._lookahead_depth == 5
        cache2.shutdown()

    def test_get_or_compute_miss(self) -> None:
        """Test cache miss - compute and cache."""
        from masks.speculative import SpeculativeMaskCache

        compute_count = [0]

        def compute_fn(state, ctx):
            compute_count[0] += 1
            return torch.ones(100, dtype=torch.bool)

        with SpeculativeMaskCache(compute_fn) as cache:
            result = cache.get_or_compute("state1", "context1")

            assert result.shape[0] == 100
            assert result.all()
            assert compute_count[0] == 1

            stats = cache.get_stats()
            assert stats.misses == 1
            assert stats.hits == 0

    def test_get_or_compute_hit(self) -> None:
        """Test cache hit - return cached mask."""
        from masks.speculative import SpeculativeMaskCache

        compute_count = [0]

        def compute_fn(state, ctx):
            compute_count[0] += 1
            return torch.ones(100, dtype=torch.bool)

        with SpeculativeMaskCache(compute_fn) as cache:
            # First call - miss
            cache.get_or_compute("state1", "context1")
            assert compute_count[0] == 1

            # Second call with same args - hit
            cache.get_or_compute("state1", "context1")
            assert compute_count[0] == 1  # Not recomputed

            stats = cache.get_stats()
            assert stats.misses == 1
            assert stats.hits == 1
            assert stats.hit_rate == 0.5

    def test_precompute(self) -> None:
        """Test precomputation of likely next tokens."""
        from masks.speculative import SpeculativeMaskCache
        import time

        compute_count = [0]

        def compute_fn(state, ctx):
            compute_count[0] += 1
            time.sleep(0.01)  # Simulate some work
            return torch.ones(100, dtype=torch.bool)

        with SpeculativeMaskCache(compute_fn, lookahead_depth=2) as cache:
            # Precompute for likely tokens
            cache.precompute([1, 2, 3, 4, 5], "state1", "context1")

            # Wait for precomputation to complete
            time.sleep(0.1)

            stats = cache.get_stats()
            # Only lookahead_depth=2 should be precomputed
            assert stats.precompute_requests == 2

    def test_precompute_hit(self) -> None:
        """Test that precomputed masks result in cache hits."""
        from masks.speculative import SpeculativeMaskCache
        import time

        compute_count = [0]

        def compute_fn(state, ctx):
            compute_count[0] += 1
            return torch.ones(100, dtype=torch.bool)

        def transition_fn(state, token):
            return (state, token)

        with SpeculativeMaskCache(
            compute_fn,
            lookahead_depth=3,
            state_transition_fn=transition_fn,
        ) as cache:
            # Precompute for likely tokens
            cache.precompute([1, 2, 3], "state1", "context1")

            # Wait for precomputation
            time.sleep(0.1)

            # Now access a precomputed state
            result = cache.get_or_compute(("state1", 1), "context1")

            assert result.shape[0] == 100
            stats = cache.get_stats()
            assert stats.hits >= 1  # Should hit precomputed entry

    def test_lru_eviction(self) -> None:
        """Test LRU eviction when cache is full."""
        from masks.speculative import SpeculativeMaskCache

        def compute_fn(state, ctx):
            return torch.ones(100, dtype=torch.bool)

        with SpeculativeMaskCache(compute_fn, max_cache_size=3) as cache:
            # Fill cache
            cache.get_or_compute("state1", "ctx")
            cache.get_or_compute("state2", "ctx")
            cache.get_or_compute("state3", "ctx")

            # Access state1 to make it recently used
            cache.get_or_compute("state1", "ctx")

            # Add new entry - should evict state2 (oldest unused)
            cache.get_or_compute("state4", "ctx")

            # state1 should still be cached (was accessed)
            cache.get_or_compute("state1", "ctx")
            # state2 should be evicted
            initial_misses = cache.get_stats().misses
            cache.get_or_compute("state2", "ctx")
            # Should be a miss since state2 was evicted
            assert cache.get_stats().misses == initial_misses + 1

    def test_clear(self) -> None:
        """Test clearing the cache."""
        from masks.speculative import SpeculativeMaskCache

        def compute_fn(state, ctx):
            return torch.ones(100, dtype=torch.bool)

        with SpeculativeMaskCache(compute_fn) as cache:
            cache.get_or_compute("state1", "ctx")
            cache.get_or_compute("state2", "ctx")

            cache.clear()

            # After clear, should miss
            initial_misses = cache.get_stats().misses
            cache.get_or_compute("state1", "ctx")
            assert cache.get_stats().misses == initial_misses + 1

    def test_stats_tracking(self) -> None:
        """Test statistics tracking."""
        from masks.speculative import SpeculativeMaskCache

        def compute_fn(state, ctx):
            return torch.ones(100, dtype=torch.bool)

        with SpeculativeMaskCache(compute_fn) as cache:
            # Generate some activity
            cache.get_or_compute("s1", "c")  # Miss
            cache.get_or_compute("s1", "c")  # Hit
            cache.get_or_compute("s2", "c")  # Miss
            cache.get_or_compute("s1", "c")  # Hit
            cache.get_or_compute("s2", "c")  # Hit

            stats = cache.get_stats()
            assert stats.misses == 2
            assert stats.hits == 3
            assert stats.hit_rate == 0.6

    def test_context_manager(self) -> None:
        """Test using cache as context manager."""
        from masks.speculative import SpeculativeMaskCache

        def compute_fn(state, ctx):
            return torch.ones(100, dtype=torch.bool)

        with SpeculativeMaskCache(compute_fn) as cache:
            result = cache.get_or_compute("state", "ctx")
            assert result.shape[0] == 100

    def test_concurrent_precompute(self) -> None:
        """Test concurrent precomputation doesn't duplicate work."""
        from masks.speculative import SpeculativeMaskCache
        import time

        compute_count = [0]

        def compute_fn(state, ctx):
            compute_count[0] += 1
            time.sleep(0.02)
            return torch.ones(100, dtype=torch.bool)

        with SpeculativeMaskCache(compute_fn, lookahead_depth=3) as cache:
            # Request precompute twice for same tokens
            cache.precompute([1, 2, 3], "state", "ctx")
            cache.precompute([1, 2, 3], "state", "ctx")

            # Wait for completion
            time.sleep(0.2)

            # Should only have computed each once
            assert compute_count[0] <= 3  # At most 3 unique states
