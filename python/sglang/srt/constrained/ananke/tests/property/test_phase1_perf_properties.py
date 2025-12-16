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
"""Property-based tests for Phase 1 Performance Components.

Tests the mathematical properties of:
1. AdaptiveTieredEvaluator (Phase 1.1)
   - Efficiency score invariants
   - Domain ordering consistency
   - Skip prediction soundness
   - Adaptive threshold convergence

2. TypeMaskCache (Phase 1.2)
   - Cache hit rate bounds
   - Mask dimensionality consistency
   - Type coverage guarantees

3. VocabPartition (Phase 1.3)
   - Classification consistency
   - Category mask validity
   - Token ID bounds
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Callable

import pytest
import torch
from hypothesis import given, strategies as st, assume, settings, HealthCheck

# Import Phase 1 components
try:
    from masks.lazy import (
        AdaptiveTieredEvaluator,
        TieredConstraintEvaluator,
        DomainStats,
        EvaluationTier,
        create_adaptive_tiered_evaluator,
    )
    from domains.types.domain import TypeMaskCache
    from zig.ffi import VocabPartition, TYPE_CATEGORY_UNKNOWN
    from core.token_classifier import TokenClassifier, TokenCategory
except ImportError:
    from sglang.srt.constrained.ananke.masks.lazy import (
        AdaptiveTieredEvaluator,
        TieredConstraintEvaluator,
        DomainStats,
        EvaluationTier,
        create_adaptive_tiered_evaluator,
    )
    from sglang.srt.constrained.ananke.domains.types.domain import TypeMaskCache
    from sglang.srt.constrained.ananke.zig.ffi import VocabPartition, TYPE_CATEGORY_UNKNOWN
    from sglang.srt.constrained.ananke.core.token_classifier import (
        TokenClassifier,
        TokenCategory,
    )


# =============================================================================
# Test Context Mock
# =============================================================================


@dataclass
class MockContext:
    """Mock generation context for testing."""
    vocab_size: int = 1000
    device: str = "cpu"


# =============================================================================
# DomainStats Properties
# =============================================================================


class TestDomainStatsProperties:
    """Property-based tests for DomainStats."""

    @given(
        st.lists(
            st.tuples(
                st.integers(min_value=1000, max_value=10_000_000),  # latency_ns
                st.floats(min_value=0.0, max_value=1.0),  # selectivity
                st.booleans(),  # was_useful
            ),
            min_size=1,
            max_size=50,
        )
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_efficiency_score_non_negative(self, evaluations: List[tuple]) -> None:
        """Efficiency score is always non-negative."""
        stats = DomainStats()
        for latency, selectivity, useful in evaluations:
            stats.record_evaluation(latency, selectivity, useful)

        assert stats.efficiency_score >= 0, "Efficiency score must be non-negative"

    @given(
        st.lists(
            st.booleans(),
            min_size=10,
            max_size=100,
        )
    )
    @settings(max_examples=50)
    def test_usefulness_rate_bounded(self, usefulness: List[bool]) -> None:
        """Usefulness rate is always between 0 and 1."""
        stats = DomainStats()
        for useful in usefulness:
            stats.record_evaluation(100_000, 0.5, useful)

        assert 0.0 <= stats.usefulness_rate <= 1.0, "Usefulness rate must be in [0, 1]"

    @given(st.integers(min_value=0, max_value=100))
    @settings(max_examples=50)
    def test_skip_rate_bounded(self, skip_count: int) -> None:
        """Skip rate is always between 0 and 1."""
        stats = DomainStats()

        # Record some evaluations
        for _ in range(10):
            stats.record_evaluation(100_000, 0.5, True)

        # Record skips
        for _ in range(skip_count):
            stats.record_skip()

        assert 0.0 <= stats.skip_rate <= 1.0, "Skip rate must be in [0, 1]"

    @given(
        st.lists(
            st.booleans(),
            min_size=11,  # Need >10 for should_skip to consider
            max_size=50,
        )
    )
    @settings(max_examples=50)
    def test_should_skip_consistency(self, usefulness: List[bool]) -> None:
        """should_skip is consistent with usefulness_rate threshold."""
        stats = DomainStats()
        for useful in usefulness:
            stats.record_evaluation(100_000, 0.5, useful)

        # should_skip requires usefulness_rate < 0.1 and evaluation_count >= 10
        if stats.evaluation_count >= 10:
            if stats.usefulness_rate < 0.1:
                assert stats.should_skip, "Should skip when usefulness < 10%"
            else:
                assert not stats.should_skip, "Should not skip when usefulness >= 10%"


# =============================================================================
# AdaptiveTieredEvaluator Properties
# =============================================================================


class TestAdaptiveTieredEvaluatorProperties:
    """Property-based tests for AdaptiveTieredEvaluator."""

    @given(st.integers(min_value=10, max_value=1000))
    @settings(max_examples=20)
    def test_target_popcount_sets_correctly(self, target: int) -> None:
        """Target popcount can be set and retrieved."""
        evaluator = create_adaptive_tiered_evaluator(target_popcount=target)
        evaluator.set_target_popcount(target)
        stats = evaluator.get_stats()
        assert stats["base_threshold"] == target

    @given(
        st.lists(
            st.sampled_from(["syntax", "types", "imports", "controlflow", "semantics"]),
            min_size=1,
            max_size=5,
            unique=True,
        )
    )
    @settings(max_examples=30)
    def test_domain_registration_idempotent(self, domains: List[str]) -> None:
        """Registering a domain twice doesn't duplicate it."""
        evaluator = create_adaptive_tiered_evaluator()

        # Create simple mask functions
        def make_mask_fn(selectivity: float):
            def mask_fn(constraint: Any, ctx: Any) -> torch.Tensor:
                vocab_size = getattr(ctx, "vocab_size", 1000)
                device = getattr(ctx, "device", "cpu")
                mask = torch.ones(vocab_size, dtype=torch.bool, device=device)
                # Make selectivity% of tokens False
                n_block = int(vocab_size * selectivity)
                if n_block > 0:
                    mask[:n_block] = False
                return mask
            return mask_fn

        # Register each domain
        for domain in domains:
            evaluator.register(domain, make_mask_fn(0.3))

        # Register again (should not duplicate)
        for domain in domains:
            evaluator.register(domain, make_mask_fn(0.3))

        # Should have exactly len(domains) registered
        assert len(evaluator._domains) == len(domains)

    @given(st.integers(min_value=1, max_value=50))
    @settings(max_examples=20)
    def test_adaptive_threshold_bounded(self, history_size: int) -> None:
        """Adaptive threshold stays within reasonable bounds."""
        evaluator = create_adaptive_tiered_evaluator(target_popcount=100)

        # Simulate history
        for _ in range(history_size):
            evaluator._popcount_history.append(50)

        threshold = evaluator._compute_adaptive_threshold()

        # Should be at least 10 (minimum) and at most 2x base (200)
        assert threshold >= 10, "Threshold should be at least 10"
        assert threshold <= 200, "Threshold should be at most 2x base"

    @given(
        st.lists(
            st.integers(min_value=1, max_value=1000),
            min_size=20,
            max_size=100,
        )
    )
    @settings(max_examples=30)
    def test_adaptive_threshold_uses_percentile(self, popcounts: List[int]) -> None:
        """Adaptive threshold uses 25th percentile of history."""
        evaluator = create_adaptive_tiered_evaluator(target_popcount=100)
        evaluator._popcount_history = popcounts.copy()

        threshold = evaluator._compute_adaptive_threshold()

        # 25th percentile of popcounts
        sorted_pops = sorted(popcounts)
        p25_idx = len(sorted_pops) // 4
        p25 = sorted_pops[p25_idx]

        # Blended with base (100): 0.7 * p25 + 0.3 * 100
        expected_approx = int(0.7 * p25 + 0.3 * 100)

        # Should be close (within clamp bounds)
        assert abs(threshold - max(10, min(expected_approx, 200))) <= 1


# =============================================================================
# TieredConstraintEvaluator Properties
# =============================================================================


class TestTieredEvaluatorProperties:
    """Property-based tests for TieredConstraintEvaluator."""

    @given(
        st.lists(
            st.sampled_from(list(EvaluationTier)),
            min_size=1,
            max_size=4,
        )
    )
    @settings(max_examples=30)
    def test_tiers_processed_in_order(self, tiers: List[EvaluationTier]) -> None:
        """Tiers are always processed in FAST -> MEDIUM -> SLOW -> OPTIONAL order."""
        evaluator = TieredConstraintEvaluator()

        # Track which tiers were seen
        seen_tiers: List[EvaluationTier] = []

        def make_tracking_fn(tier: EvaluationTier):
            def fn(constraint: Any, ctx: Any) -> torch.Tensor:
                seen_tiers.append(tier)
                return torch.ones(ctx.vocab_size, dtype=torch.bool, device=ctx.device)
            return fn

        # Register domains with different tiers
        for i, tier in enumerate(tiers):
            evaluator.register(f"domain_{i}", make_tracking_fn(tier), tier)

        # Evaluate
        constraints = {f"domain_{i}": None for i in range(len(tiers))}
        ctx = MockContext()
        evaluator.evaluate(constraints, ctx)

        # Verify order: seen_tiers should be sorted by tier value
        tier_values = [t.value for t in seen_tiers]
        assert tier_values == sorted(tier_values), "Tiers not processed in order"

    @given(st.integers(min_value=0, max_value=500))
    @settings(max_examples=20)
    def test_early_termination_on_zero_popcount(self, vocab_size: int) -> None:
        """Evaluation terminates early when popcount reaches zero."""
        assume(vocab_size > 0)

        evaluator = TieredConstraintEvaluator()

        # First domain blocks everything
        def block_all(constraint: Any, ctx: Any) -> torch.Tensor:
            return torch.zeros(ctx.vocab_size, dtype=torch.bool, device=ctx.device)

        # Second domain should not be called
        second_called = [False]

        def second_fn(constraint: Any, ctx: Any) -> torch.Tensor:
            second_called[0] = True
            return torch.ones(ctx.vocab_size, dtype=torch.bool, device=ctx.device)

        evaluator.register("block", block_all, EvaluationTier.FAST)
        evaluator.register("second", second_fn, EvaluationTier.MEDIUM)

        ctx = MockContext(vocab_size=vocab_size)
        result = evaluator.evaluate({"block": None, "second": None}, ctx)

        # Should have early termination
        assert result.early_termination, "Should terminate early on zero popcount"
        assert "second" in result.skipped_domains, "Second domain should be skipped"


# =============================================================================
# TypeMaskCache Properties
# =============================================================================


class MockTokenizer:
    """Mock tokenizer for testing TypeMaskCache."""

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size

    def decode(self, token_ids):
        return "".join(f"tok{i}" for i in token_ids)

    def get_vocab(self):
        return {f"tok{i}": i for i in range(self.vocab_size)}


class MockClassifier:
    """Mock classifier for testing TypeMaskCache."""

    def __init__(self, vocab_size: int):
        self._vocab_size = vocab_size
        self.initialized = True

    def by_category(self, category: TokenCategory) -> List[int]:
        """Return mock token IDs for a category."""
        # Return a small set of tokens for each category
        if category == TokenCategory.INT_LITERAL:
            return list(range(0, min(10, self._vocab_size)))
        elif category == TokenCategory.FLOAT_LITERAL:
            return list(range(10, min(20, self._vocab_size)))
        elif category == TokenCategory.STRING_LITERAL:
            return list(range(20, min(30, self._vocab_size)))
        return []

    def initialize(self):
        pass


class TestTypeMaskCacheProperties:
    """Property-based tests for TypeMaskCache."""

    @given(st.integers(min_value=100, max_value=10000))
    @settings(max_examples=20)
    def test_masks_have_correct_size(self, vocab_size: int) -> None:
        """All masks have the correct vocabulary size."""
        classifier = MockClassifier(vocab_size)
        cache = TypeMaskCache(classifier, vocab_size, "cpu")
        cache.initialize()

        # Check all masks have correct size
        for mask in cache._type_masks.values():
            assert mask.shape == (vocab_size,), f"Mask has wrong shape: {mask.shape}"
            assert mask.dtype == torch.bool, f"Mask has wrong dtype: {mask.dtype}"

    @given(st.integers(min_value=100, max_value=5000))
    @settings(max_examples=20)
    def test_hit_rate_bounded(self, vocab_size: int) -> None:
        """Hit rate is always between 0 and 1."""
        classifier = MockClassifier(vocab_size)
        cache = TypeMaskCache(classifier, vocab_size, "cpu")
        cache.initialize()

        # Access some types (some hits, some misses)
        from domains.types.constraint import INT, STR, FLOAT

        cache.get_mask(INT)  # Should hit
        cache.get_mask(STR)  # Should hit
        cache.get_mask(("UnknownType",))  # Should miss

        assert 0.0 <= cache.hit_rate <= 1.0, "Hit rate must be in [0, 1]"

    @given(st.integers(min_value=100, max_value=2000))
    @settings(max_examples=20)
    def test_primitive_masks_always_cached(self, vocab_size: int) -> None:
        """Primitive types are always in the cache after initialization."""
        from domains.types.constraint import INT, STR, BOOL, FLOAT, NONE, ANY

        classifier = MockClassifier(vocab_size)
        cache = TypeMaskCache(classifier, vocab_size, "cpu")
        cache.initialize()

        primitives = [INT, STR, BOOL, FLOAT, NONE, ANY]
        for prim in primitives:
            assert cache.has_mask(prim), f"Primitive {prim} not in cache"

    @given(st.integers(min_value=100, max_value=2000))
    @settings(max_examples=15)
    def test_cached_type_count_monotonic(self, vocab_size: int) -> None:
        """Cached type count doesn't decrease during initialization."""
        classifier = MockClassifier(vocab_size)
        cache = TypeMaskCache(classifier, vocab_size, "cpu")

        # Before init
        count_before = cache.cached_type_count
        assert count_before == 0

        # After init
        cache.initialize()
        count_after = cache.cached_type_count

        assert count_after >= count_before, "Cached type count should not decrease"
        assert count_after > 0, "Should have cached some types"


# =============================================================================
# VocabPartition Properties
# =============================================================================


class TestVocabPartitionProperties:
    """Property-based tests for VocabPartition."""

    @given(st.integers(min_value=10, max_value=1000))
    @settings(max_examples=20)
    def test_classification_consistent(self, vocab_size: int) -> None:
        """Same token always gets same category."""
        partition = VocabPartition(vocab_size=vocab_size, language="python")

        # Create simple token strings
        tokens = [f"tok{i}" for i in range(vocab_size)]
        partition.classify_vocabulary(tokens)

        # Query same token multiple times
        for token_id in range(min(10, vocab_size)):
            cat1 = partition.get_category(token_id)
            cat2 = partition.get_category(token_id)
            assert cat1 == cat2, f"Inconsistent category for token {token_id}"

    @given(st.integers(min_value=10, max_value=500))
    @settings(max_examples=20)
    def test_all_token_ids_valid(self, vocab_size: int) -> None:
        """All token IDs in range have valid categories."""
        partition = VocabPartition(vocab_size=vocab_size, language="python")

        tokens = [f"tok{i}" for i in range(vocab_size)]
        partition.classify_vocabulary(tokens)

        for token_id in range(vocab_size):
            cat = partition.get_category(token_id)
            assert cat >= 0, f"Invalid category {cat} for token {token_id}"

    @given(
        st.sampled_from(["python", "typescript", "rust", "go", "kotlin", "swift", "zig"])
    )
    @settings(max_examples=14)
    def test_language_specific_partition(self, language: str) -> None:
        """Partition can be created for all supported languages."""
        vocab_size = 100
        partition = VocabPartition(vocab_size=vocab_size, language=language)

        # Should not raise
        tokens = [f"tok{i}" for i in range(vocab_size)]
        partition.classify_vocabulary(tokens)

        assert partition.vocab_size == vocab_size

    @given(st.integers(min_value=10, max_value=200))
    @settings(max_examples=20)
    def test_is_in_category_consistent(self, vocab_size: int) -> None:
        """is_in_category is consistent with get_category."""
        partition = VocabPartition(vocab_size=vocab_size, language="python")

        tokens = [f"tok{i}" for i in range(vocab_size)]
        partition.classify_vocabulary(tokens)

        for token_id in range(min(20, vocab_size)):
            cat = partition.get_category(token_id)
            # Token should be in its own category
            assert partition.is_in_category(token_id, cat), \
                f"Token {token_id} not in its category {cat}"


# =============================================================================
# Integration Properties
# =============================================================================


class TestIntegrationProperties:
    """Property-based tests for integrated Phase 1 components."""

    @given(st.integers(min_value=100, max_value=500))
    @settings(max_examples=10)
    def test_evaluator_with_real_masks(self, vocab_size: int) -> None:
        """AdaptiveTieredEvaluator works with realistic mask functions."""
        evaluator = create_adaptive_tiered_evaluator(target_popcount=50)

        # Register domains with different selectivities
        def make_mask(selectivity: float):
            def fn(c: Any, ctx: Any) -> torch.Tensor:
                mask = torch.ones(ctx.vocab_size, dtype=torch.bool, device=ctx.device)
                n_block = int(ctx.vocab_size * selectivity)
                if n_block > 0:
                    mask[:n_block] = False
                return mask
            return fn

        evaluator.register("fast", make_mask(0.1), EvaluationTier.FAST)
        evaluator.register("medium", make_mask(0.3), EvaluationTier.MEDIUM)
        evaluator.register("slow", make_mask(0.5), EvaluationTier.SLOW)

        ctx = MockContext(vocab_size=vocab_size)
        constraints = {"fast": None, "medium": None, "slow": None}

        result = evaluator.evaluate(constraints, ctx)

        # Verify result is valid
        assert result.fused_mask.shape == (vocab_size,)
        assert result.final_popcount >= 0
        assert result.final_popcount <= vocab_size
        assert len(result.evaluated_domains) > 0
