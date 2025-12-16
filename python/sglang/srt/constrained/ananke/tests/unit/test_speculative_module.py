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
"""Tests for speculative decoding module.

Tests the DraftModel protocol and ConstrainedLookahead implementations
for Phase 3.1 CDSL-style speculative decoding.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple
import pytest
import torch

try:
    from ...speculative import (
        DraftContext,
        DraftResult,
        DraftModel,
        NullDraftModel,
        GreedyDraftModel,
        SamplingDraftModel,
        CachedDraftModel,
        create_draft_model,
        LookaheadConfig,
        VerificationResult,
        LookaheadStats,
        ConstraintVerifier,
        ConstrainedLookahead,
        ParallelConstrainedLookahead,
        create_constrained_lookahead,
    )
except ImportError:
    from speculative import (
        DraftContext,
        DraftResult,
        DraftModel,
        NullDraftModel,
        GreedyDraftModel,
        SamplingDraftModel,
        CachedDraftModel,
        create_draft_model,
        LookaheadConfig,
        VerificationResult,
        LookaheadStats,
        ConstraintVerifier,
        ConstrainedLookahead,
        ParallelConstrainedLookahead,
        create_constrained_lookahead,
    )


# =============================================================================
# DraftContext Tests
# =============================================================================


class TestDraftContext:
    """Tests for DraftContext dataclass."""

    def test_create_empty(self) -> None:
        """Test creating empty context."""
        ctx = DraftContext()
        assert ctx.prefix_tokens == []
        assert ctx.prefix_text == ""
        assert ctx.language == "python"
        assert ctx.temperature == 1.0

    def test_create_with_values(self) -> None:
        """Test creating context with values."""
        ctx = DraftContext(
            prefix_tokens=[1, 2, 3],
            prefix_text="def add(",
            prompt="Write an add function",
            temperature=0.7,
            language="python",
            expected_type="int",
        )
        assert ctx.prefix_tokens == [1, 2, 3]
        assert ctx.expected_type == "int"


# =============================================================================
# DraftResult Tests
# =============================================================================


class TestDraftResult:
    """Tests for DraftResult dataclass."""

    def test_create_basic(self) -> None:
        """Test creating basic result."""
        result = DraftResult(tokens=[1, 2, 3])
        assert result.tokens == [1, 2, 3]
        assert result.length == 3
        assert result.log_probs is None

    def test_create_with_log_probs(self) -> None:
        """Test creating result with log probabilities."""
        result = DraftResult(
            tokens=[1, 2, 3],
            log_probs=[-0.1, -0.2, -0.3],
            latency_ms=1.5,
        )
        assert result.log_probs == [-0.1, -0.2, -0.3]
        assert result.latency_ms == 1.5

    def test_empty_result(self) -> None:
        """Test empty result."""
        result = DraftResult(tokens=[])
        assert result.length == 0


# =============================================================================
# NullDraftModel Tests
# =============================================================================


class TestNullDraftModel:
    """Tests for NullDraftModel."""

    def test_returns_empty(self) -> None:
        """Test that null model returns empty drafts."""
        model = NullDraftModel()
        ctx = DraftContext()
        result = model.generate_draft(ctx, lookahead_length=5)

        assert result.tokens == []
        assert result.length == 0

    def test_is_draft_model(self) -> None:
        """Test that NullDraftModel satisfies protocol."""
        model = NullDraftModel()
        assert isinstance(model, DraftModel)


# =============================================================================
# GreedyDraftModel Tests
# =============================================================================


class TestGreedyDraftModel:
    """Tests for GreedyDraftModel."""

    def test_generates_tokens(self) -> None:
        """Test that greedy model generates tokens."""
        # Mock logits function that always returns highest prob for token 42
        def mock_logits(prefix: List[int]) -> torch.Tensor:
            logits = torch.zeros(100)
            logits[42] = 10.0  # High probability for token 42
            return logits

        model = GreedyDraftModel(logits_fn=mock_logits, vocab_size=100)
        ctx = DraftContext()
        result = model.generate_draft(ctx, lookahead_length=3)

        assert result.length == 3
        assert all(t == 42 for t in result.tokens)

    def test_respects_syntax_mask(self) -> None:
        """Test that greedy model respects syntax mask."""
        def mock_logits(prefix: List[int]) -> torch.Tensor:
            logits = torch.zeros(100)
            logits[42] = 10.0  # Highest prob
            logits[10] = 5.0  # Second highest
            return logits

        def mock_mask(prefix: List[int]) -> torch.Tensor:
            mask = torch.zeros(100, dtype=torch.bool)
            mask[10] = True  # Only token 10 allowed
            return mask

        model = GreedyDraftModel(
            logits_fn=mock_logits,
            syntax_mask_fn=mock_mask,
            vocab_size=100,
        )
        ctx = DraftContext()
        result = model.generate_draft(ctx, lookahead_length=3)

        # Should select token 10 (only allowed token)
        assert all(t == 10 for t in result.tokens)

    def test_is_draft_model(self) -> None:
        """Test that GreedyDraftModel satisfies protocol."""
        model = GreedyDraftModel(
            logits_fn=lambda x: torch.zeros(100),
            vocab_size=100,
        )
        assert isinstance(model, DraftModel)


# =============================================================================
# SamplingDraftModel Tests
# =============================================================================


class TestSamplingDraftModel:
    """Tests for SamplingDraftModel."""

    def test_generates_tokens(self) -> None:
        """Test that sampling model generates tokens."""
        def mock_logits(prefix: List[int]) -> torch.Tensor:
            logits = torch.zeros(100)
            logits[42] = 10.0
            return logits

        model = SamplingDraftModel(
            logits_fn=mock_logits,
            vocab_size=100,
            temperature=0.1,  # Low temp = nearly deterministic
        )
        ctx = DraftContext()
        result = model.generate_draft(ctx, lookahead_length=3)

        assert result.length == 3
        # With very low temperature, should select token 42
        assert all(t == 42 for t in result.tokens)

    def test_is_draft_model(self) -> None:
        """Test that SamplingDraftModel satisfies protocol."""
        model = SamplingDraftModel(
            logits_fn=lambda x: torch.zeros(100),
            vocab_size=100,
        )
        assert isinstance(model, DraftModel)


# =============================================================================
# CachedDraftModel Tests
# =============================================================================


class TestCachedDraftModel:
    """Tests for CachedDraftModel."""

    def test_caches_results(self) -> None:
        """Test that results are cached."""
        call_count = [0]

        class CountingModel:
            def generate_draft(
                self,
                context: DraftContext,
                lookahead_length: int = 5,
            ) -> DraftResult:
                call_count[0] += 1
                return DraftResult(tokens=[1, 2, 3])

        model = CachedDraftModel(fallback=CountingModel(), max_cache_size=10)  # type: ignore
        ctx = DraftContext(prefix_tokens=[1, 2])

        # First call - should hit fallback
        result1 = model.generate_draft(ctx, lookahead_length=3)
        assert call_count[0] == 1

        # Second call with same context - should hit cache
        result2 = model.generate_draft(ctx, lookahead_length=3)
        assert call_count[0] == 1  # No additional call

        assert result1.tokens == result2.tokens

    def test_clear_cache(self) -> None:
        """Test clearing cache."""
        call_count = [0]

        class CountingModel:
            def generate_draft(
                self,
                context: DraftContext,
                lookahead_length: int = 5,
            ) -> DraftResult:
                call_count[0] += 1
                return DraftResult(tokens=[1, 2, 3])

        model = CachedDraftModel(fallback=CountingModel(), max_cache_size=10)  # type: ignore
        ctx = DraftContext(prefix_tokens=[1, 2])

        model.generate_draft(ctx, lookahead_length=3)
        assert call_count[0] == 1

        model.clear_cache()

        model.generate_draft(ctx, lookahead_length=3)
        assert call_count[0] == 2  # Cache was cleared


# =============================================================================
# create_draft_model Tests
# =============================================================================


class TestCreateDraftModel:
    """Tests for create_draft_model factory."""

    def test_create_null(self) -> None:
        """Test creating null model."""
        model = create_draft_model("null")
        assert isinstance(model, NullDraftModel)

    def test_create_greedy(self) -> None:
        """Test creating greedy model."""
        model = create_draft_model(
            "greedy",
            logits_fn=lambda x: torch.zeros(100),
            vocab_size=100,
        )
        assert isinstance(model, GreedyDraftModel)

    def test_create_sampling(self) -> None:
        """Test creating sampling model."""
        model = create_draft_model(
            "sampling",
            logits_fn=lambda x: torch.zeros(100),
            vocab_size=100,
        )
        assert isinstance(model, SamplingDraftModel)

    def test_unknown_defaults_to_null(self) -> None:
        """Test that unknown type defaults to null."""
        model = create_draft_model("unknown_type")
        assert isinstance(model, NullDraftModel)


# =============================================================================
# LookaheadConfig Tests
# =============================================================================


class TestLookaheadConfig:
    """Tests for LookaheadConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = LookaheadConfig()
        assert config.initial_lookahead == 5
        assert config.min_lookahead == 2
        assert config.max_lookahead == 15
        assert config.adaptive is True

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = LookaheadConfig(
            initial_lookahead=10,
            min_lookahead=5,
            max_lookahead=20,
            adaptive=False,
        )
        assert config.initial_lookahead == 10
        assert config.adaptive is False


# =============================================================================
# VerificationResult Tests
# =============================================================================


class TestVerificationResult:
    """Tests for VerificationResult."""

    def test_acceptance_rate(self) -> None:
        """Test acceptance rate calculation."""
        result = VerificationResult(num_valid=3, total_draft=5)
        assert result.acceptance_rate == 0.6

    def test_acceptance_rate_empty(self) -> None:
        """Test acceptance rate with empty draft."""
        result = VerificationResult(num_valid=0, total_draft=0)
        assert result.acceptance_rate == 0.0

    def test_fully_accepted(self) -> None:
        """Test fully_accepted property."""
        full = VerificationResult(num_valid=5, total_draft=5)
        assert full.fully_accepted

        partial = VerificationResult(num_valid=3, total_draft=5)
        assert not partial.fully_accepted


# =============================================================================
# LookaheadStats Tests
# =============================================================================


class TestLookaheadStats:
    """Tests for LookaheadStats."""

    def test_overall_acceptance_rate(self) -> None:
        """Test overall acceptance rate calculation."""
        stats = LookaheadStats(
            total_tokens_drafted=100,
            total_tokens_accepted=80,
        )
        assert stats.overall_acceptance_rate == 0.8

    def test_speedup_estimate(self) -> None:
        """Test speedup estimation."""
        stats = LookaheadStats(
            total_tokens_accepted=100,
            total_draft_latency_ms=100.0,
            total_verify_latency_ms=100.0,
        )
        # Expected speedup = (100 * 20) / (100 + 100) = 10x
        assert stats.speedup_estimate == 10.0


# =============================================================================
# Mock Verifier for Testing
# =============================================================================


class MockVerifier:
    """Mock verifier for testing."""

    def __init__(self, accept_count: int = 3):
        """Initialize mock verifier.

        Args:
            accept_count: Number of tokens to accept
        """
        self.accept_count = accept_count
        self.verify_calls = 0

    def verify_draft_tokens(
        self,
        draft_tokens: List[int],
    ) -> Tuple[int, Optional[Any]]:
        """Mock verification."""
        self.verify_calls += 1
        valid = min(len(draft_tokens), self.accept_count)
        return valid, None


# =============================================================================
# ConstrainedLookahead Tests
# =============================================================================


class TestConstrainedLookahead:
    """Tests for ConstrainedLookahead."""

    def test_generate_next_basic(self) -> None:
        """Test basic token generation."""
        # Create draft model that generates [1, 2, 3, 4, 5]
        class FixedDraftModel:
            def generate_draft(
                self,
                context: DraftContext,
                lookahead_length: int = 5,
            ) -> DraftResult:
                return DraftResult(tokens=list(range(1, lookahead_length + 1)))

        verifier = MockVerifier(accept_count=3)
        lookahead = ConstrainedLookahead(
            draft_model=FixedDraftModel(),  # type: ignore
            verifier=verifier,  # type: ignore
        )

        ctx = DraftContext()
        tokens = lookahead.generate_next(ctx)

        # Should accept 3 tokens (verifier accepts 3)
        assert tokens == [1, 2, 3]
        assert verifier.verify_calls == 1

    def test_generate_next_empty_draft(self) -> None:
        """Test with empty draft."""
        verifier = MockVerifier()
        lookahead = ConstrainedLookahead(
            draft_model=NullDraftModel(),
            verifier=verifier,  # type: ignore
        )

        ctx = DraftContext()
        tokens = lookahead.generate_next(ctx)

        assert tokens == []
        assert verifier.verify_calls == 0

    def test_stats_updated(self) -> None:
        """Test that statistics are updated."""
        class FixedDraftModel:
            def generate_draft(
                self,
                context: DraftContext,
                lookahead_length: int = 5,
            ) -> DraftResult:
                return DraftResult(tokens=[1, 2, 3, 4, 5])

        verifier = MockVerifier(accept_count=3)
        lookahead = ConstrainedLookahead(
            draft_model=FixedDraftModel(),  # type: ignore
            verifier=verifier,  # type: ignore
        )

        ctx = DraftContext()
        lookahead.generate_next(ctx)

        stats = lookahead.get_stats()
        assert stats.total_drafts == 1
        assert stats.total_tokens_drafted == 5
        assert stats.total_tokens_accepted == 3

    def test_adaptive_lookahead_increase(self) -> None:
        """Test that lookahead increases with high acceptance."""
        class FixedDraftModel:
            def generate_draft(
                self,
                context: DraftContext,
                lookahead_length: int = 5,
            ) -> DraftResult:
                return DraftResult(tokens=list(range(lookahead_length)))

        # Verifier that accepts all tokens
        verifier = MockVerifier(accept_count=100)
        config = LookaheadConfig(
            initial_lookahead=5,
            acceptance_threshold_high=0.8,
            lookahead_step=2,
            adaptive=True,
        )
        lookahead = ConstrainedLookahead(
            draft_model=FixedDraftModel(),  # type: ignore
            verifier=verifier,  # type: ignore
            config=config,
        )

        ctx = DraftContext()

        # Generate multiple times to build up acceptance history
        for _ in range(5):
            lookahead.generate_next(ctx)

        # Lookahead should have increased
        assert lookahead.stats.current_lookahead > 5

    def test_reset_stats(self) -> None:
        """Test resetting statistics."""
        class FixedDraftModel:
            def generate_draft(
                self,
                context: DraftContext,
                lookahead_length: int = 5,
            ) -> DraftResult:
                return DraftResult(tokens=[1, 2, 3])

        verifier = MockVerifier()
        lookahead = ConstrainedLookahead(
            draft_model=FixedDraftModel(),  # type: ignore
            verifier=verifier,  # type: ignore
        )

        ctx = DraftContext()
        lookahead.generate_next(ctx)
        assert lookahead.stats.total_drafts == 1

        lookahead.reset_stats()
        assert lookahead.stats.total_drafts == 0


# =============================================================================
# ParallelConstrainedLookahead Tests
# =============================================================================


class TestParallelConstrainedLookahead:
    """Tests for ParallelConstrainedLookahead."""

    def test_generate_next_selects_best(self) -> None:
        """Test that parallel lookahead selects best sequence."""
        # Draft model that generates different sequences
        call_count = [0]

        class VaryingDraftModel:
            def generate_draft(
                self,
                context: DraftContext,
                lookahead_length: int = 5,
            ) -> DraftResult:
                call_count[0] += 1
                # Each call generates longer sequence
                return DraftResult(tokens=list(range(call_count[0] + 2)))

        verifier = MockVerifier(accept_count=100)  # Accept all
        parallel = ParallelConstrainedLookahead(
            draft_model=VaryingDraftModel(),  # type: ignore
            verifier=verifier,  # type: ignore
            num_sequences=3,
        )

        ctx = DraftContext()
        tokens = parallel.generate_next(ctx)

        # Should have generated 3 sequences and selected longest
        assert call_count[0] == 3
        # Third call generates [0, 1, 2, 3, 4] (5 tokens)
        assert len(tokens) == 5


# =============================================================================
# create_constrained_lookahead Tests
# =============================================================================


class TestCreateConstrainedLookahead:
    """Tests for create_constrained_lookahead factory."""

    def test_create_basic(self) -> None:
        """Test creating basic lookahead."""
        verifier = MockVerifier()
        lookahead = create_constrained_lookahead(
            draft_model=NullDraftModel(),
            verifier=verifier,  # type: ignore
        )
        assert isinstance(lookahead, ConstrainedLookahead)

    def test_create_parallel(self) -> None:
        """Test creating parallel lookahead."""
        verifier = MockVerifier()
        lookahead = create_constrained_lookahead(
            draft_model=NullDraftModel(),
            verifier=verifier,  # type: ignore
            parallel=True,
            num_sequences=3,
        )
        assert isinstance(lookahead, ParallelConstrainedLookahead)

    def test_create_with_config(self) -> None:
        """Test creating with configuration."""
        verifier = MockVerifier()
        lookahead = create_constrained_lookahead(
            draft_model=NullDraftModel(),
            verifier=verifier,  # type: ignore
            initial_lookahead=10,
            adaptive=False,
        )
        assert lookahead.config.initial_lookahead == 10
        assert lookahead.config.adaptive is False


# =============================================================================
# Integration Tests
# =============================================================================


class TestSpeculativeIntegration:
    """Integration tests for speculative decoding."""

    def test_end_to_end_flow(self) -> None:
        """Test complete speculative decoding flow."""
        # Create a draft model
        def mock_logits(prefix: List[int]) -> torch.Tensor:
            logits = torch.randn(100)
            logits[prefix[-1] + 1 if prefix else 0] = 10.0
            return logits

        draft_model = GreedyDraftModel(logits_fn=mock_logits, vocab_size=100)

        # Create verifier that accepts first 3 tokens
        verifier = MockVerifier(accept_count=3)

        # Create lookahead
        lookahead = ConstrainedLookahead(
            draft_model=draft_model,
            verifier=verifier,  # type: ignore
        )

        # Generate tokens
        ctx = DraftContext(prefix_tokens=[0])
        tokens = lookahead.generate_next(ctx)

        # Should have generated and accepted tokens
        assert len(tokens) == 3

        # Stats should be updated
        stats = lookahead.get_stats()
        assert stats.total_drafts > 0
        assert stats.total_tokens_accepted > 0

    def test_protocol_compliance(self) -> None:
        """Test that all draft models satisfy the protocol."""
        def mock_logits(x: List[int]) -> torch.Tensor:
            return torch.randn(100)

        models = [
            NullDraftModel(),
            GreedyDraftModel(mock_logits, vocab_size=100),
            SamplingDraftModel(mock_logits, vocab_size=100),
            CachedDraftModel(NullDraftModel()),  # type: ignore
        ]

        ctx = DraftContext()
        for model in models:
            assert isinstance(model, DraftModel)
            result = model.generate_draft(ctx, lookahead_length=5)
            assert isinstance(result, DraftResult)
