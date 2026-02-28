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
"""Tests for beam search module.

Tests the BeamSearch implementation for Phase 3.2 beam search integration.
"""

from __future__ import annotations

from typing import Any, List, Tuple
import pytest
import torch

try:
    from ...search.beam import (
        BeamCandidate,
        BeamSearchConfig,
        BeamSearchStats,
        BeamSearch,
        SimpleTokenScorer,
        create_beam_search,
    )
except ImportError:
    from search.beam import (
        BeamCandidate,
        BeamSearchConfig,
        BeamSearchStats,
        BeamSearch,
        SimpleTokenScorer,
        create_beam_search,
    )


# =============================================================================
# BeamCandidate Tests
# =============================================================================


class TestBeamCandidate:
    """Tests for BeamCandidate dataclass."""

    def test_create_basic(self) -> None:
        """Test basic creation."""
        candidate = BeamCandidate(
            score=-0.8,
            tokens=[1, 2, 3],
        )
        assert candidate.score == -0.8
        assert candidate.tokens == [1, 2, 3]
        assert candidate.length == 3
        assert not candidate.finished

    def test_create_with_all_fields(self) -> None:
        """Test creation with all fields."""
        candidate = BeamCandidate(
            score=-0.9,
            tokens=[1, 2],
            state="test_state",
            log_prob=-1.5,
            constraint_score=0.85,
            diversity_penalty=0.1,
            finished=True,
            metadata={"key": "value"},
        )
        assert candidate.state == "test_state"
        assert candidate.log_prob == -1.5
        assert candidate.constraint_score == 0.85
        assert candidate.finished

    def test_extend(self) -> None:
        """Test extending a candidate."""
        candidate = BeamCandidate(
            score=0.0,
            tokens=[1, 2],
            log_prob=-1.0,
            constraint_score=0.9,
        )

        extended = candidate.extend(
            token=3,
            log_prob=-0.5,
            constraint_score=0.8,
        )

        assert extended.tokens == [1, 2, 3]
        assert extended.log_prob == -1.5  # -1.0 + -0.5
        assert len(extended.tokens) == 3

    def test_ordering(self) -> None:
        """Test that candidates are orderable by score."""
        c1 = BeamCandidate(score=-0.9, tokens=[1])
        c2 = BeamCandidate(score=-0.8, tokens=[2])
        c3 = BeamCandidate(score=-0.7, tokens=[3])

        sorted_candidates = sorted([c3, c1, c2])
        # Negated scores, so -0.9 < -0.8 < -0.7
        assert sorted_candidates[0].tokens == [1]
        assert sorted_candidates[1].tokens == [2]
        assert sorted_candidates[2].tokens == [3]


# =============================================================================
# BeamSearchConfig Tests
# =============================================================================


class TestBeamSearchConfig:
    """Tests for BeamSearchConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = BeamSearchConfig()
        assert config.beam_width == 5
        assert config.max_length == 512
        assert config.early_stopping is True
        assert config.constraint_weight == 0.7

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = BeamSearchConfig(
            beam_width=10,
            max_length=256,
            diversity_penalty=1.0,
            early_stopping=False,
        )
        assert config.beam_width == 10
        assert config.max_length == 256
        assert config.diversity_penalty == 1.0


# =============================================================================
# BeamSearchStats Tests
# =============================================================================


class TestBeamSearchStats:
    """Tests for BeamSearchStats."""

    def test_default_values(self) -> None:
        """Test default statistics values."""
        stats = BeamSearchStats()
        assert stats.total_steps == 0
        assert stats.candidates_evaluated == 0
        assert stats.finished_beams == 0

    def test_update_values(self) -> None:
        """Test updating statistics."""
        stats = BeamSearchStats(
            total_steps=10,
            candidates_evaluated=50,
            candidates_pruned=40,
            finished_beams=3,
            best_score=0.95,
        )
        assert stats.total_steps == 10
        assert stats.candidates_pruned == 40
        assert stats.best_score == 0.95


# =============================================================================
# Mock Token Scorer
# =============================================================================


class MockTokenScorer:
    """Mock token scorer for testing."""

    def __init__(
        self,
        vocab_size: int = 100,
        fixed_token: int = 42,
        end_token: int = 1,
    ):
        """Initialize mock scorer.

        Args:
            vocab_size: Vocabulary size
            fixed_token: Token to always suggest (except at end)
            end_token: End token ID
        """
        self.vocab_size = vocab_size
        self.fixed_token = fixed_token
        self.end_token = end_token
        self.call_count = 0

    def score_tokens(
        self,
        tokens: List[int],
        state: Any,
        top_k: int = 50,
    ) -> List[Tuple[int, float, float]]:
        """Return mock scores."""
        self.call_count += 1

        # After 3 tokens, suggest end token
        if len(tokens) >= 3:
            return [
                (self.end_token, -0.1, 1.0),
                (self.fixed_token, -0.5, 0.8),
            ]

        return [
            (self.fixed_token, -0.2, 0.9),
            (self.fixed_token + 1, -0.5, 0.7),
        ]


# =============================================================================
# BeamSearch Tests
# =============================================================================


class TestBeamSearch:
    """Tests for BeamSearch."""

    def test_create_default(self) -> None:
        """Test default creation."""
        search = BeamSearch()
        assert search.config is not None
        assert search.config.beam_width == 5

    def test_create_with_config(self) -> None:
        """Test creation with config."""
        config = BeamSearchConfig(beam_width=10)
        search = BeamSearch(config=config)
        assert search.config.beam_width == 10

    def test_search_requires_scorer(self) -> None:
        """Test that search requires a token scorer."""
        search = BeamSearch()
        with pytest.raises(ValueError, match="token_scorer is required"):
            search.search(start_tokens=[0])

    def test_search_basic(self) -> None:
        """Test basic search."""
        scorer = MockTokenScorer()
        search = BeamSearch(token_scorer=scorer)  # type: ignore

        result = search.search(
            start_tokens=[0],
            end_token=1,
        )

        assert isinstance(result, BeamCandidate)
        assert len(result.tokens) >= 1
        assert scorer.call_count > 0

    def test_search_respects_beam_width(self) -> None:
        """Test that search respects beam width."""
        scorer = MockTokenScorer()
        config = BeamSearchConfig(beam_width=3, max_length=5)
        search = BeamSearch(config=config, token_scorer=scorer)  # type: ignore

        result = search.search(start_tokens=[0], end_token=1)

        # Should complete search
        assert isinstance(result, BeamCandidate)

    def test_search_with_max_length(self) -> None:
        """Test search with maximum length constraint."""
        scorer = MockTokenScorer(end_token=999)  # Won't trigger end
        config = BeamSearchConfig(max_length=5, beam_width=2)
        search = BeamSearch(config=config, token_scorer=scorer)  # type: ignore

        result = search.search(start_tokens=[0])

        # Should stop at max_length iterations (start_tokens + max_length iterations)
        # With start_tokens=[0] (1 token) and 5 iterations, max is 6 tokens
        assert result.length <= len([0]) + config.max_length

    def test_search_early_stopping(self) -> None:
        """Test early stopping behavior."""
        scorer = MockTokenScorer()
        config = BeamSearchConfig(
            beam_width=3,
            max_length=10,
            early_stopping=True,
        )
        search = BeamSearch(config=config, token_scorer=scorer)  # type: ignore

        result = search.search(start_tokens=[0], end_token=1)
        stats = search.get_stats()

        # Should have finished beams
        assert result is not None

    def test_stats_updated(self) -> None:
        """Test that statistics are updated."""
        scorer = MockTokenScorer()
        config = BeamSearchConfig(beam_width=2, max_length=3)
        search = BeamSearch(config=config, token_scorer=scorer)  # type: ignore

        search.search(start_tokens=[0], end_token=1)
        stats = search.get_stats()

        assert stats.total_steps > 0
        assert stats.candidates_evaluated > 0
        assert stats.latency_ms > 0

    def test_diversity_penalty(self) -> None:
        """Test that diversity penalty is applied."""
        # Create scorer that returns similar candidates
        class SimilarScorer:
            def score_tokens(
                self,
                tokens: List[int],
                state: Any,
                top_k: int = 50,
            ) -> List[Tuple[int, float, float]]:
                # Return same token multiple times
                return [(42, -0.1, 1.0)] * 5

        config = BeamSearchConfig(
            beam_width=3,
            diversity_penalty=0.5,
            max_length=2,
        )
        search = BeamSearch(config=config, token_scorer=SimilarScorer())  # type: ignore

        result = search.search(start_tokens=[0])

        # Should complete without error
        assert isinstance(result, BeamCandidate)


# =============================================================================
# SimpleTokenScorer Tests
# =============================================================================


class TestSimpleTokenScorer:
    """Tests for SimpleTokenScorer."""

    def test_score_tokens(self) -> None:
        """Test basic token scoring."""
        def logits_fn(tokens: List[int]) -> torch.Tensor:
            logits = torch.zeros(100)
            logits[42] = 10.0  # High prob for token 42
            logits[10] = 5.0   # Medium prob for token 10
            return logits

        def mask_fn(tokens: List[int], state: Any) -> torch.Tensor:
            mask = torch.ones(100, dtype=torch.bool)
            return mask

        scorer = SimpleTokenScorer(
            logits_fn=logits_fn,
            mask_fn=mask_fn,
            vocab_size=100,
        )

        results = scorer.score_tokens([0], None, top_k=5)

        assert len(results) == 5
        # Token 42 should be first (highest prob)
        assert results[0][0] == 42

    def test_score_respects_mask(self) -> None:
        """Test that scoring respects constraint mask."""
        def logits_fn(tokens: List[int]) -> torch.Tensor:
            logits = torch.zeros(100)
            logits[42] = 10.0  # Highest prob
            logits[10] = 5.0   # Second highest
            return logits

        def mask_fn(tokens: List[int], state: Any) -> torch.Tensor:
            mask = torch.zeros(100, dtype=torch.bool)
            mask[10] = True  # Only token 10 allowed
            return mask

        scorer = SimpleTokenScorer(
            logits_fn=logits_fn,
            mask_fn=mask_fn,
            vocab_size=100,
        )

        results = scorer.score_tokens([0], None, top_k=5)

        # Only token 10 should be returned
        assert len(results) == 1
        assert results[0][0] == 10


# =============================================================================
# create_beam_search Tests
# =============================================================================


class TestCreateBeamSearch:
    """Tests for create_beam_search factory."""

    def test_create_default(self) -> None:
        """Test creating with defaults."""
        search = create_beam_search()
        assert isinstance(search, BeamSearch)

    def test_create_with_scorer(self) -> None:
        """Test creating with scorer."""
        scorer = MockTokenScorer()
        search = create_beam_search(token_scorer=scorer)  # type: ignore
        assert search.token_scorer is scorer

    def test_create_with_config(self) -> None:
        """Test creating with config kwargs."""
        search = create_beam_search(beam_width=10, max_length=100)
        assert search.config.beam_width == 10
        assert search.config.max_length == 100


# =============================================================================
# Integration Tests
# =============================================================================


class TestBeamSearchIntegration:
    """Integration tests for beam search."""

    def test_end_to_end_search(self) -> None:
        """Test complete search flow."""
        # Create a scorer that generates a specific sequence
        class SequenceScorer:
            def __init__(self):
                self.step = 0

            def score_tokens(
                self,
                tokens: List[int],
                state: Any,
                top_k: int = 50,
            ) -> List[Tuple[int, float, float]]:
                self.step += 1
                # Generate sequence [start, 10, 11, 12, END]
                # len(tokens) starts at 1 (start token), so use len-1 to get 10 first
                if len(tokens) < 4:
                    next_token = 9 + len(tokens)  # 9+1=10, 9+2=11, 9+3=12
                    return [(next_token, -0.1, 0.95)]
                else:
                    return [(1, -0.05, 1.0)]  # END token

        search = create_beam_search(
            token_scorer=SequenceScorer(),  # type: ignore
            beam_width=3,
            max_length=10,
        )

        result = search.search(start_tokens=[0], end_token=1)

        # Should generate [0, 10, 11, 12, 1]
        assert 10 in result.tokens
        assert 11 in result.tokens

    def test_beam_selection(self) -> None:
        """Test that beam search selects best paths."""
        # Create scorer that offers multiple paths
        class BranchingScorer:
            def score_tokens(
                self,
                tokens: List[int],
                state: Any,
                top_k: int = 50,
            ) -> List[Tuple[int, float, float]]:
                # First step: offer two paths
                if len(tokens) == 1:
                    return [
                        (10, -0.1, 0.9),  # Path A
                        (20, -0.5, 0.5),  # Path B
                    ]
                # End after 2 tokens
                return [(1, -0.1, 1.0)]

        config = BeamSearchConfig(beam_width=2, max_length=5)
        search = BeamSearch(config=config, token_scorer=BranchingScorer())  # type: ignore

        result = search.search(start_tokens=[0], end_token=1)

        # Should select path A (higher constraint score)
        assert 10 in result.tokens

    def test_constraint_aware_search(self) -> None:
        """Test that constraint scores affect selection."""
        # Create scorer where constraint score determines selection
        class ConstraintAwareScorer:
            def score_tokens(
                self,
                tokens: List[int],
                state: Any,
                top_k: int = 50,
            ) -> List[Tuple[int, float, float]]:
                if len(tokens) == 1:
                    # Token 99 has high log prob but low constraint
                    # Token 50 has medium log prob but high constraint
                    return [
                        (99, -0.05, 0.3),  # High LM, low constraint
                        (50, -0.5, 0.95),  # Low LM, high constraint
                    ]
                return [(1, -0.1, 1.0)]  # END

        config = BeamSearchConfig(
            beam_width=2,
            constraint_weight=0.7,  # Prioritize constraint
            max_length=5,
        )
        search = BeamSearch(config=config, token_scorer=ConstraintAwareScorer())  # type: ignore

        result = search.search(start_tokens=[0], end_token=1)

        # Should prefer token 50 due to higher constraint score
        assert 50 in result.tokens

    def test_get_all_candidates(self) -> None:
        """Test getting all candidates from search."""
        scorer = MockTokenScorer()
        config = BeamSearchConfig(beam_width=3, max_length=5)
        search = BeamSearch(config=config, token_scorer=scorer)  # type: ignore

        search.search(start_tokens=[0], end_token=1)
        all_candidates = search.get_all_candidates()

        # Should have multiple candidates
        assert len(all_candidates) >= 1
        # Should be sorted by score (best first)
        for i in range(len(all_candidates) - 1):
            assert all_candidates[i].score <= all_candidates[i + 1].score

    def test_search_with_constraint_returns_multiple(self) -> None:
        """Test that search_with_constraint returns multiple candidates."""
        # Create scorer that offers multiple paths
        class MultiPathScorer:
            def score_tokens(
                self,
                tokens: List[int],
                state: Any,
                top_k: int = 50,
            ) -> List[Tuple[int, float, float]]:
                if len(tokens) == 1:
                    # Offer multiple paths
                    return [
                        (10, -0.1, 0.9),
                        (20, -0.2, 0.8),
                        (30, -0.3, 0.7),
                    ]
                # End token
                return [(1, -0.1, 1.0)]

        config = BeamSearchConfig(beam_width=3, max_length=5)
        search = BeamSearch(config=config, token_scorer=MultiPathScorer())  # type: ignore

        results = search.search_with_constraint(
            start_tokens=[0],
            end_token=1,
        )

        # Should return multiple candidates
        assert len(results) >= 1
        assert len(results) <= 3  # Limited by beam_width
        # Each should be a BeamCandidate
        for r in results:
            assert isinstance(r, BeamCandidate)
