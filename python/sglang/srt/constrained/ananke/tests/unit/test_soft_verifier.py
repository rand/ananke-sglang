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
"""Tests for soft verification (PRM integration).

Tests the SoftVerifier protocol and implementations for Phase 2.2.
"""

from __future__ import annotations

import pytest

try:
    from ...verification.soft_verifier import (
        SoftScore,
        GenerationContext,
        SoftVerifier,
        NullSoftVerifier,
        EnsembleSoftVerifier,
        HeuristicSoftVerifier,
        CombinedVerificationResult,
        create_soft_verifier,
    )
except ImportError:
    from verification.soft_verifier import (
        SoftScore,
        GenerationContext,
        SoftVerifier,
        NullSoftVerifier,
        EnsembleSoftVerifier,
        HeuristicSoftVerifier,
        CombinedVerificationResult,
        create_soft_verifier,
    )


# =============================================================================
# SoftScore Tests
# =============================================================================


class TestSoftScore:
    """Tests for SoftScore dataclass."""

    def test_create_basic(self) -> None:
        """Test basic creation."""
        score = SoftScore(score=0.8)
        assert score.score == 0.8
        assert score.confidence == 1.0
        assert score.model_name == "unknown"

    def test_score_clamping(self) -> None:
        """Test that scores are clamped to [0, 1]."""
        high = SoftScore(score=1.5)
        assert high.score == 1.0

        low = SoftScore(score=-0.5)
        assert low.score == 0.0

    def test_confidence_clamping(self) -> None:
        """Test that confidence is clamped to [0, 1]."""
        high = SoftScore(score=0.5, confidence=1.5)
        assert high.confidence == 1.0

        low = SoftScore(score=0.5, confidence=-0.5)
        assert low.confidence == 0.0

    def test_with_details(self) -> None:
        """Test score with details breakdown."""
        score = SoftScore(
            score=0.75,
            confidence=0.9,
            details={"syntax": 0.8, "types": 0.7},
            model_name="test-model",
        )
        assert score.details["syntax"] == 0.8
        assert score.model_name == "test-model"


# =============================================================================
# GenerationContext Tests
# =============================================================================


class TestGenerationContext:
    """Tests for GenerationContext."""

    def test_create_empty(self) -> None:
        """Test creating empty context."""
        ctx = GenerationContext()
        assert ctx.prompt == ""
        assert ctx.language == "python"

    def test_create_with_prompt(self) -> None:
        """Test creating with prompt."""
        ctx = GenerationContext(
            prompt="Write a function to add two numbers",
            expected_signature="def add(a: int, b: int) -> int:",
        )
        assert "add two numbers" in ctx.prompt
        assert ctx.expected_signature == "def add(a: int, b: int) -> int:"

    def test_with_test_cases(self) -> None:
        """Test context with test cases."""
        ctx = GenerationContext(
            prompt="Write add function",
            test_cases=["assert add(1, 2) == 3", "assert add(-1, 1) == 0"],
        )
        assert len(ctx.test_cases) == 2


# =============================================================================
# NullSoftVerifier Tests
# =============================================================================


class TestNullSoftVerifier:
    """Tests for NullSoftVerifier."""

    def test_returns_neutral_score(self) -> None:
        """Test that null verifier returns neutral score."""
        verifier = NullSoftVerifier()
        ctx = GenerationContext()
        score = verifier.score("def f(): pass", ctx)

        assert score.score == 0.5
        assert score.confidence == 0.1
        assert score.model_name == "null"

    def test_is_soft_verifier(self) -> None:
        """Test that NullSoftVerifier satisfies SoftVerifier protocol."""
        verifier = NullSoftVerifier()
        assert isinstance(verifier, SoftVerifier)


# =============================================================================
# HeuristicSoftVerifier Tests
# =============================================================================


class TestHeuristicSoftVerifier:
    """Tests for HeuristicSoftVerifier."""

    def test_scores_non_empty_code(self) -> None:
        """Test that non-empty code gets positive score."""
        verifier = HeuristicSoftVerifier(language="python")
        ctx = GenerationContext()
        score = verifier.score("def f(): pass", ctx)

        assert score.score > 0.0
        assert "non_empty" in score.details
        assert score.model_name == "heuristic"

    def test_scores_empty_code_low(self) -> None:
        """Test that empty code gets low score."""
        verifier = HeuristicSoftVerifier(language="python")
        ctx = GenerationContext()
        score = verifier.score("", ctx)

        assert score.score < 0.5
        assert score.details.get("non_empty", 1.0) == 0.0

    def test_rewards_docstring(self) -> None:
        """Test that code with docstring gets bonus."""
        verifier = HeuristicSoftVerifier(language="python")
        ctx = GenerationContext()

        with_doc = verifier.score('def f():\n    """A function."""\n    pass', ctx)
        without_doc = verifier.score("def f(): pass", ctx)

        assert with_doc.details["docstring"] > without_doc.details["docstring"]

    def test_rewards_type_annotations(self) -> None:
        """Test that code with type annotations gets bonus."""
        verifier = HeuristicSoftVerifier(language="python")
        ctx = GenerationContext()

        with_types = verifier.score("def f(x: int) -> int: return x", ctx)
        without_types = verifier.score("def f(x): return x", ctx)

        assert with_types.details["types"] > without_types.details["types"]

    def test_matches_expected_signature(self) -> None:
        """Test that matching signature gets bonus."""
        verifier = HeuristicSoftVerifier(language="python")
        ctx = GenerationContext(expected_signature="def add(a, b):")

        matches = verifier.score("def add(a, b): return a + b", ctx)
        no_match = verifier.score("def subtract(a, b): return a - b", ctx)

        assert matches.details["signature_match"] > no_match.details["signature_match"]

    def test_is_soft_verifier(self) -> None:
        """Test that HeuristicSoftVerifier satisfies SoftVerifier protocol."""
        verifier = HeuristicSoftVerifier()
        assert isinstance(verifier, SoftVerifier)


# =============================================================================
# EnsembleSoftVerifier Tests
# =============================================================================


class TestEnsembleSoftVerifier:
    """Tests for EnsembleSoftVerifier."""

    def test_empty_ensemble_returns_null(self) -> None:
        """Test that empty ensemble returns null score."""
        ensemble = EnsembleSoftVerifier()
        ctx = GenerationContext()
        score = ensemble.score("def f(): pass", ctx)

        assert score.score == 0.5
        assert score.model_name == "null"

    def test_single_verifier_ensemble(self) -> None:
        """Test ensemble with single verifier."""
        ensemble = EnsembleSoftVerifier()
        ensemble.add_verifier(HeuristicSoftVerifier(), weight=1.0)

        ctx = GenerationContext()
        score = ensemble.score("def f(): pass", ctx)

        assert score.score > 0.0
        assert "ensemble" in score.model_name
        assert "heuristic" in score.details

    def test_weighted_aggregation(self) -> None:
        """Test weighted aggregation of multiple verifiers."""
        # Create mock verifiers with known scores
        class HighScorer:
            def score(self, code: str, ctx: GenerationContext) -> SoftScore:
                return SoftScore(score=0.9, confidence=1.0, model_name="high")

        class LowScorer:
            def score(self, code: str, ctx: GenerationContext) -> SoftScore:
                return SoftScore(score=0.1, confidence=1.0, model_name="low")

        ensemble = EnsembleSoftVerifier(aggregation="weighted")
        ensemble.add_verifier(HighScorer(), weight=2.0)  # type: ignore
        ensemble.add_verifier(LowScorer(), weight=1.0)  # type: ignore

        ctx = GenerationContext()
        score = ensemble.score("test", ctx)

        # Weighted average: (0.9*2 + 0.1*1) / 3 = 0.633...
        assert 0.6 < score.score < 0.7

    def test_max_aggregation(self) -> None:
        """Test max aggregation."""
        class HighScorer:
            def score(self, code: str, ctx: GenerationContext) -> SoftScore:
                return SoftScore(score=0.9, model_name="high")

        class LowScorer:
            def score(self, code: str, ctx: GenerationContext) -> SoftScore:
                return SoftScore(score=0.1, model_name="low")

        ensemble = EnsembleSoftVerifier(aggregation="max")
        ensemble.add_verifier(HighScorer(), weight=1.0)  # type: ignore
        ensemble.add_verifier(LowScorer(), weight=1.0)  # type: ignore

        ctx = GenerationContext()
        score = ensemble.score("test", ctx)

        assert score.score == 0.9

    def test_mean_aggregation(self) -> None:
        """Test mean aggregation."""
        class HighScorer:
            def score(self, code: str, ctx: GenerationContext) -> SoftScore:
                return SoftScore(score=0.8, model_name="high")

        class LowScorer:
            def score(self, code: str, ctx: GenerationContext) -> SoftScore:
                return SoftScore(score=0.4, model_name="low")

        ensemble = EnsembleSoftVerifier(aggregation="mean")
        ensemble.add_verifier(HighScorer(), weight=1.0)  # type: ignore
        ensemble.add_verifier(LowScorer(), weight=1.0)  # type: ignore

        ctx = GenerationContext()
        score = ensemble.score("test", ctx)

        assert score.score == pytest.approx(0.6)

    def test_handles_failing_verifier(self) -> None:
        """Test that ensemble handles failing verifiers gracefully."""
        class FailingVerifier:
            def score(self, code: str, ctx: GenerationContext) -> SoftScore:
                raise RuntimeError("Verifier failed")

        ensemble = EnsembleSoftVerifier()
        ensemble.add_verifier(HeuristicSoftVerifier(), weight=1.0)
        ensemble.add_verifier(FailingVerifier(), weight=1.0)  # type: ignore

        ctx = GenerationContext()
        # Should not raise, should return score from working verifier
        score = ensemble.score("def f(): pass", ctx)
        assert score.score > 0.0


# =============================================================================
# CombinedVerificationResult Tests
# =============================================================================


class TestCombinedVerificationResult:
    """Tests for CombinedVerificationResult."""

    def test_combine_equal_weights(self) -> None:
        """Test combining with equal weights."""
        soft = SoftScore(score=0.8, confidence=1.0)
        result = CombinedVerificationResult.combine(
            rule_score=0.6,
            soft_score=soft,
            rule_weight=0.5,
            soft_weight=0.5,
        )

        # (0.6 * 0.5 + 0.8 * 0.5) / 1.0 = 0.7
        assert result.combined_score == pytest.approx(0.7)

    def test_combine_rule_dominated(self) -> None:
        """Test combining with rule-based dominated."""
        soft = SoftScore(score=0.8, confidence=1.0)
        result = CombinedVerificationResult.combine(
            rule_score=0.6,
            soft_score=soft,
            rule_weight=0.9,
            soft_weight=0.1,
        )

        # Combined should be closer to 0.6 than 0.8
        assert result.combined_score < 0.65

    def test_combine_low_confidence_reduces_soft_weight(self) -> None:
        """Test that low confidence reduces soft score weight."""
        high_conf = SoftScore(score=0.8, confidence=1.0)
        low_conf = SoftScore(score=0.8, confidence=0.1)

        result_high = CombinedVerificationResult.combine(
            rule_score=0.4,
            soft_score=high_conf,
            rule_weight=0.7,
            soft_weight=0.3,
        )
        result_low = CombinedVerificationResult.combine(
            rule_score=0.4,
            soft_score=low_conf,
            rule_weight=0.7,
            soft_weight=0.3,
        )

        # Low confidence should give less weight to soft score
        # So result_low.combined_score should be closer to 0.4 (rule score)
        assert result_low.combined_score < result_high.combined_score


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestCreateSoftVerifier:
    """Tests for create_soft_verifier factory."""

    def test_create_null(self) -> None:
        """Test creating null verifier."""
        verifier = create_soft_verifier("null")
        assert isinstance(verifier, NullSoftVerifier)

    def test_create_heuristic(self) -> None:
        """Test creating heuristic verifier."""
        verifier = create_soft_verifier("heuristic")
        assert isinstance(verifier, HeuristicSoftVerifier)

    def test_create_ensemble(self) -> None:
        """Test creating ensemble verifier."""
        verifier = create_soft_verifier("ensemble")
        assert isinstance(verifier, EnsembleSoftVerifier)

    def test_unknown_type_returns_null(self) -> None:
        """Test that unknown type returns null verifier."""
        verifier = create_soft_verifier("unknown_type")
        assert isinstance(verifier, NullSoftVerifier)

    def test_heuristic_respects_language(self) -> None:
        """Test that heuristic verifier respects language parameter."""
        verifier = create_soft_verifier("heuristic", language="typescript")
        assert isinstance(verifier, HeuristicSoftVerifier)
        assert verifier.language == "typescript"


# =============================================================================
# Integration Tests
# =============================================================================


class TestSoftVerifierIntegration:
    """Integration tests for soft verifier with Best-of-N selection."""

    def test_soft_verifier_protocol_compliance(self) -> None:
        """Test that all verifiers satisfy the SoftVerifier protocol."""
        verifiers = [
            NullSoftVerifier(),
            HeuristicSoftVerifier(),
            EnsembleSoftVerifier(),
        ]
        ctx = GenerationContext()

        for v in verifiers:
            assert isinstance(v, SoftVerifier)
            score = v.score("def f(): pass", ctx)
            assert isinstance(score, SoftScore)
            assert 0.0 <= score.score <= 1.0
            assert 0.0 <= score.confidence <= 1.0

    def test_end_to_end_scoring(self) -> None:
        """Test end-to-end scoring of code candidates."""
        verifier = create_soft_verifier("heuristic")
        ctx = GenerationContext(
            prompt="Write a function to add two numbers",
            expected_signature="def add(",
        )

        candidates = [
            "def add(a, b): return a + b",
            'def add(a: int, b: int) -> int:\n    """Add two numbers."""\n    return a + b',
            "",
            "not even code",
        ]

        scores = [verifier.score(c, ctx) for c in candidates]

        # Best candidate should be the one with types and docstring
        best_idx = max(range(len(scores)), key=lambda i: scores[i].score)
        assert best_idx == 1

        # Empty string should score lowest
        worst_idx = min(range(len(scores)), key=lambda i: scores[i].score)
        assert worst_idx == 2
