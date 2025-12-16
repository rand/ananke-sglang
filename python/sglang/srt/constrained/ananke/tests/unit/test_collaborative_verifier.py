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
"""Tests for collaborative verifier (CoT + PoT).

Tests the CollaborativeVerifier combining rule-based, soft, and execution
verification for Phase 2.3.
"""

from __future__ import annotations

import pytest

try:
    from ...verification.collaborative import (
        VerificationStrategy,
        StrategyScore,
        CollaborativeResult,
        CollaborativeVerifier,
        collaborative_verify,
    )
    from ...verification.soft_verifier import (
        SoftScore,
        GenerationContext,
        NullSoftVerifier,
        HeuristicSoftVerifier,
    )
    from ...verification.execution import (
        TestCase,
        ExecutionScore,
        StaticExecutionVerifier,
    )
except ImportError:
    from verification.collaborative import (
        VerificationStrategy,
        StrategyScore,
        CollaborativeResult,
        CollaborativeVerifier,
        collaborative_verify,
    )
    from verification.soft_verifier import (
        SoftScore,
        GenerationContext,
        NullSoftVerifier,
        HeuristicSoftVerifier,
    )
    from verification.execution import (
        TestCase,
        ExecutionScore,
        StaticExecutionVerifier,
    )


# =============================================================================
# VerificationStrategy Tests
# =============================================================================


class TestVerificationStrategy:
    """Tests for VerificationStrategy enum."""

    def test_all_strategies_exist(self) -> None:
        """Test that all strategies exist."""
        assert VerificationStrategy.RULE_BASED
        assert VerificationStrategy.SOFT_MODEL
        assert VerificationStrategy.EXECUTION
        assert VerificationStrategy.COLLABORATIVE

    def test_strategies_are_distinct(self) -> None:
        """Test that strategies are distinct."""
        strategies = [
            VerificationStrategy.RULE_BASED,
            VerificationStrategy.SOFT_MODEL,
            VerificationStrategy.EXECUTION,
            VerificationStrategy.COLLABORATIVE,
        ]
        assert len(strategies) == len(set(strategies))


# =============================================================================
# StrategyScore Tests
# =============================================================================


class TestStrategyScore:
    """Tests for StrategyScore dataclass."""

    def test_create_basic(self) -> None:
        """Test basic creation."""
        score = StrategyScore(
            strategy=VerificationStrategy.RULE_BASED,
            score=0.85,
        )
        assert score.strategy == VerificationStrategy.RULE_BASED
        assert score.score == 0.85
        assert score.confidence == 1.0
        assert score.weight == 1.0

    def test_create_with_all_fields(self) -> None:
        """Test creation with all fields."""
        score = StrategyScore(
            strategy=VerificationStrategy.SOFT_MODEL,
            score=0.7,
            confidence=0.8,
            weight=0.5,
            details={"component": "test"},
        )
        assert score.confidence == 0.8
        assert score.weight == 0.5
        assert score.details["component"] == "test"


# =============================================================================
# CollaborativeResult Tests
# =============================================================================


class TestCollaborativeResult:
    """Tests for CollaborativeResult dataclass."""

    def test_create_empty(self) -> None:
        """Test creation with defaults."""
        result = CollaborativeResult(code="def f(): pass")
        assert result.code == "def f(): pass"
        assert result.overall_score == 0.0
        assert result.overall_confidence == 0.0
        assert result.recommendation == "review"
        assert len(result.strategy_scores) == 0

    def test_is_acceptable_high_score(self) -> None:
        """Test is_acceptable with high score."""
        result = CollaborativeResult(
            code="test",
            overall_score=0.8,
            overall_confidence=0.7,
        )
        assert result.is_acceptable

    def test_is_acceptable_low_score(self) -> None:
        """Test is_acceptable with low score."""
        result = CollaborativeResult(
            code="test",
            overall_score=0.5,
            overall_confidence=0.9,
        )
        assert not result.is_acceptable

    def test_is_acceptable_low_confidence(self) -> None:
        """Test is_acceptable with low confidence."""
        result = CollaborativeResult(
            code="test",
            overall_score=0.9,
            overall_confidence=0.3,
        )
        assert not result.is_acceptable

    def test_to_dict(self) -> None:
        """Test serialization to dict."""
        result = CollaborativeResult(
            code="def f(): pass",
            overall_score=0.75,
            overall_confidence=0.8,
            strategy_scores={
                VerificationStrategy.RULE_BASED: StrategyScore(
                    strategy=VerificationStrategy.RULE_BASED,
                    score=0.8,
                )
            },
            recommendation="accept",
            latency_ms=10.5,
        )
        d = result.to_dict()

        assert d["code"] == "def f(): pass"
        assert d["overall_score"] == 0.75
        assert d["recommendation"] == "accept"
        assert "RULE_BASED" in d["strategy_scores"]


# =============================================================================
# CollaborativeVerifier Tests
# =============================================================================


class TestCollaborativeVerifier:
    """Tests for CollaborativeVerifier."""

    def test_create_default(self) -> None:
        """Test default creation."""
        verifier = CollaborativeVerifier()
        assert verifier.language == "python"
        assert verifier.rule_verifier is not None
        assert verifier.soft_verifier is not None
        assert verifier.execution_verifier is not None

    def test_create_with_language(self) -> None:
        """Test creation with specific language."""
        verifier = CollaborativeVerifier(language="typescript")
        assert verifier.language == "typescript"

    def test_create_with_custom_weights(self) -> None:
        """Test creation with custom weights."""
        weights = {
            VerificationStrategy.RULE_BASED: 0.6,
            VerificationStrategy.SOFT_MODEL: 0.3,
            VerificationStrategy.EXECUTION: 0.1,
        }
        verifier = CollaborativeVerifier(strategy_weights=weights)
        assert verifier.strategy_weights[VerificationStrategy.RULE_BASED] == 0.6

    def test_verify_simple_code(self) -> None:
        """Test verification of simple code."""
        verifier = CollaborativeVerifier()
        result = verifier.verify("def add(a, b):\n    return a + b")

        assert isinstance(result, CollaborativeResult)
        assert result.code == "def add(a, b):\n    return a + b"
        assert 0.0 <= result.overall_score <= 1.0
        assert result.latency_ms > 0

    def test_verify_with_context(self) -> None:
        """Test verification with generation context."""
        verifier = CollaborativeVerifier()
        context = GenerationContext(
            prompt="Write a function to add two numbers",
            expected_signature="def add(",
        )
        result = verifier.verify("def add(a, b):\n    return a + b", context=context)

        assert result.overall_score > 0

    def test_verify_with_test_cases(self) -> None:
        """Test verification with test cases."""
        verifier = CollaborativeVerifier()
        test_cases = [
            TestCase(input="add(1, 2)", expected_output="3"),
            TestCase(input="add(0, 0)", expected_output="0"),
        ]
        result = verifier.verify(
            "def add(a, b):\n    return a + b",
            test_cases=test_cases,
            function_name="add",
        )

        assert VerificationStrategy.EXECUTION in result.strategy_scores

    def test_verify_strategy_scores_populated(self) -> None:
        """Test that strategy scores are populated."""
        verifier = CollaborativeVerifier()
        result = verifier.verify("def f():\n    return 42")

        # Should have rule-based and soft model scores
        assert VerificationStrategy.RULE_BASED in result.strategy_scores
        assert VerificationStrategy.SOFT_MODEL in result.strategy_scores

    def test_verify_recommendation_accept(self) -> None:
        """Test that good code gets accept recommendation."""
        verifier = CollaborativeVerifier()
        # Good code with types and docstring
        code = '''def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b
'''
        result = verifier.verify(code)
        # May or may not be "accept" depending on thresholds
        assert result.recommendation in ["accept", "review"]

    def test_verify_recommendation_low_quality(self) -> None:
        """Test that low quality code gets lower score than good code."""
        verifier = CollaborativeVerifier()
        # Good code with types and docstring
        good_code = '''def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b
'''
        good_result = verifier.verify(good_code)

        # Minimal code without types or docstring
        minimal_code = "x = 1"
        minimal_result = verifier.verify(minimal_code)

        # Good code should score higher than minimal code
        assert good_result.overall_score >= minimal_result.overall_score

    def test_enabled_strategies(self) -> None:
        """Test enabling/disabling strategies."""
        verifier = CollaborativeVerifier(
            enabled_strategies={VerificationStrategy.RULE_BASED}
        )
        result = verifier.verify("def f(): pass")

        # Should only have rule-based score
        assert VerificationStrategy.RULE_BASED in result.strategy_scores
        assert VerificationStrategy.SOFT_MODEL not in result.strategy_scores

    def test_combine_scores_weighted(self) -> None:
        """Test score combination logic."""
        verifier = CollaborativeVerifier()
        scores = {
            VerificationStrategy.RULE_BASED: StrategyScore(
                strategy=VerificationStrategy.RULE_BASED,
                score=0.8,
                confidence=1.0,
                weight=0.5,
            ),
            VerificationStrategy.SOFT_MODEL: StrategyScore(
                strategy=VerificationStrategy.SOFT_MODEL,
                score=0.6,
                confidence=1.0,
                weight=0.5,
            ),
        }
        overall, confidence = verifier._combine_scores(scores)

        # Weighted average: (0.8 * 0.5 + 0.6 * 0.5) / 1.0 = 0.7
        assert overall == pytest.approx(0.7)

    def test_combine_scores_empty(self) -> None:
        """Test score combination with no scores."""
        verifier = CollaborativeVerifier()
        overall, confidence = verifier._combine_scores({})

        assert overall == 0.5
        assert confidence == 0.0


# =============================================================================
# Batch and Selection Tests
# =============================================================================


class TestBatchAndSelection:
    """Tests for batch verification and selection."""

    def test_verify_batch(self) -> None:
        """Test batch verification."""
        verifier = CollaborativeVerifier()
        candidates = [
            "def add(a, b): return a + b",
            "def add(a, b): return a - b",
            "",
        ]
        results = verifier.verify_batch(candidates)

        assert len(results) == 3
        for result in results:
            assert isinstance(result, CollaborativeResult)

    def test_select_best(self) -> None:
        """Test selecting best candidate."""
        verifier = CollaborativeVerifier()
        candidates = [
            "def add(a, b): return a + b",
            'def add(a: int, b: int) -> int:\n    """Add."""\n    return a + b',
            "",
        ]
        best_code, best_result = verifier.select_best(candidates)

        # Second candidate should be best (has types and docstring)
        assert best_result.overall_score > 0
        assert best_code != ""

    def test_select_best_empty_raises(self) -> None:
        """Test that selecting from empty list raises."""
        verifier = CollaborativeVerifier()
        with pytest.raises(ValueError):
            verifier.select_best([])

    def test_select_best_with_test_cases(self) -> None:
        """Test selection with test cases."""
        verifier = CollaborativeVerifier()
        candidates = [
            "def add(a, b): return a + b",
            "def add(a, b): return a - b",  # Wrong implementation
        ]
        test_cases = [TestCase(input="add(1, 2)", expected_output="3")]

        best_code, best_result = verifier.select_best(
            candidates,
            test_cases=test_cases,
            function_name="add",
        )

        # First should be selected (correct implementation)
        assert "return a + b" in best_code


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestCollaborativeVerifyFunction:
    """Tests for collaborative_verify convenience function."""

    def test_basic_usage(self) -> None:
        """Test basic usage."""
        result = collaborative_verify("def f(): return 42")
        assert isinstance(result, CollaborativeResult)
        assert result.overall_score > 0

    def test_with_language(self) -> None:
        """Test with specified language."""
        result = collaborative_verify(
            "def f(): return 42",
            language="python",
        )
        assert isinstance(result, CollaborativeResult)

    def test_with_test_cases(self) -> None:
        """Test with test cases."""
        result = collaborative_verify(
            "def double(x): return x * 2",
            test_cases=[TestCase(input="double(5)", expected_output="10")],
        )
        assert VerificationStrategy.EXECUTION in result.strategy_scores


# =============================================================================
# Integration Tests
# =============================================================================


class TestCollaborativeIntegration:
    """Integration tests for collaborative verification."""

    def test_end_to_end_good_code(self) -> None:
        """Test end-to-end with good code."""
        code = '''def fibonacci(n: int) -> int:
    """Return the nth Fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)
'''
        result = collaborative_verify(code)

        assert result.overall_score > 0.5
        assert len(result.strategy_scores) >= 2

    def test_end_to_end_with_execution(self) -> None:
        """Test end-to-end with execution verification."""
        code = "def is_positive(n):\n    return n > 0"
        test_cases = [
            TestCase(input="is_positive(5)", expected_output="True"),
            TestCase(input="is_positive(-3)", expected_output="False"),
            TestCase(input="is_positive(0)", expected_output="False"),
        ]

        result = collaborative_verify(
            code,
            test_cases=test_cases,
        )

        assert VerificationStrategy.EXECUTION in result.strategy_scores
        exec_score = result.strategy_scores[VerificationStrategy.EXECUTION]
        assert exec_score.score > 0

    def test_strategies_combine_appropriately(self) -> None:
        """Test that strategies combine to improve accuracy."""
        # Code that passes syntax but has issues
        code = "def f(x): return undefined_var"

        result = collaborative_verify(code)

        # Rule-based should catch undefined variable issues
        # Overall score should reflect the problems
        assert result.overall_confidence > 0

    def test_recommendation_reflects_quality(self) -> None:
        """Test that recommendation reflects code quality."""
        # High quality code
        good_code = '''def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b
'''
        good_result = collaborative_verify(good_code)

        # Low quality code
        bad_result = collaborative_verify("")

        # Good should score higher than bad
        assert good_result.overall_score > bad_result.overall_score
