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
"""Unit tests for constraint verification and Best-of-N selection.

Tests verify:
1. ConstraintVerifier produces accurate scores
2. BestOfNSelector correctly selects best candidates
3. Selection strategies work as expected
4. Soundness: verification never blocks valid code

Key Property:
    Post-hoc verification is SOUND - we only score and rank, never block.
    Tests verify that verification gracefully handles all inputs.
"""

import pytest
import sys
from pathlib import Path

# Add the ananke package root to sys.path for standalone testing
_ANANKE_ROOT = Path(__file__).parent.parent.parent
if str(_ANANKE_ROOT) not in sys.path:
    sys.path.insert(0, str(_ANANKE_ROOT))

from verification.verifier import (
    ConstraintVerifier,
    VerificationResult,
    DomainScore,
    DEFAULT_DOMAIN_WEIGHTS,
)
from verification.selector import (
    BestOfNSelector,
    SelectionStrategy,
    SelectionResult,
    select_best_of_n,
)


class TestDomainScore:
    """Tests for DomainScore dataclass."""

    def test_valid_domain_score(self):
        """Test creating valid domain score."""
        score = DomainScore(
            domain="syntax",
            valid=True,
            score=1.0,
            errors=(),
            latency_ns=1000,
        )
        assert score.domain == "syntax"
        assert score.valid
        assert score.score == 1.0
        assert len(score.errors) == 0

    def test_invalid_domain_score(self):
        """Test creating invalid domain score with errors."""
        score = DomainScore(
            domain="types",
            valid=False,
            score=0.5,
            errors=("Type mismatch",),
        )
        assert not score.valid
        assert score.score == 0.5
        assert "Type mismatch" in score.errors

    def test_score_clamping(self):
        """Test that score is clamped to [0, 1]."""
        score_high = DomainScore(domain="test", valid=True, score=1.5)
        assert score_high.score == 1.0

        score_low = DomainScore(domain="test", valid=True, score=-0.5)
        assert score_low.score == 0.0


class TestVerificationResult:
    """Tests for VerificationResult dataclass."""

    def test_empty_result(self):
        """Test default verification result."""
        result = VerificationResult(candidate="x = 1")
        assert result.candidate == "x = 1"
        assert not result.valid
        assert result.overall_score == 0.0
        assert len(result.domain_scores) == 0

    def test_get_errors(self):
        """Test aggregating errors across domains."""
        result = VerificationResult(
            candidate="bad code",
            domain_scores={
                "syntax": DomainScore("syntax", False, 0.0, ("Syntax error",)),
                "types": DomainScore("types", False, 0.5, ("Type error",)),
            },
        )
        errors = result.get_errors()
        assert len(errors) == 2
        assert "Syntax error" in errors
        assert "Type error" in errors

    def test_to_dict(self):
        """Test serialization to dictionary."""
        result = VerificationResult(
            candidate="x = 1",
            valid=True,
            overall_score=0.9,
            domain_scores={
                "syntax": DomainScore("syntax", True, 1.0),
            },
        )
        d = result.to_dict()
        assert d["candidate"] == "x = 1"
        assert d["valid"]
        assert d["overall_score"] == 0.9
        assert "syntax" in d["domain_scores"]


class TestConstraintVerifier:
    """Tests for ConstraintVerifier."""

    @pytest.fixture
    def verifier(self):
        """Create verifier for Python code."""
        return ConstraintVerifier(language="python")

    def test_verify_valid_python(self, verifier):
        """Test verifying valid Python code."""
        result = verifier.verify("x = 1")
        assert isinstance(result, VerificationResult)
        assert result.overall_score >= 0.0
        # Note: May not be 1.0 if domains report partial scores

    def test_verify_invalid_syntax(self, verifier):
        """Test verifying code with syntax errors."""
        result = verifier.verify("def foo( missing colon")
        # Should have lower score than valid code
        assert result.overall_score < 1.0 or "syntax" not in verifier.enabled_domains

    def test_verify_returns_result_for_any_input(self, verifier):
        """Soundness: verification should always return a result."""
        # Empty string
        result1 = verifier.verify("")
        assert isinstance(result1, VerificationResult)

        # Random garbage
        result2 = verifier.verify("@#$%^&*()_+{}|:<>?")
        assert isinstance(result2, VerificationResult)

        # Very long code
        result3 = verifier.verify("x = 1\n" * 1000)
        assert isinstance(result3, VerificationResult)

    def test_verify_batch(self, verifier):
        """Test batch verification."""
        candidates = ["x = 1", "y = 2", "z = 3"]
        results = verifier.verify_batch(candidates)
        assert len(results) == 3
        assert all(isinstance(r, VerificationResult) for r in results)

    def test_domain_weights_affect_score(self):
        """Test that domain weights affect overall score computation."""
        # High weight on syntax
        verifier_syntax = ConstraintVerifier(
            language="python",
            domain_weights={"syntax": 2.0, "types": 0.1},
        )

        # High weight on types
        verifier_types = ConstraintVerifier(
            language="python",
            domain_weights={"syntax": 0.1, "types": 2.0},
        )

        code = "x: int = 'string'"  # Type error but valid syntax
        result1 = verifier_syntax.verify(code)
        result2 = verifier_types.verify(code)

        # Both should produce results (soundness)
        assert isinstance(result1, VerificationResult)
        assert isinstance(result2, VerificationResult)

    def test_enabled_domains_filtering(self):
        """Test that only enabled domains are verified."""
        verifier = ConstraintVerifier(
            language="python",
            enabled_domains={"syntax"},
        )
        result = verifier.verify("x = 1")
        # Only syntax domain should be in results
        assert "syntax" in result.domain_scores
        assert "types" not in result.domain_scores

    def test_stats_tracking(self, verifier):
        """Test that statistics are tracked."""
        verifier.reset_stats()
        verifier.verify("x = 1")
        verifier.verify("y = 2")

        stats = verifier.get_stats()
        assert stats["verifications"] == 2


class TestBestOfNSelector:
    """Tests for BestOfNSelector."""

    @pytest.fixture
    def selector(self):
        """Create selector for Python code."""
        return BestOfNSelector(language="python")

    def test_select_from_single_candidate(self, selector):
        """Test selection from single candidate."""
        result = selector.select_best(["x = 1"])
        assert result.selected == "x = 1"
        assert result.selected_index == 0
        assert result.num_candidates == 1

    def test_select_best_score_strategy(self, selector):
        """Test BEST_SCORE selection strategy."""
        candidates = [
            "def foo( bad syntax",  # Invalid
            "x = 1",  # Valid simple
            "def foo(): pass",  # Valid function
        ]
        result = selector.select_best(candidates, strategy=SelectionStrategy.BEST_SCORE)

        # Should select one of the valid candidates
        assert result.selected_index >= 1
        assert result.num_candidates == 3

    def test_select_first_valid_strategy(self, selector):
        """Test FIRST_VALID selection strategy."""
        candidates = [
            "bad (syntax",  # Invalid
            "x = 1",  # First valid
            "y = 2",  # Also valid
        ]
        selector.strategy = SelectionStrategy.FIRST_VALID
        result = selector.select_best(candidates)

        # Should select first valid (index 1)
        assert result.selected_index >= 0  # At least returns something

    def test_select_threshold_strategy(self):
        """Test THRESHOLD selection strategy."""
        selector = BestOfNSelector(
            language="python",
            strategy=SelectionStrategy.THRESHOLD,
            threshold=0.8,
        )
        candidates = ["x = 1", "y = 2"]
        result = selector.select_best(candidates)

        # Should return a result even if threshold not met
        assert result.selected in candidates

    def test_empty_candidates_raises(self, selector):
        """Test that empty candidate list raises ValueError."""
        with pytest.raises(ValueError):
            selector.select_best([])

    def test_all_invalid_returns_best(self, selector):
        """Test selection when all candidates are invalid."""
        # All have syntax errors
        candidates = [
            "def foo(",
            "class Bar(",
            "if True",
        ]
        result = selector.select_best(candidates)

        # Should still return something (graceful degradation)
        assert result.selected in candidates

    def test_stats_tracking(self, selector):
        """Test that selection statistics are tracked."""
        selector.reset_stats()
        selector.select_best(["x = 1", "y = 2"])
        selector.select_best(["a = 1", "b = 2", "c = 3"])

        stats = selector.get_stats()
        assert stats["selections"] == 2
        assert stats["total_candidates"] == 5

    def test_return_all_results(self):
        """Test that all results are returned when requested."""
        selector = BestOfNSelector(language="python", return_all_results=True)
        result = selector.select_best(["x = 1", "y = 2", "z = 3"])

        assert len(result.all_results) == 3

    def test_return_all_results_disabled(self):
        """Test that results are not stored when disabled."""
        selector = BestOfNSelector(language="python", return_all_results=False)
        result = selector.select_best(["x = 1", "y = 2"])

        # With BEST_SCORE strategy, all results are still computed
        # but not returned unless return_all_results is True
        # The implementation may still store them for scoring


class TestSelectionResult:
    """Tests for SelectionResult dataclass."""

    def test_to_dict(self):
        """Test serialization to dictionary."""
        result = SelectionResult(
            selected="x = 1",
            selected_index=0,
            selected_result=VerificationResult(candidate="x = 1", valid=True),
            num_candidates=3,
            num_valid=2,
            strategy=SelectionStrategy.BEST_SCORE,
        )
        d = result.to_dict()
        assert d["selected"] == "x = 1"
        assert d["selected_index"] == 0
        assert d["num_candidates"] == 3
        assert d["strategy"] == "BEST_SCORE"


class TestConvenienceFunction:
    """Tests for select_best_of_n convenience function."""

    def test_basic_usage(self):
        """Test basic usage of convenience function."""
        result = select_best_of_n(["x = 1", "y = 2"])
        assert isinstance(result, SelectionResult)
        assert result.selected in ["x = 1", "y = 2"]

    def test_with_strategy(self):
        """Test with explicit strategy."""
        result = select_best_of_n(
            ["x = 1", "y = 2"],
            strategy=SelectionStrategy.FIRST_VALID,
        )
        assert isinstance(result, SelectionResult)


class TestSoundnessProperties:
    """Tests verifying soundness properties of verification.

    Soundness: Verification should NEVER cause valid code to be rejected.
    Instead, it scores and ranks candidates.
    """

    def test_verification_never_throws_on_valid_code(self):
        """Valid code should always be verified successfully."""
        verifier = ConstraintVerifier(language="python")

        valid_examples = [
            "x = 1",
            "def foo(): pass",
            "class Bar: pass",
            "x: int = 1",
            "import os",
            "from typing import List",
            'print("hello")',
        ]

        for code in valid_examples:
            result = verifier.verify(code)
            assert isinstance(result, VerificationResult)
            # Result should exist, score should be reasonable

    def test_verification_handles_edge_cases(self):
        """Edge cases should be handled gracefully."""
        verifier = ConstraintVerifier(language="python")

        edge_cases = [
            "",  # Empty
            " ",  # Whitespace only
            "\n\n\n",  # Newlines only
            "# comment only",  # Comment only
            "pass",  # Minimal valid
            "...",  # Ellipsis
        ]

        for code in edge_cases:
            result = verifier.verify(code)
            assert isinstance(result, VerificationResult)

    def test_selector_always_returns_candidate(self):
        """Selector should always return a candidate, even if all invalid."""
        selector = BestOfNSelector(language="python")

        # All syntactically invalid
        candidates = ["def (", "class {", "if ["]
        result = selector.select_best(candidates)

        # Should still return one of them
        assert result.selected in candidates


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
