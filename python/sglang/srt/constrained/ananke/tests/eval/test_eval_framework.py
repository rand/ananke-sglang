# Copyright Rand Arete @ Ananke 2025
# Licensed under the Apache License, Version 2.0
"""Tests for the eval framework."""

from __future__ import annotations

import pytest

from .config import EvalConfig, EvalTier
from .metrics import EvalMetrics, EvalResult, SatisfactionLevel
from .judges.regex_judge import RegexJudge
from .runners.syntax_satisfaction import SyntaxSatisfactionRunner


class TestEvalConfig:
    """Tests for EvalConfig."""

    def test_default_config(self):
        """Default config should have sensible defaults."""
        config = EvalConfig()
        assert config.tier == EvalTier.SYNTAX
        assert config.sample_count == 1
        assert config.languages is None
        assert config.domains is None

    def test_tier1_syntax_factory(self):
        """tier1_syntax factory should create syntax-focused config."""
        config = EvalConfig.tier1_syntax(languages={"python", "rust"})
        assert config.tier == EvalTier.SYNTAX
        assert config.languages == {"python", "rust"}

    def test_invalid_sample_count_raises(self):
        """sample_count < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="sample_count"):
            EvalConfig(sample_count=0)

    def test_invalid_timeout_raises(self):
        """Non-positive timeout should raise ValueError."""
        with pytest.raises(ValueError, match="timeout"):
            EvalConfig(timeout_per_example=-1)


class TestEvalMetrics:
    """Tests for EvalMetrics."""

    def test_empty_metrics(self):
        """Empty metrics should have zero rates."""
        metrics = EvalMetrics()
        assert metrics.total == 0
        assert metrics.satisfaction_rate == 0.0
        assert metrics.error_rate == 0.0

    def test_add_satisfied_result(self):
        """Adding a satisfied result should update counts."""
        metrics = EvalMetrics()
        result = EvalResult(
            example_id="test-001",
            satisfied=True,
            satisfaction_level=SatisfactionLevel.FULL,
        )
        metrics.add_result(result, language="python", domain="types")

        assert metrics.total == 1
        assert metrics.satisfied == 1
        assert metrics.satisfaction_rate == 1.0
        assert "python" in metrics.by_language
        assert "types" in metrics.by_domain

    def test_add_failed_result(self):
        """Adding a failed result should update counts."""
        metrics = EvalMetrics()
        result = EvalResult(
            example_id="test-001",
            satisfied=False,
            satisfaction_level=SatisfactionLevel.NONE,
        )
        metrics.add_result(result)

        assert metrics.total == 1
        assert metrics.failed == 1
        assert metrics.satisfaction_rate == 0.0

    def test_add_error_result(self):
        """Adding an error result should update error count."""
        metrics = EvalMetrics()
        result = EvalResult(
            example_id="test-001",
            satisfied=False,
            satisfaction_level=SatisfactionLevel.ERROR,
            error="Test error",
        )
        metrics.add_result(result)

        assert metrics.total == 1
        assert metrics.errors == 1
        assert metrics.error_rate == 1.0

    def test_merge_metrics(self):
        """Merging metrics should combine counts."""
        metrics1 = EvalMetrics(total=5, satisfied=3, failed=2)
        metrics2 = EvalMetrics(total=3, satisfied=2, failed=1)

        merged = metrics1.merge(metrics2)
        assert merged.total == 8
        assert merged.satisfied == 5
        assert merged.failed == 3


class TestRegexJudge:
    """Tests for RegexJudge."""

    def test_matching_pattern(self):
        """Matching pattern should return satisfied result."""
        judge = RegexJudge()
        result = judge.evaluate(
            example_id="test-001",
            output="def foo(x: int) -> int:",
            pattern=r"def\s+\w+\s*\([^)]*\)\s*->",
        )

        assert result.satisfied
        assert result.satisfaction_level == SatisfactionLevel.FULL
        assert result.metadata["pattern_matched"]

    def test_non_matching_pattern(self):
        """Non-matching pattern should return unsatisfied result."""
        judge = RegexJudge()
        result = judge.evaluate(
            example_id="test-001",
            output="function foo() { }",
            pattern=r"def\s+\w+",
        )

        assert not result.satisfied
        assert result.satisfaction_level == SatisfactionLevel.NONE

    def test_invalid_regex_returns_error(self):
        """Invalid regex should return error result."""
        judge = RegexJudge()
        result = judge.evaluate(
            example_id="test-001",
            output="test",
            pattern=r"[invalid",  # Unclosed bracket
        )

        assert not result.satisfied
        assert result.satisfaction_level == SatisfactionLevel.ERROR
        assert result.error is not None

    def test_negative_pattern_violation(self):
        """Matching a negative pattern should fail."""
        judge = RegexJudge()
        result = judge.evaluate(
            example_id="test-001",
            output="import * from module",
            pattern=r"import",
            negative_patterns=[r"import \*"],  # Forbid wildcard imports
        )

        assert not result.satisfied
        assert result.metadata["negative_violations"] == [r"import \*"]


class TestSyntaxSatisfactionRunner:
    """Tests for SyntaxSatisfactionRunner."""

    def test_get_examples_returns_iterator(self):
        """get_examples should return an iterator of examples."""
        runner = SyntaxSatisfactionRunner()
        examples = list(runner.get_examples())
        assert len(examples) > 0

    def test_filter_by_language(self):
        """Should filter examples by language."""
        config = EvalConfig.tier1_syntax(languages={"python"})
        runner = SyntaxSatisfactionRunner(config)
        examples = list(runner.get_examples())

        assert len(examples) > 0
        assert all(e.language == "python" for e in examples)

    def test_filter_by_domain(self):
        """Should filter examples by domain."""
        config = EvalConfig.tier1_syntax(domains={"types"})
        runner = SyntaxSatisfactionRunner(config)
        examples = list(runner.get_examples())

        assert len(examples) > 0
        assert all(e.domain == "types" for e in examples)

    def test_run_validation_regex_all_pass(self):
        """All regex examples should pass Tier 1 syntax validation."""
        runner = SyntaxSatisfactionRunner()
        # Only test regex examples (EBNF has known limitations)
        examples = list(runner.get_examples(include_ebnf=False))

        passed = 0
        for ex in examples:
            result = runner.validate_example(ex)
            if result.validation_passed:
                passed += 1

        # All examples with regex should have valid_outputs matching
        assert len(examples) > 0
        assert passed == len(examples), f"Failed: {len(examples) - passed}/{len(examples)}"

    def test_validate_single_example(self):
        """validate_example should return detailed results."""
        runner = SyntaxSatisfactionRunner()
        examples = list(runner.get_examples())
        assert len(examples) > 0

        result = runner.validate_example(examples[0])
        assert result.example == examples[0]
        assert result.validation_passed
        assert "valid_passed" in result.result.metadata
        assert "invalid_rejected" in result.result.metadata


class TestTier1EvalIntegration:
    """Integration tests for Tier 1 eval."""

    def test_full_tier1_eval_regex_only(self):
        """Full Tier 1 eval with regex should complete successfully."""
        config = EvalConfig.tier1_syntax()
        runner = SyntaxSatisfactionRunner(config)
        # Only test regex examples (EBNF has known limitations - see EVAL_HANDOFF.md)
        examples = list(runner.get_examples(include_ebnf=False))

        passed = 0
        for ex in examples:
            result = runner.validate_example(ex)
            if result.validation_passed:
                passed += 1

        # Check we evaluated all regex-having examples
        assert len(examples) == 92  # Known count from corpus

        # All regex should pass syntax validation
        assert passed == len(examples)

    def test_full_tier1_eval_with_ebnf(self):
        """Full Tier 1 eval including EBNF runs without errors."""
        config = EvalConfig.tier1_syntax()
        runner = SyntaxSatisfactionRunner(config)
        metrics = runner.run_validation()

        # Check we evaluated all examples including EBNF
        assert metrics.total == 136  # 92 regex + 44 EBNF

        # Check there are no errors (parse failures are OK, not errors)
        assert metrics.errors == 0

        # Check language breakdown exists
        assert "python" in metrics.by_language
        assert "rust" in metrics.by_language
        assert "typescript" in metrics.by_language
        assert "go" in metrics.by_language
        assert "zig" in metrics.by_language
        assert "kotlin" in metrics.by_language
        assert "swift" in metrics.by_language

    def test_per_language_regex_eval(self):
        """Each language's regex examples should pass independently."""
        for lang in ["python", "rust", "typescript", "go", "zig", "kotlin", "swift"]:
            config = EvalConfig.tier1_syntax(languages={lang})
            runner = SyntaxSatisfactionRunner(config)
            examples = list(runner.get_examples(include_ebnf=False))

            if len(examples) == 0:
                continue  # Some languages may have only EBNF examples

            passed = 0
            for ex in examples:
                result = runner.validate_example(ex)
                if result.validation_passed:
                    passed += 1

            assert passed == len(examples), f"Failures for {lang}: {len(examples) - passed}/{len(examples)}"
