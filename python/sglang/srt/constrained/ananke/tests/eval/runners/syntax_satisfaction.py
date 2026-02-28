# Copyright Rand Arete @ Ananke 2025
# Licensed under the Apache License, Version 2.0
"""Tier 1 eval: Syntax constraint satisfaction.

This runner evaluates whether Ananke-constrained generation produces
syntactically valid output that matches the regex/EBNF constraints.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterator, List, Optional

from ..config import EvalConfig
from ..judges.regex_judge import RegexJudge
from ..judges import HAS_EBNF_SUPPORT
from ..metrics import EvalMetrics, EvalResult, SatisfactionLevel

if HAS_EBNF_SUPPORT:
    from ..judges.ebnf_judge import EbnfJudge

try:
    from tests.fixtures.constraints import get_all_examples
    from tests.fixtures.constraints.base import ConstraintExample
except ImportError:
    from ...fixtures.constraints import get_all_examples
    from ...fixtures.constraints.base import ConstraintExample


class ConstraintType(Enum):
    """Type of syntax constraint."""

    REGEX = "regex"
    EBNF = "ebnf"
    NONE = "none"


@dataclass
class SyntaxEvalResult:
    """Result of syntax satisfaction eval for a single example."""

    example: ConstraintExample
    result: EvalResult
    valid_output_results: List[EvalResult]
    invalid_output_results: List[EvalResult]
    validation_passed: bool
    validation_message: str
    constraint_type: ConstraintType = ConstraintType.REGEX
    # Count of invalid outputs that match syntax (semantic-only issues)
    semantic_only_invalids: int = 0


class SyntaxSatisfactionRunner:
    """Runner for Tier 1 syntax constraint satisfaction eval.

    This runner performs two types of evaluation:
    1. Validation: Ensure valid_outputs match and invalid_outputs don't match
    2. Generation: Evaluate Ananke-constrained generation (when generator provided)

    Supports both regex and EBNF constraints.
    """

    def __init__(
        self,
        config: Optional[EvalConfig] = None,
        generator: Optional[Callable[[str, "ConstraintSpec"], str]] = None,
    ):
        """Initialize the runner.

        Args:
            config: Eval configuration
            generator: Optional function that generates output given (prompt, spec)
        """
        self.config = config or EvalConfig.tier1_syntax()
        self.generator = generator
        self.regex_judge = RegexJudge()
        self.ebnf_judge = EbnfJudge() if HAS_EBNF_SUPPORT else None

    def _get_constraint_type(self, example: ConstraintExample) -> ConstraintType:
        """Determine the constraint type for an example."""
        # Prefer regex if available (faster evaluation)
        if example.spec.regex:
            return ConstraintType.REGEX
        elif example.spec.ebnf:
            return ConstraintType.EBNF
        return ConstraintType.NONE

    def get_examples(
        self,
        include_ebnf: bool = True,
    ) -> Iterator[ConstraintExample]:
        """Get examples matching the config filters.

        Args:
            include_ebnf: Whether to include EBNF-only examples
        """
        examples = get_all_examples()

        for example in examples:
            # Filter by language
            if self.config.languages and example.language not in self.config.languages:
                continue

            # Filter by domain
            if self.config.domains and example.domain not in self.config.domains:
                continue

            # Filter by tags
            if self.config.tags and not set(example.tags).intersection(self.config.tags):
                continue

            # Must have syntax constraint for Tier 1 eval
            constraint_type = self._get_constraint_type(example)
            if constraint_type == ConstraintType.NONE:
                continue

            # Skip EBNF if not requested or not available
            if constraint_type == ConstraintType.EBNF:
                if not include_ebnf or not HAS_EBNF_SUPPORT:
                    continue

            yield example

    def validate_example(
        self,
        example: ConstraintExample,
        syntax_only: bool = True,
    ) -> SyntaxEvalResult:
        """Validate a single example's valid/invalid outputs against its constraint.

        For Tier 1 (Syntax) validation:
        - All valid_outputs MUST match the constraint (this is a hard requirement)
        - Invalid outputs that match are counted as "semantic-only" issues (expected)

        For full validation (syntax_only=False):
        - All valid_outputs should match the constraint
        - All invalid_outputs should NOT match the constraint

        Supports both regex and EBNF constraints.

        Args:
            example: The constraint example to validate
            syntax_only: If True, only require valid outputs to match (Tier 1)

        Returns:
            SyntaxEvalResult with validation status
        """
        constraint_type = self._get_constraint_type(example)

        if constraint_type == ConstraintType.REGEX:
            pattern = example.spec.regex
            assert pattern is not None

            # Validate valid outputs - these MUST match
            valid_passed, valid_total, valid_results = self.regex_judge.evaluate_valid_outputs(
                example_id=example.id,
                valid_outputs=example.valid_outputs,
                pattern=pattern,
            )

            # Validate invalid outputs
            invalid_rejected, invalid_total, invalid_results = (
                self.regex_judge.evaluate_invalid_outputs(
                    example_id=example.id,
                    invalid_outputs=example.invalid_outputs,
                    pattern=pattern,
                )
            )

        elif constraint_type == ConstraintType.EBNF:
            if self.ebnf_judge is None:
                return SyntaxEvalResult(
                    example=example,
                    result=EvalResult(
                        example_id=example.id,
                        satisfied=False,
                        satisfaction_level=SatisfactionLevel.ERROR,
                        error="EBNF judge not available (missing lark or llguidance)",
                    ),
                    valid_output_results=[],
                    invalid_output_results=[],
                    validation_passed=False,
                    validation_message="EBNF judge not available",
                    constraint_type=constraint_type,
                )

            ebnf = example.spec.ebnf
            assert ebnf is not None

            # Validate valid outputs - these MUST match
            valid_passed, valid_total, valid_results = self.ebnf_judge.evaluate_valid_outputs(
                example_id=example.id,
                valid_outputs=example.valid_outputs,
                ebnf=ebnf,
            )

            # Validate invalid outputs
            invalid_rejected, invalid_total, invalid_results = (
                self.ebnf_judge.evaluate_invalid_outputs(
                    example_id=example.id,
                    invalid_outputs=example.invalid_outputs,
                    ebnf=ebnf,
                )
            )

        else:
            return SyntaxEvalResult(
                example=example,
                result=EvalResult(
                    example_id=example.id,
                    satisfied=False,
                    satisfaction_level=SatisfactionLevel.ERROR,
                    error="No syntax constraint found",
                ),
                valid_output_results=[],
                invalid_output_results=[],
                validation_passed=False,
                validation_message="No syntax constraint",
                constraint_type=constraint_type,
            )

        # Count semantic-only invalids (invalid outputs that match syntax)
        semantic_only_invalids = invalid_total - invalid_rejected

        # Build summary result
        all_valid_passed = valid_passed == valid_total
        all_invalid_rejected = invalid_rejected == invalid_total

        # For syntax-only (Tier 1), we only require valid outputs to match
        if syntax_only:
            validation_passed = all_valid_passed
        else:
            validation_passed = all_valid_passed and all_invalid_rejected

        if all_valid_passed:
            if all_invalid_rejected:
                validation_message = (
                    f"OK: {valid_passed}/{valid_total} valid matched, "
                    f"{invalid_rejected}/{invalid_total} invalid rejected"
                )
            else:
                validation_message = (
                    f"OK: {valid_passed}/{valid_total} valid matched "
                    f"({semantic_only_invalids} invalid have semantic-only issues)"
                )
            satisfaction_level = SatisfactionLevel.FULL
        else:
            issues = []
            failed_valid = [r for r in valid_results if not r.satisfied]
            issues.append(
                f"{valid_total - valid_passed}/{valid_total} valid outputs didn't match"
            )
            validation_message = f"FAIL: {'; '.join(issues)}"
            satisfaction_level = SatisfactionLevel.NONE

        summary_result = EvalResult(
            example_id=example.id,
            satisfied=validation_passed,
            satisfaction_level=satisfaction_level,
            metadata={
                "valid_passed": valid_passed,
                "valid_total": valid_total,
                "invalid_rejected": invalid_rejected,
                "invalid_total": invalid_total,
                "semantic_only_invalids": semantic_only_invalids,
                "constraint_type": constraint_type.value,
                "language": example.language,
                "domain": example.domain,
            },
        )

        return SyntaxEvalResult(
            example=example,
            result=summary_result,
            valid_output_results=valid_results,
            invalid_output_results=invalid_results,
            validation_passed=validation_passed,
            validation_message=validation_message,
            constraint_type=constraint_type,
            semantic_only_invalids=semantic_only_invalids,
        )

    def run_validation(self) -> EvalMetrics:
        """Run validation on all matching examples.

        This validates that the constraint examples are well-formed:
        - valid_outputs match the regex
        - invalid_outputs don't match the regex

        Returns:
            EvalMetrics with aggregated validation results
        """
        metrics = EvalMetrics()

        for example in self.get_examples():
            eval_result = self.validate_example(example)
            metrics.add_result(
                eval_result.result,
                language=example.language,
                domain=example.domain,
            )

        return metrics

    def evaluate_generation(
        self,
        example: ConstraintExample,
    ) -> EvalResult:
        """Evaluate constrained generation for a single example.

        Args:
            example: The constraint example

        Returns:
            EvalResult with generation evaluation
        """
        if self.generator is None:
            return EvalResult(
                example_id=example.id,
                satisfied=False,
                satisfaction_level=SatisfactionLevel.ERROR,
                error="No generator provided",
            )

        pattern = example.spec.regex
        if not pattern:
            return EvalResult(
                example_id=example.id,
                satisfied=False,
                satisfaction_level=SatisfactionLevel.ERROR,
                error="Example has no regex pattern",
            )

        # Time the generation
        start_time = time.perf_counter()
        try:
            output = self.generator(example.scenario, example.spec)
            latency_ms = (time.perf_counter() - start_time) * 1000
        except Exception as e:
            return EvalResult(
                example_id=example.id,
                satisfied=False,
                satisfaction_level=SatisfactionLevel.ERROR,
                error=f"Generation failed: {e}",
            )

        # Evaluate the output
        result = self.regex_judge.evaluate(
            example_id=example.id,
            output=output,
            pattern=pattern,
        )
        result.latency_ms = latency_ms

        return result

    def run(self) -> EvalMetrics:
        """Run the full Tier 1 eval.

        If a generator is provided, evaluates generation.
        Otherwise, runs validation only.

        Returns:
            EvalMetrics with aggregated results
        """
        if self.generator is None:
            return self.run_validation()

        metrics = EvalMetrics()

        for example in self.get_examples():
            result = self.evaluate_generation(example)
            metrics.add_result(
                result,
                language=example.language,
                domain=example.domain,
            )

        return metrics
