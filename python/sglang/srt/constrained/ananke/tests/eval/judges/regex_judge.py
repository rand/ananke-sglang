# Copyright Rand Arete @ Ananke 2025
# Licensed under the Apache License, Version 2.0
"""Regex-based judge for syntax constraint satisfaction."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from ..metrics import EvalResult, SatisfactionLevel


@dataclass
class RegexMatch:
    """Details of a regex match."""

    matched: bool
    pattern: str
    match_text: Optional[str] = None
    match_start: Optional[int] = None
    match_end: Optional[int] = None
    error: Optional[str] = None


class RegexJudge:
    """Judge for regex-based syntax constraints.

    This judge evaluates whether generated output matches the regex pattern
    specified in a ConstraintSpec. It supports:
    - Single regex pattern matching
    - Multiple patterns (all must match)
    - Negative patterns (must NOT match)
    - Partial credit for multi-part patterns
    """

    # Common flags for code pattern matching
    DEFAULT_FLAGS = re.MULTILINE | re.DOTALL

    def __init__(
        self,
        flags: int = DEFAULT_FLAGS,
        allow_partial: bool = False,
    ):
        """Initialize the regex judge.

        Args:
            flags: Regex flags to use (default: MULTILINE | DOTALL)
            allow_partial: Whether to give partial credit for partial matches
        """
        self.flags = flags
        self.allow_partial = allow_partial

    def matches(self, output: str, pattern: str) -> bool:
        """Simple check if output matches pattern.

        Args:
            output: Text to check
            pattern: Regex pattern

        Returns:
            True if pattern matches output
        """
        match_result = self._match_pattern(pattern, output)
        return match_result.matched and not match_result.error

    def evaluate(
        self,
        example_id: str,
        output: str,
        pattern: str,
        negative_patterns: Optional[List[str]] = None,
    ) -> EvalResult:
        """Evaluate whether output satisfies the regex constraint.

        Args:
            example_id: Unique identifier for the example
            output: Generated output to evaluate
            pattern: Regex pattern that should match
            negative_patterns: Optional patterns that should NOT match

        Returns:
            EvalResult with satisfaction status and details
        """
        # Try to match the positive pattern
        positive_match = self._match_pattern(pattern, output)

        if positive_match.error:
            return EvalResult(
                example_id=example_id,
                satisfied=False,
                satisfaction_level=SatisfactionLevel.ERROR,
                output=output,
                error=f"Regex error: {positive_match.error}",
                metadata={"pattern": pattern},
            )

        # Check negative patterns if any
        negative_violations = []
        if negative_patterns:
            for neg_pattern in negative_patterns:
                neg_match = self._match_pattern(neg_pattern, output)
                if neg_match.error:
                    return EvalResult(
                        example_id=example_id,
                        satisfied=False,
                        satisfaction_level=SatisfactionLevel.ERROR,
                        output=output,
                        error=f"Negative pattern regex error: {neg_match.error}",
                        metadata={"negative_pattern": neg_pattern},
                    )
                if neg_match.matched:
                    negative_violations.append(neg_pattern)

        # Determine satisfaction level
        positive_satisfied = positive_match.matched
        negative_satisfied = len(negative_violations) == 0

        if positive_satisfied and negative_satisfied:
            satisfaction_level = SatisfactionLevel.FULL
            satisfied = True
        elif positive_satisfied and not negative_satisfied:
            # Matched positive but also matched forbidden patterns
            satisfaction_level = (
                SatisfactionLevel.PARTIAL if self.allow_partial else SatisfactionLevel.NONE
            )
            satisfied = False
        elif not positive_satisfied and negative_satisfied:
            # Didn't match required pattern
            satisfaction_level = SatisfactionLevel.NONE
            satisfied = False
        else:
            # Neither positive nor negative satisfied
            satisfaction_level = SatisfactionLevel.NONE
            satisfied = False

        return EvalResult(
            example_id=example_id,
            satisfied=satisfied,
            satisfaction_level=satisfaction_level,
            output=output,
            metadata={
                "pattern": pattern,
                "pattern_matched": positive_match.matched,
                "match_text": positive_match.match_text,
                "negative_violations": negative_violations if negative_violations else None,
            },
        )

    def _match_pattern(self, pattern: str, text: str) -> RegexMatch:
        """Match a single pattern against text.

        Args:
            pattern: Regex pattern
            text: Text to search

        Returns:
            RegexMatch with results
        """
        try:
            compiled = re.compile(pattern, self.flags)
            match = compiled.search(text)

            if match:
                return RegexMatch(
                    matched=True,
                    pattern=pattern,
                    match_text=match.group(0)[:200],  # Truncate long matches
                    match_start=match.start(),
                    match_end=match.end(),
                )
            else:
                return RegexMatch(
                    matched=False,
                    pattern=pattern,
                )
        except re.error as e:
            return RegexMatch(
                matched=False,
                pattern=pattern,
                error=str(e),
            )

    def evaluate_valid_outputs(
        self,
        example_id: str,
        valid_outputs: List[str],
        pattern: str,
        negative_patterns: Optional[List[str]] = None,
    ) -> Tuple[int, int, List[EvalResult]]:
        """Evaluate a list of valid outputs against the constraint.

        This is a validation method to ensure valid_outputs actually match.

        Args:
            example_id: Base example ID
            valid_outputs: List of known-valid outputs
            pattern: Regex pattern that should match
            negative_patterns: Optional patterns that should NOT match

        Returns:
            Tuple of (passed_count, total_count, results)
        """
        results = []
        passed = 0

        for i, output in enumerate(valid_outputs):
            result = self.evaluate(
                example_id=f"{example_id}-valid-{i}",
                output=output,
                pattern=pattern,
                negative_patterns=negative_patterns,
            )
            results.append(result)
            if result.satisfied:
                passed += 1

        return passed, len(valid_outputs), results

    def evaluate_invalid_outputs(
        self,
        example_id: str,
        invalid_outputs: List[str],
        pattern: str,
    ) -> Tuple[int, int, List[EvalResult]]:
        """Evaluate a list of invalid outputs against the constraint.

        This is a validation method to ensure invalid_outputs don't fully match.

        Args:
            example_id: Base example ID
            invalid_outputs: List of known-invalid outputs
            pattern: Regex pattern

        Returns:
            Tuple of (correctly_rejected_count, total_count, results)
        """
        results = []
        rejected = 0

        for i, output in enumerate(invalid_outputs):
            result = self.evaluate(
                example_id=f"{example_id}-invalid-{i}",
                output=output,
                pattern=pattern,
            )
            results.append(result)
            # Invalid outputs SHOULD NOT satisfy the constraint
            if not result.satisfied:
                rejected += 1

        return rejected, len(invalid_outputs), results
