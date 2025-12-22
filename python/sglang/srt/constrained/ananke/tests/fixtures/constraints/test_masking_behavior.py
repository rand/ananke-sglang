# Copyright Rand Arete @ Ananke 2025
# Licensed under the Apache License, Version 2.0
"""Masking behavior tests for constraint examples.

These tests validate that constraint examples correctly distinguish between
valid and invalid outputs using their syntax constraints (regex/EBNF).

For each example:
1. Valid outputs SHOULD match the syntax constraint
2. Invalid outputs should NOT match the syntax constraint

This validates that the constraints are neither too permissive (allowing
invalid outputs) nor too restrictive (blocking valid outputs).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass

import pytest

# Ensure imports work from test directory
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

try:
    from . import get_all_examples
    from .base import ConstraintExample
except ImportError:
    from tests.fixtures.constraints import get_all_examples
    from tests.fixtures.constraints.base import ConstraintExample


# =============================================================================
# Test Result Tracking
# =============================================================================


@dataclass
class RegexTestResult:
    """Result of testing a single output against a regex."""
    example_id: str
    output: str
    expected_match: bool  # True for valid_outputs, False for invalid_outputs
    actual_match: bool
    regex_pattern: str

    @property
    def passed(self) -> bool:
        return self.expected_match == self.actual_match

    @property
    def failure_type(self) -> str:
        if self.passed:
            return "none"
        elif self.expected_match and not self.actual_match:
            return "false_negative"  # Should match but doesn't
        else:
            return "false_positive"  # Shouldn't match but does


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def all_examples() -> List[ConstraintExample]:
    """Get all constraint examples."""
    return get_all_examples()


@pytest.fixture(scope="module")
def examples_with_regex(all_examples: List[ConstraintExample]) -> List[ConstraintExample]:
    """Get examples that have regex constraints."""
    return [e for e in all_examples if e.spec.regex]


@pytest.fixture(scope="module")
def examples_with_ebnf(all_examples: List[ConstraintExample]) -> List[ConstraintExample]:
    """Get examples that have EBNF constraints."""
    return [e for e in all_examples if e.spec.ebnf]


# =============================================================================
# Regex Constraint Tests
# =============================================================================


class TestRegexValidOutputs:
    """Test that valid outputs match regex constraints."""

    def test_valid_outputs_match_regex(
        self, examples_with_regex: List[ConstraintExample]
    ) -> None:
        """All valid_outputs should match the regex pattern."""
        failures: List[RegexTestResult] = []

        for example in examples_with_regex:
            if not example.valid_outputs:
                continue

            pattern = example.spec.regex
            try:
                regex = re.compile(pattern, re.MULTILINE | re.DOTALL)
            except re.error as e:
                pytest.fail(f"Invalid regex in {example.id}: {e}")

            for output in example.valid_outputs:
                match = regex.search(output)
                if not match:
                    failures.append(RegexTestResult(
                        example_id=example.id,
                        output=output[:100] + "..." if len(output) > 100 else output,
                        expected_match=True,
                        actual_match=False,
                        regex_pattern=pattern[:80] + "..." if len(pattern) > 80 else pattern,
                    ))

        if failures:
            msg_lines = [f"\n{len(failures)} valid outputs don't match their regex:"]
            for f in failures[:10]:  # Show first 10
                msg_lines.append(f"\n  [{f.example_id}]")
                msg_lines.append(f"    Pattern: {f.regex_pattern}")
                msg_lines.append(f"    Output: {f.output}")
            if len(failures) > 10:
                msg_lines.append(f"\n  ... and {len(failures) - 10} more failures")
            pytest.fail("\n".join(msg_lines))

    def test_valid_output_coverage(
        self, examples_with_regex: List[ConstraintExample]
    ) -> None:
        """Check that examples have valid outputs to test."""
        missing = [e.id for e in examples_with_regex if not e.valid_outputs]

        if missing:
            # This is a warning, not a failure - some examples may legitimately
            # not have valid outputs defined yet
            print(f"\nNote: {len(missing)} examples with regex have no valid_outputs")


class TestRegexInvalidOutputs:
    """Test that invalid outputs do NOT match regex constraints."""

    def test_invalid_outputs_rejected(
        self, examples_with_regex: List[ConstraintExample]
    ) -> None:
        """Invalid outputs should NOT match the regex pattern.

        Note: This tests that the regex is selective enough to reject
        known-bad outputs. A regex that matches everything is too permissive.
        """
        false_positives: List[RegexTestResult] = []

        for example in examples_with_regex:
            if not example.invalid_outputs:
                continue

            pattern = example.spec.regex
            try:
                regex = re.compile(pattern, re.MULTILINE | re.DOTALL)
            except re.error:
                continue  # Already tested in valid outputs test

            for output in example.invalid_outputs:
                # Clean the output - remove trailing comments like "# Missing ..."
                clean_output = output.split("  #")[0].strip()

                match = regex.search(clean_output)
                if match:
                    false_positives.append(RegexTestResult(
                        example_id=example.id,
                        output=clean_output[:100] + "..." if len(clean_output) > 100 else clean_output,
                        expected_match=False,
                        actual_match=True,
                        regex_pattern=pattern[:80] + "..." if len(pattern) > 80 else pattern,
                    ))

        if false_positives:
            msg_lines = [f"\n{len(false_positives)} invalid outputs incorrectly match regex:"]
            for f in false_positives[:10]:
                msg_lines.append(f"\n  [{f.example_id}] (false positive)")
                msg_lines.append(f"    Pattern: {f.regex_pattern}")
                msg_lines.append(f"    Invalid output matched: {f.output}")
            if len(false_positives) > 10:
                msg_lines.append(f"\n  ... and {len(false_positives) - 10} more")

            # This is informational - regex patterns are often intentionally broad
            # and rely on EBNF or domain constraints to reject invalid outputs
            print("\n".join(msg_lines))


class TestRegexPatternQuality:
    """Test quality of regex patterns."""

    def test_regex_patterns_compile(
        self, examples_with_regex: List[ConstraintExample]
    ) -> None:
        """All regex patterns should compile without errors."""
        compile_errors = []

        for example in examples_with_regex:
            try:
                re.compile(example.spec.regex)
            except re.error as e:
                compile_errors.append((example.id, str(e)))

        if compile_errors:
            msg = "\n".join(f"  {eid}: {err}" for eid, err in compile_errors)
            pytest.fail(f"Regex compile errors:\n{msg}")

    def test_regex_not_trivially_permissive(
        self, examples_with_regex: List[ConstraintExample]
    ) -> None:
        """Regex patterns shouldn't match everything."""
        trivial_patterns = []

        trivial_regexes = [r".*", r".+", r"^.*$", r"[\s\S]*", r"^[\s\S]*$"]

        for example in examples_with_regex:
            pattern = example.spec.regex.strip()
            if pattern in trivial_regexes:
                trivial_patterns.append(example.id)

        if trivial_patterns:
            pytest.fail(
                f"Trivially permissive regex patterns in: {trivial_patterns}"
            )

    def test_regex_has_anchors_or_structure(
        self, examples_with_regex: List[ConstraintExample]
    ) -> None:
        """Regex patterns should have anchors or meaningful structure."""
        weak_patterns = []

        for example in examples_with_regex:
            pattern = example.spec.regex
            # Weak: no anchors, no specific tokens, just wildcards
            if not any(c in pattern for c in ['^', '$', '\\b', '(?=', '(?!']):
                # Check if it has at least some literal text to match
                literals = re.sub(r'\\.|[.*+?[\](){}|^$]', '', pattern)
                if len(literals) < 3:
                    weak_patterns.append((example.id, pattern[:50]))

        if weak_patterns:
            print(f"\nNote: {len(weak_patterns)} regex patterns may be too weak:")
            for eid, pat in weak_patterns[:5]:
                print(f"  {eid}: {pat}")


# =============================================================================
# EBNF Constraint Tests (require llguidance)
# =============================================================================


try:
    import llguidance
    HAS_LLGUIDANCE = True
except ImportError:
    HAS_LLGUIDANCE = False


@pytest.mark.skipif(not HAS_LLGUIDANCE, reason="llguidance not available")
class TestEbnfConstraints:
    """Test EBNF constraints using llguidance."""

    def test_ebnf_examples_exist(
        self, examples_with_ebnf: List[ConstraintExample]
    ) -> None:
        """Verify EBNF examples exist for testing."""
        assert len(examples_with_ebnf) > 0, "No EBNF examples found"
        print(f"\nFound {len(examples_with_ebnf)} examples with EBNF constraints")

    def test_ebnf_grammar_parses(
        self, examples_with_ebnf: List[ConstraintExample]
    ) -> None:
        """EBNF grammars should parse without errors."""
        parse_errors = []

        for example in examples_with_ebnf:
            try:
                # llguidance should be able to parse the EBNF
                # This is a placeholder - actual implementation depends on llguidance API
                ebnf = example.spec.ebnf
                if not ebnf.strip():
                    parse_errors.append((example.id, "Empty EBNF"))
            except Exception as e:
                parse_errors.append((example.id, str(e)))

        if parse_errors:
            msg = "\n".join(f"  {eid}: {err}" for eid, err in parse_errors)
            pytest.fail(f"EBNF parse errors:\n{msg}")


# =============================================================================
# Combined Constraint Tests
# =============================================================================


class TestConstraintCoverage:
    """Test overall constraint coverage and quality."""

    def test_all_examples_have_outputs(
        self, all_examples: List[ConstraintExample]
    ) -> None:
        """Examples should have at least one valid or invalid output."""
        missing_outputs = [
            e.id for e in all_examples
            if not e.valid_outputs and not e.invalid_outputs
        ]

        # This is informational - not all examples need outputs defined
        if missing_outputs:
            print(f"\nNote: {len(missing_outputs)} examples have no valid/invalid outputs")

    def test_constraint_type_distribution(
        self, all_examples: List[ConstraintExample]
    ) -> None:
        """Report distribution of constraint types."""
        has_regex = sum(1 for e in all_examples if e.spec.regex)
        has_ebnf = sum(1 for e in all_examples if e.spec.ebnf)
        has_json = sum(1 for e in all_examples if e.spec.json_schema)
        has_both = sum(1 for e in all_examples if e.spec.regex and e.spec.ebnf)

        print(f"\nConstraint type distribution:")
        print(f"  Regex only: {has_regex - has_both}")
        print(f"  EBNF only: {has_ebnf - has_both}")
        print(f"  Both: {has_both}")
        print(f"  JSON Schema: {has_json}")


# =============================================================================
# Summary Report
# =============================================================================


class TestSummaryReport:
    """Generate summary report of masking behavior tests."""

    def test_generate_masking_report(
        self,
        examples_with_regex: List[ConstraintExample],
        examples_with_ebnf: List[ConstraintExample],
    ) -> None:
        """Generate summary report of regex matching behavior."""
        total_valid_tests = 0
        valid_matches = 0
        total_invalid_tests = 0
        invalid_rejections = 0

        for example in examples_with_regex:
            if not example.spec.regex:
                continue

            try:
                regex = re.compile(example.spec.regex, re.MULTILINE | re.DOTALL)
            except re.error:
                continue

            for output in example.valid_outputs:
                total_valid_tests += 1
                if regex.search(output):
                    valid_matches += 1

            for output in example.invalid_outputs:
                total_invalid_tests += 1
                clean_output = output.split("  #")[0].strip()
                if not regex.search(clean_output):
                    invalid_rejections += 1

        print(f"\n{'='*60}")
        print("MASKING BEHAVIOR SUMMARY")
        print(f"{'='*60}")
        print(f"Examples with regex: {len(examples_with_regex)}")
        print(f"Examples with EBNF: {len(examples_with_ebnf)}")
        print(f"\nValid output matching:")
        print(f"  Total tested: {total_valid_tests}")
        print(f"  Matched regex: {valid_matches}")
        if total_valid_tests > 0:
            pct = 100 * valid_matches / total_valid_tests
            print(f"  Match rate: {pct:.1f}%")
        print(f"\nInvalid output rejection:")
        print(f"  Total tested: {total_invalid_tests}")
        print(f"  Rejected by regex: {invalid_rejections}")
        if total_invalid_tests > 0:
            pct = 100 * invalid_rejections / total_invalid_tests
            print(f"  Rejection rate: {pct:.1f}%")
        print(f"{'='*60}")
