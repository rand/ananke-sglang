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
"""Tests for execution verifier (PoT verification).

Tests the ExecutionVerifier protocol and implementations for Phase 2.3.
"""

from __future__ import annotations

import pytest

try:
    from ...verification.execution import (
        ExecutionResult,
        TestCase,
        ExecutionScore,
        ExecutionVerifier,
        StaticExecutionVerifier,
        RestrictedExecutionVerifier,
        generate_test_cases_from_signature,
        create_execution_verifier,
    )
except ImportError:
    from verification.execution import (
        ExecutionResult,
        TestCase,
        ExecutionScore,
        ExecutionVerifier,
        StaticExecutionVerifier,
        RestrictedExecutionVerifier,
        generate_test_cases_from_signature,
        create_execution_verifier,
    )


# =============================================================================
# ExecutionResult Tests
# =============================================================================


class TestExecutionResult:
    """Tests for ExecutionResult enum."""

    def test_all_values_exist(self) -> None:
        """Test that all result values exist."""
        assert ExecutionResult.PASS
        assert ExecutionResult.FAIL
        assert ExecutionResult.ERROR
        assert ExecutionResult.TIMEOUT
        assert ExecutionResult.SKIPPED

    def test_values_are_distinct(self) -> None:
        """Test that all values are distinct."""
        values = [
            ExecutionResult.PASS,
            ExecutionResult.FAIL,
            ExecutionResult.ERROR,
            ExecutionResult.TIMEOUT,
            ExecutionResult.SKIPPED,
        ]
        assert len(values) == len(set(values))


# =============================================================================
# TestCase Tests
# =============================================================================


class TestTestCase:
    """Tests for TestCase dataclass."""

    def test_create_basic(self) -> None:
        """Test basic creation."""
        tc = TestCase(input="add(1, 2)", expected_output="3")
        assert tc.input == "add(1, 2)"
        assert tc.expected_output == "3"
        assert tc.name == ""
        assert tc.timeout_ms == 1000

    def test_create_with_name(self) -> None:
        """Test creation with name."""
        tc = TestCase(
            input="add(1, 2)",
            expected_output="3",
            name="test_add_basic",
            timeout_ms=500,
        )
        assert tc.name == "test_add_basic"
        assert tc.timeout_ms == 500

    def test_is_frozen(self) -> None:
        """Test that TestCase is immutable."""
        tc = TestCase(input="x", expected_output="y")
        with pytest.raises(AttributeError):
            tc.input = "z"  # type: ignore


# =============================================================================
# ExecutionScore Tests
# =============================================================================


class TestExecutionScore:
    """Tests for ExecutionScore dataclass."""

    def test_create_empty(self) -> None:
        """Test creation with defaults."""
        score = ExecutionScore()
        assert score.passed == 0
        assert score.failed == 0
        assert score.errors == 0
        assert score.skipped == 0
        assert score.total == 0
        assert score.score == 0.0

    def test_all_passed(self) -> None:
        """Test all_passed property."""
        score = ExecutionScore(passed=3, total=3)
        assert score.all_passed

        score2 = ExecutionScore(passed=2, failed=1, total=3)
        assert not score2.all_passed

        score3 = ExecutionScore(passed=0, total=0)
        assert not score3.all_passed  # No tests = not all passed

    def test_compute_score_all_pass(self) -> None:
        """Test score computation when all pass."""
        score = ExecutionScore(passed=5, total=5)
        assert score.compute_score() == 1.0

    def test_compute_score_all_fail(self) -> None:
        """Test score computation when all fail."""
        score = ExecutionScore(failed=5, total=5)
        assert score.compute_score() == 0.0

    def test_compute_score_mixed(self) -> None:
        """Test score computation with mixed results."""
        score = ExecutionScore(passed=2, failed=2, total=4)
        assert score.compute_score() == 0.5

    def test_compute_score_with_errors(self) -> None:
        """Test that errors contribute partially."""
        score = ExecutionScore(errors=4, total=4)
        # 4 errors * 0.25 / 4 = 0.25
        assert score.compute_score() == 0.25

    def test_compute_score_with_skipped(self) -> None:
        """Test that skipped tests contribute partially."""
        score = ExecutionScore(skipped=4, total=4)
        # 4 skipped * 0.5 / 4 = 0.5
        assert score.compute_score() == 0.5

    def test_compute_score_empty(self) -> None:
        """Test score computation with no tests."""
        score = ExecutionScore()
        assert score.compute_score() == 0.5  # Neutral score


# =============================================================================
# StaticExecutionVerifier Tests
# =============================================================================


class TestStaticExecutionVerifier:
    """Tests for StaticExecutionVerifier."""

    def test_verify_valid_function(self) -> None:
        """Test verification of valid function."""
        verifier = StaticExecutionVerifier()
        code = "def add(a, b):\n    return a + b"
        tests = [
            TestCase(input="add(1, 2)", expected_output="3"),
            TestCase(input="add(0, 0)", expected_output="0"),
        ]
        result = verifier.verify(code, tests, "add")

        assert result.passed == 2
        assert result.total == 2
        assert result.score == 1.0

    def test_verify_syntax_error(self) -> None:
        """Test verification of code with syntax error."""
        verifier = StaticExecutionVerifier()
        code = "def add(a, b:\n    return"  # Missing )
        tests = [TestCase(input="add(1, 2)", expected_output="3")]
        result = verifier.verify(code, tests)

        assert result.errors == 1
        assert result.score == 0.0

    def test_verify_no_return(self) -> None:
        """Test verification of function without return."""
        verifier = StaticExecutionVerifier()
        code = "def add(a, b):\n    c = a + b"  # No return
        tests = [TestCase(input="add(1, 2)", expected_output="3")]
        result = verifier.verify(code, tests, "add")

        assert result.failed == 1
        assert result.score == 0.0

    def test_verify_wrong_arity(self) -> None:
        """Test verification with wrong number of arguments."""
        verifier = StaticExecutionVerifier()
        code = "def add(a, b, c):\n    return a + b + c"  # 3 args
        tests = [TestCase(input="add(1, 2)", expected_output="3")]  # 2 args
        result = verifier.verify(code, tests, "add")

        assert result.failed == 1

    def test_verify_undefined_name(self) -> None:
        """Test verification with undefined name."""
        verifier = StaticExecutionVerifier()
        code = "def add(a, b):\n    return a + b + undefined_var"
        tests = [TestCase(input="add(1, 2)", expected_output="3")]
        result = verifier.verify(code, tests, "add")

        assert result.errors == 1

    def test_verify_no_tests(self) -> None:
        """Test verification with no test cases."""
        verifier = StaticExecutionVerifier()
        code = "def add(a, b):\n    return a + b"
        result = verifier.verify(code, [], "add")

        assert result.score == 0.5  # Neutral score

    def test_verify_function_not_found(self) -> None:
        """Test verification when function not found."""
        verifier = StaticExecutionVerifier()
        code = "def subtract(a, b):\n    return a - b"
        tests = [TestCase(input="add(1, 2)", expected_output="3")]
        result = verifier.verify(code, tests, "add")

        assert result.skipped == 1
        assert result.score == 0.5

    def test_is_execution_verifier(self) -> None:
        """Test that StaticExecutionVerifier satisfies protocol."""
        verifier = StaticExecutionVerifier()
        assert isinstance(verifier, ExecutionVerifier)


# =============================================================================
# RestrictedExecutionVerifier Tests
# =============================================================================


class TestRestrictedExecutionVerifier:
    """Tests for RestrictedExecutionVerifier."""

    def test_verify_simple_function(self) -> None:
        """Test execution of simple function."""
        verifier = RestrictedExecutionVerifier()
        code = "def add(a, b):\n    return a + b"
        tests = [
            TestCase(input="add(1, 2)", expected_output="3"),
            TestCase(input="add(0, 0)", expected_output="0"),
            TestCase(input="add(-1, 1)", expected_output="0"),
        ]
        result = verifier.verify(code, tests, "add")

        assert result.passed == 3
        assert result.total == 3
        assert result.all_passed
        assert result.score == 1.0

    def test_verify_failing_test(self) -> None:
        """Test execution with failing test."""
        verifier = RestrictedExecutionVerifier()
        code = "def add(a, b):\n    return a - b"  # Wrong implementation
        tests = [TestCase(input="add(1, 2)", expected_output="3")]
        result = verifier.verify(code, tests, "add")

        assert result.failed == 1
        assert result.score == 0.0

    def test_verify_syntax_error(self) -> None:
        """Test execution with syntax error."""
        verifier = RestrictedExecutionVerifier()
        code = "def add(a, b:\n    return a + b"  # Syntax error
        tests = [TestCase(input="add(1, 2)", expected_output="3")]
        result = verifier.verify(code, tests)

        assert result.errors == 1
        assert result.score == 0.0

    def test_verify_runtime_error(self) -> None:
        """Test execution with runtime error."""
        verifier = RestrictedExecutionVerifier()
        code = "def divide(a, b):\n    return a / b"
        tests = [TestCase(input="divide(1, 0)", expected_output="inf")]
        result = verifier.verify(code, tests, "divide")

        assert result.errors == 1

    def test_restricted_builtins(self) -> None:
        """Test that unsafe builtins are blocked."""
        verifier = RestrictedExecutionVerifier()
        # Try to use open() which should be blocked
        code = "def read_file(path):\n    return open(path).read()"
        tests = [TestCase(input='read_file("/etc/passwd")', expected_output='""')]
        result = verifier.verify(code, tests, "read_file")

        # Should fail because open is not in safe builtins
        assert result.errors == 1 or result.failed == 1

    def test_verify_with_list_operations(self) -> None:
        """Test execution with list operations."""
        verifier = RestrictedExecutionVerifier()
        code = "def double_list(lst):\n    return [x * 2 for x in lst]"
        tests = [
            TestCase(input="double_list([1, 2, 3])", expected_output="[2, 4, 6]"),
            TestCase(input="double_list([])", expected_output="[]"),
        ]
        result = verifier.verify(code, tests, "double_list")

        assert result.passed == 2

    def test_verify_auto_find_function(self) -> None:
        """Test that function is auto-found when name not specified."""
        verifier = RestrictedExecutionVerifier()
        code = "def mystery(x):\n    return x * 2"
        tests = [TestCase(input="mystery(5)", expected_output="10")]
        result = verifier.verify(code, tests)  # No function name

        assert result.passed == 1

    def test_verify_no_tests(self) -> None:
        """Test verification with no test cases."""
        verifier = RestrictedExecutionVerifier()
        code = "def add(a, b):\n    return a + b"
        result = verifier.verify(code, [], "add")

        assert result.score == 0.5

    def test_is_execution_verifier(self) -> None:
        """Test that RestrictedExecutionVerifier satisfies protocol."""
        verifier = RestrictedExecutionVerifier()
        assert isinstance(verifier, ExecutionVerifier)


# =============================================================================
# Test Case Generation Tests
# =============================================================================


class TestGenerateTestCases:
    """Tests for generate_test_cases_from_signature."""

    def test_generate_int_params(self) -> None:
        """Test generation with int parameters."""
        cases = generate_test_cases_from_signature(
            "def add(a: int, b: int) -> int:",
            num_cases=3,
        )
        assert len(cases) == 3
        assert "add(" in cases[0].input

    def test_generate_str_params(self) -> None:
        """Test generation with string parameters."""
        cases = generate_test_cases_from_signature(
            "def greet(name: str) -> str:",
            num_cases=2,
        )
        assert len(cases) == 2

    def test_generate_no_params(self) -> None:
        """Test generation with no parameters."""
        cases = generate_test_cases_from_signature(
            "def get_pi() -> float:",
            num_cases=3,
        )
        assert len(cases) == 1  # Only one case for no-arg function
        assert "()" in cases[0].input

    def test_generate_untyped_params(self) -> None:
        """Test generation with untyped parameters."""
        cases = generate_test_cases_from_signature(
            "def add(a, b):",
            num_cases=2,
        )
        assert len(cases) == 2

    def test_generate_invalid_signature(self) -> None:
        """Test generation with invalid signature."""
        cases = generate_test_cases_from_signature("not a function", num_cases=3)
        assert len(cases) == 0

    def test_generate_extracts_function_name(self) -> None:
        """Test that function name is extracted correctly."""
        cases = generate_test_cases_from_signature(
            "def calculate_sum(x: int, y: int) -> int:"
        )
        assert all("calculate_sum(" in tc.input for tc in cases)


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestCreateExecutionVerifier:
    """Tests for create_execution_verifier factory."""

    def test_create_static(self) -> None:
        """Test creating static verifier."""
        verifier = create_execution_verifier("static")
        assert isinstance(verifier, StaticExecutionVerifier)

    def test_create_restricted(self) -> None:
        """Test creating restricted verifier."""
        verifier = create_execution_verifier("restricted")
        assert isinstance(verifier, RestrictedExecutionVerifier)

    def test_unknown_type_defaults_to_static(self) -> None:
        """Test that unknown type defaults to static."""
        verifier = create_execution_verifier("unknown")
        assert isinstance(verifier, StaticExecutionVerifier)

    def test_create_with_kwargs(self) -> None:
        """Test creation with additional kwargs."""
        verifier = create_execution_verifier("restricted", timeout_ms=500)
        assert isinstance(verifier, RestrictedExecutionVerifier)
        assert verifier.timeout_ms == 500


# =============================================================================
# Integration Tests
# =============================================================================


class TestExecutionVerifierIntegration:
    """Integration tests for execution verifiers."""

    def test_protocol_compliance(self) -> None:
        """Test that all verifiers satisfy the protocol."""
        verifiers = [
            StaticExecutionVerifier(),
            RestrictedExecutionVerifier(),
        ]
        code = "def add(a, b):\n    return a + b"
        tests = [TestCase(input="add(1, 2)", expected_output="3")]

        for v in verifiers:
            assert isinstance(v, ExecutionVerifier)
            result = v.verify(code, tests, "add")
            assert isinstance(result, ExecutionScore)
            assert 0.0 <= result.score <= 1.0

    def test_static_vs_restricted_consistency(self) -> None:
        """Test that static and restricted give consistent results for valid code."""
        static = StaticExecutionVerifier()
        restricted = RestrictedExecutionVerifier()

        code = "def multiply(a, b):\n    return a * b"
        tests = [
            TestCase(input="multiply(2, 3)", expected_output="6"),
            TestCase(input="multiply(0, 5)", expected_output="0"),
        ]

        static_result = static.verify(code, tests, "multiply")
        restricted_result = restricted.verify(code, tests, "multiply")

        # Both should pass all tests
        assert static_result.passed == 2
        assert restricted_result.passed == 2

    def test_end_to_end_verification(self) -> None:
        """Test end-to-end verification flow."""
        # 1. Generate test cases from signature
        signature = "def is_even(n: int) -> bool:"
        cases = generate_test_cases_from_signature(signature, num_cases=2)

        # 2. Replace expected with actual expected values
        test_cases = [
            TestCase(input="is_even(0)", expected_output="True"),
            TestCase(input="is_even(1)", expected_output="False"),
            TestCase(input="is_even(42)", expected_output="True"),
        ]

        # 3. Verify code
        verifier = create_execution_verifier("restricted")
        code = "def is_even(n):\n    return n % 2 == 0"
        result = verifier.verify(code, test_cases, "is_even")

        assert result.all_passed
        assert result.score == 1.0
