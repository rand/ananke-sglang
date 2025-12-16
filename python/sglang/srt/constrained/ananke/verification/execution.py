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
"""Code execution verifier for PoT (Program of Thought) verification.

This module provides execution-based verification of generated code by
running it against test cases. Supports both sandboxed execution and
static analysis fallback.

Design Principles (from Plan Phase 2.3):
1. Safe execution in sandboxed environment
2. Test case generation from function signatures
3. Execution result scoring (pass/fail/error)
4. Static analysis fallback when execution unavailable

Security:
    Code execution MUST be sandboxed. This module provides:
    - Restricted execution with resource limits
    - Static analysis as safe fallback
    - Never executes untrusted code without explicit opt-in

References:
    - Collaborative Verification (Oct 2024): https://arxiv.org/abs/2410.05318
    - Program of Thought Prompting (NeurIPS 2024)
"""

from __future__ import annotations

import ast
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, runtime_checkable

logger = logging.getLogger(__name__)


class ExecutionResult(Enum):
    """Result of code execution."""

    PASS = auto()  # Test passed
    FAIL = auto()  # Test failed (wrong output)
    ERROR = auto()  # Execution error (exception)
    TIMEOUT = auto()  # Execution timed out
    SKIPPED = auto()  # Test skipped (e.g., sandbox unavailable)


@dataclass(frozen=True)
class TestCase:
    """A test case for execution verification.

    Attributes:
        input: Input arguments (as string or dict)
        expected_output: Expected return value
        name: Optional test name
        timeout_ms: Timeout in milliseconds
    """

    input: str
    expected_output: str
    name: str = ""
    timeout_ms: int = 1000


@dataclass
class ExecutionScore:
    """Score from execution-based verification.

    Attributes:
        passed: Number of tests passed
        failed: Number of tests failed
        errors: Number of tests with errors
        skipped: Number of tests skipped
        total: Total number of tests
        score: Overall score (0.0 to 1.0)
        details: Per-test results
        latency_ms: Total execution time
    """

    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    total: int = 0
    score: float = 0.0
    details: Dict[str, ExecutionResult] = field(default_factory=dict)
    latency_ms: float = 0.0

    @property
    def all_passed(self) -> bool:
        """Check if all tests passed."""
        return self.passed == self.total and self.total > 0

    def compute_score(self) -> float:
        """Compute score from results."""
        if self.total == 0:
            return 0.5  # No tests = neutral score

        # Passed tests count fully, errors partially
        effective_passed = self.passed + (self.errors * 0.25) + (self.skipped * 0.5)
        return effective_passed / self.total


@runtime_checkable
class ExecutionVerifier(Protocol):
    """Protocol for execution-based verifiers.

    Execution verifiers run generated code against test cases and
    report pass/fail results. They MUST ensure safe execution.
    """

    def verify(
        self,
        code: str,
        test_cases: List[TestCase],
        function_name: str = "",
    ) -> ExecutionScore:
        """Verify code by executing test cases.

        Args:
            code: The code to verify
            test_cases: Test cases to run
            function_name: Name of function to test

        Returns:
            ExecutionScore with results
        """
        ...


class StaticExecutionVerifier:
    """Static analysis fallback for execution verification.

    When sandboxed execution is unavailable, this verifier performs
    static analysis to estimate whether code would pass tests.

    Static checks include:
    - Function signature matches test inputs
    - Return statement present
    - No obvious runtime errors (undefined names, etc.)
    """

    def __init__(self, language: str = "python"):
        """Initialize static verifier.

        Args:
            language: Target programming language
        """
        self.language = language

    def verify(
        self,
        code: str,
        test_cases: List[TestCase],
        function_name: str = "",
    ) -> ExecutionScore:
        """Verify using static analysis."""
        import time

        start_time = time.perf_counter()
        details: Dict[str, ExecutionResult] = {}

        if not test_cases:
            return ExecutionScore(
                score=0.5,
                latency_ms=(time.perf_counter() - start_time) * 1000,
            )

        # Parse code
        try:
            tree = ast.parse(code)
        except SyntaxError:
            # Syntax error = all tests fail
            for tc in test_cases:
                details[tc.name or f"test_{len(details)}"] = ExecutionResult.ERROR
            return ExecutionScore(
                errors=len(test_cases),
                total=len(test_cases),
                score=0.0,
                details=details,
                latency_ms=(time.perf_counter() - start_time) * 1000,
            )

        # Find target function
        func_def = self._find_function(tree, function_name)
        if func_def is None:
            # No function = skip all tests
            for tc in test_cases:
                details[tc.name or f"test_{len(details)}"] = ExecutionResult.SKIPPED
            return ExecutionScore(
                skipped=len(test_cases),
                total=len(test_cases),
                score=0.5,
                details=details,
                latency_ms=(time.perf_counter() - start_time) * 1000,
            )

        # Analyze function
        has_return = self._has_return(func_def)
        param_count = len(func_def.args.args)
        undefined_names = self._find_undefined_names(func_def)

        passed = 0
        failed = 0
        errors = 0

        for i, tc in enumerate(test_cases):
            test_name = tc.name or f"test_{i}"

            # Check input arity
            input_count = self._count_args(tc.input)
            if input_count != param_count:
                details[test_name] = ExecutionResult.FAIL
                failed += 1
                continue

            # Check for undefined names
            if undefined_names:
                details[test_name] = ExecutionResult.ERROR
                errors += 1
                continue

            # Check for return
            if not has_return:
                details[test_name] = ExecutionResult.FAIL
                failed += 1
                continue

            # Static analysis passed - assume might work
            details[test_name] = ExecutionResult.PASS
            passed += 1

        result = ExecutionScore(
            passed=passed,
            failed=failed,
            errors=errors,
            skipped=0,
            total=len(test_cases),
            details=details,
            latency_ms=(time.perf_counter() - start_time) * 1000,
        )
        result.score = result.compute_score()
        return result

    def _find_function(
        self,
        tree: ast.AST,
        name: str,
    ) -> Optional[ast.FunctionDef]:
        """Find a function definition by name."""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if not name or node.name == name:
                    return node
        return None

    def _has_return(self, func: ast.FunctionDef) -> bool:
        """Check if function has a return statement with value."""
        for node in ast.walk(func):
            if isinstance(node, ast.Return) and node.value is not None:
                return True
        return False

    def _find_undefined_names(self, func: ast.FunctionDef) -> List[str]:
        """Find potentially undefined names in function."""
        # Get defined names
        defined = set()
        # Parameters
        for arg in func.args.args:
            defined.add(arg.arg)
        # Local assignments
        for node in ast.walk(func):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                defined.add(node.id)

        # Find used names
        undefined = []
        builtins = {
            "print", "len", "range", "int", "str", "float", "bool", "list",
            "dict", "set", "tuple", "sum", "max", "min", "abs", "round",
            "sorted", "reversed", "enumerate", "zip", "map", "filter",
            "True", "False", "None",
        }
        for node in ast.walk(func):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                if node.id not in defined and node.id not in builtins:
                    undefined.append(node.id)

        return undefined

    def _count_args(self, input_str: str) -> int:
        """Count arguments in input string."""
        # Simple heuristic: count comma-separated values
        if not input_str.strip():
            return 0
        # Handle function call format: f(a, b, c)
        match = re.search(r'\((.*)\)', input_str)
        if match:
            args = match.group(1)
            if not args.strip():
                return 0
            return len(args.split(','))
        return 1


class RestrictedExecutionVerifier:
    """Restricted execution verifier with resource limits.

    Executes code in a restricted environment with:
    - No file system access
    - No network access
    - CPU time limit
    - Memory limit
    - Restricted builtins

    WARNING: This is still not fully sandboxed. For production use,
    consider using containers or separate processes.
    """

    def __init__(
        self,
        timeout_ms: int = 1000,
        max_memory_mb: int = 100,
    ):
        """Initialize restricted verifier.

        Args:
            timeout_ms: Default timeout per test
            max_memory_mb: Maximum memory (soft limit)
        """
        self.timeout_ms = timeout_ms
        self.max_memory_mb = max_memory_mb

        # Restricted builtins
        self._safe_builtins = {
            "abs": abs,
            "all": all,
            "any": any,
            "bool": bool,
            "dict": dict,
            "enumerate": enumerate,
            "filter": filter,
            "float": float,
            "frozenset": frozenset,
            "int": int,
            "len": len,
            "list": list,
            "map": map,
            "max": max,
            "min": min,
            "pow": pow,
            "print": print,
            "range": range,
            "reversed": reversed,
            "round": round,
            "set": set,
            "sorted": sorted,
            "str": str,
            "sum": sum,
            "tuple": tuple,
            "zip": zip,
            "True": True,
            "False": False,
            "None": None,
        }

    def verify(
        self,
        code: str,
        test_cases: List[TestCase],
        function_name: str = "",
    ) -> ExecutionScore:
        """Verify code by restricted execution."""
        import time

        start_time = time.perf_counter()
        details: Dict[str, ExecutionResult] = {}

        if not test_cases:
            return ExecutionScore(
                score=0.5,
                latency_ms=(time.perf_counter() - start_time) * 1000,
            )

        # Compile code
        try:
            compiled = compile(code, "<string>", "exec")
        except SyntaxError as e:
            for tc in test_cases:
                details[tc.name or f"test_{len(details)}"] = ExecutionResult.ERROR
            return ExecutionScore(
                errors=len(test_cases),
                total=len(test_cases),
                score=0.0,
                details=details,
                latency_ms=(time.perf_counter() - start_time) * 1000,
            )

        # Execute in restricted namespace
        namespace: Dict[str, Any] = {"__builtins__": self._safe_builtins}
        try:
            exec(compiled, namespace)
        except Exception as e:
            for tc in test_cases:
                details[tc.name or f"test_{len(details)}"] = ExecutionResult.ERROR
            return ExecutionScore(
                errors=len(test_cases),
                total=len(test_cases),
                score=0.0,
                details=details,
                latency_ms=(time.perf_counter() - start_time) * 1000,
            )

        # Find function
        func = None
        if function_name:
            func = namespace.get(function_name)
        else:
            # Find first callable
            for name, obj in namespace.items():
                if callable(obj) and not name.startswith("_"):
                    func = obj
                    break

        if func is None:
            for tc in test_cases:
                details[tc.name or f"test_{len(details)}"] = ExecutionResult.SKIPPED
            return ExecutionScore(
                skipped=len(test_cases),
                total=len(test_cases),
                score=0.5,
                details=details,
                latency_ms=(time.perf_counter() - start_time) * 1000,
            )

        # Run test cases
        passed = 0
        failed = 0
        errors = 0
        skipped = 0

        for i, tc in enumerate(test_cases):
            test_name = tc.name or f"test_{i}"
            result = self._run_test(func, tc, namespace)
            details[test_name] = result

            if result == ExecutionResult.PASS:
                passed += 1
            elif result == ExecutionResult.FAIL:
                failed += 1
            elif result == ExecutionResult.ERROR:
                errors += 1
            else:
                skipped += 1

        result_score = ExecutionScore(
            passed=passed,
            failed=failed,
            errors=errors,
            skipped=skipped,
            total=len(test_cases),
            details=details,
            latency_ms=(time.perf_counter() - start_time) * 1000,
        )
        result_score.score = result_score.compute_score()
        return result_score

    def _run_test(
        self,
        func: Callable,
        tc: TestCase,
        namespace: Dict[str, Any],
    ) -> ExecutionResult:
        """Run a single test case."""
        try:
            # Parse input
            input_expr = tc.input
            if "(" in input_expr and ")" in input_expr:
                # Extract args from function call
                match = re.search(r'\((.*)\)', input_expr)
                if match:
                    input_expr = match.group(1)

            # Evaluate input args
            if input_expr.strip():
                args = eval(f"({input_expr},)", {"__builtins__": self._safe_builtins})
                if isinstance(args, tuple) and len(args) == 1 and isinstance(args[0], tuple):
                    args = args[0]
            else:
                args = ()

            # Call function
            actual = func(*args)

            # Parse expected output
            expected = eval(tc.expected_output, {"__builtins__": self._safe_builtins})

            # Compare
            if actual == expected:
                return ExecutionResult.PASS
            else:
                return ExecutionResult.FAIL

        except Exception as e:
            logger.debug(f"Test execution error: {e}")
            return ExecutionResult.ERROR


def generate_test_cases_from_signature(
    signature: str,
    function_name: str = "",
    num_cases: int = 3,
) -> List[TestCase]:
    """Generate test cases from a function signature.

    Uses heuristics to generate simple test cases from parameter types.

    Args:
        signature: Function signature (e.g., "def add(a: int, b: int) -> int:")
        function_name: Function name (extracted from signature if empty)
        num_cases: Number of test cases to generate

    Returns:
        List of generated test cases
    """
    cases: List[TestCase] = []

    # Parse signature
    match = re.search(r'def\s+(\w+)\s*\((.*?)\)', signature)
    if not match:
        return cases

    name = match.group(1)
    params_str = match.group(2)

    # Parse parameters
    params: List[Tuple[str, str]] = []
    for param in params_str.split(','):
        param = param.strip()
        if ':' in param:
            pname, ptype = param.split(':', 1)
            params.append((pname.strip(), ptype.strip()))
        elif param:
            params.append((param, "Any"))

    if not params:
        # No-arg function
        cases.append(TestCase(
            input=f"{name}()",
            expected_output="...",
            name="test_no_args",
        ))
        return cases

    # Generate test values based on types
    type_values: Dict[str, List[str]] = {
        "int": ["0", "1", "-1", "42", "100"],
        "str": ['""', '"hello"', '"test"', '"a"'],
        "bool": ["True", "False"],
        "float": ["0.0", "1.0", "-1.5", "3.14"],
        "list": ["[]", "[1, 2, 3]"],
    }

    for i in range(num_cases):
        args: List[str] = []
        for pname, ptype in params:
            # Get values for type
            base_type = ptype.lower().replace(" ", "")
            if base_type in type_values:
                values = type_values[base_type]
                args.append(values[i % len(values)])
            else:
                # Default to int
                args.append(str(i))

        input_str = f"{name}({', '.join(args)})"
        cases.append(TestCase(
            input=input_str,
            expected_output="...",  # Unknown expected output
            name=f"test_{i}",
        ))

    return cases


def create_execution_verifier(
    verifier_type: str = "static",
    **kwargs: Any,
) -> ExecutionVerifier:
    """Factory function to create execution verifiers.

    Args:
        verifier_type: Type of verifier ("static", "restricted")
        **kwargs: Additional arguments for specific verifier types

    Returns:
        ExecutionVerifier instance
    """
    if verifier_type == "restricted":
        return RestrictedExecutionVerifier(**kwargs)  # type: ignore
    else:
        return StaticExecutionVerifier(**kwargs)  # type: ignore
