"""
End-to-end tests for Qwen3-Coder-30B deployment on Modal.

Run with:
    modal run deploy/modal/tests/test_qwen3_coder.py

This test suite validates:
    1. Deployment health and readiness
    2. Basic code generation
    3. Ananke constraint enforcement
    4. Multi-language support
    5. Performance benchmarks
"""

import ast
import json
import time
from dataclasses import dataclass
from typing import Any, Optional

import modal

# =============================================================================
# Test Configuration
# =============================================================================

# App and class names for lookup
DEPLOYED_APP_NAME = "qwen3-coder-ananke"
DEPLOYED_CLASS_NAME = "Qwen3CoderAnanke"

TIMEOUT_HEALTH = 30
TIMEOUT_GENERATION = 120
MAX_LATENCY_100_TOKENS = 30  # seconds
MAX_LATENCY_200_TOKENS = 60  # seconds


@dataclass
class TestResult:
    """Result of a single test."""
    name: str
    passed: bool
    duration: float
    message: str
    details: Optional[dict] = None


# =============================================================================
# Test App
# =============================================================================

app = modal.App("qwen3-coder-tests")


@app.function(timeout=600)
def run_all_tests() -> list[dict]:
    """Run all E2E tests against the deployed Qwen3-Coder service."""
    results = []

    # Get reference to the deployed server via lookup
    Qwen3CoderAnanke = modal.Cls.from_name(DEPLOYED_APP_NAME, DEPLOYED_CLASS_NAME)
    server = Qwen3CoderAnanke()

    print("=" * 70)
    print("Qwen3-Coder-30B E2E Test Suite")
    print("=" * 70)

    # ==========================================================================
    # Phase 1: Health & Deployment Tests
    # ==========================================================================
    print("\n[Phase 1] Health & Deployment Tests")
    print("-" * 40)

    # Test 1.1: Liveness
    results.append(test_liveness(server))

    # Test 1.2: Readiness
    results.append(test_readiness(server))

    # Test 1.3: Models endpoint
    results.append(test_models_endpoint(server))

    # ==========================================================================
    # Phase 2: Basic Generation Tests
    # ==========================================================================
    print("\n[Phase 2] Basic Generation Tests")
    print("-" * 40)

    # Test 2.1: Simple Python completion
    results.append(test_simple_generation(server))

    # Test 2.2: Function completion
    results.append(test_function_completion(server))

    # Test 2.3: Chat completion
    results.append(test_chat_completion(server))

    # ==========================================================================
    # Phase 3: Ananke Constraint Tests
    # ==========================================================================
    print("\n[Phase 3] Ananke Constraint Tests")
    print("-" * 40)

    # Test 3.1: Syntax constraint
    results.append(test_syntax_constraint(server))

    # Test 3.2: Type constraint
    results.append(test_type_constraint(server))

    # Test 3.3: Multiple domains
    results.append(test_multiple_domains(server))

    # ==========================================================================
    # Phase 4: Code Quality Tests
    # ==========================================================================
    print("\n[Phase 4] Code Quality Tests")
    print("-" * 40)

    # Test 4.1: Recursive function
    results.append(test_recursive_function(server))

    # Test 4.2: Class generation
    results.append(test_class_generation(server))

    # ==========================================================================
    # Phase 5: Performance Tests
    # ==========================================================================
    print("\n[Phase 5] Performance Tests")
    print("-" * 40)

    # Test 5.1: Latency benchmark
    results.append(test_latency_benchmark(server))

    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed

    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {r.name} ({r.duration:.2f}s)")
        if not r.passed:
            print(f"         -> {r.message}")

    print("-" * 70)
    print(f"Results: {passed}/{len(results)} passed")

    if failed > 0:
        print(f"\nFAILED TESTS: {failed}")
        raise AssertionError(f"{failed} tests failed")

    print("\nAll tests passed!")
    return [_result_to_dict(r) for r in results]


def _result_to_dict(r: TestResult) -> dict:
    return {
        "name": r.name,
        "passed": r.passed,
        "duration": r.duration,
        "message": r.message,
        "details": r.details,
    }


# =============================================================================
# Phase 1: Health & Deployment Tests
# =============================================================================

def test_liveness(server: Any) -> TestResult:
    """Test that the server is alive."""
    name = "1.1 Liveness Check"
    start = time.time()

    try:
        result = server.health.remote()
        duration = time.time() - start

        if result.get("status") == "healthy":
            print(f"  [PASS] {name}")
            return TestResult(name, True, duration, "Server is healthy")
        else:
            print(f"  [FAIL] {name}: {result}")
            return TestResult(name, False, duration, f"Unhealthy: {result}")

    except Exception as e:
        duration = time.time() - start
        print(f"  [FAIL] {name}: {e}")
        return TestResult(name, False, duration, str(e))


def test_readiness(server: Any) -> TestResult:
    """Test that the server is ready to generate."""
    name = "1.2 Readiness Check"
    start = time.time()

    try:
        result = server.health_generate.remote()
        duration = time.time() - start

        if result.get("ready"):
            print(f"  [PASS] {name}")
            return TestResult(name, True, duration, "Server is ready")
        else:
            print(f"  [FAIL] {name}: {result}")
            return TestResult(name, False, duration, f"Not ready: {result}")

    except Exception as e:
        duration = time.time() - start
        print(f"  [FAIL] {name}: {e}")
        return TestResult(name, False, duration, str(e))


def test_models_endpoint(server: Any) -> TestResult:
    """Test the models endpoint."""
    name = "1.3 Models Endpoint"
    start = time.time()

    try:
        result = server.get_models.remote()
        duration = time.time() - start

        models = result.get("data", [])
        if len(models) > 0:
            model_ids = [m.get("id") for m in models]
            print(f"  [PASS] {name}: {model_ids}")
            return TestResult(
                name, True, duration,
                f"Found models: {model_ids}",
                {"models": model_ids}
            )
        else:
            print(f"  [FAIL] {name}: No models found")
            return TestResult(name, False, duration, "No models found")

    except Exception as e:
        duration = time.time() - start
        print(f"  [FAIL] {name}: {e}")
        return TestResult(name, False, duration, str(e))


# =============================================================================
# Phase 2: Basic Generation Tests
# =============================================================================

def test_simple_generation(server: Any) -> TestResult:
    """Test simple code generation."""
    name = "2.1 Simple Generation"
    start = time.time()

    try:
        result = server.generate.remote(
            prompt="def hello_world():\n    ",
            max_tokens=50,
        )
        duration = time.time() - start

        if result and len(result.strip()) > 0:
            print(f"  [PASS] {name}")
            return TestResult(
                name, True, duration,
                "Generated code successfully",
                {"output_length": len(result)}
            )
        else:
            print(f"  [FAIL] {name}: Empty result")
            return TestResult(name, False, duration, "Empty generation")

    except Exception as e:
        duration = time.time() - start
        print(f"  [FAIL] {name}: {e}")
        return TestResult(name, False, duration, str(e))


def test_function_completion(server: Any) -> TestResult:
    """Test function body completion."""
    name = "2.2 Function Completion"
    start = time.time()

    try:
        result = server.generate.remote(
            prompt="def fibonacci(n: int) -> int:\n    ",
            max_tokens=100,
        )
        duration = time.time() - start

        # Check if result contains typical fibonacci patterns
        has_base_case = "return" in result or "if" in result
        has_recursion_or_loop = "fibonacci" in result or "for" in result or "while" in result

        if has_base_case:
            print(f"  [PASS] {name}")
            return TestResult(
                name, True, duration,
                "Generated fibonacci implementation",
                {"has_recursion": has_recursion_or_loop}
            )
        else:
            print(f"  [FAIL] {name}: No valid implementation")
            return TestResult(
                name, False, duration,
                f"Invalid implementation: {result[:100]}"
            )

    except Exception as e:
        duration = time.time() - start
        print(f"  [FAIL] {name}: {e}")
        return TestResult(name, False, duration, str(e))


def test_chat_completion(server: Any) -> TestResult:
    """Test chat-based code generation."""
    name = "2.3 Chat Completion"
    start = time.time()

    try:
        result = server.chat.remote(
            messages=[
                {"role": "user", "content": "Write a Python function to reverse a string."}
            ],
            max_tokens=150,
        )
        duration = time.time() - start

        content = result.get("content", "")
        has_def = "def" in content
        has_return = "return" in content

        if has_def and has_return:
            print(f"  [PASS] {name}")
            return TestResult(
                name, True, duration,
                "Generated function via chat",
                {"output_length": len(content)}
            )
        else:
            print(f"  [FAIL] {name}: No function in response")
            return TestResult(
                name, False, duration,
                f"Missing function definition: {content[:100]}"
            )

    except Exception as e:
        duration = time.time() - start
        print(f"  [FAIL] {name}: {e}")
        return TestResult(name, False, duration, str(e))


# =============================================================================
# Phase 3: Ananke Constraint Tests
# =============================================================================

def test_syntax_constraint(server: Any) -> TestResult:
    """Test syntax-only constraint produces valid Python."""
    name = "3.1 Syntax Constraint"
    start = time.time()

    try:
        result = server.generate_constrained.remote(
            prompt="def add(a, b):\n    ",
            constraint_spec={
                "language": "python",
                "domains": ["syntax"],
            },
            max_tokens=50,
        )
        duration = time.time() - start

        text = result.get("text", "")
        full_code = f"def add(a, b):\n    {text}"

        # Try to parse as Python
        try:
            ast.parse(full_code)
            is_valid_python = True
        except SyntaxError:
            is_valid_python = False

        if is_valid_python:
            print(f"  [PASS] {name}")
            return TestResult(
                name, True, duration,
                "Generated valid Python syntax",
                {"valid_syntax": True}
            )
        else:
            print(f"  [FAIL] {name}: Invalid syntax")
            return TestResult(
                name, False, duration,
                f"Invalid Python syntax: {text[:100]}"
            )

    except Exception as e:
        duration = time.time() - start
        print(f"  [FAIL] {name}: {e}")
        return TestResult(name, False, duration, str(e))


def test_type_constraint(server: Any) -> TestResult:
    """Test type-aware constraint."""
    name = "3.2 Type Constraint"
    start = time.time()

    try:
        result = server.generate_constrained.remote(
            prompt="def multiply(a: int, b: int) -> int:\n    ",
            constraint_spec={
                "language": "python",
                "domains": ["syntax", "types"],
            },
            max_tokens=50,
        )
        duration = time.time() - start

        text = result.get("text", "")

        # Should return an int
        if "return" in text:
            print(f"  [PASS] {name}")
            return TestResult(
                name, True, duration,
                "Generated type-aware code",
                {"has_return": True}
            )
        else:
            print(f"  [FAIL] {name}: No return statement")
            return TestResult(
                name, False, duration,
                f"Missing return: {text[:100]}"
            )

    except Exception as e:
        duration = time.time() - start
        print(f"  [FAIL] {name}: {e}")
        return TestResult(name, False, duration, str(e))


def test_multiple_domains(server: Any) -> TestResult:
    """Test multiple constraint domains."""
    name = "3.3 Multiple Domains"
    start = time.time()

    try:
        result = server.generate_constrained.remote(
            prompt="from typing import List\n\ndef process(items: List[str]) -> int:\n    ",
            constraint_spec={
                "language": "python",
                "domains": ["syntax", "types", "imports"],
            },
            max_tokens=100,
        )
        duration = time.time() - start

        text = result.get("text", "")
        full_code = f"from typing import List\n\ndef process(items: List[str]) -> int:\n    {text}"

        # Try to parse
        try:
            ast.parse(full_code)
            is_valid = True
        except SyntaxError:
            is_valid = False

        if is_valid:
            print(f"  [PASS] {name}")
            return TestResult(
                name, True, duration,
                "Multiple domains work together",
                {"valid": True}
            )
        else:
            print(f"  [FAIL] {name}: Invalid output")
            return TestResult(name, False, duration, f"Invalid: {text[:100]}")

    except Exception as e:
        duration = time.time() - start
        print(f"  [FAIL] {name}: {e}")
        return TestResult(name, False, duration, str(e))


# =============================================================================
# Phase 4: Code Quality Tests
# =============================================================================

def test_recursive_function(server: Any) -> TestResult:
    """Test generation of recursive function."""
    name = "4.1 Recursive Function"
    start = time.time()

    try:
        result = server.generate.remote(
            prompt="def factorial(n: int) -> int:\n    \"\"\"Calculate factorial recursively.\"\"\"\n    ",
            max_tokens=100,
        )
        duration = time.time() - start

        # Check for recursion pattern
        has_base_case = "if" in result and ("0" in result or "1" in result)
        has_recursive_call = "factorial" in result

        if has_base_case and has_recursive_call:
            print(f"  [PASS] {name}")
            return TestResult(
                name, True, duration,
                "Generated recursive implementation",
                {"has_base_case": has_base_case, "has_recursion": has_recursive_call}
            )
        elif has_base_case:
            # Iterative is also acceptable
            print(f"  [PASS] {name} (iterative)")
            return TestResult(
                name, True, duration,
                "Generated iterative implementation",
                {"iterative": True}
            )
        else:
            print(f"  [FAIL] {name}: Incomplete implementation")
            return TestResult(
                name, False, duration,
                f"Incomplete: {result[:100]}"
            )

    except Exception as e:
        duration = time.time() - start
        print(f"  [FAIL] {name}: {e}")
        return TestResult(name, False, duration, str(e))


def test_class_generation(server: Any) -> TestResult:
    """Test class generation."""
    name = "4.2 Class Generation"
    start = time.time()

    try:
        result = server.chat.remote(
            messages=[
                {"role": "user", "content": "Write a Python class called Counter with increment() and get_value() methods."}
            ],
            max_tokens=300,
        )
        duration = time.time() - start

        content = result.get("content", "")

        has_class = "class Counter" in content or "class counter" in content.lower()
        has_init = "__init__" in content or "def __init__" in content
        has_methods = "def increment" in content.lower() or "def get_value" in content.lower()

        if has_class and has_methods:
            print(f"  [PASS] {name}")
            return TestResult(
                name, True, duration,
                "Generated valid class",
                {"has_init": has_init, "has_methods": has_methods}
            )
        else:
            print(f"  [FAIL] {name}: Incomplete class")
            return TestResult(
                name, False, duration,
                f"Missing class/methods: {content[:150]}"
            )

    except Exception as e:
        duration = time.time() - start
        print(f"  [FAIL] {name}: {e}")
        return TestResult(name, False, duration, str(e))


# =============================================================================
# Phase 5: Performance Tests
# =============================================================================

def test_latency_benchmark(server: Any) -> TestResult:
    """Benchmark generation latency."""
    name = "5.1 Latency Benchmark"
    start = time.time()

    try:
        # Generate 100 tokens
        gen_start = time.time()
        result = server.generate.remote(
            prompt="def quicksort(arr):\n    ",
            max_tokens=100,
        )
        gen_duration = time.time() - gen_start

        duration = time.time() - start

        if gen_duration <= MAX_LATENCY_100_TOKENS:
            print(f"  [PASS] {name}: {gen_duration:.2f}s for 100 tokens")
            return TestResult(
                name, True, duration,
                f"Latency: {gen_duration:.2f}s (target: <{MAX_LATENCY_100_TOKENS}s)",
                {"latency_100_tokens": gen_duration}
            )
        else:
            print(f"  [FAIL] {name}: {gen_duration:.2f}s exceeds {MAX_LATENCY_100_TOKENS}s")
            return TestResult(
                name, False, duration,
                f"Latency {gen_duration:.2f}s exceeds {MAX_LATENCY_100_TOKENS}s target",
                {"latency_100_tokens": gen_duration}
            )

    except Exception as e:
        duration = time.time() - start
        print(f"  [FAIL] {name}: {e}")
        return TestResult(name, False, duration, str(e))


# =============================================================================
# CLI Entry Point
# =============================================================================

@app.local_entrypoint()
def main():
    """Run the E2E test suite."""
    results = run_all_tests.remote()
    print(f"\nReturned {len(results)} test results")


if __name__ == "__main__":
    main()
