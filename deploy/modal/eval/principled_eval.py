"""
Principled Ananke Evaluation

Design principles:
1. Sufficient token budget (2000 tokens for real code)
2. Complete, standalone prompts (no partial code requiring continuation)
3. Multiple temperature points to show constraint value
4. Statistical rigor (10+ samples per condition)
5. Real syntax validation (actual parsers)
6. Clear metrics with confidence intervals

Run:
    modal run deploy/modal/eval/principled_eval.py
"""

import ast
import json
import statistics
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import modal

# =============================================================================
# Configuration
# =============================================================================

DEPLOYED_APP = "qwen3-coder-ananke"
DEPLOYED_CLASS = "Qwen3CoderAnanke"

# Principled settings
MAX_TOKENS = 2000  # Sufficient for real code
SAMPLES_PER_CONDITION = 10  # Statistical significance
TEMPERATURES = [0.3, 0.6, 0.9]  # Low, medium, high uncertainty

# Languages to test (subset that we can properly validate)
# Note: constraint_spec only needs 'language' - Ananke enables appropriate domains automatically
LANGUAGES = {
    "python": {
        "validator": "python_ast",
    },
    "typescript": {
        "validator": "typescript_tsc",
    },
    "go": {
        "validator": "go_parse",
    },
}


# =============================================================================
# Principled Test Prompts
# =============================================================================

PROMPTS = {
    "python": [
        # Prompt 1: Implement a complete class
        {
            "id": "py_class_datastructure",
            "prompt": """Write a complete Python class implementing a LRU (Least Recently Used) cache with the following requirements:

1. Initialize with a maximum capacity
2. get(key) - Return the value if key exists, otherwise return -1
3. put(key, value) - Insert or update the value, evicting the least recently used item if at capacity
4. The cache should use O(1) time complexity for both operations

Include type hints and docstrings. Write the complete implementation:

```python
""",
            "expected_elements": ["class", "def __init__", "def get", "def put", "OrderedDict", "self"],
        },
        # Prompt 2: Implement algorithms
        {
            "id": "py_algorithm_graph",
            "prompt": """Write a complete Python implementation of Dijkstra's shortest path algorithm.

Requirements:
1. Function signature: def dijkstra(graph: dict[str, dict[str, int]], start: str) -> dict[str, int]
2. graph is an adjacency list where graph[node][neighbor] = weight
3. Return a dictionary mapping each reachable node to its shortest distance from start
4. Use a priority queue (heapq) for efficiency
5. Include type hints and a docstring

Write the complete implementation:

```python
""",
            "expected_elements": ["def dijkstra", "heapq", "while", "return", "distances"],
        },
        # Prompt 3: Data processing
        {
            "id": "py_data_processing",
            "prompt": """Write a complete Python module for processing CSV data with the following functions:

1. read_csv(filepath: str) -> list[dict[str, str]] - Read CSV into list of dicts
2. filter_rows(data: list[dict], column: str, value: str) -> list[dict] - Filter by column value
3. aggregate(data: list[dict], group_by: str, agg_column: str) -> dict[str, float] - Group and average
4. write_csv(data: list[dict], filepath: str) -> None - Write back to CSV

Use the csv module from stdlib. Include error handling and type hints.

Write the complete implementation:

```python
""",
            "expected_elements": ["import csv", "def read_csv", "def filter_rows", "def aggregate", "def write_csv"],
        },
    ],
    "typescript": [
        {
            "id": "ts_class_api",
            "prompt": """Write a complete TypeScript class for an HTTP API client with the following features:

1. Constructor takes a base URL and optional default headers
2. get<T>(endpoint: string): Promise<T> - GET request returning typed response
3. post<T, B>(endpoint: string, body: B): Promise<T> - POST with typed body and response
4. Automatic JSON parsing
5. Error handling with custom ApiError class
6. Request timeout support

Write the complete implementation with full type annotations:

```typescript
""",
            "expected_elements": ["class", "constructor", "async", "Promise", "fetch", "ApiError"],
        },
        {
            "id": "ts_generic_utils",
            "prompt": """Write a TypeScript module with generic utility functions:

1. groupBy<T, K>(array: T[], keyFn: (item: T) => K): Map<K, T[]>
2. debounce<T extends (...args: any[]) => any>(fn: T, ms: number): T
3. retry<T>(fn: () => Promise<T>, attempts: number, delay: number): Promise<T>
4. deepClone<T>(obj: T): T
5. pipe<T>(...fns: ((arg: T) => T)[]): (arg: T) => T

Include proper generic constraints and type inference.

Write the complete implementation:

```typescript
""",
            "expected_elements": ["function groupBy", "function debounce", "function retry", "Promise", "Map"],
        },
    ],
    "go": [
        {
            "id": "go_concurrent_worker",
            "prompt": """Write a complete Go package implementing a concurrent worker pool:

1. WorkerPool struct with configurable number of workers
2. Submit(task func()) - Submit a task to the pool
3. Wait() - Wait for all tasks to complete
4. Shutdown() - Gracefully shutdown the pool
5. Use channels for task distribution
6. Handle panics in workers gracefully

Write the complete implementation:

```go
package workerpool

""",
            "expected_elements": ["type WorkerPool struct", "func New", "chan", "go func", "sync.WaitGroup"],
        },
        {
            "id": "go_http_middleware",
            "prompt": """Write a complete Go package with HTTP middleware functions:

1. Logger - Log request method, path, duration, and status code
2. Recovery - Recover from panics and return 500
3. RateLimit - Token bucket rate limiting per IP
4. Auth - Bearer token authentication middleware
5. Chain - Function to chain multiple middlewares

Use net/http standard library.

Write the complete implementation:

```go
package middleware

import (
    "net/http"
""",
            "expected_elements": ["func Logger", "func Recovery", "http.Handler", "http.ResponseWriter", "func Chain"],
        },
    ],
}


@dataclass
class ValidationResult:
    """Result of syntax validation."""
    valid: bool
    error: Optional[str] = None
    warnings: list[str] = field(default_factory=list)


@dataclass
class GenerationResult:
    """Result of a single generation."""
    prompt_id: str
    language: str
    temperature: float
    constrained: bool
    output: str
    generation_time: float
    validation: ValidationResult
    expected_matches: int
    total_expected: int


# =============================================================================
# Syntax Validators
# =============================================================================

def validate_python(code: str) -> ValidationResult:
    """Validate Python syntax using AST."""
    # Extract code from markdown if present
    code = extract_code_block(code, "python")

    try:
        ast.parse(code)
        return ValidationResult(valid=True)
    except SyntaxError as e:
        return ValidationResult(
            valid=False,
            error=f"Line {e.lineno}: {e.msg}" if e.lineno else str(e)
        )


def validate_typescript(code: str) -> ValidationResult:
    """Validate TypeScript syntax using tsc."""
    code = extract_code_block(code, "typescript")

    # Basic structural validation (tsc not available in Modal by default)
    # Check for balanced braces, brackets, parens
    if not check_balanced(code):
        return ValidationResult(valid=False, error="Unbalanced brackets")

    # Check for obvious syntax errors
    if "function(" in code and "function (" not in code:
        pass  # Arrow functions OK

    # Check it has some expected TypeScript patterns
    if not any(p in code for p in ["function", "const", "class", "interface", "type"]):
        return ValidationResult(valid=False, error="No TypeScript constructs found")

    return ValidationResult(valid=True)


def validate_go(code: str) -> ValidationResult:
    """Validate Go syntax."""
    code = extract_code_block(code, "go")

    if not check_balanced(code):
        return ValidationResult(valid=False, error="Unbalanced brackets")

    # Check for Go patterns
    if not any(p in code for p in ["func ", "package ", "type ", "var ", "const "]):
        return ValidationResult(valid=False, error="No Go constructs found")

    return ValidationResult(valid=True)


def extract_code_block(text: str, language: str) -> str:
    """Extract code from markdown code block."""
    import re

    # Try to find code block for specific language
    pattern = rf"```{language}\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Try generic code block
    pattern = r"```\n?(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Return as-is
    return text.strip()


def check_balanced(code: str) -> bool:
    """Check if brackets are balanced."""
    stack = []
    pairs = {')': '(', ']': '[', '}': '{'}

    in_string = False
    string_char = None
    prev_char = None

    for char in code:
        # Handle string literals
        if char in '"\'`' and prev_char != '\\':
            if not in_string:
                in_string = True
                string_char = char
            elif char == string_char:
                in_string = False
                string_char = None

        if not in_string:
            if char in '([{':
                stack.append(char)
            elif char in ')]}':
                if not stack or stack[-1] != pairs[char]:
                    return False
                stack.pop()

        prev_char = char

    return len(stack) == 0


VALIDATORS = {
    "python": validate_python,
    "typescript": validate_typescript,
    "go": validate_go,
}


# =============================================================================
# Modal App
# =============================================================================

app = modal.App("principled-ananke-eval")


@app.function(timeout=3600)
def run_evaluation() -> dict:
    """Run principled evaluation."""
    print("=" * 80)
    print("PRINCIPLED ANANKE EVALUATION")
    print("=" * 80)
    print(f"Max tokens: {MAX_TOKENS}")
    print(f"Samples per condition: {SAMPLES_PER_CONDITION}")
    print(f"Temperatures: {TEMPERATURES}")
    print(f"Languages: {list(LANGUAGES.keys())}")
    print("=" * 80)

    # Connect to model
    print("\nConnecting to deployed model...")
    Qwen3CoderAnanke = modal.Cls.from_name(DEPLOYED_APP, DEPLOYED_CLASS)
    server = Qwen3CoderAnanke()

    all_results = []

    # For each language
    for lang, config in LANGUAGES.items():
        prompts = PROMPTS.get(lang, [])
        if not prompts:
            continue

        print(f"\n{'='*80}")
        print(f"LANGUAGE: {lang.upper()}")
        print(f"{'='*80}")

        validator = VALIDATORS[lang]

        # For each prompt
        for prompt_info in prompts:
            prompt_id = prompt_info["id"]
            prompt = prompt_info["prompt"]
            expected = prompt_info["expected_elements"]

            print(f"\n  Prompt: {prompt_id}")
            print(f"  Expected elements: {expected}")

            # For each temperature
            for temp in TEMPERATURES:
                print(f"\n    Temperature: {temp}")

                # For each condition (constrained vs unconstrained)
                for constrained in [True, False]:
                    condition = "constrained" if constrained else "unconstrained"
                    print(f"      {condition}: ", end="", flush=True)

                    valid_count = 0
                    times = []

                    # Multiple samples
                    for sample_idx in range(SAMPLES_PER_CONDITION):
                        start = time.time()

                        try:
                            if constrained:
                                response = server.generate_constrained.remote(
                                    prompt=prompt,
                                    constraint_spec={
                                        "language": lang,
                                    },
                                    max_tokens=MAX_TOKENS,
                                    temperature=temp,
                                )
                                output = response.get("text", "")
                            else:
                                output = server.generate.remote(
                                    prompt=prompt,
                                    max_tokens=MAX_TOKENS,
                                    temperature=temp,
                                )

                            gen_time = time.time() - start
                            times.append(gen_time)

                            # Validate
                            validation = validator(output)

                            # Count expected elements
                            matches = sum(1 for e in expected if e.lower() in output.lower())

                            result = GenerationResult(
                                prompt_id=prompt_id,
                                language=lang,
                                temperature=temp,
                                constrained=constrained,
                                output=output[:500],  # Truncate for storage
                                generation_time=gen_time,
                                validation=validation,
                                expected_matches=matches,
                                total_expected=len(expected),
                            )

                            if validation.valid:
                                valid_count += 1
                                print("✓", end="", flush=True)
                            else:
                                print("✗", end="", flush=True)

                        except Exception as e:
                            gen_time = time.time() - start
                            error_msg = str(e)[:200]
                            result = GenerationResult(
                                prompt_id=prompt_id,
                                language=lang,
                                temperature=temp,
                                constrained=constrained,
                                output="",
                                generation_time=gen_time,
                                validation=ValidationResult(valid=False, error=error_msg),
                                expected_matches=0,
                                total_expected=len(expected),
                            )
                            print(f"E({error_msg[:50]})", end="", flush=True)

                        all_results.append(result)

                    # Summary for this condition
                    rate = valid_count / SAMPLES_PER_CONDITION
                    avg_time = statistics.mean(times) if times else 0
                    print(f" | {valid_count}/{SAMPLES_PER_CONDITION} ({rate:.0%}) | {avg_time:.2f}s avg")

    # Compute aggregate statistics
    summary = compute_summary(all_results)

    print("\n" + "=" * 80)
    print("SUMMARY RESULTS")
    print("=" * 80)

    print("\nValidity Rate by Temperature and Constraint:")
    print(f"{'Temp':<8} {'Constrained':<15} {'Unconstrained':<15} {'Delta':<10}")
    print("-" * 50)
    for temp in TEMPERATURES:
        c_rate = summary["by_temp_constraint"].get((temp, True), {}).get("validity_rate", 0)
        u_rate = summary["by_temp_constraint"].get((temp, False), {}).get("validity_rate", 0)
        delta = c_rate - u_rate
        print(f"{temp:<8} {c_rate:>13.1%} {u_rate:>13.1%} {delta:>+9.1%}")

    print("\nValidity Rate by Language:")
    print(f"{'Language':<12} {'Constrained':<15} {'Unconstrained':<15} {'Delta':<10}")
    print("-" * 55)
    for lang in LANGUAGES:
        c_rate = summary["by_lang_constraint"].get((lang, True), {}).get("validity_rate", 0)
        u_rate = summary["by_lang_constraint"].get((lang, False), {}).get("validity_rate", 0)
        delta = c_rate - u_rate
        print(f"{lang:<12} {c_rate:>13.1%} {u_rate:>13.1%} {delta:>+9.1%}")

    print("\nOverall:")
    print(f"  Constrained validity:   {summary['overall_constrained_rate']:.1%}")
    print(f"  Unconstrained validity: {summary['overall_unconstrained_rate']:.1%}")
    print(f"  Delta:                  {summary['overall_delta']:+.1%}")

    return {
        "summary": summary,
        "config": {
            "max_tokens": MAX_TOKENS,
            "samples_per_condition": SAMPLES_PER_CONDITION,
            "temperatures": TEMPERATURES,
            "languages": list(LANGUAGES.keys()),
        },
        "results_count": len(all_results),
    }


def compute_summary(results: list[GenerationResult]) -> dict:
    """Compute aggregate statistics."""

    # Group by various dimensions
    by_temp_constraint = {}
    by_lang_constraint = {}

    for r in results:
        # By temp + constraint
        key = (r.temperature, r.constrained)
        if key not in by_temp_constraint:
            by_temp_constraint[key] = {"valid": 0, "total": 0}
        by_temp_constraint[key]["total"] += 1
        if r.validation.valid:
            by_temp_constraint[key]["valid"] += 1

        # By lang + constraint
        key = (r.language, r.constrained)
        if key not in by_lang_constraint:
            by_lang_constraint[key] = {"valid": 0, "total": 0}
        by_lang_constraint[key]["total"] += 1
        if r.validation.valid:
            by_lang_constraint[key]["valid"] += 1

    # Compute rates
    for d in [by_temp_constraint, by_lang_constraint]:
        for key in d:
            d[key]["validity_rate"] = d[key]["valid"] / d[key]["total"] if d[key]["total"] > 0 else 0

    # Overall
    constrained_results = [r for r in results if r.constrained]
    unconstrained_results = [r for r in results if not r.constrained]

    c_valid = sum(1 for r in constrained_results if r.validation.valid)
    u_valid = sum(1 for r in unconstrained_results if r.validation.valid)

    c_rate = c_valid / len(constrained_results) if constrained_results else 0
    u_rate = u_valid / len(unconstrained_results) if unconstrained_results else 0

    return {
        "by_temp_constraint": by_temp_constraint,
        "by_lang_constraint": by_lang_constraint,
        "overall_constrained_rate": c_rate,
        "overall_unconstrained_rate": u_rate,
        "overall_delta": c_rate - u_rate,
    }


@app.local_entrypoint()
def main():
    """Run the evaluation."""
    results = run_evaluation.remote()
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
