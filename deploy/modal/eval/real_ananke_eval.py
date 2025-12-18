"""Real Ananke Evaluation - Proper constraint testing.

This evaluation actually uses Ananke's constraint features:
1. Regex constraints for syntax enforcement
2. Type bindings for type checking
3. Verification that constraints are being enforced

Run:
    modal run deploy/modal/eval/real_ananke_eval.py
"""

import ast
import json
import re
import time
from dataclasses import dataclass
from typing import Any, Optional

import modal

# =============================================================================
# Configuration
# =============================================================================

DEPLOYED_APP = "qwen3-coder-ananke"
DEPLOYED_CLASS = "Qwen3CoderAnanke"

MAX_TOKENS = 500
SAMPLES_PER_TEST = 5

app = modal.App("real-ananke-eval")


# =============================================================================
# Test Cases with REAL Constraints
# =============================================================================

TEST_CASES = [
    # Test 1: Regex constraint - must start with "def " and have type hints
    {
        "name": "regex_function_signature",
        "description": "Force function with type hints via regex",
        "prompt": "Write a function to calculate factorial:\n\n",
        "constraint_spec": {
            "language": "python",
            # Regex: function starting with 'def', having '->' return type
            "regex": r"def \w+\([^)]*\) -> \w+:\n    .*",
        },
        "validation": lambda text: (
            text.strip().startswith("def ") and
            "->" in text and
            ":" in text
        ),
        "validation_name": "starts with 'def', contains '->' type hint",
    },

    # Test 2: JSON schema constraint - structured output
    {
        "name": "json_schema_output",
        "description": "Force JSON object output",
        "prompt": "Generate a JSON object describing a user with name and age:\n\n",
        "constraint_spec": {
            "language": "python",  # Still need language for context
            "json_schema": json.dumps({
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"}
                },
                "required": ["name", "age"]
            }),
        },
        "validation": lambda text: _validate_json_schema(text, ["name", "age"]),
        "validation_name": "valid JSON with name and age fields",
    },

    # Test 3: Type bindings - provide type context
    {
        "name": "type_context_completion",
        "description": "Complete code with type context provided",
        "prompt": "# Given: items is List[int], threshold is int\n# Return items greater than threshold\ndef filter_above(items, threshold):\n    return ",
        "constraint_spec": {
            "language": "python",
            "type_bindings": [
                {"name": "items", "type_expr": "List[int]"},
                {"name": "threshold", "type_expr": "int"},
            ],
            "expected_type": "List[int]",
            "enabled_domains": ["syntax", "types"],
        },
        "validation": lambda text: (
            "[" in text and  # Should use list comprehension or filter
            "for" in text.lower() or "filter" in text.lower() or "[x" in text
        ),
        "validation_name": "returns list expression",
    },

    # Test 4: Import constraints - restrict available modules
    {
        "name": "import_constraints",
        "description": "Generate code with import restrictions",
        "prompt": "Write a function to make an HTTP request:\n\n",
        "constraint_spec": {
            "language": "python",
            "available_modules": ["requests", "urllib", "http"],
            "forbidden_imports": ["os", "subprocess", "sys"],
            "enabled_domains": ["syntax", "imports"],
        },
        "validation": lambda text: (
            ("requests" in text or "urllib" in text or "http" in text) and
            "subprocess" not in text and
            "os.system" not in text
        ),
        "validation_name": "uses allowed modules, no forbidden imports",
    },

    # Test 5: Control comparison - same prompt with and without constraints
    {
        "name": "constraint_enforcement_test",
        "description": "Verify constraints actually change output",
        "prompt": "Output: ",
        "constraint_spec": {
            "language": "python",
            "regex": r"[0-9]+",  # Only digits allowed
        },
        "validation": lambda text: text.strip().isdigit() if text.strip() else False,
        "validation_name": "output is only digits",
    },
]


def _validate_json_schema(text: str, required_fields: list) -> bool:
    """Validate text is valid JSON with required fields."""
    try:
        # Try to find JSON in the text
        text = text.strip()
        if not text.startswith("{"):
            # Look for JSON object
            start = text.find("{")
            if start == -1:
                return False
            text = text[start:]

        # Find matching closing brace
        depth = 0
        end = 0
        for i, c in enumerate(text):
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break

        if end == 0:
            return False

        json_str = text[:end]
        obj = json.loads(json_str)

        return all(field in obj for field in required_fields)
    except (json.JSONDecodeError, ValueError):
        return False


@dataclass
class TestResult:
    name: str
    constrained_valid: int
    constrained_total: int
    unconstrained_valid: int
    unconstrained_total: int
    constrained_time: float
    unconstrained_time: float
    sample_constrained: str
    sample_unconstrained: str


# =============================================================================
# Evaluation Function
# =============================================================================

@app.function(timeout=1800)
def run_evaluation() -> dict:
    """Run the real Ananke evaluation."""
    print("=" * 70)
    print("REAL ANANKE EVALUATION")
    print("=" * 70)
    print(f"Max tokens: {MAX_TOKENS}")
    print(f"Samples per test: {SAMPLES_PER_TEST}")
    print("=" * 70)

    # Connect to deployed model
    print("\nConnecting to deployed model...")
    Qwen3CoderAnanke = modal.Cls.from_name(DEPLOYED_APP, DEPLOYED_CLASS)
    server = Qwen3CoderAnanke()

    # Verify connection
    health = server.health.remote()
    print(f"Connected: {health}")

    results = []

    for test in TEST_CASES:
        print(f"\n{'='*70}")
        print(f"TEST: {test['name']}")
        print(f"Description: {test['description']}")
        print(f"Validation: {test['validation_name']}")
        print(f"{'='*70}")

        constrained_valid = 0
        unconstrained_valid = 0
        constrained_times = []
        unconstrained_times = []
        sample_constrained = ""
        sample_unconstrained = ""

        # Run with constraints
        print("\n  Constrained generation:")
        for i in range(SAMPLES_PER_TEST):
            start = time.time()
            try:
                response = server.generate_constrained.remote(
                    prompt=test["prompt"],
                    constraint_spec=test["constraint_spec"],
                    max_tokens=MAX_TOKENS,
                    temperature=0.7,
                )
                output = response.get("text", "")
                elapsed = time.time() - start
                constrained_times.append(elapsed)

                valid = test["validation"](output)
                if valid:
                    constrained_valid += 1
                    print(f"    [{i+1}] VALID ({elapsed:.2f}s)")
                else:
                    print(f"    [{i+1}] INVALID ({elapsed:.2f}s)")

                if i == 0:
                    sample_constrained = output[:200]

            except Exception as e:
                print(f"    [{i+1}] ERROR: {str(e)[:50]}")

        # Run without constraints (baseline)
        print("\n  Unconstrained generation:")
        for i in range(SAMPLES_PER_TEST):
            start = time.time()
            try:
                output = server.generate.remote(
                    prompt=test["prompt"],
                    max_tokens=MAX_TOKENS,
                    temperature=0.7,
                )
                elapsed = time.time() - start
                unconstrained_times.append(elapsed)

                valid = test["validation"](output)
                if valid:
                    unconstrained_valid += 1
                    print(f"    [{i+1}] VALID ({elapsed:.2f}s)")
                else:
                    print(f"    [{i+1}] INVALID ({elapsed:.2f}s)")

                if i == 0:
                    sample_unconstrained = output[:200]

            except Exception as e:
                print(f"    [{i+1}] ERROR: {str(e)[:50]}")

        result = TestResult(
            name=test["name"],
            constrained_valid=constrained_valid,
            constrained_total=SAMPLES_PER_TEST,
            unconstrained_valid=unconstrained_valid,
            unconstrained_total=SAMPLES_PER_TEST,
            constrained_time=sum(constrained_times) / len(constrained_times) if constrained_times else 0,
            unconstrained_time=sum(unconstrained_times) / len(unconstrained_times) if unconstrained_times else 0,
            sample_constrained=sample_constrained,
            sample_unconstrained=sample_unconstrained,
        )
        results.append(result)

        print(f"\n  Summary: Constrained {constrained_valid}/{SAMPLES_PER_TEST} vs Unconstrained {unconstrained_valid}/{SAMPLES_PER_TEST}")

    # Print final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"\n{'Test':<35} {'Constrained':<15} {'Unconstrained':<15} {'Delta':<10}")
    print("-" * 75)

    for r in results:
        c_rate = r.constrained_valid / r.constrained_total if r.constrained_total > 0 else 0
        u_rate = r.unconstrained_valid / r.unconstrained_total if r.unconstrained_total > 0 else 0
        delta = c_rate - u_rate
        print(f"{r.name:<35} {c_rate:>13.0%} {u_rate:>13.0%} {delta:>+9.0%}")

    # Overall
    total_c = sum(r.constrained_valid for r in results)
    total_c_n = sum(r.constrained_total for r in results)
    total_u = sum(r.unconstrained_valid for r in results)
    total_u_n = sum(r.unconstrained_total for r in results)

    print("-" * 75)
    c_rate = total_c / total_c_n if total_c_n > 0 else 0
    u_rate = total_u / total_u_n if total_u_n > 0 else 0
    print(f"{'OVERALL':<35} {c_rate:>13.0%} {u_rate:>13.0%} {c_rate - u_rate:>+9.0%}")

    print("\n" + "=" * 70)
    print("SAMPLE OUTPUTS")
    print("=" * 70)
    for r in results:
        print(f"\n{r.name}:")
        print(f"  Constrained: {r.sample_constrained[:100]}...")
        print(f"  Unconstrained: {r.sample_unconstrained[:100]}...")

    return {
        "results": [
            {
                "name": r.name,
                "constrained_valid": r.constrained_valid,
                "constrained_total": r.constrained_total,
                "unconstrained_valid": r.unconstrained_valid,
                "unconstrained_total": r.unconstrained_total,
            }
            for r in results
        ],
        "overall_constrained": c_rate,
        "overall_unconstrained": u_rate,
    }


@app.local_entrypoint()
def main():
    result = run_evaluation.remote()
    print(f"\nEvaluation complete!")
