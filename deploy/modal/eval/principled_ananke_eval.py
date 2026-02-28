"""Principled Ananke Evaluation - Testing Constraint Value

This evaluation tests Ananke's constraint domains with appropriate context data:
1. Syntax Domain: JSON schema, regex, EBNF (grammar enforcement)
2. Types Domain: type_bindings, expected_type (type-safe generation)
3. Imports Domain: available_modules, forbidden_imports (sandboxing)

Each test provides the SPECIFIC context needed for Ananke to provide value.

Run:
    modal run deploy/modal/eval/principled_ananke_eval.py
"""

import json
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import modal

# =============================================================================
# Configuration
# =============================================================================

DEPLOYED_APP = "qwen3-coder-ananke"
DEPLOYED_CLASS = "Qwen3CoderAnanke"

# Evaluation parameters
MAX_TOKENS = 300
SAMPLES_PER_TASK = 3
TEMPERATURE = 0.7

app = modal.App("principled-ananke-eval")


# =============================================================================
# Task Definitions
# =============================================================================

@dataclass
class EvalTask:
    """Evaluation task with proper constraint context."""
    id: str
    category: str
    description: str
    prompt: str
    constraint_spec: dict
    validator: Callable[[str], bool]
    validation_desc: str
    expected_improvement: str  # What we expect constraints to improve


TASKS = [
    # =========================================================================
    # Category A: JSON Schema (Syntax Domain)
    # =========================================================================
    EvalTask(
        id="json_user_profile",
        category="json_schema",
        description="Generate structured user profile",
        prompt="Generate a user profile with name, age, and email fields:\n\n",
        constraint_spec={
            "json_schema": json.dumps({
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                    "email": {"type": "string"}
                },
                "required": ["name", "age", "email"]
            })
        },
        validator=lambda text: _validate_json(text, ["name", "age", "email"]),
        validation_desc="Valid JSON with name, age, email fields",
        expected_improvement="100% valid JSON vs ~70% unconstrained",
    ),
    EvalTask(
        id="json_api_response",
        category="json_schema",
        description="Generate API error response",
        prompt="Generate an API error response:\n\n",
        constraint_spec={
            "json_schema": json.dumps({
                "type": "object",
                "properties": {
                    "error": {"type": "boolean"},
                    "code": {"type": "integer"},
                    "message": {"type": "string"}
                },
                "required": ["error", "code", "message"]
            })
        },
        validator=lambda text: _validate_json(text, ["error", "code", "message"]),
        validation_desc="Valid JSON with error, code, message fields",
        expected_improvement="100% valid JSON vs ~60% unconstrained",
    ),

    # =========================================================================
    # Category B: Regex Pattern (Syntax Domain)
    # =========================================================================
    EvalTask(
        id="regex_digits_only",
        category="regex",
        description="Force digit-only output",
        prompt="What is 15 * 7?",
        constraint_spec={
            "regex": r"[0-9]+"
        },
        validator=lambda text: bool(re.fullmatch(r"[0-9]+", text.strip())),
        validation_desc="Output contains only digits",
        expected_improvement="100% digits vs ~10% unconstrained",
    ),
    EvalTask(
        id="regex_function_signature",
        category="regex",
        description="Force function with return type annotation",
        prompt="Write a Python function to add two numbers:\n\n",
        constraint_spec={
            "regex": r"def \w+\([^)]*\) -> \w+:\n.*"
        },
        validator=lambda text: bool(re.match(r"def \w+\([^)]*\) -> \w+:", text.strip())),
        validation_desc="Function with -> return type annotation",
        expected_improvement="100% type hints vs ~50% unconstrained",
    ),

    # =========================================================================
    # Category C: Type Bindings (Types Domain)
    # =========================================================================
    EvalTask(
        id="type_python_sum",
        category="type_bindings",
        description="Complete function with type context (Python)",
        prompt="def sum_positives(numbers: list[int]) -> int:\n    '''Return sum of positive numbers.'''\n    return ",
        constraint_spec={
            "language": "python",
            "type_bindings": [
                {"name": "numbers", "type_expr": "list[int]"}
            ],
            "expected_type": "int"
        },
        validator=lambda text: _validates_int_expression(text),
        validation_desc="Returns integer expression",
        expected_improvement="Type context guides generation",
    ),
    EvalTask(
        id="type_python_filter",
        category="type_bindings",
        description="Complete list comprehension with types",
        prompt="def filter_evens(items: list[int]) -> list[int]:\n    '''Return only even numbers.'''\n    return ",
        constraint_spec={
            "language": "python",
            "type_bindings": [
                {"name": "items", "type_expr": "list[int]"}
            ],
            "expected_type": "list[int]"
        },
        validator=lambda text: "[" in text and "for" in text.lower(),
        validation_desc="Returns list expression",
        expected_improvement="Type context guides generation",
    ),

    # =========================================================================
    # Category D: Import Restrictions (Imports Domain)
    # =========================================================================
    EvalTask(
        id="import_safe_http",
        category="import_restrictions",
        description="Generate HTTP code without forbidden imports",
        prompt="Write a Python function to fetch a URL and return its content:\n\n",
        constraint_spec={
            "language": "python",
            "available_modules": ["requests", "urllib", "http"],
            "forbidden_imports": ["os", "subprocess", "sys", "socket", "shutil"]
        },
        validator=lambda text: (
            ("requests" in text or "urllib" in text or "http" in text) and
            "subprocess" not in text and
            "os.system" not in text
        ),
        validation_desc="Uses allowed modules, no forbidden imports",
        expected_improvement="No dangerous imports",
    ),
    EvalTask(
        id="import_json_only",
        category="import_restrictions",
        description="Generate code using only json module",
        prompt="Write a function to parse and validate JSON data:\n\n",
        constraint_spec={
            "language": "python",
            "available_modules": ["json", "typing"],
            "forbidden_imports": ["pickle", "yaml", "toml", "ast", "eval"]
        },
        validator=lambda text: "json" in text and "pickle" not in text,
        validation_desc="Uses json, avoids pickle",
        expected_improvement="Sandboxed import usage",
    ),

    # =========================================================================
    # Category E: TypeScript Type Bindings
    # =========================================================================
    EvalTask(
        id="type_typescript_map",
        category="type_bindings_ts",
        description="Complete TypeScript function with types",
        prompt="function doubleAll(nums: number[]): number[] {\n    return ",
        constraint_spec={
            "language": "typescript",
            "type_bindings": [
                {"name": "nums", "type_expr": "number[]"}
            ],
            "expected_type": "number[]"
        },
        validator=lambda text: ".map" in text or "for" in text.lower(),
        validation_desc="Returns array transformation",
        expected_improvement="Type-aware completion",
    ),
    EvalTask(
        id="type_typescript_filter",
        category="type_bindings_ts",
        description="Complete TypeScript filter with types",
        prompt="function getPositive(values: number[]): number[] {\n    return ",
        constraint_spec={
            "language": "typescript",
            "type_bindings": [
                {"name": "values", "type_expr": "number[]"}
            ],
            "expected_type": "number[]"
        },
        validator=lambda text: ".filter" in text or "[" in text,
        validation_desc="Returns filtered array",
        expected_improvement="Type-aware completion",
    ),

    # =========================================================================
    # Category F: Zig Type Bindings
    # =========================================================================
    EvalTask(
        id="type_zig_sum",
        category="type_bindings_zig",
        description="Complete Zig function with types",
        prompt="fn sumSlice(items: []const i32) i32 {\n    var total: i32 = 0;\n    for (items) |item| {\n        total += ",
        constraint_spec={
            "language": "zig",
            "type_bindings": [
                {"name": "items", "type_expr": "[]const i32"},
                {"name": "total", "type_expr": "i32"},
                {"name": "item", "type_expr": "i32"}
            ],
            "expected_type": "i32"
        },
        validator=lambda text: "item" in text,
        validation_desc="Uses item variable correctly",
        expected_improvement="Type-aware completion",
    ),

    # =========================================================================
    # Category G: Combined Constraints
    # =========================================================================
    EvalTask(
        id="combined_typed_json",
        category="combined",
        description="JSON schema with type context",
        prompt="Generate a typed configuration object:\n\n",
        constraint_spec={
            "language": "python",
            "json_schema": json.dumps({
                "type": "object",
                "properties": {
                    "debug": {"type": "boolean"},
                    "port": {"type": "integer"},
                    "host": {"type": "string"}
                },
                "required": ["debug", "port", "host"]
            }),
            "type_bindings": [
                {"name": "config", "type_expr": "dict"}
            ]
        },
        validator=lambda text: _validate_json(text, ["debug", "port", "host"]),
        validation_desc="Valid JSON config object",
        expected_improvement="Syntax + type constraints",
    ),
]


# =============================================================================
# Validators
# =============================================================================

def _validate_json(text: str, required_fields: list) -> bool:
    """Validate text is valid JSON with required fields."""
    try:
        text = text.strip()
        # Find JSON object
        start = text.find("{")
        if start == -1:
            return False

        # Find matching close brace
        depth = 0
        end = start
        for i, c in enumerate(text[start:], start):
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break

        if depth != 0:
            return False

        json_str = text[start:end]
        obj = json.loads(json_str)
        return all(f in obj for f in required_fields)
    except (json.JSONDecodeError, ValueError):
        return False


def _validates_int_expression(text: str) -> bool:
    """Check if text looks like an integer-returning expression."""
    text = text.strip().split("\n")[0]  # First line
    # Common patterns for int returns
    patterns = [
        r"sum\(",       # sum(...)
        r"\+",          # addition
        r"total",       # variable
        r"\d+",         # literal number
        r"len\(",       # length
    ]
    return any(re.search(p, text) for p in patterns)


# =============================================================================
# Results
# =============================================================================

@dataclass
class TaskResult:
    task_id: str
    category: str
    constrained_valid: int
    constrained_total: int
    unconstrained_valid: int
    unconstrained_total: int
    constrained_avg_time: float
    unconstrained_avg_time: float
    sample_constrained: str
    sample_unconstrained: str


# =============================================================================
# Evaluation
# =============================================================================

@app.function(timeout=1800)
def run_principled_eval() -> dict:
    """Run the principled Ananke evaluation."""
    print("=" * 70)
    print("PRINCIPLED ANANKE EVALUATION")
    print("=" * 70)
    print(f"Tasks: {len(TASKS)}")
    print(f"Samples per task: {SAMPLES_PER_TASK}")
    print(f"Max tokens: {MAX_TOKENS}")
    print(f"Temperature: {TEMPERATURE}")
    print("=" * 70)

    # Connect to deployed model
    print("\nConnecting to deployed model...")
    Qwen3CoderAnanke = modal.Cls.from_name(DEPLOYED_APP, DEPLOYED_CLASS)
    server = Qwen3CoderAnanke()

    health = server.health.remote()
    print(f"Connected: {health}")

    results = []

    for task in TASKS:
        print(f"\n{'='*70}")
        print(f"TASK: {task.id}")
        print(f"Category: {task.category}")
        print(f"Description: {task.description}")
        print(f"Validation: {task.validation_desc}")
        print(f"Expected: {task.expected_improvement}")
        print(f"{'='*70}")

        constrained_valid = 0
        unconstrained_valid = 0
        constrained_times = []
        unconstrained_times = []
        sample_constrained = ""
        sample_unconstrained = ""

        # Run with constraints
        print("\n  CONSTRAINED:")
        for i in range(SAMPLES_PER_TASK):
            start = time.time()
            try:
                response = server.generate_constrained.remote(
                    prompt=task.prompt,
                    constraint_spec=task.constraint_spec,
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                )
                output = response.get("text", "")
                elapsed = time.time() - start
                constrained_times.append(elapsed)

                valid = task.validator(output)
                if valid:
                    constrained_valid += 1
                    print(f"    [{i+1}] VALID ({elapsed:.2f}s): {output[:60]}...")
                else:
                    print(f"    [{i+1}] INVALID ({elapsed:.2f}s): {output[:60]}...")

                if i == 0:
                    sample_constrained = output[:200]

            except Exception as e:
                print(f"    [{i+1}] ERROR: {str(e)[:80]}")

        # Run without constraints
        print("\n  UNCONSTRAINED:")
        for i in range(SAMPLES_PER_TASK):
            start = time.time()
            try:
                output = server.generate.remote(
                    prompt=task.prompt,
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                )
                elapsed = time.time() - start
                unconstrained_times.append(elapsed)

                valid = task.validator(output)
                if valid:
                    unconstrained_valid += 1
                    print(f"    [{i+1}] VALID ({elapsed:.2f}s): {output[:60]}...")
                else:
                    print(f"    [{i+1}] INVALID ({elapsed:.2f}s): {output[:60]}...")

                if i == 0:
                    sample_unconstrained = output[:200]

            except Exception as e:
                print(f"    [{i+1}] ERROR: {str(e)[:80]}")

        result = TaskResult(
            task_id=task.id,
            category=task.category,
            constrained_valid=constrained_valid,
            constrained_total=SAMPLES_PER_TASK,
            unconstrained_valid=unconstrained_valid,
            unconstrained_total=SAMPLES_PER_TASK,
            constrained_avg_time=sum(constrained_times) / len(constrained_times) if constrained_times else 0,
            unconstrained_avg_time=sum(unconstrained_times) / len(unconstrained_times) if unconstrained_times else 0,
            sample_constrained=sample_constrained,
            sample_unconstrained=sample_unconstrained,
        )
        results.append(result)

        c_rate = constrained_valid / SAMPLES_PER_TASK if SAMPLES_PER_TASK > 0 else 0
        u_rate = unconstrained_valid / SAMPLES_PER_TASK if SAMPLES_PER_TASK > 0 else 0
        print(f"\n  Summary: Constrained {c_rate:.0%} vs Unconstrained {u_rate:.0%}")

    # Final summary
    print("\n" + "=" * 70)
    print("RESULTS BY CATEGORY")
    print("=" * 70)

    categories = {}
    for r in results:
        if r.category not in categories:
            categories[r.category] = {"c_valid": 0, "c_total": 0, "u_valid": 0, "u_total": 0}
        categories[r.category]["c_valid"] += r.constrained_valid
        categories[r.category]["c_total"] += r.constrained_total
        categories[r.category]["u_valid"] += r.unconstrained_valid
        categories[r.category]["u_total"] += r.unconstrained_total

    print(f"\n{'Category':<25} {'Constrained':<15} {'Unconstrained':<15} {'Delta':<10}")
    print("-" * 65)

    for cat, data in categories.items():
        c_rate = data["c_valid"] / data["c_total"] if data["c_total"] > 0 else 0
        u_rate = data["u_valid"] / data["u_total"] if data["u_total"] > 0 else 0
        delta = c_rate - u_rate
        print(f"{cat:<25} {c_rate:>13.0%} {u_rate:>13.0%} {delta:>+9.0%}")

    # Overall
    total_c = sum(r.constrained_valid for r in results)
    total_c_n = sum(r.constrained_total for r in results)
    total_u = sum(r.unconstrained_valid for r in results)
    total_u_n = sum(r.unconstrained_total for r in results)

    print("-" * 65)
    c_rate = total_c / total_c_n if total_c_n > 0 else 0
    u_rate = total_u / total_u_n if total_u_n > 0 else 0
    print(f"{'OVERALL':<25} {c_rate:>13.0%} {u_rate:>13.0%} {c_rate - u_rate:>+9.0%}")

    print("\n" + "=" * 70)
    print("INDIVIDUAL TASK RESULTS")
    print("=" * 70)
    print(f"\n{'Task ID':<30} {'Const':<8} {'Unconst':<8} {'Delta':<8}")
    print("-" * 55)

    for r in results:
        c_rate = r.constrained_valid / r.constrained_total if r.constrained_total > 0 else 0
        u_rate = r.unconstrained_valid / r.unconstrained_total if r.unconstrained_total > 0 else 0
        delta = c_rate - u_rate
        print(f"{r.task_id:<30} {c_rate:>6.0%} {u_rate:>8.0%} {delta:>+7.0%}")

    print("\n" + "=" * 70)
    print("SAMPLE OUTPUTS")
    print("=" * 70)

    for r in results:
        print(f"\n{r.task_id}:")
        print(f"  Constrained:   {r.sample_constrained[:100]}...")
        print(f"  Unconstrained: {r.sample_unconstrained[:100]}...")

    return {
        "results": [
            {
                "task_id": r.task_id,
                "category": r.category,
                "constrained_valid": r.constrained_valid,
                "constrained_total": r.constrained_total,
                "unconstrained_valid": r.unconstrained_valid,
                "unconstrained_total": r.unconstrained_total,
                "constrained_avg_time": r.constrained_avg_time,
                "unconstrained_avg_time": r.unconstrained_avg_time,
            }
            for r in results
        ],
        "overall_constrained": c_rate,
        "overall_unconstrained": u_rate,
        "delta": c_rate - u_rate,
    }


@app.local_entrypoint()
def main():
    print("Starting principled Ananke evaluation...")
    result = run_principled_eval.remote()
    print(f"\n{'='*70}")
    print("EVALUATION COMPLETE")
    print(f"{'='*70}")
    print(f"Overall Constrained: {result['overall_constrained']:.0%}")
    print(f"Overall Unconstrained: {result['overall_unconstrained']:.0%}")
    print(f"Delta: {result['delta']:+.0%}")
