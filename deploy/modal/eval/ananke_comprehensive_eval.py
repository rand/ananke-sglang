"""
Comprehensive Ananke Evaluation

Tests constrained generation across:
- Task types: completion, tests, fill-in-middle, refactoring, bug fixing
- Languages: Python, TypeScript, Go, Rust, Kotlin, Swift, Zig
- Constraint domains: syntax, types, imports

Run with:
    modal run deploy/modal/eval/ananke_comprehensive_eval.py
    modal run deploy/modal/eval/ananke_comprehensive_eval.py --compare
"""

import ast
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import modal

# =============================================================================
# Configuration
# =============================================================================

APP_NAME = "ananke-comprehensive-eval"
DEPLOYED_APP = "qwen3-coder-ananke"
DEPLOYED_CLASS = "Qwen3CoderAnanke"

# Languages Ananke supports
LANGUAGES = ["python", "typescript", "go", "rust", "kotlin", "swift", "zig"]

# Task types to evaluate
TASK_TYPES = [
    "completion",      # Complete a function body
    "test_creation",   # Write tests for given code
    "fill_middle",     # Fill in missing code
    "refactoring",     # Improve/refactor code
    "bug_fixing",      # Fix broken code
]


@dataclass
class TestCase:
    """A single test case for evaluation."""
    id: str
    language: str
    task_type: str
    prompt: str
    constraint_spec: dict
    expected_patterns: list[str]  # Patterns that should appear in output
    max_tokens: int = 300
    temperature: float = 0.3


@dataclass
class EvalResult:
    """Result of evaluating a test case."""
    test_id: str
    language: str
    task_type: str
    success: bool
    valid_syntax: bool
    pattern_matches: int
    total_patterns: int
    generation_time: float
    output_preview: str
    error: Optional[str] = None


# =============================================================================
# Test Cases by Language and Task Type
# =============================================================================

def get_test_cases() -> list[TestCase]:
    """Generate comprehensive test cases across languages and task types."""
    cases = []

    # =========================================================================
    # PYTHON TEST CASES
    # =========================================================================

    # Python: Code Completion
    cases.append(TestCase(
        id="py_completion_1",
        language="python",
        task_type="completion",
        prompt="""def fibonacci(n: int) -> int:
    \"\"\"Return the nth Fibonacci number.\"\"\"
""",
        constraint_spec={
            "language": "python",
            "domains": ["syntax", "types"],
        },
        expected_patterns=["if", "return", "fibonacci"],
    ))

    cases.append(TestCase(
        id="py_completion_2",
        language="python",
        task_type="completion",
        prompt="""def binary_search(arr: list[int], target: int) -> int:
    \"\"\"Return index of target in sorted array, or -1 if not found.\"\"\"
""",
        constraint_spec={
            "language": "python",
            "domains": ["syntax", "types"],
        },
        expected_patterns=["while", "return", "mid"],
    ))

    # Python: Test Creation
    cases.append(TestCase(
        id="py_test_1",
        language="python",
        task_type="test_creation",
        prompt="""# Write pytest tests for this function:
def is_palindrome(s: str) -> bool:
    return s == s[::-1]

# Tests:
import pytest

def test_is_palindrome():
""",
        constraint_spec={
            "language": "python",
            "domains": ["syntax", "imports"],
        },
        expected_patterns=["assert", "is_palindrome"],
    ))

    # Python: Fill in the Middle
    cases.append(TestCase(
        id="py_fill_1",
        language="python",
        task_type="fill_middle",
        prompt="""def process_data(items: list[dict]) -> list[str]:
    results = []
    for item in items:
        # TODO: Extract 'name' field if it exists, otherwise use 'id'
""",
        constraint_spec={
            "language": "python",
            "domains": ["syntax", "types"],
        },
        expected_patterns=["name", "id", "append"],
    ))

    # Python: Refactoring
    cases.append(TestCase(
        id="py_refactor_1",
        language="python",
        task_type="refactoring",
        prompt="""# Refactor this code to use list comprehension:
def get_even_squares(numbers):
    result = []
    for n in numbers:
        if n % 2 == 0:
            result.append(n * n)
    return result

# Refactored version:
def get_even_squares_v2(numbers):
""",
        constraint_spec={
            "language": "python",
            "domains": ["syntax"],
        },
        expected_patterns=["return", "[", "for", "if"],
    ))

    # Python: Bug Fixing
    cases.append(TestCase(
        id="py_bugfix_1",
        language="python",
        task_type="bug_fixing",
        prompt="""# Fix the bug in this function (off-by-one error):
def sum_range(start: int, end: int) -> int:
    \"\"\"Sum all integers from start to end, inclusive.\"\"\"
    total = 0
    for i in range(start, end):  # BUG: should include end
        total += i
    return total

# Fixed version:
def sum_range_fixed(start: int, end: int) -> int:
""",
        constraint_spec={
            "language": "python",
            "domains": ["syntax", "types"],
        },
        expected_patterns=["range", "end + 1", "return"],
    ))

    # =========================================================================
    # TYPESCRIPT TEST CASES
    # =========================================================================

    cases.append(TestCase(
        id="ts_completion_1",
        language="typescript",
        task_type="completion",
        prompt="""function mergeArrays<T>(arr1: T[], arr2: T[]): T[] {
    // Merge two arrays and return sorted unique values
""",
        constraint_spec={
            "language": "typescript",
            "domains": ["syntax", "types"],
        },
        expected_patterns=["return", "Set", "sort"],
        max_tokens=200,
    ))

    cases.append(TestCase(
        id="ts_test_1",
        language="typescript",
        task_type="test_creation",
        prompt="""// Write Jest tests for this function:
function capitalize(str: string): string {
    return str.charAt(0).toUpperCase() + str.slice(1);
}

// Tests:
describe('capitalize', () => {
    it('should capitalize first letter', () => {
""",
        constraint_spec={
            "language": "typescript",
            "domains": ["syntax"],
        },
        expected_patterns=["expect", "capitalize", "toBe"],
    ))

    cases.append(TestCase(
        id="ts_bugfix_1",
        language="typescript",
        task_type="bug_fixing",
        prompt="""// Fix the type error in this code:
interface User {
    id: number;
    name: string;
    email?: string;
}

function getUserEmail(user: User): string {
    return user.email;  // Error: might be undefined
}

// Fixed version:
function getUserEmailFixed(user: User): string {
""",
        constraint_spec={
            "language": "typescript",
            "domains": ["syntax", "types"],
        },
        expected_patterns=["return", "email", "||", "??"],
    ))

    # =========================================================================
    # GO TEST CASES
    # =========================================================================

    cases.append(TestCase(
        id="go_completion_1",
        language="go",
        task_type="completion",
        prompt="""func reverseString(s string) string {
    // Reverse the input string
""",
        constraint_spec={
            "language": "go",
            "domains": ["syntax", "types"],
        },
        expected_patterns=["return", "rune", "for"],
    ))

    cases.append(TestCase(
        id="go_test_1",
        language="go",
        task_type="test_creation",
        prompt="""// Write a test for this function:
func Add(a, b int) int {
    return a + b
}

// Test:
func TestAdd(t *testing.T) {
""",
        constraint_spec={
            "language": "go",
            "domains": ["syntax", "imports"],
        },
        expected_patterns=["t.Error", "Add", "expected"],
    ))

    cases.append(TestCase(
        id="go_bugfix_1",
        language="go",
        task_type="bug_fixing",
        prompt="""// Fix the nil pointer dereference:
func getLength(s *string) int {
    return len(*s)  // Panic if s is nil
}

// Fixed version:
func getLengthFixed(s *string) int {
""",
        constraint_spec={
            "language": "go",
            "domains": ["syntax", "types"],
        },
        expected_patterns=["if", "nil", "return"],
    ))

    # =========================================================================
    # RUST TEST CASES
    # =========================================================================

    cases.append(TestCase(
        id="rust_completion_1",
        language="rust",
        task_type="completion",
        prompt="""fn find_max(numbers: &[i32]) -> Option<i32> {
    // Return the maximum value, or None if empty
""",
        constraint_spec={
            "language": "rust",
            "domains": ["syntax", "types"],
        },
        expected_patterns=["Some", "None", "max"],
    ))

    cases.append(TestCase(
        id="rust_test_1",
        language="rust",
        task_type="test_creation",
        prompt="""// Write tests for this function:
fn is_even(n: i32) -> bool {
    n % 2 == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_even() {
""",
        constraint_spec={
            "language": "rust",
            "domains": ["syntax"],
        },
        expected_patterns=["assert", "is_even", "true", "false"],
    ))

    cases.append(TestCase(
        id="rust_bugfix_1",
        language="rust",
        task_type="bug_fixing",
        prompt="""// Fix the ownership error:
fn process_string(s: String) -> String {
    println!("{}", s);
    s.to_uppercase()  // s is moved in println
}

// Fixed version (use reference):
fn process_string_fixed(s: &str) -> String {
""",
        constraint_spec={
            "language": "rust",
            "domains": ["syntax", "types"],
        },
        expected_patterns=["println", "to_uppercase", "return"],
    ))

    # =========================================================================
    # KOTLIN TEST CASES
    # =========================================================================

    cases.append(TestCase(
        id="kotlin_completion_1",
        language="kotlin",
        task_type="completion",
        prompt="""fun filterPositive(numbers: List<Int>): List<Int> {
    // Return only positive numbers
""",
        constraint_spec={
            "language": "kotlin",
            "domains": ["syntax", "types"],
        },
        expected_patterns=["filter", "return", "> 0"],
    ))

    cases.append(TestCase(
        id="kotlin_bugfix_1",
        language="kotlin",
        task_type="bug_fixing",
        prompt="""// Fix the null safety issue:
fun getNameLength(name: String?): Int {
    return name.length  // Error: name might be null
}

// Fixed version:
fun getNameLengthFixed(name: String?): Int {
""",
        constraint_spec={
            "language": "kotlin",
            "domains": ["syntax", "types"],
        },
        expected_patterns=["?.", "?:", "return"],
    ))

    # =========================================================================
    # SWIFT TEST CASES
    # =========================================================================

    cases.append(TestCase(
        id="swift_completion_1",
        language="swift",
        task_type="completion",
        prompt="""func findDuplicates(_ array: [Int]) -> [Int] {
    // Return array of duplicate values
""",
        constraint_spec={
            "language": "swift",
            "domains": ["syntax", "types"],
        },
        expected_patterns=["return", "Set", "filter"],
    ))

    cases.append(TestCase(
        id="swift_bugfix_1",
        language="swift",
        task_type="bug_fixing",
        prompt="""// Fix the optional unwrapping:
func greet(name: String?) -> String {
    return "Hello, " + name!  // Force unwrap is unsafe
}

// Fixed version:
func greetFixed(name: String?) -> String {
""",
        constraint_spec={
            "language": "swift",
            "domains": ["syntax", "types"],
        },
        expected_patterns=["if let", "guard", "??", "return"],
    ))

    # =========================================================================
    # ZIG TEST CASES
    # =========================================================================

    cases.append(TestCase(
        id="zig_completion_1",
        language="zig",
        task_type="completion",
        prompt="""fn sum(items: []const i32) i32 {
    // Sum all items in the slice
""",
        constraint_spec={
            "language": "zig",
            "domains": ["syntax", "types"],
        },
        expected_patterns=["for", "return", "total"],
    ))

    cases.append(TestCase(
        id="zig_bugfix_1",
        language="zig",
        task_type="bug_fixing",
        prompt="""// Fix the error handling:
fn divide(a: i32, b: i32) i32 {
    return a / b;  // Will panic on divide by zero
}

// Fixed version with error handling:
fn divideFixed(a: i32, b: i32) !i32 {
""",
        constraint_spec={
            "language": "zig",
            "domains": ["syntax", "types"],
        },
        expected_patterns=["if", "return", "error"],
    ))

    return cases


# =============================================================================
# Syntax Validators by Language
# =============================================================================

def validate_syntax(code: str, language: str) -> tuple[bool, Optional[str]]:
    """Validate syntax for the given language."""
    if language == "python":
        return validate_python_syntax(code)
    elif language == "typescript":
        return validate_typescript_syntax(code)
    elif language == "go":
        return validate_go_syntax(code)
    elif language == "rust":
        return validate_rust_syntax(code)
    elif language == "kotlin":
        return validate_kotlin_syntax(code)
    elif language == "swift":
        return validate_swift_syntax(code)
    elif language == "zig":
        return validate_zig_syntax(code)
    else:
        return True, None  # Unknown language, assume valid


def validate_python_syntax(code: str) -> tuple[bool, Optional[str]]:
    """Validate Python syntax using ast.parse."""
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, f"Line {e.lineno}: {e.msg}"


def validate_typescript_syntax(code: str) -> tuple[bool, Optional[str]]:
    """Basic TypeScript syntax validation."""
    # Check for basic structural validity
    if not code.strip():
        return False, "Empty code"

    # Check balanced braces
    if code.count('{') != code.count('}'):
        return False, "Unbalanced braces"
    if code.count('(') != code.count(')'):
        return False, "Unbalanced parentheses"

    # Check for common syntax issues
    if re.search(r'\bfunction\s+\(', code):  # function without name
        pass  # Arrow functions are OK

    return True, None


def validate_go_syntax(code: str) -> tuple[bool, Optional[str]]:
    """Basic Go syntax validation."""
    if not code.strip():
        return False, "Empty code"

    if code.count('{') != code.count('}'):
        return False, "Unbalanced braces"

    return True, None


def validate_rust_syntax(code: str) -> tuple[bool, Optional[str]]:
    """Basic Rust syntax validation."""
    if not code.strip():
        return False, "Empty code"

    if code.count('{') != code.count('}'):
        return False, "Unbalanced braces"

    return True, None


def validate_kotlin_syntax(code: str) -> tuple[bool, Optional[str]]:
    """Basic Kotlin syntax validation."""
    if not code.strip():
        return False, "Empty code"

    if code.count('{') != code.count('}'):
        return False, "Unbalanced braces"

    return True, None


def validate_swift_syntax(code: str) -> tuple[bool, Optional[str]]:
    """Basic Swift syntax validation."""
    if not code.strip():
        return False, "Empty code"

    if code.count('{') != code.count('}'):
        return False, "Unbalanced braces"

    return True, None


def validate_zig_syntax(code: str) -> tuple[bool, Optional[str]]:
    """Basic Zig syntax validation."""
    if not code.strip():
        return False, "Empty code"

    if code.count('{') != code.count('}'):
        return False, "Unbalanced braces"

    return True, None


# =============================================================================
# Modal App
# =============================================================================

app = modal.App(APP_NAME)


@app.function(timeout=1800)
def run_evaluation(use_constraints: bool = True) -> dict:
    """Run comprehensive evaluation."""
    print("=" * 70)
    print("Ananke Comprehensive Evaluation")
    print("=" * 70)
    print(f"Constraints: {'Enabled' if use_constraints else 'Disabled'}")
    print("=" * 70)

    # Get test cases
    test_cases = get_test_cases()
    print(f"\nTotal test cases: {len(test_cases)}")

    # Group by language and task type
    by_language = {}
    by_task = {}
    for tc in test_cases:
        by_language.setdefault(tc.language, []).append(tc)
        by_task.setdefault(tc.task_type, []).append(tc)

    print("\nBy language:")
    for lang, cases in by_language.items():
        print(f"  {lang}: {len(cases)}")

    print("\nBy task type:")
    for task, cases in by_task.items():
        print(f"  {task}: {len(cases)}")

    # Connect to deployed model
    print("\nConnecting to deployed model...")
    Qwen3CoderAnanke = modal.Cls.from_name(DEPLOYED_APP, DEPLOYED_CLASS)
    server = Qwen3CoderAnanke()

    # Run evaluation
    results = []

    for i, tc in enumerate(test_cases, 1):
        print(f"\n[{i}/{len(test_cases)}] {tc.id} ({tc.language}/{tc.task_type})")

        start = time.time()
        try:
            if use_constraints:
                response = server.generate_constrained.remote(
                    prompt=tc.prompt,
                    constraint_spec=tc.constraint_spec,
                    max_tokens=tc.max_tokens,
                    temperature=tc.temperature,
                )
                output = response.get("text", "")
            else:
                output = server.generate.remote(
                    prompt=tc.prompt,
                    max_tokens=tc.max_tokens,
                    temperature=tc.temperature,
                )

            gen_time = time.time() - start

            # Extract code (remove markdown if present)
            code = extract_code(output)

            # Validate syntax
            valid, error = validate_syntax(code, tc.language)

            # Check pattern matches
            matches = sum(1 for p in tc.expected_patterns if p.lower() in code.lower())

            success = valid and matches >= len(tc.expected_patterns) // 2

            result = EvalResult(
                test_id=tc.id,
                language=tc.language,
                task_type=tc.task_type,
                success=success,
                valid_syntax=valid,
                pattern_matches=matches,
                total_patterns=len(tc.expected_patterns),
                generation_time=gen_time,
                output_preview=code[:200],
                error=error,
            )

            status = "✓" if success else "✗"
            print(f"  {status} Syntax: {'✓' if valid else '✗'} | Patterns: {matches}/{len(tc.expected_patterns)} | {gen_time:.2f}s")
            if error:
                print(f"      Error: {error[:50]}")

        except Exception as e:
            gen_time = time.time() - start
            result = EvalResult(
                test_id=tc.id,
                language=tc.language,
                task_type=tc.task_type,
                success=False,
                valid_syntax=False,
                pattern_matches=0,
                total_patterns=len(tc.expected_patterns),
                generation_time=gen_time,
                output_preview="",
                error=str(e)[:100],
            )
            print(f"  ✗ ERROR: {str(e)[:60]}")

        results.append(result)

    # Compute summary
    summary = compute_summary(results)

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"\nOverall: {summary['success_count']}/{summary['total']} ({summary['success_rate']:.1%})")
    print(f"Valid syntax: {summary['valid_syntax_count']}/{summary['total']} ({summary['syntax_rate']:.1%})")
    print(f"Avg generation time: {summary['avg_time']:.2f}s")

    print("\nBy Language:")
    for lang, stats in summary['by_language'].items():
        print(f"  {lang:12s}: {stats['success']}/{stats['total']} ({stats['rate']:.0%}) syntax={stats['syntax_rate']:.0%}")

    print("\nBy Task Type:")
    for task, stats in summary['by_task'].items():
        print(f"  {task:15s}: {stats['success']}/{stats['total']} ({stats['rate']:.0%})")

    return {
        "summary": summary,
        "results": [vars(r) for r in results],
        "config": {
            "use_constraints": use_constraints,
            "total_cases": len(test_cases),
        },
    }


def extract_code(text: str) -> str:
    """Extract code from response, removing markdown."""
    # Try to find code block
    if "```" in text:
        match = re.search(r"```(?:\w+)?\n?(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
    return text.strip()


def compute_summary(results: list[EvalResult]) -> dict:
    """Compute summary statistics."""
    total = len(results)
    success_count = sum(1 for r in results if r.success)
    valid_syntax_count = sum(1 for r in results if r.valid_syntax)
    times = [r.generation_time for r in results]

    by_language = {}
    by_task = {}

    for r in results:
        # By language
        if r.language not in by_language:
            by_language[r.language] = {"total": 0, "success": 0, "valid_syntax": 0}
        by_language[r.language]["total"] += 1
        if r.success:
            by_language[r.language]["success"] += 1
        if r.valid_syntax:
            by_language[r.language]["valid_syntax"] += 1

        # By task
        if r.task_type not in by_task:
            by_task[r.task_type] = {"total": 0, "success": 0}
        by_task[r.task_type]["total"] += 1
        if r.success:
            by_task[r.task_type]["success"] += 1

    # Compute rates
    for lang in by_language:
        stats = by_language[lang]
        stats["rate"] = stats["success"] / stats["total"] if stats["total"] > 0 else 0
        stats["syntax_rate"] = stats["valid_syntax"] / stats["total"] if stats["total"] > 0 else 0

    for task in by_task:
        stats = by_task[task]
        stats["rate"] = stats["success"] / stats["total"] if stats["total"] > 0 else 0

    return {
        "total": total,
        "success_count": success_count,
        "valid_syntax_count": valid_syntax_count,
        "success_rate": success_count / total if total > 0 else 0,
        "syntax_rate": valid_syntax_count / total if total > 0 else 0,
        "avg_time": sum(times) / len(times) if times else 0,
        "by_language": by_language,
        "by_task": by_task,
    }


@app.function(timeout=3600)
def run_comparison() -> dict:
    """Compare constrained vs unconstrained generation."""
    print("=" * 70)
    print("Ananke: Constrained vs Unconstrained Comparison")
    print("=" * 70)

    print("\n[Phase 1] Running WITH constraints...")
    constrained = run_evaluation.local(use_constraints=True)

    print("\n[Phase 2] Running WITHOUT constraints...")
    unconstrained = run_evaluation.local(use_constraints=False)

    # Compare
    c_sum = constrained["summary"]
    u_sum = unconstrained["summary"]

    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)

    print(f"\n{'Metric':<25} {'Constrained':>15} {'Unconstrained':>15} {'Delta':>10}")
    print("-" * 70)
    print(f"{'Success rate':<25} {c_sum['success_rate']:>14.1%} {u_sum['success_rate']:>14.1%} {c_sum['success_rate']-u_sum['success_rate']:>+9.1%}")
    print(f"{'Syntax validity':<25} {c_sum['syntax_rate']:>14.1%} {u_sum['syntax_rate']:>14.1%} {c_sum['syntax_rate']-u_sum['syntax_rate']:>+9.1%}")
    print(f"{'Avg time (s)':<25} {c_sum['avg_time']:>15.2f} {u_sum['avg_time']:>15.2f} {c_sum['avg_time']-u_sum['avg_time']:>+10.2f}")

    print("\nBy Language (Syntax Validity):")
    for lang in sorted(c_sum['by_language'].keys()):
        c_rate = c_sum['by_language'].get(lang, {}).get('syntax_rate', 0)
        u_rate = u_sum['by_language'].get(lang, {}).get('syntax_rate', 0)
        print(f"  {lang:12s}: Constrained {c_rate:.0%} | Unconstrained {u_rate:.0%} | Delta {c_rate-u_rate:+.0%}")

    return {
        "constrained": constrained,
        "unconstrained": unconstrained,
        "comparison": {
            "success_delta": c_sum["success_rate"] - u_sum["success_rate"],
            "syntax_delta": c_sum["syntax_rate"] - u_sum["syntax_rate"],
        },
    }


@app.local_entrypoint()
def main(compare: bool = False):
    """Run comprehensive Ananke evaluation."""
    if compare:
        results = run_comparison.remote()
    else:
        results = run_evaluation.remote(use_constraints=True)

    print(f"\nEvaluation complete!")


if __name__ == "__main__":
    main()
