"""
SWE-bench Evaluation for SGLang with Ananke Backend

Evaluates code generation on 30 principled SWE-bench Lite tasks.

This evaluation:
1. Loads problem statements from SWE-bench Lite
2. Generates patches using the deployed Qwen3-Coder with Ananke
3. Evaluates patch quality (syntax, structure, relevance)
4. Compares constrained vs unconstrained generation

Run with:
    modal run deploy/modal/eval/swebench_eval.py
"""

import ast
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import modal

# =============================================================================
# Configuration
# =============================================================================

APP_NAME = "swebench-ananke-eval"
DEPLOYED_APP = "qwen3-coder-ananke"
DEPLOYED_CLASS = "Qwen3CoderAnanke"

# Selected tasks (from principled selection)
SELECTED_TASKS = [
    "astropy__astropy-12907",
    "django__django-11848",
    "django__django-13551",
    "django__django-16408",
    "django__django-11964",
    "django__django-15252",
    "django__django-16910",
    "django__django-11815",
    "django__django-11905",
    "django__django-14016",
    "django__django-15202",
    "django__django-16816",
    "matplotlib__matplotlib-23562",
    "matplotlib__matplotlib-23913",
    "mwaskom__seaborn-2848",
    "pallets__flask-4045",
    "psf__requests-3362",
    "pydata__xarray-3364",
    "pylint-dev__pylint-7228",
    "pytest-dev__pytest-5221",
    "scikit-learn__scikit-learn-13142",
    "scikit-learn__scikit-learn-10508",
    "sphinx-doc__sphinx-8627",
    "sympy__sympy-18532",
    "sympy__sympy-13895",
    "sympy__sympy-14024",
    "sympy__sympy-16281",
    "sympy__sympy-11400",
    "sympy__sympy-18087",
    "sympy__sympy-22840",
]

# Prompt template for code generation (not diff format)
# Ananke constraints work on Python code, not diff format
CODE_COMPLETION_TEMPLATE = """You are a software engineer fixing a bug in a Python repository.

## Repository: {repo}

## Problem Statement
{problem_statement}

## Task
Write the corrected Python function or code block that fixes this issue. Output ONLY valid Python code.

Focus on:
1. Fixing the exact issue described
2. Writing syntactically correct Python
3. Following the existing code style

## Corrected Python code:
```python
"""

# Alternative: Simplified prompt for function completion
FUNCTION_FIX_TEMPLATE = """Fix this Python code issue:

{problem_statement}

Write the corrected Python code:
```python
"""


@dataclass
class EvalResult:
    """Result for a single SWE-bench task."""
    instance_id: str
    repo: str
    success: bool
    generation_time: float
    patch_generated: str
    patch_gold: str
    metrics: dict = field(default_factory=dict)
    error: Optional[str] = None


# =============================================================================
# Modal App
# =============================================================================

eval_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("datasets", "huggingface_hub")
)

app = modal.App(APP_NAME)


@app.function(image=eval_image, timeout=1800)
def run_evaluation(
    task_ids: list[str] = SELECTED_TASKS,
    use_constraints: bool = True,
    max_tokens: int = 500,
) -> dict:
    """
    Run SWE-bench evaluation on selected tasks.

    Args:
        task_ids: List of SWE-bench instance IDs to evaluate
        use_constraints: Whether to use Ananke constraints
        max_tokens: Max tokens for generation

    Returns:
        Evaluation results dict
    """
    from datasets import load_dataset

    print("=" * 70)
    print("SWE-bench Evaluation with Ananke")
    print("=" * 70)
    print(f"Tasks: {len(task_ids)}")
    print(f"Constraints: {'Ananke (syntax+types)' if use_constraints else 'None'}")
    print(f"Max tokens: {max_tokens}")
    print("=" * 70)

    # Load SWE-bench Lite
    print("\nLoading SWE-bench Lite dataset...")
    dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")

    # Index by instance_id
    task_data = {item["instance_id"]: item for item in dataset}

    # Get deployed model
    print("\nConnecting to deployed Qwen3-Coder model...")
    Qwen3CoderAnanke = modal.Cls.from_name(DEPLOYED_APP, DEPLOYED_CLASS)
    server = Qwen3CoderAnanke()

    # Run evaluation
    results = []

    for i, instance_id in enumerate(task_ids, 1):
        print(f"\n[{i}/{len(task_ids)}] {instance_id}")

        if instance_id not in task_data:
            print(f"  ERROR: Instance not found in dataset")
            results.append(EvalResult(
                instance_id=instance_id,
                repo="unknown",
                success=False,
                generation_time=0,
                patch_generated="",
                patch_gold="",
                error="Instance not found",
            ))
            continue

        item = task_data[instance_id]
        repo = item["repo"]
        problem = item["problem_statement"]
        gold_patch = item["patch"]

        # Build prompt - use code completion template (not diff format)
        # Ananke constraints validate Python syntax, so we need Python output
        prompt = CODE_COMPLETION_TEMPLATE.format(
            repo=repo,
            problem_statement=problem[:2500],  # Truncate very long problems
        )

        # Generate code fix
        start_time = time.time()
        try:
            if use_constraints:
                # Use Ananke with syntax and types constraints for Python code
                response = server.generate_constrained.remote(
                    prompt=prompt,
                    constraint_spec={
                        "language": "python",
                        "domains": ["syntax", "types"],  # Both syntax and type checking
                    },
                    max_tokens=max_tokens,
                    temperature=0.3,  # Slightly higher for more varied fixes
                )
                generated = response.get("text", "")
            else:
                generated = server.generate.remote(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=0.3,
                )

            gen_time = time.time() - start_time

            # Extract patch from response
            patch = extract_patch(generated)

            # Evaluate patch quality
            metrics = evaluate_patch(patch, gold_patch, problem)

            success = metrics.get("valid_syntax", False) and metrics.get("has_changes", False)

            result = EvalResult(
                instance_id=instance_id,
                repo=repo,
                success=success,
                generation_time=gen_time,
                patch_generated=patch[:1000],  # Truncate for storage
                patch_gold=gold_patch[:1000],
                metrics=metrics,
            )

            status = "✓" if success else "✗"
            print(f"  {status} Generated in {gen_time:.2f}s")
            print(f"    Syntax valid: {metrics.get('valid_syntax', False)}")
            if not metrics.get('valid_syntax') and metrics.get('syntax_error'):
                print(f"      Error: {metrics['syntax_error'][:60]}")
            print(f"    Code lines: {metrics.get('code_lines', 0)}")
            print(f"    Line overlap: {metrics.get('line_overlap', 0):.1%}")

        except Exception as e:
            gen_time = time.time() - start_time
            result = EvalResult(
                instance_id=instance_id,
                repo=repo,
                success=False,
                generation_time=gen_time,
                patch_generated="",
                patch_gold=gold_patch[:1000],
                error=str(e)[:200],
            )
            print(f"  ✗ ERROR: {str(e)[:100]}")

        results.append(result)

    # Compute summary statistics
    summary = compute_summary(results)

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Total tasks: {summary['total']}")
    print(f"Successful generations: {summary['successful']}")
    print(f"Valid syntax: {summary['valid_syntax']}")
    print(f"Has changes: {summary['has_changes']}")
    print(f"Success rate: {summary['success_rate']:.1%}")
    print(f"Avg generation time: {summary['avg_time']:.2f}s")
    print(f"Avg line overlap with gold: {summary['avg_overlap']:.1%}")

    print("\nBy repository:")
    for repo, stats in summary["by_repo"].items():
        print(f"  {repo}: {stats['successful']}/{stats['total']} ({stats['rate']:.0%})")

    return {
        "summary": summary,
        "results": [vars(r) for r in results],
        "config": {
            "use_constraints": use_constraints,
            "max_tokens": max_tokens,
            "task_count": len(task_ids),
        },
    }


def extract_code(response: str) -> str:
    """Extract Python code from model response."""
    # Try to find python code block
    if "```python" in response:
        match = re.search(r"```python\n(.*?)```", response, re.DOTALL)
        if match:
            return match.group(1).strip()

    # Try to find any code block
    if "```" in response:
        match = re.search(r"```\n?(.*?)```", response, re.DOTALL)
        if match:
            return match.group(1).strip()

    # Check if response starts with code-like content
    lines = response.strip().split("\n")
    if lines and (lines[0].startswith("def ") or lines[0].startswith("class ") or
                  lines[0].startswith("import ") or lines[0].startswith("from ")):
        # Find where code ends (look for markdown or explanation)
        code_lines = []
        for line in lines:
            if line.startswith("#") and len(line) > 50:  # Long comment = explanation
                break
            if line.strip().startswith("This ") or line.strip().startswith("The "):
                break
            code_lines.append(line)
        return "\n".join(code_lines).strip()

    # Return raw response (may be pure code)
    return response.strip()


# Keep old name for compatibility
def extract_patch(response: str) -> str:
    """Alias for extract_code."""
    return extract_code(response)


def evaluate_patch(generated: str, gold: str, problem: str) -> dict:
    """Evaluate generated code quality metrics."""
    metrics = {}

    # Check if code has actual content
    metrics["has_changes"] = bool(generated.strip()) and len(generated.strip()) > 10

    # Validate Python syntax directly on the generated code
    if generated.strip():
        try:
            ast.parse(generated)
            metrics["valid_syntax"] = True
            metrics["syntax_error"] = None
        except SyntaxError as e:
            metrics["valid_syntax"] = False
            metrics["syntax_error"] = f"Line {e.lineno}: {e.msg}" if e.lineno else str(e.msg)
    else:
        metrics["valid_syntax"] = False
        metrics["syntax_error"] = "Empty output"

    # Extract code additions from gold patch for comparison
    gold_code_lines = extract_added_lines_from_patch(gold)

    # Compute line overlap with gold patch's added code
    gen_lines = set(l.strip() for l in generated.split("\n") if l.strip())
    gold_lines = set(l.strip() for l in gold_code_lines if l.strip())
    if gold_lines:
        overlap = len(gen_lines & gold_lines) / len(gold_lines)
        metrics["line_overlap"] = overlap
    else:
        metrics["line_overlap"] = 0

    # Check if key terms from problem appear in code
    problem_terms = extract_key_terms(problem)
    code_text = generated.lower()
    term_matches = sum(1 for t in problem_terms if t.lower() in code_text)
    metrics["problem_relevance"] = term_matches / max(len(problem_terms), 1)

    # Code metrics
    metrics["code_lines"] = len([l for l in generated.split("\n") if l.strip()])
    metrics["has_function"] = "def " in generated
    metrics["has_class"] = "class " in generated

    return metrics


def extract_added_lines_from_patch(patch: str) -> list[str]:
    """Extract added lines (+ lines) from a unified diff patch."""
    added = []
    for line in patch.split("\n"):
        if line.startswith("+") and not line.startswith("+++"):
            # Remove the + prefix
            added.append(line[1:])
    return added


def extract_python_from_patch(patch: str) -> str:
    """Extract Python code from a patch for syntax checking."""
    lines = []
    for line in patch.split("\n"):
        # Skip diff headers
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("@@"):
            continue
        # Get added lines (without the + prefix)
        if line.startswith("+") and not line.startswith("+++"):
            lines.append(line[1:])
        # Or unchanged context lines
        elif line.startswith(" "):
            lines.append(line[1:])
        # Or raw code (no diff format)
        elif not line.startswith("-"):
            lines.append(line)

    code = "\n".join(lines)
    # Only return if it looks like Python
    if any(kw in code for kw in ["def ", "class ", "import ", "return ", "if ", "for "]):
        return code
    return ""


def is_valid_diff_format(patch: str) -> bool:
    """Check if patch looks like valid unified diff format."""
    lines = patch.split("\n")
    has_plus = any(l.startswith("+") for l in lines)
    has_minus = any(l.startswith("-") for l in lines)
    has_header = any(l.startswith("@@") or l.startswith("diff") for l in lines)
    return (has_plus or has_minus) and len(lines) > 1


def extract_key_terms(problem: str) -> list[str]:
    """Extract key terms from problem statement."""
    # Simple extraction: words that look like identifiers
    words = re.findall(r'\b[a-z_][a-z0-9_]{2,}\b', problem.lower())
    # Filter common words
    common = {"the", "and", "for", "that", "this", "with", "are", "not", "but", "from"}
    return [w for w in set(words) if w not in common][:20]


def compute_summary(results: list[EvalResult]) -> dict:
    """Compute summary statistics from results."""
    total = len(results)
    successful = sum(1 for r in results if r.success)
    valid_syntax = sum(1 for r in results if r.metrics.get("valid_syntax", False))
    has_changes = sum(1 for r in results if r.metrics.get("has_changes", False))

    times = [r.generation_time for r in results if r.generation_time > 0]
    overlaps = [r.metrics.get("line_overlap", 0) for r in results if r.metrics]

    # By repository
    by_repo = {}
    for r in results:
        repo = r.repo.split("/")[-1] if "/" in r.repo else r.repo
        if repo not in by_repo:
            by_repo[repo] = {"total": 0, "successful": 0}
        by_repo[repo]["total"] += 1
        if r.success:
            by_repo[repo]["successful"] += 1

    for repo in by_repo:
        by_repo[repo]["rate"] = by_repo[repo]["successful"] / by_repo[repo]["total"]

    return {
        "total": total,
        "successful": successful,
        "valid_syntax": valid_syntax,
        "has_changes": has_changes,
        "success_rate": successful / total if total > 0 else 0,
        "avg_time": sum(times) / len(times) if times else 0,
        "avg_overlap": sum(overlaps) / len(overlaps) if overlaps else 0,
        "by_repo": by_repo,
    }


@app.function(image=eval_image, timeout=3600)
def run_comparison(task_ids: list[str] = SELECTED_TASKS) -> dict:
    """
    Run comparison between constrained and unconstrained generation.
    """
    print("=" * 70)
    print("SWE-bench: Constrained vs Unconstrained Comparison")
    print("=" * 70)

    # Run with constraints
    print("\n[Phase 1] Running with Ananke constraints...")
    constrained = run_evaluation.local(task_ids, use_constraints=True)

    # Run without constraints
    print("\n[Phase 2] Running without constraints...")
    unconstrained = run_evaluation.local(task_ids, use_constraints=False)

    # Compare results
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)

    c_sum = constrained["summary"]
    u_sum = unconstrained["summary"]

    print(f"\n{'Metric':<25} {'Constrained':>15} {'Unconstrained':>15} {'Delta':>10}")
    print("-" * 70)
    print(f"{'Success rate':<25} {c_sum['success_rate']:>14.1%} {u_sum['success_rate']:>14.1%} {c_sum['success_rate']-u_sum['success_rate']:>+9.1%}")
    print(f"{'Valid syntax':<25} {c_sum['valid_syntax']:>15} {u_sum['valid_syntax']:>15} {c_sum['valid_syntax']-u_sum['valid_syntax']:>+10}")
    print(f"{'Has changes':<25} {c_sum['has_changes']:>15} {u_sum['has_changes']:>15} {c_sum['has_changes']-u_sum['has_changes']:>+10}")
    print(f"{'Avg time (s)':<25} {c_sum['avg_time']:>15.2f} {u_sum['avg_time']:>15.2f} {c_sum['avg_time']-u_sum['avg_time']:>+10.2f}")
    print(f"{'Avg gold overlap':<25} {c_sum['avg_overlap']:>14.1%} {u_sum['avg_overlap']:>14.1%} {c_sum['avg_overlap']-u_sum['avg_overlap']:>+9.1%}")

    return {
        "constrained": constrained,
        "unconstrained": unconstrained,
        "comparison": {
            "success_delta": c_sum["success_rate"] - u_sum["success_rate"],
            "syntax_delta": c_sum["valid_syntax"] - u_sum["valid_syntax"],
            "time_delta": c_sum["avg_time"] - u_sum["avg_time"],
        },
    }


@app.local_entrypoint()
def main(
    compare: bool = False,
    task_count: int = 30,
):
    """Run SWE-bench evaluation."""
    tasks = SELECTED_TASKS[:task_count]

    if compare:
        results = run_comparison.remote(tasks)
    else:
        results = run_evaluation.remote(tasks, use_constraints=True)

    # Save results
    output_path = "/tmp/swebench_eval_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
