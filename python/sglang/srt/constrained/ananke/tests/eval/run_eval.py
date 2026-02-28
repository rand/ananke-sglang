#!/usr/bin/env python
# Copyright Rand Arete @ Ananke 2025
# Licensed under the Apache License, Version 2.0
"""CLI entry point for Ananke constraint evaluation.

This script runs the constraint evaluation framework:
- Level 1: Validate constraint examples (no LLM, runs locally)
- Level 2: Generation eval (requires Modal deployed model)
- Level 3: Error analysis and report generation

Usage:
    # Run Level 1 validation (local, no LLM required)
    python -m tests.eval.run_eval --level 1

    # Run Level 2 generation eval (requires Modal)
    modal run tests/eval/run_eval.py --level 2

    # Run Level 2 with sample size
    modal run tests/eval/run_eval.py --level 2 --samples 10

    # Analyze existing results
    python -m tests.eval.run_eval --analyze-failures results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# Try Modal import for remote execution
try:
    import modal
    HAS_MODAL = True
except ImportError:
    HAS_MODAL = False


def run_level1(
    languages: Optional[list[str]] = None,
    domains: Optional[list[str]] = None,
    verbose: bool = True,
) -> dict:
    """Run Level 1: Constraint validation (no LLM required).

    Validates that all constraint examples are well-formed:
    - valid_outputs match their regex/EBNF constraints
    - EBNF grammars compile successfully

    Returns:
        Results dictionary with pass/fail status
    """
    from tests.eval.runners.syntax_satisfaction import SyntaxSatisfactionRunner
    from tests.eval.config import EvalConfig

    print("=" * 70)
    print("LEVEL 1: CONSTRAINT VALIDATION")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")

    # Configure
    config = EvalConfig.tier1_syntax()
    if languages:
        config.languages = set(languages)
    if domains:
        config.domains = set(domains)

    # Run validation
    runner = SyntaxSatisfactionRunner(config)
    start = time.time()
    metrics = runner.run_validation()
    elapsed = time.time() - start

    # Build results
    summary = metrics.summary()
    results = {
        "level": 1,
        "timestamp": datetime.now().isoformat(),
        "runtime_seconds": elapsed,
        "summary": {
            "total": summary["total"],
            "passed": summary["satisfied"],
            "failed": summary["failed"],
            "errors": summary["errors"],
            "pass_rate": summary["satisfaction_rate"],
        },
        "by_language": summary.get("by_language", {}),
        "by_domain": summary.get("by_domain", {}),
    }

    # Print summary
    print(f"\nResults:")
    print(f"  Total: {summary['total']}")
    print(f"  Passed: {summary['satisfied']} ({100*summary['satisfaction_rate']:.1f}%)")
    print(f"  Failed: {summary['failed']}")
    print(f"  Errors: {summary['errors']}")
    print(f"  Runtime: {elapsed:.1f}s")

    if summary["by_language"]:
        print("\nBy Language:")
        for lang, stats in summary["by_language"].items():
            rate = stats.get("satisfaction_rate", 0) * 100
            print(f"  {lang}: {stats.get('satisfied', 0)}/{stats.get('total', 0)} ({rate:.1f}%)")

    return results


def run_level2_local(
    samples: int = 10,
    languages: Optional[list[str]] = None,
    domains: Optional[list[str]] = None,
    verbose: bool = True,
) -> dict:
    """Run Level 2 generation eval locally (requires Modal server).

    Note: This function requires the Modal server to be deployed and accessible.
    """
    if not HAS_MODAL:
        print("ERROR: Modal not installed. Run: pip install modal")
        return {"error": "Modal not installed"}

    print("=" * 70)
    print("LEVEL 2: GENERATION EVAL")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Samples: {samples}")

    # Import here to avoid issues
    from tests.eval.runners.generation_eval import GenerationEvalRunner
    from tests.eval.reports.error_analysis import generate_report
    from tests.eval.config import EvalConfig

    # Connect to Modal server
    print("\nConnecting to Modal server...")
    try:
        Qwen3CoderAnanke = modal.Cls.from_name("qwen3-coder-ananke", "Qwen3CoderAnanke")
        server = Qwen3CoderAnanke()
    except Exception as e:
        print(f"ERROR: Could not connect to Modal server: {e}")
        return {"error": str(e)}

    # Warm up server
    print("Warming up server...")
    try:
        health = server.health.remote()
        if health.get("status") != "healthy":
            print(f"Server not healthy: {health}")
            return {"error": "Server not healthy"}
        print("Server ready!")
    except Exception as e:
        print(f"ERROR: Server warmup failed: {e}")
        return {"error": str(e)}

    # Create generate function
    def generate_fn(prompt: str, constraint_spec: dict, max_tokens: int) -> str:
        result = server.generate_constrained.remote(
            prompt=prompt,
            constraint_spec=constraint_spec,
            max_tokens=max_tokens,
            temperature=0.3,
        )
        if isinstance(result, dict):
            return result.get("text", "")
        return str(result)

    # Configure
    config = EvalConfig.tier1_syntax()
    if languages:
        config.languages = set(languages)
    if domains:
        config.domains = set(domains)

    # Run eval
    runner = GenerationEvalRunner(generate_fn, config)

    def progress(current, total):
        if verbose:
            print(f"  [{current}/{total}] ", end="\r", flush=True)

    start = time.time()
    metrics, results, failures = runner.run_sample(n=samples)
    elapsed = time.time() - start

    # Generate report
    report = generate_report(metrics, results, failures)

    print("\n")
    print(report.generate_text_report())

    # Return results
    return {
        "level": 2,
        "timestamp": datetime.now().isoformat(),
        "runtime_seconds": elapsed,
        "samples": samples,
        "report": report.to_dict(),
    }


def analyze_failures(results_file: str) -> None:
    """Analyze failures from a previous eval run."""
    print("=" * 70)
    print("FAILURE ANALYSIS")
    print("=" * 70)

    with open(results_file) as f:
        data = json.load(f)

    if "report" in data:
        report = data["report"]
    else:
        report = data

    print(f"File: {results_file}")
    print(f"Timestamp: {report.get('timestamp', 'unknown')}")

    summary = report.get("summary", report)
    print(f"\nSummary:")
    print(f"  Total: {summary.get('total', 'N/A')}")
    print(f"  Passed: {summary.get('passed', 'N/A')}")
    print(f"  Failed: {summary.get('failed', 'N/A')}")
    print(f"  Pass rate: {summary.get('pass_rate', 0)*100:.1f}%")

    if "by_failure_category" in report:
        print("\nFailure Categories:")
        for cat, count in sorted(
            report["by_failure_category"].items(),
            key=lambda x: -x[1]
        ):
            print(f"  {cat}: {count}")

    if "failures" in report:
        print(f"\nDetailed Failures ({len(report['failures'])} total):")
        for i, f in enumerate(report["failures"][:10], 1):
            print(f"\n[{i}] {f.get('example_id', 'unknown')}")
            print(f"    Category: {f.get('category', 'unknown')}")
            print(f"    Details: {f.get('details', 'N/A')}")


def main():
    parser = argparse.ArgumentParser(
        description="Ananke Constraint Evaluation CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--level",
        type=int,
        choices=[1, 2],
        default=1,
        help="Eval level: 1=validation (no LLM), 2=generation (requires Modal)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10,
        help="Number of samples for Level 2 eval",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        help="Filter by languages (e.g., python rust)",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        help="Filter by domains (e.g., types imports)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path for results JSON",
    )
    parser.add_argument(
        "--analyze-failures",
        type=str,
        metavar="FILE",
        help="Analyze failures from a previous results file",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity",
    )

    args = parser.parse_args()

    # Analyze failures mode
    if args.analyze_failures:
        analyze_failures(args.analyze_failures)
        return 0

    # Run eval
    if args.level == 1:
        results = run_level1(
            languages=args.languages,
            domains=args.domains,
            verbose=not args.quiet,
        )
    else:
        results = run_level2_local(
            samples=args.samples,
            languages=args.languages,
            domains=args.domains,
            verbose=not args.quiet,
        )

    # Check for errors
    if "error" in results:
        print(f"\nERROR: {results['error']}")
        return 1

    # Save results
    if args.output:
        output_path = args.output
    else:
        timestamp = int(time.time())
        output_path = f"eval_results_level{args.level}_{timestamp}.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    # Return exit code based on pass rate
    if args.level == 1:
        return 0 if results["summary"]["pass_rate"] == 1.0 else 1
    else:
        report = results.get("report", {})
        pass_rate = report.get("summary", {}).get("pass_rate", 0)
        return 0 if pass_rate >= 0.9 else 1


# Modal entry point for remote execution
if HAS_MODAL:
    app = modal.App("ananke-constraint-eval")

    @app.local_entrypoint()
    def modal_main(
        level: int = 2,
        samples: int = 10,
    ):
        """Modal entry point for Level 2 eval."""
        if level == 1:
            results = run_level1()
        else:
            results = run_level2_local(samples=samples)

        timestamp = int(time.time())
        output_path = f"eval_results_level{level}_{timestamp}.json"

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    sys.exit(main())
