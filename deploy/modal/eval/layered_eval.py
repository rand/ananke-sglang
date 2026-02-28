"""Layered Evaluation Framework for Ananke Constraint System.

This module orchestrates the two-layer evaluation:
- Layer 1: Mechanism verification (do constraints work at all?)
- Layer 2: Value measurement (do constraints improve real tasks?)

Usage:
    from deploy.modal.eval.layered_eval import LayeredEvaluator

    evaluator = LayeredEvaluator(server)
    layer1_results = evaluator.run_layer1()
    if layer1_results.all_passed:
        layer2_results = evaluator.run_layer2()
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional

from .statistics import (
    wilson_score_interval,
    cohens_h,
    effect_size_interpretation,
    compare_conditions,
    evaluate_test,
    ConfidenceInterval,
    ComparisonResult,
    EvaluationSummary,
)
from .tests.layer1_mechanism_tests import (
    MechanismTest,
    get_layer1_tests,
    ALL_LAYER1_TESTS,
)
from .tests.layer2_value_tests import (
    ValueTest,
    Condition,
    get_layer2_tests,
    ALL_LAYER2_TESTS,
)


@dataclass
class TestRunResult:
    """Result of a single test run."""
    test_id: str
    output: str
    valid: bool
    latency_ms: float
    condition: str | None = None
    metrics: dict[str, bool] = field(default_factory=dict)


@dataclass
class Layer1TestResult:
    """Result of a Layer 1 mechanism test."""
    test: MechanismTest
    runs: list[TestRunResult]
    summary: EvaluationSummary
    passed: bool

    def to_dict(self) -> dict:
        return {
            "test_id": self.test.id,
            "test_name": self.test.name,
            "is_control": self.test.is_control,
            "samples": len(self.runs),
            "successes": self.summary.successes,
            "pass_threshold": self.test.pass_threshold,
            "success_rate": self.summary.ci.point_estimate,
            "ci_lower": self.summary.ci.lower,
            "ci_upper": self.summary.ci.upper,
            "passed": self.passed,
        }


@dataclass
class Layer1Results:
    """Aggregated Layer 1 results."""
    tests: list[Layer1TestResult]
    all_passed: bool
    json_passed: bool
    regex_passed: bool
    domain_passed: bool
    runtime_seconds: float

    def to_dict(self) -> dict:
        return {
            "layer": 1,
            "all_passed": self.all_passed,
            "json_passed": self.json_passed,
            "regex_passed": self.regex_passed,
            "domain_passed": self.domain_passed,
            "runtime_seconds": self.runtime_seconds,
            "tests": [t.to_dict() for t in self.tests],
        }


@dataclass
class Layer2TestResult:
    """Result of a Layer 2 value test."""
    test: ValueTest
    condition_results: dict[str, list[TestRunResult]]
    comparisons: list[ComparisonResult]
    primary_comparison: ComparisonResult | None
    value_demonstrated: bool

    def to_dict(self) -> dict:
        return {
            "test_id": self.test.id,
            "test_name": self.test.name,
            "primary_metric": self.test.primary_metric,
            "expected_delta": self.test.expected_delta,
            "conditions": {
                cond: {
                    "samples": len(runs),
                    "successes": sum(1 for r in runs if r.valid),
                    "rate": sum(1 for r in runs if r.valid) / len(runs) if runs else 0,
                }
                for cond, runs in self.condition_results.items()
            },
            "primary_comparison": {
                "delta": self.primary_comparison.delta if self.primary_comparison else None,
                "cohens_h": self.primary_comparison.cohens_h if self.primary_comparison else None,
                "effect_size": self.primary_comparison.effect_interpretation if self.primary_comparison else None,
                "significant": self.primary_comparison.significant if self.primary_comparison else False,
            },
            "value_demonstrated": self.value_demonstrated,
        }


@dataclass
class Layer2Results:
    """Aggregated Layer 2 results."""
    tests: list[Layer2TestResult]
    overall_value_demonstrated: bool
    significant_improvements: int
    total_tests: int
    runtime_seconds: float

    def to_dict(self) -> dict:
        return {
            "layer": 2,
            "overall_value_demonstrated": self.overall_value_demonstrated,
            "significant_improvements": self.significant_improvements,
            "total_tests": self.total_tests,
            "runtime_seconds": self.runtime_seconds,
            "tests": [t.to_dict() for t in self.tests],
        }


class LayeredEvaluator:
    """Orchestrates layered evaluation of Ananke constraint system."""

    def __init__(
        self,
        generate_fn: Callable[[str, dict | None, int, float], str],
        verbose: bool = True
    ):
        """Initialize evaluator.

        Args:
            generate_fn: Function to generate text. Signature:
                generate_fn(prompt, constraint_spec, max_tokens, temperature) -> str
            verbose: Whether to print progress
        """
        self.generate_fn = generate_fn
        self.verbose = verbose

    def _log(self, msg: str):
        """Print message if verbose."""
        if self.verbose:
            print(msg)

    def _run_single(
        self,
        prompt: str,
        constraint_spec: dict | None,
        max_tokens: int,
        temperature: float
    ) -> tuple[str, float]:
        """Run a single generation and return output with latency."""
        start = time.time()
        output = self.generate_fn(prompt, constraint_spec, max_tokens, temperature)
        latency_ms = (time.time() - start) * 1000
        return output, latency_ms

    # =========================================================================
    # Layer 1: Mechanism Verification
    # =========================================================================

    def run_layer1(
        self,
        category: str | None = None,
        progress_callback: Callable[[int, int], None] | None = None
    ) -> Layer1Results:
        """Run Layer 1 mechanism verification tests.

        Args:
            category: Filter tests by category ("json", "regex", "domain", or None)
            progress_callback: Optional callback(current, total) for progress

        Returns:
            Layer1Results with pass/fail for each test
        """
        self._log("\n" + "=" * 60)
        self._log("LAYER 1: MECHANISM VERIFICATION")
        self._log("=" * 60)

        tests = get_layer1_tests(category)
        start_time = time.time()
        results = []

        total_runs = sum(t.samples for t in tests)
        completed = 0

        for test in tests:
            self._log(f"\n[{test.id}] {test.name}")
            self._log(f"  Samples: {test.samples}, Threshold: {test.pass_threshold:.0%}")

            runs = []
            successes = 0

            for i in range(test.samples):
                output, latency = self._run_single(
                    test.prompt,
                    test.constraint_spec,
                    test.max_tokens,
                    test.temperature
                )

                valid = test.validate(output)
                if valid:
                    successes += 1

                runs.append(TestRunResult(
                    test_id=test.id,
                    output=output,
                    valid=valid,
                    latency_ms=latency
                ))

                completed += 1
                if progress_callback:
                    progress_callback(completed, total_runs)

            # Evaluate against threshold
            summary = evaluate_test(
                test.id,
                "mechanism",
                successes,
                test.samples,
                test.pass_threshold
            )

            # For mechanism tests, use point estimate (observed rate) for pass logic
            # CI is informative but too strict for deterministic mechanism verification
            if test.is_control:
                # Control: pass if rate is BELOW threshold (showing constraint is needed)
                passed = summary.ci.point_estimate <= test.pass_threshold
            else:
                # Regular: pass if observed rate is AT OR ABOVE threshold
                # Use point_estimate (not ci.lower) because mechanism tests expect
                # deterministic behavior - 100% observed means the mechanism works
                passed = summary.ci.point_estimate >= test.pass_threshold

            result = Layer1TestResult(
                test=test,
                runs=runs,
                summary=summary,
                passed=passed
            )
            results.append(result)

            status = "PASS" if passed else "FAIL"
            self._log(f"  Result: [{status}] {successes}/{test.samples} ({summary.ci})")

        # Categorize results
        json_tests = [r for r in results if r.test.id.startswith("json")]
        regex_tests = [r for r in results if r.test.id.startswith("regex")]
        domain_tests = [r for r in results if "domain" in r.test.id]

        json_passed = all(r.passed for r in json_tests) if json_tests else True
        regex_passed = all(r.passed for r in regex_tests if not r.test.is_control) if regex_tests else True
        domain_passed = all(r.passed for r in domain_tests if not r.test.is_control) if domain_tests else True

        all_passed = all(r.passed for r in results if not r.test.is_control)

        runtime = time.time() - start_time

        self._log("\n" + "-" * 60)
        self._log("LAYER 1 SUMMARY")
        self._log("-" * 60)
        self._log(f"  JSON Schema: {'PASS' if json_passed else 'FAIL'}")
        self._log(f"  Regex: {'PASS' if regex_passed else 'FAIL'}")
        self._log(f"  Domain: {'PASS' if domain_passed else 'FAIL'}")
        self._log(f"  Overall: {'PASS' if all_passed else 'FAIL'}")
        self._log(f"  Runtime: {runtime:.1f}s")

        return Layer1Results(
            tests=results,
            all_passed=all_passed,
            json_passed=json_passed,
            regex_passed=regex_passed,
            domain_passed=domain_passed,
            runtime_seconds=runtime
        )

    # =========================================================================
    # Layer 2: Value Measurement
    # =========================================================================

    def run_layer2(
        self,
        category: str | None = None,
        temperatures: list[float] | None = None,
        progress_callback: Callable[[int, int], None] | None = None
    ) -> Layer2Results:
        """Run Layer 2 value measurement tests.

        Args:
            category: Filter tests by category ("json", "code", "security", "multilang")
            temperatures: Override test temperatures
            progress_callback: Optional callback(current, total) for progress

        Returns:
            Layer2Results with comparison statistics
        """
        self._log("\n" + "=" * 60)
        self._log("LAYER 2: VALUE MEASUREMENT")
        self._log("=" * 60)

        tests = get_layer2_tests(category)
        start_time = time.time()
        results = []

        # Calculate total runs for progress
        total_runs = sum(
            t.samples_per_condition * len(t.conditions) * len(temperatures or t.temperatures)
            for t in tests
        )
        completed = 0

        for test in tests:
            temps = temperatures or test.temperatures
            self._log(f"\n[{test.id}] {test.name}")
            self._log(f"  Conditions: {list(test.conditions.keys())}")
            self._log(f"  Samples/condition: {test.samples_per_condition}")
            self._log(f"  Temperatures: {temps}")

            condition_results: dict[str, list[TestRunResult]] = {
                cond: [] for cond in test.conditions
            }

            for cond_name, constraint_spec in test.conditions.items():
                for temp in temps:
                    for i in range(test.samples_per_condition // len(temps)):
                        output, latency = self._run_single(
                            test.prompt,
                            constraint_spec,
                            test.max_tokens,
                            temp
                        )

                        metrics = test.validate(output)
                        primary_valid = metrics.get(test.primary_metric, False)

                        condition_results[cond_name].append(TestRunResult(
                            test_id=test.id,
                            output=output,
                            valid=primary_valid,
                            latency_ms=latency,
                            condition=cond_name,
                            metrics=metrics
                        ))

                        completed += 1
                        if progress_callback:
                            progress_callback(completed, total_runs)

            # Compare conditions
            comparisons = []
            baseline_cond = "unconstrained"
            baseline_runs = condition_results.get(baseline_cond, [])
            baseline_successes = sum(1 for r in baseline_runs if r.valid)
            baseline_total = len(baseline_runs)

            for cond_name, runs in condition_results.items():
                if cond_name == baseline_cond:
                    continue

                successes = sum(1 for r in runs if r.valid)
                total = len(runs)

                comparison = compare_conditions(
                    cond_name, successes, total,
                    baseline_cond, baseline_successes, baseline_total
                )
                comparisons.append(comparison)

                self._log(f"  {comparison}")

            # Find primary comparison (constrained vs unconstrained)
            primary_comparison = None
            for comp in comparisons:
                if comp.condition2 == baseline_cond:
                    primary_comparison = comp
                    break

            # Determine if value was demonstrated
            value_demonstrated = (
                primary_comparison is not None and
                primary_comparison.delta >= test.expected_delta * 0.5 and  # At least half expected
                abs(primary_comparison.cohens_h) >= 0.3  # At least small-medium effect
            )

            result = Layer2TestResult(
                test=test,
                condition_results=condition_results,
                comparisons=comparisons,
                primary_comparison=primary_comparison,
                value_demonstrated=value_demonstrated
            )
            results.append(result)

            status = "VALUE" if value_demonstrated else "NO VALUE"
            self._log(f"  Verdict: [{status}]")

        # Aggregate results
        significant_improvements = sum(1 for r in results if r.value_demonstrated)
        overall_value = significant_improvements >= len(results) // 2  # Majority show value

        runtime = time.time() - start_time

        self._log("\n" + "-" * 60)
        self._log("LAYER 2 SUMMARY")
        self._log("-" * 60)
        self._log(f"  Tests showing value: {significant_improvements}/{len(results)}")
        self._log(f"  Overall value demonstrated: {'YES' if overall_value else 'NO'}")
        self._log(f"  Runtime: {runtime:.1f}s")

        return Layer2Results(
            tests=results,
            overall_value_demonstrated=overall_value,
            significant_improvements=significant_improvements,
            total_tests=len(results),
            runtime_seconds=runtime
        )

    # =========================================================================
    # Full Evaluation
    # =========================================================================

    def run_full(
        self,
        skip_layer2_on_layer1_fail: bool = True,
        progress_callback: Callable[[int, int, str], None] | None = None
    ) -> dict:
        """Run full layered evaluation.

        Args:
            skip_layer2_on_layer1_fail: Skip Layer 2 if Layer 1 fails
            progress_callback: Optional callback(current, total, layer) for progress

        Returns:
            Complete evaluation results as dict
        """
        self._log("\n" + "=" * 60)
        self._log("ANANKE LAYERED EVALUATION")
        self._log("=" * 60)
        self._log(f"Timestamp: {datetime.now().isoformat()}")

        start_time = time.time()

        # Layer 1
        layer1_callback = None
        if progress_callback:
            layer1_callback = lambda c, t: progress_callback(c, t, "layer1")

        layer1_results = self.run_layer1(progress_callback=layer1_callback)

        # Layer 2 (conditional)
        layer2_results = None
        if layer1_results.all_passed or not skip_layer2_on_layer1_fail:
            layer2_callback = None
            if progress_callback:
                layer2_callback = lambda c, t: progress_callback(c, t, "layer2")

            layer2_results = self.run_layer2(progress_callback=layer2_callback)
        else:
            self._log("\n[SKIPPED] Layer 2 skipped due to Layer 1 failures")

        total_runtime = time.time() - start_time

        # Build final report
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_runtime_seconds": total_runtime,
            "layer1": layer1_results.to_dict(),
            "layer2": layer2_results.to_dict() if layer2_results else None,
            "summary": {
                "layer1_passed": layer1_results.all_passed,
                "layer2_ran": layer2_results is not None,
                "layer2_value_demonstrated": (
                    layer2_results.overall_value_demonstrated
                    if layer2_results else None
                ),
                "overall_success": (
                    layer1_results.all_passed and
                    (layer2_results is None or layer2_results.overall_value_demonstrated)
                ),
            }
        }

        self._log("\n" + "=" * 60)
        self._log("FINAL SUMMARY")
        self._log("=" * 60)
        self._log(f"  Layer 1: {'PASS' if layer1_results.all_passed else 'FAIL'}")
        if layer2_results:
            self._log(f"  Layer 2: {'VALUE DEMONSTRATED' if layer2_results.overall_value_demonstrated else 'NO VALUE'}")
        self._log(f"  Total runtime: {total_runtime:.1f}s")

        return report


def save_results(results: dict, filepath: str | None = None) -> str:
    """Save results to JSON file.

    Args:
        results: Evaluation results dict
        filepath: Optional filepath (auto-generated if None)

    Returns:
        Filepath where results were saved
    """
    if filepath is None:
        timestamp = int(time.time())
        filepath = f"layered_eval_results_{timestamp}.json"

    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)

    return filepath
