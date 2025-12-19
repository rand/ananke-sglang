# Copyright Rand Arete @ Ananke 2025
# Licensed under the Apache License, Version 2.0
"""Error analysis report for generation eval.

Following Hamel's eval methodology:
- Error analysis reveals actual failure modes
- Categorization enables targeted improvements
- Focus on traces (what actually happened) not metrics
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any

from ..metrics import EvalMetrics
from ..runners.generation_eval import GenerationResult, FailureAnalysis


@dataclass
class ErrorAnalysisReport:
    """Structured error analysis report."""

    # Summary stats
    total_examples: int
    passed: int
    failed: int
    pass_rate: float

    # Breakdown by category
    by_language: Dict[str, Dict[str, int]]
    by_domain: Dict[str, Dict[str, int]]
    by_failure_category: Dict[str, int]
    by_constraint_type: Dict[str, Dict[str, int]]

    # Detailed failures
    failures: List[FailureAnalysis]

    # Timing
    avg_latency_ms: float
    p95_latency_ms: float

    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "summary": {
                "total": self.total_examples,
                "passed": self.passed,
                "failed": self.failed,
                "pass_rate": self.pass_rate,
            },
            "by_language": self.by_language,
            "by_domain": self.by_domain,
            "by_failure_category": self.by_failure_category,
            "by_constraint_type": self.by_constraint_type,
            "latency": {
                "avg_ms": self.avg_latency_ms,
                "p95_ms": self.p95_latency_ms,
            },
            "failures": [
                {
                    "example_id": f.example_id,
                    "language": f.language,
                    "domain": f.domain,
                    "constraint_type": f.constraint_type,
                    "category": f.failure_category,
                    "details": f.details,
                    "expected_pattern": f.expected_pattern[:100] + "..." if len(f.expected_pattern) > 100 else f.expected_pattern,
                    "actual_output": f.actual_output[:200] + "..." if len(f.actual_output) > 200 else f.actual_output,
                }
                for f in self.failures
            ],
        }

    def generate_text_report(self) -> str:
        """Generate human-readable text report."""
        lines = [
            "=" * 70,
            "ANANKE CONSTRAINT GENERATION ERROR ANALYSIS",
            "=" * 70,
            f"Timestamp: {self.timestamp}",
            "",
            "SUMMARY",
            "-" * 70,
            f"Total examples: {self.total_examples}",
            f"Passed: {self.passed} ({100*self.pass_rate:.1f}%)",
            f"Failed: {self.failed} ({100*(1-self.pass_rate):.1f}%)",
            f"Avg latency: {self.avg_latency_ms:.1f}ms",
            f"P95 latency: {self.p95_latency_ms:.1f}ms",
            "",
            "BY LANGUAGE",
            "-" * 70,
        ]

        for lang, stats in sorted(self.by_language.items()):
            total = stats.get("total", 0)
            passed = stats.get("passed", 0)
            rate = passed / total * 100 if total > 0 else 0
            lines.append(f"  {lang}: {passed}/{total} ({rate:.1f}%)")

        lines.extend([
            "",
            "BY DOMAIN",
            "-" * 70,
        ])

        for domain, stats in sorted(self.by_domain.items()):
            total = stats.get("total", 0)
            passed = stats.get("passed", 0)
            rate = passed / total * 100 if total > 0 else 0
            lines.append(f"  {domain}: {passed}/{total} ({rate:.1f}%)")

        lines.extend([
            "",
            "BY CONSTRAINT TYPE",
            "-" * 70,
        ])

        for ctype, stats in sorted(self.by_constraint_type.items()):
            total = stats.get("total", 0)
            passed = stats.get("passed", 0)
            rate = passed / total * 100 if total > 0 else 0
            lines.append(f"  {ctype}: {passed}/{total} ({rate:.1f}%)")

        if self.by_failure_category:
            lines.extend([
                "",
                "FAILURE CATEGORIES",
                "-" * 70,
            ])

            for category, count in sorted(
                self.by_failure_category.items(),
                key=lambda x: -x[1]
            ):
                lines.append(f"  {category}: {count}")

        if self.failures:
            lines.extend([
                "",
                "DETAILED FAILURES",
                "-" * 70,
            ])

            for i, f in enumerate(self.failures[:20], 1):
                lines.extend([
                    f"",
                    f"[{i}] {f.example_id}",
                    f"    Language: {f.language}, Domain: {f.domain}",
                    f"    Constraint: {f.constraint_type}",
                    f"    Category: {f.failure_category}",
                    f"    Details: {f.details}",
                    f"    Output: {f.actual_output[:100]}{'...' if len(f.actual_output) > 100 else ''}",
                ])

            if len(self.failures) > 20:
                lines.append(f"\n  ... and {len(self.failures) - 20} more failures")

        lines.extend([
            "",
            "=" * 70,
            "END OF REPORT",
            "=" * 70,
        ])

        return "\n".join(lines)


def generate_report(
    metrics: EvalMetrics,
    results: List[GenerationResult],
    failures: List[FailureAnalysis],
) -> ErrorAnalysisReport:
    """Generate error analysis report from eval results.

    Args:
        metrics: Aggregated metrics from eval run
        results: All generation results
        failures: Categorized failure analyses

    Returns:
        ErrorAnalysisReport with breakdown and categorization
    """
    # Basic stats
    total = len(results)
    passed = sum(1 for r in results if r.matches_constraint)
    failed = total - passed
    pass_rate = passed / total if total > 0 else 0

    # By language
    by_language: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "passed": 0})
    for r in results:
        by_language[r.language]["total"] += 1
        if r.matches_constraint:
            by_language[r.language]["passed"] += 1

    # By domain
    by_domain: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "passed": 0})
    for r in results:
        by_domain[r.domain]["total"] += 1
        if r.matches_constraint:
            by_domain[r.domain]["passed"] += 1

    # By constraint type
    by_constraint_type: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "passed": 0})
    for r in results:
        by_constraint_type[r.constraint_type]["total"] += 1
        if r.matches_constraint:
            by_constraint_type[r.constraint_type]["passed"] += 1

    # By failure category
    by_failure_category: Dict[str, int] = defaultdict(int)
    for f in failures:
        by_failure_category[f.failure_category] += 1

    # Latency stats
    latencies = [r.latency_ms for r in results if r.latency_ms > 0]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    sorted_latencies = sorted(latencies)
    p95_idx = int(len(sorted_latencies) * 0.95)
    p95_latency = sorted_latencies[p95_idx] if sorted_latencies else 0

    return ErrorAnalysisReport(
        total_examples=total,
        passed=passed,
        failed=failed,
        pass_rate=pass_rate,
        by_language=dict(by_language),
        by_domain=dict(by_domain),
        by_failure_category=dict(by_failure_category),
        by_constraint_type=dict(by_constraint_type),
        failures=failures,
        avg_latency_ms=avg_latency,
        p95_latency_ms=p95_latency,
    )
