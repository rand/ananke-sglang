# Copyright Rand Arete @ Ananke 2025
# Licensed under the Apache License, Version 2.0
"""Eval reporting and result aggregation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import EvalConfig, EvalTier, OutputMode
from ..metrics import EvalMetrics, EvalResult, SatisfactionLevel


@dataclass
class EvalReport:
    """Complete eval report with results and metadata.

    Attributes:
        name: Name of the eval run
        tier: Capability tier evaluated
        config: Configuration used
        metrics: Aggregated metrics
        timestamp: When the eval was run
        duration_seconds: Total duration of the eval
    """

    name: str
    tier: EvalTier
    config: EvalConfig
    metrics: EvalMetrics
    timestamp: datetime
    duration_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary for serialization."""
        return {
            "name": self.name,
            "tier": self.tier.name,
            "timestamp": self.timestamp.isoformat(),
            "duration_seconds": round(self.duration_seconds, 2),
            "summary": self.metrics.summary(),
            "config": {
                "tier": self.config.tier.name,
                "languages": list(self.config.languages) if self.config.languages else None,
                "domains": list(self.config.domains) if self.config.domains else None,
                "sample_count": self.config.sample_count,
                "compare_baseline": self.config.compare_baseline,
            },
        }

    def save_json(self, path: Path) -> None:
        """Save report as JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def print_summary(self) -> None:
        """Print a human-readable summary to stdout."""
        print(f"\n{'=' * 60}")
        print(f"  Eval Report: {self.name}")
        print(f"{'=' * 60}")
        print(f"  Tier: {self.tier.name}")
        print(f"  Timestamp: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Duration: {self.duration_seconds:.2f}s")
        print()
        print(f"  RESULTS")
        print(f"  {'-' * 40}")
        print(f"  Total examples:     {self.metrics.total}")
        print(f"  Satisfied:          {self.metrics.satisfied} ({self.metrics.satisfaction_rate:.1%})")
        if self.metrics.partial > 0:
            print(f"  Partial:            {self.metrics.partial} ({self.metrics.partial_rate:.1%})")
        print(f"  Failed:             {self.metrics.failed}")
        if self.metrics.errors > 0:
            print(f"  Errors:             {self.metrics.errors} ({self.metrics.error_rate:.1%})")

        if self.metrics.baseline_improvement_rate is not None:
            print()
            print(f"  Baseline improvement: {self.metrics.baseline_improvement_rate:.1%}")

        # By language
        if self.metrics.by_language:
            print()
            print(f"  BY LANGUAGE")
            print(f"  {'-' * 40}")
            for lang, lang_metrics in sorted(self.metrics.by_language.items()):
                rate = lang_metrics.satisfaction_rate
                print(f"  {lang:15} {lang_metrics.satisfied:3}/{lang_metrics.total:3} ({rate:.1%})")

        # By domain
        if self.metrics.by_domain:
            print()
            print(f"  BY DOMAIN")
            print(f"  {'-' * 40}")
            for domain, domain_metrics in sorted(self.metrics.by_domain.items()):
                rate = domain_metrics.satisfaction_rate
                print(f"  {domain:15} {domain_metrics.satisfied:3}/{domain_metrics.total:3} ({rate:.1%})")

        print(f"{'=' * 60}\n")

    def print_failures(self, max_failures: int = 10) -> None:
        """Print details of failed examples."""
        failures = [
            r for r in self.metrics.results
            if not r.satisfied and r.satisfaction_level != SatisfactionLevel.ERROR
        ]

        if not failures:
            print("No failures to report.")
            return

        print(f"\n{'=' * 60}")
        print(f"  FAILURES ({len(failures)} total, showing {min(len(failures), max_failures)})")
        print(f"{'=' * 60}")

        for result in failures[:max_failures]:
            print(f"\n  Example: {result.example_id}")
            if result.metadata:
                if "pattern" in result.metadata:
                    print(f"  Pattern: {result.metadata['pattern'][:80]}...")
                if "negative_violations" in result.metadata and result.metadata["negative_violations"]:
                    print(f"  Negative violations: {result.metadata['negative_violations']}")
            if result.output:
                output_preview = result.output[:200].replace("\n", "\\n")
                print(f"  Output preview: {output_preview}...")

    def print_errors(self) -> None:
        """Print details of error cases."""
        errors = [
            r for r in self.metrics.results
            if r.satisfaction_level == SatisfactionLevel.ERROR
        ]

        if not errors:
            print("No errors to report.")
            return

        print(f"\n{'=' * 60}")
        print(f"  ERRORS ({len(errors)} total)")
        print(f"{'=' * 60}")

        for result in errors:
            print(f"\n  Example: {result.example_id}")
            print(f"  Error: {result.error}")


def generate_report(
    name: str,
    tier: EvalTier,
    config: EvalConfig,
    metrics: EvalMetrics,
    duration_seconds: float,
) -> EvalReport:
    """Generate a complete eval report.

    Args:
        name: Name of the eval run
        tier: Capability tier evaluated
        config: Configuration used
        metrics: Aggregated metrics
        duration_seconds: Total duration of the eval

    Returns:
        Complete EvalReport
    """
    return EvalReport(
        name=name,
        tier=tier,
        config=config,
        metrics=metrics,
        timestamp=datetime.now(),
        duration_seconds=duration_seconds,
    )
