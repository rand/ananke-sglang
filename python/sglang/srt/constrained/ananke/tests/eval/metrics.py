# Copyright Rand Arete @ Ananke 2025
# Licensed under the Apache License, Version 2.0
"""Metric definitions and aggregation for evals."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


def clopper_pearson_ci(
    successes: int, total: int, confidence: float = 0.95
) -> Tuple[float, float]:
    """Calculate Clopper-Pearson exact binomial confidence interval.

    This provides conservative (wider) confidence intervals that are
    guaranteed to have at least the stated coverage.

    Args:
        successes: Number of successful outcomes
        total: Total number of trials
        confidence: Confidence level (default 0.95 for 95% CI)

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if total == 0:
        return (0.0, 1.0)

    alpha = 1 - confidence

    # Use beta distribution quantiles (scipy-free implementation)
    # For Clopper-Pearson, we use the relationship between
    # binomial and beta distributions
    try:
        from scipy.stats import beta

        if successes == 0:
            lower = 0.0
        else:
            lower = beta.ppf(alpha / 2, successes, total - successes + 1)

        if successes == total:
            upper = 1.0
        else:
            upper = beta.ppf(1 - alpha / 2, successes + 1, total - successes)

        return (lower, upper)
    except ImportError:
        # Fallback: Wilson score interval (simpler, no scipy dependency)
        return wilson_score_ci(successes, total, confidence)


def wilson_score_ci(
    successes: int, total: int, confidence: float = 0.95
) -> Tuple[float, float]:
    """Calculate Wilson score confidence interval (fallback when scipy unavailable).

    This is a simpler approximation that works well for most sample sizes.

    Args:
        successes: Number of successful outcomes
        total: Total number of trials
        confidence: Confidence level (default 0.95 for 95% CI)

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if total == 0:
        return (0.0, 1.0)

    # Z-score for confidence level (approximations for common values)
    z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z = z_scores.get(confidence, 1.96)

    p = successes / total
    denominator = 1 + z * z / total

    centre = p + z * z / (2 * total)
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total)

    lower = max(0.0, (centre - margin) / denominator)
    upper = min(1.0, (centre + margin) / denominator)

    return (lower, upper)


class SatisfactionLevel(Enum):
    """Level of constraint satisfaction."""

    FULL = "full"  # Completely satisfies constraint
    PARTIAL = "partial"  # Partially satisfies (useful for multi-part constraints)
    NONE = "none"  # Does not satisfy constraint
    ERROR = "error"  # Error during evaluation


@dataclass
class EvalResult:
    """Result of evaluating a single example.

    Attributes:
        example_id: Unique identifier for the example
        satisfied: Whether the constraint was satisfied
        satisfaction_level: Granular satisfaction level
        output: The generated output (if available)
        baseline_output: Unconstrained baseline output (if compared)
        baseline_satisfied: Whether baseline satisfied constraint
        latency_ms: Time to generate output in milliseconds
        error: Error message if evaluation failed
        metadata: Additional metadata about the evaluation
    """

    example_id: str
    satisfied: bool
    satisfaction_level: SatisfactionLevel = SatisfactionLevel.NONE
    output: Optional[str] = None
    baseline_output: Optional[str] = None
    baseline_satisfied: Optional[bool] = None
    latency_ms: Optional[float] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_error(self) -> bool:
        """Check if this result is an error."""
        return self.satisfaction_level == SatisfactionLevel.ERROR

    @property
    def improvement_over_baseline(self) -> Optional[bool]:
        """Check if constrained generation improved over baseline."""
        if self.baseline_satisfied is None:
            return None
        return self.satisfied and not self.baseline_satisfied


@dataclass
class EvalMetrics:
    """Aggregated metrics across multiple eval results.

    Attributes:
        total: Total number of examples evaluated
        satisfied: Number of examples with satisfied constraints
        partial: Number of examples with partial satisfaction
        failed: Number of examples with unsatisfied constraints
        errors: Number of examples with evaluation errors
        results: Individual results for each example
        by_language: Results grouped by language
        by_domain: Results grouped by domain
    """

    total: int = 0
    satisfied: int = 0
    partial: int = 0
    failed: int = 0
    errors: int = 0
    results: List[EvalResult] = field(default_factory=list)
    by_language: Dict[str, "EvalMetrics"] = field(default_factory=dict)
    by_domain: Dict[str, "EvalMetrics"] = field(default_factory=dict)
    by_complexity: Dict[str, "EvalMetrics"] = field(default_factory=dict)
    baseline_stats: Optional[Dict[str, int]] = None

    @property
    def satisfaction_rate(self) -> float:
        """Calculate constraint satisfaction rate."""
        if self.total == 0:
            return 0.0
        return self.satisfied / self.total

    @property
    def partial_rate(self) -> float:
        """Calculate partial satisfaction rate."""
        if self.total == 0:
            return 0.0
        return self.partial / self.total

    @property
    def error_rate(self) -> float:
        """Calculate error rate."""
        if self.total == 0:
            return 0.0
        return self.errors / self.total

    @property
    def satisfaction_ci(self) -> Tuple[float, float]:
        """Calculate 95% confidence interval for satisfaction rate."""
        return clopper_pearson_ci(self.satisfied, self.total)

    @property
    def satisfaction_rate_with_ci(self) -> str:
        """Format satisfaction rate with confidence interval."""
        rate = self.satisfaction_rate
        lower, upper = self.satisfaction_ci
        return f"{rate:.1%} [{lower:.1%}, {upper:.1%}]"

    @property
    def baseline_improvement_rate(self) -> Optional[float]:
        """Calculate rate of improvement over baseline."""
        with_baseline = [r for r in self.results if r.baseline_satisfied is not None]
        if not with_baseline:
            return None
        improvements = sum(1 for r in with_baseline if r.improvement_over_baseline)
        return improvements / len(with_baseline)

    def add_result(
        self,
        result: EvalResult,
        language: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> None:
        """Add a result to the metrics."""
        self.total += 1
        self.results.append(result)

        if result.is_error:
            self.errors += 1
        elif result.satisfaction_level == SatisfactionLevel.FULL:
            self.satisfied += 1
        elif result.satisfaction_level == SatisfactionLevel.PARTIAL:
            self.partial += 1
        else:
            self.failed += 1

        # Update language-specific metrics
        if language:
            if language not in self.by_language:
                self.by_language[language] = EvalMetrics()
            self.by_language[language].add_result(result)

        # Update domain-specific metrics
        if domain:
            if domain not in self.by_domain:
                self.by_domain[domain] = EvalMetrics()
            self.by_domain[domain].add_result(result)

    def merge(self, other: "EvalMetrics") -> "EvalMetrics":
        """Merge two metrics objects."""
        merged = EvalMetrics(
            total=self.total + other.total,
            satisfied=self.satisfied + other.satisfied,
            partial=self.partial + other.partial,
            failed=self.failed + other.failed,
            errors=self.errors + other.errors,
            results=self.results + other.results,
        )

        # Merge by_language
        for lang, metrics in other.by_language.items():
            if lang in merged.by_language:
                merged.by_language[lang] = merged.by_language[lang].merge(metrics)
            else:
                merged.by_language[lang] = metrics

        # Merge by_domain
        for domain, metrics in other.by_domain.items():
            if domain in merged.by_domain:
                merged.by_domain[domain] = merged.by_domain[domain].merge(metrics)
            else:
                merged.by_domain[domain] = metrics

        return merged

    def summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        ci_lower, ci_upper = self.satisfaction_ci
        result = {
            "total": self.total,
            "satisfied": self.satisfied,
            "partial": self.partial,
            "failed": self.failed,
            "errors": self.errors,
            "satisfaction_rate": round(self.satisfaction_rate, 4),
            "satisfaction_ci_95": {
                "lower": round(ci_lower, 4),
                "upper": round(ci_upper, 4),
            },
            "satisfaction_rate_formatted": self.satisfaction_rate_with_ci,
            "error_rate": round(self.error_rate, 4),
            "baseline_improvement_rate": (
                round(self.baseline_improvement_rate, 4)
                if self.baseline_improvement_rate is not None
                else None
            ),
            "by_language": {
                lang: metrics.summary() for lang, metrics in self.by_language.items()
            },
            "by_domain": {
                domain: metrics.summary()
                for domain, metrics in self.by_domain.items()
            },
            "by_complexity": {
                complexity: metrics.summary()
                for complexity, metrics in self.by_complexity.items()
            },
        }

        # Add baseline comparison stats if available
        if self.baseline_stats:
            total_with_baseline = sum(self.baseline_stats.values())
            if total_with_baseline > 0:
                result["baseline_comparison"] = {
                    "total_compared": total_with_baseline,
                    "constraint_helped": self.baseline_stats["constraint_helped"],
                    "constraint_hurt": self.baseline_stats["constraint_hurt"],
                    "both_passed": self.baseline_stats["both_passed"],
                    "both_failed": self.baseline_stats["both_failed"],
                    "constraint_value_rate": round(
                        self.baseline_stats["constraint_helped"] / total_with_baseline, 4
                    ),
                }

        return result
