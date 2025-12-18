"""Statistical analysis utilities for Ananke evaluation.

Provides:
- Wilson score confidence intervals for proportions
- Cohen's h effect size for comparing proportions
- Power analysis for sample size estimation
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple


@dataclass
class ConfidenceInterval:
    """Confidence interval for a proportion."""

    point_estimate: float
    lower: float
    upper: float
    confidence: float
    n: int
    successes: int

    def __str__(self) -> str:
        return f"{self.point_estimate:.1%} [{self.lower:.1%}, {self.upper:.1%}] (n={self.n})"

    def contains(self, value: float) -> bool:
        """Check if value is within the confidence interval."""
        return self.lower <= value <= self.upper


def wilson_score_interval(
    successes: int,
    total: int,
    confidence: float = 0.95
) -> ConfidenceInterval:
    """Compute Wilson score confidence interval for a proportion.

    The Wilson score interval is more accurate than the normal approximation,
    especially for small samples or proportions near 0 or 1.

    Args:
        successes: Number of successes
        total: Total number of trials
        confidence: Confidence level (default 0.95 for 95% CI)

    Returns:
        ConfidenceInterval with point estimate and bounds
    """
    if total == 0:
        return ConfidenceInterval(
            point_estimate=0.0,
            lower=0.0,
            upper=1.0,
            confidence=confidence,
            n=0,
            successes=0
        )

    # Z-score for confidence level (two-tailed)
    # Common values: 90%=1.645, 95%=1.96, 99%=2.576
    if confidence >= 0.99:
        z = 2.576
    elif confidence >= 0.95:
        z = 1.96
    elif confidence >= 0.90:
        z = 1.645
    else:
        # Approximate using inverse normal (good enough for common cases)
        z = 1.96  # Default to 95%

    p_hat = successes / total
    n = total

    # Wilson score interval formula
    denominator = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denominator
    margin = z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denominator

    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)

    return ConfidenceInterval(
        point_estimate=p_hat,
        lower=lower,
        upper=upper,
        confidence=confidence,
        n=total,
        successes=successes
    )


def cohens_h(p1: float, p2: float) -> float:
    """Compute Cohen's h effect size for comparing two proportions.

    Cohen's h measures the difference between two proportions using
    the arcsine transformation, which is more appropriate than simple
    subtraction for proportions.

    Interpretation (Cohen's conventions):
    - |h| < 0.2: Small effect
    - 0.2 <= |h| < 0.5: Small to medium effect
    - 0.5 <= |h| < 0.8: Medium to large effect
    - |h| >= 0.8: Large effect

    Args:
        p1: First proportion (e.g., constrained success rate)
        p2: Second proportion (e.g., unconstrained success rate)

    Returns:
        Cohen's h effect size (positive if p1 > p2)
    """
    # Arcsine transformation
    phi1 = 2 * math.asin(math.sqrt(p1))
    phi2 = 2 * math.asin(math.sqrt(p2))

    return phi1 - phi2


def effect_size_interpretation(h: float) -> str:
    """Interpret Cohen's h effect size."""
    abs_h = abs(h)
    if abs_h < 0.2:
        return "negligible"
    elif abs_h < 0.5:
        return "small"
    elif abs_h < 0.8:
        return "medium"
    else:
        return "large"


@dataclass
class ComparisonResult:
    """Result of comparing two conditions."""

    condition1: str
    condition2: str
    rate1: ConfidenceInterval
    rate2: ConfidenceInterval
    delta: float  # rate1 - rate2
    cohens_h: float
    effect_interpretation: str
    significant: bool  # CIs don't overlap

    def __str__(self) -> str:
        direction = "+" if self.delta > 0 else ""
        return (
            f"{self.condition1} vs {self.condition2}: "
            f"{direction}{self.delta:.1%} "
            f"(h={self.cohens_h:.2f}, {self.effect_interpretation})"
        )


def compare_conditions(
    name1: str,
    successes1: int,
    total1: int,
    name2: str,
    successes2: int,
    total2: int,
    confidence: float = 0.95
) -> ComparisonResult:
    """Compare two experimental conditions.

    Args:
        name1: Name of first condition
        successes1: Successes in first condition
        total1: Total trials in first condition
        name2: Name of second condition
        successes2: Successes in second condition
        total2: Total trials in second condition
        confidence: Confidence level for intervals

    Returns:
        ComparisonResult with statistical analysis
    """
    ci1 = wilson_score_interval(successes1, total1, confidence)
    ci2 = wilson_score_interval(successes2, total2, confidence)

    delta = ci1.point_estimate - ci2.point_estimate
    h = cohens_h(ci1.point_estimate, ci2.point_estimate)

    # Check if CIs overlap (simple significance test)
    significant = ci1.lower > ci2.upper or ci2.lower > ci1.upper

    return ComparisonResult(
        condition1=name1,
        condition2=name2,
        rate1=ci1,
        rate2=ci2,
        delta=delta,
        cohens_h=h,
        effect_interpretation=effect_size_interpretation(h),
        significant=significant
    )


def required_sample_size(
    p1: float,
    p2: float,
    alpha: float = 0.05,
    power: float = 0.80
) -> int:
    """Estimate required sample size per group for comparing proportions.

    Uses approximation based on arcsine transformation.

    Args:
        p1: Expected proportion in group 1
        p2: Expected proportion in group 2
        alpha: Significance level (default 0.05)
        power: Desired statistical power (default 0.80)

    Returns:
        Required sample size per group
    """
    # Z-scores
    z_alpha = 1.96 if alpha <= 0.05 else 1.645  # Two-tailed
    z_beta = 0.84 if power >= 0.80 else 0.52  # One-tailed

    # Cohen's h
    h = abs(cohens_h(p1, p2))

    if h == 0:
        return float('inf')  # Can't detect no effect

    # Sample size formula for comparing proportions
    n = 2 * ((z_alpha + z_beta) / h) ** 2

    return math.ceil(n)


@dataclass
class EvaluationSummary:
    """Summary of an evaluation run."""

    test_name: str
    condition: str
    total: int
    successes: int
    failures: int
    ci: ConfidenceInterval
    pass_threshold: float  # Required rate to pass (e.g., 1.0 for 100%)
    passed: bool

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{status}] {self.test_name} ({self.condition}): "
            f"{self.ci.point_estimate:.1%} "
            f"[{self.ci.lower:.1%}, {self.ci.upper:.1%}] "
            f"(need {self.pass_threshold:.0%})"
        )


def evaluate_test(
    test_name: str,
    condition: str,
    successes: int,
    total: int,
    pass_threshold: float,
    confidence: float = 0.95
) -> EvaluationSummary:
    """Evaluate a single test against a pass threshold.

    Args:
        test_name: Name of the test
        condition: Experimental condition
        successes: Number of successes
        total: Total trials
        pass_threshold: Required success rate to pass (0.0 to 1.0)
        confidence: Confidence level for CI

    Returns:
        EvaluationSummary with pass/fail determination
    """
    ci = wilson_score_interval(successes, total, confidence)

    # For pass_threshold of 1.0 (100%), we need lower bound >= threshold
    # For pass_threshold of 0.0 (0%), we need upper bound <= threshold
    if pass_threshold >= 0.5:
        # High threshold: lower bound must meet it
        passed = ci.lower >= pass_threshold - 0.01  # Small tolerance
    else:
        # Low threshold: upper bound must not exceed it
        passed = ci.upper <= pass_threshold + 0.01

    return EvaluationSummary(
        test_name=test_name,
        condition=condition,
        total=total,
        successes=successes,
        failures=total - successes,
        ci=ci,
        pass_threshold=pass_threshold,
        passed=passed
    )


if __name__ == "__main__":
    # Demo usage
    print("Wilson Score Interval Examples:")
    print("-" * 40)

    # Example: 18/20 successes
    ci = wilson_score_interval(18, 20)
    print(f"18/20 successes: {ci}")

    # Example: 0/20 successes (edge case)
    ci = wilson_score_interval(0, 20)
    print(f"0/20 successes: {ci}")

    # Example: 20/20 successes (edge case)
    ci = wilson_score_interval(20, 20)
    print(f"20/20 successes: {ci}")

    print("\nCohen's h Examples:")
    print("-" * 40)

    # Compare 90% vs 70%
    h = cohens_h(0.90, 0.70)
    print(f"90% vs 70%: h={h:.2f} ({effect_size_interpretation(h)})")

    # Compare 100% vs 70%
    h = cohens_h(1.00, 0.70)
    print(f"100% vs 70%: h={h:.2f} ({effect_size_interpretation(h)})")

    print("\nCondition Comparison:")
    print("-" * 40)

    result = compare_conditions(
        "constrained", 28, 30,
        "unconstrained", 18, 30
    )
    print(result)
    print(f"  Significant: {result.significant}")

    print("\nRequired Sample Sizes:")
    print("-" * 40)

    n = required_sample_size(0.95, 0.70)
    print(f"Detect 95% vs 70% (25% delta): n={n} per group")

    n = required_sample_size(1.00, 0.85)
    print(f"Detect 100% vs 85% (15% delta): n={n} per group")
