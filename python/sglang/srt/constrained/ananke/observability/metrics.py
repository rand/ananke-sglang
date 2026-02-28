# Copyright Rand Arete @ Ananke 2025
# Licensed under the Apache License, Version 2.0
"""Metric data structures for Ananke observability."""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MaskMetrics:
    """Metrics for mask application and popcount distribution.

    Tracks the distribution of allowed tokens (popcount) across
    mask applications to help understand constraint selectivity.

    Attributes:
        popcounts: List of popcount values recorded
        vocab_size: Vocabulary size for computing selectivity
        domain: Domain that produced this mask (if applicable)
    """

    popcounts: List[int] = field(default_factory=list)
    vocab_size: int = 32000
    domain: Optional[str] = None

    def record(self, popcount: int) -> None:
        """Record a popcount observation."""
        self.popcounts.append(popcount)

    @property
    def count(self) -> int:
        """Number of observations."""
        return len(self.popcounts)

    @property
    def mean(self) -> float:
        """Mean popcount."""
        return statistics.mean(self.popcounts) if self.popcounts else 0.0

    @property
    def median(self) -> float:
        """Median popcount."""
        return statistics.median(self.popcounts) if self.popcounts else 0.0

    @property
    def stdev(self) -> float:
        """Standard deviation of popcount."""
        return statistics.stdev(self.popcounts) if len(self.popcounts) > 1 else 0.0

    @property
    def min(self) -> int:
        """Minimum popcount."""
        return min(self.popcounts) if self.popcounts else 0

    @property
    def max(self) -> int:
        """Maximum popcount."""
        return max(self.popcounts) if self.popcounts else 0

    @property
    def mean_selectivity(self) -> float:
        """Mean selectivity (1 - popcount/vocab_size)."""
        if not self.popcounts or self.vocab_size == 0:
            return 0.0
        return 1.0 - (self.mean / self.vocab_size)

    def percentile(self, p: float) -> float:
        """Calculate percentile (0-100)."""
        if not self.popcounts:
            return 0.0
        sorted_vals = sorted(self.popcounts)
        k = (len(sorted_vals) - 1) * (p / 100.0)
        f = int(k)
        c = f + 1 if f + 1 < len(sorted_vals) else f
        return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)

    def histogram(self, bins: int = 10) -> Dict[str, int]:
        """Compute histogram of popcount distribution."""
        if not self.popcounts:
            return {}

        min_val, max_val = min(self.popcounts), max(self.popcounts)
        if min_val == max_val:
            return {f"{min_val}": len(self.popcounts)}

        bin_width = (max_val - min_val) / bins
        histogram: Dict[str, int] = {}

        for val in self.popcounts:
            bin_idx = min(int((val - min_val) / bin_width), bins - 1)
            bin_start = min_val + bin_idx * bin_width
            bin_end = bin_start + bin_width
            label = f"{int(bin_start)}-{int(bin_end)}"
            histogram[label] = histogram.get(label, 0) + 1

        return histogram

    def to_dict(self) -> Dict[str, Any]:
        """Export metrics as dictionary."""
        return {
            "domain": self.domain,
            "count": self.count,
            "mean": round(self.mean, 2),
            "median": round(self.median, 2),
            "stdev": round(self.stdev, 2),
            "min": self.min,
            "max": self.max,
            "mean_selectivity": round(self.mean_selectivity, 4),
            "p50": round(self.percentile(50), 2),
            "p90": round(self.percentile(90), 2),
            "p99": round(self.percentile(99), 2),
        }


@dataclass
class DomainLatencyMetrics:
    """Metrics for per-domain latency tracking.

    Tracks the time spent in each constraint domain during
    mask computation for performance analysis.

    Attributes:
        domain: Domain name
        latencies_ms: List of latency observations in milliseconds
    """

    domain: str
    latencies_ms: List[float] = field(default_factory=list)

    def record(self, latency_ms: float) -> None:
        """Record a latency observation in milliseconds."""
        self.latencies_ms.append(latency_ms)

    @property
    def count(self) -> int:
        """Number of observations."""
        return len(self.latencies_ms)

    @property
    def total_ms(self) -> float:
        """Total time spent in this domain."""
        return sum(self.latencies_ms)

    @property
    def mean_ms(self) -> float:
        """Mean latency in milliseconds."""
        return statistics.mean(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def median_ms(self) -> float:
        """Median latency in milliseconds."""
        return statistics.median(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def p99_ms(self) -> float:
        """99th percentile latency."""
        if not self.latencies_ms:
            return 0.0
        sorted_vals = sorted(self.latencies_ms)
        idx = int(len(sorted_vals) * 0.99)
        return sorted_vals[min(idx, len(sorted_vals) - 1)]

    def to_dict(self) -> Dict[str, Any]:
        """Export metrics as dictionary."""
        return {
            "domain": self.domain,
            "count": self.count,
            "total_ms": round(self.total_ms, 3),
            "mean_ms": round(self.mean_ms, 3),
            "median_ms": round(self.median_ms, 3),
            "p99_ms": round(self.p99_ms, 3),
        }


@dataclass
class RelaxationEventMetrics:
    """Metrics for constraint relaxation events.

    Tracks when and why constraints are relaxed during generation,
    useful for understanding constraint tightness and tuning.

    Attributes:
        total_events: Total relaxation events
        events_by_domain: Count of relaxations per domain
        popcount_before: Popcount values before relaxation
        popcount_after: Popcount values after relaxation
        threshold_hits: How often threshold was hit
    """

    total_events: int = 0
    events_by_domain: Dict[str, int] = field(default_factory=dict)
    popcount_before: List[int] = field(default_factory=list)
    popcount_after: List[int] = field(default_factory=list)
    threshold_hits: int = 0

    def record_relaxation(
        self,
        domain: str,
        popcount_before: int,
        popcount_after: int,
        hit_threshold: bool = True,
    ) -> None:
        """Record a relaxation event."""
        self.total_events += 1
        self.events_by_domain[domain] = self.events_by_domain.get(domain, 0) + 1
        self.popcount_before.append(popcount_before)
        self.popcount_after.append(popcount_after)
        if hit_threshold:
            self.threshold_hits += 1

    @property
    def relaxation_rate(self) -> float:
        """Rate of relaxation events (events / total mask applications)."""
        # This needs to be computed externally with total applications
        return 0.0

    @property
    def mean_popcount_gain(self) -> float:
        """Mean increase in popcount from relaxation."""
        if not self.popcount_before:
            return 0.0
        gains = [
            after - before
            for before, after in zip(self.popcount_before, self.popcount_after)
        ]
        return statistics.mean(gains) if gains else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Export metrics as dictionary."""
        return {
            "total_events": self.total_events,
            "events_by_domain": dict(self.events_by_domain),
            "threshold_hits": self.threshold_hits,
            "mean_popcount_gain": round(self.mean_popcount_gain, 2),
        }


@dataclass
class RequestMetrics:
    """Aggregate metrics for a single generation request.

    Combines mask, latency, and relaxation metrics for
    a complete picture of request performance.
    """

    request_id: str
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    tokens_generated: int = 0
    mask_applications: int = 0
    total_mask_latency_ms: float = 0.0
    domain_latencies: Dict[str, DomainLatencyMetrics] = field(default_factory=dict)
    final_popcounts: List[int] = field(default_factory=list)
    relaxation_events: int = 0
    early_terminated: bool = False

    def record_mask_application(
        self,
        latency_ms: float,
        final_popcount: int,
        domain_latencies: Optional[Dict[str, float]] = None,
    ) -> None:
        """Record a mask application."""
        self.mask_applications += 1
        self.total_mask_latency_ms += latency_ms
        self.final_popcounts.append(final_popcount)

        if domain_latencies:
            for domain, lat in domain_latencies.items():
                if domain not in self.domain_latencies:
                    self.domain_latencies[domain] = DomainLatencyMetrics(domain)
                self.domain_latencies[domain].record(lat)

    def finish(self, tokens: int, early_terminated: bool = False) -> None:
        """Mark request as finished."""
        self.end_time = time.time()
        self.tokens_generated = tokens
        self.early_terminated = early_terminated

    @property
    def duration_ms(self) -> float:
        """Total request duration in milliseconds."""
        if self.end_time is None:
            return (time.time() - self.start_time) * 1000
        return (self.end_time - self.start_time) * 1000

    @property
    def mean_mask_latency_ms(self) -> float:
        """Mean latency per mask application."""
        if self.mask_applications == 0:
            return 0.0
        return self.total_mask_latency_ms / self.mask_applications

    @property
    def tokens_per_second(self) -> float:
        """Generation throughput."""
        duration_s = self.duration_ms / 1000
        if duration_s == 0:
            return 0.0
        return self.tokens_generated / duration_s

    def to_dict(self) -> Dict[str, Any]:
        """Export metrics as dictionary."""
        return {
            "request_id": self.request_id,
            "duration_ms": round(self.duration_ms, 2),
            "tokens_generated": self.tokens_generated,
            "tokens_per_second": round(self.tokens_per_second, 2),
            "mask_applications": self.mask_applications,
            "mean_mask_latency_ms": round(self.mean_mask_latency_ms, 3),
            "relaxation_events": self.relaxation_events,
            "early_terminated": self.early_terminated,
            "domain_latencies": {
                k: v.to_dict() for k, v in self.domain_latencies.items()
            },
        }
