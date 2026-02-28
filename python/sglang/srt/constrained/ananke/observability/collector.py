# Copyright Rand Arete @ Ananke 2025
# Licensed under the Apache License, Version 2.0
"""Central metrics collection for Ananke observability."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .metrics import (
    DomainLatencyMetrics,
    MaskMetrics,
    RelaxationEventMetrics,
    RequestMetrics,
)

logger = logging.getLogger(__name__)


@dataclass
class MetricsCollector:
    """Central collector for Ananke metrics.

    Thread-safe collector that aggregates metrics across requests
    and provides summary statistics and export capabilities.

    Attributes:
        vocab_size: Vocabulary size for selectivity calculations
        max_history: Maximum number of requests to keep in history
        enable_detailed: Enable detailed per-request tracking
    """

    vocab_size: int = 32000
    max_history: int = 1000
    enable_detailed: bool = True

    # Aggregate metrics
    _mask_metrics: MaskMetrics = field(default_factory=MaskMetrics)
    _domain_latencies: Dict[str, DomainLatencyMetrics] = field(default_factory=dict)
    _relaxation_metrics: RelaxationEventMetrics = field(
        default_factory=RelaxationEventMetrics
    )

    # Request tracking
    _active_requests: Dict[str, RequestMetrics] = field(default_factory=dict)
    _completed_requests: List[RequestMetrics] = field(default_factory=list)

    # Thread safety
    _lock: threading.Lock = field(default_factory=threading.Lock)

    # Callbacks for real-time alerts
    _alert_callbacks: List[Callable[[str, Dict[str, Any]], None]] = field(
        default_factory=list
    )

    def __post_init__(self) -> None:
        """Initialize metrics with vocab_size."""
        self._mask_metrics = MaskMetrics(vocab_size=self.vocab_size)

    def start_request(self, request_id: Optional[str] = None) -> str:
        """Start tracking a new request.

        Args:
            request_id: Optional request ID (generated if not provided)

        Returns:
            The request ID
        """
        request_id = request_id or str(uuid.uuid4())[:8]
        with self._lock:
            self._active_requests[request_id] = RequestMetrics(request_id=request_id)
        return request_id

    def finish_request(
        self,
        request_id: str,
        tokens: int,
        early_terminated: bool = False,
    ) -> Optional[RequestMetrics]:
        """Finish tracking a request.

        Args:
            request_id: The request ID
            tokens: Number of tokens generated
            early_terminated: Whether generation terminated early

        Returns:
            The completed RequestMetrics, or None if not found
        """
        with self._lock:
            if request_id not in self._active_requests:
                return None

            metrics = self._active_requests.pop(request_id)
            metrics.finish(tokens, early_terminated)

            if self.enable_detailed:
                self._completed_requests.append(metrics)
                # Trim history
                if len(self._completed_requests) > self.max_history:
                    self._completed_requests = self._completed_requests[-self.max_history :]

            return metrics

    def record_mask_application(
        self,
        popcount: int,
        domain: Optional[str] = None,
        latency_ms: Optional[float] = None,
        request_id: Optional[str] = None,
        domain_latencies: Optional[Dict[str, float]] = None,
    ) -> None:
        """Record a mask application.

        Args:
            popcount: Number of allowed tokens
            domain: Domain that produced the mask
            latency_ms: Total mask computation latency
            request_id: Optional request ID for per-request tracking
            domain_latencies: Per-domain latency breakdown
        """
        with self._lock:
            # Aggregate metrics
            self._mask_metrics.record(popcount)

            # Domain latencies
            if domain_latencies:
                for d, lat in domain_latencies.items():
                    if d not in self._domain_latencies:
                        self._domain_latencies[d] = DomainLatencyMetrics(d)
                    self._domain_latencies[d].record(lat)

            # Per-request tracking
            if request_id and request_id in self._active_requests:
                self._active_requests[request_id].record_mask_application(
                    latency_ms or 0.0, popcount, domain_latencies
                )

            # Check for alerts
            selectivity = 1.0 - (popcount / self.vocab_size) if self.vocab_size else 0
            if selectivity > 0.99:  # Very restrictive
                self._trigger_alert(
                    "high_selectivity",
                    {
                        "popcount": popcount,
                        "selectivity": selectivity,
                        "domain": domain,
                    },
                )

    def record_domain_latency(self, domain: str, latency_ms: float) -> None:
        """Record latency for a specific domain.

        Args:
            domain: Domain name
            latency_ms: Latency in milliseconds
        """
        with self._lock:
            if domain not in self._domain_latencies:
                self._domain_latencies[domain] = DomainLatencyMetrics(domain)
            self._domain_latencies[domain].record(latency_ms)

    def record_relaxation(
        self,
        domain: str,
        popcount_before: int,
        popcount_after: int,
        request_id: Optional[str] = None,
    ) -> None:
        """Record a constraint relaxation event.

        Args:
            domain: Domain that was relaxed
            popcount_before: Popcount before relaxation
            popcount_after: Popcount after relaxation
            request_id: Optional request ID
        """
        with self._lock:
            self._relaxation_metrics.record_relaxation(
                domain, popcount_before, popcount_after
            )

            if request_id and request_id in self._active_requests:
                self._active_requests[request_id].relaxation_events += 1

            # Trigger alert for relaxation
            self._trigger_alert(
                "relaxation_event",
                {
                    "domain": domain,
                    "popcount_before": popcount_before,
                    "popcount_after": popcount_after,
                    "gain": popcount_after - popcount_before,
                },
            )

    def register_alert_callback(
        self, callback: Callable[[str, Dict[str, Any]], None]
    ) -> None:
        """Register a callback for alerts.

        Args:
            callback: Function(alert_type, data) to call on alerts
        """
        with self._lock:
            self._alert_callbacks.append(callback)

    def _trigger_alert(self, alert_type: str, data: Dict[str, Any]) -> None:
        """Trigger alert callbacks (called with lock held)."""
        for callback in self._alert_callbacks:
            try:
                callback(alert_type, data)
            except Exception as e:
                logger.warning(f"Alert callback error: {e}")

    def get_mask_metrics(self) -> MaskMetrics:
        """Get aggregate mask metrics."""
        with self._lock:
            return self._mask_metrics

    def get_domain_latencies(self) -> Dict[str, DomainLatencyMetrics]:
        """Get per-domain latency metrics."""
        with self._lock:
            return dict(self._domain_latencies)

    def get_relaxation_metrics(self) -> RelaxationEventMetrics:
        """Get relaxation event metrics."""
        with self._lock:
            return self._relaxation_metrics

    def get_recent_requests(self, n: int = 10) -> List[RequestMetrics]:
        """Get the N most recent completed requests."""
        with self._lock:
            return self._completed_requests[-n:]

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all metrics.

        Returns:
            Dictionary with aggregate statistics
        """
        with self._lock:
            return {
                "mask_metrics": self._mask_metrics.to_dict(),
                "domain_latencies": {
                    k: v.to_dict() for k, v in self._domain_latencies.items()
                },
                "relaxation": self._relaxation_metrics.to_dict(),
                "active_requests": len(self._active_requests),
                "completed_requests": len(self._completed_requests),
            }

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._mask_metrics = MaskMetrics(vocab_size=self.vocab_size)
            self._domain_latencies.clear()
            self._relaxation_metrics = RelaxationEventMetrics()
            self._active_requests.clear()
            self._completed_requests.clear()


# Global collector instance
_global_collector: Optional[MetricsCollector] = None
_global_lock = threading.Lock()


def get_global_collector() -> MetricsCollector:
    """Get or create the global metrics collector."""
    global _global_collector
    with _global_lock:
        if _global_collector is None:
            _global_collector = MetricsCollector()
        return _global_collector


def set_global_collector(collector: MetricsCollector) -> None:
    """Set the global metrics collector."""
    global _global_collector
    with _global_lock:
        _global_collector = collector
