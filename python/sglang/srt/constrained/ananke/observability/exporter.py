# Copyright Rand Arete @ Ananke 2025
# Licensed under the Apache License, Version 2.0
"""Metrics exporters for various backends."""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from .collector import MetricsCollector

logger = logging.getLogger(__name__)


class MetricsExporter(ABC):
    """Base class for metrics exporters."""

    @abstractmethod
    def export(self, collector: MetricsCollector) -> None:
        """Export metrics from the collector."""
        pass

    @abstractmethod
    def export_alert(self, alert_type: str, data: Dict[str, Any]) -> None:
        """Export an alert."""
        pass


@dataclass
class LogExporter(MetricsExporter):
    """Export metrics to the logging system.

    Attributes:
        log_level: Logging level for metrics (default: INFO)
        alert_level: Logging level for alerts (default: WARNING)
        format: Output format ('json' or 'text')
    """

    log_level: int = logging.INFO
    alert_level: int = logging.WARNING
    format: str = "json"

    def export(self, collector: MetricsCollector) -> None:
        """Export metrics to logs."""
        summary = collector.get_summary()

        if self.format == "json":
            message = json.dumps(summary, indent=2)
        else:
            message = self._format_text(summary)

        logger.log(self.log_level, f"Ananke Metrics:\n{message}")

    def export_alert(self, alert_type: str, data: Dict[str, Any]) -> None:
        """Export an alert to logs."""
        if self.format == "json":
            message = json.dumps({"alert_type": alert_type, **data})
        else:
            message = f"{alert_type}: {data}"

        logger.log(self.alert_level, f"Ananke Alert: {message}")

    def _format_text(self, summary: Dict[str, Any]) -> str:
        """Format summary as human-readable text."""
        lines = []

        # Mask metrics
        mask = summary.get("mask_metrics", {})
        lines.append(f"Mask Metrics:")
        lines.append(f"  Count: {mask.get('count', 0)}")
        lines.append(f"  Mean popcount: {mask.get('mean', 0):.1f}")
        lines.append(f"  Selectivity: {mask.get('mean_selectivity', 0):.2%}")
        lines.append(f"  P50/P90/P99: {mask.get('p50', 0):.0f}/{mask.get('p90', 0):.0f}/{mask.get('p99', 0):.0f}")

        # Domain latencies
        latencies = summary.get("domain_latencies", {})
        if latencies:
            lines.append(f"\nDomain Latencies:")
            for domain, metrics in latencies.items():
                lines.append(
                    f"  {domain}: mean={metrics.get('mean_ms', 0):.3f}ms, "
                    f"p99={metrics.get('p99_ms', 0):.3f}ms, "
                    f"total={metrics.get('total_ms', 0):.1f}ms"
                )

        # Relaxation
        relaxation = summary.get("relaxation", {})
        if relaxation.get("total_events", 0) > 0:
            lines.append(f"\nRelaxation Events:")
            lines.append(f"  Total: {relaxation.get('total_events', 0)}")
            lines.append(f"  By domain: {relaxation.get('events_by_domain', {})}")
            lines.append(f"  Mean gain: {relaxation.get('mean_popcount_gain', 0):.1f}")

        return "\n".join(lines)


@dataclass
class CallbackExporter(MetricsExporter):
    """Export metrics via callback functions.

    Useful for integrating with external monitoring systems.

    Attributes:
        metrics_callback: Called with metrics summary dict
        alert_callback: Called with (alert_type, data) tuple
    """

    metrics_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    alert_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None

    def export(self, collector: MetricsCollector) -> None:
        """Export metrics via callback."""
        if self.metrics_callback:
            summary = collector.get_summary()
            try:
                self.metrics_callback(summary)
            except Exception as e:
                logger.warning(f"Metrics callback error: {e}")

    def export_alert(self, alert_type: str, data: Dict[str, Any]) -> None:
        """Export alert via callback."""
        if self.alert_callback:
            try:
                self.alert_callback(alert_type, data)
            except Exception as e:
                logger.warning(f"Alert callback error: {e}")


class PeriodicExporter:
    """Periodically export metrics from a collector.

    Runs in a background thread and exports metrics at a fixed interval.

    Example:
        >>> exporter = PeriodicExporter(
        ...     collector=collector,
        ...     exporters=[LogExporter()],
        ...     interval_seconds=60,
        ... )
        >>> exporter.start()
        >>> # ... later ...
        >>> exporter.stop()
    """

    def __init__(
        self,
        collector: MetricsCollector,
        exporters: list[MetricsExporter],
        interval_seconds: float = 60.0,
    ):
        """Initialize periodic exporter.

        Args:
            collector: Metrics collector to export from
            exporters: List of exporters to use
            interval_seconds: Export interval
        """
        self.collector = collector
        self.exporters = exporters
        self.interval_seconds = interval_seconds
        self._running = False
        self._thread: Optional[Any] = None

    def start(self) -> None:
        """Start periodic export."""
        if self._running:
            return

        import threading

        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop periodic export."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None

    def _run(self) -> None:
        """Background export loop."""
        while self._running:
            try:
                for exporter in self.exporters:
                    exporter.export(self.collector)
            except Exception as e:
                logger.warning(f"Periodic export error: {e}")

            time.sleep(self.interval_seconds)

    def export_now(self) -> None:
        """Trigger an immediate export."""
        for exporter in self.exporters:
            exporter.export(self.collector)
