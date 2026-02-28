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


class PrometheusExporter(MetricsExporter):
    """Export metrics to Prometheus.

    Supports both pull mode (HTTP endpoint) and push mode (Pushgateway).
    Requires the ``prometheus_client`` package (optional dependency).

    Attributes:
        namespace: Metric name prefix (default: "ananke")
        push_gateway: Optional Pushgateway URL for push mode
        job_name: Job name for Pushgateway (default: "ananke")
    """

    def __init__(
        self,
        namespace: str = "ananke",
        push_gateway: Optional[str] = None,
        job_name: str = "ananke",
    ) -> None:
        try:
            import prometheus_client
        except ImportError:
            raise ImportError(
                "prometheus_client is required for PrometheusExporter. "
                "Install with: pip install prometheus-client"
            )

        self._pc = prometheus_client
        self.namespace = namespace
        self.push_gateway = push_gateway
        self.job_name = job_name

        self._registry = prometheus_client.CollectorRegistry()
        self._build_metrics()

    def _build_metrics(self) -> None:
        pc = self._pc
        ns = self.namespace
        reg = self._registry

        self._mask_count = pc.Counter(
            f"{ns}_mask_applications_total",
            "Total mask applications",
            registry=reg,
        )
        self._mask_popcount = pc.Gauge(
            f"{ns}_mask_popcount_mean",
            "Mean popcount of masks",
            registry=reg,
        )
        self._mask_selectivity = pc.Gauge(
            f"{ns}_mask_selectivity",
            "Mean mask selectivity (1 - popcount/vocab_size)",
            registry=reg,
        )
        self._mask_p99 = pc.Gauge(
            f"{ns}_mask_popcount_p99",
            "99th percentile mask popcount",
            registry=reg,
        )
        self._domain_latency = pc.Gauge(
            f"{ns}_domain_latency_mean_ms",
            "Mean domain latency in milliseconds",
            ["domain"],
            registry=reg,
        )
        self._domain_latency_p99 = pc.Gauge(
            f"{ns}_domain_latency_p99_ms",
            "P99 domain latency in milliseconds",
            ["domain"],
            registry=reg,
        )
        self._relaxation_total = pc.Counter(
            f"{ns}_relaxation_events_total",
            "Total constraint relaxation events",
            registry=reg,
        )
        self._relaxation_by_domain = pc.Counter(
            f"{ns}_relaxation_events_by_domain",
            "Relaxation events by domain",
            ["domain"],
            registry=reg,
        )
        self._active_requests = pc.Gauge(
            f"{ns}_active_requests",
            "Currently active requests",
            registry=reg,
        )
        self._completed_requests = pc.Counter(
            f"{ns}_completed_requests_total",
            "Total completed requests",
            registry=reg,
        )

    def export(self, collector: MetricsCollector) -> None:
        summary = collector.get_summary()

        mask = summary.get("mask_metrics", {})
        count = mask.get("count", 0)
        if count > 0:
            self._mask_count._value.set(count)
            self._mask_popcount.set(mask.get("mean", 0))
            self._mask_selectivity.set(mask.get("mean_selectivity", 0))
            self._mask_p99.set(mask.get("p99", 0))

        for domain, metrics in summary.get("domain_latencies", {}).items():
            self._domain_latency.labels(domain=domain).set(
                metrics.get("mean_ms", 0)
            )
            self._domain_latency_p99.labels(domain=domain).set(
                metrics.get("p99_ms", 0)
            )

        relaxation = summary.get("relaxation", {})
        total_events = relaxation.get("total_events", 0)
        if total_events > 0:
            self._relaxation_total._value.set(total_events)
            for domain, count in relaxation.get("events_by_domain", {}).items():
                self._relaxation_by_domain.labels(domain=domain)._value.set(count)

        self._active_requests.set(summary.get("active_requests", 0))
        self._completed_requests._value.set(summary.get("completed_requests", 0))

        if self.push_gateway:
            try:
                self._pc.push_to_gateway(
                    self.push_gateway,
                    job=self.job_name,
                    registry=self._registry,
                )
            except Exception as e:
                logger.warning(f"Prometheus push failed: {e}")

    def export_alert(self, alert_type: str, data: Dict[str, Any]) -> None:
        # Prometheus doesn't have a native alert export mechanism.
        # Alerts are typically derived from metric thresholds in Alertmanager.
        logger.debug(f"Prometheus alert (use Alertmanager rules): {alert_type}: {data}")

    @property
    def registry(self) -> Any:
        """Access the Prometheus registry for custom HTTP server setup."""
        return self._registry


class OTLPExporter(MetricsExporter):
    """Export metrics via OpenTelemetry Protocol (OTLP).

    Supports gRPC and HTTP exporters. Requires the ``opentelemetry-sdk``
    and ``opentelemetry-exporter-otlp`` packages (optional dependencies).

    Attributes:
        service_name: Service name in OTLP resource (default: "ananke")
        endpoint: OTLP endpoint URL (default: "http://localhost:4317")
        protocol: Transport protocol - "grpc" or "http" (default: "grpc")
        insecure: Use insecure connection for gRPC (default: True)
    """

    def __init__(
        self,
        service_name: str = "ananke",
        endpoint: str = "http://localhost:4317",
        protocol: str = "grpc",
        insecure: bool = True,
    ) -> None:
        try:
            from opentelemetry import metrics as otel_metrics
            from opentelemetry.sdk.metrics import MeterProvider
            from opentelemetry.sdk.resources import Resource
        except ImportError:
            raise ImportError(
                "opentelemetry-sdk is required for OTLPExporter. "
                "Install with: pip install opentelemetry-sdk opentelemetry-exporter-otlp"
            )

        self.service_name = service_name
        self.endpoint = endpoint
        self.protocol = protocol

        resource = Resource.create({"service.name": service_name})

        if protocol == "grpc":
            try:
                from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
                    OTLPMetricExporter,
                )
            except ImportError:
                raise ImportError(
                    "opentelemetry-exporter-otlp-proto-grpc is required for gRPC protocol. "
                    "Install with: pip install opentelemetry-exporter-otlp-proto-grpc"
                )
            otlp_exporter = OTLPMetricExporter(
                endpoint=endpoint, insecure=insecure
            )
        else:
            try:
                from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
                    OTLPMetricExporter,
                )
            except ImportError:
                raise ImportError(
                    "opentelemetry-exporter-otlp-proto-http is required for HTTP protocol. "
                    "Install with: pip install opentelemetry-exporter-otlp-proto-http"
                )
            otlp_exporter = OTLPMetricExporter(endpoint=endpoint)

        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

        reader = PeriodicExportingMetricReader(
            otlp_exporter, export_interval_millis=60000
        )
        self._provider = MeterProvider(resource=resource, metric_readers=[reader])
        self._meter = self._provider.get_meter("ananke.observability")

        self._build_instruments()
        self._last_mask_count = 0
        self._last_relaxation_total = 0
        self._last_completed = 0

    def _build_instruments(self) -> None:
        self._mask_counter = self._meter.create_counter(
            "ananke.mask_applications",
            description="Total mask applications",
        )
        self._mask_popcount_hist = self._meter.create_histogram(
            "ananke.mask_popcount",
            description="Mask popcount distribution",
            unit="tokens",
        )
        self._mask_selectivity_gauge = self._meter.create_gauge(
            "ananke.mask_selectivity",
            description="Mean mask selectivity",
        )
        self._domain_latency_hist = self._meter.create_histogram(
            "ananke.domain_latency",
            description="Per-domain latency",
            unit="ms",
        )
        self._relaxation_counter = self._meter.create_counter(
            "ananke.relaxation_events",
            description="Total relaxation events",
        )
        self._active_gauge = self._meter.create_gauge(
            "ananke.active_requests",
            description="Currently active requests",
        )

    def export(self, collector: MetricsCollector) -> None:
        summary = collector.get_summary()

        mask = summary.get("mask_metrics", {})
        current_count = mask.get("count", 0)
        delta = current_count - self._last_mask_count
        if delta > 0:
            self._mask_counter.add(delta)
            self._last_mask_count = current_count

        mean_popcount = mask.get("mean", 0)
        if mean_popcount > 0:
            self._mask_popcount_hist.record(mean_popcount)
        self._mask_selectivity_gauge.set(mask.get("mean_selectivity", 0))

        for domain, metrics in summary.get("domain_latencies", {}).items():
            mean_ms = metrics.get("mean_ms", 0)
            if mean_ms > 0:
                self._domain_latency_hist.record(
                    mean_ms, {"domain": domain}
                )

        relaxation = summary.get("relaxation", {})
        relax_total = relaxation.get("total_events", 0)
        relax_delta = relax_total - self._last_relaxation_total
        if relax_delta > 0:
            self._relaxation_counter.add(relax_delta)
            self._last_relaxation_total = relax_total

        self._active_gauge.set(summary.get("active_requests", 0))

    def export_alert(self, alert_type: str, data: Dict[str, Any]) -> None:
        # OTLP alerts are typically handled via traces/logs, not metrics.
        # Record as a metric event with attributes for now.
        logger.debug(f"OTLP alert (consider traces/logs pipeline): {alert_type}: {data}")

    def shutdown(self) -> None:
        """Shut down the OTLP provider, flushing pending metrics."""
        self._provider.shutdown()


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
