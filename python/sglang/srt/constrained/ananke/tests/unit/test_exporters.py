# Copyright Rand Arete @ Ananke 2025
# Licensed under the Apache License, Version 2.0
"""Tests for Prometheus and OTLP observability exporters."""

from __future__ import annotations

import logging
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from observability.collector import MetricsCollector
from observability.exporter import (
    CallbackExporter,
    LogExporter,
    MetricsExporter,
    PeriodicExporter,
)


@pytest.fixture
def collector() -> MetricsCollector:
    """Create a collector with sample data."""
    c = MetricsCollector(vocab_size=32000)
    c.record_mask_application(popcount=15000, domain="syntax", latency_ms=0.5)
    c.record_mask_application(popcount=8000, domain="types", latency_ms=1.2)
    c.record_mask_application(popcount=20000, domain="syntax", latency_ms=0.3)
    c.record_domain_latency("syntax", 0.5)
    c.record_domain_latency("types", 1.2)
    c.record_relaxation("types", popcount_before=100, popcount_after=5000)
    return c


class TestPrometheusExporter:
    """Tests for PrometheusExporter."""

    def test_import_error_without_package(self) -> None:
        """PrometheusExporter raises ImportError without prometheus_client."""
        with patch.dict("sys.modules", {"prometheus_client": None}):
            from observability.exporter import PrometheusExporter

            with pytest.raises(ImportError, match="prometheus_client"):
                PrometheusExporter()

    def test_creation_with_mock(self) -> None:
        """PrometheusExporter can be created when prometheus_client is available."""
        prometheus_client = pytest.importorskip(
            "prometheus_client", reason="prometheus_client required"
        )
        from observability.exporter import PrometheusExporter

        exporter = PrometheusExporter(namespace="test_ananke")
        assert exporter.namespace == "test_ananke"
        assert exporter.push_gateway is None
        assert exporter.registry is not None

    def test_export_populates_metrics(self, collector: MetricsCollector) -> None:
        """export() populates Prometheus metrics from collector."""
        prometheus_client = pytest.importorskip(
            "prometheus_client", reason="prometheus_client required"
        )
        from observability.exporter import PrometheusExporter

        exporter = PrometheusExporter(namespace="test_export")
        exporter.export(collector)

        # Verify metrics were set by generating output from the registry
        output = prometheus_client.generate_latest(exporter.registry).decode()
        assert "test_export_mask_applications_total" in output
        assert "test_export_mask_selectivity" in output
        assert "test_export_active_requests" in output

    def test_export_domain_latencies(self, collector: MetricsCollector) -> None:
        """export() records per-domain latency metrics."""
        prometheus_client = pytest.importorskip(
            "prometheus_client", reason="prometheus_client required"
        )
        from observability.exporter import PrometheusExporter

        exporter = PrometheusExporter(namespace="test_latency")
        exporter.export(collector)

        output = prometheus_client.generate_latest(exporter.registry).decode()
        assert "test_latency_domain_latency_mean_ms" in output
        assert 'domain="syntax"' in output or "domain=\"types\"" in output

    def test_export_alert_logs_debug(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """export_alert() logs at debug level."""
        pytest.importorskip("prometheus_client", reason="prometheus_client required")
        from observability.exporter import PrometheusExporter

        exporter = PrometheusExporter()
        with caplog.at_level(logging.DEBUG, logger="observability.exporter"):
            exporter.export_alert("high_selectivity", {"popcount": 50})

        assert any("Prometheus alert" in r.message for r in caplog.records)

    def test_push_gateway_mode(self, collector: MetricsCollector) -> None:
        """export() pushes to gateway when push_gateway is configured."""
        pc = pytest.importorskip(
            "prometheus_client", reason="prometheus_client required"
        )
        from observability.exporter import PrometheusExporter

        exporter = PrometheusExporter(
            push_gateway="http://localhost:9091", job_name="test"
        )
        assert exporter.push_gateway == "http://localhost:9091"

        # Mock push_to_gateway to avoid actual network call
        with patch.object(pc, "push_to_gateway") as mock_push:
            exporter.export(collector)
            mock_push.assert_called_once_with(
                "http://localhost:9091",
                job="test",
                registry=exporter.registry,
            )


class TestOTLPExporter:
    """Tests for OTLPExporter."""

    def test_import_error_without_package(self) -> None:
        """OTLPExporter raises ImportError without opentelemetry-sdk."""
        with patch.dict(
            "sys.modules",
            {"opentelemetry": None, "opentelemetry.metrics": None},
        ):
            from observability.exporter import OTLPExporter

            with pytest.raises(ImportError, match="opentelemetry-sdk"):
                OTLPExporter()

    def test_creation_with_mock(self) -> None:
        """OTLPExporter can be created when opentelemetry packages are available."""
        otel_sdk = pytest.importorskip(
            "opentelemetry.sdk", reason="opentelemetry-sdk required"
        )
        pytest.importorskip(
            "opentelemetry.exporter.otlp.proto.grpc.metric_exporter",
            reason="opentelemetry-exporter-otlp-proto-grpc required",
        )
        from observability.exporter import OTLPExporter

        exporter = OTLPExporter(
            service_name="test_ananke", endpoint="http://localhost:4317"
        )
        assert exporter.service_name == "test_ananke"
        assert exporter.endpoint == "http://localhost:4317"
        assert exporter.protocol == "grpc"

    def test_export_computes_deltas(self) -> None:
        """export() tracks deltas for counter-style metrics."""
        otel_sdk = pytest.importorskip(
            "opentelemetry.sdk", reason="opentelemetry-sdk required"
        )
        pytest.importorskip(
            "opentelemetry.exporter.otlp.proto.grpc.metric_exporter",
            reason="opentelemetry-exporter-otlp-proto-grpc required",
        )
        from observability.exporter import OTLPExporter

        exporter = OTLPExporter(service_name="test_delta")
        collector = MetricsCollector(vocab_size=32000)

        # First export with some data
        collector.record_mask_application(popcount=10000, domain="syntax")
        exporter.export(collector)
        assert exporter._last_mask_count == 1

        # Second export with more data
        collector.record_mask_application(popcount=15000, domain="types")
        exporter.export(collector)
        assert exporter._last_mask_count == 2

    def test_export_alert_logs_debug(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """export_alert() logs at debug level."""
        otel_sdk = pytest.importorskip(
            "opentelemetry.sdk", reason="opentelemetry-sdk required"
        )
        pytest.importorskip(
            "opentelemetry.exporter.otlp.proto.grpc.metric_exporter",
            reason="opentelemetry-exporter-otlp-proto-grpc required",
        )
        from observability.exporter import OTLPExporter

        exporter = OTLPExporter()
        with caplog.at_level(logging.DEBUG, logger="observability.exporter"):
            exporter.export_alert("test_alert", {"key": "value"})

        assert any("OTLP alert" in r.message for r in caplog.records)

    def test_shutdown(self) -> None:
        """shutdown() calls provider.shutdown()."""
        otel_sdk = pytest.importorskip(
            "opentelemetry.sdk", reason="opentelemetry-sdk required"
        )
        pytest.importorskip(
            "opentelemetry.exporter.otlp.proto.grpc.metric_exporter",
            reason="opentelemetry-exporter-otlp-proto-grpc required",
        )
        from observability.exporter import OTLPExporter

        exporter = OTLPExporter(service_name="test_shutdown")
        # Should not raise
        exporter.shutdown()


class TestExistingExporters:
    """Regression tests for existing exporters (LogExporter, CallbackExporter)."""

    def test_log_exporter_json(
        self, collector: MetricsCollector, caplog: pytest.LogCaptureFixture
    ) -> None:
        """LogExporter in json format exports valid JSON."""
        exporter = LogExporter(format="json")
        with caplog.at_level(logging.INFO):
            exporter.export(collector)

        assert any("Ananke Metrics" in r.message for r in caplog.records)

    def test_log_exporter_text(
        self, collector: MetricsCollector, caplog: pytest.LogCaptureFixture
    ) -> None:
        """LogExporter in text format exports readable text."""
        exporter = LogExporter(format="text")
        with caplog.at_level(logging.INFO):
            exporter.export(collector)

        assert any("Mask Metrics" in r.message for r in caplog.records)

    def test_callback_exporter(self, collector: MetricsCollector) -> None:
        """CallbackExporter calls the provided callback."""
        captured: Dict[str, Any] = {}

        def on_metrics(summary: Dict[str, Any]) -> None:
            captured.update(summary)

        exporter = CallbackExporter(metrics_callback=on_metrics)
        exporter.export(collector)

        assert "mask_metrics" in captured
        assert captured["mask_metrics"]["count"] == 3

    def test_periodic_exporter_export_now(
        self, collector: MetricsCollector
    ) -> None:
        """PeriodicExporter.export_now() triggers immediate export."""
        captured: list[Dict[str, Any]] = []

        def on_metrics(summary: Dict[str, Any]) -> None:
            captured.append(summary)

        callback_exporter = CallbackExporter(metrics_callback=on_metrics)
        periodic = PeriodicExporter(
            collector=collector, exporters=[callback_exporter], interval_seconds=999
        )
        periodic.export_now()

        assert len(captured) == 1
        assert captured[0]["mask_metrics"]["count"] == 3
