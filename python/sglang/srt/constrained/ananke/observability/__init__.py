# Copyright Rand Arete @ Ananke 2025
# Licensed under the Apache License, Version 2.0
"""Observability module for Ananke constrained generation.

This module provides structured metrics collection, aggregation, and export
for monitoring Ananke performance in production.

Key Components:
- MetricsCollector: Central metrics collection point
- MaskMetrics: Popcount distribution tracking
- LatencyMetrics: Per-domain latency breakdown
- RelaxationMetrics: Relaxation event tracking
- MetricsExporter: Export to various backends (logs, prometheus, etc.)

Example:
    >>> from sglang.srt.constrained.ananke.observability import MetricsCollector
    >>> collector = MetricsCollector()
    >>> collector.record_mask_application(popcount=1500, vocab_size=32000, domain="types")
    >>> collector.record_domain_latency("types", 0.5)  # 0.5ms
    >>> print(collector.get_summary())
"""

from .collector import MetricsCollector
from .metrics import (
    DomainLatencyMetrics,
    MaskMetrics,
    RelaxationEventMetrics,
    RequestMetrics,
)
from .exporter import (
    MetricsExporter,
    LogExporter,
    CallbackExporter,
    PrometheusExporter,
    OTLPExporter,
)

__all__ = [
    "MetricsCollector",
    "MaskMetrics",
    "DomainLatencyMetrics",
    "RelaxationEventMetrics",
    "RequestMetrics",
    "MetricsExporter",
    "LogExporter",
    "CallbackExporter",
    "PrometheusExporter",
    "OTLPExporter",
]
