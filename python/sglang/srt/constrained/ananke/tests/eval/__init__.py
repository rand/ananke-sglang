# Copyright Rand Arete @ Ananke 2025
# Licensed under the Apache License, Version 2.0
"""Ananke constraint evaluation framework.

This module provides evals that measure Ananke's constraint satisfaction
on real LLM outputs. The framework is organized into:

- config.py: Eval configuration and settings
- metrics.py: Metric definitions and aggregation
- runners/: Eval runners for each tier
- judges/: Judges for evaluating constraint satisfaction
- reports/: Result aggregation and reporting

See EVAL_HANDOFF.md for detailed documentation on eval tiers and strategy.
"""

from .config import EvalConfig
from .metrics import EvalMetrics, EvalResult

__all__ = ["EvalConfig", "EvalMetrics", "EvalResult"]
