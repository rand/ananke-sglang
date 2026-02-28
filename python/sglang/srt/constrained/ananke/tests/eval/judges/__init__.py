# Copyright Rand Arete @ Ananke 2025
# Licensed under the Apache License, Version 2.0
"""Judges for evaluating constraint satisfaction."""

from .regex_judge import RegexJudge

# EBNF judge requires optional dependencies
try:
    from .ebnf_judge import EbnfJudge, HAS_EBNF_SUPPORT
except ImportError:
    EbnfJudge = None  # type: ignore
    HAS_EBNF_SUPPORT = False

__all__ = ["RegexJudge", "EbnfJudge", "HAS_EBNF_SUPPORT"]
