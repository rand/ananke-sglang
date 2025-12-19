# Copyright Rand Arete @ Ananke 2025
# Licensed under the Apache License, Version 2.0
"""Eval configuration and settings."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Set


# Language-specific constraint configuration
# Tuned based on eval results showing language-specific pass rates
LANGUAGE_CONSTRAINT_CONFIG: Dict[str, Dict[str, Any]] = {
    "typescript": {
        "max_tokens": 1024,
        "evaluation_strategy": "TIERED",
        "enable_domains": ["types", "imports", "syntax"],
        "expected_pass_rate": 0.84,  # Best performer
    },
    "go": {
        "max_tokens": 1024,
        "evaluation_strategy": "TIERED",
        "enable_domains": ["types", "imports", "syntax"],
        "expected_pass_rate": 0.62,
    },
    "kotlin": {
        "max_tokens": 1024,
        "evaluation_strategy": "TIERED",
        "enable_domains": ["types", "imports", "syntax", "controlflow"],
        "expected_pass_rate": 0.55,
    },
    "rust": {
        "max_tokens": 1024,
        "evaluation_strategy": "TIERED",
        "enable_domains": ["types", "imports", "syntax"],
        "expected_pass_rate": 0.50,
    },
    "swift": {
        "max_tokens": 1024,
        "evaluation_strategy": "TIERED",
        "enable_domains": ["types", "imports", "syntax"],
        "expected_pass_rate": 0.50,
    },
    "python": {
        "max_tokens": 1024,
        "evaluation_strategy": "TIERED",
        "enable_domains": ["types", "imports", "syntax", "semantics"],
        "expected_pass_rate": 0.44,
    },
    "zig": {
        "max_tokens": 1024,  # Zig code is verbose
        "evaluation_strategy": "ADAPTIVE",
        "enable_domains": ["syntax", "imports"],
        "skip_domains": ["comptime"],  # Known to fail
        "expected_pass_rate": 0.37,  # Worst performer
        "notes": "Model struggles with Zig syntax; avoid comptime constraints",
    },
}

# Domain-specific configuration
DOMAIN_CONSTRAINT_CONFIG: Dict[str, Dict[str, Any]] = {
    "imports": {
        "complexity": "syntactic",
        "expected_pass_rate": 0.86,
        "notes": "Simpler structural patterns work well",
    },
    "types": {
        "complexity": "structural",
        "expected_pass_rate": 0.71,
        "notes": "Type annotations are well-handled",
    },
    "semantics": {
        "complexity": "semantic",
        "expected_pass_rate": 0.64,
        "notes": "Moderate complexity",
    },
    "syntax": {
        "complexity": "structural",
        "expected_pass_rate": 0.54,
        "notes": "Mixed results",
    },
    "controlflow": {
        "complexity": "structural",
        "expected_pass_rate": 0.41,
        "notes": "Complex patterns struggle",
    },
    "coroutines": {
        "complexity": "semantic",
        "expected_pass_rate": 0.20,
        "notes": "Async patterns are hard",
    },
    "comptime": {
        "complexity": "semantic",
        "expected_pass_rate": 0.0,
        "notes": "Zig comptime constraints not achievable with regex",
    },
}


def get_language_config(language: str) -> Dict[str, Any]:
    """Get constraint configuration for a language."""
    return LANGUAGE_CONSTRAINT_CONFIG.get(language, {
        "max_tokens": 1024,
        "evaluation_strategy": "TIERED",
        "enable_domains": ["types", "imports", "syntax"],
        "expected_pass_rate": 0.50,
    })


def get_domain_config(domain: str) -> Dict[str, Any]:
    """Get constraint configuration for a domain."""
    return DOMAIN_CONSTRAINT_CONFIG.get(domain, {
        "complexity": "structural",
        "expected_pass_rate": 0.50,
    })


class EvalTier(Enum):
    """Capability tier for eval complexity."""

    SYNTAX = 1  # Regex/EBNF/JSON schema matching
    TYPE = 2  # Type-correct code
    IMPORT = 3  # Correct dependencies
    CONTROL_FLOW = 4  # Context-appropriate code
    SEMANTIC = 5  # Semantically valid code


class OutputMode(Enum):
    """Output mode for eval results."""

    QUIET = "quiet"  # Summary only
    NORMAL = "normal"  # Per-example results
    VERBOSE = "verbose"  # Full details including output samples


@dataclass
class EvalConfig:
    """Configuration for running evals.

    Attributes:
        tier: Maximum capability tier to evaluate
        languages: Languages to include (None = all)
        domains: Domains to include (None = all)
        tags: Tags to filter by (None = all)
        sample_count: Number of samples per example for statistical eval
        output_mode: Verbosity of output
        output_dir: Directory for reports
        compare_baseline: Whether to compare with unconstrained baseline
        timeout_per_example: Timeout in seconds per example
        seed: Random seed for reproducibility
    """

    tier: EvalTier = EvalTier.SYNTAX
    languages: Optional[Set[str]] = None
    domains: Optional[Set[str]] = None
    tags: Optional[Set[str]] = None
    sample_count: int = 1
    output_mode: OutputMode = OutputMode.NORMAL
    output_dir: Path = field(default_factory=lambda: Path("eval_results"))
    compare_baseline: bool = False
    timeout_per_example: float = 30.0
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.sample_count < 1:
            raise ValueError("sample_count must be at least 1")
        if self.timeout_per_example <= 0:
            raise ValueError("timeout_per_example must be positive")

    @classmethod
    def tier1_syntax(
        cls,
        languages: Optional[Set[str]] = None,
        domains: Optional[Set[str]] = None,
    ) -> EvalConfig:
        """Quick config for Tier 1 syntax constraint satisfaction."""
        return cls(
            tier=EvalTier.SYNTAX,
            languages=languages,
            domains=domains,
            sample_count=1,
            compare_baseline=False,
        )

    @classmethod
    def full_syntax(
        cls,
        sample_count: int = 10,
    ) -> EvalConfig:
        """Full Tier 1 eval with baseline comparison."""
        return cls(
            tier=EvalTier.SYNTAX,
            sample_count=sample_count,
            compare_baseline=True,
            output_mode=OutputMode.VERBOSE,
        )
