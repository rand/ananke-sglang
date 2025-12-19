# Copyright Rand Arete @ Ananke 2025
# Licensed under the Apache License, Version 2.0
"""Rust constraint examples for Ananke.

This package contains comprehensive Rust-specific constraint examples demonstrating:
- Type system: ownership, lifetimes, trait bounds
- Import constraints: no_std, unsafe restrictions, feature gating
- Control flow: match exhaustiveness, ? operator, guard clauses
- Semantics: panic-freedom, memory safety, unsafe block documentation
- Syntax: serde schemas, semver patterns, macro DSLs
- Ownership deep dive: Pin, 'static, Arc<Mutex<T>>, RefCell

All examples use realistic Rust systems programming scenarios with proper
type annotations, safety comments, and idiomatic patterns.
"""

from __future__ import annotations

from typing import List

from ..base import ConstraintExample

# Import domain-specific examples
from .types import RUST_TYPE_EXAMPLES
from .imports import RUST_IMPORT_EXAMPLES
from .controlflow import RUST_CONTROLFLOW_EXAMPLES
from .semantics import RUST_SEMANTIC_EXAMPLES
from .syntax import RUST_SYNTAX_EXAMPLES
from .ownership import RUST_OWNERSHIP_EXAMPLES

# =============================================================================
# Unified Export
# =============================================================================

ALL_RUST_EXAMPLES: List[ConstraintExample] = [
    *RUST_TYPE_EXAMPLES,
    *RUST_IMPORT_EXAMPLES,
    *RUST_CONTROLFLOW_EXAMPLES,
    *RUST_SEMANTIC_EXAMPLES,
    *RUST_SYNTAX_EXAMPLES,
    *RUST_OWNERSHIP_EXAMPLES,
]

__all__ = [
    "ALL_RUST_EXAMPLES",
    "RUST_TYPE_EXAMPLES",
    "RUST_IMPORT_EXAMPLES",
    "RUST_CONTROLFLOW_EXAMPLES",
    "RUST_SEMANTIC_EXAMPLES",
    "RUST_SYNTAX_EXAMPLES",
    "RUST_OWNERSHIP_EXAMPLES",
]
