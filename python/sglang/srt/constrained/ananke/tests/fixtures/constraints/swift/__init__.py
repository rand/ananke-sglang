# Copyright Rand Arete @ Ananke 2025
# Licensed under the Apache License, Version 2.0
"""Swift constraint examples for Ananke.

This module contains realistic constraint examples for Swift, organized by domain:

- types.py: Type system constraints (optionals, protocols, actors)
- imports.py: Import and framework availability constraints
- controlflow.py: Control flow constraints (guard, async/await, result builders)
- semantics.py: Semantic constraints (force unwrap, actor state, Sendable)
- syntax.py: Syntax constraints (Codable, bundle IDs, SwiftUI DSL)
- actors.py: Deep dive into actor isolation, Sendable, and async patterns

All examples demonstrate modern Swift 5.9+ features including:
- Swift 6 strict concurrency (actors, Sendable, @MainActor)
- SwiftUI view builders and result builders
- Optional chaining and guard-let patterns
- Platform availability checks (#available)
- Codable for JSON serialization
"""

from __future__ import annotations

from typing import List

from ..base import ConstraintExample

# Import domain-specific examples
from .types import SWIFT_TYPE_EXAMPLES
from .imports import SWIFT_IMPORT_EXAMPLES
from .controlflow import SWIFT_CONTROLFLOW_EXAMPLES
from .semantics import SWIFT_SEMANTIC_EXAMPLES
from .syntax import SWIFT_SYNTAX_EXAMPLES
from .actors import SWIFT_ACTOR_EXAMPLES

# =============================================================================
# Aggregate All Swift Examples
# =============================================================================

ALL_SWIFT_EXAMPLES: List[ConstraintExample] = [
    *SWIFT_TYPE_EXAMPLES,
    *SWIFT_IMPORT_EXAMPLES,
    *SWIFT_CONTROLFLOW_EXAMPLES,
    *SWIFT_SEMANTIC_EXAMPLES,
    *SWIFT_SYNTAX_EXAMPLES,
    *SWIFT_ACTOR_EXAMPLES,
]

__all__ = [
    # Individual domain exports
    "SWIFT_TYPE_EXAMPLES",
    "SWIFT_IMPORT_EXAMPLES",
    "SWIFT_CONTROLFLOW_EXAMPLES",
    "SWIFT_SEMANTIC_EXAMPLES",
    "SWIFT_SYNTAX_EXAMPLES",
    "SWIFT_ACTOR_EXAMPLES",
    # Aggregate export
    "ALL_SWIFT_EXAMPLES",
]
