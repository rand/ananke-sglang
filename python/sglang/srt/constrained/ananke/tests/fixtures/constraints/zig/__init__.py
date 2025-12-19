# Copyright Rand Arete @ Ananke 2025
# Licensed under the Apache License, Version 2.0
"""Zig constraint examples for Ananke.

This module provides comprehensive constraint examples for Zig language features,
organized by domain:

- types.py: Type system constraints (comptime types, error unions, sentinel arrays)
- imports.py: Import/module constraints (platform restrictions, build options, allocators)
- controlflow.py: Control flow constraints (errdefer, comptime branches, unreachable)
- semantics.py: Semantic constraints (allocation safety, memory leaks, comptime assertions)
- syntax.py: Syntax constraints (Build.zig, naming conventions, format strings)
- comptime.py: Advanced comptime features (generics, allocators, error sets)

Usage:
    from tests.fixtures.constraints.zig import ALL_ZIG_EXAMPLES
    from tests.fixtures.constraints.zig import ZIG_TYPE_EXAMPLES, ZIG_COMPTIME_EXAMPLES
"""

from __future__ import annotations

from typing import List

try:
    from .types import ZIG_TYPE_EXAMPLES
    from .imports import ZIG_IMPORT_EXAMPLES
    from .controlflow import ZIG_CONTROLFLOW_EXAMPLES
    from .semantics import ZIG_SEMANTICS_EXAMPLES
    from .syntax import ZIG_SYNTAX_EXAMPLES
    from .comptime import ZIG_COMPTIME_EXAMPLES
    from ..base import ConstraintExample
except ImportError:
    # Fallback for different import contexts
    from types import ZIG_TYPE_EXAMPLES
    from imports import ZIG_IMPORT_EXAMPLES
    from controlflow import ZIG_CONTROLFLOW_EXAMPLES
    from semantics import ZIG_SEMANTICS_EXAMPLES
    from syntax import ZIG_SYNTAX_EXAMPLES
    from comptime import ZIG_COMPTIME_EXAMPLES
    from base import ConstraintExample


# Combine all Zig examples
ALL_ZIG_EXAMPLES: List[ConstraintExample] = [
    *ZIG_TYPE_EXAMPLES,
    *ZIG_IMPORT_EXAMPLES,
    *ZIG_CONTROLFLOW_EXAMPLES,
    *ZIG_SEMANTICS_EXAMPLES,
    *ZIG_SYNTAX_EXAMPLES,
    *ZIG_COMPTIME_EXAMPLES,
]


__all__ = [
    "ALL_ZIG_EXAMPLES",
    "ZIG_TYPE_EXAMPLES",
    "ZIG_IMPORT_EXAMPLES",
    "ZIG_CONTROLFLOW_EXAMPLES",
    "ZIG_SEMANTICS_EXAMPLES",
    "ZIG_SYNTAX_EXAMPLES",
    "ZIG_COMPTIME_EXAMPLES",
]


# Module metadata
__version__ = "1.0.0"
__author__ = "Rand Arete @ Ananke"


def get_example_by_id(example_id: str) -> ConstraintExample | None:
    """Get a specific Zig example by ID.

    Args:
        example_id: Example identifier (e.g., "zig-types-001")

    Returns:
        The matching ConstraintExample or None if not found
    """
    for example in ALL_ZIG_EXAMPLES:
        if example.id == example_id:
            return example
    return None


def get_examples_by_tag(tag: str) -> List[ConstraintExample]:
    """Get all Zig examples with a specific tag.

    Args:
        tag: Tag to filter by (e.g., "comptime", "error-handling")

    Returns:
        List of matching ConstraintExample objects
    """
    return [ex for ex in ALL_ZIG_EXAMPLES if tag in ex.tags]


def get_examples_by_domain(domain: str) -> List[ConstraintExample]:
    """Get all Zig examples for a specific domain.

    Args:
        domain: Domain name (e.g., "types", "imports", "controlflow")

    Returns:
        List of matching ConstraintExample objects
    """
    return [ex for ex in ALL_ZIG_EXAMPLES if ex.domain == domain]


def get_example_count() -> int:
    """Get total number of Zig constraint examples.

    Returns:
        Total count of examples across all domains
    """
    return len(ALL_ZIG_EXAMPLES)


def get_domain_counts() -> dict[str, int]:
    """Get count of examples per domain.

    Returns:
        Dictionary mapping domain names to example counts
    """
    counts: dict[str, int] = {}
    for example in ALL_ZIG_EXAMPLES:
        counts[example.domain] = counts.get(example.domain, 0) + 1
    return counts


def get_all_tags() -> set[str]:
    """Get all unique tags used across Zig examples.

    Returns:
        Set of all tag strings
    """
    tags: set[str] = set()
    for example in ALL_ZIG_EXAMPLES:
        tags.update(example.tags)
    return tags
