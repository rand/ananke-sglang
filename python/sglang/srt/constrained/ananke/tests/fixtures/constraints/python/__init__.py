# Copyright Rand Arete @ Ananke 2025
# Licensed under the Apache License, Version 2.0
"""Python constraint examples for Ananke.

This module aggregates all Python-specific constraint examples across all
constraint domains: types, imports, control flow, semantics, and syntax.

The examples demonstrate realistic developer workflows and how Ananke's
multi-domain constraint system masks tokens to enforce correctness.

Available collections:
- PYTHON_TYPE_EXAMPLES: Type system constraints (generics, protocols, narrowing)
- PYTHON_IMPORT_EXAMPLES: Import constraints (security, compatibility, TYPE_CHECKING)
- PYTHON_CONTROLFLOW_EXAMPLES: Control flow constraints (async/await, loops, exceptions)
- PYTHON_SEMANTICS_EXAMPLES: Semantic constraints (pre/postconditions, invariants)
- PYTHON_SYNTAX_EXAMPLES: Syntax constraints (JSON schema, regex, EBNF)
- PYTHON_DEEP_DIVE_EXAMPLES: Complex multi-domain examples (dataclass validation)
- ALL_PYTHON_EXAMPLES: Combined list of all examples

Usage:
    >>> from tests.fixtures.constraints.python import ALL_PYTHON_EXAMPLES
    >>> from tests.fixtures.constraints.python import PYTHON_TYPE_EXAMPLES
    >>>
    >>> # Get all Python examples
    >>> examples = ALL_PYTHON_EXAMPLES
    >>> print(f"Total Python examples: {len(examples)}")
    >>>
    >>> # Filter by tag
    >>> async_examples = [e for e in ALL_PYTHON_EXAMPLES if "async" in e.tags]
    >>>
    >>> # Access specific domain
    >>> type_examples = PYTHON_TYPE_EXAMPLES
    >>> for example in type_examples:
    ...     print(f"{example.id}: {example.name}")
"""

from __future__ import annotations

from typing import List

from ..base import ConstraintExample

# Import all domain-specific examples
from .types import PYTHON_TYPE_EXAMPLES
from .imports import PYTHON_IMPORT_EXAMPLES
from .controlflow import PYTHON_CONTROLFLOW_EXAMPLES
from .semantics import PYTHON_SEMANTICS_EXAMPLES
from .syntax import PYTHON_SYNTAX_EXAMPLES
from .dataclass_validation import PYTHON_DEEP_DIVE_EXAMPLES

__all__ = [
    "PYTHON_TYPE_EXAMPLES",
    "PYTHON_IMPORT_EXAMPLES",
    "PYTHON_CONTROLFLOW_EXAMPLES",
    "PYTHON_SEMANTICS_EXAMPLES",
    "PYTHON_SYNTAX_EXAMPLES",
    "PYTHON_DEEP_DIVE_EXAMPLES",
    "ALL_PYTHON_EXAMPLES",
]

# Combine all examples into a single list
ALL_PYTHON_EXAMPLES: List[ConstraintExample] = [
    *PYTHON_TYPE_EXAMPLES,
    *PYTHON_IMPORT_EXAMPLES,
    *PYTHON_CONTROLFLOW_EXAMPLES,
    *PYTHON_SEMANTICS_EXAMPLES,
    *PYTHON_SYNTAX_EXAMPLES,
    *PYTHON_DEEP_DIVE_EXAMPLES,
]

# Validation: Check for unique IDs
_ids = [e.id for e in ALL_PYTHON_EXAMPLES]
if len(_ids) != len(set(_ids)):
    duplicates = [id_ for id_ in _ids if _ids.count(id_) > 1]
    raise ValueError(f"Duplicate example IDs found: {set(duplicates)}")

# Validation: Check all examples have language="python"
_wrong_lang = [e.id for e in ALL_PYTHON_EXAMPLES if e.language != "python"]
if _wrong_lang:
    raise ValueError(f"Examples with wrong language: {_wrong_lang}")

# Summary statistics (computed at import time)
_by_domain = {}
for example in ALL_PYTHON_EXAMPLES:
    domain = example.domain
    _by_domain[domain] = _by_domain.get(domain, 0) + 1

_SUMMARY = {
    "total_examples": len(ALL_PYTHON_EXAMPLES),
    "by_domain": _by_domain,
    "example_ids": [e.id for e in ALL_PYTHON_EXAMPLES],
}


def get_summary() -> dict:
    """Get summary statistics about Python constraint examples.

    Returns:
        Dictionary with total count, breakdown by domain, and list of IDs
    """
    return _SUMMARY.copy()


def get_by_tag(tag: str) -> List[ConstraintExample]:
    """Get all Python examples with a specific tag.

    Args:
        tag: Tag to filter by (e.g., "async", "generics", "validation")

    Returns:
        List of examples with the specified tag
    """
    return [e for e in ALL_PYTHON_EXAMPLES if tag in e.tags]


def get_by_domain(domain: str) -> List[ConstraintExample]:
    """Get all Python examples for a specific domain.

    Args:
        domain: Domain name ("types", "imports", "controlflow", "semantics", "syntax")

    Returns:
        List of examples in the specified domain
    """
    return [e for e in ALL_PYTHON_EXAMPLES if e.domain == domain]


def get_by_id(example_id: str) -> ConstraintExample | None:
    """Get a specific Python example by ID.

    Args:
        example_id: Example ID (e.g., "py-types-001")

    Returns:
        The example if found, None otherwise
    """
    for example in ALL_PYTHON_EXAMPLES:
        if example.id == example_id:
            return example
    return None
