# Copyright Rand Arete @ Ananke 2025
# Licensed under the Apache License, Version 2.0
"""Kotlin constraint examples for Ananke.

This package contains realistic constraint examples for Kotlin across all
domains: types, imports, control flow, semantics, syntax, and a deep dive
into coroutines.

Kotlin-specific features covered:
- Nullable types with safe call operators (?., ?:, !!)
- Sealed classes with exhaustive when expressions
- Suspend functions and coroutine contexts
- Platform-specific imports (JVM, JS, Native)
- Kotlinx.coroutines Flow, StateFlow, SharedFlow
- Annotation processing (kotlinx.serialization, Room)
- Type-safe builder DSLs
- Smart casts and contracts
- Collection safety (getOrNull, indices)
- Framework patterns (Spring, Ktor)

Usage:
    from tests.fixtures.constraints.kotlin import ALL_KOTLIN_EXAMPLES
    from tests.fixtures.constraints.kotlin.types import KOTLIN_TYPE_EXAMPLES
    from tests.fixtures.constraints.kotlin.coroutines import KOTLIN_COROUTINES_EXAMPLES
"""

from __future__ import annotations

from typing import List

from ..base import ConstraintExample

# Import all domain examples
from .types import KOTLIN_TYPE_EXAMPLES
from .imports import KOTLIN_IMPORT_EXAMPLES
from .controlflow import KOTLIN_CONTROLFLOW_EXAMPLES
from .semantics import KOTLIN_SEMANTICS_EXAMPLES
from .syntax import KOTLIN_SYNTAX_EXAMPLES
from .coroutines import KOTLIN_COROUTINES_EXAMPLES

__all__ = [
    "KOTLIN_TYPE_EXAMPLES",
    "KOTLIN_IMPORT_EXAMPLES",
    "KOTLIN_CONTROLFLOW_EXAMPLES",
    "KOTLIN_SEMANTICS_EXAMPLES",
    "KOTLIN_SYNTAX_EXAMPLES",
    "KOTLIN_COROUTINES_EXAMPLES",
    "ALL_KOTLIN_EXAMPLES",
    "get_kotlin_examples_by_domain",
]


# Combine all examples
ALL_KOTLIN_EXAMPLES: List[ConstraintExample] = [
    *KOTLIN_TYPE_EXAMPLES,
    *KOTLIN_IMPORT_EXAMPLES,
    *KOTLIN_CONTROLFLOW_EXAMPLES,
    *KOTLIN_SEMANTICS_EXAMPLES,
    *KOTLIN_SYNTAX_EXAMPLES,
    *KOTLIN_COROUTINES_EXAMPLES,
]


def get_kotlin_examples_by_domain(domain: str) -> List[ConstraintExample]:
    """Get Kotlin examples for a specific domain.

    Args:
        domain: Domain name (types, imports, controlflow, semantics, syntax, coroutines)

    Returns:
        List of constraint examples for that domain
    """
    domain_map = {
        "types": KOTLIN_TYPE_EXAMPLES,
        "imports": KOTLIN_IMPORT_EXAMPLES,
        "controlflow": KOTLIN_CONTROLFLOW_EXAMPLES,
        "semantics": KOTLIN_SEMANTICS_EXAMPLES,
        "syntax": KOTLIN_SYNTAX_EXAMPLES,
        "coroutines": KOTLIN_COROUTINES_EXAMPLES,
    }
    return domain_map.get(domain, [])


# Statistics
_TOTAL_EXAMPLES = len(ALL_KOTLIN_EXAMPLES)
_EXAMPLES_BY_DOMAIN = {
    "types": len(KOTLIN_TYPE_EXAMPLES),
    "imports": len(KOTLIN_IMPORT_EXAMPLES),
    "controlflow": len(KOTLIN_CONTROLFLOW_EXAMPLES),
    "semantics": len(KOTLIN_SEMANTICS_EXAMPLES),
    "syntax": len(KOTLIN_SYNTAX_EXAMPLES),
    "coroutines": len(KOTLIN_COROUTINES_EXAMPLES),
}

# Module docstring enhancement
__doc__ += f"""

Statistics:
- Total examples: {_TOTAL_EXAMPLES}
- Types: {_EXAMPLES_BY_DOMAIN['types']}
- Imports: {_EXAMPLES_BY_DOMAIN['imports']}
- Control Flow: {_EXAMPLES_BY_DOMAIN['controlflow']}
- Semantics: {_EXAMPLES_BY_DOMAIN['semantics']}
- Syntax: {_EXAMPLES_BY_DOMAIN['syntax']}
- Coroutines (deep dive): {_EXAMPLES_BY_DOMAIN['coroutines']}
"""
