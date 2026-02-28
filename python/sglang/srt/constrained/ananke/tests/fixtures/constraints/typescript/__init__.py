# Copyright Rand Arete @ Ananke 2025
# Licensed under the Apache License, Version 2.0
"""TypeScript constraint examples for Ananke.

This package provides comprehensive constraint examples for TypeScript,
demonstrating how Ananke's domain-based masking system enforces type safety,
import correctness, control flow soundness, semantic constraints, and syntax
validation during code generation.

Modules:
    types: Type-level constraints (conditional types, mapped types, template literals)
    imports: Import constraints (ESM/CommonJS, type-only imports, path aliases)
    controlflow: Control flow constraints (discriminated unions, async/await, never)
    semantics: Semantic constraints (non-null assertions, readonly, narrowing)
    syntax: Syntax constraints (OpenAPI schemas, URL patterns, GraphQL)
    conditional_types: Advanced type system features (recursive types, distributive types)

Usage:
    >>> from .typescript import ALL_TYPESCRIPT_EXAMPLES
    >>> # Get all TypeScript examples
    >>> examples = ALL_TYPESCRIPT_EXAMPLES
    >>> # Filter by domain
    >>> type_examples = [e for e in examples if e.domain == "types"]
"""

from __future__ import annotations

from typing import List

from ..base import ConstraintExample
from .types import TYPESCRIPT_TYPE_EXAMPLES
from .imports import TYPESCRIPT_IMPORT_EXAMPLES
from .controlflow import TYPESCRIPT_CONTROLFLOW_EXAMPLES
from .semantics import TYPESCRIPT_SEMANTICS_EXAMPLES
from .syntax import TYPESCRIPT_SYNTAX_EXAMPLES
from .conditional_types import TYPESCRIPT_CONDITIONAL_TYPES_EXAMPLES

# Combine all TypeScript examples
ALL_TYPESCRIPT_EXAMPLES: List[ConstraintExample] = [
    *TYPESCRIPT_TYPE_EXAMPLES,
    *TYPESCRIPT_IMPORT_EXAMPLES,
    *TYPESCRIPT_CONTROLFLOW_EXAMPLES,
    *TYPESCRIPT_SEMANTICS_EXAMPLES,
    *TYPESCRIPT_SYNTAX_EXAMPLES,
    *TYPESCRIPT_CONDITIONAL_TYPES_EXAMPLES,
]

__all__ = [
    "ALL_TYPESCRIPT_EXAMPLES",
    "TYPESCRIPT_TYPE_EXAMPLES",
    "TYPESCRIPT_IMPORT_EXAMPLES",
    "TYPESCRIPT_CONTROLFLOW_EXAMPLES",
    "TYPESCRIPT_SEMANTICS_EXAMPLES",
    "TYPESCRIPT_SYNTAX_EXAMPLES",
    "TYPESCRIPT_CONDITIONAL_TYPES_EXAMPLES",
]
