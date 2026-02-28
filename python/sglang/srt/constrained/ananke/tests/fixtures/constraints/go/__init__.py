# Copyright Rand Arete @ Ananke 2025
# Licensed under the Apache License, Version 2.0
"""Go constraint examples for Ananke.

This package contains comprehensive constraint examples for Go, demonstrating
how Ananke's domain-based masking system enforces Go-specific constraints
during code generation.

Modules:
    types: Interface satisfaction, channel directions, generic constraints
    imports: Package path restrictions, unsafe blocking, stdlib constraints
    controlflow: Defer patterns, goroutine contexts, error handling
    semantics: Nil checks, context propagation, resource cleanup
    syntax: Protobuf tags, import path validation, template syntax
    interfaces: Deep dive into Go's interface system (embedding, assertions, patterns)

Example Usage:
    >>> from .types import GO_TYPE_EXAMPLES
    >>> from . import ALL_GO_EXAMPLES
    >>> print(f"Total Go examples: {len(ALL_GO_EXAMPLES)}")
"""

from __future__ import annotations

from typing import List

from ..base import ConstraintExample

# Import domain-specific examples
from .types import GO_TYPE_EXAMPLES
from .imports import GO_IMPORT_EXAMPLES
from .controlflow import GO_CONTROLFLOW_EXAMPLES
from .semantics import GO_SEMANTIC_EXAMPLES
from .syntax import GO_SYNTAX_EXAMPLES
from .interfaces import GO_INTERFACE_EXAMPLES

# Combine all examples
ALL_GO_EXAMPLES: List[ConstraintExample] = (
    GO_TYPE_EXAMPLES
    + GO_IMPORT_EXAMPLES
    + GO_CONTROLFLOW_EXAMPLES
    + GO_SEMANTIC_EXAMPLES
    + GO_SYNTAX_EXAMPLES
    + GO_INTERFACE_EXAMPLES
)

__all__ = [
    "GO_TYPE_EXAMPLES",
    "GO_IMPORT_EXAMPLES",
    "GO_CONTROLFLOW_EXAMPLES",
    "GO_SEMANTIC_EXAMPLES",
    "GO_SYNTAX_EXAMPLES",
    "GO_INTERFACE_EXAMPLES",
    "ALL_GO_EXAMPLES",
]
