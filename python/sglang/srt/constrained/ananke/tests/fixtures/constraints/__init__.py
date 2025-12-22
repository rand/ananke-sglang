# Copyright Rand Arete @ Ananke 2025
# Licensed under the Apache License, Version 2.0
"""Constraint example fixtures for Ananke testing and documentation.

This package contains realistic constraint examples for all supported
languages and domains, organized as:

- `python/` - Python-specific constraint examples
- `rust/` - Rust-specific constraint examples
- `zig/` - Zig-specific constraint examples
- `typescript/` - TypeScript-specific constraint examples
- `go/` - Go-specific constraint examples
- `kotlin/` - Kotlin-specific constraint examples
- `swift/` - Swift-specific constraint examples
- `cross_language/` - Examples showing same workflow across languages

Each language module contains domain-specific submodules:
- `types.py` - Type system constraint examples
- `imports.py` - Import/module constraint examples
- `controlflow.py` - Control flow constraint examples
- `semantics.py` - Semantic constraint examples
- `syntax.py` - Syntax/grammar constraint examples

Usage:
    from tests.fixtures.constraints import get_all_examples, get_examples_by_language
    from tests.fixtures.constraints.python.types import PYTHON_TYPE_EXAMPLES
"""

from __future__ import annotations

from typing import List, Optional

from .base import ConstraintExample, ExampleCatalog, LANGUAGES, DOMAINS

__all__ = [
    "ConstraintExample",
    "ExampleCatalog",
    "LANGUAGES",
    "DOMAINS",
    "get_all_examples",
    "get_examples_by_language",
    "get_examples_by_domain",
]


def get_all_examples() -> List[ConstraintExample]:
    """Get all constraint examples from all languages and domains."""
    examples = []

    # Import each language module and collect examples
    try:
        from .python import ALL_PYTHON_EXAMPLES
        examples.extend(ALL_PYTHON_EXAMPLES)
    except ImportError:
        pass

    try:
        from .rust import ALL_RUST_EXAMPLES
        examples.extend(ALL_RUST_EXAMPLES)
    except ImportError:
        pass

    try:
        from .zig import ALL_ZIG_EXAMPLES
        examples.extend(ALL_ZIG_EXAMPLES)
    except ImportError:
        pass

    try:
        from .typescript import ALL_TYPESCRIPT_EXAMPLES
        examples.extend(ALL_TYPESCRIPT_EXAMPLES)
    except ImportError:
        pass

    try:
        from .go import ALL_GO_EXAMPLES
        examples.extend(ALL_GO_EXAMPLES)
    except ImportError:
        pass

    try:
        from .kotlin import ALL_KOTLIN_EXAMPLES
        examples.extend(ALL_KOTLIN_EXAMPLES)
    except ImportError:
        pass

    try:
        from .swift import ALL_SWIFT_EXAMPLES
        examples.extend(ALL_SWIFT_EXAMPLES)
    except ImportError:
        pass

    try:
        from .cross_language import ALL_CROSS_LANGUAGE_EXAMPLES
        examples.extend(ALL_CROSS_LANGUAGE_EXAMPLES)
    except ImportError:
        pass

    return examples


def get_examples_by_language(language: str) -> List[ConstraintExample]:
    """Get all examples for a specific language."""
    return [e for e in get_all_examples() if e.language == language]


def get_examples_by_domain(domain: str) -> List[ConstraintExample]:
    """Get all examples for a specific domain."""
    return [e for e in get_all_examples() if e.domain == domain]
