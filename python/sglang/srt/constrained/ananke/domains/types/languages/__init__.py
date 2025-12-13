# Copyright Rand Arete @ Ananke 2025
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Language-specific type systems for Python, TypeScript, Rust, Zig, Go.

This package provides type system implementations for multiple programming
languages, enabling language-aware type checking during code generation.

Currently supported:
- Python (PEP 484 compatible, mypy/pyright semantics)

Planned:
- TypeScript
- Rust
- Zig
- Go

Usage:
    >>> from domains.types.languages import get_type_system, PythonTypeSystem
    >>>
    >>> # Get a type system by name
    >>> ts = get_type_system("python")
    >>>
    >>> # Parse type annotations
    >>> list_int = ts.parse_type_annotation("List[int]")
    >>>
    >>> # Check assignability
    >>> ts.check_assignable(INT, FLOAT)  # True (int -> float)
    >>>
    >>> # Format types
    >>> ts.format_type(list_int)  # "list[int]"
"""

from domains.types.languages.base import (
    LanguageTypeSystem,
    TypeSystemCapabilities,
    TypeParseError,
    LiteralInfo,
    LiteralKind,
    get_type_system,
    supported_languages,
)

from domains.types.languages.python import (
    PythonTypeSystem,
    BYTES,
    COMPLEX,
    OBJECT,
)


__all__ = [
    # Base
    "LanguageTypeSystem",
    "TypeSystemCapabilities",
    "TypeParseError",
    "LiteralInfo",
    "LiteralKind",
    "get_type_system",
    "supported_languages",
    # Python
    "PythonTypeSystem",
    "BYTES",
    "COMPLEX",
    "OBJECT",
]
