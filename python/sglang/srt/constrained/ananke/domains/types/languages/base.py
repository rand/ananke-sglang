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
"""Base class for language-specific type systems.

This module defines the LanguageTypeSystem abstract base class that
language-specific implementations (Python, TypeScript, Rust, Zig, Go)
must implement.

Each language type system provides:
- Type parsing from string annotations
- Literal type inference
- Builtin type definitions
- Assignability checking (subtyping)
- Type formatting

References:
    - Python: PEP 484, mypy, pyright
    - TypeScript: TypeScript Handbook
    - Rust: The Rust Reference
    - Zig: Zig Language Reference
    - Go: Go Language Specification
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, FrozenSet, List, Optional, Tuple, Union

from domains.types.constraint import (
    Type,
    TypeVar,
    PrimitiveType,
    FunctionType,
    ListType,
    DictType,
    TupleType,
    SetType,
    UnionType,
    ClassType,
    AnyType,
    NeverType,
    HoleType,
    INT,
    STR,
    BOOL,
    FLOAT,
    NONE,
    ANY,
    NEVER,
)


class TypeParseError(Exception):
    """Error parsing a type annotation string."""

    def __init__(self, annotation: str, message: str):
        self.annotation = annotation
        self.message = message
        super().__init__(f"Failed to parse type '{annotation}': {message}")


class LiteralKind(Enum):
    """The kind of literal value."""

    INTEGER = auto()
    FLOAT = auto()
    STRING = auto()
    CHARACTER = auto()  # For Rust char, Zig u8/u21
    BOOLEAN = auto()
    NONE = auto()
    LIST = auto()
    DICT = auto()
    TUPLE = auto()
    SET = auto()


@dataclass(frozen=True, slots=True)
class LiteralInfo:
    """Information about a literal value.

    Attributes:
        kind: The kind of literal
        value: The actual value (if known)
        text: The source text representation
    """

    kind: LiteralKind
    value: Optional[Any] = None
    text: Optional[str] = None


@dataclass(frozen=True, slots=True)
class TypeSystemCapabilities:
    """Capabilities of a language type system.

    Attributes:
        supports_generics: Whether the language supports generic types
        supports_union_types: Whether union types are supported
        supports_optional_types: Whether optional/nullable types are supported
        supports_type_inference: Whether type inference is supported
        supports_protocols: Whether structural subtyping is supported
        supports_variance: Whether type parameter variance is tracked
        supports_overloading: Whether function overloading is supported
        supports_ownership: Whether ownership/borrowing is tracked (Rust)
        supports_comptime: Whether compile-time evaluation is tracked (Zig)
        supports_error_unions: Whether error union types are supported (Zig E!T)
        supports_lifetime_bounds: Whether lifetime annotations are supported (Rust 'a)
        supports_sentinels: Whether sentinel-terminated types are supported (Zig [*:0]T)
        supports_allocators: Whether explicit allocator parameters are tracked (Zig)
    """

    supports_generics: bool = True
    supports_union_types: bool = True
    supports_optional_types: bool = True
    supports_type_inference: bool = True
    supports_protocols: bool = False
    supports_variance: bool = False
    supports_overloading: bool = False
    supports_ownership: bool = False
    supports_comptime: bool = False
    supports_error_unions: bool = False
    supports_lifetime_bounds: bool = False
    supports_sentinels: bool = False
    supports_allocators: bool = False


class LanguageTypeSystem(ABC):
    """Abstract base class for language-specific type systems.

    Implementations provide language-specific type parsing, inference,
    and checking. The type system is used by the TypeDomain to enforce
    type correctness during code generation.

    Subclasses must implement:
    - parse_type_annotation: Parse a type from string
    - infer_literal_type: Infer type of a literal value
    - get_builtin_types: Return builtin type bindings
    - check_assignable: Check if one type is assignable to another
    - format_type: Convert type to string representation
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the language (e.g., 'python', 'typescript')."""
        pass

    @property
    @abstractmethod
    def capabilities(self) -> TypeSystemCapabilities:
        """Return the capabilities of this type system."""
        pass

    @abstractmethod
    def parse_type_annotation(self, annotation: str) -> Type:
        """Parse a type annotation string into a Type.

        Args:
            annotation: The type annotation string (e.g., "int", "List[str]")

        Returns:
            The parsed Type

        Raises:
            TypeParseError: If the annotation cannot be parsed
        """
        pass

    @abstractmethod
    def infer_literal_type(self, literal: LiteralInfo) -> Type:
        """Infer the type of a literal value.

        Args:
            literal: Information about the literal

        Returns:
            The inferred type
        """
        pass

    @abstractmethod
    def get_builtin_types(self) -> Dict[str, Type]:
        """Return a dictionary of builtin type bindings.

        Returns:
            Map from name to Type for all builtin types
        """
        pass

    @abstractmethod
    def check_assignable(self, source: Type, target: Type) -> bool:
        """Check if source type is assignable to target type.

        This implements the language's subtyping rules.

        Args:
            source: The type being assigned from
            target: The type being assigned to

        Returns:
            True if source can be assigned to target
        """
        pass

    @abstractmethod
    def format_type(self, typ: Type) -> str:
        """Format a type as a string in this language's syntax.

        Args:
            typ: The type to format

        Returns:
            String representation in this language's syntax
        """
        pass

    def get_builtin_functions(self) -> Dict[str, FunctionType]:
        """Return builtin function signatures.

        Override this to provide language-specific builtin functions.

        Returns:
            Map from function name to FunctionType
        """
        return {}

    def get_common_imports(self) -> Dict[str, Type]:
        """Return commonly imported types.

        Override this to provide common library types.

        Returns:
            Map from qualified name to Type
        """
        return {}

    def supports_type(self, typ: Type) -> bool:
        """Check if a type is supported by this language.

        Args:
            typ: The type to check

        Returns:
            True if this language supports the type
        """
        # Default: support all types
        return True

    def normalize_type(self, typ: Type) -> Type:
        """Normalize a type to canonical form.

        Some languages may have multiple representations for the same type.
        This normalizes to a canonical form for comparison.

        Args:
            typ: The type to normalize

        Returns:
            The normalized type
        """
        # Default: no normalization
        return typ

    def lub(self, types: List[Type]) -> Optional[Type]:
        """Compute the least upper bound of a list of types.

        The LUB is the most specific type that all given types are
        assignable to. Used for type inference in conditionals.

        Args:
            types: The types to compute LUB for

        Returns:
            The LUB, or None if no common supertype exists
        """
        if not types:
            return NEVER

        if len(types) == 1:
            return types[0]

        # Default: return union of types if supported
        if self.capabilities.supports_union_types:
            return UnionType(frozenset(types))

        # Otherwise try to find a common supertype
        result = types[0]
        for typ in types[1:]:
            if self.check_assignable(typ, result):
                continue
            elif self.check_assignable(result, typ):
                result = typ
            else:
                # No common supertype - return Any
                return ANY

        return result

    def glb(self, types: List[Type]) -> Optional[Type]:
        """Compute the greatest lower bound of a list of types.

        The GLB is the most general type that is assignable to all
        given types. Used for type refinement.

        Args:
            types: The types to compute GLB for

        Returns:
            The GLB, or None if no common subtype exists
        """
        if not types:
            return ANY

        if len(types) == 1:
            return types[0]

        # Default: find the most specific type that is assignable to all
        result = types[0]
        for typ in types[1:]:
            if self.check_assignable(result, typ):
                continue
            elif self.check_assignable(typ, result):
                result = typ
            else:
                # No common subtype - return Never
                return NEVER

        return result


def get_type_system(language: str) -> LanguageTypeSystem:
    """Get the type system for a language.

    Args:
        language: The language name (e.g., 'python', 'zig', 'rust')

    Returns:
        The LanguageTypeSystem for that language

    Raises:
        ValueError: If the language is not supported
    """
    # Import here to avoid circular imports
    from domains.types.languages.python import PythonTypeSystem

    # Build systems dict with available implementations
    systems: Dict[str, type] = {
        "python": PythonTypeSystem,
        "py": PythonTypeSystem,
    }

    # Try to import Zig type system if available
    try:
        from domains.types.languages.zig import ZigTypeSystem
        systems["zig"] = ZigTypeSystem
    except ImportError:
        pass

    # Try to import Rust type system if available
    try:
        from domains.types.languages.rust import RustTypeSystem
        systems["rust"] = RustTypeSystem
        systems["rs"] = RustTypeSystem
    except ImportError:
        pass

    # Try to import TypeScript type system if available
    try:
        from domains.types.languages.typescript import TypeScriptTypeSystem
        systems["typescript"] = TypeScriptTypeSystem
        systems["ts"] = TypeScriptTypeSystem
        systems["javascript"] = TypeScriptTypeSystem  # JS is a subset of TS
        systems["js"] = TypeScriptTypeSystem
    except ImportError:
        pass

    # Try to import Go type system if available
    try:
        from domains.types.languages.go import GoTypeSystem
        systems["go"] = GoTypeSystem
    except ImportError:
        pass

    # Try to import Kotlin type system if available
    try:
        from domains.types.languages.kotlin import KotlinTypeSystem
        systems["kotlin"] = KotlinTypeSystem
        systems["kt"] = KotlinTypeSystem
    except ImportError:
        pass

    # Try to import Swift type system if available
    try:
        from domains.types.languages.swift import SwiftTypeSystem
        systems["swift"] = SwiftTypeSystem
    except ImportError:
        pass

    language_lower = language.lower()
    if language_lower not in systems:
        supported = ", ".join(sorted(set(systems.keys())))
        raise ValueError(
            f"Unsupported language '{language}'. Supported: {supported}"
        )

    return systems[language_lower]()


def supported_languages() -> List[str]:
    """Return a list of supported language names.

    Returns:
        List of language name strings (canonical names only, not aliases)
    """
    languages = ["python"]

    # Check for Zig support
    try:
        from domains.types.languages.zig import ZigTypeSystem
        languages.append("zig")
    except ImportError:
        pass

    # Check for Rust support
    try:
        from domains.types.languages.rust import RustTypeSystem
        languages.append("rust")
    except ImportError:
        pass

    # Check for TypeScript support
    try:
        from domains.types.languages.typescript import TypeScriptTypeSystem
        languages.append("typescript")
    except ImportError:
        pass

    # Check for Go support
    try:
        from domains.types.languages.go import GoTypeSystem
        languages.append("go")
    except ImportError:
        pass

    # Check for Kotlin support
    try:
        from domains.types.languages.kotlin import KotlinTypeSystem
        languages.append("kotlin")
    except ImportError:
        pass

    # Check for Swift support
    try:
        from domains.types.languages.swift import SwiftTypeSystem
        languages.append("swift")
    except ImportError:
        pass

    return languages
