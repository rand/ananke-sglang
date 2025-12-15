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
"""Swift type system implementation.

This module implements the Swift type system for constrained decoding,
supporting Swift's type hierarchy, protocols, and generics.

Features:
- Primitive types: Int, Double, Bool, String, etc.
- Optional types: T?, T!
- Array, Dictionary, Set types
- Function types: (T) -> R
- Protocol types and protocol composition
- Generic types with constraints

References:
    - Swift Language Guide: https://docs.swift.org/swift-book/
    - Swift Type System: https://docs.swift.org/swift-book/LanguageGuide/Types.html
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    FrozenSet,
    List,
    Literal,
    Optional,
    Tuple,
)

from domains.types.constraint import (
    Type,
    PrimitiveType,
    FunctionType as BaseFunctionType,
    AnyType,
    NeverType,
)
from domains.types.languages.base import (
    LanguageTypeSystem,
    TypeSystemCapabilities,
    TypeParseError,
    LiteralInfo,
    LiteralKind,
)


# =============================================================================
# Swift Primitive Types
# =============================================================================

# Integer types
SWIFT_INT = PrimitiveType("Int")
SWIFT_INT8 = PrimitiveType("Int8")
SWIFT_INT16 = PrimitiveType("Int16")
SWIFT_INT32 = PrimitiveType("Int32")
SWIFT_INT64 = PrimitiveType("Int64")

SWIFT_UINT = PrimitiveType("UInt")
SWIFT_UINT8 = PrimitiveType("UInt8")
SWIFT_UINT16 = PrimitiveType("UInt16")
SWIFT_UINT32 = PrimitiveType("UInt32")
SWIFT_UINT64 = PrimitiveType("UInt64")

# Floating point
SWIFT_FLOAT = PrimitiveType("Float")
SWIFT_DOUBLE = PrimitiveType("Double")
SWIFT_FLOAT16 = PrimitiveType("Float16")
SWIFT_FLOAT80 = PrimitiveType("Float80")

# Boolean
SWIFT_BOOL = PrimitiveType("Bool")

# Character and String
SWIFT_CHARACTER = PrimitiveType("Character")
SWIFT_STRING = PrimitiveType("String")

# Special types
SWIFT_VOID = PrimitiveType("Void")
SWIFT_NEVER = NeverType()
SWIFT_ANY = AnyType()
SWIFT_ANY_OBJECT = PrimitiveType("AnyObject")


# Primitive type lookup
SWIFT_PRIMITIVES: Dict[str, Type] = {
    "Int": SWIFT_INT,
    "Int8": SWIFT_INT8,
    "Int16": SWIFT_INT16,
    "Int32": SWIFT_INT32,
    "Int64": SWIFT_INT64,
    "UInt": SWIFT_UINT,
    "UInt8": SWIFT_UINT8,
    "UInt16": SWIFT_UINT16,
    "UInt32": SWIFT_UINT32,
    "UInt64": SWIFT_UINT64,
    "Float": SWIFT_FLOAT,
    "Double": SWIFT_DOUBLE,
    "Float16": SWIFT_FLOAT16,
    "Float80": SWIFT_FLOAT80,
    "Bool": SWIFT_BOOL,
    "Character": SWIFT_CHARACTER,
    "String": SWIFT_STRING,
    "Void": SWIFT_VOID,
    "Never": SWIFT_NEVER,
    "Any": SWIFT_ANY,
    "AnyObject": SWIFT_ANY_OBJECT,
}


# =============================================================================
# Swift Composite Types
# =============================================================================

@dataclass(frozen=True)
class SwiftOptionalType(Type):
    """Swift optional type: T?"""
    wrapped: Type

    def __str__(self) -> str:
        wrapped_str = str(self.wrapped)
        if isinstance(self.wrapped, SwiftFunctionType):
            return f"({wrapped_str})?"
        return f"{wrapped_str}?"


@dataclass(frozen=True)
class SwiftImplicitlyUnwrappedOptionalType(Type):
    """Swift implicitly unwrapped optional type: T!"""
    wrapped: Type

    def __str__(self) -> str:
        wrapped_str = str(self.wrapped)
        if isinstance(self.wrapped, SwiftFunctionType):
            return f"({wrapped_str})!"
        return f"{wrapped_str}!"


@dataclass(frozen=True)
class SwiftArrayType(Type):
    """Swift array type: [T]"""
    element: Type

    def __str__(self) -> str:
        return f"[{self.element}]"


@dataclass(frozen=True)
class SwiftDictionaryType(Type):
    """Swift dictionary type: [K: V]"""
    key: Type
    value: Type

    def __str__(self) -> str:
        return f"[{self.key}: {self.value}]"


@dataclass(frozen=True)
class SwiftSetType(Type):
    """Swift set type: Set<T>"""
    element: Type

    def __str__(self) -> str:
        return f"Set<{self.element}>"


@dataclass(frozen=True)
class SwiftTupleType(Type):
    """Swift tuple type: (T1, T2, ...)"""
    elements: Tuple[Type, ...]
    labels: Tuple[Optional[str], ...] = ()

    def __str__(self) -> str:
        if self.labels and len(self.labels) == len(self.elements):
            parts = []
            for label, elem in zip(self.labels, self.elements):
                if label:
                    parts.append(f"{label}: {elem}")
                else:
                    parts.append(str(elem))
            return f"({', '.join(parts)})"
        return f"({', '.join(str(e) for e in self.elements)})"


@dataclass(frozen=True)
class SwiftFunctionType(Type):
    """Swift function type: (P1, P2) -> R or (P1, P2) throws -> R"""
    parameters: Tuple[Type, ...]
    return_type: Type
    throws: bool = False
    async_: bool = False  # async in Python is reserved

    def __str__(self) -> str:
        params = ", ".join(str(p) for p in self.parameters)
        prefix = "async " if self.async_ else ""
        throws_str = "throws " if self.throws else ""
        return f"{prefix}({params}) {throws_str}-> {self.return_type}"


@dataclass(frozen=True)
class SwiftClosureType(Type):
    """Swift closure type with attributes: @escaping (T) -> R"""
    function: SwiftFunctionType
    escaping: bool = False
    autoclosure: bool = False

    def __str__(self) -> str:
        attrs = []
        if self.escaping:
            attrs.append("@escaping")
        if self.autoclosure:
            attrs.append("@autoclosure")
        prefix = " ".join(attrs) + " " if attrs else ""
        return f"{prefix}{self.function}"


@dataclass(frozen=True)
class SwiftProtocolType(Type):
    """Swift protocol type."""
    name: str
    associated_types: Tuple[Tuple[str, Optional[Type]], ...] = ()

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class SwiftProtocolCompositionType(Type):
    """Swift protocol composition: P1 & P2"""
    protocols: Tuple[Type, ...]

    def __str__(self) -> str:
        return " & ".join(str(p) for p in self.protocols)


@dataclass(frozen=True)
class SwiftGenericType(Type):
    """Swift generic type: Array<T>"""
    base: str
    type_args: Tuple[Type, ...]

    def __str__(self) -> str:
        args = ", ".join(str(a) for a in self.type_args)
        return f"{self.base}<{args}>"


@dataclass(frozen=True)
class SwiftTypeParameter(Type):
    """Swift type parameter: T, T: Protocol"""
    name: str
    constraints: Tuple[Type, ...] = ()

    def __str__(self) -> str:
        if self.constraints:
            constrs = " & ".join(str(c) for c in self.constraints)
            return f"{self.name}: {constrs}"
        return self.name


@dataclass(frozen=True)
class SwiftMetatypeType(Type):
    """Swift metatype: T.Type or T.Protocol"""
    instance_type: Type
    is_protocol: bool = False

    def __str__(self) -> str:
        suffix = "Protocol" if self.is_protocol else "Type"
        return f"{self.instance_type}.{suffix}"


@dataclass(frozen=True)
class SwiftExistentialType(Type):
    """Swift existential type: any Protocol"""
    protocol: Type

    def __str__(self) -> str:
        return f"any {self.protocol}"


@dataclass(frozen=True)
class SwiftOpaqueType(Type):
    """Swift opaque type: some Protocol"""
    constraint: Type

    def __str__(self) -> str:
        return f"some {self.constraint}"


@dataclass(frozen=True)
class SwiftResultType(Type):
    """Swift Result type: Result<Success, Failure>"""
    success: Type
    failure: Type

    def __str__(self) -> str:
        return f"Result<{self.success}, {self.failure}>"


@dataclass(frozen=True)
class SwiftNamedType(Type):
    """Swift named type (class, struct, enum)."""
    name: str
    is_class: bool = False
    is_struct: bool = True
    is_enum: bool = False

    def __str__(self) -> str:
        return self.name


# =============================================================================
# Swift Type System
# =============================================================================

class SwiftTypeSystem(LanguageTypeSystem):
    """Swift type system implementation."""

    def __init__(self):
        self._type_cache: Dict[str, Type] = {}

    @property
    def name(self) -> str:
        return "swift"

    @property
    def capabilities(self) -> TypeSystemCapabilities:
        return TypeSystemCapabilities(
            supports_generics=True,
            supports_union_types=False,         # Use enums with associated values
            supports_optional_types=True,       # T?
            supports_type_inference=True,
            supports_protocols=True,            # Protocol-oriented programming
            supports_variance=True,             # Covariance for protocols
            supports_overloading=True,
            supports_ownership=False,           # ARC, not manual
            supports_comptime=False,
            supports_error_unions=False,        # Uses throws/Result
            supports_lifetime_bounds=False,
            supports_sentinels=False,
            supports_allocators=False,
        )

    def parse_type_annotation(self, annotation: str) -> Type:
        """Parse a Swift type annotation string."""
        annotation = annotation.strip()

        if not annotation:
            raise TypeParseError("Empty type annotation")

        if annotation in self._type_cache:
            return self._type_cache[annotation]

        result = self._parse_type(annotation)
        self._type_cache[annotation] = result
        return result

    def _parse_type(self, text: str) -> Type:
        """Internal type parsing."""
        text = text.strip()

        # Check primitives
        if text in SWIFT_PRIMITIVES:
            return SWIFT_PRIMITIVES[text]

        # Optional type: T?
        if text.endswith("?") and not text.endswith("??"):
            inner = self._parse_type(text[:-1])
            return SwiftOptionalType(inner)

        # Implicitly unwrapped optional: T!
        if text.endswith("!"):
            inner = self._parse_type(text[:-1])
            return SwiftImplicitlyUnwrappedOptionalType(inner)

        # Array type: [T]
        if text.startswith("[") and text.endswith("]") and ":" not in text:
            inner = self._parse_type(text[1:-1])
            return SwiftArrayType(inner)

        # Dictionary type: [K: V]
        if text.startswith("[") and text.endswith("]") and ":" in text:
            colon_idx = self._find_colon_in_brackets(text)
            if colon_idx > 0:
                key = self._parse_type(text[1:colon_idx])
                value = self._parse_type(text[colon_idx + 1:-1])
                return SwiftDictionaryType(key, value)

        # Tuple type: (T1, T2)
        if text.startswith("(") and text.endswith(")") and "->" not in text:
            inner = text[1:-1]
            if inner:
                elements = self._split_type_list(inner)
                types = [self._parse_type(e.strip()) for e in elements]
                return SwiftTupleType(tuple(types))
            return SWIFT_VOID

        # Function type: (P) -> R
        if "->" in text:
            return self._parse_function_type(text)

        # Protocol composition: P1 & P2
        if " & " in text:
            parts = text.split(" & ")
            protocols = [self._parse_type(p.strip()) for p in parts]
            return SwiftProtocolCompositionType(tuple(protocols))

        # Opaque type: some Protocol
        if text.startswith("some "):
            constraint = self._parse_type(text[5:])
            return SwiftOpaqueType(constraint)

        # Existential type: any Protocol
        if text.startswith("any "):
            protocol = self._parse_type(text[4:])
            return SwiftExistentialType(protocol)

        # Generic type: Array<T>
        if "<" in text and text.endswith(">"):
            bracket_idx = text.index("<")
            base = text[:bracket_idx]
            args_str = text[bracket_idx + 1:-1]
            args = [self._parse_type(a.strip()) for a in self._split_type_list(args_str)]

            # Check for specific types
            if base == "Set" and len(args) == 1:
                return SwiftSetType(args[0])
            if base == "Result" and len(args) == 2:
                return SwiftResultType(args[0], args[1])
            if base == "Optional" and len(args) == 1:
                return SwiftOptionalType(args[0])
            if base == "Array" and len(args) == 1:
                return SwiftArrayType(args[0])
            if base == "Dictionary" and len(args) == 2:
                return SwiftDictionaryType(args[0], args[1])

            return SwiftGenericType(base, tuple(args))

        # Metatype: T.Type or T.Protocol
        if text.endswith(".Type"):
            inner = self._parse_type(text[:-5])
            return SwiftMetatypeType(inner, is_protocol=False)
        if text.endswith(".Protocol"):
            inner = self._parse_type(text[:-9])
            return SwiftMetatypeType(inner, is_protocol=True)

        # Simple type name
        if re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", text):
            return SwiftNamedType(text)

        # Qualified type: Module.Type
        if "." in text and not text.endswith(".Type") and not text.endswith(".Protocol"):
            return SwiftNamedType(text)

        raise TypeParseError(f"Cannot parse Swift type: {text}")

    def _parse_function_type(self, text: str) -> SwiftFunctionType:
        """Parse a function type."""
        text = text.strip()

        # Check for async
        is_async = text.startswith("async ")
        if is_async:
            text = text[6:].strip()

        # Find the arrow
        arrow_idx = self._find_arrow(text)
        if arrow_idx == -1:
            raise TypeParseError(f"Invalid function type: {text}")

        params_part = text[:arrow_idx].strip()
        return_part = text[arrow_idx + 2:].strip()

        # Check for throws before return type
        throws = False
        if return_part.startswith("throws "):
            throws = True
            return_part = return_part[7:].strip()

        # Parse parameters
        if params_part.startswith("(") and params_part.endswith(")"):
            params_str = params_part[1:-1]
        else:
            params_str = params_part

        params: List[Type] = []
        if params_str.strip():
            for p in self._split_type_list(params_str):
                p = p.strip()
                if p:
                    # Handle labeled parameters
                    if ":" in p and not p.startswith("["):
                        parts = p.split(":", 1)
                        p = parts[1].strip()
                    params.append(self._parse_type(p))

        return_type = self._parse_type(return_part)

        return SwiftFunctionType(
            parameters=tuple(params),
            return_type=return_type,
            throws=throws,
            async_=is_async,
        )

    def _find_arrow(self, text: str) -> int:
        """Find -> respecting brackets."""
        depth = 0
        for i in range(len(text) - 1):
            if text[i] in "(<[":
                depth += 1
            elif text[i] in ")>]":
                depth -= 1
            elif depth == 0 and text[i:i+2] == "->":
                return i
        return -1

    def _find_colon_in_brackets(self, text: str) -> int:
        """Find : in dictionary type [K: V]."""
        depth = 0
        for i, char in enumerate(text):
            if char in "(<[":
                depth += 1
            elif char in ")>]":
                depth -= 1
            elif char == ":" and depth == 1:
                return i
        return -1

    def _split_type_list(self, text: str) -> List[str]:
        """Split comma-separated type list respecting brackets."""
        parts = []
        current: List[str] = []
        depth = 0

        for char in text:
            if char in "(<[":
                depth += 1
                current.append(char)
            elif char in ")>]":
                depth -= 1
                current.append(char)
            elif char == "," and depth == 0:
                parts.append("".join(current))
                current = []
            else:
                current.append(char)

        if current:
            parts.append("".join(current))

        return parts

    def infer_literal_type(self, literal: LiteralInfo) -> Type:
        """Infer the type of a literal value."""
        if literal.kind == LiteralKind.INTEGER:
            return SWIFT_INT
        elif literal.kind == LiteralKind.FLOAT:
            return SWIFT_DOUBLE
        elif literal.kind == LiteralKind.STRING:
            return SWIFT_STRING
        elif literal.kind == LiteralKind.BOOLEAN:
            return SWIFT_BOOL
        elif literal.kind == LiteralKind.NONE:
            # nil needs context
            return SwiftOptionalType(SWIFT_ANY)
        elif literal.kind == LiteralKind.CHAR:
            return SWIFT_CHARACTER
        else:
            return SWIFT_ANY

    def check_assignable(self, source: Type, target: Type) -> bool:
        """Check if source type is assignable to target type."""
        # Same type
        if source == target:
            return True

        # Never is assignable to everything
        if isinstance(source, NeverType):
            return True

        # Any accepts anything
        if isinstance(target, AnyType):
            return True

        # Non-optional to optional
        if isinstance(target, SwiftOptionalType) and not isinstance(source, SwiftOptionalType):
            return self.check_assignable(source, target.wrapped)

        # Optional to optional
        if isinstance(source, SwiftOptionalType) and isinstance(target, SwiftOptionalType):
            return self.check_assignable(source.wrapped, target.wrapped)

        # Array covariance
        if isinstance(source, SwiftArrayType) and isinstance(target, SwiftArrayType):
            return self.check_assignable(source.element, target.element)

        # Dictionary covariance
        if isinstance(source, SwiftDictionaryType) and isinstance(target, SwiftDictionaryType):
            return (self.check_assignable(source.key, target.key) and
                    self.check_assignable(source.value, target.value))

        # Function type assignability
        if isinstance(source, SwiftFunctionType) and isinstance(target, SwiftFunctionType):
            return self._check_function_assignable(source, target)

        # Protocol conformance would go here (simplified)
        if isinstance(target, SwiftProtocolType):
            return True  # Simplified - real check would verify conformance

        return False

    def _check_function_assignable(
        self,
        source: SwiftFunctionType,
        target: SwiftFunctionType
    ) -> bool:
        """Check function type assignability."""
        if len(source.parameters) != len(target.parameters):
            return False

        # throws must match (throws -> nonthrowing is not allowed)
        if source.throws and not target.throws:
            return False

        # async must match
        if source.async_ != target.async_:
            return False

        # Parameters are contravariant
        for sp, tp in zip(source.parameters, target.parameters):
            if not self.check_assignable(tp, sp):
                return False

        # Return type is covariant
        return self.check_assignable(source.return_type, target.return_type)

    def format_type(self, typ: Type) -> str:
        """Format a type for display."""
        return str(typ)

    def get_builtin_types(self) -> Dict[str, Type]:
        """Get all builtin types."""
        return SWIFT_PRIMITIVES.copy()

    def get_builtin_functions(self) -> Dict[str, BaseFunctionType]:
        """Get builtin functions with their types."""
        return {
            "print": BaseFunctionType((SWIFT_ANY,), SWIFT_VOID),
            "debugPrint": BaseFunctionType((SWIFT_ANY,), SWIFT_VOID),
            "type": BaseFunctionType((SWIFT_ANY,), SWIFT_ANY),
            "abs": BaseFunctionType((SWIFT_INT,), SWIFT_INT),
            "min": BaseFunctionType((SWIFT_INT, SWIFT_INT), SWIFT_INT),
            "max": BaseFunctionType((SWIFT_INT, SWIFT_INT), SWIFT_INT),
            "fatalError": BaseFunctionType((SWIFT_STRING,), SWIFT_NEVER),
        }

    def lub(self, types: List[Type]) -> Type:
        """Compute least upper bound of types."""
        if not types:
            return SWIFT_NEVER
        if len(types) == 1:
            return types[0]

        unique = list(dict.fromkeys(types))
        if len(unique) == 1:
            return unique[0]

        # Check for optionals
        has_optional = any(isinstance(t, SwiftOptionalType) for t in unique)
        if has_optional:
            return SwiftOptionalType(SWIFT_ANY)

        return SWIFT_ANY

    def glb(self, types: List[Type]) -> Type:
        """Compute greatest lower bound of types."""
        if not types:
            return SWIFT_ANY
        if len(types) == 1:
            return types[0]

        unique = list(dict.fromkeys(types))
        if len(unique) == 1:
            return unique[0]

        return SWIFT_NEVER
