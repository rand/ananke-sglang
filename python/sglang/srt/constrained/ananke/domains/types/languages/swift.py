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
        elif literal.kind == LiteralKind.CHARACTER:
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

        # Protocol conformance
        if isinstance(target, SwiftProtocolType):
            return self._check_protocol_conformance(source, target)

        # Protocol composition (P1 & P2)
        if isinstance(target, SwiftProtocolCompositionType):
            return self._check_protocol_composition_conformance(source, target)

        # Existential type (any Protocol)
        if isinstance(target, SwiftExistentialType):
            return self._check_existential_assignable(source, target)

        # Opaque type (some Protocol) - source must conform to constraint
        if isinstance(target, SwiftOpaqueType):
            return self._check_opaque_assignable(source, target)

        # Generic type assignability
        if isinstance(source, SwiftGenericType) and isinstance(target, SwiftGenericType):
            return self._check_generic_assignable(source, target)

        # Named type assignability (struct/class/enum)
        if isinstance(source, SwiftNamedType) and isinstance(target, SwiftNamedType):
            return self._check_named_type_assignable(source, target)

        # Set covariance
        if isinstance(source, SwiftSetType) and isinstance(target, SwiftSetType):
            return self.check_assignable(source.element, target.element)

        # Result type assignability
        if isinstance(source, SwiftResultType) and isinstance(target, SwiftResultType):
            return (self.check_assignable(source.success, target.success) and
                    self.check_assignable(source.failure, target.failure))

        # Tuple assignability
        if isinstance(source, SwiftTupleType) and isinstance(target, SwiftTupleType):
            return self._check_tuple_assignable(source, target)

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

    def _check_protocol_conformance(self, source: Type, target: SwiftProtocolType) -> bool:
        """Check if a type conforms to a protocol.

        Swift uses nominal protocol conformance - a type must explicitly
        declare conformance. This implementation provides heuristic checking
        for common Swift protocols since full conformance tracking would
        require parsing entire codebases.

        Common protocols checked:
        - Equatable, Hashable, Comparable, Identifiable
        - Codable (Encodable & Decodable)
        - Sequence, Collection, IteratorProtocol
        - CustomStringConvertible, CustomDebugStringConvertible
        - Error
        """
        protocol_name = target.name

        # Get the source type name for conformance checking
        source_name = self._get_type_name(source)

        # Check known conformances
        return self._type_conforms_to_protocol(source, source_name, protocol_name)

    def _get_type_name(self, typ: Type) -> str:
        """Extract the base type name from a type."""
        if isinstance(typ, PrimitiveType):
            return typ.name
        elif isinstance(typ, SwiftNamedType):
            return typ.name
        elif isinstance(typ, SwiftGenericType):
            return typ.base
        elif isinstance(typ, SwiftArrayType):
            return "Array"
        elif isinstance(typ, SwiftDictionaryType):
            return "Dictionary"
        elif isinstance(typ, SwiftSetType):
            return "Set"
        elif isinstance(typ, SwiftOptionalType):
            return "Optional"
        elif isinstance(typ, SwiftResultType):
            return "Result"
        elif isinstance(typ, SwiftTupleType):
            return "Tuple"
        elif isinstance(typ, SwiftProtocolType):
            return typ.name
        return ""

    def _type_conforms_to_protocol(self, typ: Type, type_name: str, protocol: str) -> bool:
        """Check if a type conforms to a specific protocol.

        This provides heuristic conformance checking for common Swift types.
        """
        # Known protocol conformances for Swift standard library types
        # Structure: type_name -> set of protocols it conforms to
        conformances: Dict[str, FrozenSet[str]] = {
            # Primitives
            "Int": frozenset({"Equatable", "Hashable", "Comparable", "Codable",
                             "SignedNumeric", "BinaryInteger", "FixedWidthInteger",
                             "CustomStringConvertible", "LosslessStringConvertible"}),
            "Int8": frozenset({"Equatable", "Hashable", "Comparable", "Codable",
                              "SignedNumeric", "BinaryInteger", "FixedWidthInteger"}),
            "Int16": frozenset({"Equatable", "Hashable", "Comparable", "Codable",
                               "SignedNumeric", "BinaryInteger", "FixedWidthInteger"}),
            "Int32": frozenset({"Equatable", "Hashable", "Comparable", "Codable",
                               "SignedNumeric", "BinaryInteger", "FixedWidthInteger"}),
            "Int64": frozenset({"Equatable", "Hashable", "Comparable", "Codable",
                               "SignedNumeric", "BinaryInteger", "FixedWidthInteger"}),
            "UInt": frozenset({"Equatable", "Hashable", "Comparable", "Codable",
                              "UnsignedInteger", "BinaryInteger", "FixedWidthInteger"}),
            "UInt8": frozenset({"Equatable", "Hashable", "Comparable", "Codable",
                               "UnsignedInteger", "BinaryInteger", "FixedWidthInteger"}),
            "UInt16": frozenset({"Equatable", "Hashable", "Comparable", "Codable",
                                "UnsignedInteger", "BinaryInteger", "FixedWidthInteger"}),
            "UInt32": frozenset({"Equatable", "Hashable", "Comparable", "Codable",
                                "UnsignedInteger", "BinaryInteger", "FixedWidthInteger"}),
            "UInt64": frozenset({"Equatable", "Hashable", "Comparable", "Codable",
                                "UnsignedInteger", "BinaryInteger", "FixedWidthInteger"}),
            "Float": frozenset({"Equatable", "Hashable", "Comparable", "Codable",
                               "FloatingPoint", "BinaryFloatingPoint"}),
            "Double": frozenset({"Equatable", "Hashable", "Comparable", "Codable",
                                "FloatingPoint", "BinaryFloatingPoint"}),
            "Float16": frozenset({"Equatable", "Hashable", "Comparable", "Codable",
                                 "FloatingPoint", "BinaryFloatingPoint"}),
            "Bool": frozenset({"Equatable", "Hashable", "Codable", "CustomStringConvertible"}),
            "Character": frozenset({"Equatable", "Hashable", "Comparable",
                                   "CustomStringConvertible"}),
            "String": frozenset({"Equatable", "Hashable", "Comparable", "Codable",
                                "Collection", "Sequence", "BidirectionalCollection",
                                "StringProtocol", "CustomStringConvertible",
                                "ExpressibleByStringLiteral", "LosslessStringConvertible"}),

            # Collections
            "Array": frozenset({"Equatable", "Hashable", "Codable", "Sequence", "Collection",
                               "MutableCollection", "RandomAccessCollection",
                               "RangeReplaceableCollection", "ExpressibleByArrayLiteral"}),
            "Dictionary": frozenset({"Equatable", "Codable", "Sequence", "Collection",
                                    "ExpressibleByDictionaryLiteral"}),
            "Set": frozenset({"Equatable", "Hashable", "Codable", "Sequence", "Collection",
                             "SetAlgebra", "ExpressibleByArrayLiteral"}),
            "Optional": frozenset({"Equatable", "Hashable", "ExpressibleByNilLiteral"}),

            # Result type
            "Result": frozenset({"Equatable", "Hashable"}),

            # Common Foundation types
            "URL": frozenset({"Equatable", "Hashable", "Codable"}),
            "Data": frozenset({"Equatable", "Hashable", "Codable", "Collection", "Sequence"}),
            "Date": frozenset({"Equatable", "Hashable", "Comparable", "Codable"}),
            "UUID": frozenset({"Equatable", "Hashable", "Codable", "CustomStringConvertible"}),

            # Common SwiftUI types
            "Color": frozenset({"Equatable", "Hashable", "View"}),
            "Text": frozenset({"View"}),
            "Image": frozenset({"View"}),
        }

        # Check direct conformance
        if type_name in conformances:
            if protocol in conformances[type_name]:
                return True

        # Check conditional conformances for generic types
        if isinstance(typ, SwiftArrayType):
            # Array<Element> conforms to Equatable/Hashable if Element does
            if protocol in ("Equatable", "Hashable", "Codable"):
                element_name = self._get_type_name(typ.element)
                if element_name in conformances:
                    return protocol in conformances.get(element_name, frozenset())
            # Array always conforms to Sequence/Collection
            if protocol in ("Sequence", "Collection", "RandomAccessCollection"):
                return True

        if isinstance(typ, SwiftDictionaryType):
            # Dictionary<Key, Value> conditional conformances
            if protocol == "Codable":
                key_name = self._get_type_name(typ.key)
                value_name = self._get_type_name(typ.value)
                key_codable = "Codable" in conformances.get(key_name, frozenset())
                value_codable = "Codable" in conformances.get(value_name, frozenset())
                return key_codable and value_codable
            if protocol in ("Sequence", "Collection"):
                return True

        if isinstance(typ, SwiftSetType):
            if protocol in ("Sequence", "Collection", "SetAlgebra"):
                return True

        if isinstance(typ, SwiftOptionalType):
            # Optional<Wrapped> conforms to Equatable/Hashable if Wrapped does
            if protocol in ("Equatable", "Hashable"):
                wrapped_name = self._get_type_name(typ.wrapped)
                if wrapped_name in conformances:
                    return protocol in conformances.get(wrapped_name, frozenset())

        # Protocol inheritance - check if protocol inherits from target
        protocol_hierarchy = {
            "BinaryInteger": {"Numeric", "Hashable", "Equatable", "Comparable", "Strideable"},
            "SignedNumeric": {"Numeric"},
            "UnsignedInteger": {"BinaryInteger", "Numeric"},
            "FloatingPoint": {"Numeric", "Hashable", "Equatable", "Comparable", "Strideable"},
            "BinaryFloatingPoint": {"FloatingPoint"},
            "Collection": {"Sequence"},
            "MutableCollection": {"Collection", "Sequence"},
            "BidirectionalCollection": {"Collection", "Sequence"},
            "RandomAccessCollection": {"BidirectionalCollection", "Collection", "Sequence"},
            "RangeReplaceableCollection": {"Collection", "Sequence"},
            "StringProtocol": {"Collection", "Sequence", "Hashable", "Equatable", "Comparable"},
            "Codable": {"Encodable", "Decodable"},
        }

        # Check if the type's protocols include something that inherits from target
        if type_name in conformances:
            for conformed_protocol in conformances[type_name]:
                if conformed_protocol in protocol_hierarchy:
                    if protocol in protocol_hierarchy[conformed_protocol]:
                        return True

        return False

    def _check_protocol_composition_conformance(
        self,
        source: Type,
        target: SwiftProtocolCompositionType
    ) -> bool:
        """Check conformance to a protocol composition (P1 & P2).

        Source must conform to ALL protocols in the composition.
        """
        for protocol in target.protocols:
            if isinstance(protocol, SwiftProtocolType):
                if not self._check_protocol_conformance(source, protocol):
                    return False
            elif isinstance(protocol, SwiftNamedType):
                # Treat as protocol
                proto = SwiftProtocolType(protocol.name)
                if not self._check_protocol_conformance(source, proto):
                    return False
            else:
                # Unknown protocol type
                return False
        return True

    def _check_existential_assignable(
        self,
        source: Type,
        target: SwiftExistentialType
    ) -> bool:
        """Check if source can be assigned to 'any Protocol' type.

        In Swift 5.6+, existential types are written as 'any Protocol'.
        Any type that conforms to the protocol can be assigned.
        """
        protocol = target.protocol
        if isinstance(protocol, SwiftProtocolType):
            return self._check_protocol_conformance(source, protocol)
        elif isinstance(protocol, SwiftProtocolCompositionType):
            return self._check_protocol_composition_conformance(source, protocol)
        elif isinstance(protocol, SwiftNamedType):
            return self._check_protocol_conformance(source, SwiftProtocolType(protocol.name))
        return False

    def _check_opaque_assignable(
        self,
        source: Type,
        target: SwiftOpaqueType
    ) -> bool:
        """Check if source can satisfy 'some Protocol' constraint.

        Opaque return types (some Protocol) require conformance to the constraint.
        The key difference from existentials is that opaque types preserve
        the underlying type identity.
        """
        constraint = target.constraint
        if isinstance(constraint, SwiftProtocolType):
            return self._check_protocol_conformance(source, constraint)
        elif isinstance(constraint, SwiftProtocolCompositionType):
            return self._check_protocol_composition_conformance(source, constraint)
        elif isinstance(constraint, SwiftNamedType):
            return self._check_protocol_conformance(source, SwiftProtocolType(constraint.name))
        return False

    def _check_generic_assignable(
        self,
        source: SwiftGenericType,
        target: SwiftGenericType
    ) -> bool:
        """Check generic type assignability.

        Generic types must have matching base types and compatible type arguments.
        Swift collections are generally covariant for their element types.
        """
        if source.base != target.base:
            return False

        if len(source.type_args) != len(target.type_args):
            return False

        # Swift standard library collections are covariant
        covariant_types = {"Array", "Set", "Optional", "Result"}
        is_covariant = source.base in covariant_types

        for source_arg, target_arg in zip(source.type_args, target.type_args):
            if is_covariant:
                if not self.check_assignable(source_arg, target_arg):
                    return False
            else:
                # Invariant by default
                if source_arg != target_arg:
                    return False

        return True

    def _check_named_type_assignable(
        self,
        source: SwiftNamedType,
        target: SwiftNamedType
    ) -> bool:
        """Check named type assignability.

        Swift uses nominal typing - types must explicitly match or have
        a declared inheritance/conformance relationship.
        """
        # Exact match
        if source.name == target.name:
            return True

        # Check known inheritance relationships
        return self._is_known_subtype(source.name, target.name)

    def _is_known_subtype(self, child: str, parent: str) -> bool:
        """Check known type hierarchy relationships.

        This provides heuristic subtype checking for common Swift types.
        Full hierarchy tracking would require parsing type definitions.
        """
        hierarchies: Dict[str, FrozenSet[str]] = {
            # Error types
            "LocalizedError": frozenset({"Error"}),
            "DecodingError": frozenset({"Error"}),
            "EncodingError": frozenset({"Error"}),
            "URLError": frozenset({"Error"}),
            "CocoaError": frozenset({"Error"}),

            # Common Foundation hierarchies
            "NSObject": frozenset({"AnyObject"}),
            "NSString": frozenset({"NSObject", "AnyObject"}),
            "NSArray": frozenset({"NSObject", "AnyObject"}),
            "NSDictionary": frozenset({"NSObject", "AnyObject"}),

            # Number types (not directly inherited but related)
            "Int": frozenset({"SignedInteger", "BinaryInteger", "Numeric"}),
            "Double": frozenset({"BinaryFloatingPoint", "FloatingPoint", "Numeric"}),
        }

        if child in hierarchies:
            return parent in hierarchies[child]

        return False

    def _check_tuple_assignable(
        self,
        source: SwiftTupleType,
        target: SwiftTupleType
    ) -> bool:
        """Check tuple type assignability.

        Tuples must have same arity and assignable element types.
        Labels are structural - unlabeled can match labeled.
        """
        if len(source.elements) != len(target.elements):
            return False

        for src_elem, tgt_elem in zip(source.elements, target.elements):
            if not self.check_assignable(src_elem, tgt_elem):
                return False

        return True

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
