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
"""Kotlin type system implementation.

This module implements the Kotlin type system for constrained decoding,
supporting Kotlin's null-safe type hierarchy, generics, and function types.

Features:
- Primitive types: Byte, Short, Int, Long, Float, Double, Boolean, Char
- Nullable types: T?
- Array types: Array<T>, IntArray, etc.
- Collection types: List<T>, Set<T>, Map<K, V>
- Function types: (P1, P2) -> R
- Lambda with receiver: T.() -> R
- Generic types with variance (in, out)

References:
    - Kotlin Language Specification: https://kotlinlang.org/spec/
    - Kotlin Type System: https://kotlinlang.org/docs/basic-types.html
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
    Union,
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
# Kotlin Primitive Types
# =============================================================================

# Numeric types
KOTLIN_BYTE = PrimitiveType("Byte")
KOTLIN_SHORT = PrimitiveType("Short")
KOTLIN_INT = PrimitiveType("Int")
KOTLIN_LONG = PrimitiveType("Long")
KOTLIN_FLOAT = PrimitiveType("Float")
KOTLIN_DOUBLE = PrimitiveType("Double")

# Unsigned types
KOTLIN_UBYTE = PrimitiveType("UByte")
KOTLIN_USHORT = PrimitiveType("UShort")
KOTLIN_UINT = PrimitiveType("UInt")
KOTLIN_ULONG = PrimitiveType("ULong")

# Boolean and Char
KOTLIN_BOOLEAN = PrimitiveType("Boolean")
KOTLIN_CHAR = PrimitiveType("Char")

# String
KOTLIN_STRING = PrimitiveType("String")

# Special types
KOTLIN_UNIT = PrimitiveType("Unit")
KOTLIN_ANY = AnyType()  # Top type
KOTLIN_NOTHING = NeverType()  # Bottom type


# Primitive type lookup
KOTLIN_PRIMITIVES: Dict[str, Type] = {
    "Byte": KOTLIN_BYTE,
    "Short": KOTLIN_SHORT,
    "Int": KOTLIN_INT,
    "Long": KOTLIN_LONG,
    "Float": KOTLIN_FLOAT,
    "Double": KOTLIN_DOUBLE,
    "UByte": KOTLIN_UBYTE,
    "UShort": KOTLIN_USHORT,
    "UInt": KOTLIN_UINT,
    "ULong": KOTLIN_ULONG,
    "Boolean": KOTLIN_BOOLEAN,
    "Char": KOTLIN_CHAR,
    "String": KOTLIN_STRING,
    "Unit": KOTLIN_UNIT,
    "Any": KOTLIN_ANY,
    "Nothing": KOTLIN_NOTHING,
}


# =============================================================================
# Kotlin Composite Types
# =============================================================================

@dataclass(frozen=True)
class KotlinNullableType(Type):
    """Kotlin nullable type: T?"""
    inner: Type

    def __str__(self) -> str:
        inner_str = str(self.inner)
        # Don't double-mark nullable or wrap function types without parens
        if isinstance(self.inner, KotlinFunctionType):
            return f"({inner_str})?"
        return f"{inner_str}?"


@dataclass(frozen=True)
class KotlinArrayType(Type):
    """Kotlin array type: Array<T>"""
    element: Type

    def __str__(self) -> str:
        return f"Array<{self.element}>"


@dataclass(frozen=True)
class KotlinPrimitiveArrayType(Type):
    """Kotlin primitive array types: IntArray, etc."""
    element_type: str  # "Int", "Long", etc.

    def __str__(self) -> str:
        return f"{self.element_type}Array"


@dataclass(frozen=True)
class KotlinListType(Type):
    """Kotlin list type: List<T>"""
    element: Type
    mutable: bool = False

    def __str__(self) -> str:
        prefix = "MutableList" if self.mutable else "List"
        return f"{prefix}<{self.element}>"


@dataclass(frozen=True)
class KotlinSetType(Type):
    """Kotlin set type: Set<T>"""
    element: Type
    mutable: bool = False

    def __str__(self) -> str:
        prefix = "MutableSet" if self.mutable else "Set"
        return f"{prefix}<{self.element}>"


@dataclass(frozen=True)
class KotlinMapType(Type):
    """Kotlin map type: Map<K, V>"""
    key: Type
    value: Type
    mutable: bool = False

    def __str__(self) -> str:
        prefix = "MutableMap" if self.mutable else "Map"
        return f"{prefix}<{self.key}, {self.value}>"


@dataclass(frozen=True)
class KotlinPairType(Type):
    """Kotlin Pair type: Pair<A, B>"""
    first: Type
    second: Type

    def __str__(self) -> str:
        return f"Pair<{self.first}, {self.second}>"


@dataclass(frozen=True)
class KotlinFunctionType(Type):
    """Kotlin function type: (P1, P2) -> R or T.() -> R"""
    parameters: Tuple[Type, ...]
    return_type: Type
    receiver: Optional[Type] = None
    is_suspend: bool = False

    def __str__(self) -> str:
        params = ", ".join(str(p) for p in self.parameters)
        suspend = "suspend " if self.is_suspend else ""
        if self.receiver:
            return f"{suspend}{self.receiver}.({params}) -> {self.return_type}"
        return f"{suspend}({params}) -> {self.return_type}"


@dataclass(frozen=True)
class KotlinTypeParameter(Type):
    """Kotlin type parameter: T, out T, in T"""
    name: str
    variance: Literal["invariant", "in", "out"] = "invariant"
    bound: Optional[Type] = None

    def __str__(self) -> str:
        variance_str = ""
        if self.variance == "in":
            variance_str = "in "
        elif self.variance == "out":
            variance_str = "out "

        if self.bound:
            return f"{variance_str}{self.name} : {self.bound}"
        return f"{variance_str}{self.name}"


@dataclass(frozen=True)
class KotlinGenericType(Type):
    """Kotlin generic type instantiation: List<Int>"""
    base: str
    type_args: Tuple[Type, ...]

    def __str__(self) -> str:
        args = ", ".join(str(a) for a in self.type_args)
        return f"{self.base}<{args}>"


@dataclass(frozen=True)
class KotlinStarProjection(Type):
    """Kotlin star projection: *"""

    def __str__(self) -> str:
        return "*"


@dataclass(frozen=True)
class KotlinClassType(Type):
    """Kotlin class or interface type."""
    name: str
    type_params: Tuple[KotlinTypeParameter, ...] = ()
    is_interface: bool = False
    is_data: bool = False
    is_sealed: bool = False
    is_object: bool = False
    is_inline: bool = False  # value class

    def __str__(self) -> str:
        if self.type_params:
            params = ", ".join(str(p) for p in self.type_params)
            return f"{self.name}<{params}>"
        return self.name


@dataclass(frozen=True)
class KotlinEnumType(Type):
    """Kotlin enum class type."""
    name: str
    entries: Tuple[str, ...] = ()

    def __str__(self) -> str:
        return self.name


# =============================================================================
# Kotlin Type System
# =============================================================================

class KotlinTypeSystem(LanguageTypeSystem):
    """Kotlin type system implementation."""

    def __init__(self):
        self._type_cache: Dict[str, Type] = {}

    @property
    def name(self) -> str:
        return "kotlin"

    @property
    def capabilities(self) -> TypeSystemCapabilities:
        return TypeSystemCapabilities(
            supports_generics=True,
            supports_union_types=False,         # No union types (use sealed classes)
            supports_optional_types=True,       # T?
            supports_type_inference=True,       # val x = ...
            supports_protocols=True,            # interfaces
            supports_variance=True,             # in/out variance
            supports_overloading=True,          # Function overloading
            supports_ownership=False,
            supports_comptime=False,
            supports_error_unions=False,
            supports_lifetime_bounds=False,
            supports_sentinels=False,
            supports_allocators=False,
        )

    def parse_type_annotation(self, annotation: str) -> Type:
        """Parse a Kotlin type annotation string."""
        annotation = annotation.strip()

        if not annotation:
            raise TypeParseError("Empty type annotation")

        # Check cache
        if annotation in self._type_cache:
            return self._type_cache[annotation]

        result = self._parse_type(annotation)
        self._type_cache[annotation] = result
        return result

    def _parse_type(self, text: str) -> Type:
        """Internal type parsing."""
        text = text.strip()

        # Check primitives
        if text in KOTLIN_PRIMITIVES:
            return KOTLIN_PRIMITIVES[text]

        # Nullable type: T?
        if text.endswith("?"):
            inner = self._parse_type(text[:-1])
            return KotlinNullableType(inner)

        # Star projection
        if text == "*":
            return KotlinStarProjection()

        # Primitive arrays: IntArray, etc.
        primitive_arrays = {
            "IntArray": "Int",
            "LongArray": "Long",
            "ShortArray": "Short",
            "ByteArray": "Byte",
            "FloatArray": "Float",
            "DoubleArray": "Double",
            "BooleanArray": "Boolean",
            "CharArray": "Char",
            "UIntArray": "UInt",
            "ULongArray": "ULong",
            "UShortArray": "UShort",
            "UByteArray": "UByte",
        }
        if text in primitive_arrays:
            return KotlinPrimitiveArrayType(primitive_arrays[text])

        # Function type: (params) -> return
        if "->" in text:
            return self._parse_function_type(text)

        # Generic types: List<T>, Map<K, V>, etc.
        if "<" in text and text.endswith(">"):
            return self._parse_generic_type(text)

        # Simple type name or qualified name
        if re.match(r"^[a-zA-Z_][a-zA-Z0-9_.]*$", text):
            return KotlinClassType(text)

        raise TypeParseError(f"Cannot parse Kotlin type: {text}")

    def _parse_function_type(self, text: str) -> KotlinFunctionType:
        """Parse a function type like (Int, String) -> Boolean."""
        text = text.strip()

        # Check for suspend
        is_suspend = text.startswith("suspend ")
        if is_suspend:
            text = text[8:].strip()

        # Check for receiver type: T.() -> R
        receiver = None
        if "." in text and not text.startswith("("):
            # Find the dot before the opening paren
            paren_idx = text.find("(")
            if paren_idx > 0:
                dot_idx = text.rfind(".", 0, paren_idx)
                if dot_idx > 0:
                    receiver = self._parse_type(text[:dot_idx])
                    text = text[dot_idx + 1:]

        # Find the arrow
        arrow_idx = self._find_arrow(text)
        if arrow_idx == -1:
            raise TypeParseError(f"Invalid function type: {text}")

        params_part = text[:arrow_idx].strip()
        return_part = text[arrow_idx + 2:].strip()

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
                    params.append(self._parse_type(p))

        # Parse return type
        return_type = self._parse_type(return_part)

        return KotlinFunctionType(
            parameters=tuple(params),
            return_type=return_type,
            receiver=receiver,
            is_suspend=is_suspend,
        )

    def _find_arrow(self, text: str) -> int:
        """Find the position of -> respecting brackets."""
        depth = 0
        for i in range(len(text) - 1):
            if text[i] in "(<":
                depth += 1
            elif text[i] in ")>":
                depth -= 1
            elif depth == 0 and text[i:i+2] == "->":
                return i
        return -1

    def _parse_generic_type(self, text: str) -> Type:
        """Parse a generic type like List<Int> or Map<String, Int>."""
        bracket_idx = text.index("<")
        base = text[:bracket_idx]
        args_str = text[bracket_idx + 1:-1]

        # Parse type arguments
        type_args: List[Type] = []
        for arg in self._split_type_list(args_str):
            arg = arg.strip()
            if arg:
                # Handle variance modifiers in type arguments
                if arg.startswith("out "):
                    inner = self._parse_type(arg[4:])
                    type_args.append(KotlinTypeParameter("_", "out", inner))
                elif arg.startswith("in "):
                    inner = self._parse_type(arg[3:])
                    type_args.append(KotlinTypeParameter("_", "in", inner))
                else:
                    type_args.append(self._parse_type(arg))

        # Check for built-in collection types
        if base == "Array":
            if len(type_args) == 1:
                return KotlinArrayType(type_args[0])
        elif base == "List":
            if len(type_args) == 1:
                return KotlinListType(type_args[0], mutable=False)
        elif base == "MutableList":
            if len(type_args) == 1:
                return KotlinListType(type_args[0], mutable=True)
        elif base == "Set":
            if len(type_args) == 1:
                return KotlinSetType(type_args[0], mutable=False)
        elif base == "MutableSet":
            if len(type_args) == 1:
                return KotlinSetType(type_args[0], mutable=True)
        elif base == "Map":
            if len(type_args) == 2:
                return KotlinMapType(type_args[0], type_args[1], mutable=False)
        elif base == "MutableMap":
            if len(type_args) == 2:
                return KotlinMapType(type_args[0], type_args[1], mutable=True)
        elif base == "Pair":
            if len(type_args) == 2:
                return KotlinPairType(type_args[0], type_args[1])

        return KotlinGenericType(base, tuple(type_args))

    def _split_type_list(self, text: str) -> List[str]:
        """Split a comma-separated type list respecting brackets."""
        parts = []
        current: List[str] = []
        depth = 0

        for char in text:
            if char in "(<":
                depth += 1
                current.append(char)
            elif char in ")>":
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
            # Kotlin infers Int by default, Long if too big
            value = literal.raw_value
            if isinstance(value, int):
                if -2147483648 <= value <= 2147483647:
                    return KOTLIN_INT
                return KOTLIN_LONG
            return KOTLIN_INT
        elif literal.kind == LiteralKind.FLOAT:
            return KOTLIN_DOUBLE  # Kotlin defaults to Double
        elif literal.kind == LiteralKind.STRING:
            return KOTLIN_STRING
        elif literal.kind == LiteralKind.BOOLEAN:
            return KOTLIN_BOOLEAN
        elif literal.kind == LiteralKind.NONE:
            # null has type Nothing?
            return KotlinNullableType(KOTLIN_NOTHING)
        elif literal.kind == LiteralKind.CHAR:
            return KOTLIN_CHAR
        else:
            return KOTLIN_ANY

    def check_assignable(self, source: Type, target: Type) -> bool:
        """Check if source type is assignable to target type."""
        # Same type
        if source == target:
            return True

        # Nothing is assignable to everything
        if isinstance(source, NeverType):
            return True

        # Any accepts anything
        if isinstance(target, AnyType):
            return True

        # Nothing? (null literal) is assignable to any nullable type
        if isinstance(source, KotlinNullableType) and isinstance(source.inner, NeverType):
            if isinstance(target, KotlinNullableType):
                return True

        # Non-nullable to nullable
        if isinstance(target, KotlinNullableType) and not isinstance(source, KotlinNullableType):
            return self.check_assignable(source, target.inner)

        # Nullable to nullable (with inner check)
        if isinstance(source, KotlinNullableType) and isinstance(target, KotlinNullableType):
            return self.check_assignable(source.inner, target.inner)

        # Numeric conversions (Kotlin doesn't auto-widen, but we allow explicit cases)
        if isinstance(source, PrimitiveType) and isinstance(target, PrimitiveType):
            return self._check_numeric_assignable(source, target)

        # Collection covariance (immutable List<Derived> to List<Base>)
        if isinstance(source, KotlinListType) and isinstance(target, KotlinListType):
            if not source.mutable and not target.mutable:
                return self.check_assignable(source.element, target.element)
            return source.element == target.element and source.mutable == target.mutable

        if isinstance(source, KotlinSetType) and isinstance(target, KotlinSetType):
            if not source.mutable and not target.mutable:
                return self.check_assignable(source.element, target.element)
            return source.element == target.element and source.mutable == target.mutable

        # Map covariance (for immutable)
        if isinstance(source, KotlinMapType) and isinstance(target, KotlinMapType):
            if not source.mutable and not target.mutable:
                return (self.check_assignable(source.key, target.key) and
                        self.check_assignable(source.value, target.value))
            return (source.key == target.key and source.value == target.value
                    and source.mutable == target.mutable)

        # Array invariance
        if isinstance(source, KotlinArrayType) and isinstance(target, KotlinArrayType):
            return source.element == target.element

        # Function type assignability (contravariant params, covariant return)
        if isinstance(source, KotlinFunctionType) and isinstance(target, KotlinFunctionType):
            return self._check_function_assignable(source, target)

        return False

    def _check_numeric_assignable(self, source: PrimitiveType, target: PrimitiveType) -> bool:
        """Check numeric type assignability (Kotlin is strict about this)."""
        # Kotlin doesn't auto-widen primitives
        return source.name == target.name

    def _check_function_assignable(
        self,
        source: KotlinFunctionType,
        target: KotlinFunctionType
    ) -> bool:
        """Check function type assignability."""
        # Same number of parameters
        if len(source.parameters) != len(target.parameters):
            return False

        # Suspend must match
        if source.is_suspend != target.is_suspend:
            return False

        # Receiver must be compatible (contravariant)
        if source.receiver and target.receiver:
            if not self.check_assignable(target.receiver, source.receiver):
                return False
        elif source.receiver or target.receiver:
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
        return KOTLIN_PRIMITIVES.copy()

    def get_builtin_functions(self) -> Dict[str, BaseFunctionType]:
        """Get builtin functions with their types."""
        return {
            "println": BaseFunctionType((KOTLIN_ANY,), KOTLIN_UNIT),
            "print": BaseFunctionType((KOTLIN_ANY,), KOTLIN_UNIT),
            "readln": BaseFunctionType((), KOTLIN_STRING),
            "listOf": BaseFunctionType((KOTLIN_ANY,), KOTLIN_ANY),
            "mutableListOf": BaseFunctionType((KOTLIN_ANY,), KOTLIN_ANY),
            "setOf": BaseFunctionType((KOTLIN_ANY,), KOTLIN_ANY),
            "mutableSetOf": BaseFunctionType((KOTLIN_ANY,), KOTLIN_ANY),
            "mapOf": BaseFunctionType((KOTLIN_ANY,), KOTLIN_ANY),
            "mutableMapOf": BaseFunctionType((KOTLIN_ANY,), KOTLIN_ANY),
            "arrayOf": BaseFunctionType((KOTLIN_ANY,), KOTLIN_ANY),
            "require": BaseFunctionType((KOTLIN_BOOLEAN,), KOTLIN_UNIT),
            "check": BaseFunctionType((KOTLIN_BOOLEAN,), KOTLIN_UNIT),
            "error": BaseFunctionType((KOTLIN_ANY,), KOTLIN_NOTHING),
            "TODO": BaseFunctionType((), KOTLIN_NOTHING),
        }

    def lub(self, types: List[Type]) -> Type:
        """Compute least upper bound of types."""
        if not types:
            return KOTLIN_NOTHING
        if len(types) == 1:
            return types[0]

        # Remove duplicates
        unique = list(dict.fromkeys(types))
        if len(unique) == 1:
            return unique[0]

        # Check if any is nullable
        has_nullable = any(isinstance(t, KotlinNullableType) for t in unique)

        # For Kotlin, LUB defaults to Any (or Any? if nullable)
        if has_nullable:
            return KotlinNullableType(KOTLIN_ANY)
        return KOTLIN_ANY

    def glb(self, types: List[Type]) -> Type:
        """Compute greatest lower bound of types."""
        if not types:
            return KOTLIN_ANY
        if len(types) == 1:
            return types[0]

        # Remove duplicates
        unique = list(dict.fromkeys(types))
        if len(unique) == 1:
            return unique[0]

        # For Kotlin, GLB is Nothing
        return KOTLIN_NOTHING
