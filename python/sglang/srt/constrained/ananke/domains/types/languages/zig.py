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
"""Zig type system implementation.

This module implements the Zig type system with full comptime support.
Key features:

- Primitive types: i8-i128, u8-u128, f16-f128, bool, void, noreturn
- Comptime types: comptime_int, comptime_float, type, anytype
- Pointer types: *T, *const T, [*]T, [*:0]T, [*c]T
- Optional types: ?T
- Error union types: E!T, anyerror!T
- Slice types: []T, []const T, [:0]T
- Array types: [N]T, [N:sentinel]T
- Struct, enum, union types
- Comptime evaluation semantics

References:
    - Zig Language Reference: https://ziglang.org/documentation/master/
    - Zig Standard Library: https://ziglang.org/documentation/master/std/
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

from domains.types.constraint import (
    Type,
    TypeVar,
    PrimitiveType,
    FunctionType,
    TupleType,
    AnyType,
    NeverType,
    HoleType,
    ANY,
    NEVER,
)
from domains.types.extended_types import (
    PointerType,
    SliceType,
    ArrayType,
    ErrorSetType,
    ResultType,
    OptionalType as ExtOptionalType,
    ManyPointerType,
    CPointerType,
    ComptimeType,
    ANYERROR,
)
from domains.types.languages.base import (
    LanguageTypeSystem,
    TypeSystemCapabilities,
    TypeParseError,
    LiteralInfo,
    LiteralKind,
)


# =============================================================================
# Zig Primitive Types
# =============================================================================

# Signed integers
ZIG_I8 = PrimitiveType("i8")
ZIG_I16 = PrimitiveType("i16")
ZIG_I32 = PrimitiveType("i32")
ZIG_I64 = PrimitiveType("i64")
ZIG_I128 = PrimitiveType("i128")
ZIG_ISIZE = PrimitiveType("isize")

# Unsigned integers
ZIG_U8 = PrimitiveType("u8")
ZIG_U16 = PrimitiveType("u16")
ZIG_U32 = PrimitiveType("u32")
ZIG_U64 = PrimitiveType("u64")
ZIG_U128 = PrimitiveType("u128")
ZIG_USIZE = PrimitiveType("usize")

# Floating point
ZIG_F16 = PrimitiveType("f16")
ZIG_F32 = PrimitiveType("f32")
ZIG_F64 = PrimitiveType("f64")
ZIG_F80 = PrimitiveType("f80")
ZIG_F128 = PrimitiveType("f128")

# Special types
ZIG_BOOL = PrimitiveType("bool")
ZIG_VOID = PrimitiveType("void")
ZIG_NORETURN = PrimitiveType("noreturn")
ZIG_TYPE = PrimitiveType("type")
ZIG_ANYTYPE = PrimitiveType("anytype")
ZIG_ANYOPAQUE = PrimitiveType("anyopaque")
ZIG_ANYFRAME = PrimitiveType("anyframe")
ZIG_ANYERROR = PrimitiveType("anyerror")

# Comptime types
ZIG_COMPTIME_INT = PrimitiveType("comptime_int")
ZIG_COMPTIME_FLOAT = PrimitiveType("comptime_float")

# Special values (represented as types for constraint purposes)
ZIG_UNDEFINED = PrimitiveType("undefined")
ZIG_NULL = PrimitiveType("null")

# C interop types
ZIG_C_CHAR = PrimitiveType("c_char")
ZIG_C_SHORT = PrimitiveType("c_short")
ZIG_C_USHORT = PrimitiveType("c_ushort")
ZIG_C_INT = PrimitiveType("c_int")
ZIG_C_UINT = PrimitiveType("c_uint")
ZIG_C_LONG = PrimitiveType("c_long")
ZIG_C_ULONG = PrimitiveType("c_ulong")
ZIG_C_LONGLONG = PrimitiveType("c_longlong")
ZIG_C_ULONGLONG = PrimitiveType("c_ulonglong")
ZIG_C_LONGDOUBLE = PrimitiveType("c_longdouble")


# =============================================================================
# Zig-Specific Type Classes
# =============================================================================

@dataclass(frozen=True, slots=True)
class ZigOptionalType(Type):
    """Zig optional type ?T.

    Represents a value that may be null or a valid T.
    """
    inner: Type

    def free_type_vars(self) -> FrozenSet[str]:
        return self.inner.free_type_vars()

    def substitute(self, substitution: Dict[str, Type]) -> "ZigOptionalType":
        return ZigOptionalType(inner=self.inner.substitute(substitution))

    def __str__(self) -> str:
        return f"?{self.inner}"


@dataclass(frozen=True, slots=True)
class ZigErrorUnionType(Type):
    """Zig error union type E!T.

    Represents a value that is either an error from error set E,
    or a valid value of type T.
    """
    error_set: Type  # ErrorSetType or anyerror
    payload: Type

    def free_type_vars(self) -> FrozenSet[str]:
        return self.error_set.free_type_vars() | self.payload.free_type_vars()

    def substitute(self, substitution: Dict[str, Type]) -> "ZigErrorUnionType":
        return ZigErrorUnionType(
            error_set=self.error_set.substitute(substitution),
            payload=self.payload.substitute(substitution),
        )

    def __str__(self) -> str:
        return f"{self.error_set}!{self.payload}"


@dataclass(frozen=True, slots=True)
class ZigPointerType(Type):
    """Full Zig pointer type with all qualifiers.

    Represents *T, *const T, *volatile T, *align(N) T, etc.
    """
    pointee: Type
    is_const: bool = False
    is_volatile: bool = False
    is_allowzero: bool = False
    alignment: Optional[int] = None
    sentinel: Optional[Any] = None
    address_space: Optional[str] = None

    def free_type_vars(self) -> FrozenSet[str]:
        return self.pointee.free_type_vars()

    def substitute(self, substitution: Dict[str, Type]) -> "ZigPointerType":
        return ZigPointerType(
            pointee=self.pointee.substitute(substitution),
            is_const=self.is_const,
            is_volatile=self.is_volatile,
            is_allowzero=self.is_allowzero,
            alignment=self.alignment,
            sentinel=self.sentinel,
            address_space=self.address_space,
        )

    def __str__(self) -> str:
        parts = ["*"]
        if self.is_allowzero:
            parts.append("allowzero ")
        if self.alignment is not None:
            parts.append(f"align({self.alignment}) ")
        if self.address_space:
            parts.append(f"addrspace(.{self.address_space}) ")
        if self.is_const:
            parts.append("const ")
        if self.is_volatile:
            parts.append("volatile ")
        parts.append(str(self.pointee))
        return "".join(parts)


@dataclass(frozen=True, slots=True)
class ZigSliceType(Type):
    """Zig slice type []T or [:sentinel]T.

    A pointer + length pair providing bounds-checked access.
    """
    element: Type
    is_const: bool = False
    sentinel: Optional[Any] = None

    def free_type_vars(self) -> FrozenSet[str]:
        return self.element.free_type_vars()

    def substitute(self, substitution: Dict[str, Type]) -> "ZigSliceType":
        return ZigSliceType(
            element=self.element.substitute(substitution),
            is_const=self.is_const,
            sentinel=self.sentinel,
        )

    def __str__(self) -> str:
        sentinel_str = f":{self.sentinel}" if self.sentinel is not None else ""
        const_str = "const " if self.is_const else ""
        return f"[{sentinel_str}]{const_str}{self.element}"


@dataclass(frozen=True, slots=True)
class ZigArrayType(Type):
    """Zig array type [N]T or [N:sentinel]T."""
    element: Type
    length: Optional[int] = None  # None for comptime-known length
    sentinel: Optional[Any] = None

    def free_type_vars(self) -> FrozenSet[str]:
        return self.element.free_type_vars()

    def substitute(self, substitution: Dict[str, Type]) -> "ZigArrayType":
        return ZigArrayType(
            element=self.element.substitute(substitution),
            length=self.length,
            sentinel=self.sentinel,
        )

    def __str__(self) -> str:
        len_str = str(self.length) if self.length is not None else "_"
        sentinel_str = f":{self.sentinel}" if self.sentinel is not None else ""
        return f"[{len_str}{sentinel_str}]{self.element}"


@dataclass(frozen=True, slots=True)
class ZigManyPointerType(Type):
    """Zig many-item pointer [*]T or [*:sentinel]T."""
    element: Type
    is_const: bool = False
    sentinel: Optional[Any] = None
    alignment: Optional[int] = None

    def free_type_vars(self) -> FrozenSet[str]:
        return self.element.free_type_vars()

    def substitute(self, substitution: Dict[str, Type]) -> "ZigManyPointerType":
        return ZigManyPointerType(
            element=self.element.substitute(substitution),
            is_const=self.is_const,
            sentinel=self.sentinel,
            alignment=self.alignment,
        )

    def __str__(self) -> str:
        sentinel_str = f":{self.sentinel}" if self.sentinel is not None else ""
        const_str = "const " if self.is_const else ""
        return f"[*{sentinel_str}]{const_str}{self.element}"


@dataclass(frozen=True, slots=True)
class ZigCPointerType(Type):
    """Zig C pointer [*c]T for C interoperability."""
    element: Type
    is_const: bool = False

    def free_type_vars(self) -> FrozenSet[str]:
        return self.element.free_type_vars()

    def substitute(self, substitution: Dict[str, Type]) -> "ZigCPointerType":
        return ZigCPointerType(
            element=self.element.substitute(substitution),
            is_const=self.is_const,
        )

    def __str__(self) -> str:
        const_str = "const " if self.is_const else ""
        return f"[*c]{const_str}{self.element}"


@dataclass(frozen=True, slots=True)
class ZigStructType(Type):
    """Zig struct type."""
    name: Optional[str] = None
    fields: Tuple[Tuple[str, Type], ...] = ()
    is_packed: bool = False
    is_extern: bool = False
    backing_integer: Optional[Type] = None

    def free_type_vars(self) -> FrozenSet[str]:
        result: FrozenSet[str] = frozenset()
        for _, field_type in self.fields:
            result = result | field_type.free_type_vars()
        if self.backing_integer:
            result = result | self.backing_integer.free_type_vars()
        return result

    def substitute(self, substitution: Dict[str, Type]) -> "ZigStructType":
        new_fields = tuple(
            (name, field_type.substitute(substitution))
            for name, field_type in self.fields
        )
        new_backing = (
            self.backing_integer.substitute(substitution)
            if self.backing_integer else None
        )
        return ZigStructType(
            name=self.name,
            fields=new_fields,
            is_packed=self.is_packed,
            is_extern=self.is_extern,
            backing_integer=new_backing,
        )

    def __str__(self) -> str:
        if self.name:
            return self.name
        prefix = ""
        if self.is_packed:
            prefix = "packed "
        elif self.is_extern:
            prefix = "extern "
        return f"{prefix}struct{{...}}"


@dataclass(frozen=True, slots=True)
class ZigEnumType(Type):
    """Zig enum type."""
    name: Optional[str] = None
    tag_type: Optional[Type] = None
    variants: FrozenSet[str] = frozenset()
    is_extern: bool = False

    def free_type_vars(self) -> FrozenSet[str]:
        if self.tag_type:
            return self.tag_type.free_type_vars()
        return frozenset()

    def substitute(self, substitution: Dict[str, Type]) -> "ZigEnumType":
        new_tag = (
            self.tag_type.substitute(substitution)
            if self.tag_type else None
        )
        return ZigEnumType(
            name=self.name,
            tag_type=new_tag,
            variants=self.variants,
            is_extern=self.is_extern,
        )

    def __str__(self) -> str:
        if self.name:
            return self.name
        return "enum{...}"


@dataclass(frozen=True, slots=True)
class ZigUnionType(Type):
    """Zig union type (tagged or untagged)."""
    name: Optional[str] = None
    tag_type: Optional[Type] = None  # None for bare unions
    variants: Tuple[Tuple[str, Type], ...] = ()
    is_packed: bool = False
    is_extern: bool = False

    def free_type_vars(self) -> FrozenSet[str]:
        result: FrozenSet[str] = frozenset()
        if self.tag_type:
            result = result | self.tag_type.free_type_vars()
        for _, variant_type in self.variants:
            result = result | variant_type.free_type_vars()
        return result

    def substitute(self, substitution: Dict[str, Type]) -> "ZigUnionType":
        new_tag = (
            self.tag_type.substitute(substitution)
            if self.tag_type else None
        )
        new_variants = tuple(
            (name, variant_type.substitute(substitution))
            for name, variant_type in self.variants
        )
        return ZigUnionType(
            name=self.name,
            tag_type=new_tag,
            variants=new_variants,
            is_packed=self.is_packed,
            is_extern=self.is_extern,
        )

    def __str__(self) -> str:
        if self.name:
            return self.name
        return "union{...}"


@dataclass(frozen=True, slots=True)
class ZigFunctionType(Type):
    """Zig function type with calling convention."""
    params: Tuple[Type, ...]
    return_type: Type
    calling_convention: Optional[str] = None  # .C, .Naked, etc.
    is_generic: bool = False
    is_inline: bool = False

    def free_type_vars(self) -> FrozenSet[str]:
        result: FrozenSet[str] = frozenset()
        for param in self.params:
            result = result | param.free_type_vars()
        result = result | self.return_type.free_type_vars()
        return result

    def substitute(self, substitution: Dict[str, Type]) -> "ZigFunctionType":
        new_params = tuple(p.substitute(substitution) for p in self.params)
        new_return = self.return_type.substitute(substitution)
        return ZigFunctionType(
            params=new_params,
            return_type=new_return,
            calling_convention=self.calling_convention,
            is_generic=self.is_generic,
            is_inline=self.is_inline,
        )

    def __str__(self) -> str:
        cc = f"callconv(.{self.calling_convention}) " if self.calling_convention else ""
        params_str = ", ".join(str(p) for p in self.params)
        return f"fn({params_str}) {cc}{self.return_type}"


@dataclass(frozen=True, slots=True)
class ZigVectorType(Type):
    """Zig SIMD vector type @Vector(N, T)."""
    length: int
    element: Type

    def free_type_vars(self) -> FrozenSet[str]:
        return self.element.free_type_vars()

    def substitute(self, substitution: Dict[str, Type]) -> "ZigVectorType":
        return ZigVectorType(
            length=self.length,
            element=self.element.substitute(substitution),
        )

    def __str__(self) -> str:
        return f"@Vector({self.length}, {self.element})"


# =============================================================================
# Zig Type System Implementation
# =============================================================================

class ZigTypeSystem(LanguageTypeSystem):
    """Zig type system with full comptime support.

    Implements type parsing, inference, and checking for the Zig language.
    Key features include:

    - Comptime integer/float coercion to concrete types
    - Pointer and slice type handling with qualifiers
    - Error union type checking
    - Optional type handling
    - Struct/enum/union type support
    """

    def __init__(self) -> None:
        """Initialize the Zig type system."""
        self._builtin_types = self._build_builtin_types()
        self._builtin_functions = self._build_builtin_functions()

    @property
    def name(self) -> str:
        return "zig"

    @property
    def capabilities(self) -> TypeSystemCapabilities:
        return TypeSystemCapabilities(
            supports_generics=True,
            supports_union_types=True,
            supports_optional_types=True,
            supports_type_inference=True,
            supports_protocols=False,
            supports_variance=False,
            supports_overloading=False,
            supports_ownership=False,
            supports_comptime=True,
            supports_error_unions=True,
            supports_lifetime_bounds=False,
            supports_sentinels=True,
            supports_allocators=True,
        )

    def _build_builtin_types(self) -> Dict[str, Type]:
        """Build the dictionary of Zig builtin types."""
        return {
            # Signed integers
            "i8": ZIG_I8,
            "i16": ZIG_I16,
            "i32": ZIG_I32,
            "i64": ZIG_I64,
            "i128": ZIG_I128,
            "isize": ZIG_ISIZE,
            # Unsigned integers
            "u8": ZIG_U8,
            "u16": ZIG_U16,
            "u32": ZIG_U32,
            "u64": ZIG_U64,
            "u128": ZIG_U128,
            "usize": ZIG_USIZE,
            # Floats
            "f16": ZIG_F16,
            "f32": ZIG_F32,
            "f64": ZIG_F64,
            "f80": ZIG_F80,
            "f128": ZIG_F128,
            # Special
            "bool": ZIG_BOOL,
            "void": ZIG_VOID,
            "noreturn": ZIG_NORETURN,
            "type": ZIG_TYPE,
            "anytype": ZIG_ANYTYPE,
            "anyopaque": ZIG_ANYOPAQUE,
            "anyframe": ZIG_ANYFRAME,
            "anyerror": ANYERROR,
            # Comptime
            "comptime_int": ZIG_COMPTIME_INT,
            "comptime_float": ZIG_COMPTIME_FLOAT,
            # Special values
            "undefined": ZIG_UNDEFINED,
            "null": ZIG_NULL,
            # C types
            "c_char": ZIG_C_CHAR,
            "c_short": ZIG_C_SHORT,
            "c_ushort": ZIG_C_USHORT,
            "c_int": ZIG_C_INT,
            "c_uint": ZIG_C_UINT,
            "c_long": ZIG_C_LONG,
            "c_ulong": ZIG_C_ULONG,
            "c_longlong": ZIG_C_LONGLONG,
            "c_ulonglong": ZIG_C_ULONGLONG,
            "c_longdouble": ZIG_C_LONGDOUBLE,
        }

    def _build_builtin_functions(self) -> Dict[str, FunctionType]:
        """Build dictionary of Zig builtin function signatures."""
        # Zig builtins use @name syntax and are handled specially
        # These are simplified signatures for type checking
        return {
            # Type introspection
            "@TypeOf": ZigFunctionType((ANY,), ZIG_TYPE),
            "@typeInfo": ZigFunctionType((ZIG_TYPE,), ZigStructType(name="std.builtin.Type")),
            "@typeName": ZigFunctionType((ZIG_TYPE,), ZigSliceType(ZIG_U8, is_const=True)),
            "@Type": ZigFunctionType((ZigStructType(name="std.builtin.Type"),), ZIG_TYPE),
            # Size/alignment
            "@sizeOf": ZigFunctionType((ZIG_TYPE,), ZIG_COMPTIME_INT),
            "@alignOf": ZigFunctionType((ZIG_TYPE,), ZIG_COMPTIME_INT),
            "@bitSizeOf": ZigFunctionType((ZIG_TYPE,), ZIG_COMPTIME_INT),
            "@offsetOf": ZigFunctionType((ZIG_TYPE, ZigSliceType(ZIG_U8, is_const=True)), ZIG_COMPTIME_INT),
            # Casts
            "@as": ZigFunctionType((ZIG_TYPE, ANY), ANY),
            "@intCast": ZigFunctionType((ANY,), ANY),
            "@floatCast": ZigFunctionType((ANY,), ANY),
            "@ptrCast": ZigFunctionType((ANY,), ANY),
            "@alignCast": ZigFunctionType((ANY,), ANY),
            "@enumFromInt": ZigFunctionType((ANY,), ANY),
            "@intFromEnum": ZigFunctionType((ANY,), ZIG_COMPTIME_INT),
            "@intFromPtr": ZigFunctionType((ANY,), ZIG_USIZE),
            "@ptrFromInt": ZigFunctionType((ZIG_USIZE,), ANY),
            "@intFromBool": ZigFunctionType((ZIG_BOOL,), ZIG_U1),
            "@intFromFloat": ZigFunctionType((ANY,), ANY),
            "@floatFromInt": ZigFunctionType((ANY,), ANY),
            "@truncate": ZigFunctionType((ANY,), ANY),
            "@bitCast": ZigFunctionType((ANY,), ANY),
            "@errorCast": ZigFunctionType((ANY,), ANY),
            # Memory
            "@memcpy": ZigFunctionType((ANY, ANY), ZIG_VOID),
            "@memset": ZigFunctionType((ANY, ANY), ZIG_VOID),
            # Math
            "@min": ZigFunctionType((ANY, ANY), ANY),
            "@max": ZigFunctionType((ANY, ANY), ANY),
            "@clz": ZigFunctionType((ANY,), ANY),
            "@ctz": ZigFunctionType((ANY,), ANY),
            "@popCount": ZigFunctionType((ANY,), ANY),
            "@byteSwap": ZigFunctionType((ANY,), ANY),
            "@bitReverse": ZigFunctionType((ANY,), ANY),
            "@addWithOverflow": ZigFunctionType((ANY, ANY), TupleType((ANY, ZIG_BOOL))),
            "@subWithOverflow": ZigFunctionType((ANY, ANY), TupleType((ANY, ZIG_BOOL))),
            "@mulWithOverflow": ZigFunctionType((ANY, ANY), TupleType((ANY, ZIG_BOOL))),
            "@shlWithOverflow": ZigFunctionType((ANY, ANY), TupleType((ANY, ZIG_BOOL))),
            # Errors
            "@errorName": ZigFunctionType((ANYERROR,), ZigSliceType(ZIG_U8, is_const=True)),
            "@errorFromInt": ZigFunctionType((ZIG_U16,), ANYERROR),
            "@intFromError": ZigFunctionType((ANYERROR,), ZIG_U16),
            # Compilation
            "@compileError": ZigFunctionType((ZigSliceType(ZIG_U8, is_const=True),), ZIG_NORETURN),
            "@compileLog": ZigFunctionType((ANY,), ZIG_VOID),
            # Debug
            "@panic": ZigFunctionType((ZigSliceType(ZIG_U8, is_const=True),), ZIG_NORETURN),
            "@breakpoint": ZigFunctionType((), ZIG_VOID),
            # Import/embed
            "@import": ZigFunctionType((ZigSliceType(ZIG_U8, is_const=True),), ZIG_TYPE),
            "@embedFile": ZigFunctionType((ZigSliceType(ZIG_U8, is_const=True),), ZigSliceType(ZIG_U8, is_const=True)),
            # SIMD
            "@Vector": ZigFunctionType((ZIG_COMPTIME_INT, ZIG_TYPE), ZIG_TYPE),
            "@splat": ZigFunctionType((ANY,), ANY),
            "@reduce": ZigFunctionType((ANY, ANY), ANY),
            "@shuffle": ZigFunctionType((ANY, ANY, ANY, ANY), ANY),
            "@select": ZigFunctionType((ANY, ANY, ANY, ANY), ANY),
            # Misc
            "@src": ZigFunctionType((), ZigStructType(name="std.builtin.SourceLocation")),
            "@This": ZigFunctionType((), ZIG_TYPE),
            "@hasDecl": ZigFunctionType((ZIG_TYPE, ZigSliceType(ZIG_U8, is_const=True)), ZIG_BOOL),
            "@hasField": ZigFunctionType((ZIG_TYPE, ZigSliceType(ZIG_U8, is_const=True)), ZIG_BOOL),
            "@field": ZigFunctionType((ANY, ZigSliceType(ZIG_U8, is_const=True)), ANY),
            "@call": ZigFunctionType((ANY, ANY, ANY), ANY),
            "@tagName": ZigFunctionType((ANY,), ZigSliceType(ZIG_U8, is_const=True)),
        }

    def parse_type_annotation(self, annotation: str) -> Type:
        """Parse a Zig type annotation string.

        Supports:
        - Primitives: i32, u8, f64, bool, void, etc.
        - Pointers: *T, *const T, *volatile T
        - Many-pointers: [*]T, [*:0]T, [*c]T
        - Slices: []T, []const T, [:0]T
        - Arrays: [N]T, [N:0]T
        - Optionals: ?T
        - Error unions: E!T, anyerror!T
        - Functions: fn(T) R
        - Vectors: @Vector(N, T)

        Args:
            annotation: The type annotation string

        Returns:
            The parsed Type

        Raises:
            TypeParseError: If parsing fails
        """
        annotation = annotation.strip()

        if not annotation:
            raise TypeParseError(annotation, "Empty annotation")

        return self._parse_type(annotation)

    def _parse_type(self, s: str) -> Type:
        """Parse a type string recursively."""
        s = s.strip()

        # Check for function type first: fn(...) T
        # This must come before error union check because fn(T) E!T
        # should parse as fn returning E!T, not as error union
        if s.startswith("fn(") or s.startswith("fn ("):
            return self._parse_function_type(s)

        # Check for error union: E!T
        if "!" in s and not s.startswith("@"):
            excl_pos = self._find_error_union_split(s)
            if excl_pos is not None:
                error_part = s[:excl_pos].strip()
                payload_part = s[excl_pos + 1:].strip()
                error_type = self._parse_type(error_part) if error_part else ANYERROR
                payload_type = self._parse_type(payload_part)
                return ZigErrorUnionType(error_set=error_type, payload=payload_type)

        # Check for optional: ?T
        if s.startswith("?"):
            inner = self._parse_type(s[1:])
            return ZigOptionalType(inner=inner)

        # Check for single pointer: *T, *const T, *volatile T
        if s.startswith("*") and not s.startswith("[*"):
            return self._parse_single_pointer(s[1:])

        # Check for slice: []T, []const T, [:0]T
        if s.startswith("[]"):
            return self._parse_slice(s[2:])

        # Check for many-pointer or array: [*]T, [*:0]T, [*c]T, [N]T
        if s.startswith("["):
            return self._parse_bracket_type(s)

        # Check for builtins: @Vector, @TypeOf, etc.
        if s.startswith("@"):
            return self._parse_builtin_type(s)

        # Check for struct/enum/union literals
        if s.startswith("struct{") or s.startswith("packed struct{") or s.startswith("extern struct{"):
            return self._parse_struct_literal(s)
        if s.startswith("enum{") or s.startswith("enum("):
            return self._parse_enum_literal(s)
        if s.startswith("union{") or s.startswith("union("):
            return self._parse_union_literal(s)

        # Check for error set: error{...}
        if s.startswith("error{"):
            return self._parse_error_set(s)

        # Check builtins
        if s in self._builtin_types:
            return self._builtin_types[s]

        # Check for arbitrary-width integers: u3, i17, etc.
        int_match = re.match(r"^([iu])(\d+)$", s)
        if int_match:
            sign = int_match.group(1)
            bits = int(int_match.group(2))
            if bits > 0 and bits <= 65535:  # Zig supports up to u65535
                return PrimitiveType(s)

        # Assume it's a named type (struct, enum, etc.)
        if self._is_valid_identifier(s):
            return ZigStructType(name=s)

        raise TypeParseError(s, f"Cannot parse type: {s}")

    def _parse_single_pointer(self, rest: str) -> ZigPointerType:
        """Parse *T, *const T, *volatile T, etc."""
        is_const = False
        is_volatile = False
        is_allowzero = False
        alignment = None
        address_space = None

        # Parse qualifiers
        while rest:
            rest = rest.strip()
            if rest.startswith("allowzero "):
                is_allowzero = True
                rest = rest[10:]
            elif rest.startswith("align("):
                end = rest.index(")")
                alignment = int(rest[6:end])
                rest = rest[end + 1:].strip()
            elif rest.startswith("addrspace("):
                end = rest.index(")")
                address_space = rest[10:end].strip(".")
                rest = rest[end + 1:].strip()
            elif rest.startswith("const "):
                is_const = True
                rest = rest[6:]
            elif rest.startswith("volatile "):
                is_volatile = True
                rest = rest[9:]
            else:
                break

        pointee = self._parse_type(rest)
        return ZigPointerType(
            pointee=pointee,
            is_const=is_const,
            is_volatile=is_volatile,
            is_allowzero=is_allowzero,
            alignment=alignment,
            address_space=address_space,
        )

    def _parse_slice(self, rest: str) -> ZigSliceType:
        """Parse []T, []const T, [:0]T."""
        is_const = False
        sentinel = None

        rest = rest.strip()

        # Check for sentinel: starts with :
        if rest.startswith(":"):
            # Find end of sentinel
            sentinel_end = 0
            for i, c in enumerate(rest[1:], 1):
                if c == "]":
                    sentinel_end = i
                    break
            if sentinel_end > 0:
                sentinel = rest[1:sentinel_end]
                rest = rest[sentinel_end + 1:].strip()

        # Check for const
        if rest.startswith("const "):
            is_const = True
            rest = rest[6:]

        element = self._parse_type(rest)
        return ZigSliceType(element=element, is_const=is_const, sentinel=sentinel)

    def _parse_bracket_type(self, s: str) -> Type:
        """Parse [*]T, [*:0]T, [*c]T, [N]T, [N:0]T."""
        # Find matching ]
        try:
            bracket_end = s.index("]")
        except ValueError:
            raise TypeParseError(s, "Unclosed bracket in type")
        inside = s[1:bracket_end]
        rest = s[bracket_end + 1:].strip()

        # Many-pointer: [*]T
        if inside == "*":
            is_const = rest.startswith("const ")
            if is_const:
                rest = rest[6:]
            element = self._parse_type(rest)
            return ZigManyPointerType(element=element, is_const=is_const)

        # Many-pointer with sentinel: [*:0]T
        if inside.startswith("*:"):
            sentinel = inside[2:]
            is_const = rest.startswith("const ")
            if is_const:
                rest = rest[6:]
            element = self._parse_type(rest)
            return ZigManyPointerType(element=element, is_const=is_const, sentinel=sentinel)

        # C pointer: [*c]T
        if inside == "*c":
            is_const = rest.startswith("const ")
            if is_const:
                rest = rest[6:]
            element = self._parse_type(rest)
            return ZigCPointerType(element=element, is_const=is_const)

        # Slice with sentinel: [:0]T (already handled by _parse_slice if starts with [])
        if inside.startswith(":"):
            sentinel = inside[1:]
            is_const = rest.startswith("const ")
            if is_const:
                rest = rest[6:]
            element = self._parse_type(rest)
            return ZigSliceType(element=element, is_const=is_const, sentinel=sentinel)

        # Array: [N]T or [N:sentinel]T
        if ":" in inside:
            parts = inside.split(":", 1)
            length_str = parts[0].strip()
            sentinel = parts[1].strip()
            length = int(length_str) if length_str.isdigit() else None
            element = self._parse_type(rest)
            return ZigArrayType(element=element, length=length, sentinel=sentinel)
        else:
            length = int(inside) if inside.isdigit() else None
            element = self._parse_type(rest)
            return ZigArrayType(element=element, length=length)

    def _parse_function_type(self, s: str) -> ZigFunctionType:
        """Parse fn(params) return_type."""
        # Find opening paren
        paren_start = s.index("(")
        paren_end = self._find_matching_paren(s, paren_start)

        params_str = s[paren_start + 1:paren_end].strip()
        rest = s[paren_end + 1:].strip()

        # Parse parameters
        params: List[Type] = []
        if params_str:
            param_strs = self._split_params(params_str)
            for p in param_strs:
                p = p.strip()
                # Handle comptime and other qualifiers
                if p.startswith("comptime "):
                    p = p[9:]
                if ":" in p:
                    # Named parameter: name: Type
                    p = p.split(":", 1)[1].strip()
                if p:
                    params.append(self._parse_type(p))

        # Parse return type
        # Handle calling convention
        calling_convention = None
        if rest.startswith("callconv("):
            cc_end = rest.index(")")
            calling_convention = rest[9:cc_end].strip(".")
            rest = rest[cc_end + 1:].strip()

        return_type = self._parse_type(rest) if rest else ZIG_VOID

        return ZigFunctionType(
            params=tuple(params),
            return_type=return_type,
            calling_convention=calling_convention,
        )

    def _parse_builtin_type(self, s: str) -> Type:
        """Parse @Vector, @TypeOf result, etc."""
        if s.startswith("@Vector("):
            # @Vector(N, T)
            end = self._find_matching_paren(s, 7)
            args = s[8:end].split(",", 1)
            if len(args) == 2:
                length = int(args[0].strip())
                element = self._parse_type(args[1].strip())
                return ZigVectorType(length=length, element=element)

        # Most @builtins return type, treat as generic
        return ZIG_TYPE

    def _parse_struct_literal(self, s: str) -> ZigStructType:
        """Parse struct{...} literal with field extraction.

        Parses struct definitions like:
        - struct { x: i32, y: i32 }
        - packed struct { flags: u8, data: u32 }
        - extern struct { handle: *anyopaque }
        """
        is_packed = "packed" in s[:20]
        is_extern = "extern" in s[:20]

        # Find the content between braces
        brace_start = s.find("{")
        brace_end = s.rfind("}")

        if brace_start == -1 or brace_end == -1:
            return ZigStructType(is_packed=is_packed, is_extern=is_extern)

        content = s[brace_start + 1:brace_end].strip()

        if not content:
            return ZigStructType(is_packed=is_packed, is_extern=is_extern)

        # Parse fields
        fields = self._parse_struct_fields(content)

        return ZigStructType(
            fields=fields,
            is_packed=is_packed,
            is_extern=is_extern,
        )

    def _parse_struct_fields(self, content: str) -> Tuple[Tuple[str, Type], ...]:
        """Parse struct field definitions.

        Handles field declarations like:
        - name: Type
        - name: Type = default_value
        - comptime name: Type
        """
        fields: List[Tuple[str, Type]] = []

        # Split by comma, respecting nesting
        field_strs = self._split_fields(content)

        for field_str in field_strs:
            field_str = field_str.strip()
            if not field_str:
                continue

            # Skip functions and decls
            if field_str.startswith("fn ") or field_str.startswith("const "):
                continue

            # Remove comptime prefix if present
            if field_str.startswith("comptime "):
                field_str = field_str[9:].strip()

            # Find the colon separating name from type
            colon_idx = field_str.find(":")
            if colon_idx == -1:
                continue

            field_name = field_str[:colon_idx].strip()

            # Handle default values: name: Type = value
            type_part = field_str[colon_idx + 1:].strip()
            eq_idx = self._find_default_value_eq(type_part)
            if eq_idx != -1:
                type_part = type_part[:eq_idx].strip()

            # Skip invalid field names
            if not field_name or not self._is_valid_identifier(field_name):
                continue

            try:
                field_type = self._parse_type(type_part)
                fields.append((field_name, field_type))
            except (TypeParseError, ValueError):
                # Skip unparseable fields
                continue

        return tuple(fields)

    def _find_default_value_eq(self, s: str) -> int:
        """Find = for default value, not inside nested expressions."""
        depth = 0
        for i, c in enumerate(s):
            if c in "([{":
                depth += 1
            elif c in ")]}":
                depth -= 1
            elif c == "=" and depth == 0:
                return i
        return -1

    def _split_fields(self, content: str) -> List[str]:
        """Split struct/union/enum content by commas, respecting nesting."""
        fields: List[str] = []
        current: List[str] = []
        depth = 0

        for c in content:
            if c in "([{":
                depth += 1
                current.append(c)
            elif c in ")]}":
                depth -= 1
                current.append(c)
            elif c == "," and depth == 0:
                fields.append("".join(current))
                current = []
            else:
                current.append(c)

        if current:
            fields.append("".join(current))

        return fields

    def _parse_enum_literal(self, s: str) -> ZigEnumType:
        """Parse enum{...} or enum(tag){...} with variant extraction.

        Parses enum definitions like:
        - enum { foo, bar, baz }
        - enum(u8) { a, b, c }
        """
        is_extern = "extern" in s[:20]

        # Check for tag type: enum(tag_type)
        tag_type: Optional[Type] = None
        tag_start = s.find("(")
        brace_start = s.find("{")

        if tag_start != -1 and tag_start < brace_start:
            tag_end = s.find(")", tag_start)
            if tag_end != -1:
                tag_str = s[tag_start + 1:tag_end].strip()
                try:
                    tag_type = self._parse_type(tag_str)
                except (TypeParseError, ValueError):
                    pass

        # Find the content between braces
        brace_end = s.rfind("}")

        if brace_start == -1 or brace_end == -1:
            return ZigEnumType(tag_type=tag_type, is_extern=is_extern)

        content = s[brace_start + 1:brace_end].strip()

        if not content:
            return ZigEnumType(tag_type=tag_type, is_extern=is_extern)

        # Parse variants
        variants = self._parse_enum_variants(content)

        return ZigEnumType(
            tag_type=tag_type,
            variants=variants,
            is_extern=is_extern,
        )

    def _parse_enum_variants(self, content: str) -> FrozenSet[str]:
        """Parse enum variant names.

        Handles variants like:
        - simple_name
        - name = value
        """
        variants: Set[str] = set()

        variant_strs = self._split_fields(content)

        for variant_str in variant_strs:
            variant_str = variant_str.strip()
            if not variant_str:
                continue

            # Skip functions
            if variant_str.startswith("fn "):
                continue

            # Handle value assignment: name = value
            eq_idx = variant_str.find("=")
            if eq_idx != -1:
                variant_str = variant_str[:eq_idx].strip()

            # Extract variant name
            if self._is_valid_identifier(variant_str):
                variants.add(variant_str)

        return frozenset(variants)

    def _parse_union_literal(self, s: str) -> ZigUnionType:
        """Parse union{...} or union(tag){...} with variant extraction.

        Parses union definitions like:
        - union { value: i32, ptr: *u8 }
        - union(enum) { a: i32, b: f64 }
        """
        is_packed = "packed" in s[:20]
        is_extern = "extern" in s[:20]

        # Check for tag type: union(tag_type)
        tag_type: Optional[Type] = None
        tag_start = s.find("(")
        brace_start = s.find("{")

        if tag_start != -1 and tag_start < brace_start:
            tag_end = s.find(")", tag_start)
            if tag_end != -1:
                tag_str = s[tag_start + 1:tag_end].strip()
                if tag_str == "enum":
                    # union(enum) - auto-generated tag
                    tag_type = ZigEnumType()
                else:
                    try:
                        tag_type = self._parse_type(tag_str)
                    except (TypeParseError, ValueError):
                        pass

        # Find the content between braces
        brace_end = s.rfind("}")

        if brace_start == -1 or brace_end == -1:
            return ZigUnionType(tag_type=tag_type, is_packed=is_packed, is_extern=is_extern)

        content = s[brace_start + 1:brace_end].strip()

        if not content:
            return ZigUnionType(tag_type=tag_type, is_packed=is_packed, is_extern=is_extern)

        # Parse variants (same format as struct fields for typed unions)
        variants = self._parse_union_variants(content)

        return ZigUnionType(
            tag_type=tag_type,
            variants=variants,
            is_packed=is_packed,
            is_extern=is_extern,
        )

    def _parse_union_variants(self, content: str) -> Tuple[Tuple[str, Type], ...]:
        """Parse union variant definitions.

        Handles variants like:
        - name: Type
        - name: void (for enum-like variants)
        """
        variants: List[Tuple[str, Type]] = []

        variant_strs = self._split_fields(content)

        for variant_str in variant_strs:
            variant_str = variant_str.strip()
            if not variant_str:
                continue

            # Skip functions
            if variant_str.startswith("fn "):
                continue

            # Find the colon separating name from type
            colon_idx = variant_str.find(":")
            if colon_idx == -1:
                # Bare name - treat as void type
                if self._is_valid_identifier(variant_str):
                    variants.append((variant_str, ZIG_VOID))
                continue

            variant_name = variant_str[:colon_idx].strip()
            type_part = variant_str[colon_idx + 1:].strip()

            if not variant_name or not self._is_valid_identifier(variant_name):
                continue

            try:
                variant_type = self._parse_type(type_part)
                variants.append((variant_name, variant_type))
            except (TypeParseError, ValueError):
                # Treat as void if unparseable
                variants.append((variant_name, ZIG_VOID))

        return tuple(variants)

    def _parse_error_set(self, s: str) -> ErrorSetType:
        """Parse error{A, B, C}."""
        if s == "anyerror":
            return ANYERROR

        # Extract error names
        start = s.index("{")
        end = s.rindex("}")
        content = s[start + 1:end].strip()

        if not content:
            return ErrorSetType(error_names=frozenset())

        names = frozenset(n.strip() for n in content.split(",") if n.strip())
        return ErrorSetType(error_names=names)

    def _find_error_union_split(self, s: str) -> Optional[int]:
        """Find the ! that separates error type from payload in E!T."""
        depth = 0
        for i, c in enumerate(s):
            if c in "([{":
                depth += 1
            elif c in ")]}":
                depth -= 1
            elif c == "!" and depth == 0:
                return i
        return None

    def _find_matching_paren(self, s: str, start: int) -> int:
        """Find the matching closing parenthesis."""
        depth = 1
        for i in range(start + 1, len(s)):
            if s[i] == "(":
                depth += 1
            elif s[i] == ")":
                depth -= 1
                if depth == 0:
                    return i
        raise TypeParseError(s, "Unmatched parenthesis")

    def _split_params(self, s: str) -> List[str]:
        """Split function parameters by comma, respecting nesting."""
        params = []
        current = []
        depth = 0

        for c in s:
            if c in "([{":
                depth += 1
                current.append(c)
            elif c in ")]}":
                depth -= 1
                current.append(c)
            elif c == "," and depth == 0:
                params.append("".join(current))
                current = []
            else:
                current.append(c)

        if current:
            params.append("".join(current))

        return params

    def _is_valid_identifier(self, s: str) -> bool:
        """Check if s is a valid Zig identifier."""
        if not s:
            return False
        if s[0].isdigit():
            return False
        return all(c.isalnum() or c == "_" or c == "." for c in s)

    def infer_literal_type(self, literal: LiteralInfo) -> Type:
        """Infer the type of a Zig literal.

        Key difference from other languages: integer literals have
        type comptime_int, not i32. Float literals have comptime_float.
        """
        if literal.kind == LiteralKind.INTEGER:
            return ZIG_COMPTIME_INT
        elif literal.kind == LiteralKind.FLOAT:
            return ZIG_COMPTIME_FLOAT
        elif literal.kind == LiteralKind.STRING:
            # String literals are *const [N:0]u8
            if literal.text:
                length = len(literal.text) - 2  # Remove quotes
                return ZigPointerType(
                    pointee=ZigArrayType(element=ZIG_U8, length=length, sentinel=0),
                    is_const=True,
                )
            return ZigSliceType(element=ZIG_U8, is_const=True)
        elif literal.kind == LiteralKind.BOOLEAN:
            return ZIG_BOOL
        elif literal.kind == LiteralKind.NONE:
            return ZIG_NULL
        else:
            return ANY

    def get_builtin_types(self) -> Dict[str, Type]:
        """Return Zig builtin types."""
        return self._builtin_types.copy()

    def get_builtin_functions(self) -> Dict[str, FunctionType]:
        """Return Zig builtin function signatures."""
        return self._builtin_functions.copy()

    def check_assignable(self, source: Type, target: Type) -> bool:
        """Check if source type is assignable to target type.

        Zig assignability rules include:
        - comptime_int assignable to any integer (if value fits)
        - comptime_float assignable to any float
        - Pointer coercion rules
        - Error union coercion
        - Optional coercion (T -> ?T)
        - anytype accepts anything
        """
        # Handle special types
        if isinstance(target, AnyType) or isinstance(source, AnyType):
            return True

        if isinstance(source, NeverType):
            return True

        if isinstance(target, NeverType):
            return False

        # Handle holes
        if isinstance(source, HoleType) or isinstance(target, HoleType):
            return True

        # Same type
        if source == target:
            return True

        # anytype accepts anything (in both directions for generic context)
        if target == ZIG_ANYTYPE or source == ZIG_ANYTYPE:
            return True

        # noreturn is assignable to anything (bottom type)
        if source == ZIG_NORETURN:
            return True

        # void is only assignable to void
        if target == ZIG_VOID:
            return source == ZIG_VOID

        # Comptime integer coercion
        if source == ZIG_COMPTIME_INT:
            return self._is_integer_type(target)

        # Comptime float coercion
        if source == ZIG_COMPTIME_FLOAT:
            return self._is_float_type(target)

        # undefined is assignable to anything
        if source == ZIG_UNDEFINED:
            return True

        # null is assignable to optional types
        if source == ZIG_NULL:
            return isinstance(target, ZigOptionalType)

        # T is assignable to ?T
        if isinstance(target, ZigOptionalType):
            return self.check_assignable(source, target.inner)

        # Error union coercion: T -> E!T
        if isinstance(target, ZigErrorUnionType):
            # Payload type matches
            if self.check_assignable(source, target.payload):
                return True
            # Error type matches
            if isinstance(source, ErrorSetType):
                return self._check_error_set_assignable(source, target.error_set)

        # Pointer coercion
        if isinstance(source, (ZigPointerType, ZigManyPointerType, ZigCPointerType)):
            if isinstance(target, (ZigPointerType, ZigManyPointerType, ZigCPointerType)):
                return self._check_pointer_coercion(source, target)

        # Slice coercion
        if isinstance(source, ZigSliceType) and isinstance(target, ZigSliceType):
            if source.is_const or not target.is_const:
                return self.check_assignable(source.element, target.element)

        # Array to slice coercion
        if isinstance(source, ZigArrayType) and isinstance(target, ZigSliceType):
            return self.check_assignable(source.element, target.element)

        # Pointer to slice coercion: *[N]T -> []T
        if isinstance(source, ZigPointerType) and isinstance(target, ZigSliceType):
            if isinstance(source.pointee, ZigArrayType):
                return self.check_assignable(source.pointee.element, target.element)

        # Error set coercion
        if isinstance(source, ErrorSetType) and isinstance(target, ErrorSetType):
            return self._check_error_set_assignable(source, target)

        # Struct/enum/union assignability
        if isinstance(source, ZigStructType) and isinstance(target, ZigStructType):
            return self._check_struct_assignable(source, target)

        if isinstance(source, ZigEnumType) and isinstance(target, ZigEnumType):
            return self._check_enum_assignable(source, target)

        if isinstance(source, ZigUnionType) and isinstance(target, ZigUnionType):
            return self._check_union_assignable(source, target)

        # Vector type assignability
        if isinstance(source, ZigVectorType) and isinstance(target, ZigVectorType):
            return self._check_vector_assignable(source, target)

        # Function type compatibility
        if isinstance(source, ZigFunctionType) and isinstance(target, ZigFunctionType):
            return self._check_function_assignable(source, target)

        return False

    def _check_struct_assignable(self, source: ZigStructType, target: ZigStructType) -> bool:
        """Check struct type assignability.

        Zig uses nominal typing for named structs but structural typing
        for anonymous structs. Rules:
        - Named structs: must have same name
        - Anonymous structs: structural compatibility (all target fields present in source)
        """
        # Named structs use nominal typing
        if source.name and target.name:
            return source.name == target.name

        # Named source to anonymous target - check structural compatibility
        if target.fields and source.fields:
            return self._check_struct_fields_compatible(source.fields, target.fields)

        # Anonymous to named - not allowed without explicit cast
        if not source.name and target.name:
            return False

        # Named source to empty anonymous target - allowed
        if source.name and not target.fields:
            return True

        # Both anonymous - structural compatibility
        if not source.name and not target.name:
            if target.fields:
                return self._check_struct_fields_compatible(source.fields, target.fields)
            return True  # Empty target accepts anything

        return False

    def _check_struct_fields_compatible(
        self,
        source_fields: Tuple[Tuple[str, Type], ...],
        target_fields: Tuple[Tuple[str, Type], ...]
    ) -> bool:
        """Check if source struct has all required target fields with compatible types.

        Structural subtyping: source can have extra fields, but must have all
        target fields with assignable types.
        """
        source_field_map = {name: typ for name, typ in source_fields}

        for field_name, target_type in target_fields:
            if field_name not in source_field_map:
                return False
            if not self.check_assignable(source_field_map[field_name], target_type):
                return False

        return True

    def _check_enum_assignable(self, source: ZigEnumType, target: ZigEnumType) -> bool:
        """Check enum type assignability.

        Zig uses nominal typing for named enums.
        - Named enums: must have same name
        - Anonymous enums: structural compatibility (source variants subset of target)
        """
        # Named enums use nominal typing
        if source.name and target.name:
            return source.name == target.name

        # Check tag type compatibility
        if source.tag_type and target.tag_type:
            if not self.check_assignable(source.tag_type, target.tag_type):
                return False

        # Anonymous enum assignability - source variants must be subset
        if source.variants and target.variants:
            return source.variants.issubset(target.variants)

        # Empty variants treated as compatible
        return True

    def _check_union_assignable(self, source: ZigUnionType, target: ZigUnionType) -> bool:
        """Check union type assignability.

        Zig uses nominal typing for named unions.
        - Tagged unions: source variants must be subset with compatible payload types
        - Bare unions: less strict, based on memory layout
        """
        # Named unions use nominal typing
        if source.name and target.name:
            return source.name == target.name

        # Check tag type compatibility for tagged unions
        if source.tag_type and target.tag_type:
            if not self.check_assignable(source.tag_type, target.tag_type):
                return False

        # Check variant compatibility
        if source.variants and target.variants:
            return self._check_union_variants_compatible(source.variants, target.variants)

        return True

    def _check_union_variants_compatible(
        self,
        source_variants: Tuple[Tuple[str, Type], ...],
        target_variants: Tuple[Tuple[str, Type], ...]
    ) -> bool:
        """Check if source union variants are compatible with target.

        For tagged unions, source must have all target variants with
        assignable payload types.
        """
        source_variant_map = {name: typ for name, typ in source_variants}
        target_variant_map = {name: typ for name, typ in target_variants}

        # Source must have at least the target's variants
        for variant_name, target_type in target_variant_map.items():
            if variant_name not in source_variant_map:
                return False
            if not self.check_assignable(source_variant_map[variant_name], target_type):
                return False

        return True

    def _check_vector_assignable(self, source: ZigVectorType, target: ZigVectorType) -> bool:
        """Check SIMD vector type assignability.

        Vectors must have same length and compatible element types.
        """
        if source.length != target.length:
            return False
        return self.check_assignable(source.element, target.element)

    def get_struct_field_type(self, struct_type: ZigStructType, field_name: str) -> Optional[Type]:
        """Get the type of a struct field by name.

        Useful for field access type checking.
        """
        for name, typ in struct_type.fields:
            if name == field_name:
                return typ
        return None

    def get_union_variant_type(self, union_type: ZigUnionType, variant_name: str) -> Optional[Type]:
        """Get the payload type of a union variant by name.

        Useful for switch/payload capture type checking.
        """
        for name, typ in union_type.variants:
            if name == variant_name:
                return typ
        return None

    def has_enum_variant(self, enum_type: ZigEnumType, variant_name: str) -> bool:
        """Check if an enum has a specific variant."""
        return variant_name in enum_type.variants

    def _is_integer_type(self, typ: Type) -> bool:
        """Check if type is an integer type."""
        if isinstance(typ, PrimitiveType):
            name = typ.name
            if name in ("i8", "i16", "i32", "i64", "i128", "isize",
                       "u8", "u16", "u32", "u64", "u128", "usize",
                       "c_char", "c_short", "c_ushort", "c_int", "c_uint",
                       "c_long", "c_ulong", "c_longlong", "c_ulonglong",
                       "comptime_int"):
                return True
            # Arbitrary width integers: u3, i17, etc.
            if re.match(r"^[iu]\d+$", name):
                return True
        return False

    def _is_float_type(self, typ: Type) -> bool:
        """Check if type is a float type."""
        if isinstance(typ, PrimitiveType):
            return typ.name in ("f16", "f32", "f64", "f80", "f128",
                               "c_longdouble", "comptime_float")
        return False

    def _check_pointer_coercion(self, source: Type, target: Type) -> bool:
        """Check pointer coercion rules."""
        # Get pointee types
        if isinstance(source, ZigPointerType):
            source_pointee = source.pointee
            source_const = source.is_const
        elif isinstance(source, ZigManyPointerType):
            source_pointee = source.element
            source_const = source.is_const
        elif isinstance(source, ZigCPointerType):
            source_pointee = source.element
            source_const = source.is_const
        else:
            return False

        if isinstance(target, ZigPointerType):
            target_pointee = target.pointee
            target_const = target.is_const
        elif isinstance(target, ZigManyPointerType):
            target_pointee = target.element
            target_const = target.is_const
        elif isinstance(target, ZigCPointerType):
            target_pointee = target.element
            target_const = target.is_const
        else:
            return False

        # Can't remove const
        if source_const and not target_const:
            return False

        # Pointee types must be compatible
        return self.check_assignable(source_pointee, target_pointee)

    def _check_error_set_assignable(self, source: ErrorSetType, target: Type) -> bool:
        """Check error set assignability."""
        if isinstance(target, ErrorSetType):
            # anyerror accepts any error set
            if target.is_anyerror:
                return True
            # Source must be subset of target
            if source.error_names and target.error_names:
                return source.error_names.issubset(target.error_names)
        return False

    def _check_function_assignable(self, source: ZigFunctionType, target: ZigFunctionType) -> bool:
        """Check function type compatibility."""
        if len(source.params) != len(target.params):
            return False

        # Contravariant in parameters
        for sp, tp in zip(source.params, target.params):
            if not self.check_assignable(tp, sp):
                return False

        # Covariant in return type
        return self.check_assignable(source.return_type, target.return_type)

    def format_type(self, typ: Type) -> str:
        """Format a type as Zig syntax."""
        if isinstance(typ, PrimitiveType):
            return typ.name

        if isinstance(typ, AnyType):
            return "anytype"

        if isinstance(typ, NeverType):
            return "noreturn"

        if isinstance(typ, HoleType):
            return f"@compileError(\"hole_{typ.hole_id}\")"

        if isinstance(typ, ZigOptionalType):
            return f"?{self.format_type(typ.inner)}"

        if isinstance(typ, ZigErrorUnionType):
            return f"{self.format_type(typ.error_set)}!{self.format_type(typ.payload)}"

        if isinstance(typ, ZigPointerType):
            parts = ["*"]
            if typ.is_allowzero:
                parts.append("allowzero ")
            if typ.alignment is not None:
                parts.append(f"align({typ.alignment}) ")
            if typ.is_const:
                parts.append("const ")
            if typ.is_volatile:
                parts.append("volatile ")
            parts.append(self.format_type(typ.pointee))
            return "".join(parts)

        if isinstance(typ, ZigSliceType):
            sentinel_str = f":{typ.sentinel}" if typ.sentinel is not None else ""
            const_str = "const " if typ.is_const else ""
            return f"[{sentinel_str}]{const_str}{self.format_type(typ.element)}"

        if isinstance(typ, ZigArrayType):
            len_str = str(typ.length) if typ.length is not None else "_"
            sentinel_str = f":{typ.sentinel}" if typ.sentinel is not None else ""
            return f"[{len_str}{sentinel_str}]{self.format_type(typ.element)}"

        if isinstance(typ, ZigManyPointerType):
            sentinel_str = f":{typ.sentinel}" if typ.sentinel is not None else ""
            const_str = "const " if typ.is_const else ""
            return f"[*{sentinel_str}]{const_str}{self.format_type(typ.element)}"

        if isinstance(typ, ZigCPointerType):
            const_str = "const " if typ.is_const else ""
            return f"[*c]{const_str}{self.format_type(typ.element)}"

        if isinstance(typ, ErrorSetType):
            if typ.is_anyerror:
                return "anyerror"
            if typ.error_names:
                names = ", ".join(sorted(typ.error_names))
                return f"error{{{names}}}"
            return "error{}"

        if isinstance(typ, ZigFunctionType):
            params = ", ".join(self.format_type(p) for p in typ.params)
            ret = self.format_type(typ.return_type)
            cc = f"callconv(.{typ.calling_convention}) " if typ.calling_convention else ""
            return f"fn({params}) {cc}{ret}"

        if isinstance(typ, ZigStructType):
            return typ.name or "struct{...}"

        if isinstance(typ, ZigEnumType):
            return typ.name or "enum{...}"

        if isinstance(typ, ZigUnionType):
            return typ.name or "union{...}"

        if isinstance(typ, ZigVectorType):
            return f"@Vector({typ.length}, {self.format_type(typ.element)})"

        return str(typ)

    def get_common_imports(self) -> Dict[str, Type]:
        """Return commonly imported types from std."""
        return {
            "std.mem.Allocator": ZigStructType(name="std.mem.Allocator"),
            "std.ArrayList": ZigStructType(name="std.ArrayList"),
            "std.AutoHashMap": ZigStructType(name="std.AutoHashMap"),
            "std.StringHashMap": ZigStructType(name="std.StringHashMap"),
            "std.BoundedArray": ZigStructType(name="std.BoundedArray"),
            "std.io.Reader": ZigStructType(name="std.io.Reader"),
            "std.io.Writer": ZigStructType(name="std.io.Writer"),
            "std.fs.File": ZigStructType(name="std.fs.File"),
            "std.fs.Dir": ZigStructType(name="std.fs.Dir"),
            "std.Thread": ZigStructType(name="std.Thread"),
            "std.Mutex": ZigStructType(name="std.Thread.Mutex"),
        }

    def normalize_type(self, typ: Type) -> Type:
        """Normalize Zig type to canonical form."""
        # Zig types are generally already in canonical form
        return typ


# Convenience alias
ZIG_U1 = PrimitiveType("u1")
