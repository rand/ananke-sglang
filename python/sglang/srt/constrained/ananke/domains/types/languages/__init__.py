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
"""Language-specific type systems for Python, Zig, Rust, TypeScript, Go, and more.

This package provides type system implementations for multiple programming
languages, enabling language-aware type checking during code generation.

Currently supported:
- Python (PEP 484 compatible, mypy/pyright semantics)
- Zig (with full comptime support)
- Rust (with ownership and lifetime support)
- TypeScript (with structural typing)
- Go (with interfaces and generics)

Usage:
    >>> from domains.types.languages import get_type_system, PythonTypeSystem
    >>>
    >>> # Get a type system by name
    >>> ts = get_type_system("python")
    >>> zig_ts = get_type_system("zig")
    >>> rust_ts = get_type_system("rust")
    >>>
    >>> # Parse type annotations
    >>> list_int = ts.parse_type_annotation("List[int]")
    >>> zig_opt = zig_ts.parse_type_annotation("?i32")
    >>> rust_ref = rust_ts.parse_type_annotation("&mut i32")
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

from domains.types.languages.zig import (
    ZigTypeSystem,
    # Primitives
    ZIG_I8, ZIG_I16, ZIG_I32, ZIG_I64, ZIG_I128, ZIG_ISIZE,
    ZIG_U8, ZIG_U16, ZIG_U32, ZIG_U64, ZIG_U128, ZIG_USIZE,
    ZIG_F16, ZIG_F32, ZIG_F64, ZIG_F80, ZIG_F128,
    ZIG_BOOL, ZIG_VOID, ZIG_NORETURN,
    ZIG_COMPTIME_INT, ZIG_COMPTIME_FLOAT,
    ZIG_TYPE, ZIG_ANYTYPE, ZIG_ANYOPAQUE,
    # Compound types
    ZigOptionalType,
    ZigErrorUnionType,
    ZigPointerType,
    ZigSliceType,
    ZigArrayType,
    ZigFunctionType,
    ZigVectorType,
)

from domains.types.languages.rust import (
    RustTypeSystem,
    # Primitives
    RUST_I8, RUST_I16, RUST_I32, RUST_I64, RUST_I128, RUST_ISIZE,
    RUST_U8, RUST_U16, RUST_U32, RUST_U64, RUST_U128, RUST_USIZE,
    RUST_F32, RUST_F64,
    RUST_BOOL, RUST_CHAR, RUST_STR,
    RUST_UNIT, RUST_NEVER,
    # Compound types
    RustReferenceType,
    RustSliceType,
    RustArrayType,
    RustOptionType,
    RustResultType,
    RustBoxType,
    RustRcType,
    RustVecType,
    RustStringType,
    RustFunctionType,
    RustDynTraitType,
    RustImplTraitType,
)

from domains.types.languages.typescript import (
    TypeScriptTypeSystem,
    # Primitives
    TS_STRING, TS_NUMBER, TS_BOOLEAN, TS_BIGINT, TS_SYMBOL,
    TS_UNDEFINED, TS_NULL, TS_VOID, TS_OBJECT, TS_UNKNOWN,
    TS_ANY, TS_NEVER,
    # Helper types
    TSParameter,
    # Compound types
    TSLiteralType,
    TSArrayType,
    TSTupleType,
    TSObjectType,
    TSFunctionType,
    TSTypeParameter,
    TSUnionType,
    TSIntersectionType,
    TSConditionalType,
    TSMappedType,
    TSIndexedAccessType,
    TSKeyofType,
    TSTypeofType,
    TSTemplateLiteralType,
    TSTypeReference,
    TSInferType,
)

from domains.types.languages.go import (
    GoTypeSystem,
    # Primitives
    GO_BOOL, GO_STRING,
    GO_INT, GO_INT8, GO_INT16, GO_INT32, GO_INT64,
    GO_UINT, GO_UINT8, GO_UINT16, GO_UINT32, GO_UINT64,
    GO_FLOAT32, GO_FLOAT64,
    GO_COMPLEX64, GO_COMPLEX128,
    GO_BYTE, GO_RUNE,
    GO_ANY, GO_ERROR,
    GO_UNTYPED_INT, GO_UNTYPED_FLOAT, GO_UNTYPED_STRING,
    # Compound types
    GoArrayType,
    GoSliceType,
    GoMapType,
    GoPointerType,
    GoChannelType,
    GoFunctionType,
    GoInterfaceType,
    GoStructType,
    GoGenericType,
    GoNamedType,
)

from domains.types.languages.kotlin import (
    KotlinTypeSystem,
    # Primitives
    KOTLIN_BYTE, KOTLIN_SHORT, KOTLIN_INT, KOTLIN_LONG,
    KOTLIN_FLOAT, KOTLIN_DOUBLE,
    KOTLIN_UBYTE, KOTLIN_USHORT, KOTLIN_UINT, KOTLIN_ULONG,
    KOTLIN_BOOLEAN, KOTLIN_CHAR, KOTLIN_STRING,
    KOTLIN_UNIT, KOTLIN_ANY, KOTLIN_NOTHING,
    # Compound types
    KotlinNullableType,
    KotlinArrayType,
    KotlinPrimitiveArrayType,
    KotlinListType,
    KotlinSetType,
    KotlinMapType,
    KotlinPairType,
    KotlinFunctionType,
    KotlinTypeParameter,
    KotlinGenericType,
    KotlinStarProjection,
    KotlinClassType,
    KotlinEnumType,
)

from domains.types.languages.swift import (
    SwiftTypeSystem,
    # Primitives
    SWIFT_INT, SWIFT_INT8, SWIFT_INT16, SWIFT_INT32, SWIFT_INT64,
    SWIFT_UINT, SWIFT_UINT8, SWIFT_UINT16, SWIFT_UINT32, SWIFT_UINT64,
    SWIFT_FLOAT, SWIFT_DOUBLE, SWIFT_FLOAT16, SWIFT_FLOAT80,
    SWIFT_BOOL, SWIFT_CHARACTER, SWIFT_STRING,
    SWIFT_VOID, SWIFT_NEVER, SWIFT_ANY, SWIFT_ANY_OBJECT,
    # Compound types
    SwiftOptionalType,
    SwiftImplicitlyUnwrappedOptionalType,
    SwiftArrayType,
    SwiftDictionaryType,
    SwiftSetType,
    SwiftTupleType,
    SwiftFunctionType,
    SwiftClosureType,
    SwiftProtocolType,
    SwiftProtocolCompositionType,
    SwiftGenericType,
    SwiftTypeParameter,
    SwiftMetatypeType,
    SwiftExistentialType,
    SwiftOpaqueType,
    SwiftResultType,
    SwiftNamedType,
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
    # Zig primitives
    "ZigTypeSystem",
    "ZIG_I8", "ZIG_I16", "ZIG_I32", "ZIG_I64", "ZIG_I128", "ZIG_ISIZE",
    "ZIG_U8", "ZIG_U16", "ZIG_U32", "ZIG_U64", "ZIG_U128", "ZIG_USIZE",
    "ZIG_F16", "ZIG_F32", "ZIG_F64", "ZIG_F80", "ZIG_F128",
    "ZIG_BOOL", "ZIG_VOID", "ZIG_NORETURN",
    "ZIG_COMPTIME_INT", "ZIG_COMPTIME_FLOAT",
    "ZIG_TYPE", "ZIG_ANYTYPE", "ZIG_ANYOPAQUE",
    # Zig compound types
    "ZigOptionalType",
    "ZigErrorUnionType",
    "ZigPointerType",
    "ZigSliceType",
    "ZigArrayType",
    "ZigFunctionType",
    "ZigVectorType",
    # Rust primitives
    "RustTypeSystem",
    "RUST_I8", "RUST_I16", "RUST_I32", "RUST_I64", "RUST_I128", "RUST_ISIZE",
    "RUST_U8", "RUST_U16", "RUST_U32", "RUST_U64", "RUST_U128", "RUST_USIZE",
    "RUST_F32", "RUST_F64",
    "RUST_BOOL", "RUST_CHAR", "RUST_STR",
    "RUST_UNIT", "RUST_NEVER",
    # Rust compound types
    "RustReferenceType",
    "RustSliceType",
    "RustArrayType",
    "RustOptionType",
    "RustResultType",
    "RustBoxType",
    "RustRcType",
    "RustVecType",
    "RustStringType",
    "RustFunctionType",
    "RustDynTraitType",
    "RustImplTraitType",
    # TypeScript
    "TypeScriptTypeSystem",
    "TS_STRING", "TS_NUMBER", "TS_BOOLEAN", "TS_BIGINT", "TS_SYMBOL",
    "TS_UNDEFINED", "TS_NULL", "TS_VOID", "TS_OBJECT", "TS_UNKNOWN",
    "TS_ANY", "TS_NEVER",
    "TSParameter",
    "TSLiteralType",
    "TSArrayType",
    "TSTupleType",
    "TSObjectType",
    "TSFunctionType",
    "TSTypeParameter",
    "TSUnionType",
    "TSIntersectionType",
    "TSConditionalType",
    "TSMappedType",
    "TSIndexedAccessType",
    "TSKeyofType",
    "TSTypeofType",
    "TSTemplateLiteralType",
    "TSTypeReference",
    "TSInferType",
    # Go
    "GoTypeSystem",
    "GO_BOOL", "GO_STRING",
    "GO_INT", "GO_INT8", "GO_INT16", "GO_INT32", "GO_INT64",
    "GO_UINT", "GO_UINT8", "GO_UINT16", "GO_UINT32", "GO_UINT64",
    "GO_FLOAT32", "GO_FLOAT64",
    "GO_COMPLEX64", "GO_COMPLEX128",
    "GO_BYTE", "GO_RUNE",
    "GO_ANY", "GO_ERROR",
    "GO_UNTYPED_INT", "GO_UNTYPED_FLOAT", "GO_UNTYPED_STRING",
    "GoArrayType",
    "GoSliceType",
    "GoMapType",
    "GoPointerType",
    "GoChannelType",
    "GoFunctionType",
    "GoInterfaceType",
    "GoStructType",
    "GoGenericType",
    "GoNamedType",
    # Kotlin
    "KotlinTypeSystem",
    "KOTLIN_BYTE", "KOTLIN_SHORT", "KOTLIN_INT", "KOTLIN_LONG",
    "KOTLIN_FLOAT", "KOTLIN_DOUBLE",
    "KOTLIN_UBYTE", "KOTLIN_USHORT", "KOTLIN_UINT", "KOTLIN_ULONG",
    "KOTLIN_BOOLEAN", "KOTLIN_CHAR", "KOTLIN_STRING",
    "KOTLIN_UNIT", "KOTLIN_ANY", "KOTLIN_NOTHING",
    "KotlinNullableType",
    "KotlinArrayType",
    "KotlinPrimitiveArrayType",
    "KotlinListType",
    "KotlinSetType",
    "KotlinMapType",
    "KotlinPairType",
    "KotlinFunctionType",
    "KotlinTypeParameter",
    "KotlinGenericType",
    "KotlinStarProjection",
    "KotlinClassType",
    "KotlinEnumType",
    # Swift
    "SwiftTypeSystem",
    "SWIFT_INT", "SWIFT_INT8", "SWIFT_INT16", "SWIFT_INT32", "SWIFT_INT64",
    "SWIFT_UINT", "SWIFT_UINT8", "SWIFT_UINT16", "SWIFT_UINT32", "SWIFT_UINT64",
    "SWIFT_FLOAT", "SWIFT_DOUBLE", "SWIFT_FLOAT16", "SWIFT_FLOAT80",
    "SWIFT_BOOL", "SWIFT_CHARACTER", "SWIFT_STRING",
    "SWIFT_VOID", "SWIFT_NEVER", "SWIFT_ANY", "SWIFT_ANY_OBJECT",
    "SwiftOptionalType",
    "SwiftImplicitlyUnwrappedOptionalType",
    "SwiftArrayType",
    "SwiftDictionaryType",
    "SwiftSetType",
    "SwiftTupleType",
    "SwiftFunctionType",
    "SwiftClosureType",
    "SwiftProtocolType",
    "SwiftProtocolCompositionType",
    "SwiftGenericType",
    "SwiftTypeParameter",
    "SwiftMetatypeType",
    "SwiftExistentialType",
    "SwiftOpaqueType",
    "SwiftResultType",
    "SwiftNamedType",
]
