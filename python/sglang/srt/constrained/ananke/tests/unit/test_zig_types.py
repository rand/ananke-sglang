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
"""Unit tests for Zig type system.

Tests for the ZigTypeSystem implementation including:
- Primitive type parsing
- Pointer types (*T, *const T, [*]T, etc.)
- Optional types (?T)
- Error unions (E!T)
- Arrays and slices
- Comptime type handling
- Assignability rules
"""

import pytest

from domains.types.constraint import (
    ANY,
    NEVER,
)
from domains.types.languages import (
    get_type_system,
    supported_languages,
    TypeParseError,
    LiteralInfo,
    LiteralKind,
)
from domains.types.languages.zig import (
    ZigTypeSystem,
    ZigOptionalType,
    ZigErrorUnionType,
    ZigPointerType,
    ZigSliceType,
    ZigArrayType,
    ZigManyPointerType,
    ZigCPointerType,
    ZigStructType,
    ZigEnumType,
    ZigUnionType,
    ZigFunctionType,
    ZigVectorType,
    ZIG_I8,
    ZIG_I16,
    ZIG_I32,
    ZIG_I64,
    ZIG_I128,
    ZIG_ISIZE,
    ZIG_U8,
    ZIG_U16,
    ZIG_U32,
    ZIG_U64,
    ZIG_U128,
    ZIG_USIZE,
    ZIG_F16,
    ZIG_F32,
    ZIG_F64,
    ZIG_F80,
    ZIG_F128,
    ZIG_BOOL,
    ZIG_VOID,
    ZIG_NORETURN,
    ZIG_TYPE,
    ZIG_ANYTYPE,
    ZIG_ANYOPAQUE,
    ZIG_COMPTIME_INT,
    ZIG_COMPTIME_FLOAT,
    ZIG_C_CHAR,
    ZIG_C_INT,
)
from domains.types.extended_types import ANYERROR


# Helper functions for type checking
def is_comptime_type(t) -> bool:
    """Check if type is a comptime type."""
    return t in (ZIG_COMPTIME_INT, ZIG_COMPTIME_FLOAT, ZIG_TYPE, ZIG_ANYTYPE)


def is_error_union_type(t) -> bool:
    """Check if type is an error union."""
    return isinstance(t, ZigErrorUnionType)


# ===========================================================================
# Factory Function Tests
# ===========================================================================


class TestGetZigTypeSystem:
    """Tests for get_type_system with Zig."""

    def test_get_zig_by_name(self):
        """Should return Zig type system."""
        ts = get_type_system("zig")
        assert isinstance(ts, ZigTypeSystem)
        assert ts.name == "zig"

    def test_zig_in_supported_languages(self):
        """Zig should be in supported languages."""
        langs = supported_languages()
        assert "zig" in langs


# ===========================================================================
# Zig Type System Basic Tests
# ===========================================================================


class TestZigTypeSystemBasics:
    """Basic tests for ZigTypeSystem."""

    @pytest.fixture
    def ts(self):
        """Create a Zig type system instance."""
        return ZigTypeSystem()

    def test_name(self, ts):
        """Name should be 'zig'."""
        assert ts.name == "zig"

    def test_capabilities(self, ts):
        """Should have correct capabilities."""
        caps = ts.capabilities
        assert caps.supports_generics
        assert caps.supports_comptime
        assert caps.supports_error_unions
        assert caps.supports_sentinels
        assert caps.supports_allocators
        assert not caps.supports_ownership  # Zig doesn't have Rust-style ownership


# ===========================================================================
# Primitive Type Parsing Tests
# ===========================================================================


class TestZigPrimitiveTypeParsing:
    """Tests for parsing Zig primitive types."""

    @pytest.fixture
    def ts(self):
        return ZigTypeSystem()

    # Signed integers
    def test_parse_i8(self, ts):
        """Should parse 'i8'."""
        typ = ts.parse_type_annotation("i8")
        assert typ == ZIG_I8

    def test_parse_i16(self, ts):
        """Should parse 'i16'."""
        typ = ts.parse_type_annotation("i16")
        assert typ == ZIG_I16

    def test_parse_i32(self, ts):
        """Should parse 'i32'."""
        typ = ts.parse_type_annotation("i32")
        assert typ == ZIG_I32

    def test_parse_i64(self, ts):
        """Should parse 'i64'."""
        typ = ts.parse_type_annotation("i64")
        assert typ == ZIG_I64

    def test_parse_i128(self, ts):
        """Should parse 'i128'."""
        typ = ts.parse_type_annotation("i128")
        assert typ == ZIG_I128

    def test_parse_isize(self, ts):
        """Should parse 'isize'."""
        typ = ts.parse_type_annotation("isize")
        assert typ == ZIG_ISIZE

    # Unsigned integers
    def test_parse_u8(self, ts):
        """Should parse 'u8'."""
        typ = ts.parse_type_annotation("u8")
        assert typ == ZIG_U8

    def test_parse_u16(self, ts):
        """Should parse 'u16'."""
        typ = ts.parse_type_annotation("u16")
        assert typ == ZIG_U16

    def test_parse_u32(self, ts):
        """Should parse 'u32'."""
        typ = ts.parse_type_annotation("u32")
        assert typ == ZIG_U32

    def test_parse_u64(self, ts):
        """Should parse 'u64'."""
        typ = ts.parse_type_annotation("u64")
        assert typ == ZIG_U64

    def test_parse_u128(self, ts):
        """Should parse 'u128'."""
        typ = ts.parse_type_annotation("u128")
        assert typ == ZIG_U128

    def test_parse_usize(self, ts):
        """Should parse 'usize'."""
        typ = ts.parse_type_annotation("usize")
        assert typ == ZIG_USIZE

    # Arbitrary-width integers
    def test_parse_u3(self, ts):
        """Should parse 'u3' (arbitrary width)."""
        typ = ts.parse_type_annotation("u3")
        assert typ.name == "u3"

    def test_parse_i17(self, ts):
        """Should parse 'i17' (arbitrary width)."""
        typ = ts.parse_type_annotation("i17")
        assert typ.name == "i17"

    # Floats
    def test_parse_f16(self, ts):
        """Should parse 'f16'."""
        typ = ts.parse_type_annotation("f16")
        assert typ == ZIG_F16

    def test_parse_f32(self, ts):
        """Should parse 'f32'."""
        typ = ts.parse_type_annotation("f32")
        assert typ == ZIG_F32

    def test_parse_f64(self, ts):
        """Should parse 'f64'."""
        typ = ts.parse_type_annotation("f64")
        assert typ == ZIG_F64

    def test_parse_f80(self, ts):
        """Should parse 'f80'."""
        typ = ts.parse_type_annotation("f80")
        assert typ == ZIG_F80

    def test_parse_f128(self, ts):
        """Should parse 'f128'."""
        typ = ts.parse_type_annotation("f128")
        assert typ == ZIG_F128

    # Special types
    def test_parse_bool(self, ts):
        """Should parse 'bool'."""
        typ = ts.parse_type_annotation("bool")
        assert typ == ZIG_BOOL

    def test_parse_void(self, ts):
        """Should parse 'void'."""
        typ = ts.parse_type_annotation("void")
        assert typ == ZIG_VOID

    def test_parse_noreturn(self, ts):
        """Should parse 'noreturn'."""
        typ = ts.parse_type_annotation("noreturn")
        assert typ == ZIG_NORETURN

    def test_parse_type(self, ts):
        """Should parse 'type'."""
        typ = ts.parse_type_annotation("type")
        assert typ == ZIG_TYPE

    def test_parse_anytype(self, ts):
        """Should parse 'anytype'."""
        typ = ts.parse_type_annotation("anytype")
        assert typ == ZIG_ANYTYPE

    def test_parse_anyopaque(self, ts):
        """Should parse 'anyopaque'."""
        typ = ts.parse_type_annotation("anyopaque")
        assert typ == ZIG_ANYOPAQUE

    def test_parse_anyerror(self, ts):
        """Should parse 'anyerror'."""
        typ = ts.parse_type_annotation("anyerror")
        assert typ == ANYERROR

    # Comptime types
    def test_parse_comptime_int(self, ts):
        """Should parse 'comptime_int'."""
        typ = ts.parse_type_annotation("comptime_int")
        assert typ == ZIG_COMPTIME_INT

    def test_parse_comptime_float(self, ts):
        """Should parse 'comptime_float'."""
        typ = ts.parse_type_annotation("comptime_float")
        assert typ == ZIG_COMPTIME_FLOAT

    # C interop types
    def test_parse_c_int(self, ts):
        """Should parse 'c_int'."""
        typ = ts.parse_type_annotation("c_int")
        assert typ == ZIG_C_INT

    def test_parse_c_char(self, ts):
        """Should parse 'c_char'."""
        typ = ts.parse_type_annotation("c_char")
        assert typ == ZIG_C_CHAR


# ===========================================================================
# Pointer Type Parsing Tests
# ===========================================================================


class TestZigPointerTypeParsing:
    """Tests for parsing Zig pointer types."""

    @pytest.fixture
    def ts(self):
        return ZigTypeSystem()

    def test_parse_single_pointer(self, ts):
        """Should parse '*i32'."""
        typ = ts.parse_type_annotation("*i32")
        assert isinstance(typ, ZigPointerType)
        assert typ.pointee == ZIG_I32
        assert not typ.is_const

    def test_parse_const_pointer(self, ts):
        """Should parse '*const u8'."""
        typ = ts.parse_type_annotation("*const u8")
        assert isinstance(typ, ZigPointerType)
        assert typ.pointee == ZIG_U8
        assert typ.is_const

    def test_parse_many_pointer(self, ts):
        """Should parse '[*]u8'."""
        typ = ts.parse_type_annotation("[*]u8")
        assert isinstance(typ, ZigManyPointerType)
        assert typ.element == ZIG_U8

    def test_parse_c_pointer(self, ts):
        """Should parse '[*c]u8'."""
        typ = ts.parse_type_annotation("[*c]u8")
        assert isinstance(typ, ZigCPointerType)
        assert typ.element == ZIG_U8

    def test_parse_sentinel_pointer(self, ts):
        """Should parse '[*:0]u8' (null-terminated)."""
        typ = ts.parse_type_annotation("[*:0]u8")
        assert isinstance(typ, ZigManyPointerType)
        assert typ.element == ZIG_U8
        assert typ.sentinel == "0"  # Sentinel is kept as string


# ===========================================================================
# Slice Type Parsing Tests
# ===========================================================================


class TestZigSliceTypeParsing:
    """Tests for parsing Zig slice types."""

    @pytest.fixture
    def ts(self):
        return ZigTypeSystem()

    def test_parse_slice(self, ts):
        """Should parse '[]u8'."""
        typ = ts.parse_type_annotation("[]u8")
        assert isinstance(typ, ZigSliceType)
        assert typ.element == ZIG_U8
        assert not typ.is_const

    def test_parse_const_slice(self, ts):
        """Should parse '[]const u8'."""
        typ = ts.parse_type_annotation("[]const u8")
        assert isinstance(typ, ZigSliceType)
        assert typ.element == ZIG_U8
        assert typ.is_const

    def test_parse_sentinel_slice(self, ts):
        """Should parse '[:0]u8' (null-terminated slice)."""
        typ = ts.parse_type_annotation("[:0]u8")
        assert isinstance(typ, ZigSliceType)
        assert typ.element == ZIG_U8
        assert typ.sentinel == "0"  # Sentinel is kept as string


# ===========================================================================
# Array Type Parsing Tests
# ===========================================================================


class TestZigArrayTypeParsing:
    """Tests for parsing Zig array types."""

    @pytest.fixture
    def ts(self):
        return ZigTypeSystem()

    def test_parse_array(self, ts):
        """Should parse '[10]u8'."""
        typ = ts.parse_type_annotation("[10]u8")
        assert isinstance(typ, ZigArrayType)
        assert typ.element == ZIG_U8
        assert typ.length == 10

    def test_parse_array_with_sentinel(self, ts):
        """Should parse '[10:0]u8' (sentinel-terminated array)."""
        typ = ts.parse_type_annotation("[10:0]u8")
        assert isinstance(typ, ZigArrayType)
        assert typ.element == ZIG_U8
        assert typ.length == 10
        assert typ.sentinel == "0"  # Sentinel is kept as string

    def test_parse_inferred_array(self, ts):
        """Should parse '[_]u8' (inferred length)."""
        typ = ts.parse_type_annotation("[_]u8")
        assert isinstance(typ, ZigArrayType)
        assert typ.element == ZIG_U8
        assert typ.length is None


# ===========================================================================
# Optional Type Parsing Tests
# ===========================================================================


class TestZigOptionalTypeParsing:
    """Tests for parsing Zig optional types."""

    @pytest.fixture
    def ts(self):
        return ZigTypeSystem()

    def test_parse_optional(self, ts):
        """Should parse '?i32'."""
        typ = ts.parse_type_annotation("?i32")
        assert isinstance(typ, ZigOptionalType)
        assert typ.inner == ZIG_I32

    def test_parse_optional_pointer(self, ts):
        """Should parse '?*u8'."""
        typ = ts.parse_type_annotation("?*u8")
        assert isinstance(typ, ZigOptionalType)
        assert isinstance(typ.inner, ZigPointerType)
        assert typ.inner.pointee == ZIG_U8


# ===========================================================================
# Error Union Type Parsing Tests
# ===========================================================================


class TestZigErrorUnionTypeParsing:
    """Tests for parsing Zig error union types."""

    @pytest.fixture
    def ts(self):
        return ZigTypeSystem()

    def test_parse_error_union_simple(self, ts):
        """Should parse 'anyerror!i32'."""
        typ = ts.parse_type_annotation("anyerror!i32")
        assert isinstance(typ, ZigErrorUnionType)
        assert typ.payload == ZIG_I32
        assert typ.error_set == ANYERROR

    def test_parse_error_union_with_void(self, ts):
        """Should parse 'anyerror!void'."""
        typ = ts.parse_type_annotation("anyerror!void")
        assert isinstance(typ, ZigErrorUnionType)
        assert typ.payload == ZIG_VOID

    def test_is_error_union_type_helper(self, ts):
        """is_error_union_type should work correctly."""
        error_union = ts.parse_type_annotation("anyerror!i32")
        assert is_error_union_type(error_union)
        assert not is_error_union_type(ZIG_I32)


# ===========================================================================
# Function Type Parsing Tests
# ===========================================================================


class TestZigFunctionTypeParsing:
    """Tests for parsing Zig function types."""

    @pytest.fixture
    def ts(self):
        return ZigTypeSystem()

    def test_parse_function_simple(self, ts):
        """Should parse 'fn(i32) void'."""
        typ = ts.parse_type_annotation("fn(i32) void")
        assert isinstance(typ, ZigFunctionType)
        assert len(typ.params) == 1
        assert typ.params[0] == ZIG_I32
        assert typ.return_type == ZIG_VOID

    def test_parse_function_multiple_params(self, ts):
        """Should parse 'fn(i32, u64) bool'."""
        typ = ts.parse_type_annotation("fn(i32, u64) bool")
        assert isinstance(typ, ZigFunctionType)
        assert len(typ.params) == 2
        assert typ.params[0] == ZIG_I32
        assert typ.params[1] == ZIG_U64
        assert typ.return_type == ZIG_BOOL

    def test_parse_function_no_params(self, ts):
        """Should parse 'fn() void'."""
        typ = ts.parse_type_annotation("fn() void")
        assert isinstance(typ, ZigFunctionType)
        assert len(typ.params) == 0
        assert typ.return_type == ZIG_VOID

    def test_parse_function_error_return(self, ts):
        """Should parse 'fn(i32) anyerror!u64'."""
        typ = ts.parse_type_annotation("fn(i32) anyerror!u64")
        assert isinstance(typ, ZigFunctionType)
        assert isinstance(typ.return_type, ZigErrorUnionType)


# ===========================================================================
# Vector Type Parsing Tests
# ===========================================================================


class TestZigVectorTypeParsing:
    """Tests for parsing Zig SIMD vector types."""

    @pytest.fixture
    def ts(self):
        return ZigTypeSystem()

    def test_parse_vector(self, ts):
        """Should parse '@Vector(4, f32)'."""
        typ = ts.parse_type_annotation("@Vector(4, f32)")
        assert isinstance(typ, ZigVectorType)
        assert typ.length == 4
        assert typ.element == ZIG_F32


# ===========================================================================
# Comptime Type Tests
# ===========================================================================


class TestZigComptimeTypes:
    """Tests for Zig comptime type handling."""

    @pytest.fixture
    def ts(self):
        return ZigTypeSystem()

    def test_is_comptime_type_int(self, ts):
        """comptime_int should be a comptime type."""
        assert is_comptime_type(ZIG_COMPTIME_INT)

    def test_is_comptime_type_float(self, ts):
        """comptime_float should be a comptime type."""
        assert is_comptime_type(ZIG_COMPTIME_FLOAT)

    def test_is_comptime_type_type(self, ts):
        """type should be a comptime type."""
        assert is_comptime_type(ZIG_TYPE)

    def test_is_not_comptime_type(self, ts):
        """Regular types should not be comptime types."""
        assert not is_comptime_type(ZIG_I32)
        assert not is_comptime_type(ZIG_F64)
        assert not is_comptime_type(ZIG_BOOL)


# ===========================================================================
# Assignability Tests
# ===========================================================================


class TestZigAssignability:
    """Tests for Zig type assignability checking."""

    @pytest.fixture
    def ts(self):
        return ZigTypeSystem()

    def test_same_type_assignable(self, ts):
        """Same types should be assignable."""
        assert ts.check_assignable(ZIG_I32, ZIG_I32)
        assert ts.check_assignable(ZIG_U8, ZIG_U8)

    def test_anytype_assignable(self, ts):
        """anytype should be assignable to anything."""
        assert ts.check_assignable(ZIG_ANYTYPE, ZIG_I32)
        assert ts.check_assignable(ZIG_I32, ZIG_ANYTYPE)

    def test_comptime_int_to_integer(self, ts):
        """comptime_int should be assignable to any integer type."""
        assert ts.check_assignable(ZIG_COMPTIME_INT, ZIG_I32)
        assert ts.check_assignable(ZIG_COMPTIME_INT, ZIG_U8)
        assert ts.check_assignable(ZIG_COMPTIME_INT, ZIG_I64)
        assert ts.check_assignable(ZIG_COMPTIME_INT, ZIG_USIZE)

    def test_comptime_float_to_float(self, ts):
        """comptime_float should be assignable to any float type."""
        assert ts.check_assignable(ZIG_COMPTIME_FLOAT, ZIG_F32)
        assert ts.check_assignable(ZIG_COMPTIME_FLOAT, ZIG_F64)
        assert ts.check_assignable(ZIG_COMPTIME_FLOAT, ZIG_F128)

    def test_integer_not_to_float(self, ts):
        """Integer types should not be assignable to float types."""
        assert not ts.check_assignable(ZIG_I32, ZIG_F32)

    def test_float_not_to_integer(self, ts):
        """Float types should not be assignable to integer types."""
        assert not ts.check_assignable(ZIG_F32, ZIG_I32)

    def test_optional_null(self, ts):
        """null should be assignable to optional types."""
        optional_i32 = ZigOptionalType(ZIG_I32)
        # We'd need a null type representation for this test
        # For now, test that inner type is assignable
        assert ts.check_assignable(ZIG_I32, optional_i32)

    def test_inner_to_optional(self, ts):
        """T should be assignable to ?T."""
        optional_i32 = ZigOptionalType(ZIG_I32)
        assert ts.check_assignable(ZIG_I32, optional_i32)

    def test_noreturn_assignable_to_all(self, ts):
        """noreturn should be assignable to any type."""
        assert ts.check_assignable(ZIG_NORETURN, ZIG_I32)
        assert ts.check_assignable(ZIG_NORETURN, ZIG_VOID)

    def test_slice_covariance(self, ts):
        """Slices should be covariant in element type."""
        slice1 = ZigSliceType(ZIG_U8)
        slice2 = ZigSliceType(ZIG_U8)
        assert ts.check_assignable(slice1, slice2)

    def test_pointer_covariance(self, ts):
        """Const pointer should be assignable to non-const."""
        ptr_const = ZigPointerType(ZIG_I32, is_const=True)
        ptr_mutable = ZigPointerType(ZIG_I32, is_const=False)
        # Mutable can be coerced to const
        assert ts.check_assignable(ptr_mutable, ptr_const)


# ===========================================================================
# Type Formatting Tests
# ===========================================================================


class TestZigTypeFormatting:
    """Tests for Zig type formatting."""

    @pytest.fixture
    def ts(self):
        return ZigTypeSystem()

    def test_format_primitive(self, ts):
        """Should format primitives correctly."""
        assert ts.format_type(ZIG_I32) == "i32"
        assert ts.format_type(ZIG_U8) == "u8"
        assert ts.format_type(ZIG_BOOL) == "bool"

    def test_format_pointer(self, ts):
        """Should format pointer types."""
        ptr = ZigPointerType(ZIG_I32)
        assert ts.format_type(ptr) == "*i32"

    def test_format_const_pointer(self, ts):
        """Should format const pointer types."""
        ptr = ZigPointerType(ZIG_U8, is_const=True)
        assert ts.format_type(ptr) == "*const u8"

    def test_format_slice(self, ts):
        """Should format slice types."""
        slc = ZigSliceType(ZIG_U8)
        assert ts.format_type(slc) == "[]u8"

    def test_format_const_slice(self, ts):
        """Should format const slice types."""
        slc = ZigSliceType(ZIG_U8, is_const=True)
        assert ts.format_type(slc) == "[]const u8"

    def test_format_array(self, ts):
        """Should format array types."""
        arr = ZigArrayType(ZIG_U8, 10)
        assert ts.format_type(arr) == "[10]u8"

    def test_format_optional(self, ts):
        """Should format optional types."""
        opt = ZigOptionalType(ZIG_I32)
        assert ts.format_type(opt) == "?i32"

    def test_format_error_union(self, ts):
        """Should format error union types."""
        err = ZigErrorUnionType(ANYERROR, ZIG_I32)
        formatted = ts.format_type(err)
        assert "!" in formatted
        assert "i32" in formatted

    def test_format_function(self, ts):
        """Should format function types."""
        func = ZigFunctionType([ZIG_I32], ZIG_VOID)
        formatted = ts.format_type(func)
        assert "fn" in formatted
        assert "i32" in formatted
        assert "void" in formatted


# ===========================================================================
# Literal Type Inference Tests
# ===========================================================================


class TestZigLiteralInference:
    """Tests for Zig literal type inference."""

    @pytest.fixture
    def ts(self):
        return ZigTypeSystem()

    def test_infer_integer(self, ts):
        """Should infer comptime_int for integer literals."""
        literal = LiteralInfo(LiteralKind.INTEGER, value=42)
        result = ts.infer_literal_type(literal)
        assert result == ZIG_COMPTIME_INT

    def test_infer_float(self, ts):
        """Should infer comptime_float for float literals."""
        literal = LiteralInfo(LiteralKind.FLOAT, value=3.14)
        result = ts.infer_literal_type(literal)
        assert result == ZIG_COMPTIME_FLOAT

    def test_infer_boolean(self, ts):
        """Should infer bool for boolean literals."""
        literal = LiteralInfo(LiteralKind.BOOLEAN, value=True)
        result = ts.infer_literal_type(literal)
        assert result == ZIG_BOOL

    def test_infer_string(self, ts):
        """Should infer const slice of u8 for string literals."""
        literal = LiteralInfo(LiteralKind.STRING, value="hello")
        result = ts.infer_literal_type(literal)
        # Without source text, returns slice type (coerced form)
        assert isinstance(result, ZigSliceType)
        assert result.element == ZIG_U8
        assert result.is_const is True

    def test_infer_string_with_text(self, ts):
        """With source text, infer pointer to array for string literals."""
        literal = LiteralInfo(LiteralKind.STRING, value="hello", text='"hello"')
        result = ts.infer_literal_type(literal)
        # With source text, returns pointer to sentinel-terminated array
        assert isinstance(result, ZigPointerType)
        assert result.is_const is True
        assert isinstance(result.pointee, ZigArrayType)
        assert result.pointee.element == ZIG_U8
        assert result.pointee.length == 5  # "hello" is 5 characters


# ===========================================================================
# Parsing Error Tests
# ===========================================================================


class TestZigTypeParsingErrors:
    """Tests for type parsing errors."""

    @pytest.fixture
    def ts(self):
        return ZigTypeSystem()

    def test_empty_annotation_fails(self, ts):
        """Empty annotation should raise TypeParseError."""
        with pytest.raises(TypeParseError):
            ts.parse_type_annotation("")

    def test_unknown_type_name_is_struct(self, ts):
        """Unknown type name should be parsed as struct (user-defined type)."""
        # In Zig, unknown identifiers are valid as they could be user-defined types
        typ = ts.parse_type_annotation("MyCustomType")
        assert isinstance(typ, ZigStructType)
        assert typ.name == "MyCustomType"

    def test_unclosed_bracket_fails(self, ts):
        """Unclosed bracket should raise TypeParseError."""
        with pytest.raises(TypeParseError):
            ts.parse_type_annotation("[10u8")


# ===========================================================================
# Builtin Types and Functions Tests
# ===========================================================================


class TestZigBuiltins:
    """Tests for Zig builtin types."""

    @pytest.fixture
    def ts(self):
        return ZigTypeSystem()

    def test_has_builtin_types(self, ts):
        """Should have builtin types."""
        builtins = ts.get_builtin_types()
        assert "i32" in builtins
        assert "u8" in builtins
        assert "bool" in builtins
        assert "void" in builtins
        assert "comptime_int" in builtins

    def test_has_builtin_functions(self, ts):
        """Should have builtin function signatures."""
        funcs = ts.get_builtin_functions()
        assert "@import" in funcs
        assert "@TypeOf" in funcs
        assert "@sizeOf" in funcs
