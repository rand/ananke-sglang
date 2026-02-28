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
"""Tests for Go type system."""

import pytest

from domains.types.languages.go import (
    GoTypeSystem,
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
    GO_BOOL,
    GO_STRING,
    GO_INT,
    GO_INT8,
    GO_INT16,
    GO_INT32,
    GO_INT64,
    GO_UINT,
    GO_UINT8,
    GO_UINT16,
    GO_UINT32,
    GO_UINT64,
    GO_FLOAT32,
    GO_FLOAT64,
    GO_COMPLEX64,
    GO_COMPLEX128,
    GO_BYTE,
    GO_RUNE,
    GO_ANY,
    GO_ERROR,
    GO_UNTYPED_INT,
    GO_UNTYPED_FLOAT,
    GO_UNTYPED_STRING,
)
from domains.types.languages.base import LiteralInfo, LiteralKind


class TestGoTypeSystemCreation:
    """Tests for GoTypeSystem creation."""

    def test_create_type_system(self):
        """Should create a Go type system."""
        ts = GoTypeSystem()
        assert ts.name == "go"

    def test_capabilities(self):
        """Should report correct capabilities."""
        ts = GoTypeSystem()
        caps = ts.capabilities

        assert caps.supports_generics is True
        assert caps.supports_protocols is True  # Structural interfaces
        assert caps.supports_ownership is False
        assert caps.supports_comptime is False


class TestGoPrimitiveTypes:
    """Tests for Go primitive type parsing."""

    @pytest.fixture
    def ts(self):
        return GoTypeSystem()

    def test_parse_bool(self, ts):
        """Should parse bool type."""
        result = ts.parse_type_annotation("bool")
        assert result == GO_BOOL

    def test_parse_string(self, ts):
        """Should parse string type."""
        result = ts.parse_type_annotation("string")
        assert result == GO_STRING

    def test_parse_int(self, ts):
        """Should parse int type."""
        result = ts.parse_type_annotation("int")
        assert result == GO_INT

    def test_parse_int8(self, ts):
        """Should parse int8 type."""
        result = ts.parse_type_annotation("int8")
        assert result == GO_INT8

    def test_parse_int16(self, ts):
        """Should parse int16 type."""
        result = ts.parse_type_annotation("int16")
        assert result == GO_INT16

    def test_parse_int32(self, ts):
        """Should parse int32 type."""
        result = ts.parse_type_annotation("int32")
        assert result == GO_INT32

    def test_parse_int64(self, ts):
        """Should parse int64 type."""
        result = ts.parse_type_annotation("int64")
        assert result == GO_INT64

    def test_parse_uint(self, ts):
        """Should parse uint type."""
        result = ts.parse_type_annotation("uint")
        assert result == GO_UINT

    def test_parse_uint8(self, ts):
        """Should parse uint8 type."""
        result = ts.parse_type_annotation("uint8")
        assert result == GO_UINT8

    def test_parse_byte(self, ts):
        """Should parse byte as uint8."""
        result = ts.parse_type_annotation("byte")
        assert result == GO_BYTE
        assert result == GO_UINT8

    def test_parse_rune(self, ts):
        """Should parse rune as int32."""
        result = ts.parse_type_annotation("rune")
        assert result == GO_RUNE
        assert result == GO_INT32

    def test_parse_float32(self, ts):
        """Should parse float32 type."""
        result = ts.parse_type_annotation("float32")
        assert result == GO_FLOAT32

    def test_parse_float64(self, ts):
        """Should parse float64 type."""
        result = ts.parse_type_annotation("float64")
        assert result == GO_FLOAT64

    def test_parse_complex64(self, ts):
        """Should parse complex64 type."""
        result = ts.parse_type_annotation("complex64")
        assert result == GO_COMPLEX64

    def test_parse_complex128(self, ts):
        """Should parse complex128 type."""
        result = ts.parse_type_annotation("complex128")
        assert result == GO_COMPLEX128

    def test_parse_error(self, ts):
        """Should parse error type."""
        result = ts.parse_type_annotation("error")
        assert result == GO_ERROR

    def test_parse_any(self, ts):
        """Should parse any type."""
        result = ts.parse_type_annotation("any")
        assert result == GO_ANY


class TestGoCompositeTypes:
    """Tests for Go composite type parsing."""

    @pytest.fixture
    def ts(self):
        return GoTypeSystem()

    def test_parse_pointer(self, ts):
        """Should parse pointer type."""
        result = ts.parse_type_annotation("*int")
        assert isinstance(result, GoPointerType)
        assert result.pointee == GO_INT

    def test_parse_nested_pointer(self, ts):
        """Should parse nested pointer type."""
        result = ts.parse_type_annotation("**int")
        assert isinstance(result, GoPointerType)
        assert isinstance(result.pointee, GoPointerType)
        assert result.pointee.pointee == GO_INT

    def test_parse_slice(self, ts):
        """Should parse slice type."""
        result = ts.parse_type_annotation("[]int")
        assert isinstance(result, GoSliceType)
        assert result.element == GO_INT

    def test_parse_slice_of_pointers(self, ts):
        """Should parse slice of pointers."""
        result = ts.parse_type_annotation("[]*string")
        assert isinstance(result, GoSliceType)
        assert isinstance(result.element, GoPointerType)
        assert result.element.pointee == GO_STRING

    def test_parse_array(self, ts):
        """Should parse array type."""
        result = ts.parse_type_annotation("[10]int")
        assert isinstance(result, GoArrayType)
        assert result.length == 10
        assert result.element == GO_INT

    def test_parse_map(self, ts):
        """Should parse map type."""
        result = ts.parse_type_annotation("map[string]int")
        assert isinstance(result, GoMapType)
        assert result.key == GO_STRING
        assert result.value == GO_INT

    def test_parse_map_nested(self, ts):
        """Should parse nested map type."""
        result = ts.parse_type_annotation("map[string][]int")
        assert isinstance(result, GoMapType)
        assert result.key == GO_STRING
        assert isinstance(result.value, GoSliceType)

    def test_parse_channel_bidirectional(self, ts):
        """Should parse bidirectional channel."""
        result = ts.parse_type_annotation("chan int")
        assert isinstance(result, GoChannelType)
        assert result.element == GO_INT
        assert result.direction == "bidirectional"

    def test_parse_channel_send(self, ts):
        """Should parse send-only channel."""
        result = ts.parse_type_annotation("chan<- int")
        assert isinstance(result, GoChannelType)
        assert result.element == GO_INT
        assert result.direction == "send"

    def test_parse_channel_receive(self, ts):
        """Should parse receive-only channel."""
        result = ts.parse_type_annotation("<-chan int")
        assert isinstance(result, GoChannelType)
        assert result.element == GO_INT
        assert result.direction == "receive"


class TestGoFunctionTypes:
    """Tests for Go function type parsing."""

    @pytest.fixture
    def ts(self):
        return GoTypeSystem()

    def test_parse_func_no_params_no_return(self, ts):
        """Should parse func with no params and no return."""
        result = ts.parse_type_annotation("func()")
        assert isinstance(result, GoFunctionType)
        assert result.parameters == ()
        assert result.returns == ()

    def test_parse_func_with_param(self, ts):
        """Should parse func with parameter."""
        result = ts.parse_type_annotation("func(x int)")
        assert isinstance(result, GoFunctionType)
        assert len(result.parameters) == 1

    def test_parse_func_with_return(self, ts):
        """Should parse func with return type."""
        result = ts.parse_type_annotation("func() int")
        assert isinstance(result, GoFunctionType)
        assert len(result.returns) == 1
        assert result.returns[0] == GO_INT

    def test_parse_func_multiple_returns(self, ts):
        """Should parse func with multiple returns."""
        result = ts.parse_type_annotation("func() (int, error)")
        assert isinstance(result, GoFunctionType)
        assert len(result.returns) == 2


class TestGoGenericTypes:
    """Tests for Go generic type parsing."""

    @pytest.fixture
    def ts(self):
        return GoTypeSystem()

    def test_parse_generic_single_param(self, ts):
        """Should parse generic with single type param."""
        result = ts.parse_type_annotation("List[int]")
        assert isinstance(result, GoGenericType)
        assert result.base == "List"
        assert len(result.type_args) == 1
        assert result.type_args[0] == GO_INT

    def test_parse_generic_multiple_params(self, ts):
        """Should parse generic with multiple type params."""
        result = ts.parse_type_annotation("Map[string, int]")
        assert isinstance(result, GoGenericType)
        assert result.base == "Map"
        assert len(result.type_args) == 2


class TestGoAssignability:
    """Tests for Go type assignability."""

    @pytest.fixture
    def ts(self):
        return GoTypeSystem()

    def test_same_type_assignable(self, ts):
        """Same types should be assignable."""
        assert ts.check_assignable(GO_INT, GO_INT)
        assert ts.check_assignable(GO_STRING, GO_STRING)
        assert ts.check_assignable(GO_BOOL, GO_BOOL)

    def test_different_primitives_not_assignable(self, ts):
        """Different primitives should not be assignable."""
        assert not ts.check_assignable(GO_INT, GO_STRING)
        assert not ts.check_assignable(GO_FLOAT64, GO_INT)

    def test_any_accepts_all(self, ts):
        """Any should accept all types."""
        assert ts.check_assignable(GO_INT, GO_ANY)
        assert ts.check_assignable(GO_STRING, GO_ANY)
        assert ts.check_assignable(GoSliceType(GO_INT), GO_ANY)

    def test_untyped_int_assignable(self, ts):
        """Untyped int should be assignable to numeric types."""
        assert ts.check_assignable(GO_UNTYPED_INT, GO_INT)
        assert ts.check_assignable(GO_UNTYPED_INT, GO_INT64)
        assert ts.check_assignable(GO_UNTYPED_INT, GO_FLOAT64)

    def test_untyped_string_assignable(self, ts):
        """Untyped string should be assignable to string."""
        assert ts.check_assignable(GO_UNTYPED_STRING, GO_STRING)

    def test_slice_assignability(self, ts):
        """Slices with same element type should be assignable."""
        slice1 = GoSliceType(GO_INT)
        slice2 = GoSliceType(GO_INT)
        assert ts.check_assignable(slice1, slice2)

    def test_pointer_assignability(self, ts):
        """Pointers with same pointee should be assignable."""
        ptr1 = GoPointerType(GO_INT)
        ptr2 = GoPointerType(GO_INT)
        assert ts.check_assignable(ptr1, ptr2)

    def test_channel_assignability(self, ts):
        """Bidirectional channel assignable to directional."""
        bidi = GoChannelType(GO_INT, "bidirectional")
        send = GoChannelType(GO_INT, "send")
        recv = GoChannelType(GO_INT, "receive")

        assert ts.check_assignable(bidi, send)
        assert ts.check_assignable(bidi, recv)
        assert not ts.check_assignable(send, recv)


class TestGoTypeFormatting:
    """Tests for Go type formatting."""

    @pytest.fixture
    def ts(self):
        return GoTypeSystem()

    def test_format_primitive(self, ts):
        """Should format primitive types."""
        assert ts.format_type(GO_INT) == "int"
        assert ts.format_type(GO_STRING) == "string"

    def test_format_pointer(self, ts):
        """Should format pointer type."""
        ptr = GoPointerType(GO_INT)
        assert ts.format_type(ptr) == "*int"

    def test_format_slice(self, ts):
        """Should format slice type."""
        s = GoSliceType(GO_STRING)
        assert ts.format_type(s) == "[]string"

    def test_format_array(self, ts):
        """Should format array type."""
        arr = GoArrayType(5, GO_INT)
        assert ts.format_type(arr) == "[5]int"

    def test_format_map(self, ts):
        """Should format map type."""
        m = GoMapType(GO_STRING, GO_INT)
        assert ts.format_type(m) == "map[string]int"

    def test_format_channel(self, ts):
        """Should format channel types."""
        bidi = GoChannelType(GO_INT)
        send = GoChannelType(GO_INT, "send")
        recv = GoChannelType(GO_INT, "receive")

        assert ts.format_type(bidi) == "chan int"
        assert ts.format_type(send) == "chan<- int"
        assert ts.format_type(recv) == "<-chan int"


class TestGoLiteralInference:
    """Tests for Go literal type inference."""

    @pytest.fixture
    def ts(self):
        return GoTypeSystem()

    def test_infer_int_literal(self, ts):
        """Should infer untyped int for integer literals."""
        lit = LiteralInfo(kind=LiteralKind.INTEGER, value=42)
        result = ts.infer_literal_type(lit)
        assert result == GO_UNTYPED_INT

    def test_infer_float_literal(self, ts):
        """Should infer untyped float for float literals."""
        lit = LiteralInfo(kind=LiteralKind.FLOAT, value=3.14)
        result = ts.infer_literal_type(lit)
        assert result == GO_UNTYPED_FLOAT

    def test_infer_string_literal(self, ts):
        """Should infer untyped string for string literals."""
        lit = LiteralInfo(kind=LiteralKind.STRING, value="hello")
        result = ts.infer_literal_type(lit)
        assert result == GO_UNTYPED_STRING


class TestGoBuiltinFunctions:
    """Tests for Go builtin functions."""

    @pytest.fixture
    def ts(self):
        return GoTypeSystem()

    def test_get_builtins(self, ts):
        """Should return builtin functions."""
        builtins = ts.get_builtin_functions()
        assert "len" in builtins
        assert "cap" in builtins
        assert "make" in builtins
        assert "new" in builtins
        assert "append" in builtins
        assert "panic" in builtins


class TestGoLUBGLB:
    """Tests for Go LUB and GLB operations."""

    @pytest.fixture
    def ts(self):
        return GoTypeSystem()

    def test_lub_same_types(self, ts):
        """LUB of same types is that type."""
        result = ts.lub([GO_INT, GO_INT])
        assert result == GO_INT

    def test_lub_different_types(self, ts):
        """LUB of different types is any."""
        result = ts.lub([GO_INT, GO_STRING])
        assert result == GO_ANY

    def test_glb_same_types(self, ts):
        """GLB of same types is that type."""
        result = ts.glb([GO_INT, GO_INT])
        assert result == GO_INT
