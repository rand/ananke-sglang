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
"""Tests for Swift type system."""

import pytest

from domains.types.languages.swift import (
    SwiftTypeSystem,
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
    SWIFT_INT,
    SWIFT_INT8,
    SWIFT_INT16,
    SWIFT_INT32,
    SWIFT_INT64,
    SWIFT_UINT,
    SWIFT_UINT8,
    SWIFT_UINT16,
    SWIFT_UINT32,
    SWIFT_UINT64,
    SWIFT_FLOAT,
    SWIFT_DOUBLE,
    SWIFT_FLOAT16,
    SWIFT_FLOAT80,
    SWIFT_BOOL,
    SWIFT_CHARACTER,
    SWIFT_STRING,
    SWIFT_VOID,
    SWIFT_NEVER,
    SWIFT_ANY,
    SWIFT_ANY_OBJECT,
)
from domains.types.languages.base import LiteralInfo, LiteralKind


class TestSwiftTypeSystemCreation:
    """Tests for SwiftTypeSystem creation."""

    def test_create_type_system(self):
        """Should create a Swift type system."""
        ts = SwiftTypeSystem()
        assert ts.name == "swift"

    def test_capabilities(self):
        """Should report correct capabilities."""
        ts = SwiftTypeSystem()
        caps = ts.capabilities

        assert caps.supports_generics is True
        assert caps.supports_optional_types is True
        assert caps.supports_protocols is True
        assert caps.supports_variance is True
        assert caps.supports_overloading is True
        assert caps.supports_union_types is False


class TestSwiftPrimitiveTypes:
    """Tests for Swift primitive type parsing."""

    @pytest.fixture
    def ts(self):
        return SwiftTypeSystem()

    def test_parse_int(self, ts):
        """Should parse Int type."""
        result = ts.parse_type_annotation("Int")
        assert result == SWIFT_INT

    def test_parse_int8(self, ts):
        """Should parse Int8 type."""
        result = ts.parse_type_annotation("Int8")
        assert result == SWIFT_INT8

    def test_parse_int16(self, ts):
        """Should parse Int16 type."""
        result = ts.parse_type_annotation("Int16")
        assert result == SWIFT_INT16

    def test_parse_int32(self, ts):
        """Should parse Int32 type."""
        result = ts.parse_type_annotation("Int32")
        assert result == SWIFT_INT32

    def test_parse_int64(self, ts):
        """Should parse Int64 type."""
        result = ts.parse_type_annotation("Int64")
        assert result == SWIFT_INT64

    def test_parse_uint(self, ts):
        """Should parse UInt type."""
        result = ts.parse_type_annotation("UInt")
        assert result == SWIFT_UINT

    def test_parse_float(self, ts):
        """Should parse Float type."""
        result = ts.parse_type_annotation("Float")
        assert result == SWIFT_FLOAT

    def test_parse_double(self, ts):
        """Should parse Double type."""
        result = ts.parse_type_annotation("Double")
        assert result == SWIFT_DOUBLE

    def test_parse_bool(self, ts):
        """Should parse Bool type."""
        result = ts.parse_type_annotation("Bool")
        assert result == SWIFT_BOOL

    def test_parse_character(self, ts):
        """Should parse Character type."""
        result = ts.parse_type_annotation("Character")
        assert result == SWIFT_CHARACTER

    def test_parse_string(self, ts):
        """Should parse String type."""
        result = ts.parse_type_annotation("String")
        assert result == SWIFT_STRING

    def test_parse_void(self, ts):
        """Should parse Void type."""
        result = ts.parse_type_annotation("Void")
        assert result == SWIFT_VOID

    def test_parse_never(self, ts):
        """Should parse Never type."""
        result = ts.parse_type_annotation("Never")
        assert result == SWIFT_NEVER

    def test_parse_any(self, ts):
        """Should parse Any type."""
        result = ts.parse_type_annotation("Any")
        assert result == SWIFT_ANY

    def test_parse_any_object(self, ts):
        """Should parse AnyObject type."""
        result = ts.parse_type_annotation("AnyObject")
        assert result == SWIFT_ANY_OBJECT


class TestSwiftOptionalTypes:
    """Tests for Swift optional type parsing."""

    @pytest.fixture
    def ts(self):
        return SwiftTypeSystem()

    def test_parse_optional_int(self, ts):
        """Should parse Int? type."""
        result = ts.parse_type_annotation("Int?")
        assert isinstance(result, SwiftOptionalType)
        assert result.wrapped == SWIFT_INT

    def test_parse_optional_string(self, ts):
        """Should parse String? type."""
        result = ts.parse_type_annotation("String?")
        assert isinstance(result, SwiftOptionalType)
        assert result.wrapped == SWIFT_STRING

    def test_parse_implicitly_unwrapped_optional(self, ts):
        """Should parse Int! type."""
        result = ts.parse_type_annotation("Int!")
        assert isinstance(result, SwiftImplicitlyUnwrappedOptionalType)
        assert result.wrapped == SWIFT_INT

    def test_optional_formatting(self, ts):
        """Should format optional types correctly."""
        result = ts.parse_type_annotation("Int?")
        assert ts.format_type(result) == "Int?"

    def test_parse_optional_generic_form(self, ts):
        """Should parse Optional<Int> type."""
        result = ts.parse_type_annotation("Optional<Int>")
        assert isinstance(result, SwiftOptionalType)
        assert result.wrapped == SWIFT_INT


class TestSwiftArrayTypes:
    """Tests for Swift array type parsing."""

    @pytest.fixture
    def ts(self):
        return SwiftTypeSystem()

    def test_parse_array_short_form(self, ts):
        """Should parse [Int] type."""
        result = ts.parse_type_annotation("[Int]")
        assert isinstance(result, SwiftArrayType)
        assert result.element == SWIFT_INT

    def test_parse_array_generic_form(self, ts):
        """Should parse Array<Int> type."""
        result = ts.parse_type_annotation("Array<Int>")
        assert isinstance(result, SwiftArrayType)
        assert result.element == SWIFT_INT

    def test_array_formatting(self, ts):
        """Should format array types correctly."""
        result = ts.parse_type_annotation("[String]")
        assert ts.format_type(result) == "[String]"


class TestSwiftDictionaryTypes:
    """Tests for Swift dictionary type parsing."""

    @pytest.fixture
    def ts(self):
        return SwiftTypeSystem()

    def test_parse_dictionary_short_form(self, ts):
        """Should parse [String: Int] type."""
        result = ts.parse_type_annotation("[String: Int]")
        assert isinstance(result, SwiftDictionaryType)
        assert result.key == SWIFT_STRING
        assert result.value == SWIFT_INT

    def test_parse_dictionary_generic_form(self, ts):
        """Should parse Dictionary<String, Int> type."""
        result = ts.parse_type_annotation("Dictionary<String, Int>")
        assert isinstance(result, SwiftDictionaryType)
        assert result.key == SWIFT_STRING
        assert result.value == SWIFT_INT

    def test_dictionary_formatting(self, ts):
        """Should format dictionary types correctly."""
        result = ts.parse_type_annotation("[String: Int]")
        assert ts.format_type(result) == "[String: Int]"


class TestSwiftSetTypes:
    """Tests for Swift set type parsing."""

    @pytest.fixture
    def ts(self):
        return SwiftTypeSystem()

    def test_parse_set(self, ts):
        """Should parse Set<Int> type."""
        result = ts.parse_type_annotation("Set<Int>")
        assert isinstance(result, SwiftSetType)
        assert result.element == SWIFT_INT

    def test_set_formatting(self, ts):
        """Should format set types correctly."""
        result = ts.parse_type_annotation("Set<String>")
        assert ts.format_type(result) == "Set<String>"


class TestSwiftTupleTypes:
    """Tests for Swift tuple type parsing."""

    @pytest.fixture
    def ts(self):
        return SwiftTypeSystem()

    def test_parse_tuple(self, ts):
        """Should parse (Int, String) type."""
        result = ts.parse_type_annotation("(Int, String)")
        assert isinstance(result, SwiftTupleType)
        assert len(result.elements) == 2
        assert result.elements[0] == SWIFT_INT
        assert result.elements[1] == SWIFT_STRING

    def test_parse_empty_tuple_as_void(self, ts):
        """Should parse () as Void."""
        result = ts.parse_type_annotation("()")
        assert result == SWIFT_VOID


class TestSwiftFunctionTypes:
    """Tests for Swift function type parsing."""

    @pytest.fixture
    def ts(self):
        return SwiftTypeSystem()

    def test_parse_function_no_params(self, ts):
        """Should parse () -> Void type."""
        result = ts.parse_type_annotation("() -> Void")
        assert isinstance(result, SwiftFunctionType)
        assert result.parameters == ()
        assert result.return_type == SWIFT_VOID

    def test_parse_function_with_param(self, ts):
        """Should parse (Int) -> String type."""
        result = ts.parse_type_annotation("(Int) -> String")
        assert isinstance(result, SwiftFunctionType)
        assert len(result.parameters) == 1
        assert result.parameters[0] == SWIFT_INT
        assert result.return_type == SWIFT_STRING

    def test_parse_function_multiple_params(self, ts):
        """Should parse (Int, String) -> Bool type."""
        result = ts.parse_type_annotation("(Int, String) -> Bool")
        assert isinstance(result, SwiftFunctionType)
        assert len(result.parameters) == 2
        assert result.parameters[0] == SWIFT_INT
        assert result.parameters[1] == SWIFT_STRING
        assert result.return_type == SWIFT_BOOL

    def test_parse_async_function(self, ts):
        """Should parse async () -> Int type."""
        result = ts.parse_type_annotation("async () -> Int")
        assert isinstance(result, SwiftFunctionType)
        assert result.async_ is True
        assert result.return_type == SWIFT_INT


class TestSwiftProtocolTypes:
    """Tests for Swift protocol type parsing."""

    @pytest.fixture
    def ts(self):
        return SwiftTypeSystem()

    def test_parse_protocol_composition(self, ts):
        """Should parse protocol composition."""
        result = ts.parse_type_annotation("Equatable & Hashable")
        assert isinstance(result, SwiftProtocolCompositionType)
        assert len(result.protocols) == 2

    def test_parse_existential_type(self, ts):
        """Should parse any Protocol type."""
        result = ts.parse_type_annotation("any Equatable")
        assert isinstance(result, SwiftExistentialType)

    def test_parse_opaque_type(self, ts):
        """Should parse some Protocol type."""
        result = ts.parse_type_annotation("some View")
        assert isinstance(result, SwiftOpaqueType)


class TestSwiftGenericTypes:
    """Tests for Swift generic type parsing."""

    @pytest.fixture
    def ts(self):
        return SwiftTypeSystem()

    def test_parse_result_type(self, ts):
        """Should parse Result<Int, Error> type."""
        result = ts.parse_type_annotation("Result<Int, Error>")
        assert isinstance(result, SwiftResultType)
        assert result.success == SWIFT_INT

    def test_parse_generic_type(self, ts):
        """Should parse generic type."""
        result = ts.parse_type_annotation("Publisher<Int, Error>")
        assert isinstance(result, SwiftGenericType)
        assert result.base == "Publisher"
        assert len(result.type_args) == 2


class TestSwiftMetatypeTypes:
    """Tests for Swift metatype parsing."""

    @pytest.fixture
    def ts(self):
        return SwiftTypeSystem()

    def test_parse_metatype(self, ts):
        """Should parse Int.Type metatype."""
        result = ts.parse_type_annotation("Int.Type")
        assert isinstance(result, SwiftMetatypeType)
        assert result.instance_type == SWIFT_INT
        assert result.is_protocol is False

    def test_parse_protocol_metatype(self, ts):
        """Should parse Protocol.Protocol metatype."""
        # Parse a named type first
        result = ts.parse_type_annotation("Equatable.Protocol")
        assert isinstance(result, SwiftMetatypeType)
        assert result.is_protocol is True


class TestSwiftAssignability:
    """Tests for Swift type assignability."""

    @pytest.fixture
    def ts(self):
        return SwiftTypeSystem()

    def test_same_type_assignable(self, ts):
        """Same types should be assignable."""
        assert ts.check_assignable(SWIFT_INT, SWIFT_INT)
        assert ts.check_assignable(SWIFT_STRING, SWIFT_STRING)
        assert ts.check_assignable(SWIFT_BOOL, SWIFT_BOOL)

    def test_different_primitives_not_assignable(self, ts):
        """Different primitives should not be assignable."""
        assert not ts.check_assignable(SWIFT_INT, SWIFT_STRING)
        assert not ts.check_assignable(SWIFT_DOUBLE, SWIFT_INT)

    def test_any_accepts_all(self, ts):
        """Any should accept all types."""
        assert ts.check_assignable(SWIFT_INT, SWIFT_ANY)
        assert ts.check_assignable(SWIFT_STRING, SWIFT_ANY)

    def test_never_to_all(self, ts):
        """Never should be assignable to all types."""
        assert ts.check_assignable(SWIFT_NEVER, SWIFT_INT)
        assert ts.check_assignable(SWIFT_NEVER, SWIFT_STRING)

    def test_non_optional_to_optional(self, ts):
        """Non-optional should be assignable to optional."""
        optional = SwiftOptionalType(SWIFT_INT)
        assert ts.check_assignable(SWIFT_INT, optional)

    def test_optional_to_optional(self, ts):
        """Optional should be assignable to same optional."""
        optional1 = SwiftOptionalType(SWIFT_INT)
        optional2 = SwiftOptionalType(SWIFT_INT)
        assert ts.check_assignable(optional1, optional2)

    def test_array_covariance(self, ts):
        """Arrays should be covariant in Swift."""
        # Never array should be assignable to Int array
        arr_never = SwiftArrayType(SWIFT_NEVER)
        arr_int = SwiftArrayType(SWIFT_INT)
        assert ts.check_assignable(arr_never, arr_int)

    def test_dictionary_covariance(self, ts):
        """Dictionaries should be covariant."""
        dict1 = SwiftDictionaryType(SWIFT_STRING, SWIFT_INT)
        dict2 = SwiftDictionaryType(SWIFT_STRING, SWIFT_INT)
        assert ts.check_assignable(dict1, dict2)


class TestSwiftTypeFormatting:
    """Tests for Swift type formatting."""

    @pytest.fixture
    def ts(self):
        return SwiftTypeSystem()

    def test_format_primitive(self, ts):
        """Should format primitive types."""
        assert ts.format_type(SWIFT_INT) == "Int"
        assert ts.format_type(SWIFT_STRING) == "String"

    def test_format_optional(self, ts):
        """Should format optional type."""
        optional = SwiftOptionalType(SWIFT_INT)
        assert ts.format_type(optional) == "Int?"

    def test_format_array(self, ts):
        """Should format array type."""
        arr = SwiftArrayType(SWIFT_INT)
        assert ts.format_type(arr) == "[Int]"

    def test_format_dictionary(self, ts):
        """Should format dictionary type."""
        d = SwiftDictionaryType(SWIFT_STRING, SWIFT_INT)
        assert ts.format_type(d) == "[String: Int]"

    def test_format_function(self, ts):
        """Should format function type."""
        fn = SwiftFunctionType((SWIFT_INT,), SWIFT_STRING)
        assert ts.format_type(fn) == "(Int) -> String"


class TestSwiftLiteralInference:
    """Tests for Swift literal type inference."""

    @pytest.fixture
    def ts(self):
        return SwiftTypeSystem()

    def test_infer_int_literal(self, ts):
        """Should infer Int for integer literals."""
        lit = LiteralInfo(kind=LiteralKind.INTEGER, value=42)
        result = ts.infer_literal_type(lit)
        assert result == SWIFT_INT

    def test_infer_float_literal(self, ts):
        """Should infer Double for float literals."""
        lit = LiteralInfo(kind=LiteralKind.FLOAT, value=3.14)
        result = ts.infer_literal_type(lit)
        assert result == SWIFT_DOUBLE

    def test_infer_string_literal(self, ts):
        """Should infer String for string literals."""
        lit = LiteralInfo(kind=LiteralKind.STRING, value="hello")
        result = ts.infer_literal_type(lit)
        assert result == SWIFT_STRING

    def test_infer_boolean_literal(self, ts):
        """Should infer Bool for boolean literals."""
        lit = LiteralInfo(kind=LiteralKind.BOOLEAN, value=True)
        result = ts.infer_literal_type(lit)
        assert result == SWIFT_BOOL

    def test_infer_nil_literal(self, ts):
        """Should infer optional Any for nil."""
        lit = LiteralInfo(kind=LiteralKind.NONE, value=None)
        result = ts.infer_literal_type(lit)
        assert isinstance(result, SwiftOptionalType)

    def test_infer_char_literal(self, ts):
        """Should infer Character for char literals."""
        lit = LiteralInfo(kind=LiteralKind.CHARACTER, value='a')
        result = ts.infer_literal_type(lit)
        assert result == SWIFT_CHARACTER


class TestSwiftBuiltinFunctions:
    """Tests for Swift builtin functions."""

    @pytest.fixture
    def ts(self):
        return SwiftTypeSystem()

    def test_get_builtins(self, ts):
        """Should return builtin functions."""
        builtins = ts.get_builtin_functions()
        assert "print" in builtins
        assert "abs" in builtins
        assert "min" in builtins
        assert "max" in builtins

    def test_get_builtin_types(self, ts):
        """Should return builtin types."""
        types = ts.get_builtin_types()
        assert "Int" in types
        assert "String" in types
        assert "Bool" in types
        assert "Any" in types


class TestSwiftLUBGLB:
    """Tests for Swift LUB and GLB operations."""

    @pytest.fixture
    def ts(self):
        return SwiftTypeSystem()

    def test_lub_same_types(self, ts):
        """LUB of same types is that type."""
        result = ts.lub([SWIFT_INT, SWIFT_INT])
        assert result == SWIFT_INT

    def test_lub_different_types(self, ts):
        """LUB of different types is Any."""
        result = ts.lub([SWIFT_INT, SWIFT_STRING])
        assert result == SWIFT_ANY

    def test_lub_with_optional(self, ts):
        """LUB with optional types is optional Any."""
        optional = SwiftOptionalType(SWIFT_INT)
        result = ts.lub([optional, SWIFT_STRING])
        assert isinstance(result, SwiftOptionalType)

    def test_glb_same_types(self, ts):
        """GLB of same types is that type."""
        result = ts.glb([SWIFT_INT, SWIFT_INT])
        assert result == SWIFT_INT

    def test_glb_different_types(self, ts):
        """GLB of different types is Never."""
        result = ts.glb([SWIFT_INT, SWIFT_STRING])
        assert result == SWIFT_NEVER
