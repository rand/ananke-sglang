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
"""Tests for Kotlin type system."""

import pytest

from domains.types.languages.kotlin import (
    KotlinTypeSystem,
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
    KOTLIN_BYTE,
    KOTLIN_SHORT,
    KOTLIN_INT,
    KOTLIN_LONG,
    KOTLIN_FLOAT,
    KOTLIN_DOUBLE,
    KOTLIN_UBYTE,
    KOTLIN_USHORT,
    KOTLIN_UINT,
    KOTLIN_ULONG,
    KOTLIN_BOOLEAN,
    KOTLIN_CHAR,
    KOTLIN_STRING,
    KOTLIN_UNIT,
    KOTLIN_ANY,
    KOTLIN_NOTHING,
)
from domains.types.languages.base import LiteralInfo, LiteralKind


class TestKotlinTypeSystemCreation:
    """Tests for KotlinTypeSystem creation."""

    def test_create_type_system(self):
        """Should create a Kotlin type system."""
        ts = KotlinTypeSystem()
        assert ts.name == "kotlin"

    def test_capabilities(self):
        """Should report correct capabilities."""
        ts = KotlinTypeSystem()
        caps = ts.capabilities

        assert caps.supports_generics is True
        assert caps.supports_optional_types is True
        assert caps.supports_protocols is True  # interfaces
        assert caps.supports_variance is True
        assert caps.supports_overloading is True
        assert caps.supports_union_types is False


class TestKotlinPrimitiveTypes:
    """Tests for Kotlin primitive type parsing."""

    @pytest.fixture
    def ts(self):
        return KotlinTypeSystem()

    def test_parse_int(self, ts):
        """Should parse Int type."""
        result = ts.parse_type_annotation("Int")
        assert result == KOTLIN_INT

    def test_parse_long(self, ts):
        """Should parse Long type."""
        result = ts.parse_type_annotation("Long")
        assert result == KOTLIN_LONG

    def test_parse_short(self, ts):
        """Should parse Short type."""
        result = ts.parse_type_annotation("Short")
        assert result == KOTLIN_SHORT

    def test_parse_byte(self, ts):
        """Should parse Byte type."""
        result = ts.parse_type_annotation("Byte")
        assert result == KOTLIN_BYTE

    def test_parse_float(self, ts):
        """Should parse Float type."""
        result = ts.parse_type_annotation("Float")
        assert result == KOTLIN_FLOAT

    def test_parse_double(self, ts):
        """Should parse Double type."""
        result = ts.parse_type_annotation("Double")
        assert result == KOTLIN_DOUBLE

    def test_parse_boolean(self, ts):
        """Should parse Boolean type."""
        result = ts.parse_type_annotation("Boolean")
        assert result == KOTLIN_BOOLEAN

    def test_parse_char(self, ts):
        """Should parse Char type."""
        result = ts.parse_type_annotation("Char")
        assert result == KOTLIN_CHAR

    def test_parse_string(self, ts):
        """Should parse String type."""
        result = ts.parse_type_annotation("String")
        assert result == KOTLIN_STRING

    def test_parse_unit(self, ts):
        """Should parse Unit type."""
        result = ts.parse_type_annotation("Unit")
        assert result == KOTLIN_UNIT

    def test_parse_any(self, ts):
        """Should parse Any type."""
        result = ts.parse_type_annotation("Any")
        assert result == KOTLIN_ANY

    def test_parse_nothing(self, ts):
        """Should parse Nothing type."""
        result = ts.parse_type_annotation("Nothing")
        assert result == KOTLIN_NOTHING

    def test_parse_unsigned_types(self, ts):
        """Should parse unsigned types."""
        assert ts.parse_type_annotation("UInt") == KOTLIN_UINT
        assert ts.parse_type_annotation("ULong") == KOTLIN_ULONG
        assert ts.parse_type_annotation("UShort") == KOTLIN_USHORT
        assert ts.parse_type_annotation("UByte") == KOTLIN_UBYTE


class TestKotlinNullableTypes:
    """Tests for Kotlin nullable type parsing."""

    @pytest.fixture
    def ts(self):
        return KotlinTypeSystem()

    def test_parse_nullable_int(self, ts):
        """Should parse Int? type."""
        result = ts.parse_type_annotation("Int?")
        assert isinstance(result, KotlinNullableType)
        assert result.inner == KOTLIN_INT

    def test_parse_nullable_string(self, ts):
        """Should parse String? type."""
        result = ts.parse_type_annotation("String?")
        assert isinstance(result, KotlinNullableType)
        assert result.inner == KOTLIN_STRING

    def test_nullable_formatting(self, ts):
        """Should format nullable types correctly."""
        result = ts.parse_type_annotation("Int?")
        assert ts.format_type(result) == "Int?"


class TestKotlinArrayTypes:
    """Tests for Kotlin array type parsing."""

    @pytest.fixture
    def ts(self):
        return KotlinTypeSystem()

    def test_parse_generic_array(self, ts):
        """Should parse Array<Int> type."""
        result = ts.parse_type_annotation("Array<Int>")
        assert isinstance(result, KotlinArrayType)
        assert result.element == KOTLIN_INT

    def test_parse_primitive_int_array(self, ts):
        """Should parse IntArray type."""
        result = ts.parse_type_annotation("IntArray")
        assert isinstance(result, KotlinPrimitiveArrayType)
        assert result.element_type == "Int"

    def test_parse_primitive_long_array(self, ts):
        """Should parse LongArray type."""
        result = ts.parse_type_annotation("LongArray")
        assert isinstance(result, KotlinPrimitiveArrayType)
        assert result.element_type == "Long"

    def test_parse_primitive_byte_array(self, ts):
        """Should parse ByteArray type."""
        result = ts.parse_type_annotation("ByteArray")
        assert isinstance(result, KotlinPrimitiveArrayType)
        assert result.element_type == "Byte"

    def test_array_formatting(self, ts):
        """Should format array types correctly."""
        result = ts.parse_type_annotation("Array<String>")
        assert ts.format_type(result) == "Array<String>"


class TestKotlinCollectionTypes:
    """Tests for Kotlin collection type parsing."""

    @pytest.fixture
    def ts(self):
        return KotlinTypeSystem()

    def test_parse_list(self, ts):
        """Should parse List<Int> type."""
        result = ts.parse_type_annotation("List<Int>")
        assert isinstance(result, KotlinListType)
        assert result.element == KOTLIN_INT
        assert result.mutable is False

    def test_parse_mutable_list(self, ts):
        """Should parse MutableList<Int> type."""
        result = ts.parse_type_annotation("MutableList<Int>")
        assert isinstance(result, KotlinListType)
        assert result.element == KOTLIN_INT
        assert result.mutable is True

    def test_parse_set(self, ts):
        """Should parse Set<String> type."""
        result = ts.parse_type_annotation("Set<String>")
        assert isinstance(result, KotlinSetType)
        assert result.element == KOTLIN_STRING
        assert result.mutable is False

    def test_parse_mutable_set(self, ts):
        """Should parse MutableSet<String> type."""
        result = ts.parse_type_annotation("MutableSet<String>")
        assert isinstance(result, KotlinSetType)
        assert result.element == KOTLIN_STRING
        assert result.mutable is True

    def test_parse_map(self, ts):
        """Should parse Map<String, Int> type."""
        result = ts.parse_type_annotation("Map<String, Int>")
        assert isinstance(result, KotlinMapType)
        assert result.key == KOTLIN_STRING
        assert result.value == KOTLIN_INT
        assert result.mutable is False

    def test_parse_mutable_map(self, ts):
        """Should parse MutableMap<String, Int> type."""
        result = ts.parse_type_annotation("MutableMap<String, Int>")
        assert isinstance(result, KotlinMapType)
        assert result.key == KOTLIN_STRING
        assert result.value == KOTLIN_INT
        assert result.mutable is True

    def test_parse_pair(self, ts):
        """Should parse Pair<Int, String> type."""
        result = ts.parse_type_annotation("Pair<Int, String>")
        assert isinstance(result, KotlinPairType)
        assert result.first == KOTLIN_INT
        assert result.second == KOTLIN_STRING


class TestKotlinFunctionTypes:
    """Tests for Kotlin function type parsing."""

    @pytest.fixture
    def ts(self):
        return KotlinTypeSystem()

    def test_parse_function_no_params(self, ts):
        """Should parse () -> Unit type."""
        result = ts.parse_type_annotation("() -> Unit")
        assert isinstance(result, KotlinFunctionType)
        assert result.parameters == ()
        assert result.return_type == KOTLIN_UNIT

    def test_parse_function_with_param(self, ts):
        """Should parse (Int) -> String type."""
        result = ts.parse_type_annotation("(Int) -> String")
        assert isinstance(result, KotlinFunctionType)
        assert len(result.parameters) == 1
        assert result.parameters[0] == KOTLIN_INT
        assert result.return_type == KOTLIN_STRING

    def test_parse_function_multiple_params(self, ts):
        """Should parse (Int, String) -> Boolean type."""
        result = ts.parse_type_annotation("(Int, String) -> Boolean")
        assert isinstance(result, KotlinFunctionType)
        assert len(result.parameters) == 2
        assert result.parameters[0] == KOTLIN_INT
        assert result.parameters[1] == KOTLIN_STRING
        assert result.return_type == KOTLIN_BOOLEAN

    def test_parse_suspend_function(self, ts):
        """Should parse suspend () -> Int type."""
        result = ts.parse_type_annotation("suspend () -> Int")
        assert isinstance(result, KotlinFunctionType)
        assert result.is_suspend is True
        assert result.return_type == KOTLIN_INT


class TestKotlinGenericTypes:
    """Tests for Kotlin generic type parsing."""

    @pytest.fixture
    def ts(self):
        return KotlinTypeSystem()

    def test_parse_generic_type(self, ts):
        """Should parse generic type."""
        result = ts.parse_type_annotation("Result<Int, Exception>")
        assert isinstance(result, KotlinGenericType)
        assert result.base == "Result"
        assert len(result.type_args) == 2

    def test_parse_star_projection(self, ts):
        """Should parse star projection."""
        result = ts.parse_type_annotation("*")
        assert isinstance(result, KotlinStarProjection)


class TestKotlinAssignability:
    """Tests for Kotlin type assignability."""

    @pytest.fixture
    def ts(self):
        return KotlinTypeSystem()

    def test_same_type_assignable(self, ts):
        """Same types should be assignable."""
        assert ts.check_assignable(KOTLIN_INT, KOTLIN_INT)
        assert ts.check_assignable(KOTLIN_STRING, KOTLIN_STRING)
        assert ts.check_assignable(KOTLIN_BOOLEAN, KOTLIN_BOOLEAN)

    def test_different_primitives_not_assignable(self, ts):
        """Different primitives should not be assignable (Kotlin is strict)."""
        assert not ts.check_assignable(KOTLIN_INT, KOTLIN_STRING)
        assert not ts.check_assignable(KOTLIN_INT, KOTLIN_LONG)

    def test_any_accepts_all(self, ts):
        """Any should accept all types."""
        assert ts.check_assignable(KOTLIN_INT, KOTLIN_ANY)
        assert ts.check_assignable(KOTLIN_STRING, KOTLIN_ANY)

    def test_nothing_to_all(self, ts):
        """Nothing should be assignable to all types."""
        assert ts.check_assignable(KOTLIN_NOTHING, KOTLIN_INT)
        assert ts.check_assignable(KOTLIN_NOTHING, KOTLIN_STRING)

    def test_non_nullable_to_nullable(self, ts):
        """Non-nullable should be assignable to nullable."""
        nullable = KotlinNullableType(KOTLIN_INT)
        assert ts.check_assignable(KOTLIN_INT, nullable)

    def test_nullable_to_nullable(self, ts):
        """Nullable should be assignable to same nullable."""
        nullable1 = KotlinNullableType(KOTLIN_INT)
        nullable2 = KotlinNullableType(KOTLIN_INT)
        assert ts.check_assignable(nullable1, nullable2)

    def test_nullable_not_to_non_nullable(self, ts):
        """Nullable should NOT be assignable to non-nullable."""
        nullable = KotlinNullableType(KOTLIN_INT)
        assert not ts.check_assignable(nullable, KOTLIN_INT)

    def test_list_covariance(self, ts):
        """Immutable List should be covariant."""
        # List<Nothing> should be assignable to List<Int>
        list_nothing = KotlinListType(KOTLIN_NOTHING, mutable=False)
        list_int = KotlinListType(KOTLIN_INT, mutable=False)
        assert ts.check_assignable(list_nothing, list_int)

    def test_mutable_list_invariance(self, ts):
        """Mutable List should be invariant."""
        mlist1 = KotlinListType(KOTLIN_INT, mutable=True)
        mlist2 = KotlinListType(KOTLIN_INT, mutable=True)
        assert ts.check_assignable(mlist1, mlist2)

    def test_array_invariance(self, ts):
        """Arrays should be invariant."""
        arr1 = KotlinArrayType(KOTLIN_INT)
        arr2 = KotlinArrayType(KOTLIN_INT)
        assert ts.check_assignable(arr1, arr2)


class TestKotlinTypeFormatting:
    """Tests for Kotlin type formatting."""

    @pytest.fixture
    def ts(self):
        return KotlinTypeSystem()

    def test_format_primitive(self, ts):
        """Should format primitive types."""
        assert ts.format_type(KOTLIN_INT) == "Int"
        assert ts.format_type(KOTLIN_STRING) == "String"

    def test_format_nullable(self, ts):
        """Should format nullable type."""
        nullable = KotlinNullableType(KOTLIN_INT)
        assert ts.format_type(nullable) == "Int?"

    def test_format_list(self, ts):
        """Should format list type."""
        lst = KotlinListType(KOTLIN_INT, mutable=False)
        assert ts.format_type(lst) == "List<Int>"

    def test_format_mutable_list(self, ts):
        """Should format mutable list type."""
        lst = KotlinListType(KOTLIN_INT, mutable=True)
        assert ts.format_type(lst) == "MutableList<Int>"

    def test_format_map(self, ts):
        """Should format map type."""
        m = KotlinMapType(KOTLIN_STRING, KOTLIN_INT, mutable=False)
        assert ts.format_type(m) == "Map<String, Int>"

    def test_format_function(self, ts):
        """Should format function type."""
        fn = KotlinFunctionType((KOTLIN_INT,), KOTLIN_STRING)
        assert ts.format_type(fn) == "(Int) -> String"


class TestKotlinLiteralInference:
    """Tests for Kotlin literal type inference."""

    @pytest.fixture
    def ts(self):
        return KotlinTypeSystem()

    def test_infer_int_literal(self, ts):
        """Should infer Int for small integer literals."""
        lit = LiteralInfo(kind=LiteralKind.INTEGER, value=42)
        result = ts.infer_literal_type(lit)
        assert result == KOTLIN_INT

    def test_infer_long_literal(self, ts):
        """Should infer Long for large integer literals."""
        lit = LiteralInfo(kind=LiteralKind.INTEGER, value=2**32)
        result = ts.infer_literal_type(lit)
        assert result == KOTLIN_LONG

    def test_infer_float_literal(self, ts):
        """Should infer Double for float literals."""
        lit = LiteralInfo(kind=LiteralKind.FLOAT, value=3.14)
        result = ts.infer_literal_type(lit)
        assert result == KOTLIN_DOUBLE

    def test_infer_string_literal(self, ts):
        """Should infer String for string literals."""
        lit = LiteralInfo(kind=LiteralKind.STRING, value="hello")
        result = ts.infer_literal_type(lit)
        assert result == KOTLIN_STRING

    def test_infer_boolean_literal(self, ts):
        """Should infer Boolean for boolean literals."""
        lit = LiteralInfo(kind=LiteralKind.BOOLEAN, value=True)
        result = ts.infer_literal_type(lit)
        assert result == KOTLIN_BOOLEAN

    def test_infer_null_literal(self, ts):
        """Should infer Nothing? for null."""
        lit = LiteralInfo(kind=LiteralKind.NONE, value=None)
        result = ts.infer_literal_type(lit)
        assert isinstance(result, KotlinNullableType)
        assert result.inner == KOTLIN_NOTHING

    def test_infer_char_literal(self, ts):
        """Should infer Char for char literals."""
        lit = LiteralInfo(kind=LiteralKind.CHARACTER, value='a')
        result = ts.infer_literal_type(lit)
        assert result == KOTLIN_CHAR


class TestKotlinBuiltinFunctions:
    """Tests for Kotlin builtin functions."""

    @pytest.fixture
    def ts(self):
        return KotlinTypeSystem()

    def test_get_builtins(self, ts):
        """Should return builtin functions."""
        builtins = ts.get_builtin_functions()
        assert "println" in builtins
        assert "print" in builtins
        assert "listOf" in builtins
        assert "mapOf" in builtins

    def test_get_builtin_types(self, ts):
        """Should return builtin types."""
        types = ts.get_builtin_types()
        assert "Int" in types
        assert "String" in types
        assert "Boolean" in types
        assert "Any" in types


class TestKotlinLUBGLB:
    """Tests for Kotlin LUB and GLB operations."""

    @pytest.fixture
    def ts(self):
        return KotlinTypeSystem()

    def test_lub_same_types(self, ts):
        """LUB of same types is that type."""
        result = ts.lub([KOTLIN_INT, KOTLIN_INT])
        assert result == KOTLIN_INT

    def test_lub_different_types(self, ts):
        """LUB of different types is Any."""
        result = ts.lub([KOTLIN_INT, KOTLIN_STRING])
        assert result == KOTLIN_ANY

    def test_lub_with_nullable(self, ts):
        """LUB with nullable types is nullable Any."""
        nullable = KotlinNullableType(KOTLIN_INT)
        result = ts.lub([nullable, KOTLIN_STRING])
        assert isinstance(result, KotlinNullableType)

    def test_glb_same_types(self, ts):
        """GLB of same types is that type."""
        result = ts.glb([KOTLIN_INT, KOTLIN_INT])
        assert result == KOTLIN_INT

    def test_glb_different_types(self, ts):
        """GLB of different types is Nothing."""
        result = ts.glb([KOTLIN_INT, KOTLIN_STRING])
        assert result == KOTLIN_NOTHING
