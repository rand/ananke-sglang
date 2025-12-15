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
"""Unit tests for TypeScript type system.

Tests for the TypeScriptTypeSystem implementation including:
- Primitive type parsing
- Literal types
- Array and tuple types
- Object types
- Function types
- Union and intersection types
- Generic types
- Utility types
- Structural typing (assignability)
- any/unknown/never handling
"""

import pytest

from domains.types.constraint import ANY, NEVER
from domains.types.languages import (
    get_type_system,
    supported_languages,
    TypeParseError,
    LiteralInfo,
    LiteralKind,
)
from domains.types.languages.typescript import (
    TypeScriptTypeSystem,
    # Helper types
    TSParameter,
    # Type classes
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
    # Primitive constants
    TS_STRING,
    TS_NUMBER,
    TS_BOOLEAN,
    TS_BIGINT,
    TS_SYMBOL,
    TS_UNDEFINED,
    TS_NULL,
    TS_VOID,
    TS_OBJECT,
    TS_UNKNOWN,
    TS_ANY,
    TS_NEVER,
)


# ===========================================================================
# Factory Function Tests
# ===========================================================================


class TestGetTypeScriptTypeSystem:
    """Tests for get_type_system with TypeScript."""

    def test_get_typescript_by_name(self):
        """Should return TypeScript type system."""
        ts = get_type_system("typescript")
        assert isinstance(ts, TypeScriptTypeSystem)
        assert ts.name == "typescript"

    def test_get_typescript_by_alias_ts(self):
        """Should accept 'ts' as alias."""
        ts = get_type_system("ts")
        assert isinstance(ts, TypeScriptTypeSystem)

    def test_get_typescript_by_alias_javascript(self):
        """Should accept 'javascript' as alias."""
        ts = get_type_system("javascript")
        assert isinstance(ts, TypeScriptTypeSystem)

    def test_get_typescript_by_alias_js(self):
        """Should accept 'js' as alias."""
        ts = get_type_system("js")
        assert isinstance(ts, TypeScriptTypeSystem)

    def test_typescript_in_supported_languages(self):
        """TypeScript should be in supported languages."""
        langs = supported_languages()
        assert "typescript" in langs


# ===========================================================================
# TypeScript Type System Basic Tests
# ===========================================================================


class TestTypeScriptTypeSystemBasics:
    """Basic tests for TypeScriptTypeSystem."""

    @pytest.fixture
    def ts(self):
        """Create a TypeScript type system instance."""
        return TypeScriptTypeSystem()

    def test_name(self, ts):
        """Name should be 'typescript'."""
        assert ts.name == "typescript"

    def test_capabilities(self, ts):
        """Should have correct capabilities."""
        caps = ts.capabilities
        assert caps.supports_generics
        assert caps.supports_union_types
        assert caps.supports_optional_types
        assert caps.supports_type_inference
        assert caps.supports_protocols  # Structural typing
        assert caps.supports_variance
        assert caps.supports_overloading
        # TypeScript doesn't have these
        assert not caps.supports_ownership
        assert not caps.supports_comptime
        assert not caps.supports_error_unions


# ===========================================================================
# Primitive Type Parsing Tests
# ===========================================================================


class TestTypeScriptPrimitiveTypeParsing:
    """Tests for parsing TypeScript primitive types."""

    @pytest.fixture
    def ts(self):
        return TypeScriptTypeSystem()

    def test_parse_string(self, ts):
        """Should parse 'string'."""
        typ = ts.parse_type_annotation("string")
        assert typ == TS_STRING

    def test_parse_number(self, ts):
        """Should parse 'number'."""
        typ = ts.parse_type_annotation("number")
        assert typ == TS_NUMBER

    def test_parse_boolean(self, ts):
        """Should parse 'boolean'."""
        typ = ts.parse_type_annotation("boolean")
        assert typ == TS_BOOLEAN

    def test_parse_bigint(self, ts):
        """Should parse 'bigint'."""
        typ = ts.parse_type_annotation("bigint")
        assert typ == TS_BIGINT

    def test_parse_symbol(self, ts):
        """Should parse 'symbol'."""
        typ = ts.parse_type_annotation("symbol")
        assert typ == TS_SYMBOL

    def test_parse_undefined(self, ts):
        """Should parse 'undefined'."""
        typ = ts.parse_type_annotation("undefined")
        assert typ == TS_UNDEFINED

    def test_parse_null(self, ts):
        """Should parse 'null'."""
        typ = ts.parse_type_annotation("null")
        assert typ == TS_NULL

    def test_parse_void(self, ts):
        """Should parse 'void'."""
        typ = ts.parse_type_annotation("void")
        assert typ == TS_VOID

    def test_parse_object(self, ts):
        """Should parse 'object'."""
        typ = ts.parse_type_annotation("object")
        assert typ == TS_OBJECT

    def test_parse_any(self, ts):
        """Should parse 'any'."""
        typ = ts.parse_type_annotation("any")
        assert typ == TS_ANY

    def test_parse_unknown(self, ts):
        """Should parse 'unknown'."""
        typ = ts.parse_type_annotation("unknown")
        assert typ == TS_UNKNOWN

    def test_parse_never(self, ts):
        """Should parse 'never'."""
        typ = ts.parse_type_annotation("never")
        assert typ == TS_NEVER


# ===========================================================================
# Literal Type Parsing Tests
# ===========================================================================


class TestTypeScriptLiteralTypeParsing:
    """Tests for parsing TypeScript literal types."""

    @pytest.fixture
    def ts(self):
        return TypeScriptTypeSystem()

    def test_parse_string_literal_single(self, ts):
        """Should parse single-quoted string literal."""
        typ = ts.parse_type_annotation("'hello'")
        assert isinstance(typ, TSLiteralType)
        assert typ.kind == "string"
        assert typ.value == "hello"

    def test_parse_string_literal_double(self, ts):
        """Should parse double-quoted string literal."""
        typ = ts.parse_type_annotation('"world"')
        assert isinstance(typ, TSLiteralType)
        assert typ.kind == "string"
        assert typ.value == "world"

    def test_parse_number_literal_integer(self, ts):
        """Should parse integer literal."""
        typ = ts.parse_type_annotation("42")
        assert isinstance(typ, TSLiteralType)
        assert typ.kind == "number"
        assert typ.value == 42

    def test_parse_number_literal_float(self, ts):
        """Should parse float literal."""
        typ = ts.parse_type_annotation("3.14")
        assert isinstance(typ, TSLiteralType)
        assert typ.kind == "number"
        assert typ.value == 3.14

    def test_parse_boolean_literal_true(self, ts):
        """Should parse true literal."""
        typ = ts.parse_type_annotation("true")
        assert isinstance(typ, TSLiteralType)
        assert typ.kind == "boolean"
        assert typ.value is True

    def test_parse_boolean_literal_false(self, ts):
        """Should parse false literal."""
        typ = ts.parse_type_annotation("false")
        assert isinstance(typ, TSLiteralType)
        assert typ.kind == "boolean"
        assert typ.value is False


# ===========================================================================
# Array Type Parsing Tests
# ===========================================================================


class TestTypeScriptArrayTypeParsing:
    """Tests for parsing TypeScript array types."""

    @pytest.fixture
    def ts(self):
        return TypeScriptTypeSystem()

    def test_parse_array_bracket_syntax(self, ts):
        """Should parse T[] syntax."""
        typ = ts.parse_type_annotation("string[]")
        assert isinstance(typ, TSArrayType)
        assert typ.element == TS_STRING

    def test_parse_array_generic_syntax(self, ts):
        """Should parse Array<T> syntax."""
        typ = ts.parse_type_annotation("Array<number>")
        assert isinstance(typ, TSArrayType)
        assert typ.element == TS_NUMBER

    def test_parse_readonly_array(self, ts):
        """Should parse ReadonlyArray<T>."""
        typ = ts.parse_type_annotation("ReadonlyArray<boolean>")
        assert isinstance(typ, TSArrayType)
        assert typ.element == TS_BOOLEAN
        assert typ.is_readonly

    def test_parse_nested_array(self, ts):
        """Should parse nested arrays."""
        typ = ts.parse_type_annotation("string[][]")
        assert isinstance(typ, TSArrayType)
        assert isinstance(typ.element, TSArrayType)
        assert typ.element.element == TS_STRING


# ===========================================================================
# Tuple Type Parsing Tests
# ===========================================================================


class TestTypeScriptTupleTypeParsing:
    """Tests for parsing TypeScript tuple types."""

    @pytest.fixture
    def ts(self):
        return TypeScriptTypeSystem()

    def test_parse_simple_tuple(self, ts):
        """Should parse simple tuple."""
        typ = ts.parse_type_annotation("[string, number]")
        assert isinstance(typ, TSTupleType)
        assert len(typ.elements) == 2
        assert typ.elements[0] == TS_STRING
        assert typ.elements[1] == TS_NUMBER

    def test_parse_single_element_tuple(self, ts):
        """Should parse single-element tuple."""
        typ = ts.parse_type_annotation("[string]")
        assert isinstance(typ, TSTupleType)
        assert len(typ.elements) == 1
        assert typ.elements[0] == TS_STRING

    def test_parse_three_element_tuple(self, ts):
        """Should parse three-element tuple."""
        typ = ts.parse_type_annotation("[string, number, boolean]")
        assert isinstance(typ, TSTupleType)
        assert len(typ.elements) == 3

    def test_parse_rest_tuple(self, ts):
        """Should parse tuple with rest element."""
        typ = ts.parse_type_annotation("[string, ...number[]]")
        assert isinstance(typ, TSTupleType)
        assert len(typ.elements) == 1
        assert typ.elements[0] == TS_STRING
        assert typ.rest_element == TS_NUMBER


# ===========================================================================
# Object Type Parsing Tests
# ===========================================================================


class TestTypeScriptObjectTypeParsing:
    """Tests for parsing TypeScript object types."""

    @pytest.fixture
    def ts(self):
        return TypeScriptTypeSystem()

    def test_parse_empty_object(self, ts):
        """Should parse empty object type."""
        typ = ts.parse_type_annotation("{}")
        assert isinstance(typ, TSObjectType)
        assert len(typ.properties) == 0

    def test_parse_object_single_property(self, ts):
        """Should parse object with single property."""
        typ = ts.parse_type_annotation("{ name: string }")
        assert isinstance(typ, TSObjectType)
        assert typ.get_property("name") == TS_STRING

    def test_parse_object_multiple_properties(self, ts):
        """Should parse object with multiple properties."""
        typ = ts.parse_type_annotation("{ name: string; age: number }")
        assert isinstance(typ, TSObjectType)
        assert typ.get_property("name") == TS_STRING
        assert typ.get_property("age") == TS_NUMBER

    def test_parse_object_optional_property(self, ts):
        """Should parse object with optional property."""
        typ = ts.parse_type_annotation("{ name: string; age?: number }")
        assert isinstance(typ, TSObjectType)
        assert typ.get_property("name") == TS_STRING
        assert "age" in typ.optional_properties

    def test_parse_index_signature(self, ts):
        """Should parse index signature."""
        typ = ts.parse_type_annotation("{ [key: string]: number }")
        assert isinstance(typ, TSObjectType)
        assert typ.index_signature is not None
        key_type, value_type = typ.index_signature
        assert key_type == TS_STRING
        assert value_type == TS_NUMBER


# ===========================================================================
# Function Type Parsing Tests
# ===========================================================================


class TestTypeScriptFunctionTypeParsing:
    """Tests for parsing TypeScript function types."""

    @pytest.fixture
    def ts(self):
        return TypeScriptTypeSystem()

    def test_parse_simple_function(self, ts):
        """Should parse simple function type."""
        typ = ts.parse_type_annotation("(x: number) => string")
        assert isinstance(typ, TSFunctionType)
        assert len(typ.parameters) == 1
        # Parameters are (name, type, optional) tuples
        assert typ.parameters[0][0] == "x"
        assert typ.parameters[0][1] == TS_NUMBER
        assert typ.return_type == TS_STRING

    def test_parse_function_no_params(self, ts):
        """Should parse function with no parameters."""
        typ = ts.parse_type_annotation("() => void")
        assert isinstance(typ, TSFunctionType)
        assert len(typ.parameters) == 0
        assert typ.return_type == TS_VOID

    def test_parse_function_multiple_params(self, ts):
        """Should parse function with multiple parameters."""
        typ = ts.parse_type_annotation("(x: number, y: string) => boolean")
        assert isinstance(typ, TSFunctionType)
        assert len(typ.parameters) == 2
        # Parameters are (name, type, optional) tuples
        assert typ.parameters[0][0] == "x"
        assert typ.parameters[1][0] == "y"

    def test_parse_function_optional_param(self, ts):
        """Should parse function with optional parameter."""
        typ = ts.parse_type_annotation("(x: number, y?: string) => void")
        assert isinstance(typ, TSFunctionType)
        # Parameters are (name, type, optional) tuples
        assert not typ.parameters[0][2]  # x is not optional
        assert typ.parameters[1][2]  # y is optional

    def test_parse_function_rest_param(self, ts):
        """Should parse function with rest parameter."""
        typ = ts.parse_type_annotation("(...args: number[]) => void")
        assert isinstance(typ, TSFunctionType)
        assert typ.rest_parameter is not None


# ===========================================================================
# Union Type Parsing Tests
# ===========================================================================


class TestTypeScriptUnionTypeParsing:
    """Tests for parsing TypeScript union types."""

    @pytest.fixture
    def ts(self):
        return TypeScriptTypeSystem()

    def test_parse_simple_union(self, ts):
        """Should parse simple union."""
        typ = ts.parse_type_annotation("string | number")
        assert isinstance(typ, TSUnionType)
        assert TS_STRING in typ.members
        assert TS_NUMBER in typ.members

    def test_parse_three_way_union(self, ts):
        """Should parse three-way union."""
        typ = ts.parse_type_annotation("string | number | boolean")
        assert isinstance(typ, TSUnionType)
        assert len(typ.members) == 3

    def test_parse_nullable_type(self, ts):
        """Should parse nullable type (T | null)."""
        typ = ts.parse_type_annotation("string | null")
        assert isinstance(typ, TSUnionType)
        assert TS_STRING in typ.members
        assert TS_NULL in typ.members

    def test_parse_optional_undefined(self, ts):
        """Should parse optional type (T | undefined)."""
        typ = ts.parse_type_annotation("string | undefined")
        assert isinstance(typ, TSUnionType)
        assert TS_STRING in typ.members
        assert TS_UNDEFINED in typ.members


# ===========================================================================
# Intersection Type Parsing Tests
# ===========================================================================


class TestTypeScriptIntersectionTypeParsing:
    """Tests for parsing TypeScript intersection types."""

    @pytest.fixture
    def ts(self):
        return TypeScriptTypeSystem()

    def test_parse_simple_intersection(self, ts):
        """Should parse simple intersection."""
        typ = ts.parse_type_annotation("{ a: string } & { b: number }")
        assert isinstance(typ, TSIntersectionType)
        assert len(typ.members) == 2


# ===========================================================================
# Generic Type Parsing Tests
# ===========================================================================


class TestTypeScriptGenericTypeParsing:
    """Tests for parsing TypeScript generic types."""

    @pytest.fixture
    def ts(self):
        return TypeScriptTypeSystem()

    def test_parse_promise(self, ts):
        """Should parse Promise<T>."""
        typ = ts.parse_type_annotation("Promise<string>")
        assert isinstance(typ, TSTypeReference)
        assert typ.name == "Promise"
        assert len(typ.type_arguments) == 1

    def test_parse_map(self, ts):
        """Should parse Map<K, V>."""
        typ = ts.parse_type_annotation("Map<string, number>")
        assert isinstance(typ, TSTypeReference)
        assert typ.name == "Map"
        assert len(typ.type_arguments) == 2

    def test_parse_record(self, ts):
        """Should parse Record<K, V>."""
        typ = ts.parse_type_annotation("Record<string, number>")
        assert isinstance(typ, TSTypeReference)
        assert typ.name == "Record"


# ===========================================================================
# Assignability Tests - Basic
# ===========================================================================


class TestTypeScriptAssignabilityBasic:
    """Tests for TypeScript assignability rules - basic cases."""

    @pytest.fixture
    def ts(self):
        return TypeScriptTypeSystem()

    def test_reflexive_string(self, ts):
        """string should be assignable to string."""
        assert ts.check_assignable(TS_STRING, TS_STRING)

    def test_reflexive_number(self, ts):
        """number should be assignable to number."""
        assert ts.check_assignable(TS_NUMBER, TS_NUMBER)

    def test_different_primitives_not_assignable(self, ts):
        """Different primitives should not be assignable."""
        assert not ts.check_assignable(TS_STRING, TS_NUMBER)
        assert not ts.check_assignable(TS_NUMBER, TS_BOOLEAN)


# ===========================================================================
# Assignability Tests - any/unknown/never
# ===========================================================================


class TestTypeScriptAssignabilityAnyUnknownNever:
    """Tests for any/unknown/never assignability."""

    @pytest.fixture
    def ts(self):
        return TypeScriptTypeSystem()

    # any tests
    def test_any_accepts_all(self, ts):
        """any should accept all types."""
        assert ts.check_assignable(TS_STRING, TS_ANY)
        assert ts.check_assignable(TS_NUMBER, TS_ANY)
        assert ts.check_assignable(TS_BOOLEAN, TS_ANY)

    def test_any_assignable_to_all(self, ts):
        """any should be assignable to all types."""
        assert ts.check_assignable(TS_ANY, TS_STRING)
        assert ts.check_assignable(TS_ANY, TS_NUMBER)

    # unknown tests
    def test_unknown_accepts_all(self, ts):
        """unknown should accept all types."""
        assert ts.check_assignable(TS_STRING, TS_UNKNOWN)
        assert ts.check_assignable(TS_NUMBER, TS_UNKNOWN)
        assert ts.check_assignable(TS_ANY, TS_UNKNOWN)

    def test_unknown_not_assignable_without_narrowing(self, ts):
        """unknown should not be assignable to specific types."""
        assert not ts.check_assignable(TS_UNKNOWN, TS_STRING)
        assert not ts.check_assignable(TS_UNKNOWN, TS_NUMBER)

    def test_unknown_assignable_to_any(self, ts):
        """unknown should be assignable to any."""
        assert ts.check_assignable(TS_UNKNOWN, TS_ANY)

    # never tests
    def test_never_assignable_to_all(self, ts):
        """never should be assignable to all types."""
        assert ts.check_assignable(TS_NEVER, TS_STRING)
        assert ts.check_assignable(TS_NEVER, TS_NUMBER)
        assert ts.check_assignable(TS_NEVER, TS_ANY)

    def test_nothing_assignable_to_never(self, ts):
        """No type should be assignable to never."""
        assert not ts.check_assignable(TS_STRING, TS_NEVER)
        assert not ts.check_assignable(TS_NUMBER, TS_NEVER)
        assert not ts.check_assignable(TS_ANY, TS_NEVER)

    def test_never_assignable_to_never(self, ts):
        """never should be assignable to itself."""
        assert ts.check_assignable(TS_NEVER, TS_NEVER)


# ===========================================================================
# Assignability Tests - Union Types
# ===========================================================================


class TestTypeScriptAssignabilityUnion:
    """Tests for union type assignability."""

    @pytest.fixture
    def ts(self):
        return TypeScriptTypeSystem()

    def test_member_assignable_to_union(self, ts):
        """Union member should be assignable to union."""
        union = TSUnionType(frozenset([TS_STRING, TS_NUMBER]))
        assert ts.check_assignable(TS_STRING, union)
        assert ts.check_assignable(TS_NUMBER, union)

    def test_non_member_not_assignable_to_union(self, ts):
        """Non-member should not be assignable to union."""
        union = TSUnionType(frozenset([TS_STRING, TS_NUMBER]))
        assert not ts.check_assignable(TS_BOOLEAN, union)

    def test_union_assignable_to_wider_union(self, ts):
        """Narrower union should be assignable to wider union."""
        narrow = TSUnionType(frozenset([TS_STRING, TS_NUMBER]))
        wide = TSUnionType(frozenset([TS_STRING, TS_NUMBER, TS_BOOLEAN]))
        assert ts.check_assignable(narrow, wide)


# ===========================================================================
# Assignability Tests - Structural Typing
# ===========================================================================


class TestTypeScriptStructuralTyping:
    """Tests for structural typing (the key TypeScript feature)."""

    @pytest.fixture
    def ts(self):
        return TypeScriptTypeSystem()

    def test_object_extra_property_compatible(self, ts):
        """Object with extra property should be compatible."""
        source = TSObjectType(
            properties=(("name", TS_STRING), ("age", TS_NUMBER), ("extra", TS_BOOLEAN)),
            optional_properties=frozenset(),
        )
        target = TSObjectType(
            properties=(("name", TS_STRING), ("age", TS_NUMBER)),
            optional_properties=frozenset(),
        )
        assert ts.check_assignable(source, target)

    def test_object_missing_required_property(self, ts):
        """Object missing required property should not be compatible."""
        source = TSObjectType(
            properties=(("name", TS_STRING),),
            optional_properties=frozenset(),
        )
        target = TSObjectType(
            properties=(("name", TS_STRING), ("age", TS_NUMBER)),
            optional_properties=frozenset(),
        )
        assert not ts.check_assignable(source, target)

    def test_object_optional_property_can_be_missing(self, ts):
        """Source can omit optional properties from target."""
        source = TSObjectType(
            properties=(("name", TS_STRING),),
            optional_properties=frozenset(),
        )
        target = TSObjectType(
            properties=(("name", TS_STRING), ("age", TS_NUMBER)),
            optional_properties=frozenset(["age"]),
        )
        assert ts.check_assignable(source, target)


# ===========================================================================
# Assignability Tests - Arrays
# ===========================================================================


class TestTypeScriptAssignabilityArrays:
    """Tests for array type assignability."""

    @pytest.fixture
    def ts(self):
        return TypeScriptTypeSystem()

    def test_same_array_type(self, ts):
        """Same array types should be assignable."""
        arr1 = TSArrayType(TS_STRING)
        arr2 = TSArrayType(TS_STRING)
        assert ts.check_assignable(arr1, arr2)

    def test_different_array_element_not_assignable(self, ts):
        """Arrays with different element types should not be assignable."""
        arr1 = TSArrayType(TS_STRING)
        arr2 = TSArrayType(TS_NUMBER)
        assert not ts.check_assignable(arr1, arr2)


# ===========================================================================
# Literal Type Inference Tests
# ===========================================================================


class TestTypeScriptLiteralInference:
    """Tests for literal type inference."""

    @pytest.fixture
    def ts(self):
        return TypeScriptTypeSystem()

    def test_infer_integer_literal(self, ts):
        """Should infer number for integer."""
        lit = LiteralInfo(kind=LiteralKind.INTEGER, value=42)
        typ = ts.infer_literal_type(lit)
        assert typ == TS_NUMBER

    def test_infer_float_literal(self, ts):
        """Should infer number for float."""
        lit = LiteralInfo(kind=LiteralKind.FLOAT, value=3.14)
        typ = ts.infer_literal_type(lit)
        assert typ == TS_NUMBER

    def test_infer_string_literal(self, ts):
        """Should infer string for string."""
        lit = LiteralInfo(kind=LiteralKind.STRING, value="hello")
        typ = ts.infer_literal_type(lit)
        assert typ == TS_STRING

    def test_infer_boolean_literal(self, ts):
        """Should infer boolean for boolean."""
        lit = LiteralInfo(kind=LiteralKind.BOOLEAN, value=True)
        typ = ts.infer_literal_type(lit)
        assert typ == TS_BOOLEAN

    def test_infer_none_literal(self, ts):
        """Should infer undefined for none."""
        lit = LiteralInfo(kind=LiteralKind.NONE)
        typ = ts.infer_literal_type(lit)
        assert typ == TS_UNDEFINED


# ===========================================================================
# Builtin Types Tests
# ===========================================================================


class TestTypeScriptBuiltinTypes:
    """Tests for builtin types availability."""

    @pytest.fixture
    def ts(self):
        return TypeScriptTypeSystem()

    def test_primitive_types_in_builtins(self, ts):
        """Primitive types should be in builtins."""
        builtins = ts.get_builtin_types()
        assert "string" in builtins
        assert "number" in builtins
        assert "boolean" in builtins
        assert "bigint" in builtins
        assert "symbol" in builtins

    def test_special_types_in_builtins(self, ts):
        """Special types should be in builtins."""
        builtins = ts.get_builtin_types()
        assert "any" in builtins
        assert "unknown" in builtins
        assert "never" in builtins
        assert "void" in builtins

    def test_utility_types_in_builtins(self, ts):
        """Utility types should be in builtins."""
        builtins = ts.get_builtin_types()
        assert "Array" in builtins
        assert "Promise" in builtins
        assert "Map" in builtins
        assert "Set" in builtins
        assert "Record" in builtins
        assert "Partial" in builtins
        assert "Required" in builtins
        assert "Readonly" in builtins
        assert "Pick" in builtins
        assert "Omit" in builtins


# ===========================================================================
# Type Formatting Tests
# ===========================================================================


class TestTypeScriptTypeFormatting:
    """Tests for type formatting."""

    @pytest.fixture
    def ts(self):
        return TypeScriptTypeSystem()

    def test_format_primitives(self, ts):
        """Should format primitive types."""
        assert ts.format_type(TS_STRING) == "string"
        assert ts.format_type(TS_NUMBER) == "number"
        assert ts.format_type(TS_BOOLEAN) == "boolean"

    def test_format_array(self, ts):
        """Should format array types."""
        arr = TSArrayType(TS_STRING)
        assert ts.format_type(arr) == "string[]"

    def test_format_tuple(self, ts):
        """Should format tuple types."""
        tup = TSTupleType(elements=(TS_STRING, TS_NUMBER))
        assert ts.format_type(tup) == "[string, number]"

    def test_format_union(self, ts):
        """Should format union types."""
        union = TSUnionType(frozenset([TS_STRING, TS_NUMBER]))
        formatted = ts.format_type(union)
        # Order may vary due to frozenset
        assert "string" in formatted
        assert "number" in formatted
        assert "|" in formatted

    def test_format_function(self, ts):
        """Should format function types."""
        # TSFunctionType uses tuples for parameters: (name, type, optional)
        func = TSFunctionType(
            parameters=(("x", TS_NUMBER, False),),
            return_type=TS_STRING,
        )
        formatted = ts.format_type(func)
        assert "=>" in formatted


# ===========================================================================
# LUB/GLB Tests
# ===========================================================================


class TestTypeScriptLUBGLB:
    """Tests for LUB and GLB operations."""

    @pytest.fixture
    def ts(self):
        return TypeScriptTypeSystem()

    def test_lub_same_types(self, ts):
        """LUB of same type should be that type."""
        result = ts.lub([TS_STRING, TS_STRING])
        assert result == TS_STRING

    def test_lub_different_primitives(self, ts):
        """LUB of different primitives should be union."""
        result = ts.lub([TS_STRING, TS_NUMBER])
        assert isinstance(result, TSUnionType)
        assert TS_STRING in result.members
        assert TS_NUMBER in result.members

    def test_glb_same_types(self, ts):
        """GLB of same type should be that type."""
        result = ts.glb([TS_STRING, TS_STRING])
        assert result == TS_STRING


# ===========================================================================
# Error Handling Tests
# ===========================================================================


class TestTypeScriptErrorHandling:
    """Tests for error handling."""

    @pytest.fixture
    def ts(self):
        return TypeScriptTypeSystem()

    def test_parse_invalid_type(self, ts):
        """Should raise error for invalid type."""
        # Unbalanced brackets are syntactically invalid
        with pytest.raises(TypeParseError):
            ts.parse_type_annotation("Array<string")


# ===========================================================================
# Conditional Type Evaluation Tests
# ===========================================================================


class TestConditionalTypeEvaluation:
    """Tests for conditional type evaluation."""

    @pytest.fixture
    def ts(self):
        return TypeScriptTypeSystem()

    def test_conditional_true_branch(self, ts):
        """Should return true branch when condition is met."""
        cond = TSConditionalType(
            check_type=TS_STRING,
            extends_type=TS_STRING,
            true_type=TS_NUMBER,
            false_type=TS_BOOLEAN,
        )
        result = ts.evaluate_conditional(cond)
        assert result == TS_NUMBER

    def test_conditional_false_branch(self, ts):
        """Should return false branch when condition is not met."""
        cond = TSConditionalType(
            check_type=TS_NUMBER,
            extends_type=TS_STRING,
            true_type=TS_NUMBER,
            false_type=TS_BOOLEAN,
        )
        result = ts.evaluate_conditional(cond)
        assert result == TS_BOOLEAN

    def test_conditional_distributes_over_union(self, ts):
        """Should distribute conditional over union types."""
        # string | number extends string ? true : false
        # Should become: (string extends string ? true : false) | (number extends string ? true : false)
        # Which is: true | false
        cond = TSConditionalType(
            check_type=TSUnionType(members=frozenset({TS_STRING, TS_NUMBER})),
            extends_type=TS_STRING,
            true_type=TSLiteralType(kind="boolean", value=True),
            false_type=TSLiteralType(kind="boolean", value=False),
        )
        result = ts.evaluate_conditional(cond)
        assert isinstance(result, TSUnionType)
        assert len(result.members) == 2


# ===========================================================================
# Utility Type Operation Tests
# ===========================================================================


class TestUtilityTypeOperations:
    """Tests for utility type operations."""

    @pytest.fixture
    def ts(self):
        return TypeScriptTypeSystem()

    def test_partial_makes_all_optional(self, ts):
        """Partial<T> should make all properties optional."""
        obj = TSObjectType(
            properties=(("name", TS_STRING), ("age", TS_NUMBER)),
            optional_properties=frozenset(),
        )
        result = ts.apply_utility_type("Partial", [obj])
        assert isinstance(result, TSObjectType)
        assert "name" in result.optional_properties
        assert "age" in result.optional_properties

    def test_required_makes_all_required(self, ts):
        """Required<T> should make all properties required."""
        obj = TSObjectType(
            properties=(("name", TS_STRING), ("age", TS_NUMBER)),
            optional_properties=frozenset(["age"]),
        )
        result = ts.apply_utility_type("Required", [obj])
        assert isinstance(result, TSObjectType)
        assert len(result.optional_properties) == 0

    def test_pick_selects_properties(self, ts):
        """Pick<T, K> should select specified properties."""
        obj = TSObjectType(
            properties=(("name", TS_STRING), ("age", TS_NUMBER), ("email", TS_STRING)),
            optional_properties=frozenset(),
        )
        keys = TSUnionType(members=frozenset({
            TSLiteralType(kind="string", value="name"),
            TSLiteralType(kind="string", value="email"),
        }))
        result = ts.apply_utility_type("Pick", [obj, keys])
        assert isinstance(result, TSObjectType)
        assert len(result.properties) == 2
        assert "name" in result
        assert "email" in result
        assert "age" not in result

    def test_omit_removes_properties(self, ts):
        """Omit<T, K> should remove specified properties."""
        obj = TSObjectType(
            properties=(("name", TS_STRING), ("age", TS_NUMBER), ("email", TS_STRING)),
            optional_properties=frozenset(),
        )
        keys = TSLiteralType(kind="string", value="age")
        result = ts.apply_utility_type("Omit", [obj, keys])
        assert isinstance(result, TSObjectType)
        assert len(result.properties) == 2
        assert "name" in result
        assert "email" in result
        assert "age" not in result

    def test_record_creates_index_signature(self, ts):
        """Record<K, V> should create object with index signature."""
        result = ts.apply_utility_type("Record", [TS_STRING, TS_NUMBER])
        assert isinstance(result, TSObjectType)
        assert result.index_signature is not None
        assert result.index_signature[0] == TS_STRING
        assert result.index_signature[1] == TS_NUMBER

    def test_exclude_removes_from_union(self, ts):
        """Exclude<T, U> should remove types from union."""
        union = TSUnionType(members=frozenset({TS_STRING, TS_NUMBER, TS_BOOLEAN}))
        result = ts.apply_utility_type("Exclude", [union, TS_STRING])
        assert isinstance(result, TSUnionType)
        assert TS_STRING not in result.members
        assert TS_NUMBER in result.members
        assert TS_BOOLEAN in result.members

    def test_extract_keeps_matching_types(self, ts):
        """Extract<T, U> should keep types assignable to U."""
        union = TSUnionType(members=frozenset({TS_STRING, TS_NUMBER, TS_BOOLEAN}))
        result = ts.apply_utility_type("Extract", [union, TS_STRING])
        assert result == TS_STRING

    def test_nonnullable_removes_null_undefined(self, ts):
        """NonNullable<T> should remove null and undefined."""
        union = TSUnionType(members=frozenset({TS_STRING, TS_NULL, TS_UNDEFINED}))
        result = ts.apply_utility_type("NonNullable", [union])
        assert result == TS_STRING

    def test_returntype_extracts_return(self, ts):
        """ReturnType<T> should extract function return type."""
        func = TSFunctionType(
            parameters=(("x", TS_NUMBER, False),),
            return_type=TS_STRING,
        )
        result = ts.apply_utility_type("ReturnType", [func])
        assert result == TS_STRING

    def test_parameters_extracts_params_as_tuple(self, ts):
        """Parameters<T> should extract function parameters as tuple."""
        func = TSFunctionType(
            parameters=(("x", TS_NUMBER, False), ("y", TS_STRING, False)),
            return_type=TS_BOOLEAN,
        )
        result = ts.apply_utility_type("Parameters", [func])
        assert isinstance(result, TSTupleType)
        assert len(result.elements) == 2
        assert result.elements[0] == TS_NUMBER
        assert result.elements[1] == TS_STRING


# ===========================================================================
# API Ergonomics Tests
# ===========================================================================


class TestAPIErgonomics:
    """Tests for new API ergonomic features."""

    def test_object_type_contains(self):
        """Should support 'in' operator for property check."""
        obj = TSObjectType(
            properties=(("name", TS_STRING), ("age", TS_NUMBER)),
            optional_properties=frozenset(),
        )
        assert "name" in obj
        assert "age" in obj
        assert "email" not in obj

    def test_object_type_getitem(self):
        """Should support bracket access for properties."""
        obj = TSObjectType(
            properties=(("name", TS_STRING), ("age", TS_NUMBER)),
            optional_properties=frozenset(),
        )
        assert obj["name"] == TS_STRING
        assert obj["age"] == TS_NUMBER
        with pytest.raises(KeyError):
            _ = obj["email"]

    def test_object_type_property_names(self):
        """Should return all property names."""
        obj = TSObjectType(
            properties=(("name", TS_STRING), ("age", TS_NUMBER)),
            optional_properties=frozenset(),
        )
        names = obj.property_names()
        assert names == frozenset({"name", "age"})

    def test_function_get_parameter(self):
        """Should return TSParameter with named access."""
        func = TSFunctionType(
            parameters=(("x", TS_NUMBER, False), ("y", TS_STRING, True)),
            return_type=TS_BOOLEAN,
        )
        param = func.get_parameter(0)
        assert param.name == "x"
        assert param.type == TS_NUMBER
        assert param.optional is False

        param2 = func.get_parameter(1)
        assert param2.name == "y"
        assert param2.optional is True

    def test_function_get_parameter_by_name(self):
        """Should find parameter by name."""
        func = TSFunctionType(
            parameters=(("x", TS_NUMBER, False), ("y", TS_STRING, True)),
            return_type=TS_BOOLEAN,
        )
        param = func.get_parameter_by_name("y")
        assert param is not None
        assert param.name == "y"
        assert param.type == TS_STRING

        missing = func.get_parameter_by_name("z")
        assert missing is None

    def test_function_iter_parameters(self):
        """Should iterate over parameters as TSParameter."""
        func = TSFunctionType(
            parameters=(("x", TS_NUMBER, False), ("y", TS_STRING, True)),
            return_type=TS_BOOLEAN,
        )
        params = list(func.iter_parameters())
        assert len(params) == 2
        assert all(isinstance(p, TSParameter) for p in params)

    def test_function_required_parameters(self):
        """Should return only required parameters."""
        func = TSFunctionType(
            parameters=(("x", TS_NUMBER, False), ("y", TS_STRING, True)),
            return_type=TS_BOOLEAN,
        )
        required = func.required_parameters
        assert len(required) == 1
        assert required[0].name == "x"

    def test_function_optional_parameters(self):
        """Should return only optional parameters."""
        func = TSFunctionType(
            parameters=(("x", TS_NUMBER, False), ("y", TS_STRING, True)),
            return_type=TS_BOOLEAN,
        )
        optional = func.optional_parameters
        assert len(optional) == 1
        assert optional[0].name == "y"
