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
"""Unit tests for language type systems.

Tests for the base LanguageTypeSystem ABC and Python type system implementation.
"""

import pytest

from domains.types.constraint import (
    INT,
    STR,
    BOOL,
    FLOAT,
    NONE,
    ANY,
    NEVER,
    TypeVar,
    ListType,
    DictType,
    TupleType,
    SetType,
    FunctionType,
    UnionType,
    ClassType,
    OptionalType,
)
from domains.types.languages import (
    get_type_system,
    supported_languages,
    TypeParseError,
    LiteralInfo,
    LiteralKind,
)
from domains.types.languages.python import (
    PythonTypeSystem,
    BYTES,
    COMPLEX,
    is_optional_type,
    get_optional_inner,
)


# ===========================================================================
# Factory Function Tests
# ===========================================================================


class TestGetTypeSystem:
    """Tests for get_type_system factory function."""

    def test_get_python_by_name(self):
        """Should return Python type system."""
        ts = get_type_system("python")
        assert isinstance(ts, PythonTypeSystem)
        assert ts.name == "python"

    def test_get_python_by_alias(self):
        """Should accept 'py' as alias."""
        ts = get_type_system("py")
        assert isinstance(ts, PythonTypeSystem)

    def test_case_insensitive(self):
        """Should be case insensitive."""
        ts = get_type_system("Python")
        assert isinstance(ts, PythonTypeSystem)

    def test_unsupported_language(self):
        """Should raise ValueError for unsupported language."""
        with pytest.raises(ValueError, match="Unsupported language"):
            get_type_system("cobol")


class TestSupportedLanguages:
    """Tests for supported_languages function."""

    def test_returns_list(self):
        """Should return list of supported languages."""
        langs = supported_languages()
        assert isinstance(langs, list)
        assert "python" in langs


# ===========================================================================
# Python Type System Tests
# ===========================================================================


class TestPythonTypeSystemBasics:
    """Basic tests for PythonTypeSystem."""

    @pytest.fixture
    def ts(self):
        """Create a Python type system instance."""
        return PythonTypeSystem()

    def test_name(self, ts):
        """Name should be 'python'."""
        assert ts.name == "python"

    def test_capabilities(self, ts):
        """Should have correct capabilities."""
        caps = ts.capabilities
        assert caps.supports_generics
        assert caps.supports_union_types
        assert caps.supports_optional_types
        assert caps.supports_type_inference
        assert caps.supports_protocols
        assert caps.supports_variance
        assert not caps.supports_ownership  # Python doesn't have ownership
        assert not caps.supports_comptime  # Python doesn't have comptime


class TestPythonTypeParsingPrimitives:
    """Tests for parsing primitive types."""

    @pytest.fixture
    def ts(self):
        return PythonTypeSystem()

    def test_parse_int(self, ts):
        """Should parse 'int'."""
        typ = ts.parse_type_annotation("int")
        assert typ == INT

    def test_parse_str(self, ts):
        """Should parse 'str'."""
        typ = ts.parse_type_annotation("str")
        assert typ == STR

    def test_parse_bool(self, ts):
        """Should parse 'bool'."""
        typ = ts.parse_type_annotation("bool")
        assert typ == BOOL

    def test_parse_float(self, ts):
        """Should parse 'float'."""
        typ = ts.parse_type_annotation("float")
        assert typ == FLOAT

    def test_parse_bytes(self, ts):
        """Should parse 'bytes'."""
        typ = ts.parse_type_annotation("bytes")
        assert typ == BYTES

    def test_parse_none(self, ts):
        """Should parse 'None'."""
        typ = ts.parse_type_annotation("None")
        assert typ == NONE

    def test_parse_any(self, ts):
        """Should parse 'Any'."""
        typ = ts.parse_type_annotation("Any")
        assert typ == ANY

    def test_parse_never(self, ts):
        """Should parse 'Never'."""
        typ = ts.parse_type_annotation("Never")
        assert typ == NEVER


class TestPythonTypeParsingGenerics:
    """Tests for parsing generic types."""

    @pytest.fixture
    def ts(self):
        return PythonTypeSystem()

    def test_parse_list(self, ts):
        """Should parse 'List[int]'."""
        typ = ts.parse_type_annotation("List[int]")
        assert isinstance(typ, ListType)
        assert typ.element == INT

    def test_parse_lowercase_list(self, ts):
        """Should parse 'list[str]'."""
        typ = ts.parse_type_annotation("list[str]")
        assert isinstance(typ, ListType)
        assert typ.element == STR

    def test_parse_set(self, ts):
        """Should parse 'Set[int]'."""
        typ = ts.parse_type_annotation("Set[int]")
        assert isinstance(typ, SetType)
        assert typ.element == INT

    def test_parse_dict(self, ts):
        """Should parse 'Dict[str, int]'."""
        typ = ts.parse_type_annotation("Dict[str, int]")
        assert isinstance(typ, DictType)
        assert typ.key == STR
        assert typ.value == INT

    def test_parse_tuple(self, ts):
        """Should parse 'Tuple[int, str]'."""
        typ = ts.parse_type_annotation("Tuple[int, str]")
        assert isinstance(typ, TupleType)
        assert typ.elements == (INT, STR)

    def test_parse_empty_tuple(self, ts):
        """Should parse 'Tuple[()]'."""
        typ = ts.parse_type_annotation("Tuple[()]")
        assert isinstance(typ, TupleType)
        assert typ.elements == ()

    def test_parse_nested_generic(self, ts):
        """Should parse 'List[Dict[str, int]]'."""
        typ = ts.parse_type_annotation("List[Dict[str, int]]")
        assert isinstance(typ, ListType)
        inner = typ.element
        assert isinstance(inner, DictType)
        assert inner.key == STR
        assert inner.value == INT


class TestPythonTypeParsingUnionOptional:
    """Tests for parsing union and optional types."""

    @pytest.fixture
    def ts(self):
        return PythonTypeSystem()

    def test_parse_union(self, ts):
        """Should parse 'Union[int, str]'."""
        typ = ts.parse_type_annotation("Union[int, str]")
        assert isinstance(typ, UnionType)
        assert INT in typ.members
        assert STR in typ.members

    def test_parse_optional(self, ts):
        """Should parse 'Optional[int]'."""
        typ = ts.parse_type_annotation("Optional[int]")
        assert isinstance(typ, UnionType)
        assert INT in typ.members
        assert NONE in typ.members

    def test_parse_pipe_syntax(self, ts):
        """Should parse 'int | str'."""
        typ = ts.parse_type_annotation("int | str")
        assert isinstance(typ, UnionType)
        assert INT in typ.members
        assert STR in typ.members

    def test_parse_optional_pipe(self, ts):
        """Should parse 'int | None'."""
        typ = ts.parse_type_annotation("int | None")
        assert isinstance(typ, UnionType)
        assert INT in typ.members
        assert NONE in typ.members


class TestPythonTypeParsingCallable:
    """Tests for parsing Callable types."""

    @pytest.fixture
    def ts(self):
        return PythonTypeSystem()

    def test_parse_callable_simple(self, ts):
        """Should parse 'Callable[[int], str]'."""
        typ = ts.parse_type_annotation("Callable[[int], str]")
        assert isinstance(typ, FunctionType)
        assert typ.params == (INT,)
        assert typ.returns == STR

    def test_parse_callable_multiple_params(self, ts):
        """Should parse 'Callable[[int, str], bool]'."""
        typ = ts.parse_type_annotation("Callable[[int, str], bool]")
        assert isinstance(typ, FunctionType)
        assert typ.params == (INT, STR)
        assert typ.returns == BOOL

    def test_parse_callable_no_params(self, ts):
        """Should parse 'Callable[[], int]'."""
        typ = ts.parse_type_annotation("Callable[[], int]")
        assert isinstance(typ, FunctionType)
        assert typ.params == ()
        assert typ.returns == INT

    def test_parse_callable_ellipsis(self, ts):
        """Should parse 'Callable[..., int]'."""
        typ = ts.parse_type_annotation("Callable[..., int]")
        assert isinstance(typ, FunctionType)
        assert typ.returns == INT


class TestPythonTypeParsingErrors:
    """Tests for type parsing errors."""

    @pytest.fixture
    def ts(self):
        return PythonTypeSystem()

    def test_empty_annotation_fails(self, ts):
        """Empty annotation should raise TypeParseError."""
        with pytest.raises(TypeParseError):
            ts.parse_type_annotation("")

    def test_invalid_dict_args(self, ts):
        """Dict with wrong number of args should fail."""
        with pytest.raises(TypeParseError):
            ts.parse_type_annotation("Dict[str]")


class TestPythonAssignability:
    """Tests for type assignability checking."""

    @pytest.fixture
    def ts(self):
        return PythonTypeSystem()

    def test_same_type_assignable(self, ts):
        """Same types should be assignable."""
        assert ts.check_assignable(INT, INT)
        assert ts.check_assignable(STR, STR)

    def test_any_assignable_to_all(self, ts):
        """Any should be assignable to anything."""
        assert ts.check_assignable(ANY, INT)
        assert ts.check_assignable(INT, ANY)

    def test_never_assignable_to_all(self, ts):
        """Never should be assignable to anything."""
        assert ts.check_assignable(NEVER, INT)
        assert not ts.check_assignable(INT, NEVER)

    def test_int_to_float(self, ts):
        """int should be assignable to float."""
        assert ts.check_assignable(INT, FLOAT)

    def test_float_not_to_int(self, ts):
        """float should not be assignable to int."""
        assert not ts.check_assignable(FLOAT, INT)

    def test_bool_to_int(self, ts):
        """bool should be assignable to int."""
        assert ts.check_assignable(BOOL, INT)

    def test_none_to_optional(self, ts):
        """None should be assignable to Optional[T]."""
        optional_int = OptionalType(INT)
        assert ts.check_assignable(NONE, optional_int)

    def test_t_to_optional_t(self, ts):
        """T should be assignable to Optional[T]."""
        optional_int = OptionalType(INT)
        assert ts.check_assignable(INT, optional_int)

    def test_union_target(self, ts):
        """Should check assignability to union."""
        union = UnionType(frozenset({INT, STR}))
        assert ts.check_assignable(INT, union)
        assert ts.check_assignable(STR, union)
        assert not ts.check_assignable(FLOAT, union)

    def test_union_source(self, ts):
        """Should check assignability from union."""
        union = UnionType(frozenset({INT, BOOL}))
        # All members must be assignable to target
        assert ts.check_assignable(union, INT)  # Both int and bool are assignable to int

    def test_list_covariance(self, ts):
        """List should be covariant."""
        list_int = ListType(INT)
        list_float = ListType(FLOAT)
        # int is assignable to float, so List[int] is assignable to List[float]
        assert ts.check_assignable(list_int, list_float)

    def test_dict_covariance(self, ts):
        """Dict should be covariant in both key and value."""
        dict1 = DictType(STR, INT)
        dict2 = DictType(STR, FLOAT)
        assert ts.check_assignable(dict1, dict2)

    def test_function_contravariance(self, ts):
        """Function params should be contravariant."""
        # (float) -> int should be assignable to (int) -> int
        # because int is assignable to float (param is contravariant)
        func1 = FunctionType((FLOAT,), INT)
        func2 = FunctionType((INT,), INT)
        assert ts.check_assignable(func1, func2)

    def test_function_covariant_return(self, ts):
        """Function return should be covariant."""
        func1 = FunctionType((INT,), INT)
        func2 = FunctionType((INT,), FLOAT)
        assert ts.check_assignable(func1, func2)

    def test_tuple_assignability(self, ts):
        """Tuple elements should match."""
        tuple1 = TupleType((INT, STR))
        tuple2 = TupleType((INT, STR))
        assert ts.check_assignable(tuple1, tuple2)

    def test_tuple_length_mismatch(self, ts):
        """Tuples with different lengths should not be assignable."""
        tuple1 = TupleType((INT,))
        tuple2 = TupleType((INT, STR))
        assert not ts.check_assignable(tuple1, tuple2)


class TestPythonTypeFormatting:
    """Tests for type formatting."""

    @pytest.fixture
    def ts(self):
        return PythonTypeSystem()

    def test_format_primitive(self, ts):
        """Should format primitives correctly."""
        assert ts.format_type(INT) == "int"
        assert ts.format_type(STR) == "str"
        assert ts.format_type(NONE) == "None"

    def test_format_list(self, ts):
        """Should format list types."""
        assert ts.format_type(ListType(INT)) == "list[int]"

    def test_format_dict(self, ts):
        """Should format dict types."""
        assert ts.format_type(DictType(STR, INT)) == "dict[str, int]"

    def test_format_tuple(self, ts):
        """Should format tuple types."""
        assert ts.format_type(TupleType((INT, STR))) == "tuple[int, str]"

    def test_format_function(self, ts):
        """Should format function types."""
        result = ts.format_type(FunctionType((INT,), STR))
        assert "Callable" in result
        assert "int" in result
        assert "str" in result

    def test_format_optional(self, ts):
        """Should format optional as union with None."""
        typ = OptionalType(INT)
        formatted = ts.format_type(typ)
        assert "None" in formatted
        assert "int" in formatted


class TestPythonLiteralInference:
    """Tests for literal type inference."""

    @pytest.fixture
    def ts(self):
        return PythonTypeSystem()

    def test_infer_integer(self, ts):
        """Should infer int for integer literals."""
        literal = LiteralInfo(LiteralKind.INTEGER, value=42)
        assert ts.infer_literal_type(literal) == INT

    def test_infer_float(self, ts):
        """Should infer float for float literals."""
        literal = LiteralInfo(LiteralKind.FLOAT, value=3.14)
        assert ts.infer_literal_type(literal) == FLOAT

    def test_infer_string(self, ts):
        """Should infer str for string literals."""
        literal = LiteralInfo(LiteralKind.STRING, value="hello")
        assert ts.infer_literal_type(literal) == STR

    def test_infer_boolean(self, ts):
        """Should infer bool for boolean literals."""
        literal = LiteralInfo(LiteralKind.BOOLEAN, value=True)
        assert ts.infer_literal_type(literal) == BOOL

    def test_infer_none(self, ts):
        """Should infer None for None literals."""
        literal = LiteralInfo(LiteralKind.NONE)
        assert ts.infer_literal_type(literal) == NONE


class TestPythonBuiltins:
    """Tests for builtin types and functions."""

    @pytest.fixture
    def ts(self):
        return PythonTypeSystem()

    def test_has_builtin_types(self, ts):
        """Should have builtin types."""
        builtins = ts.get_builtin_types()
        assert "int" in builtins
        assert "str" in builtins
        assert "list" in builtins
        assert "dict" in builtins

    def test_has_builtin_functions(self, ts):
        """Should have builtin function signatures."""
        funcs = ts.get_builtin_functions()
        assert "len" in funcs
        assert "print" in funcs
        assert "range" in funcs

    def test_len_signature(self, ts):
        """len should take any and return int."""
        funcs = ts.get_builtin_functions()
        len_type = funcs["len"]
        assert isinstance(len_type, FunctionType)
        assert len_type.returns == INT


class TestOptionalTypeHelpers:
    """Tests for optional type helper functions."""

    def test_is_optional_type_true(self):
        """Should detect optional types."""
        optional = OptionalType(INT)
        assert is_optional_type(optional)

    def test_is_optional_type_false_for_non_union(self):
        """Should return False for non-union types."""
        assert not is_optional_type(INT)
        assert not is_optional_type(ListType(INT))

    def test_is_optional_type_false_for_union_without_none(self):
        """Should return False for unions without None."""
        union = UnionType(frozenset({INT, STR}))
        assert not is_optional_type(union)

    def test_get_optional_inner(self):
        """Should extract inner type from optional."""
        optional = OptionalType(INT)
        inner = get_optional_inner(optional)
        assert inner == INT

    def test_get_optional_inner_union(self):
        """Should handle union optionals."""
        union = UnionType(frozenset({INT, STR, NONE}))
        inner = get_optional_inner(union)
        assert isinstance(inner, UnionType)
        assert INT in inner.members
        assert STR in inner.members
        assert NONE not in inner.members
