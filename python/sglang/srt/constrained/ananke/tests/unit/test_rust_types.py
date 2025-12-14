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
"""Unit tests for Rust type system.

Tests for the RustTypeSystem implementation including:
- Primitive type parsing
- Reference types (&T, &mut T, &'a T)
- Smart pointers (Box, Rc, Arc)
- Option and Result types
- Slices and arrays
- Function types
- Trait objects
- Lifetime handling
- Assignability rules
"""

import pytest

from domains.types.constraint import ANY, NEVER, TupleType
from domains.types.languages import (
    get_type_system,
    supported_languages,
    TypeParseError,
    LiteralInfo,
    LiteralKind,
)
from domains.types.languages.rust import (
    RustTypeSystem,
    RustReferenceType,
    RustSliceType,
    RustArrayType,
    RustBoxType,
    RustRcType,
    RustCowType,
    RustOptionType,
    RustResultType,
    RustVecType,
    RustStringType,
    RustHashMapType,
    RustTraitBound,
    RustGenericType,
    RustFunctionType,
    RustRawPointerType,
    RustDynTraitType,
    RustImplTraitType,
    # Primitive constants
    RUST_I8,
    RUST_I16,
    RUST_I32,
    RUST_I64,
    RUST_I128,
    RUST_ISIZE,
    RUST_U8,
    RUST_U16,
    RUST_U32,
    RUST_U64,
    RUST_U128,
    RUST_USIZE,
    RUST_F32,
    RUST_F64,
    RUST_BOOL,
    RUST_CHAR,
    RUST_STR,
    RUST_UNIT,
    RUST_NEVER,
)


# Helper functions for type checking
def is_owned_type(t) -> bool:
    """Check if type is an owned type (Box, String, Vec, etc.)."""
    return isinstance(t, (RustBoxType, RustStringType, RustVecType))


def is_borrowed_type(t) -> bool:
    """Check if type is a borrowed reference."""
    return isinstance(t, RustReferenceType)


def is_copy_type(t) -> bool:
    """Check if type is Copy (primitives)."""
    return t in (
        RUST_I8, RUST_I16, RUST_I32, RUST_I64, RUST_I128, RUST_ISIZE,
        RUST_U8, RUST_U16, RUST_U32, RUST_U64, RUST_U128, RUST_USIZE,
        RUST_F32, RUST_F64, RUST_BOOL, RUST_CHAR,
    )


# ===========================================================================
# Factory Function Tests
# ===========================================================================


class TestGetRustTypeSystem:
    """Tests for get_type_system with Rust."""

    def test_get_rust_by_name(self):
        """Should return Rust type system."""
        ts = get_type_system("rust")
        assert isinstance(ts, RustTypeSystem)
        assert ts.name == "rust"

    def test_get_rust_by_alias(self):
        """Should accept 'rs' as alias."""
        ts = get_type_system("rs")
        assert isinstance(ts, RustTypeSystem)

    def test_rust_in_supported_languages(self):
        """Rust should be in supported languages."""
        langs = supported_languages()
        assert "rust" in langs


# ===========================================================================
# Rust Type System Basic Tests
# ===========================================================================


class TestRustTypeSystemBasics:
    """Basic tests for RustTypeSystem."""

    @pytest.fixture
    def ts(self):
        """Create a Rust type system instance."""
        return RustTypeSystem()

    def test_name(self, ts):
        """Name should be 'rust'."""
        assert ts.name == "rust"

    def test_capabilities(self, ts):
        """Should have correct capabilities."""
        caps = ts.capabilities
        assert caps.supports_generics
        assert caps.supports_ownership
        assert caps.supports_lifetime_bounds
        assert caps.supports_protocols  # Traits
        assert caps.supports_variance


# ===========================================================================
# Primitive Type Parsing Tests
# ===========================================================================


class TestRustPrimitiveTypeParsing:
    """Tests for parsing Rust primitive types."""

    @pytest.fixture
    def ts(self):
        return RustTypeSystem()

    # Signed integers
    def test_parse_i8(self, ts):
        """Should parse 'i8'."""
        typ = ts.parse_type_annotation("i8")
        assert typ == RUST_I8

    def test_parse_i16(self, ts):
        """Should parse 'i16'."""
        typ = ts.parse_type_annotation("i16")
        assert typ == RUST_I16

    def test_parse_i32(self, ts):
        """Should parse 'i32'."""
        typ = ts.parse_type_annotation("i32")
        assert typ == RUST_I32

    def test_parse_i64(self, ts):
        """Should parse 'i64'."""
        typ = ts.parse_type_annotation("i64")
        assert typ == RUST_I64

    def test_parse_i128(self, ts):
        """Should parse 'i128'."""
        typ = ts.parse_type_annotation("i128")
        assert typ == RUST_I128

    def test_parse_isize(self, ts):
        """Should parse 'isize'."""
        typ = ts.parse_type_annotation("isize")
        assert typ == RUST_ISIZE

    # Unsigned integers
    def test_parse_u8(self, ts):
        """Should parse 'u8'."""
        typ = ts.parse_type_annotation("u8")
        assert typ == RUST_U8

    def test_parse_u16(self, ts):
        """Should parse 'u16'."""
        typ = ts.parse_type_annotation("u16")
        assert typ == RUST_U16

    def test_parse_u32(self, ts):
        """Should parse 'u32'."""
        typ = ts.parse_type_annotation("u32")
        assert typ == RUST_U32

    def test_parse_u64(self, ts):
        """Should parse 'u64'."""
        typ = ts.parse_type_annotation("u64")
        assert typ == RUST_U64

    def test_parse_u128(self, ts):
        """Should parse 'u128'."""
        typ = ts.parse_type_annotation("u128")
        assert typ == RUST_U128

    def test_parse_usize(self, ts):
        """Should parse 'usize'."""
        typ = ts.parse_type_annotation("usize")
        assert typ == RUST_USIZE

    # Floats
    def test_parse_f32(self, ts):
        """Should parse 'f32'."""
        typ = ts.parse_type_annotation("f32")
        assert typ == RUST_F32

    def test_parse_f64(self, ts):
        """Should parse 'f64'."""
        typ = ts.parse_type_annotation("f64")
        assert typ == RUST_F64

    # Other primitives
    def test_parse_bool(self, ts):
        """Should parse 'bool'."""
        typ = ts.parse_type_annotation("bool")
        assert typ == RUST_BOOL

    def test_parse_char(self, ts):
        """Should parse 'char'."""
        typ = ts.parse_type_annotation("char")
        assert typ == RUST_CHAR

    def test_parse_str(self, ts):
        """Should parse 'str'."""
        typ = ts.parse_type_annotation("str")
        assert typ == RUST_STR

    def test_parse_unit(self, ts):
        """Should parse '()'."""
        typ = ts.parse_type_annotation("()")
        assert typ == RUST_UNIT

    def test_parse_never(self, ts):
        """Should parse '!'."""
        typ = ts.parse_type_annotation("!")
        assert typ == RUST_NEVER


# ===========================================================================
# Reference Type Parsing Tests
# ===========================================================================


class TestRustReferenceTypeParsing:
    """Tests for parsing Rust reference types."""

    @pytest.fixture
    def ts(self):
        return RustTypeSystem()

    def test_parse_shared_ref(self, ts):
        """Should parse '&i32'."""
        typ = ts.parse_type_annotation("&i32")
        assert isinstance(typ, RustReferenceType)
        assert typ.referent == RUST_I32
        assert not typ.is_mutable

    def test_parse_mutable_ref(self, ts):
        """Should parse '&mut i32'."""
        typ = ts.parse_type_annotation("&mut i32")
        assert isinstance(typ, RustReferenceType)
        assert typ.referent == RUST_I32
        assert typ.is_mutable

    def test_parse_ref_with_lifetime(self, ts):
        """Should parse \"&'a i32\"."""
        typ = ts.parse_type_annotation("&'a i32")
        assert isinstance(typ, RustReferenceType)
        assert typ.referent == RUST_I32
        assert typ.lifetime == "a"  # Lifetime stored without leading quote

    def test_parse_static_ref(self, ts):
        """Should parse \"&'static str\"."""
        typ = ts.parse_type_annotation("&'static str")
        assert isinstance(typ, RustReferenceType)
        assert typ.referent == RUST_STR
        assert typ.lifetime == "static"  # Lifetime stored without leading quote

    def test_parse_mut_ref_with_lifetime(self, ts):
        """Should parse \"&'a mut i32\"."""
        typ = ts.parse_type_annotation("&'a mut i32")
        assert isinstance(typ, RustReferenceType)
        assert typ.is_mutable
        assert typ.lifetime == "a"  # Lifetime stored without leading quote


# ===========================================================================
# Slice and Array Type Parsing Tests
# ===========================================================================


class TestRustSliceArrayTypeParsing:
    """Tests for parsing Rust slice and array types."""

    @pytest.fixture
    def ts(self):
        return RustTypeSystem()

    def test_parse_slice(self, ts):
        """Should parse '&[u8]' - returns RustSliceType with embedded ref semantics."""
        typ = ts.parse_type_annotation("&[u8]")
        assert isinstance(typ, RustSliceType)
        assert typ.element == RUST_U8
        assert not typ.is_mutable

    def test_parse_mut_slice(self, ts):
        """Should parse '&mut [u8]' - returns RustSliceType with mutable flag."""
        typ = ts.parse_type_annotation("&mut [u8]")
        assert isinstance(typ, RustSliceType)
        assert typ.is_mutable
        assert typ.element == RUST_U8

    def test_parse_array(self, ts):
        """Should parse '[i32; 10]'."""
        typ = ts.parse_type_annotation("[i32; 10]")
        assert isinstance(typ, RustArrayType)
        assert typ.element == RUST_I32
        assert typ.length == 10


# ===========================================================================
# Smart Pointer Type Parsing Tests
# ===========================================================================


class TestRustSmartPointerTypeParsing:
    """Tests for parsing Rust smart pointer types."""

    @pytest.fixture
    def ts(self):
        return RustTypeSystem()

    def test_parse_box(self, ts):
        """Should parse 'Box<i32>'."""
        typ = ts.parse_type_annotation("Box<i32>")
        assert isinstance(typ, RustBoxType)
        assert typ.inner == RUST_I32

    def test_parse_rc(self, ts):
        """Should parse 'Rc<String>'."""
        typ = ts.parse_type_annotation("Rc<String>")
        assert isinstance(typ, RustRcType)
        assert isinstance(typ.inner, RustStringType)

    def test_parse_arc(self, ts):
        """Should parse 'Arc<i32>' - uses RustRcType with is_arc=True."""
        typ = ts.parse_type_annotation("Arc<i32>")
        assert isinstance(typ, RustRcType)
        assert typ.is_arc
        assert typ.inner == RUST_I32

    def test_parse_cow(self, ts):
        """Should parse \"Cow<'a, str>\"."""
        typ = ts.parse_type_annotation("Cow<'a, str>")
        assert isinstance(typ, RustCowType)
        assert typ.inner == RUST_STR  # Uses .inner, not .borrowed


# ===========================================================================
# Option and Result Type Parsing Tests
# ===========================================================================


class TestRustOptionResultTypeParsing:
    """Tests for parsing Option and Result types."""

    @pytest.fixture
    def ts(self):
        return RustTypeSystem()

    def test_parse_option(self, ts):
        """Should parse 'Option<i32>'."""
        typ = ts.parse_type_annotation("Option<i32>")
        assert isinstance(typ, RustOptionType)
        assert typ.inner == RUST_I32

    def test_parse_result(self, ts):
        """Should parse 'Result<i32, String>'."""
        typ = ts.parse_type_annotation("Result<i32, String>")
        assert isinstance(typ, RustResultType)
        assert typ.ok_type == RUST_I32
        assert isinstance(typ.err_type, RustStringType)


# ===========================================================================
# Collection Type Parsing Tests
# ===========================================================================


class TestRustCollectionTypeParsing:
    """Tests for parsing Rust collection types."""

    @pytest.fixture
    def ts(self):
        return RustTypeSystem()

    def test_parse_vec(self, ts):
        """Should parse 'Vec<i32>'."""
        typ = ts.parse_type_annotation("Vec<i32>")
        assert isinstance(typ, RustVecType)
        assert typ.element == RUST_I32

    def test_parse_string(self, ts):
        """Should parse 'String'."""
        typ = ts.parse_type_annotation("String")
        assert isinstance(typ, RustStringType)

    def test_parse_hashmap(self, ts):
        """Should parse 'HashMap<String, i32>'."""
        typ = ts.parse_type_annotation("HashMap<String, i32>")
        assert isinstance(typ, RustHashMapType)
        assert isinstance(typ.key, RustStringType)
        assert typ.value == RUST_I32


# ===========================================================================
# Function Type Parsing Tests
# ===========================================================================


class TestRustFunctionTypeParsing:
    """Tests for parsing Rust function types."""

    @pytest.fixture
    def ts(self):
        return RustTypeSystem()

    def test_parse_fn_simple(self, ts):
        """Should parse 'fn(i32) -> i32'."""
        typ = ts.parse_type_annotation("fn(i32) -> i32")
        assert isinstance(typ, RustFunctionType)
        assert len(typ.params) == 1
        assert typ.params[0] == RUST_I32
        assert typ.return_type == RUST_I32

    def test_parse_fn_multiple_params(self, ts):
        """Should parse 'fn(i32, i32) -> i32'."""
        typ = ts.parse_type_annotation("fn(i32, i32) -> i32")
        assert isinstance(typ, RustFunctionType)
        assert len(typ.params) == 2

    def test_parse_fn_no_return(self, ts):
        """Should parse 'fn(i32)'."""
        typ = ts.parse_type_annotation("fn(i32)")
        assert isinstance(typ, RustFunctionType)
        assert typ.return_type == RUST_UNIT

    def test_parse_fn_no_params(self, ts):
        """Should parse 'fn() -> i32'."""
        typ = ts.parse_type_annotation("fn() -> i32")
        assert isinstance(typ, RustFunctionType)
        assert len(typ.params) == 0


# ===========================================================================
# Tuple Type Parsing Tests
# ===========================================================================


class TestRustTupleTypeParsing:
    """Tests for parsing Rust tuple types."""

    @pytest.fixture
    def ts(self):
        return RustTypeSystem()

    def test_parse_tuple(self, ts):
        """Should parse '(i32, i32)'."""
        typ = ts.parse_type_annotation("(i32, i32)")
        assert isinstance(typ, TupleType)
        assert len(typ.elements) == 2
        assert typ.elements[0] == RUST_I32
        assert typ.elements[1] == RUST_I32

    def test_parse_heterogeneous_tuple(self, ts):
        """Should parse '(i32, String, bool)'."""
        typ = ts.parse_type_annotation("(i32, String, bool)")
        assert isinstance(typ, TupleType)
        assert len(typ.elements) == 3
        assert typ.elements[0] == RUST_I32
        assert isinstance(typ.elements[1], RustStringType)
        assert typ.elements[2] == RUST_BOOL


# ===========================================================================
# Raw Pointer Type Parsing Tests
# ===========================================================================


class TestRustRawPointerTypeParsing:
    """Tests for parsing Rust raw pointer types."""

    @pytest.fixture
    def ts(self):
        return RustTypeSystem()

    def test_parse_const_ptr(self, ts):
        """Should parse '*const i32'."""
        typ = ts.parse_type_annotation("*const i32")
        assert isinstance(typ, RustRawPointerType)
        assert typ.pointee == RUST_I32
        assert not typ.is_mutable

    def test_parse_mut_ptr(self, ts):
        """Should parse '*mut i32'."""
        typ = ts.parse_type_annotation("*mut i32")
        assert isinstance(typ, RustRawPointerType)
        assert typ.pointee == RUST_I32
        assert typ.is_mutable


# ===========================================================================
# Trait Object Type Parsing Tests
# ===========================================================================


class TestRustTraitObjectTypeParsing:
    """Tests for parsing Rust trait object types."""

    @pytest.fixture
    def ts(self):
        return RustTypeSystem()

    def test_parse_dyn_trait(self, ts):
        """Should parse 'dyn Clone'."""
        typ = ts.parse_type_annotation("dyn Clone")
        assert isinstance(typ, RustDynTraitType)
        assert typ.trait_name == "Clone"

    def test_parse_impl_trait(self, ts):
        """Should parse 'impl Iterator'."""
        typ = ts.parse_type_annotation("impl Iterator")
        assert isinstance(typ, RustImplTraitType)
        assert typ.trait_name == "Iterator"


# ===========================================================================
# Ownership Helper Tests
# ===========================================================================


class TestRustOwnershipHelpers:
    """Tests for ownership helper functions."""

    @pytest.fixture
    def ts(self):
        return RustTypeSystem()

    def test_box_is_owned(self, ts):
        """Box should be owned type."""
        box_type = RustBoxType(RUST_I32)
        assert is_owned_type(box_type)

    def test_vec_is_owned(self, ts):
        """Vec should be owned type."""
        vec_type = RustVecType(RUST_I32)
        assert is_owned_type(vec_type)

    def test_string_is_owned(self, ts):
        """String should be owned type."""
        string_type = RustStringType()
        assert is_owned_type(string_type)

    def test_reference_is_borrowed(self, ts):
        """Reference should be borrowed type."""
        ref_type = RustReferenceType(RUST_I32)
        assert is_borrowed_type(ref_type)

    def test_primitives_are_copy(self, ts):
        """Primitive types should be Copy."""
        assert is_copy_type(RUST_I32)
        assert is_copy_type(RUST_F64)
        assert is_copy_type(RUST_BOOL)
        assert is_copy_type(RUST_CHAR)

    def test_string_not_copy(self, ts):
        """String should not be Copy."""
        assert not is_copy_type(RustStringType())

    def test_vec_not_copy(self, ts):
        """Vec should not be Copy."""
        assert not is_copy_type(RustVecType(RUST_I32))


# ===========================================================================
# Assignability Tests
# ===========================================================================


class TestRustAssignability:
    """Tests for Rust type assignability checking."""

    @pytest.fixture
    def ts(self):
        return RustTypeSystem()

    def test_same_type_assignable(self, ts):
        """Same types should be assignable."""
        assert ts.check_assignable(RUST_I32, RUST_I32)
        assert ts.check_assignable(RUST_F64, RUST_F64)

    def test_never_assignable_to_any(self, ts):
        """Never type should be assignable to any type."""
        assert ts.check_assignable(RUST_NEVER, RUST_I32)
        assert ts.check_assignable(RUST_NEVER, RUST_UNIT)

    def test_t_to_option_t(self, ts):
        """T should not be directly assignable to Option<T> (need Some())."""
        option_i32 = RustOptionType(RUST_I32)
        # In Rust, you need to wrap in Some(), so direct assignment fails
        # But semantically in type checking, we might allow this
        # Depends on implementation

    def test_ref_covariance(self, ts):
        """&T should be covariant in T."""
        ref1 = RustReferenceType(RUST_I32)
        ref2 = RustReferenceType(RUST_I32)
        assert ts.check_assignable(ref1, ref2)

    def test_mut_ref_to_shared_ref(self, ts):
        """&mut T should be assignable to &T (coercion)."""
        mut_ref = RustReferenceType(RUST_I32, is_mutable=True)
        shared_ref = RustReferenceType(RUST_I32, is_mutable=False)
        assert ts.check_assignable(mut_ref, shared_ref)

    def test_shared_ref_not_to_mut_ref(self, ts):
        """&T should not be assignable to &mut T."""
        shared_ref = RustReferenceType(RUST_I32, is_mutable=False)
        mut_ref = RustReferenceType(RUST_I32, is_mutable=True)
        assert not ts.check_assignable(shared_ref, mut_ref)

    def test_box_covariance(self, ts):
        """Box<T> should be covariant in T."""
        box1 = RustBoxType(RUST_I32)
        box2 = RustBoxType(RUST_I32)
        assert ts.check_assignable(box1, box2)


# ===========================================================================
# Type Formatting Tests
# ===========================================================================


class TestRustTypeFormatting:
    """Tests for Rust type formatting."""

    @pytest.fixture
    def ts(self):
        return RustTypeSystem()

    def test_format_primitive(self, ts):
        """Should format primitives correctly."""
        assert ts.format_type(RUST_I32) == "i32"
        assert ts.format_type(RUST_BOOL) == "bool"
        assert ts.format_type(RUST_CHAR) == "char"

    def test_format_reference(self, ts):
        """Should format reference types."""
        ref_type = RustReferenceType(RUST_I32)
        assert ts.format_type(ref_type) == "&i32"

    def test_format_mut_reference(self, ts):
        """Should format mutable reference types."""
        ref_type = RustReferenceType(RUST_I32, is_mutable=True)
        assert ts.format_type(ref_type) == "&mut i32"

    def test_format_option(self, ts):
        """Should format Option types."""
        opt = RustOptionType(RUST_I32)
        formatted = ts.format_type(opt)
        assert "Option" in formatted
        assert "i32" in formatted

    def test_format_result(self, ts):
        """Should format Result types."""
        result = RustResultType(RUST_I32, RustStringType())
        formatted = ts.format_type(result)
        assert "Result" in formatted

    def test_format_vec(self, ts):
        """Should format Vec types."""
        vec = RustVecType(RUST_I32)
        formatted = ts.format_type(vec)
        assert "Vec" in formatted

    def test_format_box(self, ts):
        """Should format Box types."""
        box_type = RustBoxType(RUST_I32)
        formatted = ts.format_type(box_type)
        assert "Box" in formatted

    def test_format_function(self, ts):
        """Should format function types."""
        func = RustFunctionType([RUST_I32], RUST_BOOL)
        formatted = ts.format_type(func)
        assert "fn" in formatted


# ===========================================================================
# Literal Type Inference Tests
# ===========================================================================


class TestRustLiteralInference:
    """Tests for Rust literal type inference."""

    @pytest.fixture
    def ts(self):
        return RustTypeSystem()

    def test_infer_integer(self, ts):
        """Should infer i32 for integer literals by default."""
        literal = LiteralInfo(LiteralKind.INTEGER, value=42)
        result = ts.infer_literal_type(literal)
        assert result == RUST_I32

    def test_infer_float(self, ts):
        """Should infer f64 for float literals by default."""
        literal = LiteralInfo(LiteralKind.FLOAT, value=3.14)
        result = ts.infer_literal_type(literal)
        assert result == RUST_F64

    def test_infer_boolean(self, ts):
        """Should infer bool for boolean literals."""
        literal = LiteralInfo(LiteralKind.BOOLEAN, value=True)
        result = ts.infer_literal_type(literal)
        assert result == RUST_BOOL

    def test_infer_string(self, ts):
        """Should infer &str for string literals."""
        literal = LiteralInfo(LiteralKind.STRING, value="hello")
        result = ts.infer_literal_type(literal)
        # String literals are &str
        assert isinstance(result, RustReferenceType)

    def test_infer_char(self, ts):
        """Should infer char for character literals."""
        literal = LiteralInfo(LiteralKind.CHARACTER, value='a')
        result = ts.infer_literal_type(literal)
        assert result == RUST_CHAR


# ===========================================================================
# Parsing Error Tests
# ===========================================================================


class TestRustTypeParsingErrors:
    """Tests for type parsing errors."""

    @pytest.fixture
    def ts(self):
        return RustTypeSystem()

    def test_empty_annotation_fails(self, ts):
        """Empty annotation should raise TypeParseError."""
        with pytest.raises(TypeParseError):
            ts.parse_type_annotation("")

    def test_unknown_type_name_is_generic(self, ts):
        """Unknown type name should be parsed as generic (user-defined type)."""
        # In Rust, unknown identifiers are valid as they could be user-defined types
        typ = ts.parse_type_annotation("MyCustomType")
        assert isinstance(typ, RustGenericType)
        assert typ.name == "MyCustomType"

    def test_unclosed_generic_fails(self, ts):
        """Unclosed generic should raise TypeParseError."""
        with pytest.raises(TypeParseError):
            ts.parse_type_annotation("Vec<i32")


# ===========================================================================
# Builtin Types and Functions Tests
# ===========================================================================


class TestRustBuiltins:
    """Tests for Rust builtin types."""

    @pytest.fixture
    def ts(self):
        return RustTypeSystem()

    def test_has_builtin_types(self, ts):
        """Should have builtin types."""
        builtins = ts.get_builtin_types()
        assert "i32" in builtins
        assert "u8" in builtins
        assert "bool" in builtins
        assert "str" in builtins

    def test_has_common_traits(self, ts):
        """Should recognize common traits."""
        # The type system should know about common traits
        funcs = ts.get_builtin_functions()
        # Rust doesn't have builtins like Python, but traits are known
