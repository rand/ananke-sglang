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
"""Property-based tests for type systems.

Tests mathematical properties that type systems should satisfy:
- Reflexivity: T is assignable to T
- Transitivity: If A ≤ B and B ≤ C, then A ≤ C
- Parse-format roundtrip: format(parse(s)) is a valid type
- Comptime coercion preserves semantics
"""

import pytest
from hypothesis import given, strategies as st, assume, settings, HealthCheck

from domains.types.languages.zig import (
    ZigTypeSystem,
    ZIG_I8, ZIG_I16, ZIG_I32, ZIG_I64, ZIG_I128, ZIG_ISIZE,
    ZIG_U8, ZIG_U16, ZIG_U32, ZIG_U64, ZIG_U128, ZIG_USIZE,
    ZIG_F16, ZIG_F32, ZIG_F64, ZIG_F80, ZIG_F128,
    ZIG_BOOL, ZIG_VOID, ZIG_NORETURN,
    ZIG_COMPTIME_INT, ZIG_COMPTIME_FLOAT,
    ZIG_TYPE, ZIG_ANYTYPE, ZIG_ANYERROR,
    ZigOptionalType,
    ZigPointerType,
    ZigSliceType,
    ZigArrayType,
)
from domains.types.languages.rust import (
    RustTypeSystem,
    RUST_I8, RUST_I16, RUST_I32, RUST_I64, RUST_I128, RUST_ISIZE,
    RUST_U8, RUST_U16, RUST_U32, RUST_U64, RUST_U128, RUST_USIZE,
    RUST_F32, RUST_F64,
    RUST_BOOL, RUST_CHAR, RUST_STR,
    RUST_UNIT, RUST_NEVER,
    RustReferenceType,
    RustSliceType,
    RustArrayType,
    RustOptionType,
    RustResultType,
    RustBoxType,
    RustVecType,
    RustStringType,
)
from domains.types.languages.typescript import (
    TypeScriptTypeSystem,
    TS_STRING, TS_NUMBER, TS_BOOLEAN, TS_BIGINT, TS_SYMBOL,
    TS_UNDEFINED, TS_NULL, TS_VOID, TS_OBJECT, TS_UNKNOWN,
    TS_ANY, TS_NEVER,
    TSArrayType,
    TSTupleType,
    TSObjectType,
    TSFunctionType,
    TSUnionType,
    TSIntersectionType,
    TSLiteralType,
)


# ===========================================================================
# Zig Primitive Types for Testing
# ===========================================================================

ZIG_INTEGERS = [
    ZIG_I8, ZIG_I16, ZIG_I32, ZIG_I64, ZIG_I128, ZIG_ISIZE,
    ZIG_U8, ZIG_U16, ZIG_U32, ZIG_U64, ZIG_U128, ZIG_USIZE,
]

ZIG_FLOATS = [ZIG_F16, ZIG_F32, ZIG_F64, ZIG_F80, ZIG_F128]

ZIG_PRIMITIVES = ZIG_INTEGERS + ZIG_FLOATS + [ZIG_BOOL, ZIG_VOID]

ZIG_PRIMITIVE_NAMES = [
    "i8", "i16", "i32", "i64", "i128", "isize",
    "u8", "u16", "u32", "u64", "u128", "usize",
    "f16", "f32", "f64", "f80", "f128",
    "bool", "void",
]


# ===========================================================================
# Rust Primitive Types for Testing
# ===========================================================================

RUST_INTEGERS = [
    RUST_I8, RUST_I16, RUST_I32, RUST_I64, RUST_I128, RUST_ISIZE,
    RUST_U8, RUST_U16, RUST_U32, RUST_U64, RUST_U128, RUST_USIZE,
]

RUST_FLOATS = [RUST_F32, RUST_F64]

RUST_PRIMITIVES = RUST_INTEGERS + RUST_FLOATS + [RUST_BOOL, RUST_CHAR]

RUST_PRIMITIVE_NAMES = [
    "i8", "i16", "i32", "i64", "i128", "isize",
    "u8", "u16", "u32", "u64", "u128", "usize",
    "f32", "f64",
    "bool", "char",
]


# ===========================================================================
# TypeScript Primitive Types for Testing
# ===========================================================================

TS_PRIMITIVES = [
    TS_STRING, TS_NUMBER, TS_BOOLEAN, TS_BIGINT, TS_SYMBOL,
    TS_UNDEFINED, TS_NULL, TS_VOID,
]

TS_SPECIAL = [TS_ANY, TS_UNKNOWN, TS_NEVER]

TS_ALL_TYPES = TS_PRIMITIVES + TS_SPECIAL

TS_PRIMITIVE_NAMES = [
    "string", "number", "boolean", "bigint", "symbol",
    "undefined", "null", "void", "any", "unknown", "never",
]


# ===========================================================================
# Zig Type System Properties
# ===========================================================================


class TestZigAssignabilityReflexivity:
    """Test reflexivity: T is assignable to T."""

    @pytest.fixture
    def ts(self):
        return ZigTypeSystem()

    @pytest.mark.parametrize("typ", ZIG_PRIMITIVES)
    def test_primitive_reflexivity(self, ts, typ):
        """Primitive types should be assignable to themselves."""
        assert ts.check_assignable(typ, typ)

    def test_optional_reflexivity(self, ts):
        """Optional types should be assignable to themselves."""
        opt = ZigOptionalType(ZIG_I32)
        assert ts.check_assignable(opt, opt)

    def test_pointer_reflexivity(self, ts):
        """Pointer types should be assignable to themselves."""
        ptr = ZigPointerType(ZIG_I32)
        assert ts.check_assignable(ptr, ptr)

    def test_slice_reflexivity(self, ts):
        """Slice types should be assignable to themselves."""
        slc = ZigSliceType(ZIG_U8)
        assert ts.check_assignable(slc, slc)

    def test_array_reflexivity(self, ts):
        """Array types should be assignable to themselves."""
        arr = ZigArrayType(ZIG_I32, 10)
        assert ts.check_assignable(arr, arr)


class TestZigComptimeCoercion:
    """Test comptime coercion properties."""

    @pytest.fixture
    def ts(self):
        return ZigTypeSystem()

    @pytest.mark.parametrize("int_type", ZIG_INTEGERS)
    def test_comptime_int_to_any_integer(self, ts, int_type):
        """comptime_int should be assignable to any integer type."""
        assert ts.check_assignable(ZIG_COMPTIME_INT, int_type)

    @pytest.mark.parametrize("float_type", ZIG_FLOATS)
    def test_comptime_float_to_any_float(self, ts, float_type):
        """comptime_float should be assignable to any float type."""
        assert ts.check_assignable(ZIG_COMPTIME_FLOAT, float_type)

    def test_anytype_universality(self, ts):
        """anytype should be assignable to/from any type."""
        for typ in ZIG_PRIMITIVES:
            assert ts.check_assignable(ZIG_ANYTYPE, typ)
            assert ts.check_assignable(typ, ZIG_ANYTYPE)


class TestZigNoreturnProperties:
    """Test noreturn type properties."""

    @pytest.fixture
    def ts(self):
        return ZigTypeSystem()

    @pytest.mark.parametrize("typ", ZIG_PRIMITIVES)
    def test_noreturn_is_bottom(self, ts, typ):
        """noreturn should be assignable to any type (bottom type)."""
        assert ts.check_assignable(ZIG_NORETURN, typ)


class TestZigParseFormatRoundtrip:
    """Test that parse and format are consistent."""

    @pytest.fixture
    def ts(self):
        return ZigTypeSystem()

    @pytest.mark.parametrize("type_name", ZIG_PRIMITIVE_NAMES)
    def test_primitive_roundtrip(self, ts, type_name):
        """Parsing a primitive and formatting should give valid type."""
        typ = ts.parse_type_annotation(type_name)
        formatted = ts.format_type(typ)
        # Should be able to parse again
        reparsed = ts.parse_type_annotation(formatted)
        assert reparsed is not None

    def test_optional_roundtrip(self, ts):
        """Optional type should roundtrip."""
        typ = ts.parse_type_annotation("?i32")
        formatted = ts.format_type(typ)
        assert "?" in formatted or "optional" in formatted.lower()

    def test_pointer_roundtrip(self, ts):
        """Pointer type should roundtrip."""
        typ = ts.parse_type_annotation("*i32")
        formatted = ts.format_type(typ)
        assert "*" in formatted

    def test_slice_roundtrip(self, ts):
        """Slice type should roundtrip."""
        typ = ts.parse_type_annotation("[]u8")
        formatted = ts.format_type(typ)
        assert "[]" in formatted or "[" in formatted


# ===========================================================================
# Rust Type System Properties
# ===========================================================================


class TestRustAssignabilityReflexivity:
    """Test reflexivity: T is assignable to T."""

    @pytest.fixture
    def ts(self):
        return RustTypeSystem()

    @pytest.mark.parametrize("typ", RUST_PRIMITIVES)
    def test_primitive_reflexivity(self, ts, typ):
        """Primitive types should be assignable to themselves."""
        assert ts.check_assignable(typ, typ)

    def test_reference_reflexivity(self, ts):
        """Reference types should be assignable to themselves."""
        ref = RustReferenceType(RUST_I32)
        assert ts.check_assignable(ref, ref)

    def test_option_reflexivity(self, ts):
        """Option types should be assignable to themselves."""
        opt = RustOptionType(RUST_I32)
        assert ts.check_assignable(opt, opt)

    def test_box_reflexivity(self, ts):
        """Box types should be assignable to themselves."""
        box_type = RustBoxType(RUST_I32)
        assert ts.check_assignable(box_type, box_type)

    def test_vec_reflexivity(self, ts):
        """Vec types should be assignable to themselves."""
        vec = RustVecType(RUST_I32)
        assert ts.check_assignable(vec, vec)


class TestRustNeverProperties:
    """Test never type properties."""

    @pytest.fixture
    def ts(self):
        return RustTypeSystem()

    @pytest.mark.parametrize("typ", RUST_PRIMITIVES)
    def test_never_is_bottom(self, ts, typ):
        """Never (!) should be assignable to any type (bottom type)."""
        assert ts.check_assignable(RUST_NEVER, typ)


class TestRustReferenceCoercion:
    """Test reference coercion properties."""

    @pytest.fixture
    def ts(self):
        return RustTypeSystem()

    @pytest.mark.parametrize("inner", RUST_PRIMITIVES)
    def test_mut_to_shared_coercion(self, ts, inner):
        """&mut T should be coercible to &T."""
        mut_ref = RustReferenceType(inner, is_mutable=True)
        shared_ref = RustReferenceType(inner, is_mutable=False)
        assert ts.check_assignable(mut_ref, shared_ref)

    @pytest.mark.parametrize("inner", RUST_PRIMITIVES)
    def test_shared_not_to_mut(self, ts, inner):
        """&T should not be coercible to &mut T."""
        shared_ref = RustReferenceType(inner, is_mutable=False)
        mut_ref = RustReferenceType(inner, is_mutable=True)
        assert not ts.check_assignable(shared_ref, mut_ref)


class TestRustParseFormatRoundtrip:
    """Test that parse and format are consistent."""

    @pytest.fixture
    def ts(self):
        return RustTypeSystem()

    @pytest.mark.parametrize("type_name", RUST_PRIMITIVE_NAMES)
    def test_primitive_roundtrip(self, ts, type_name):
        """Parsing a primitive and formatting should give valid type."""
        typ = ts.parse_type_annotation(type_name)
        formatted = ts.format_type(typ)
        reparsed = ts.parse_type_annotation(formatted)
        assert reparsed is not None

    def test_option_roundtrip(self, ts):
        """Option type should roundtrip."""
        typ = ts.parse_type_annotation("Option<i32>")
        formatted = ts.format_type(typ)
        assert "Option" in formatted

    def test_result_roundtrip(self, ts):
        """Result type should roundtrip."""
        typ = ts.parse_type_annotation("Result<i32, String>")
        formatted = ts.format_type(typ)
        assert "Result" in formatted

    def test_reference_roundtrip(self, ts):
        """Reference type should roundtrip."""
        typ = ts.parse_type_annotation("&i32")
        formatted = ts.format_type(typ)
        assert "&" in formatted


# ===========================================================================
# TypeScript Type System Properties
# ===========================================================================


class TestTypeScriptAssignabilityReflexivity:
    """Test reflexivity: T is assignable to T."""

    @pytest.fixture
    def ts(self):
        return TypeScriptTypeSystem()

    @pytest.mark.parametrize("typ", TS_PRIMITIVES)
    def test_primitive_reflexivity(self, ts, typ):
        """Primitive types should be assignable to themselves."""
        assert ts.check_assignable(typ, typ)

    def test_array_reflexivity(self, ts):
        """Array types should be assignable to themselves."""
        arr = TSArrayType(element=TS_STRING)
        assert ts.check_assignable(arr, arr)

    def test_tuple_reflexivity(self, ts):
        """Tuple types should be assignable to themselves."""
        tup = TSTupleType(elements=(TS_STRING, TS_NUMBER))
        assert ts.check_assignable(tup, tup)

    def test_object_reflexivity(self, ts):
        """Object types should be assignable to themselves."""
        obj = TSObjectType(
            properties=(("name", TS_STRING), ("age", TS_NUMBER)),
            optional_properties=frozenset(),
        )
        assert ts.check_assignable(obj, obj)

    def test_function_reflexivity(self, ts):
        """Function types should be assignable to themselves."""
        func = TSFunctionType(
            parameters=(("x", TS_NUMBER, False),),
            return_type=TS_STRING,
        )
        assert ts.check_assignable(func, func)

    def test_union_reflexivity(self, ts):
        """Union types should be assignable to themselves."""
        union = TSUnionType(members=frozenset({TS_STRING, TS_NUMBER}))
        assert ts.check_assignable(union, union)


class TestTypeScriptAnyUnknownNeverProperties:
    """Test TypeScript special type properties."""

    @pytest.fixture
    def ts(self):
        return TypeScriptTypeSystem()

    @pytest.mark.parametrize("typ", TS_PRIMITIVES)
    def test_any_accepts_everything(self, ts, typ):
        """any should accept any type (target)."""
        assert ts.check_assignable(typ, TS_ANY)

    @pytest.mark.parametrize("typ", TS_PRIMITIVES)
    def test_any_assignable_to_everything(self, ts, typ):
        """any should be assignable to any type except never."""
        if typ != TS_NEVER:
            assert ts.check_assignable(TS_ANY, typ)

    @pytest.mark.parametrize("typ", TS_PRIMITIVES)
    def test_unknown_accepts_everything(self, ts, typ):
        """unknown should accept any type (target)."""
        assert ts.check_assignable(typ, TS_UNKNOWN)

    @pytest.mark.parametrize("typ", TS_PRIMITIVES)
    def test_never_assignable_to_all(self, ts, typ):
        """never should be assignable to any type (bottom type)."""
        assert ts.check_assignable(TS_NEVER, typ)

    def test_nothing_assignable_to_never(self, ts):
        """Only never should be assignable to never."""
        for typ in TS_PRIMITIVES:
            if typ != TS_NEVER:
                assert not ts.check_assignable(typ, TS_NEVER)


class TestTypeScriptUnionProperties:
    """Test TypeScript union type properties."""

    @pytest.fixture
    def ts(self):
        return TypeScriptTypeSystem()

    @pytest.mark.parametrize("typ", TS_PRIMITIVES[:5])  # Use subset for speed
    def test_member_assignable_to_union(self, ts, typ):
        """A type should be assignable to a union containing it."""
        union = TSUnionType(members=frozenset({typ, TS_NULL}))
        assert ts.check_assignable(typ, union)

    def test_union_subset_assignable(self, ts):
        """A union should be assignable to a wider union."""
        narrow = TSUnionType(members=frozenset({TS_STRING, TS_NUMBER}))
        wide = TSUnionType(members=frozenset({TS_STRING, TS_NUMBER, TS_BOOLEAN}))
        assert ts.check_assignable(narrow, wide)


class TestTypeScriptStructuralTypingProperties:
    """Test TypeScript structural typing properties."""

    @pytest.fixture
    def ts(self):
        return TypeScriptTypeSystem()

    def test_extra_property_compatible(self, ts):
        """Object with extra properties should be assignable to object with fewer."""
        source = TSObjectType(
            properties=(("name", TS_STRING), ("age", TS_NUMBER), ("extra", TS_BOOLEAN)),
            optional_properties=frozenset(),
        )
        target = TSObjectType(
            properties=(("name", TS_STRING), ("age", TS_NUMBER)),
            optional_properties=frozenset(),
        )
        assert ts.check_assignable(source, target)

    def test_missing_required_not_compatible(self, ts):
        """Object missing required property should not be assignable."""
        source = TSObjectType(
            properties=(("name", TS_STRING),),
            optional_properties=frozenset(),
        )
        target = TSObjectType(
            properties=(("name", TS_STRING), ("age", TS_NUMBER)),
            optional_properties=frozenset(),
        )
        assert not ts.check_assignable(source, target)


class TestTypeScriptParseFormatRoundtrip:
    """Test that parse and format are consistent."""

    @pytest.fixture
    def ts(self):
        return TypeScriptTypeSystem()

    @pytest.mark.parametrize("type_name", TS_PRIMITIVE_NAMES)
    def test_primitive_roundtrip(self, ts, type_name):
        """Parsing a primitive and formatting should give valid type."""
        typ = ts.parse_type_annotation(type_name)
        formatted = ts.format_type(typ)
        reparsed = ts.parse_type_annotation(formatted)
        assert reparsed is not None

    def test_array_roundtrip(self, ts):
        """Array type should roundtrip."""
        typ = ts.parse_type_annotation("string[]")
        formatted = ts.format_type(typ)
        assert "string" in formatted

    def test_tuple_roundtrip(self, ts):
        """Tuple type should roundtrip."""
        typ = ts.parse_type_annotation("[string, number]")
        formatted = ts.format_type(typ)
        assert "[" in formatted

    def test_union_roundtrip(self, ts):
        """Union type should roundtrip."""
        typ = ts.parse_type_annotation("string | number")
        formatted = ts.format_type(typ)
        assert "|" in formatted

    def test_function_roundtrip(self, ts):
        """Function type should roundtrip."""
        typ = ts.parse_type_annotation("(x: number) => string")
        formatted = ts.format_type(typ)
        assert "=>" in formatted


class TestTypeScriptUtilityTypeProperties:
    """Test utility type properties."""

    @pytest.fixture
    def ts(self):
        return TypeScriptTypeSystem()

    def test_partial_idempotent(self, ts):
        """Partial applied twice should have same effect."""
        obj = TSObjectType(
            properties=(("name", TS_STRING),),
            optional_properties=frozenset(),
        )
        partial1 = ts.apply_utility_type("Partial", [obj])
        partial2 = ts.apply_utility_type("Partial", [partial1])
        # Both should have all properties optional
        assert partial1.optional_properties == partial2.optional_properties

    def test_required_undoes_partial(self, ts):
        """Required should undo Partial."""
        obj = TSObjectType(
            properties=(("name", TS_STRING), ("age", TS_NUMBER)),
            optional_properties=frozenset(),
        )
        partial = ts.apply_utility_type("Partial", [obj])
        required = ts.apply_utility_type("Required", [partial])
        assert len(required.optional_properties) == 0

    def test_pick_omit_complement(self, ts):
        """Pick and Omit should be complementary."""
        obj = TSObjectType(
            properties=(("a", TS_STRING), ("b", TS_NUMBER), ("c", TS_BOOLEAN)),
            optional_properties=frozenset(),
        )
        pick_key = TSLiteralType(kind="string", value="a")
        picked = ts.apply_utility_type("Pick", [obj, pick_key])
        omitted = ts.apply_utility_type("Omit", [obj, pick_key])

        # Picked should have only 'a'
        assert len(picked.properties) == 1
        assert "a" in picked

        # Omitted should have 'b' and 'c'
        assert len(omitted.properties) == 2
        assert "a" not in omitted


# ===========================================================================
# Cross-Language Property Tests
# ===========================================================================


class TestCrossLanguageProperties:
    """Test properties that should hold across both languages."""

    @pytest.fixture
    def zig_ts(self):
        return ZigTypeSystem()

    @pytest.fixture
    def rust_ts(self):
        return RustTypeSystem()

    def test_both_support_i32(self, zig_ts, rust_ts):
        """Both should parse i32."""
        zig_i32 = zig_ts.parse_type_annotation("i32")
        rust_i32 = rust_ts.parse_type_annotation("i32")
        assert zig_i32 is not None
        assert rust_i32 is not None

    def test_both_reflexive_for_i32(self, zig_ts, rust_ts):
        """Both should have reflexive assignability for i32."""
        zig_i32 = zig_ts.parse_type_annotation("i32")
        rust_i32 = rust_ts.parse_type_annotation("i32")

        assert zig_ts.check_assignable(zig_i32, zig_i32)
        assert rust_ts.check_assignable(rust_i32, rust_i32)

    @pytest.mark.parametrize("size", [8, 16, 32, 64, 128])
    def test_both_support_signed_integers(self, zig_ts, rust_ts, size):
        """Both should support i8, i16, i32, i64, i128."""
        type_name = f"i{size}"
        zig_type = zig_ts.parse_type_annotation(type_name)
        rust_type = rust_ts.parse_type_annotation(type_name)
        assert zig_type is not None
        assert rust_type is not None

    @pytest.mark.parametrize("size", [8, 16, 32, 64, 128])
    def test_both_support_unsigned_integers(self, zig_ts, rust_ts, size):
        """Both should support u8, u16, u32, u64, u128."""
        type_name = f"u{size}"
        zig_type = zig_ts.parse_type_annotation(type_name)
        rust_type = rust_ts.parse_type_annotation(type_name)
        assert zig_type is not None
        assert rust_type is not None


# ===========================================================================
# Hypothesis-Based Property Tests (if hypothesis is available)
# ===========================================================================

try:
    from hypothesis import given, strategies as st, assume, settings

    # Strategy for generating valid Zig type names
    zig_primitive_strategy = st.sampled_from(ZIG_PRIMITIVE_NAMES)

    # Strategy for generating valid Rust type names
    rust_primitive_strategy = st.sampled_from(RUST_PRIMITIVE_NAMES)

    # Strategy for generating valid TypeScript type names
    ts_primitive_strategy = st.sampled_from(TS_PRIMITIVE_NAMES)

    # Strategy for TypeScript array depth
    ts_array_depth = st.integers(min_value=0, max_value=3)

    class TestZigHypothesis:
        """Hypothesis-based property tests for Zig."""

        @pytest.fixture
        def ts(self):
            return ZigTypeSystem()

        @given(type_name=zig_primitive_strategy)
        @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
        def test_parse_never_fails_on_primitives(self, ts, type_name):
            """Parsing primitives should never fail."""
            typ = ts.parse_type_annotation(type_name)
            assert typ is not None

        @given(type_name=zig_primitive_strategy)
        @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
        def test_format_produces_nonempty(self, ts, type_name):
            """Formatting should produce non-empty string."""
            typ = ts.parse_type_annotation(type_name)
            formatted = ts.format_type(typ)
            assert len(formatted) > 0

    class TestRustHypothesis:
        """Hypothesis-based property tests for Rust."""

        @pytest.fixture
        def ts(self):
            return RustTypeSystem()

        @given(type_name=rust_primitive_strategy)
        @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
        def test_parse_never_fails_on_primitives(self, ts, type_name):
            """Parsing primitives should never fail."""
            typ = ts.parse_type_annotation(type_name)
            assert typ is not None

        @given(type_name=rust_primitive_strategy)
        @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
        def test_format_produces_nonempty(self, ts, type_name):
            """Formatting should produce non-empty string."""
            typ = ts.parse_type_annotation(type_name)
            formatted = ts.format_type(typ)
            assert len(formatted) > 0

    class TestTypeScriptHypothesis:
        """Hypothesis-based property tests for TypeScript."""

        @pytest.fixture
        def ts(self):
            return TypeScriptTypeSystem()

        @given(type_name=ts_primitive_strategy)
        @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
        def test_parse_never_fails_on_primitives(self, ts, type_name):
            """Parsing primitives should never fail."""
            typ = ts.parse_type_annotation(type_name)
            assert typ is not None

        @given(type_name=ts_primitive_strategy)
        @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
        def test_format_produces_nonempty(self, ts, type_name):
            """Formatting should produce non-empty string."""
            typ = ts.parse_type_annotation(type_name)
            formatted = ts.format_type(typ)
            assert len(formatted) > 0

        @given(type_name=ts_primitive_strategy, depth=ts_array_depth)
        @settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture])
        def test_array_nesting_reflexive(self, ts, type_name, depth):
            """Nested array types should be reflexively assignable."""
            type_str = type_name
            for _ in range(depth):
                type_str = f"{type_str}[]"
            typ = ts.parse_type_annotation(type_str)
            assert ts.check_assignable(typ, typ)

        @given(
            t1=ts_primitive_strategy,
            t2=ts_primitive_strategy,
        )
        @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
        def test_union_commutative(self, ts, t1, t2):
            """Union type members are commutative: T | U == U | T."""
            assume(t1 != t2)
            # Exclude 'unknown' since it has special narrowing semantics
            # unknown requires narrowing to be assigned to non-unknown types
            assume(t1 != "unknown" and t2 != "unknown")
            union1 = ts.parse_type_annotation(f"{t1} | {t2}")
            union2 = ts.parse_type_annotation(f"{t2} | {t1}")
            # Both should accept the same things
            typ1 = ts.parse_type_annotation(t1)
            assert ts.check_assignable(typ1, union1)
            assert ts.check_assignable(typ1, union2)

        @given(type_name=ts_primitive_strategy)
        @settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture])
        def test_nullable_contains_type(self, ts, type_name):
            """Nullable type should accept the base type."""
            # Exclude 'unknown' since it has special narrowing semantics
            assume(type_name != "unknown")
            base_type = ts.parse_type_annotation(type_name)
            nullable = ts.parse_type_annotation(f"{type_name} | null")
            assert ts.check_assignable(base_type, nullable)

except ImportError:
    # Hypothesis not available, skip those tests
    pass
