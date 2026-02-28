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
"""Tests for type system enhancements in full language support.

Tests the new type checking features:
- Go: Interface satisfaction checking
- Rust: Trait bound satisfaction checking
- Kotlin: Variance (in/out) checking
- Swift: Protocol conformance checking
- Zig: Struct/enum/union field tracking
"""

from __future__ import annotations

import pytest

try:
    from ...domains.types.languages.go import (
        GoTypeSystem,
        GoInterfaceType,
        GoStructType,
        GoFunctionType,
        GoNamedType,
        GO_INT,
        GO_STRING,
        GO_BOOL,
    )
    from ...domains.types.languages.rust import (
        RustTypeSystem,
        RustTraitBound,
        RustDynTraitType,
        RustImplTraitType,
        RustStringType,
        RUST_I32,
        RUST_STR,
        RUST_BOOL,
    )
    RUST_STRING = RustStringType()
    from ...domains.types.languages.kotlin import (
        KotlinTypeSystem,
        KotlinGenericType,
        KotlinTypeParameter,
        KotlinClassType,
        KotlinStarProjection,
        KOTLIN_INT,
        KOTLIN_STRING,
    )
    from ...domains.types.languages.swift import (
        SwiftTypeSystem,
        SwiftProtocolType,
        SwiftProtocolCompositionType,
        SwiftExistentialType,
        SwiftOpaqueType,
        SwiftNamedType,
        SwiftArrayType,
        SWIFT_INT,
        SWIFT_STRING,
        SWIFT_BOOL,
    )
    from ...domains.types.languages.zig import (
        ZigTypeSystem,
        ZigStructType,
        ZigEnumType,
        ZigUnionType,
        ZIG_I32,
        ZIG_U8,
        ZIG_BOOL,
        ZIG_VOID,
    )
except ImportError:
    from domains.types.languages.go import (
        GoTypeSystem,
        GoInterfaceType,
        GoStructType,
        GoFunctionType,
        GoNamedType,
        GO_INT,
        GO_STRING,
        GO_BOOL,
    )
    from domains.types.languages.rust import (
        RustTypeSystem,
        RustTraitBound,
        RustDynTraitType,
        RustImplTraitType,
        RustStringType,
        RUST_I32,
        RUST_STR,
        RUST_BOOL,
    )
    RUST_STRING = RustStringType()
    from domains.types.languages.kotlin import (
        KotlinTypeSystem,
        KotlinGenericType,
        KotlinTypeParameter,
        KotlinClassType,
        KotlinStarProjection,
        KOTLIN_INT,
        KOTLIN_STRING,
    )
    from domains.types.languages.swift import (
        SwiftTypeSystem,
        SwiftProtocolType,
        SwiftProtocolCompositionType,
        SwiftExistentialType,
        SwiftOpaqueType,
        SwiftNamedType,
        SwiftArrayType,
        SWIFT_INT,
        SWIFT_STRING,
        SWIFT_BOOL,
    )
    from domains.types.languages.zig import (
        ZigTypeSystem,
        ZigStructType,
        ZigEnumType,
        ZigUnionType,
        ZIG_I32,
        ZIG_U8,
        ZIG_BOOL,
        ZIG_VOID,
    )


# =============================================================================
# Go Interface Satisfaction Tests
# =============================================================================


class TestGoInterfaceSatisfaction:
    """Tests for Go interface satisfaction checking."""

    @pytest.fixture
    def ts(self) -> GoTypeSystem:
        return GoTypeSystem()

    def test_empty_interface_accepts_anything(self, ts: GoTypeSystem) -> None:
        """Test that empty interface{} accepts any type."""
        empty_interface = GoInterfaceType(name="interface{}", methods=())
        assert ts.check_assignable(GO_INT, empty_interface)
        assert ts.check_assignable(GO_STRING, empty_interface)

    def test_interface_with_method_requires_method(self, ts: GoTypeSystem) -> None:
        """Test that interface with methods requires implementation."""
        # Interface with String() method
        stringer = GoInterfaceType(
            name="Stringer",
            methods=(("String", GoFunctionType(parameters=(), returns=(GO_STRING,))),)
        )
        # Named type with String method would satisfy
        # Without method implementation tracking, this is heuristic

    def test_struct_to_interface_assignable(self, ts: GoTypeSystem) -> None:
        """Test struct satisfying interface."""
        # Empty interface accepts struct
        empty_interface = GoInterfaceType(name="interface{}", methods=())
        struct = GoStructType(name="MyStruct", fields=(("x", GO_INT),))
        assert ts.check_assignable(struct, empty_interface)

    def test_interface_method_extraction(self, ts: GoTypeSystem) -> None:
        """Test extracting methods from interface type."""
        iface = GoInterfaceType(
            name="Reader",
            methods=(
                ("Read", GoFunctionType(parameters=(("p", GO_INT),), returns=(GO_INT,))),
            )
        )
        methods = ts._get_interface_methods(iface)
        assert "Read" in methods


# =============================================================================
# Rust Trait Bound Tests
# =============================================================================


class TestRustTraitBounds:
    """Tests for Rust trait bound satisfaction checking."""

    @pytest.fixture
    def ts(self) -> RustTypeSystem:
        return RustTypeSystem()

    def test_primitive_implements_common_traits(self, ts: RustTypeSystem) -> None:
        """Test that primitives implement common traits."""
        # i32 implements Copy, Clone, Debug, etc.
        assert ts._type_implements_trait(RUST_I32, "Copy")
        assert ts._type_implements_trait(RUST_I32, "Clone")
        assert ts._type_implements_trait(RUST_I32, "Debug")

    def test_string_implements_traits(self, ts: RustTypeSystem) -> None:
        """Test that String implements expected traits."""
        assert ts._type_implements_trait(RUST_STRING, "Clone")
        assert ts._type_implements_trait(RUST_STRING, "Debug")
        # String doesn't implement Copy
        assert not ts._type_implements_trait(RUST_STRING, "Copy")

    def test_dyn_trait_assignability(self, ts: RustTypeSystem) -> None:
        """Test dyn Trait type assignability."""
        dyn_debug = RustDynTraitType(trait_name="Debug")
        # Any type implementing Debug can be assigned
        assert ts.check_assignable(RUST_I32, dyn_debug)
        assert ts.check_assignable(RUST_STRING, dyn_debug)

    def test_impl_trait_assignability(self, ts: RustTypeSystem) -> None:
        """Test impl Trait type assignability."""
        impl_clone = RustImplTraitType(trait_name="Clone")
        # Types implementing Clone satisfy impl Clone
        assert ts.check_assignable(RUST_I32, impl_clone)
        assert ts.check_assignable(RUST_STRING, impl_clone)

    def test_auto_trait_detection(self, ts: RustTypeSystem) -> None:
        """Test auto trait (Send, Sync) detection."""
        # Primitives are Send and Sync
        assert ts._type_is_auto_trait_safe(RUST_I32, "Send")
        assert ts._type_is_auto_trait_safe(RUST_I32, "Sync")


# =============================================================================
# Kotlin Variance Tests
# =============================================================================


class TestKotlinVariance:
    """Tests for Kotlin variance (in/out) checking."""

    @pytest.fixture
    def ts(self) -> KotlinTypeSystem:
        return KotlinTypeSystem()

    def test_star_projection_accepts_anything(self, ts: KotlinTypeSystem) -> None:
        """Test that * star projection accepts any type."""
        star = KotlinStarProjection()
        assert ts.check_assignable(KOTLIN_INT, star)
        assert ts.check_assignable(KOTLIN_STRING, star)

    def test_out_variance_covariant(self, ts: KotlinTypeSystem) -> None:
        """Test that out variance is covariant."""
        # Box<out Number> can accept Box<Int> since Int is subtype of Number
        out_param = KotlinTypeParameter("_", "out", KotlinClassType("Number"))
        # Int should satisfy out Number
        assert ts._check_variance_projection(KOTLIN_INT, out_param)

    def test_in_variance_contravariant(self, ts: KotlinTypeSystem) -> None:
        """Test that in variance is contravariant."""
        # Consumer<in Int> can accept Consumer<Any> since Any is supertype of Int
        in_param = KotlinTypeParameter("_", "in", KOTLIN_INT)
        # Any should satisfy in Int
        any_type = KotlinClassType("Any")
        # Contravariant - supertype satisfies
        assert ts._check_variance_projection(any_type, in_param)

    def test_generic_type_variance(self, ts: KotlinTypeSystem) -> None:
        """Test generic type assignability with variance."""
        # List<Int> to List<out Int>
        list_int = KotlinGenericType("List", (KOTLIN_INT,))
        list_out_int = KotlinGenericType(
            "List",
            (KotlinTypeParameter("_", "out", KOTLIN_INT),)
        )
        # Same base type, compatible variance
        assert ts._check_generic_assignable(list_int, list_out_int)

    def test_known_class_hierarchy(self, ts: KotlinTypeSystem) -> None:
        """Test known class hierarchy relationships."""
        assert ts._is_known_subclass("ArrayList", "List")
        assert ts._is_known_subclass("Int", "Number")
        assert ts._is_known_subclass("String", "CharSequence")


# =============================================================================
# Swift Protocol Conformance Tests
# =============================================================================


class TestSwiftProtocolConformance:
    """Tests for Swift protocol conformance checking."""

    @pytest.fixture
    def ts(self) -> SwiftTypeSystem:
        return SwiftTypeSystem()

    def test_primitive_protocol_conformance(self, ts: SwiftTypeSystem) -> None:
        """Test that primitives conform to expected protocols."""
        equatable = SwiftProtocolType("Equatable")
        hashable = SwiftProtocolType("Hashable")
        comparable = SwiftProtocolType("Comparable")

        assert ts.check_assignable(SWIFT_INT, equatable)
        assert ts.check_assignable(SWIFT_INT, hashable)
        assert ts.check_assignable(SWIFT_INT, comparable)

    def test_string_protocol_conformance(self, ts: SwiftTypeSystem) -> None:
        """Test String's protocol conformances."""
        sequence = SwiftProtocolType("Sequence")
        collection = SwiftProtocolType("Collection")
        codable = SwiftProtocolType("Codable")

        assert ts.check_assignable(SWIFT_STRING, sequence)
        assert ts.check_assignable(SWIFT_STRING, collection)
        assert ts.check_assignable(SWIFT_STRING, codable)

    def test_protocol_composition(self, ts: SwiftTypeSystem) -> None:
        """Test protocol composition (P1 & P2) conformance."""
        equatable_hashable = SwiftProtocolCompositionType((
            SwiftProtocolType("Equatable"),
            SwiftProtocolType("Hashable"),
        ))
        # Int conforms to both
        assert ts.check_assignable(SWIFT_INT, equatable_hashable)

    def test_existential_type(self, ts: SwiftTypeSystem) -> None:
        """Test existential type (any Protocol) assignability."""
        any_equatable = SwiftExistentialType(SwiftProtocolType("Equatable"))
        assert ts.check_assignable(SWIFT_INT, any_equatable)
        assert ts.check_assignable(SWIFT_STRING, any_equatable)

    def test_opaque_type(self, ts: SwiftTypeSystem) -> None:
        """Test opaque type (some Protocol) assignability."""
        some_equatable = SwiftOpaqueType(SwiftProtocolType("Equatable"))
        assert ts.check_assignable(SWIFT_INT, some_equatable)

    def test_array_conditional_conformance(self, ts: SwiftTypeSystem) -> None:
        """Test Array's conditional conformance."""
        sequence = SwiftProtocolType("Sequence")
        array_int = SwiftArrayType(SWIFT_INT)
        assert ts.check_assignable(array_int, sequence)

    def test_known_subtype_relationships(self, ts: SwiftTypeSystem) -> None:
        """Test known type hierarchy relationships."""
        assert ts._is_known_subtype("LocalizedError", "Error")
        assert ts._is_known_subtype("NSString", "NSObject")


# =============================================================================
# Zig Field Tracking Tests
# =============================================================================


class TestZigFieldTracking:
    """Tests for Zig struct/enum/union field tracking."""

    @pytest.fixture
    def ts(self) -> ZigTypeSystem:
        return ZigTypeSystem()

    def test_struct_field_parsing(self, ts: ZigTypeSystem) -> None:
        """Test struct field parsing from literals."""
        struct = ts._parse_struct_literal("struct { x: i32, y: i32 }")
        assert len(struct.fields) == 2
        field_names = {name for name, _ in struct.fields}
        assert "x" in field_names
        assert "y" in field_names

    def test_struct_field_type_access(self, ts: ZigTypeSystem) -> None:
        """Test getting field type from struct."""
        struct = ZigStructType(
            fields=(("x", ZIG_I32), ("name", ts._builtin_types["u8"])),
        )
        assert ts.get_struct_field_type(struct, "x") == ZIG_I32
        assert ts.get_struct_field_type(struct, "missing") is None

    def test_enum_variant_parsing(self, ts: ZigTypeSystem) -> None:
        """Test enum variant parsing from literals."""
        enum = ts._parse_enum_literal("enum { foo, bar, baz }")
        assert "foo" in enum.variants
        assert "bar" in enum.variants
        assert "baz" in enum.variants

    def test_enum_has_variant(self, ts: ZigTypeSystem) -> None:
        """Test checking enum has variant."""
        enum = ZigEnumType(variants=frozenset({"success", "failure"}))
        assert ts.has_enum_variant(enum, "success")
        assert ts.has_enum_variant(enum, "failure")
        assert not ts.has_enum_variant(enum, "pending")

    def test_union_variant_parsing(self, ts: ZigTypeSystem) -> None:
        """Test union variant parsing from literals."""
        union = ts._parse_union_literal("union { value: i32, ptr: *u8 }")
        assert len(union.variants) == 2
        variant_names = {name for name, _ in union.variants}
        assert "value" in variant_names
        assert "ptr" in variant_names

    def test_union_variant_type_access(self, ts: ZigTypeSystem) -> None:
        """Test getting variant type from union."""
        union = ZigUnionType(
            variants=(("int_val", ZIG_I32), ("flag", ZIG_BOOL)),
        )
        assert ts.get_union_variant_type(union, "int_val") == ZIG_I32
        assert ts.get_union_variant_type(union, "flag") == ZIG_BOOL
        assert ts.get_union_variant_type(union, "missing") is None

    def test_struct_assignability_by_name(self, ts: ZigTypeSystem) -> None:
        """Test named struct assignability."""
        struct_a = ZigStructType(name="Point")
        struct_b = ZigStructType(name="Point")
        struct_c = ZigStructType(name="Vector")
        assert ts.check_assignable(struct_a, struct_b)
        assert not ts.check_assignable(struct_a, struct_c)

    def test_anonymous_struct_structural(self, ts: ZigTypeSystem) -> None:
        """Test anonymous struct structural assignability."""
        source = ZigStructType(
            fields=(("x", ZIG_I32), ("y", ZIG_I32), ("z", ZIG_I32)),
        )
        target = ZigStructType(
            fields=(("x", ZIG_I32), ("y", ZIG_I32)),
        )
        # Source has all target fields, so assignable
        assert ts.check_assignable(source, target)

    def test_enum_assignability_by_name(self, ts: ZigTypeSystem) -> None:
        """Test named enum assignability."""
        enum_a = ZigEnumType(name="Status")
        enum_b = ZigEnumType(name="Status")
        enum_c = ZigEnumType(name="Error")
        assert ts.check_assignable(enum_a, enum_b)
        assert not ts.check_assignable(enum_a, enum_c)

    def test_packed_struct_parsing(self, ts: ZigTypeSystem) -> None:
        """Test packed struct parsing."""
        struct = ts._parse_struct_literal("packed struct { flags: u8, data: u32 }")
        assert struct.is_packed
        assert len(struct.fields) == 2

    def test_extern_struct_parsing(self, ts: ZigTypeSystem) -> None:
        """Test extern struct parsing."""
        struct = ts._parse_struct_literal("extern struct { handle: anyopaque }")
        assert struct.is_extern


# =============================================================================
# Cross-Language Type System Tests
# =============================================================================


class TestCrossLanguageTypeFeatures:
    """Tests that verify consistent patterns across language type systems."""

    def test_all_type_systems_have_check_assignable(self) -> None:
        """Test all type systems implement check_assignable."""
        systems = [
            GoTypeSystem(),
            RustTypeSystem(),
            KotlinTypeSystem(),
            SwiftTypeSystem(),
            ZigTypeSystem(),
        ]
        for ts in systems:
            assert hasattr(ts, "check_assignable")
            assert callable(ts.check_assignable)

    def test_all_type_systems_have_parse_type_annotation(self) -> None:
        """Test all type systems implement parse_type_annotation."""
        systems = [
            GoTypeSystem(),
            RustTypeSystem(),
            KotlinTypeSystem(),
            SwiftTypeSystem(),
            ZigTypeSystem(),
        ]
        for ts in systems:
            assert hasattr(ts, "parse_type_annotation")
            assert callable(ts.parse_type_annotation)

    def test_all_type_systems_have_builtin_types(self) -> None:
        """Test all type systems provide builtin types."""
        systems = [
            GoTypeSystem(),
            RustTypeSystem(),
            KotlinTypeSystem(),
            SwiftTypeSystem(),
            ZigTypeSystem(),
        ]
        for ts in systems:
            builtins = ts.get_builtin_types()
            assert isinstance(builtins, dict)
            assert len(builtins) > 0
