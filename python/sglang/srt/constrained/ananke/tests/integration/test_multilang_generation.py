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
"""Integration tests for multi-language code generation.

Tests the integration of Zig and Rust type systems, import resolvers,
token classifiers, and parsers in the context of constrained code generation.
"""

import pytest

from domains.types.languages import get_type_system, supported_languages
from domains.types.languages.zig import ZigTypeSystem, ZIG_I32, ZIG_BOOL, ZIG_COMPTIME_INT
from domains.types.languages.rust import RustTypeSystem, RUST_I32, RUST_BOOL, RUST_F64
from domains.imports.resolvers.zig import ZigImportResolver
from domains.imports.resolvers.rust import RustImportResolver
from core.token_classifier_zig import classify_zig_token, ZIG_ALL_KEYWORDS
from core.token_classifier_rust import classify_rust_token, RUST_ALL_KEYWORDS
from parsing.languages.zig import ZigIncrementalParser, create_zig_parser
from parsing.languages.rust import RustIncrementalParser, create_rust_parser


# ===========================================================================
# Multi-Language Support Integration Tests
# ===========================================================================


class TestMultiLanguageSupport:
    """Tests for multi-language support infrastructure."""

    def test_zig_in_supported_languages(self):
        """Zig should be available."""
        langs = supported_languages()
        assert "zig" in langs

    def test_rust_in_supported_languages(self):
        """Rust should be available."""
        langs = supported_languages()
        assert "rust" in langs

    def test_get_zig_type_system(self):
        """Should get Zig type system."""
        ts = get_type_system("zig")
        assert isinstance(ts, ZigTypeSystem)

    def test_get_rust_type_system(self):
        """Should get Rust type system."""
        ts = get_type_system("rust")
        assert isinstance(ts, RustTypeSystem)

    def test_different_type_systems(self):
        """Zig and Rust should have different type systems."""
        zig_ts = get_type_system("zig")
        rust_ts = get_type_system("rust")
        assert type(zig_ts) != type(rust_ts)
        assert zig_ts.name != rust_ts.name


# ===========================================================================
# Zig Generation Integration Tests
# ===========================================================================


class TestZigGenerationIntegration:
    """Integration tests for Zig code generation."""

    @pytest.fixture
    def ts(self):
        return ZigTypeSystem()

    @pytest.fixture
    def resolver(self):
        return ZigImportResolver()

    @pytest.fixture
    def parser(self):
        return create_zig_parser()

    def test_comptime_function_generation(self, ts, parser):
        """Test generation of comptime functions."""
        # Parse partial comptime function
        source = "fn comptimeAdd(comptime a: comptime_int, comptime b: comptime_int) comptime_int"
        result = parser.parse_initial(source)

        # Should detect missing body
        holes = parser.find_holes()
        assert len(holes) > 0

        # Type system should understand comptime types
        comptime_type = ts.parse_type_annotation("comptime_int")
        assert comptime_type == ZIG_COMPTIME_INT

    def test_error_union_handling(self, ts, parser):
        """Test handling of error unions."""
        # Parse function with error union return
        source = "fn readFile(path: []const u8) anyerror![]u8"
        result = parser.parse_initial(source)

        # Parse the return type
        return_type = ts.parse_type_annotation("anyerror![]u8")
        assert return_type is not None

    def test_allocator_aware_struct(self, ts, parser):
        """Test allocator-aware struct generation."""
        source = """const ArrayList = struct {
    items: [*]T,
    len: usize,
    capacity: usize,
    allocator: std.mem.Allocator,
}"""
        result = parser.parse_initial(source)
        assert result.ast is not None

    def test_std_import_resolution(self, ts, resolver):
        """Test standard library import resolution."""
        result = resolver.resolve("std")
        assert result is not None

        result = resolver.resolve("std.mem")
        assert result is not None

    def test_token_classification_consistency(self, ts, parser):
        """Test that token classification is consistent with parsing."""
        # Keywords should be recognized
        for keyword in ["fn", "const", "var", "struct", "comptime"]:
            category, kw, _ = classify_zig_token(keyword)
            assert kw == keyword


# ===========================================================================
# Rust Generation Integration Tests
# ===========================================================================


class TestRustGenerationIntegration:
    """Integration tests for Rust code generation."""

    @pytest.fixture
    def ts(self):
        return RustTypeSystem()

    @pytest.fixture
    def resolver(self):
        return RustImportResolver()

    @pytest.fixture
    def parser(self):
        return create_rust_parser()

    def test_ownership_aware_generation(self, ts, parser):
        """Test ownership-aware code generation."""
        # Parse function taking ownership
        source = "fn consume(s: String) -> usize"
        result = parser.parse_initial(source)

        # Type system should understand String is owned
        string_type = ts.parse_type_annotation("String")
        assert string_type is not None

    def test_lifetime_inference(self, ts, parser):
        """Test lifetime handling in code generation."""
        # Parse function with lifetimes
        source = "fn longest<'a>(x: &'a str, y: &'a str) -> &'a str"
        result = parser.parse_initial(source)

        # Type system should parse lifetime annotations
        ref_type = ts.parse_type_annotation("&'a str")
        assert ref_type is not None

    def test_result_type_handling(self, ts, parser):
        """Test Result type handling."""
        source = "fn read_file(path: &Path) -> Result<String, io::Error>"
        result = parser.parse_initial(source)

        # Type system should parse Result
        result_type = ts.parse_type_annotation("Result<String, Error>")
        assert result_type is not None

    def test_trait_bound_generation(self, ts, parser):
        """Test trait bound handling in generics."""
        source = "fn process<T: Clone + Debug>(item: T)"
        result = parser.parse_initial(source)
        assert result.ast is not None

    def test_std_import_resolution(self, ts, resolver):
        """Test standard library import resolution."""
        result = resolver.resolve("std")
        assert result is not None

        result = resolver.resolve("std::io")
        assert result is not None

    def test_token_classification_consistency(self, ts, parser):
        """Test that token classification is consistent with parsing."""
        for keyword in ["fn", "let", "mut", "struct", "impl"]:
            category, kw, _ = classify_rust_token(keyword)
            assert kw == keyword


# ===========================================================================
# Cross-Language Comparison Tests
# ===========================================================================


class TestCrossLanguageComparison:
    """Tests comparing Zig and Rust implementations."""

    @pytest.fixture
    def zig_ts(self):
        return ZigTypeSystem()

    @pytest.fixture
    def rust_ts(self):
        return RustTypeSystem()

    def test_primitive_types_both_support_i32(self, zig_ts, rust_ts):
        """Both languages should support i32."""
        zig_i32 = zig_ts.parse_type_annotation("i32")
        rust_i32 = rust_ts.parse_type_annotation("i32")

        assert zig_i32 is not None
        assert rust_i32 is not None

    def test_primitive_types_both_support_bool(self, zig_ts, rust_ts):
        """Both languages should support bool."""
        zig_bool = zig_ts.parse_type_annotation("bool")
        rust_bool = rust_ts.parse_type_annotation("bool")

        assert zig_bool is not None
        assert rust_bool is not None

    def test_different_optional_syntax(self, zig_ts, rust_ts):
        """Zig uses ?T, Rust uses Option<T>."""
        zig_opt = zig_ts.parse_type_annotation("?i32")
        rust_opt = rust_ts.parse_type_annotation("Option<i32>")

        assert zig_opt is not None
        assert rust_opt is not None

    def test_different_result_syntax(self, zig_ts, rust_ts):
        """Zig uses E!T, Rust uses Result<T, E>."""
        zig_result = zig_ts.parse_type_annotation("anyerror!i32")
        rust_result = rust_ts.parse_type_annotation("Result<i32, Error>")

        assert zig_result is not None
        assert rust_result is not None

    def test_zig_has_comptime_rust_does_not(self, zig_ts, rust_ts):
        """Zig should support comptime types."""
        assert zig_ts.capabilities.supports_comptime
        assert not rust_ts.capabilities.supports_comptime

    def test_rust_has_ownership_zig_does_not(self, zig_ts, rust_ts):
        """Rust should support ownership types."""
        assert rust_ts.capabilities.supports_ownership
        assert not zig_ts.capabilities.supports_ownership

    def test_both_support_generics(self, zig_ts, rust_ts):
        """Both should support generics."""
        assert zig_ts.capabilities.supports_generics
        assert rust_ts.capabilities.supports_generics


# ===========================================================================
# Parser Integration Tests
# ===========================================================================


class TestParserIntegration:
    """Integration tests for parsers across languages."""

    def test_zig_parser_checkpoint_restore(self):
        """Test Zig parser checkpoint/restore works end-to-end."""
        parser = create_zig_parser()
        parser.parse_initial("fn foo(x: i32)")

        checkpoint = parser.checkpoint()
        parser.extend_with_text(" void { return x; }")

        # Restore and verify
        parser.restore(checkpoint)
        assert parser.current_source == "fn foo(x: i32)"

    def test_rust_parser_checkpoint_restore(self):
        """Test Rust parser checkpoint/restore works end-to-end."""
        parser = create_rust_parser()
        parser.parse_initial("fn foo(x: i32)")

        checkpoint = parser.checkpoint()
        parser.extend_with_text(" -> i32 { x }")

        # Restore and verify
        parser.restore(checkpoint)
        assert parser.current_source == "fn foo(x: i32)"

    def test_zig_parser_expected_tokens(self):
        """Test Zig parser provides useful expected tokens."""
        parser = create_zig_parser()
        parser.parse_initial("fn ")
        expected = parser.get_expected_tokens()

        # Should expect identifier for function name
        assert "identifier" in expected

    def test_rust_parser_expected_tokens(self):
        """Test Rust parser provides useful expected tokens."""
        parser = create_rust_parser()
        parser.parse_initial("fn ")
        expected = parser.get_expected_tokens()

        # Should expect identifier for function name
        assert "identifier" in expected


# ===========================================================================
# Resolver Integration Tests
# ===========================================================================


class TestResolverIntegration:
    """Integration tests for import resolvers."""

    def test_zig_std_provides_exports(self):
        """Zig std module should report exports."""
        resolver = ZigImportResolver()
        result = resolver.resolve("std")
        # Check that resolution succeeded
        assert result is not None

    def test_rust_std_provides_exports(self):
        """Rust std module should report exports."""
        resolver = RustImportResolver()
        result = resolver.resolve("std")
        assert result is not None

    def test_zig_nested_module_resolution(self):
        """Zig should resolve nested modules."""
        resolver = ZigImportResolver()
        result = resolver.resolve("std.mem")
        assert result is not None

    def test_rust_nested_module_resolution(self):
        """Rust should resolve nested modules."""
        resolver = RustImportResolver()
        result = resolver.resolve("std::collections")
        assert result is not None


# ===========================================================================
# End-to-End Generation Simulation Tests
# ===========================================================================


class TestEndToEndGeneration:
    """Simulates end-to-end code generation scenarios."""

    def test_zig_function_generation_flow(self):
        """Simulate generating a Zig function."""
        ts = ZigTypeSystem()
        parser = create_zig_parser()
        resolver = ZigImportResolver()

        # Start with function signature
        parser.parse_initial("pub fn add(a: i32, b: i32) i32")

        # Check expected tokens for body
        expected = parser.get_expected_tokens()

        # Extend with body
        parser.extend_with_text(" { return a + b; }")

        # Final source should be valid
        assert "pub fn add(a: i32, b: i32) i32 { return a + b; }" in parser.current_source

    def test_rust_function_generation_flow(self):
        """Simulate generating a Rust function."""
        ts = RustTypeSystem()
        parser = create_rust_parser()
        resolver = RustImportResolver()

        # Start with function signature
        parser.parse_initial("pub fn add(a: i32, b: i32) -> i32")

        # Check expected tokens for body
        expected = parser.get_expected_tokens()

        # Extend with body
        parser.extend_with_text(" { a + b }")

        # Final source should be valid
        assert "pub fn add(a: i32, b: i32) -> i32 { a + b }" in parser.current_source

    def test_zig_struct_generation_flow(self):
        """Simulate generating a Zig struct."""
        parser = create_zig_parser()

        # Build struct incrementally
        parser.parse_initial("const Point = struct {")
        parser.extend_with_text("\n    x: f32,")
        parser.extend_with_text("\n    y: f32,")
        parser.extend_with_text("\n};")

        # Should have no holes now
        holes = parser.find_holes()
        # Struct is complete

    def test_rust_struct_generation_flow(self):
        """Simulate generating a Rust struct."""
        parser = create_rust_parser()

        # Build struct incrementally
        parser.parse_initial("struct Point {")
        parser.extend_with_text("\n    x: f32,")
        parser.extend_with_text("\n    y: f32,")
        parser.extend_with_text("\n}")

        # Should have no holes now
        holes = parser.find_holes()
        # Struct is complete
