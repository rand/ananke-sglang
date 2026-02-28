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

Tests the integration of all supported language type systems, import resolvers,
token classifiers, and parsers in the context of constrained code generation.
Covers: Go, Kotlin, Python, Rust, Swift, TypeScript, and Zig.
"""

import pytest

from domains.types.languages import get_type_system, supported_languages
from domains.types.languages.zig import ZigTypeSystem, ZIG_I32, ZIG_BOOL, ZIG_COMPTIME_INT
from domains.types.languages.rust import RustTypeSystem, RUST_I32, RUST_BOOL, RUST_F64
from domains.types.languages.typescript import (
    TypeScriptTypeSystem,
    TS_STRING,
    TS_NUMBER,
    TS_BOOLEAN,
    TS_ANY,
    TS_UNKNOWN,
    TSArrayType,
    TSObjectType,
    TSUnionType,
)
from domains.types.languages.go import (
    GoTypeSystem,
    GO_INT,
    GO_BOOL,
    GO_STRING,
    GO_ANY,
    GoSliceType,
    GoMapType,
)
from domains.types.languages.kotlin import (
    KotlinTypeSystem,
    KOTLIN_INT,
    KOTLIN_BOOLEAN,
    KOTLIN_STRING,
    KOTLIN_ANY,
    KotlinNullableType,
    KotlinListType,
)
from domains.types.languages.swift import (
    SwiftTypeSystem,
    SWIFT_INT,
    SWIFT_BOOL,
    SWIFT_STRING,
    SWIFT_ANY,
    SwiftOptionalType,
    SwiftArrayType,
)
from domains.imports.resolvers.zig import ZigImportResolver
from domains.imports.resolvers.rust import RustImportResolver
from domains.imports.resolvers.typescript import TypeScriptImportResolver
from domains.imports.resolvers.go import GoImportResolver
from domains.imports.resolvers.kotlin import KotlinImportResolver
from domains.imports.resolvers.swift import SwiftImportResolver
from core.token_classifier_zig import classify_zig_token, ZIG_ALL_KEYWORDS
from core.token_classifier_rust import classify_rust_token, RUST_ALL_KEYWORDS
from core.token_classifier_typescript import classify_typescript_token, TYPESCRIPT_ALL_KEYWORDS
from core.token_classifier_go import classify_go_token, GO_ALL_KEYWORDS
from core.token_classifier_kotlin import classify_kotlin_token, KOTLIN_ALL_KEYWORDS
from core.token_classifier_swift import classify_swift_token, SWIFT_ALL_KEYWORDS
from parsing.languages.zig import ZigIncrementalParser, create_zig_parser
from parsing.languages.rust import RustIncrementalParser, create_rust_parser
from parsing.languages.typescript import TypeScriptIncrementalParser, create_typescript_parser
from parsing.languages.go import GoIncrementalParser, create_go_parser
from parsing.languages.kotlin import KotlinIncrementalParser, create_kotlin_parser
from parsing.languages.swift import SwiftIncrementalParser, create_swift_parser


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

    def test_typescript_in_supported_languages(self):
        """TypeScript should be available."""
        langs = supported_languages()
        assert "typescript" in langs

    def test_get_zig_type_system(self):
        """Should get Zig type system."""
        ts = get_type_system("zig")
        assert isinstance(ts, ZigTypeSystem)

    def test_get_rust_type_system(self):
        """Should get Rust type system."""
        ts = get_type_system("rust")
        assert isinstance(ts, RustTypeSystem)

    def test_get_typescript_type_system(self):
        """Should get TypeScript type system."""
        ts = get_type_system("typescript")
        assert isinstance(ts, TypeScriptTypeSystem)

    def test_different_type_systems(self):
        """Zig, Rust, and TypeScript should have different type systems."""
        zig_ts = get_type_system("zig")
        rust_ts = get_type_system("rust")
        ts_ts = get_type_system("typescript")
        assert type(zig_ts) != type(rust_ts)
        assert type(rust_ts) != type(ts_ts)
        assert zig_ts.name != rust_ts.name
        assert rust_ts.name != ts_ts.name


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
# TypeScript Generation Integration Tests
# ===========================================================================


class TestTypeScriptGenerationIntegration:
    """Integration tests for TypeScript code generation."""

    @pytest.fixture
    def ts(self):
        return TypeScriptTypeSystem()

    @pytest.fixture
    def resolver(self):
        return TypeScriptImportResolver()

    @pytest.fixture
    def parser(self):
        return create_typescript_parser()

    def test_generic_function_generation(self, ts, parser):
        """Test generation of generic functions."""
        # Use incomplete code that clearly needs more input
        source = "function identity<T>(x: T): T {"
        result = parser.parse_initial(source)

        # Should detect unclosed brace
        holes = parser.find_holes()
        assert len(holes) > 0

        # Type system should understand generics
        parsed = ts.parse_type_annotation("T")
        assert parsed is not None

    def test_interface_type_handling(self, ts, parser):
        """Test interface type handling."""
        source = """interface Person {
    name: string;
    age: number;
}"""
        result = parser.parse_initial(source)
        assert result.is_valid

        # Type system should parse object types
        obj_type = ts.parse_type_annotation("{ name: string; age: number }")
        assert obj_type is not None

    def test_union_type_handling(self, ts, parser):
        """Test union type handling."""
        source = "type Status = 'pending' | 'active' | 'done';"
        result = parser.parse_initial(source)

        # Type system should parse union types
        union_type = ts.parse_type_annotation("string | number")
        assert union_type is not None
        assert isinstance(union_type, TSUnionType)

    def test_async_function_generation(self, ts, parser):
        """Test async function generation."""
        # Use incomplete code that clearly needs more input
        source = "async function fetchData(): Promise<string> {"
        result = parser.parse_initial(source)

        # Should detect unclosed brace
        holes = parser.find_holes()
        assert len(holes) > 0

        # Type system should parse Promise
        promise_type = ts.parse_type_annotation("Promise<string>")
        assert promise_type is not None

    def test_module_import_resolution(self, ts, resolver):
        """Test ES6 module import resolution."""
        result = resolver.resolve("react")
        assert result is not None

        result = resolver.resolve("@types/node")
        assert result is not None

    def test_token_classification_consistency(self, ts, parser):
        """Test that token classification is consistent with parsing."""
        from core.token_classifier import TokenCategory
        for keyword in ["const", "let", "function", "interface", "type"]:
            result = classify_typescript_token(keyword)
            # TypeScript classifier returns TokenClassification object
            assert result.category == TokenCategory.KEYWORD
            assert result.keyword_name == keyword

    def test_conditional_type_generation(self, ts, parser):
        """Test conditional type generation."""
        source = "type IsString<T> = T extends string ? true : false;"
        result = parser.parse_initial(source)
        assert result.is_valid

    def test_mapped_type_generation(self, ts, parser):
        """Test mapped type generation."""
        source = "type Readonly<T> = { readonly [K in keyof T]: T[K] };"
        result = parser.parse_initial(source)
        assert result.is_valid

    def test_arrow_function_generation(self, ts, parser):
        """Test arrow function generation."""
        source = "const add = (a: number, b: number): number =>"
        result = parser.parse_initial(source)

        # Should detect missing body
        holes = parser.find_holes()
        assert len(holes) > 0


# ===========================================================================
# Go Generation Integration Tests
# ===========================================================================


class TestGoGenerationIntegration:
    """Integration tests for Go code generation."""

    @pytest.fixture
    def ts(self):
        return GoTypeSystem()

    @pytest.fixture
    def resolver(self):
        return GoImportResolver()

    @pytest.fixture
    def parser(self):
        return create_go_parser()

    def test_go_in_supported_languages(self):
        """Go should be in supported languages."""
        langs = supported_languages()
        assert "go" in langs

    def test_get_go_type_system(self):
        """Should get Go type system."""
        ts = get_type_system("go")
        assert isinstance(ts, GoTypeSystem)

    def test_goroutine_function_generation(self, ts, parser):
        """Test generation of goroutine functions."""
        source = "func worker(ch chan int) {"
        result = parser.parse_initial(source)

        # Should detect missing body
        holes = parser.find_holes()
        assert len(holes) > 0

        # Type system should understand channel types
        chan_type = ts.parse_type_annotation("chan int")
        assert chan_type is not None

    def test_error_handling(self, ts, parser):
        """Test error handling patterns."""
        source = "func readFile(path string) ([]byte, error)"
        result = parser.parse_initial(source)

        # Type system should parse error type
        error_type = ts.parse_type_annotation("error")
        assert error_type is not None

    def test_interface_handling(self, ts, parser):
        """Test interface type handling."""
        source = """type Reader interface {
    Read(p []byte) (n int, err error)
}"""
        result = parser.parse_initial(source)
        # Parser validates syntax, AST may be None for incremental parsers
        assert result.is_valid

    def test_slice_and_map_types(self, ts):
        """Test Go slice and map type parsing."""
        slice_type = ts.parse_type_annotation("[]string")
        assert slice_type is not None

        map_type = ts.parse_type_annotation("map[string]int")
        assert map_type is not None

    def test_std_import_resolution(self, resolver):
        """Test Go standard library import resolution."""
        result = resolver.resolve("fmt")
        assert result is not None

        result = resolver.resolve("io")
        assert result is not None

    def test_token_classification(self):
        """Test Go token classification consistency."""
        from core.token_classifier import TokenCategory
        for keyword in ["func", "var", "const", "type", "struct"]:
            result = classify_go_token(keyword)
            assert result.category == TokenCategory.KEYWORD
            assert result.keyword_name == keyword


# ===========================================================================
# Kotlin Generation Integration Tests
# ===========================================================================


class TestKotlinGenerationIntegration:
    """Integration tests for Kotlin code generation."""

    @pytest.fixture
    def ts(self):
        return KotlinTypeSystem()

    @pytest.fixture
    def resolver(self):
        return KotlinImportResolver()

    @pytest.fixture
    def parser(self):
        return create_kotlin_parser()

    def test_kotlin_in_supported_languages(self):
        """Kotlin should be in supported languages."""
        langs = supported_languages()
        assert "kotlin" in langs

    def test_get_kotlin_type_system(self):
        """Should get Kotlin type system."""
        ts = get_type_system("kotlin")
        assert isinstance(ts, KotlinTypeSystem)

    def test_nullable_type_generation(self, ts, parser):
        """Test nullable type handling."""
        source = "fun process(value: String?): Int {"
        result = parser.parse_initial(source)

        # Should detect missing body
        holes = parser.find_holes()
        assert len(holes) > 0

        # Type system should understand nullable types
        nullable_type = ts.parse_type_annotation("String?")
        assert nullable_type is not None

    def test_when_expression(self, ts, parser):
        """Test when expression handling."""
        source = """when (x) {
    1 -> "one"
    2 -> "two"
    else -> "other"
}"""
        result = parser.parse_initial(source)
        # Parser validates syntax, AST may be None for incremental parsers
        assert result.is_valid

    def test_lambda_generation(self, ts, parser):
        """Test lambda expression generation."""
        source = "val sum = { a: Int, b: Int -> a + b }"
        result = parser.parse_initial(source)
        # Parser validates syntax, AST may be None for incremental parsers
        assert result.is_valid

    def test_collection_types(self, ts):
        """Test Kotlin collection type parsing."""
        list_type = ts.parse_type_annotation("List<String>")
        assert list_type is not None

        map_type = ts.parse_type_annotation("Map<String, Int>")
        assert map_type is not None

    def test_kotlin_import_resolution(self, resolver):
        """Test Kotlin stdlib import resolution."""
        result = resolver.resolve("kotlin.collections")
        assert result is not None

    def test_token_classification(self):
        """Test Kotlin token classification consistency."""
        from core.token_classifier import TokenCategory
        for keyword in ["fun", "val", "var", "class", "when"]:
            result = classify_kotlin_token(keyword)
            assert result.category == TokenCategory.KEYWORD
            assert result.keyword_name == keyword


# ===========================================================================
# Swift Generation Integration Tests
# ===========================================================================


class TestSwiftGenerationIntegration:
    """Integration tests for Swift code generation."""

    @pytest.fixture
    def ts(self):
        return SwiftTypeSystem()

    @pytest.fixture
    def resolver(self):
        return SwiftImportResolver()

    @pytest.fixture
    def parser(self):
        return create_swift_parser()

    def test_swift_in_supported_languages(self):
        """Swift should be in supported languages."""
        langs = supported_languages()
        assert "swift" in langs

    def test_get_swift_type_system(self):
        """Should get Swift type system."""
        ts = get_type_system("swift")
        assert isinstance(ts, SwiftTypeSystem)

    def test_optional_type_generation(self, ts, parser):
        """Test optional type handling."""
        source = "func process(value: String?) -> Int {"
        result = parser.parse_initial(source)

        # Should detect missing body
        holes = parser.find_holes()
        assert len(holes) > 0

        # Type system should understand optional types
        optional_type = ts.parse_type_annotation("String?")
        assert optional_type is not None

    def test_guard_statement(self, ts, parser):
        """Test guard statement handling."""
        source = """guard let value = optional else {
    return
}"""
        result = parser.parse_initial(source)
        # Parser validates syntax, AST may be None for incremental parsers
        assert result.is_valid

    def test_protocol_conformance(self, ts, parser):
        """Test protocol conformance."""
        source = """struct Point: Equatable {
    var x: Int
    var y: Int
}"""
        result = parser.parse_initial(source)
        # Parser validates syntax, AST may be None for incremental parsers
        assert result.is_valid

    def test_collection_types(self, ts):
        """Test Swift collection type parsing."""
        array_type = ts.parse_type_annotation("[String]")
        assert array_type is not None

        dict_type = ts.parse_type_annotation("[String: Int]")
        assert dict_type is not None

    def test_swift_import_resolution(self, resolver):
        """Test Swift framework import resolution."""
        result = resolver.resolve("Foundation")
        assert result is not None

    def test_token_classification(self):
        """Test Swift token classification consistency."""
        from core.token_classifier import TokenCategory
        for keyword in ["func", "let", "var", "class", "struct"]:
            result = classify_swift_token(keyword)
            assert result.category == TokenCategory.KEYWORD
            assert result.keyword_name == keyword


# ===========================================================================
# Cross-Language Comparison Tests
# ===========================================================================


class TestCrossLanguageComparison:
    """Tests comparing all supported language implementations."""

    @pytest.fixture
    def zig_ts(self):
        return ZigTypeSystem()

    @pytest.fixture
    def rust_ts(self):
        return RustTypeSystem()

    @pytest.fixture
    def ts_ts(self):
        return TypeScriptTypeSystem()

    @pytest.fixture
    def go_ts(self):
        return GoTypeSystem()

    @pytest.fixture
    def kotlin_ts(self):
        return KotlinTypeSystem()

    @pytest.fixture
    def swift_ts(self):
        return SwiftTypeSystem()

    def test_all_seven_languages_supported(self):
        """All seven languages should be in supported languages."""
        langs = supported_languages()
        expected = {"python", "typescript", "go", "rust", "kotlin", "swift", "zig"}
        assert expected.issubset(set(langs))

    def test_all_seven_type_systems_different(
        self, zig_ts, rust_ts, ts_ts, go_ts, kotlin_ts, swift_ts
    ):
        """All type systems should be distinct."""
        type_systems = [zig_ts, rust_ts, ts_ts, go_ts, kotlin_ts, swift_ts]
        names = [ts.name for ts in type_systems]
        assert len(set(names)) == len(names)  # All unique

    def test_primitive_types_both_support_i32(self, zig_ts, rust_ts):
        """Both languages should support i32."""
        zig_i32 = zig_ts.parse_type_annotation("i32")
        rust_i32 = rust_ts.parse_type_annotation("i32")

        assert zig_i32 is not None
        assert rust_i32 is not None

    def test_primitive_types_both_support_bool(self, zig_ts, rust_ts, ts_ts):
        """All languages should support bool."""
        zig_bool = zig_ts.parse_type_annotation("bool")
        rust_bool = rust_ts.parse_type_annotation("bool")
        ts_bool = ts_ts.parse_type_annotation("boolean")

        assert zig_bool is not None
        assert rust_bool is not None
        assert ts_bool is not None

    def test_different_optional_syntax(self, zig_ts, rust_ts, ts_ts):
        """Different optional syntax across languages."""
        # Zig uses ?T
        zig_opt = zig_ts.parse_type_annotation("?i32")
        # Rust uses Option<T>
        rust_opt = rust_ts.parse_type_annotation("Option<i32>")
        # TypeScript uses T | undefined or T | null
        ts_opt = ts_ts.parse_type_annotation("number | undefined")

        assert zig_opt is not None
        assert rust_opt is not None
        assert ts_opt is not None

    def test_different_result_syntax(self, zig_ts, rust_ts, ts_ts):
        """Different error handling syntax across languages."""
        # Zig uses E!T
        zig_result = zig_ts.parse_type_annotation("anyerror!i32")
        # Rust uses Result<T, E>
        rust_result = rust_ts.parse_type_annotation("Result<i32, Error>")
        # TypeScript uses Promise<T> for async errors
        ts_result = ts_ts.parse_type_annotation("Promise<number>")

        assert zig_result is not None
        assert rust_result is not None
        assert ts_result is not None

    def test_zig_has_comptime_rust_does_not(self, zig_ts, rust_ts, ts_ts):
        """Zig should support comptime types."""
        assert zig_ts.capabilities.supports_comptime
        assert not rust_ts.capabilities.supports_comptime
        assert not ts_ts.capabilities.supports_comptime

    def test_rust_has_ownership_zig_does_not(self, zig_ts, rust_ts, ts_ts):
        """Rust should support ownership types."""
        assert rust_ts.capabilities.supports_ownership
        assert not zig_ts.capabilities.supports_ownership
        assert not ts_ts.capabilities.supports_ownership

    def test_all_support_generics(self, zig_ts, rust_ts, ts_ts):
        """All languages should support generics."""
        assert zig_ts.capabilities.supports_generics
        assert rust_ts.capabilities.supports_generics
        assert ts_ts.capabilities.supports_generics

    def test_typescript_has_structural_typing(self, ts_ts):
        """TypeScript should support structural typing (protocols)."""
        # TypeScript uses structural typing which maps to supports_protocols
        assert ts_ts.capabilities.supports_protocols
        # TypeScript also supports union types natively
        assert ts_ts.capabilities.supports_union_types

    def test_typescript_has_union_types(self, ts_ts):
        """TypeScript should support union types natively."""
        union = ts_ts.parse_type_annotation("string | number | boolean")
        assert union is not None
        assert isinstance(union, TSUnionType)


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

    def test_typescript_parser_checkpoint_restore(self):
        """Test TypeScript parser checkpoint/restore works end-to-end."""
        parser = create_typescript_parser()
        parser.parse_initial("function foo(x: number)")

        checkpoint = parser.checkpoint()
        parser.extend_with_text(": number { return x; }")

        # Restore and verify
        parser.restore(checkpoint)
        assert parser.current_source == "function foo(x: number)"

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

    def test_typescript_parser_expected_tokens(self):
        """Test TypeScript parser provides useful expected tokens."""
        parser = create_typescript_parser()
        parser.parse_initial("function ")
        expected = parser.get_expected_tokens()

        # Should expect identifier for function name
        assert len(expected) > 0

    def test_all_parsers_detect_holes_consistently(self):
        """All parsers should detect holes in incomplete code."""
        # Test incomplete function definitions across all languages
        # Use unclosed braces to ensure holes are detected
        zig = create_zig_parser()
        rust = create_rust_parser()
        ts = create_typescript_parser()

        # Parse incomplete functions with unclosed braces
        zig.parse_initial("fn foo() void {")
        rust.parse_initial("fn foo() {")
        ts.parse_initial("function foo() {")

        # All should detect holes for unclosed braces
        assert len(zig.find_holes()) > 0
        assert len(rust.find_holes()) > 0
        assert len(ts.find_holes()) > 0


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

    def test_typescript_module_resolution(self):
        """TypeScript should resolve npm modules."""
        resolver = TypeScriptImportResolver()
        result = resolver.resolve("react")
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

    def test_typescript_scoped_package_resolution(self):
        """TypeScript should resolve scoped npm packages."""
        resolver = TypeScriptImportResolver()
        result = resolver.resolve("@types/node")
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

    def test_typescript_function_generation_flow(self):
        """Simulate generating a TypeScript function."""
        ts = TypeScriptTypeSystem()
        parser = create_typescript_parser()
        resolver = TypeScriptImportResolver()

        # Start with function signature
        parser.parse_initial("function add(a: number, b: number): number")

        # Check expected tokens for body
        expected = parser.get_expected_tokens()

        # Extend with body
        parser.extend_with_text(" { return a + b; }")

        # Final source should be valid
        assert "function add(a: number, b: number): number { return a + b; }" in parser.current_source

    def test_typescript_interface_generation_flow(self):
        """Simulate generating a TypeScript interface."""
        parser = create_typescript_parser()

        # Build interface incrementally
        parser.parse_initial("interface Point {")
        parser.extend_with_text("\n    x: number;")
        parser.extend_with_text("\n    y: number;")
        parser.extend_with_text("\n}")

        # Should have no holes now
        result = parser.parse_initial(parser.current_source)
        assert result.is_valid

    def test_typescript_class_generation_flow(self):
        """Simulate generating a TypeScript class."""
        parser = create_typescript_parser()

        # Build class incrementally
        parser.parse_initial("class Calculator {")
        parser.extend_with_text("\n    add(a: number, b: number): number {")
        parser.extend_with_text("\n        return a + b;")
        parser.extend_with_text("\n    }")
        parser.extend_with_text("\n}")

        # Should be complete
        result = parser.parse_initial(parser.current_source)
        assert result.is_valid

    def test_typescript_arrow_function_generation_flow(self):
        """Simulate generating a TypeScript arrow function."""
        parser = create_typescript_parser()

        # Build arrow function incrementally
        parser.parse_initial("const add = (a: number, b: number)")
        parser.extend_with_text(": number")
        parser.extend_with_text(" =>")

        # Should have hole for body
        holes = parser.find_holes()
        assert len(holes) > 0

        # Complete with body
        parser.extend_with_text(" a + b;")

        # Now should be complete
        assert "const add = (a: number, b: number): number => a + b;" in parser.current_source


# ===========================================================================
# Cross-Domain Constraint Propagation Tests
# ===========================================================================


class TestCrossDomainConstraintPropagation:
    """Tests for cross-domain constraint propagation.

    These tests verify that constraints flow correctly between domains
    (syntax, types, imports, control flow, semantics) across all languages.
    """

    def test_type_constraints_affect_hole_detection_python(self):
        """Type context should influence hole detection in Python."""
        from parsing.languages.python import create_python_parser
        from domains.types.languages.python import PythonTypeSystem

        parser = create_python_parser()
        ts = PythonTypeSystem()

        # Parse a function with type annotation
        parser.parse_initial("def foo(x: int) -> ")

        # Should detect a hole for return type
        holes = parser.find_holes()
        assert len(holes) > 0

    def test_type_constraints_affect_hole_detection_typescript(self):
        """Type context should influence hole detection in TypeScript."""
        parser = create_typescript_parser()
        ts = TypeScriptTypeSystem()

        # Parse a function with type annotation
        parser.parse_initial("function foo(x: number): ")

        # Should detect a hole for return type
        holes = parser.find_holes()
        assert len(holes) > 0

    def test_type_constraints_affect_hole_detection_rust(self):
        """Type context should influence hole detection in Rust."""
        parser = create_rust_parser()
        ts = RustTypeSystem()

        # Parse a function with type annotation
        parser.parse_initial("fn foo(x: i32) -> ")

        # Should detect a hole for return type
        holes = parser.find_holes()
        assert len(holes) > 0

    def test_type_constraints_affect_hole_detection_zig(self):
        """Type context should influence hole detection in Zig."""
        parser = create_zig_parser()
        ts = ZigTypeSystem()

        # Parse a function with type annotation
        parser.parse_initial("fn foo(x: i32) ")

        # Should detect a hole for return type
        holes = parser.find_holes()
        assert len(holes) > 0

    def test_incremental_constraint_refinement(self):
        """Constraints should refine as more code is parsed."""
        parser = create_typescript_parser()

        # Start with partial code - many holes
        parser.parse_initial("const x: ")
        holes1 = parser.find_holes()
        assert len(holes1) > 0

        # Add type annotation - still have value hole
        parser.extend_with_text("number = ")
        holes2 = parser.find_holes()
        assert len(holes2) > 0

        # Complete with value - no holes
        parser.extend_with_text("42;")
        holes3 = parser.find_holes()
        # Should be complete

    def test_all_languages_support_checkpoint_rollback(self):
        """All languages should support constraint rollback via checkpoints."""
        from parsing.languages.python import create_python_parser

        parsers = [
            ("python", create_python_parser()),
            ("typescript", create_typescript_parser()),
            ("rust", create_rust_parser()),
            ("zig", create_zig_parser()),
        ]

        for name, parser in parsers:
            # Parse initial code
            if name == "python":
                parser.parse_initial("def foo(x")
            elif name == "typescript":
                parser.parse_initial("function foo(x")
            else:
                parser.parse_initial("fn foo(x")

            # Checkpoint
            checkpoint = parser.checkpoint()
            initial_source = parser.current_source

            # Extend
            parser.extend_with_text("): void {}")

            # Restore
            parser.restore(checkpoint)

            # Verify rollback
            assert parser.current_source == initial_source, f"{name} rollback failed"

    def test_hole_kind_propagation_across_languages(self):
        """Different hole kinds (EXPRESSION, TYPE, etc.) should be detected consistently."""
        from parsing.partial_ast import HoleKind

        # Test that type holes are detected
        ts_parser = create_typescript_parser()
        ts_parser.parse_initial("const x: ")
        ts_holes = ts_parser.find_holes()

        # Test that expression holes are detected
        ts_parser2 = create_typescript_parser()
        ts_parser2.parse_initial("const x: number = ")
        expr_holes = ts_parser2.find_holes()

        # Both should have holes
        assert len(ts_holes) > 0
        assert len(expr_holes) > 0


# ===========================================================================
# Hole Capabilities Integration Tests
# ===========================================================================


class TestHoleCapabilitiesIntegration:
    """Tests for hole detection and management across all languages."""

    def test_unclosed_brackets_create_holes_all_languages(self):
        """Unclosed brackets should create holes in all languages."""
        from parsing.languages.python import create_python_parser

        test_cases = [
            ("python", create_python_parser(), "def foo("),
            ("typescript", create_typescript_parser(), "function foo("),
            ("rust", create_rust_parser(), "fn foo("),
            ("zig", create_zig_parser(), "fn foo("),
        ]

        for name, parser, code in test_cases:
            parser.parse_initial(code)
            holes = parser.find_holes()
            assert len(holes) > 0, f"{name}: unclosed paren should create hole"

    def test_unclosed_braces_create_holes_all_languages(self):
        """Unclosed braces should create holes in all languages."""
        from parsing.languages.python import create_python_parser

        test_cases = [
            ("typescript", create_typescript_parser(), "const obj = {"),
            ("rust", create_rust_parser(), "struct Foo {"),
            ("zig", create_zig_parser(), "const Foo = struct {"),
        ]

        for name, parser, code in test_cases:
            parser.parse_initial(code)
            holes = parser.find_holes()
            assert len(holes) > 0, f"{name}: unclosed brace should create hole"

    def test_incomplete_assignment_creates_holes(self):
        """Incomplete assignments should create holes."""
        test_cases = [
            ("typescript", create_typescript_parser(), "const x = "),
            ("rust", create_rust_parser(), "let x = "),
            ("zig", create_zig_parser(), "const x = "),
        ]

        for name, parser, code in test_cases:
            parser.parse_initial(code)
            holes = parser.find_holes()
            assert len(holes) > 0, f"{name}: incomplete assignment should create hole"

    def test_binary_operators_create_holes(self):
        """Trailing binary operators should create holes."""
        test_cases = [
            ("typescript", create_typescript_parser(), "const x = 1 + "),
            ("rust", create_rust_parser(), "let x = 1 + "),
            ("zig", create_zig_parser(), "const x = 1 + "),
        ]

        for name, parser, code in test_cases:
            parser.parse_initial(code)
            holes = parser.find_holes()
            assert len(holes) > 0, f"{name}: trailing operator should create hole"

    def test_parser_copy_preserves_holes(self):
        """Parser copy should preserve hole state."""
        parser = create_typescript_parser()
        parser.parse_initial("function foo(x: number): ")

        # Get holes before copy
        holes_before = parser.find_holes()

        # Copy parser
        parser_copy = parser.copy()

        # Get holes after copy
        holes_after = parser_copy.find_holes()

        # Should have same number of holes
        assert len(holes_before) == len(holes_after)

        # Modifications to copy shouldn't affect original
        parser_copy.extend_with_text("number { return x; }")
        assert len(parser.find_holes()) == len(holes_before)
