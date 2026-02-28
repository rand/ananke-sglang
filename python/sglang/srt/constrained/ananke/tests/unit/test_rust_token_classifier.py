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
"""Unit tests for Rust token classification.

Tests for the Rust token classifier including:
- Keyword classification
- Macro recognition
- Primitive type recognition
- Operator classification
- Literal parsing
"""

import pytest

from core.token_classifier import TokenCategory
from core.token_classifier_rust import (
    # Keyword sets
    RUST_ALL_KEYWORDS,
    RUST_CONTROL_KEYWORDS,
    RUST_DEFINITION_KEYWORDS,
    RUST_TYPE_KEYWORDS,
    RUST_MEMORY_KEYWORDS,
    RUST_SAFETY_KEYWORDS,
    RUST_ASYNC_KEYWORDS,
    RUST_OPERATOR_KEYWORDS,
    RUST_RESERVED_KEYWORDS,
    # Macros
    RUST_STD_MACROS,
    RUST_STD_MACRO_NAMES,
    # Types
    RUST_PRIMITIVE_TYPES,
    RUST_STD_TYPES,
    RUST_COMMON_TRAITS,
    # Other sets
    RUST_OPERATORS,
    RUST_DELIMITERS,
    RUST_BOOL_LITERALS,
    RUST_COMMON_ATTRIBUTES,
    RUST_COMMON_LIFETIMES,
    # Functions
    classify_rust_token,
    is_rust_string,
    is_rust_raw_string,
    is_rust_byte_string,
    is_rust_char,
    is_rust_byte,
    is_rust_identifier,
    classify_rust_numeric,
    get_rust_keyword_category,
    get_rust_keywords,
    get_rust_macros,
    get_rust_operators,
    get_rust_delimiters,
    get_rust_primitive_types,
    get_rust_std_types,
    get_rust_common_traits,
    get_rust_attributes,
)


# ===========================================================================
# Keyword Set Tests
# ===========================================================================


class TestRustKeywordSets:
    """Tests for Rust keyword sets."""

    def test_control_keywords_content(self):
        """Control keywords should contain expected values."""
        assert "if" in RUST_CONTROL_KEYWORDS
        assert "else" in RUST_CONTROL_KEYWORDS
        assert "match" in RUST_CONTROL_KEYWORDS
        assert "loop" in RUST_CONTROL_KEYWORDS
        assert "while" in RUST_CONTROL_KEYWORDS
        assert "for" in RUST_CONTROL_KEYWORDS
        assert "return" in RUST_CONTROL_KEYWORDS

    def test_definition_keywords_content(self):
        """Definition keywords should contain expected values."""
        assert "fn" in RUST_DEFINITION_KEYWORDS
        assert "let" in RUST_DEFINITION_KEYWORDS
        assert "const" in RUST_DEFINITION_KEYWORDS
        assert "static" in RUST_DEFINITION_KEYWORDS
        assert "struct" in RUST_DEFINITION_KEYWORDS
        assert "enum" in RUST_DEFINITION_KEYWORDS
        assert "trait" in RUST_DEFINITION_KEYWORDS
        assert "impl" in RUST_DEFINITION_KEYWORDS
        assert "mod" in RUST_DEFINITION_KEYWORDS
        assert "use" in RUST_DEFINITION_KEYWORDS

    def test_type_keywords_content(self):
        """Type keywords should contain expected values."""
        assert "where" in RUST_TYPE_KEYWORDS
        assert "dyn" in RUST_TYPE_KEYWORDS
        assert "Self" in RUST_TYPE_KEYWORDS
        assert "self" in RUST_TYPE_KEYWORDS

    def test_memory_keywords_content(self):
        """Memory keywords should contain expected values."""
        assert "mut" in RUST_MEMORY_KEYWORDS
        assert "ref" in RUST_MEMORY_KEYWORDS
        assert "move" in RUST_MEMORY_KEYWORDS

    def test_safety_keywords_content(self):
        """Safety keywords should contain expected values."""
        assert "unsafe" in RUST_SAFETY_KEYWORDS

    def test_async_keywords_content(self):
        """Async keywords should contain expected values."""
        assert "async" in RUST_ASYNC_KEYWORDS
        assert "await" in RUST_ASYNC_KEYWORDS

    def test_reserved_keywords_content(self):
        """Reserved keywords should contain expected values."""
        assert "abstract" in RUST_RESERVED_KEYWORDS
        assert "yield" in RUST_RESERVED_KEYWORDS
        assert "try" in RUST_RESERVED_KEYWORDS

    def test_all_keywords_union(self):
        """All keywords should be union of categories."""
        for kw in RUST_CONTROL_KEYWORDS:
            assert kw in RUST_ALL_KEYWORDS
        for kw in RUST_DEFINITION_KEYWORDS:
            assert kw in RUST_ALL_KEYWORDS
        for kw in RUST_TYPE_KEYWORDS:
            assert kw in RUST_ALL_KEYWORDS


# ===========================================================================
# Macro Tests
# ===========================================================================


class TestRustMacros:
    """Tests for Rust macro recognition."""

    def test_printing_macros(self):
        """Should have printing macros."""
        assert "println!" in RUST_STD_MACROS
        assert "print!" in RUST_STD_MACROS
        assert "eprintln!" in RUST_STD_MACROS
        assert "format!" in RUST_STD_MACROS

    def test_assertion_macros(self):
        """Should have assertion macros."""
        assert "assert!" in RUST_STD_MACROS
        assert "assert_eq!" in RUST_STD_MACROS
        assert "assert_ne!" in RUST_STD_MACROS
        assert "debug_assert!" in RUST_STD_MACROS

    def test_panic_macros(self):
        """Should have panic macros."""
        assert "panic!" in RUST_STD_MACROS
        assert "todo!" in RUST_STD_MACROS
        assert "unimplemented!" in RUST_STD_MACROS
        assert "unreachable!" in RUST_STD_MACROS

    def test_collection_macros(self):
        """Should have collection macros."""
        assert "vec!" in RUST_STD_MACROS

    def test_debug_macros(self):
        """Should have debug macros."""
        assert "dbg!" in RUST_STD_MACROS

    def test_macro_names_without_bang(self):
        """Should have macro names without !."""
        assert "println" in RUST_STD_MACRO_NAMES
        assert "vec" in RUST_STD_MACRO_NAMES


# ===========================================================================
# Primitive Type Tests
# ===========================================================================


class TestRustPrimitiveTypes:
    """Tests for Rust primitive type recognition."""

    def test_signed_integers(self):
        """Should have signed integer types."""
        assert "i8" in RUST_PRIMITIVE_TYPES
        assert "i16" in RUST_PRIMITIVE_TYPES
        assert "i32" in RUST_PRIMITIVE_TYPES
        assert "i64" in RUST_PRIMITIVE_TYPES
        assert "i128" in RUST_PRIMITIVE_TYPES
        assert "isize" in RUST_PRIMITIVE_TYPES

    def test_unsigned_integers(self):
        """Should have unsigned integer types."""
        assert "u8" in RUST_PRIMITIVE_TYPES
        assert "u16" in RUST_PRIMITIVE_TYPES
        assert "u32" in RUST_PRIMITIVE_TYPES
        assert "u64" in RUST_PRIMITIVE_TYPES
        assert "u128" in RUST_PRIMITIVE_TYPES
        assert "usize" in RUST_PRIMITIVE_TYPES

    def test_floats(self):
        """Should have float types."""
        assert "f32" in RUST_PRIMITIVE_TYPES
        assert "f64" in RUST_PRIMITIVE_TYPES

    def test_other_primitives(self):
        """Should have other primitive types."""
        assert "bool" in RUST_PRIMITIVE_TYPES
        assert "char" in RUST_PRIMITIVE_TYPES
        assert "str" in RUST_PRIMITIVE_TYPES

    def test_special_types(self):
        """Should have special types."""
        assert "()" in RUST_PRIMITIVE_TYPES
        assert "!" in RUST_PRIMITIVE_TYPES


# ===========================================================================
# Standard Library Types Tests
# ===========================================================================


class TestRustStdTypes:
    """Tests for Rust standard library types."""

    def test_smart_pointers(self):
        """Should have smart pointer types."""
        assert "Box" in RUST_STD_TYPES
        assert "Rc" in RUST_STD_TYPES
        assert "Arc" in RUST_STD_TYPES
        assert "Cell" in RUST_STD_TYPES
        assert "RefCell" in RUST_STD_TYPES

    def test_option_result(self):
        """Should have Option and Result."""
        assert "Option" in RUST_STD_TYPES
        assert "Some" in RUST_STD_TYPES
        assert "None" in RUST_STD_TYPES
        assert "Result" in RUST_STD_TYPES
        assert "Ok" in RUST_STD_TYPES
        assert "Err" in RUST_STD_TYPES

    def test_collections(self):
        """Should have collection types."""
        assert "Vec" in RUST_STD_TYPES
        assert "HashMap" in RUST_STD_TYPES
        assert "HashSet" in RUST_STD_TYPES
        assert "BTreeMap" in RUST_STD_TYPES

    def test_strings(self):
        """Should have string types."""
        assert "String" in RUST_STD_TYPES
        assert "PathBuf" in RUST_STD_TYPES


# ===========================================================================
# Common Traits Tests
# ===========================================================================


class TestRustCommonTraits:
    """Tests for Rust common traits."""

    def test_core_traits(self):
        """Should have core traits."""
        assert "Clone" in RUST_COMMON_TRAITS
        assert "Copy" in RUST_COMMON_TRAITS
        assert "Debug" in RUST_COMMON_TRAITS
        assert "Default" in RUST_COMMON_TRAITS
        assert "Display" in RUST_COMMON_TRAITS

    def test_comparison_traits(self):
        """Should have comparison traits."""
        assert "Eq" in RUST_COMMON_TRAITS
        assert "PartialEq" in RUST_COMMON_TRAITS
        assert "Ord" in RUST_COMMON_TRAITS
        assert "PartialOrd" in RUST_COMMON_TRAITS
        assert "Hash" in RUST_COMMON_TRAITS

    def test_conversion_traits(self):
        """Should have conversion traits."""
        assert "From" in RUST_COMMON_TRAITS
        assert "Into" in RUST_COMMON_TRAITS
        assert "TryFrom" in RUST_COMMON_TRAITS
        assert "TryInto" in RUST_COMMON_TRAITS

    def test_memory_traits(self):
        """Should have memory traits."""
        assert "Drop" in RUST_COMMON_TRAITS
        assert "Sized" in RUST_COMMON_TRAITS
        assert "Send" in RUST_COMMON_TRAITS
        assert "Sync" in RUST_COMMON_TRAITS

    def test_iterator_traits(self):
        """Should have iterator traits."""
        assert "Iterator" in RUST_COMMON_TRAITS
        assert "IntoIterator" in RUST_COMMON_TRAITS

    def test_function_traits(self):
        """Should have function traits."""
        assert "Fn" in RUST_COMMON_TRAITS
        assert "FnMut" in RUST_COMMON_TRAITS
        assert "FnOnce" in RUST_COMMON_TRAITS


# ===========================================================================
# Operator Tests
# ===========================================================================


class TestRustOperators:
    """Tests for Rust operator recognition."""

    def test_arithmetic_operators(self):
        """Should have arithmetic operators."""
        assert "+" in RUST_OPERATORS
        assert "-" in RUST_OPERATORS
        assert "*" in RUST_OPERATORS
        assert "/" in RUST_OPERATORS
        assert "%" in RUST_OPERATORS

    def test_comparison_operators(self):
        """Should have comparison operators."""
        assert "==" in RUST_OPERATORS
        assert "!=" in RUST_OPERATORS
        assert "<" in RUST_OPERATORS
        assert ">" in RUST_OPERATORS
        assert "<=" in RUST_OPERATORS
        assert ">=" in RUST_OPERATORS

    def test_logical_operators(self):
        """Should have logical operators."""
        assert "&&" in RUST_OPERATORS
        assert "||" in RUST_OPERATORS
        assert "!" in RUST_OPERATORS

    def test_bitwise_operators(self):
        """Should have bitwise operators."""
        assert "&" in RUST_OPERATORS
        assert "|" in RUST_OPERATORS
        assert "^" in RUST_OPERATORS
        assert "<<" in RUST_OPERATORS
        assert ">>" in RUST_OPERATORS

    def test_assignment_operators(self):
        """Should have assignment operators."""
        assert "=" in RUST_OPERATORS
        assert "+=" in RUST_OPERATORS
        assert "-=" in RUST_OPERATORS

    def test_special_operators(self):
        """Should have special operators."""
        assert ".." in RUST_OPERATORS  # Range
        assert "..=" in RUST_OPERATORS  # Inclusive range
        assert "::" in RUST_OPERATORS  # Path
        assert "?" in RUST_OPERATORS  # Error propagation
        assert "=>" in RUST_OPERATORS  # Match arm
        assert "->" in RUST_OPERATORS  # Return type


# ===========================================================================
# Token Classification Tests
# ===========================================================================


class TestClassifyRustToken:
    """Tests for classify_rust_token function."""

    def test_classify_keyword(self):
        """Should classify keywords correctly."""
        category, keyword, value = classify_rust_token("fn")
        assert category == TokenCategory.KEYWORD
        assert keyword == "fn"

    def test_classify_macro(self):
        """Should classify macros correctly."""
        category, keyword, value = classify_rust_token("println!")
        assert category == TokenCategory.BUILTIN
        assert keyword == "println!"

    def test_classify_primitive_type(self):
        """Should classify primitive types as builtins."""
        category, keyword, value = classify_rust_token("i32")
        assert category == TokenCategory.BUILTIN

    def test_classify_std_type(self):
        """Should classify std types as builtins."""
        category, keyword, value = classify_rust_token("Vec")
        assert category == TokenCategory.BUILTIN

    def test_classify_trait(self):
        """Should classify traits as builtins."""
        category, keyword, value = classify_rust_token("Clone")
        assert category == TokenCategory.BUILTIN

    def test_classify_operator(self):
        """Should classify operators correctly."""
        category, keyword, value = classify_rust_token("+")
        assert category == TokenCategory.OPERATOR

    def test_classify_delimiter(self):
        """Should classify delimiters correctly."""
        category, keyword, value = classify_rust_token("(")
        assert category == TokenCategory.DELIMITER

    def test_classify_bool_literal_true(self):
        """Should classify 'true' as bool literal."""
        category, keyword, value = classify_rust_token("true")
        assert category == TokenCategory.BOOL_LITERAL
        assert value is True

    def test_classify_bool_literal_false(self):
        """Should classify 'false' as bool literal."""
        category, keyword, value = classify_rust_token("false")
        assert category == TokenCategory.BOOL_LITERAL
        assert value is False

    def test_classify_identifier(self):
        """Should classify identifiers correctly."""
        category, keyword, value = classify_rust_token("my_variable")
        assert category == TokenCategory.IDENTIFIER

    def test_classify_lifetime(self):
        """Should classify lifetimes as builtins."""
        category, keyword, value = classify_rust_token("'a")
        assert category == TokenCategory.BUILTIN

    def test_classify_whitespace(self):
        """Should classify whitespace correctly."""
        category, keyword, value = classify_rust_token("   ")
        assert category == TokenCategory.WHITESPACE

    def test_classify_comment(self):
        """Should classify comments correctly."""
        category, keyword, value = classify_rust_token("// comment")
        assert category == TokenCategory.COMMENT


# ===========================================================================
# Numeric Classification Tests
# ===========================================================================


class TestClassifyRustNumeric:
    """Tests for numeric literal classification."""

    def test_decimal_integer(self):
        """Should classify decimal integers."""
        result = classify_rust_numeric("123")
        assert result is not None
        category, keyword, value = result
        assert category == TokenCategory.INT_LITERAL
        assert value == 123

    def test_integer_with_underscore(self):
        """Should handle underscores in integers."""
        result = classify_rust_numeric("1_000_000")
        assert result is not None
        category, keyword, value = result
        assert category == TokenCategory.INT_LITERAL
        assert value == 1000000

    def test_hex_integer(self):
        """Should classify hex integers."""
        result = classify_rust_numeric("0x1A")
        assert result is not None
        category, keyword, value = result
        assert category == TokenCategory.INT_LITERAL
        assert value == 26

    def test_octal_integer(self):
        """Should classify octal integers."""
        result = classify_rust_numeric("0o17")
        assert result is not None
        category, keyword, value = result
        assert category == TokenCategory.INT_LITERAL
        assert value == 15

    def test_binary_integer(self):
        """Should classify binary integers."""
        result = classify_rust_numeric("0b1010")
        assert result is not None
        category, keyword, value = result
        assert category == TokenCategory.INT_LITERAL
        assert value == 10

    def test_decimal_float(self):
        """Should classify decimal floats."""
        result = classify_rust_numeric("3.14")
        assert result is not None
        category, keyword, value = result
        assert category == TokenCategory.FLOAT_LITERAL
        assert abs(value - 3.14) < 0.001

    def test_float_with_exponent(self):
        """Should classify floats with exponent."""
        result = classify_rust_numeric("1.0e10")
        assert result is not None
        category, keyword, value = result
        assert category == TokenCategory.FLOAT_LITERAL

    def test_integer_with_type_suffix(self):
        """Should handle type suffix on integers."""
        result = classify_rust_numeric("42i32")
        assert result is not None
        category, keyword, value = result
        assert category == TokenCategory.INT_LITERAL


# ===========================================================================
# String/Char Tests
# ===========================================================================


class TestRustStringChar:
    """Tests for string and character detection."""

    def test_is_string(self):
        """Should detect string literals."""
        assert is_rust_string('"hello"')
        assert is_rust_string('"')

    def test_is_not_string(self):
        """Should not detect non-strings."""
        assert not is_rust_string("hello")
        assert not is_rust_string('b"bytes"')

    def test_is_raw_string(self):
        """Should detect raw string literals."""
        assert is_rust_raw_string('r"raw"')
        assert is_rust_raw_string('r#"raw"#')
        assert is_rust_raw_string('br"raw bytes"')

    def test_is_byte_string(self):
        """Should detect byte string literals."""
        assert is_rust_byte_string('b"bytes"')
        assert not is_rust_byte_string('"not bytes"')

    def test_is_char(self):
        """Should detect character literals."""
        assert is_rust_char("'a'")
        assert is_rust_char("'\\n'")
        # But not lifetimes
        assert not is_rust_char("'static")

    def test_is_byte(self):
        """Should detect byte literals."""
        assert is_rust_byte("b'a'")


# ===========================================================================
# Identifier Tests
# ===========================================================================


class TestRustIdentifier:
    """Tests for identifier validation."""

    def test_simple_identifier(self):
        """Should accept simple identifiers."""
        assert is_rust_identifier("foo")
        assert is_rust_identifier("bar123")
        assert is_rust_identifier("_private")

    def test_identifier_with_underscore(self):
        """Should accept identifiers with underscores."""
        assert is_rust_identifier("foo_bar")
        assert is_rust_identifier("_foo_bar_")

    def test_starts_with_digit_invalid(self):
        """Should reject identifiers starting with digit."""
        assert not is_rust_identifier("123abc")
        assert not is_rust_identifier("1foo")

    def test_raw_identifier(self):
        """Should accept raw identifiers."""
        assert is_rust_identifier("r#fn")
        assert is_rust_identifier("r#type")

    def test_empty_invalid(self):
        """Should reject empty identifiers."""
        assert not is_rust_identifier("")


# ===========================================================================
# Keyword Category Tests
# ===========================================================================


class TestRustKeywordCategory:
    """Tests for keyword category retrieval."""

    def test_control_category(self):
        """Should identify control keywords."""
        assert get_rust_keyword_category("if") == "control"
        assert get_rust_keyword_category("match") == "control"
        assert get_rust_keyword_category("return") == "control"

    def test_definition_category(self):
        """Should identify definition keywords."""
        assert get_rust_keyword_category("fn") == "definition"
        assert get_rust_keyword_category("struct") == "definition"
        assert get_rust_keyword_category("let") == "definition"

    def test_type_category(self):
        """Should identify type keywords."""
        assert get_rust_keyword_category("where") == "type"
        assert get_rust_keyword_category("dyn") == "type"

    def test_memory_category(self):
        """Should identify memory keywords."""
        assert get_rust_keyword_category("mut") == "memory"
        assert get_rust_keyword_category("ref") == "memory"

    def test_safety_category(self):
        """Should identify safety keywords."""
        assert get_rust_keyword_category("unsafe") == "safety"

    def test_async_category(self):
        """Should identify async keywords."""
        assert get_rust_keyword_category("async") == "async"
        assert get_rust_keyword_category("await") == "async"

    def test_reserved_category(self):
        """Should identify reserved keywords."""
        assert get_rust_keyword_category("yield") == "reserved"

    def test_unknown_category(self):
        """Should return unknown for non-keywords."""
        assert get_rust_keyword_category("notakeyword") == "unknown"


# ===========================================================================
# Attribute Tests
# ===========================================================================


class TestRustAttributes:
    """Tests for Rust attribute recognition."""

    def test_derive_attribute(self):
        """Should have derive attribute."""
        assert "derive" in RUST_COMMON_ATTRIBUTES

    def test_testing_attributes(self):
        """Should have testing attributes."""
        assert "test" in RUST_COMMON_ATTRIBUTES
        assert "ignore" in RUST_COMMON_ATTRIBUTES
        assert "should_panic" in RUST_COMMON_ATTRIBUTES

    def test_configuration_attributes(self):
        """Should have configuration attributes."""
        assert "cfg" in RUST_COMMON_ATTRIBUTES
        assert "cfg_attr" in RUST_COMMON_ATTRIBUTES

    def test_lint_attributes(self):
        """Should have lint attributes."""
        assert "allow" in RUST_COMMON_ATTRIBUTES
        assert "warn" in RUST_COMMON_ATTRIBUTES
        assert "deny" in RUST_COMMON_ATTRIBUTES

    def test_code_gen_attributes(self):
        """Should have code generation attributes."""
        assert "inline" in RUST_COMMON_ATTRIBUTES


# ===========================================================================
# Getter Function Tests
# ===========================================================================


class TestRustGetterFunctions:
    """Tests for getter functions."""

    def test_get_keywords(self):
        """get_rust_keywords should return keywords."""
        keywords = get_rust_keywords()
        assert "fn" in keywords
        assert "let" in keywords
        assert "if" in keywords

    def test_get_macros(self):
        """get_rust_macros should return macros."""
        macros = get_rust_macros()
        assert "println!" in macros
        assert "vec!" in macros

    def test_get_operators(self):
        """get_rust_operators should return operators."""
        operators = get_rust_operators()
        assert "+" in operators
        assert "==" in operators

    def test_get_delimiters(self):
        """get_rust_delimiters should return delimiters."""
        delimiters = get_rust_delimiters()
        assert "(" in delimiters
        assert "{" in delimiters

    def test_get_primitive_types(self):
        """get_rust_primitive_types should return types."""
        types = get_rust_primitive_types()
        assert "i32" in types
        assert "bool" in types

    def test_get_std_types(self):
        """get_rust_std_types should return std types."""
        types = get_rust_std_types()
        assert "Vec" in types
        assert "String" in types

    def test_get_common_traits(self):
        """get_rust_common_traits should return traits."""
        traits = get_rust_common_traits()
        assert "Clone" in traits
        assert "Debug" in traits

    def test_get_attributes(self):
        """get_rust_attributes should return attributes."""
        attrs = get_rust_attributes()
        assert "derive" in attrs
        assert "test" in attrs
