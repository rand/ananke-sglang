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
"""Unit tests for Zig token classification.

Tests for the Zig token classifier including:
- Keyword classification
- Builtin function recognition
- Primitive type recognition
- Operator classification
- Literal parsing
"""

import pytest

from core.token_classifier import TokenCategory
from core.token_classifier_zig import (
    # Keyword sets
    ZIG_ALL_KEYWORDS,
    ZIG_CONTROL_KEYWORDS,
    ZIG_ERROR_KEYWORDS,
    ZIG_DEFINITION_KEYWORDS,
    ZIG_MEMORY_KEYWORDS,
    ZIG_ASYNC_KEYWORDS,
    ZIG_TYPE_KEYWORDS,
    ZIG_OPERATOR_KEYWORDS,
    ZIG_ASM_KEYWORDS,
    # Builtins
    ZIG_BUILTINS,
    # Types
    ZIG_PRIMITIVE_TYPES,
    # Other sets
    ZIG_OPERATORS,
    ZIG_DELIMITERS,
    ZIG_BOOL_LITERALS,
    ZIG_NULL_LITERALS,
    ZIG_STD_COMMON,
    # Functions
    classify_zig_token,
    is_zig_string,
    is_zig_char,
    is_zig_identifier,
    classify_zig_numeric,
    get_zig_keyword_category,
    get_zig_keywords,
    get_zig_builtins,
    get_zig_operators,
    get_zig_delimiters,
    get_zig_primitive_types,
)


# ===========================================================================
# Keyword Set Tests
# ===========================================================================


class TestZigKeywordSets:
    """Tests for Zig keyword sets."""

    def test_control_keywords_content(self):
        """Control keywords should contain expected values."""
        assert "if" in ZIG_CONTROL_KEYWORDS
        assert "else" in ZIG_CONTROL_KEYWORDS
        assert "for" in ZIG_CONTROL_KEYWORDS
        assert "while" in ZIG_CONTROL_KEYWORDS
        assert "switch" in ZIG_CONTROL_KEYWORDS
        assert "return" in ZIG_CONTROL_KEYWORDS

    def test_error_keywords_content(self):
        """Error keywords should contain expected values."""
        assert "try" in ZIG_ERROR_KEYWORDS
        assert "catch" in ZIG_ERROR_KEYWORDS
        assert "orelse" in ZIG_ERROR_KEYWORDS
        assert "error" in ZIG_ERROR_KEYWORDS

    def test_definition_keywords_content(self):
        """Definition keywords should contain expected values."""
        assert "fn" in ZIG_DEFINITION_KEYWORDS
        assert "const" in ZIG_DEFINITION_KEYWORDS
        assert "var" in ZIG_DEFINITION_KEYWORDS
        assert "struct" in ZIG_DEFINITION_KEYWORDS
        assert "enum" in ZIG_DEFINITION_KEYWORDS
        assert "union" in ZIG_DEFINITION_KEYWORDS
        assert "comptime" in ZIG_DEFINITION_KEYWORDS

    def test_memory_keywords_content(self):
        """Memory keywords should contain expected values."""
        assert "defer" in ZIG_MEMORY_KEYWORDS
        assert "errdefer" in ZIG_MEMORY_KEYWORDS
        assert "noalias" in ZIG_MEMORY_KEYWORDS
        assert "volatile" in ZIG_MEMORY_KEYWORDS

    def test_async_keywords_content(self):
        """Async keywords should contain expected values."""
        assert "async" in ZIG_ASYNC_KEYWORDS
        assert "await" in ZIG_ASYNC_KEYWORDS
        assert "suspend" in ZIG_ASYNC_KEYWORDS
        assert "resume" in ZIG_ASYNC_KEYWORDS

    def test_all_keywords_union(self):
        """All keywords should be union of categories."""
        for kw in ZIG_CONTROL_KEYWORDS:
            assert kw in ZIG_ALL_KEYWORDS
        for kw in ZIG_ERROR_KEYWORDS:
            assert kw in ZIG_ALL_KEYWORDS
        for kw in ZIG_DEFINITION_KEYWORDS:
            assert kw in ZIG_ALL_KEYWORDS


# ===========================================================================
# Builtin Tests
# ===========================================================================


class TestZigBuiltins:
    """Tests for Zig builtin functions."""

    def test_type_introspection_builtins(self):
        """Should have type introspection builtins."""
        assert "@TypeOf" in ZIG_BUILTINS
        assert "@typeInfo" in ZIG_BUILTINS
        assert "@typeName" in ZIG_BUILTINS
        assert "@Type" in ZIG_BUILTINS

    def test_size_builtins(self):
        """Should have size/alignment builtins."""
        assert "@sizeOf" in ZIG_BUILTINS
        assert "@alignOf" in ZIG_BUILTINS
        assert "@bitSizeOf" in ZIG_BUILTINS

    def test_cast_builtins(self):
        """Should have casting builtins."""
        assert "@as" in ZIG_BUILTINS
        assert "@intCast" in ZIG_BUILTINS
        assert "@floatCast" in ZIG_BUILTINS
        assert "@ptrCast" in ZIG_BUILTINS
        assert "@bitCast" in ZIG_BUILTINS

    def test_math_builtins(self):
        """Should have math builtins."""
        assert "@min" in ZIG_BUILTINS
        assert "@max" in ZIG_BUILTINS
        assert "@sqrt" in ZIG_BUILTINS
        assert "@abs" in ZIG_BUILTINS

    def test_memory_builtins(self):
        """Should have memory builtins."""
        assert "@memcpy" in ZIG_BUILTINS
        assert "@memset" in ZIG_BUILTINS

    def test_import_builtins(self):
        """Should have import builtins."""
        assert "@import" in ZIG_BUILTINS
        assert "@embedFile" in ZIG_BUILTINS
        assert "@cImport" in ZIG_BUILTINS

    def test_compile_builtins(self):
        """Should have compilation builtins."""
        assert "@compileError" in ZIG_BUILTINS
        assert "@compileLog" in ZIG_BUILTINS


# ===========================================================================
# Primitive Type Tests
# ===========================================================================


class TestZigPrimitiveTypes:
    """Tests for Zig primitive type recognition."""

    def test_signed_integers(self):
        """Should have signed integer types."""
        assert "i8" in ZIG_PRIMITIVE_TYPES
        assert "i16" in ZIG_PRIMITIVE_TYPES
        assert "i32" in ZIG_PRIMITIVE_TYPES
        assert "i64" in ZIG_PRIMITIVE_TYPES
        assert "i128" in ZIG_PRIMITIVE_TYPES
        assert "isize" in ZIG_PRIMITIVE_TYPES

    def test_unsigned_integers(self):
        """Should have unsigned integer types."""
        assert "u8" in ZIG_PRIMITIVE_TYPES
        assert "u16" in ZIG_PRIMITIVE_TYPES
        assert "u32" in ZIG_PRIMITIVE_TYPES
        assert "u64" in ZIG_PRIMITIVE_TYPES
        assert "u128" in ZIG_PRIMITIVE_TYPES
        assert "usize" in ZIG_PRIMITIVE_TYPES

    def test_floats(self):
        """Should have float types."""
        assert "f16" in ZIG_PRIMITIVE_TYPES
        assert "f32" in ZIG_PRIMITIVE_TYPES
        assert "f64" in ZIG_PRIMITIVE_TYPES
        assert "f80" in ZIG_PRIMITIVE_TYPES
        assert "f128" in ZIG_PRIMITIVE_TYPES

    def test_special_types(self):
        """Should have special types."""
        assert "bool" in ZIG_PRIMITIVE_TYPES
        assert "void" in ZIG_PRIMITIVE_TYPES
        assert "noreturn" in ZIG_PRIMITIVE_TYPES
        assert "type" in ZIG_PRIMITIVE_TYPES
        assert "anytype" in ZIG_PRIMITIVE_TYPES

    def test_comptime_types(self):
        """Should have comptime types."""
        assert "comptime_int" in ZIG_PRIMITIVE_TYPES
        assert "comptime_float" in ZIG_PRIMITIVE_TYPES

    def test_c_interop_types(self):
        """Should have C interop types."""
        assert "c_int" in ZIG_PRIMITIVE_TYPES
        assert "c_char" in ZIG_PRIMITIVE_TYPES
        assert "c_long" in ZIG_PRIMITIVE_TYPES


# ===========================================================================
# Operator Tests
# ===========================================================================


class TestZigOperators:
    """Tests for Zig operator recognition."""

    def test_arithmetic_operators(self):
        """Should have arithmetic operators."""
        assert "+" in ZIG_OPERATORS
        assert "-" in ZIG_OPERATORS
        assert "*" in ZIG_OPERATORS
        assert "/" in ZIG_OPERATORS
        assert "%" in ZIG_OPERATORS

    def test_wrapping_operators(self):
        """Should have wrapping arithmetic operators."""
        assert "+%" in ZIG_OPERATORS
        assert "-%" in ZIG_OPERATORS
        assert "*%" in ZIG_OPERATORS

    def test_saturating_operators(self):
        """Should have saturating arithmetic operators."""
        assert "+|" in ZIG_OPERATORS
        assert "-|" in ZIG_OPERATORS
        assert "*|" in ZIG_OPERATORS

    def test_comparison_operators(self):
        """Should have comparison operators."""
        assert "==" in ZIG_OPERATORS
        assert "!=" in ZIG_OPERATORS
        assert "<" in ZIG_OPERATORS
        assert ">" in ZIG_OPERATORS
        assert "<=" in ZIG_OPERATORS
        assert ">=" in ZIG_OPERATORS

    def test_bitwise_operators(self):
        """Should have bitwise operators."""
        assert "&" in ZIG_OPERATORS
        assert "|" in ZIG_OPERATORS
        assert "^" in ZIG_OPERATORS
        assert "~" in ZIG_OPERATORS
        assert "<<" in ZIG_OPERATORS
        assert ">>" in ZIG_OPERATORS

    def test_assignment_operators(self):
        """Should have assignment operators."""
        assert "=" in ZIG_OPERATORS
        assert "+=" in ZIG_OPERATORS
        assert "-=" in ZIG_OPERATORS
        assert "*=" in ZIG_OPERATORS

    def test_pointer_operators(self):
        """Should have pointer/optional operators."""
        assert ".*" in ZIG_OPERATORS
        assert ".?" in ZIG_OPERATORS

    def test_special_operators(self):
        """Should have special operators."""
        assert "?" in ZIG_OPERATORS  # Optional type
        assert "!" in ZIG_OPERATORS  # Error union


# ===========================================================================
# Token Classification Tests
# ===========================================================================


class TestClassifyZigToken:
    """Tests for classify_zig_token function."""

    def test_classify_keyword(self):
        """Should classify keywords correctly."""
        category, keyword, value = classify_zig_token("fn")
        assert category == TokenCategory.KEYWORD
        assert keyword == "fn"

    def test_classify_builtin(self):
        """Should classify builtins correctly."""
        category, keyword, value = classify_zig_token("@import")
        assert category == TokenCategory.BUILTIN
        assert keyword == "@import"

    def test_classify_primitive_type(self):
        """Should classify primitive types as builtins."""
        category, keyword, value = classify_zig_token("i32")
        assert category == TokenCategory.BUILTIN

    def test_classify_operator(self):
        """Should classify operators correctly."""
        category, keyword, value = classify_zig_token("+")
        assert category == TokenCategory.OPERATOR

    def test_classify_delimiter(self):
        """Should classify delimiters correctly."""
        category, keyword, value = classify_zig_token("(")
        assert category == TokenCategory.DELIMITER

    def test_classify_bool_literal_true(self):
        """Should classify 'true' as bool literal."""
        category, keyword, value = classify_zig_token("true")
        assert category == TokenCategory.BOOL_LITERAL
        assert value is True

    def test_classify_bool_literal_false(self):
        """Should classify 'false' as bool literal."""
        category, keyword, value = classify_zig_token("false")
        assert category == TokenCategory.BOOL_LITERAL
        assert value is False

    def test_classify_null_literal(self):
        """Should classify 'null' as none literal."""
        category, keyword, value = classify_zig_token("null")
        assert category == TokenCategory.NONE_LITERAL

    def test_classify_undefined_literal(self):
        """Should classify 'undefined' as none literal."""
        category, keyword, value = classify_zig_token("undefined")
        assert category == TokenCategory.NONE_LITERAL

    def test_classify_identifier(self):
        """Should classify identifiers correctly."""
        category, keyword, value = classify_zig_token("myVariable")
        assert category == TokenCategory.IDENTIFIER

    def test_classify_whitespace(self):
        """Should classify whitespace correctly."""
        category, keyword, value = classify_zig_token("   ")
        assert category == TokenCategory.WHITESPACE

    def test_classify_comment(self):
        """Should classify comments correctly."""
        category, keyword, value = classify_zig_token("// comment")
        assert category == TokenCategory.COMMENT

    def test_classify_arbitrary_int_type(self):
        """Should classify arbitrary-width integer types."""
        category, keyword, value = classify_zig_token("u3")
        assert category == TokenCategory.BUILTIN

        category, keyword, value = classify_zig_token("i17")
        assert category == TokenCategory.BUILTIN


# ===========================================================================
# Numeric Classification Tests
# ===========================================================================


class TestClassifyZigNumeric:
    """Tests for numeric literal classification."""

    def test_decimal_integer(self):
        """Should classify decimal integers."""
        result = classify_zig_numeric("123")
        assert result is not None
        category, keyword, value = result
        assert category == TokenCategory.INT_LITERAL
        assert value == 123

    def test_decimal_integer_with_underscore(self):
        """Should handle underscores in integers."""
        result = classify_zig_numeric("1_000_000")
        assert result is not None
        category, keyword, value = result
        assert category == TokenCategory.INT_LITERAL
        assert value == 1000000

    def test_hex_integer(self):
        """Should classify hex integers."""
        result = classify_zig_numeric("0x1A")
        assert result is not None
        category, keyword, value = result
        assert category == TokenCategory.INT_LITERAL
        assert value == 26

    def test_octal_integer(self):
        """Should classify octal integers."""
        result = classify_zig_numeric("0o17")
        assert result is not None
        category, keyword, value = result
        assert category == TokenCategory.INT_LITERAL
        assert value == 15

    def test_binary_integer(self):
        """Should classify binary integers."""
        result = classify_zig_numeric("0b1010")
        assert result is not None
        category, keyword, value = result
        assert category == TokenCategory.INT_LITERAL
        assert value == 10

    def test_decimal_float(self):
        """Should classify decimal floats."""
        result = classify_zig_numeric("3.14")
        assert result is not None
        category, keyword, value = result
        assert category == TokenCategory.FLOAT_LITERAL
        assert abs(value - 3.14) < 0.001

    def test_float_with_exponent(self):
        """Should classify floats with exponent."""
        result = classify_zig_numeric("1.0e10")
        assert result is not None
        category, keyword, value = result
        assert category == TokenCategory.FLOAT_LITERAL
        assert value == 1.0e10

    def test_hex_float(self):
        """Should classify hex floats."""
        result = classify_zig_numeric("0x1.0p0")
        assert result is not None
        category, keyword, value = result
        assert category == TokenCategory.FLOAT_LITERAL
        assert value == 1.0


# ===========================================================================
# String/Char Tests
# ===========================================================================


class TestZigStringChar:
    """Tests for string and character detection."""

    def test_is_string_double_quote(self):
        """Should detect double-quoted strings."""
        assert is_zig_string('"hello"')
        assert is_zig_string('"')

    def test_is_string_multiline(self):
        """Should detect multiline strings."""
        assert is_zig_string("\\\\multiline")

    def test_is_not_string(self):
        """Should not detect non-strings."""
        assert not is_zig_string("hello")
        assert not is_zig_string("123")

    def test_is_char(self):
        """Should detect character literals."""
        assert is_zig_char("'a'")
        assert is_zig_char("'\\n'")

    def test_is_not_char(self):
        """Should not detect non-chars."""
        assert not is_zig_char("a")
        assert not is_zig_char("'")  # Too short


# ===========================================================================
# Identifier Tests
# ===========================================================================


class TestZigIdentifier:
    """Tests for identifier validation."""

    def test_simple_identifier(self):
        """Should accept simple identifiers."""
        assert is_zig_identifier("foo")
        assert is_zig_identifier("bar123")
        assert is_zig_identifier("_private")

    def test_identifier_with_underscore(self):
        """Should accept identifiers with underscores."""
        assert is_zig_identifier("foo_bar")
        assert is_zig_identifier("_foo_bar_")

    def test_starts_with_digit_invalid(self):
        """Should reject identifiers starting with digit."""
        assert not is_zig_identifier("123abc")
        assert not is_zig_identifier("1foo")

    def test_raw_identifier(self):
        """Should accept raw identifiers."""
        assert is_zig_identifier('@"while"')
        assert is_zig_identifier('@"fn"')

    def test_empty_invalid(self):
        """Should reject empty identifiers."""
        assert not is_zig_identifier("")


# ===========================================================================
# Keyword Category Tests
# ===========================================================================


class TestZigKeywordCategory:
    """Tests for keyword category retrieval."""

    def test_control_category(self):
        """Should identify control keywords."""
        assert get_zig_keyword_category("if") == "control"
        assert get_zig_keyword_category("for") == "control"
        assert get_zig_keyword_category("return") == "control"

    def test_error_category(self):
        """Should identify error keywords."""
        assert get_zig_keyword_category("try") == "error"
        assert get_zig_keyword_category("catch") == "error"

    def test_definition_category(self):
        """Should identify definition keywords."""
        assert get_zig_keyword_category("fn") == "definition"
        assert get_zig_keyword_category("struct") == "definition"
        assert get_zig_keyword_category("const") == "definition"

    def test_memory_category(self):
        """Should identify memory keywords."""
        assert get_zig_keyword_category("defer") == "memory"
        assert get_zig_keyword_category("noalias") == "memory"

    def test_async_category(self):
        """Should identify async keywords."""
        assert get_zig_keyword_category("async") == "async"
        assert get_zig_keyword_category("await") == "async"

    def test_unknown_category(self):
        """Should return unknown for non-keywords."""
        assert get_zig_keyword_category("notakeyword") == "unknown"


# ===========================================================================
# Getter Function Tests
# ===========================================================================


class TestZigGetterFunctions:
    """Tests for getter functions."""

    def test_get_keywords(self):
        """get_zig_keywords should return keywords."""
        keywords = get_zig_keywords()
        assert "fn" in keywords
        assert "const" in keywords
        assert "if" in keywords

    def test_get_builtins(self):
        """get_zig_builtins should return builtins."""
        builtins = get_zig_builtins()
        assert "@import" in builtins
        assert "@TypeOf" in builtins

    def test_get_operators(self):
        """get_zig_operators should return operators."""
        operators = get_zig_operators()
        assert "+" in operators
        assert "==" in operators

    def test_get_delimiters(self):
        """get_zig_delimiters should return delimiters."""
        delimiters = get_zig_delimiters()
        assert "(" in delimiters
        assert "{" in delimiters

    def test_get_primitive_types(self):
        """get_zig_primitive_types should return types."""
        types = get_zig_primitive_types()
        assert "i32" in types
        assert "bool" in types
