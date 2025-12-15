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
"""Tests for Kotlin token classifier."""

import pytest

from core.token_classifier import TokenCategory
from core.token_classifier_kotlin import (
    KOTLIN_HARD_KEYWORDS,
    KOTLIN_SOFT_KEYWORDS,
    KOTLIN_MODIFIER_KEYWORDS,
    KOTLIN_ALL_KEYWORDS,
    KOTLIN_BUILTIN_TYPES,
    KOTLIN_BUILTIN_FUNCTIONS,
    KOTLIN_ALL_BUILTINS,
    KOTLIN_ALL_OPERATORS,
    KOTLIN_DELIMITERS,
    is_kotlin_keyword,
    is_kotlin_builtin,
    is_kotlin_builtin_type,
    is_kotlin_builtin_function,
    is_kotlin_operator,
    is_kotlin_delimiter,
    is_kotlin_int_literal,
    is_kotlin_float_literal,
    is_kotlin_char_literal,
    is_kotlin_string_literal,
    is_kotlin_bool_literal,
    is_kotlin_null_literal,
    is_kotlin_identifier,
    parse_kotlin_int_literal,
    parse_kotlin_float_literal,
    classify_kotlin_token,
    get_kotlin_keywords,
    get_kotlin_builtins,
    get_kotlin_operators,
    get_kotlin_delimiters,
)


class TestKotlinKeywords:
    """Tests for Kotlin keyword detection."""

    def test_hard_keywords(self):
        """Should recognize hard keywords."""
        assert "fun" in KOTLIN_HARD_KEYWORDS
        assert "val" in KOTLIN_HARD_KEYWORDS
        assert "var" in KOTLIN_HARD_KEYWORDS
        assert "class" in KOTLIN_HARD_KEYWORDS
        assert "interface" in KOTLIN_HARD_KEYWORDS
        assert "if" in KOTLIN_HARD_KEYWORDS
        assert "when" in KOTLIN_HARD_KEYWORDS
        assert "for" in KOTLIN_HARD_KEYWORDS
        assert "while" in KOTLIN_HARD_KEYWORDS
        assert "return" in KOTLIN_HARD_KEYWORDS

    def test_soft_keywords(self):
        """Should recognize soft keywords."""
        assert "by" in KOTLIN_SOFT_KEYWORDS
        assert "catch" in KOTLIN_SOFT_KEYWORDS
        assert "constructor" in KOTLIN_SOFT_KEYWORDS
        assert "finally" in KOTLIN_SOFT_KEYWORDS
        assert "get" in KOTLIN_SOFT_KEYWORDS
        assert "set" in KOTLIN_SOFT_KEYWORDS
        assert "import" in KOTLIN_SOFT_KEYWORDS
        assert "where" in KOTLIN_SOFT_KEYWORDS

    def test_modifier_keywords(self):
        """Should recognize modifier keywords."""
        assert "public" in KOTLIN_MODIFIER_KEYWORDS
        assert "private" in KOTLIN_MODIFIER_KEYWORDS
        assert "protected" in KOTLIN_MODIFIER_KEYWORDS
        assert "internal" in KOTLIN_MODIFIER_KEYWORDS
        assert "abstract" in KOTLIN_MODIFIER_KEYWORDS
        assert "open" in KOTLIN_MODIFIER_KEYWORDS
        assert "sealed" in KOTLIN_MODIFIER_KEYWORDS
        assert "data" in KOTLIN_MODIFIER_KEYWORDS
        assert "suspend" in KOTLIN_MODIFIER_KEYWORDS

    def test_is_kotlin_keyword(self):
        """Should detect keywords."""
        assert is_kotlin_keyword("fun")
        assert is_kotlin_keyword("class")
        assert is_kotlin_keyword("abstract")
        assert not is_kotlin_keyword("myFunction")
        assert not is_kotlin_keyword("println")

    def test_get_kotlin_keywords(self):
        """Should return all keywords."""
        keywords = get_kotlin_keywords()
        assert keywords == KOTLIN_ALL_KEYWORDS
        assert len(keywords) > 50


class TestKotlinBuiltins:
    """Tests for Kotlin builtin detection."""

    def test_builtin_types(self):
        """Should recognize builtin types."""
        assert "Int" in KOTLIN_BUILTIN_TYPES
        assert "Long" in KOTLIN_BUILTIN_TYPES
        assert "String" in KOTLIN_BUILTIN_TYPES
        assert "Boolean" in KOTLIN_BUILTIN_TYPES
        assert "List" in KOTLIN_BUILTIN_TYPES
        assert "Map" in KOTLIN_BUILTIN_TYPES
        assert "Array" in KOTLIN_BUILTIN_TYPES

    def test_builtin_functions(self):
        """Should recognize builtin functions."""
        assert "println" in KOTLIN_BUILTIN_FUNCTIONS
        assert "print" in KOTLIN_BUILTIN_FUNCTIONS
        assert "listOf" in KOTLIN_BUILTIN_FUNCTIONS
        assert "mapOf" in KOTLIN_BUILTIN_FUNCTIONS
        assert "arrayOf" in KOTLIN_BUILTIN_FUNCTIONS
        assert "require" in KOTLIN_BUILTIN_FUNCTIONS
        assert "let" in KOTLIN_BUILTIN_FUNCTIONS
        assert "apply" in KOTLIN_BUILTIN_FUNCTIONS

    def test_is_kotlin_builtin(self):
        """Should detect builtins."""
        assert is_kotlin_builtin("Int")
        assert is_kotlin_builtin("println")
        assert is_kotlin_builtin("listOf")
        assert not is_kotlin_builtin("myFunction")
        assert not is_kotlin_builtin("fun")

    def test_is_kotlin_builtin_type(self):
        """Should detect builtin types."""
        assert is_kotlin_builtin_type("Int")
        assert is_kotlin_builtin_type("String")
        assert not is_kotlin_builtin_type("println")

    def test_is_kotlin_builtin_function(self):
        """Should detect builtin functions."""
        assert is_kotlin_builtin_function("println")
        assert is_kotlin_builtin_function("listOf")
        assert not is_kotlin_builtin_function("Int")


class TestKotlinOperators:
    """Tests for Kotlin operator detection."""

    def test_arithmetic_operators(self):
        """Should recognize arithmetic operators."""
        assert "+" in KOTLIN_ALL_OPERATORS
        assert "-" in KOTLIN_ALL_OPERATORS
        assert "*" in KOTLIN_ALL_OPERATORS
        assert "/" in KOTLIN_ALL_OPERATORS
        assert "%" in KOTLIN_ALL_OPERATORS

    def test_comparison_operators(self):
        """Should recognize comparison operators."""
        assert "==" in KOTLIN_ALL_OPERATORS
        assert "!=" in KOTLIN_ALL_OPERATORS
        assert "<" in KOTLIN_ALL_OPERATORS
        assert ">" in KOTLIN_ALL_OPERATORS
        assert "===" in KOTLIN_ALL_OPERATORS  # referential equality
        assert "!==" in KOTLIN_ALL_OPERATORS

    def test_null_safety_operators(self):
        """Should recognize null safety operators."""
        assert "?." in KOTLIN_ALL_OPERATORS
        assert "?:" in KOTLIN_ALL_OPERATORS  # elvis
        assert "!!" in KOTLIN_ALL_OPERATORS

    def test_is_kotlin_operator(self):
        """Should detect operators."""
        assert is_kotlin_operator("+")
        assert is_kotlin_operator("?.")
        assert is_kotlin_operator("::")
        assert not is_kotlin_operator("fun")


class TestKotlinDelimiters:
    """Tests for Kotlin delimiter detection."""

    def test_delimiters(self):
        """Should recognize delimiters."""
        assert "(" in KOTLIN_DELIMITERS
        assert ")" in KOTLIN_DELIMITERS
        assert "{" in KOTLIN_DELIMITERS
        assert "}" in KOTLIN_DELIMITERS
        assert "[" in KOTLIN_DELIMITERS
        assert "]" in KOTLIN_DELIMITERS
        assert "," in KOTLIN_DELIMITERS
        assert ":" in KOTLIN_DELIMITERS

    def test_is_kotlin_delimiter(self):
        """Should detect delimiters."""
        assert is_kotlin_delimiter("(")
        assert is_kotlin_delimiter("{")
        assert not is_kotlin_delimiter("+")


class TestKotlinLiterals:
    """Tests for Kotlin literal detection."""

    def test_int_literals(self):
        """Should detect integer literals."""
        assert is_kotlin_int_literal("42")
        assert is_kotlin_int_literal("1_000_000")
        assert is_kotlin_int_literal("0xFF")
        assert is_kotlin_int_literal("0b1010")
        assert is_kotlin_int_literal("100L")
        assert not is_kotlin_int_literal("3.14")
        assert not is_kotlin_int_literal("abc")

    def test_parse_int_literals(self):
        """Should parse integer literals."""
        assert parse_kotlin_int_literal("42") == 42
        assert parse_kotlin_int_literal("1_000") == 1000
        assert parse_kotlin_int_literal("0xFF") == 255
        assert parse_kotlin_int_literal("0b1010") == 10
        assert parse_kotlin_int_literal("100L") == 100

    def test_float_literals(self):
        """Should detect float literals."""
        assert is_kotlin_float_literal("3.14")
        assert is_kotlin_float_literal("2.0e10")
        assert is_kotlin_float_literal("1.5f")
        assert not is_kotlin_float_literal("42")
        assert not is_kotlin_float_literal("abc")

    def test_parse_float_literals(self):
        """Should parse float literals."""
        assert parse_kotlin_float_literal("3.14") == 3.14
        assert parse_kotlin_float_literal("2.0e10") == 2.0e10
        assert parse_kotlin_float_literal("1.5f") == 1.5

    def test_char_literals(self):
        """Should detect character literals."""
        assert is_kotlin_char_literal("'a'")
        assert is_kotlin_char_literal("'\\n'")
        assert is_kotlin_char_literal("'\\u0041'")
        assert not is_kotlin_char_literal("'ab'")
        assert not is_kotlin_char_literal("\"a\"")

    def test_string_literals(self):
        """Should detect string literals."""
        assert is_kotlin_string_literal('"hello"')
        assert is_kotlin_string_literal('"hello $name"')
        assert not is_kotlin_string_literal("hello")
        assert not is_kotlin_string_literal("'h'")

    def test_bool_literals(self):
        """Should detect boolean literals."""
        assert is_kotlin_bool_literal("true")
        assert is_kotlin_bool_literal("false")
        assert not is_kotlin_bool_literal("True")
        assert not is_kotlin_bool_literal("1")

    def test_null_literal(self):
        """Should detect null literal."""
        assert is_kotlin_null_literal("null")
        assert not is_kotlin_null_literal("NULL")
        assert not is_kotlin_null_literal("nil")


class TestKotlinIdentifiers:
    """Tests for Kotlin identifier detection."""

    def test_valid_identifiers(self):
        """Should detect valid identifiers."""
        assert is_kotlin_identifier("myVar")
        assert is_kotlin_identifier("_private")
        assert is_kotlin_identifier("myFunction123")
        assert is_kotlin_identifier("camelCase")

    def test_invalid_identifiers(self):
        """Should reject invalid identifiers."""
        assert not is_kotlin_identifier("fun")  # keyword
        assert not is_kotlin_identifier("class")  # keyword
        assert not is_kotlin_identifier("Int")  # builtin
        assert not is_kotlin_identifier("123abc")  # starts with digit


class TestKotlinTokenClassification:
    """Tests for Kotlin token classification."""

    def test_classify_keyword(self):
        """Should classify keywords."""
        result = classify_kotlin_token("fun")
        assert result.category == TokenCategory.KEYWORD
        assert result.keyword_name == "fun"

    def test_classify_builtin_type(self):
        """Should classify builtin types."""
        result = classify_kotlin_token("Int")
        assert result.category == TokenCategory.BUILTIN
        assert result.keyword_name == "Int"

    def test_classify_builtin_function(self):
        """Should classify builtin functions."""
        result = classify_kotlin_token("println")
        assert result.category == TokenCategory.BUILTIN

    def test_classify_int_literal(self):
        """Should classify integer literals."""
        result = classify_kotlin_token("42")
        assert result.category == TokenCategory.INT_LITERAL
        assert result.literal_value == 42

    def test_classify_float_literal(self):
        """Should classify float literals."""
        result = classify_kotlin_token("3.14")
        assert result.category == TokenCategory.FLOAT_LITERAL

    def test_classify_string_literal(self):
        """Should classify string literals."""
        result = classify_kotlin_token('"hello"')
        assert result.category == TokenCategory.STRING_LITERAL

    def test_classify_bool_literal(self):
        """Should classify boolean literals as keywords (they're in keyword set)."""
        # In Kotlin, true/false are keywords that happen to be literals
        result = classify_kotlin_token("true")
        assert result.category == TokenCategory.KEYWORD
        assert result.keyword_name == "true"

    def test_classify_null_literal(self):
        """Should classify null as keyword (it's in keyword set)."""
        # In Kotlin, null is a keyword
        result = classify_kotlin_token("null")
        assert result.category == TokenCategory.KEYWORD
        assert result.keyword_name == "null"

    def test_classify_operator(self):
        """Should classify operators."""
        result = classify_kotlin_token("+")
        assert result.category == TokenCategory.OPERATOR

    def test_classify_delimiter(self):
        """Should classify delimiters."""
        result = classify_kotlin_token("(")
        assert result.category == TokenCategory.DELIMITER

    def test_classify_identifier(self):
        """Should classify identifiers."""
        result = classify_kotlin_token("myVariable")
        assert result.category == TokenCategory.IDENTIFIER

    def test_classify_whitespace(self):
        """Should classify whitespace."""
        result = classify_kotlin_token("   ")
        assert result.category == TokenCategory.WHITESPACE

    def test_classify_comment(self):
        """Should classify comments."""
        result = classify_kotlin_token("// comment")
        assert result.category == TokenCategory.COMMENT
