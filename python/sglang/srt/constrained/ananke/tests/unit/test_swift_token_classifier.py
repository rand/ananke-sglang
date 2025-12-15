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
"""Tests for Swift token classifier."""

import pytest

from core.token_classifier import TokenCategory
from core.token_classifier_swift import (
    SWIFT_DECLARATION_KEYWORDS,
    SWIFT_STATEMENT_KEYWORDS,
    SWIFT_EXPRESSION_KEYWORDS,
    SWIFT_CONTEXT_KEYWORDS,
    SWIFT_ALL_KEYWORDS,
    SWIFT_BUILTIN_TYPES,
    SWIFT_BUILTIN_FUNCTIONS,
    SWIFT_ALL_BUILTINS,
    SWIFT_ALL_OPERATORS,
    SWIFT_DELIMITERS,
    is_swift_keyword,
    is_swift_builtin,
    is_swift_builtin_type,
    is_swift_builtin_function,
    is_swift_operator,
    is_swift_delimiter,
    is_swift_int_literal,
    is_swift_float_literal,
    is_swift_string_literal,
    is_swift_bool_literal,
    is_swift_nil_literal,
    is_swift_identifier,
    parse_swift_int_literal,
    parse_swift_float_literal,
    classify_swift_token,
    get_swift_keywords,
    get_swift_builtins,
    get_swift_operators,
    get_swift_delimiters,
)


class TestSwiftKeywords:
    """Tests for Swift keyword detection."""

    def test_declaration_keywords(self):
        """Should recognize declaration keywords."""
        assert "func" in SWIFT_DECLARATION_KEYWORDS
        assert "var" in SWIFT_DECLARATION_KEYWORDS
        assert "let" in SWIFT_DECLARATION_KEYWORDS
        assert "class" in SWIFT_DECLARATION_KEYWORDS
        assert "struct" in SWIFT_DECLARATION_KEYWORDS
        assert "enum" in SWIFT_DECLARATION_KEYWORDS
        assert "protocol" in SWIFT_DECLARATION_KEYWORDS
        assert "extension" in SWIFT_DECLARATION_KEYWORDS
        assert "import" in SWIFT_DECLARATION_KEYWORDS

    def test_statement_keywords(self):
        """Should recognize statement keywords."""
        assert "if" in SWIFT_STATEMENT_KEYWORDS
        assert "else" in SWIFT_STATEMENT_KEYWORDS
        assert "for" in SWIFT_STATEMENT_KEYWORDS
        assert "while" in SWIFT_STATEMENT_KEYWORDS
        assert "switch" in SWIFT_STATEMENT_KEYWORDS
        assert "case" in SWIFT_STATEMENT_KEYWORDS
        assert "return" in SWIFT_STATEMENT_KEYWORDS
        assert "guard" in SWIFT_STATEMENT_KEYWORDS
        assert "defer" in SWIFT_STATEMENT_KEYWORDS

    def test_expression_keywords(self):
        """Should recognize expression keywords."""
        assert "as" in SWIFT_EXPRESSION_KEYWORDS
        assert "is" in SWIFT_EXPRESSION_KEYWORDS
        assert "try" in SWIFT_EXPRESSION_KEYWORDS
        assert "true" in SWIFT_EXPRESSION_KEYWORDS
        assert "false" in SWIFT_EXPRESSION_KEYWORDS
        assert "nil" in SWIFT_EXPRESSION_KEYWORDS
        assert "self" in SWIFT_EXPRESSION_KEYWORDS
        assert "super" in SWIFT_EXPRESSION_KEYWORDS

    def test_context_keywords(self):
        """Should recognize context keywords."""
        assert "get" in SWIFT_CONTEXT_KEYWORDS
        assert "set" in SWIFT_CONTEXT_KEYWORDS
        assert "willSet" in SWIFT_CONTEXT_KEYWORDS
        assert "didSet" in SWIFT_CONTEXT_KEYWORDS
        assert "mutating" in SWIFT_CONTEXT_KEYWORDS
        assert "lazy" in SWIFT_CONTEXT_KEYWORDS
        assert "weak" in SWIFT_CONTEXT_KEYWORDS

    def test_is_swift_keyword(self):
        """Should detect keywords."""
        assert is_swift_keyword("func")
        assert is_swift_keyword("class")
        assert is_swift_keyword("guard")
        assert not is_swift_keyword("myFunction")
        assert not is_swift_keyword("print")

    def test_get_swift_keywords(self):
        """Should return all keywords."""
        keywords = get_swift_keywords()
        assert keywords == SWIFT_ALL_KEYWORDS
        assert len(keywords) > 50


class TestSwiftBuiltins:
    """Tests for Swift builtin detection."""

    def test_builtin_types(self):
        """Should recognize builtin types."""
        assert "Int" in SWIFT_BUILTIN_TYPES
        assert "Double" in SWIFT_BUILTIN_TYPES
        assert "String" in SWIFT_BUILTIN_TYPES
        assert "Bool" in SWIFT_BUILTIN_TYPES
        assert "Array" in SWIFT_BUILTIN_TYPES
        assert "Dictionary" in SWIFT_BUILTIN_TYPES
        assert "Optional" in SWIFT_BUILTIN_TYPES
        assert "Result" in SWIFT_BUILTIN_TYPES

    def test_builtin_functions(self):
        """Should recognize builtin functions."""
        assert "print" in SWIFT_BUILTIN_FUNCTIONS
        assert "debugPrint" in SWIFT_BUILTIN_FUNCTIONS
        assert "abs" in SWIFT_BUILTIN_FUNCTIONS
        assert "min" in SWIFT_BUILTIN_FUNCTIONS
        assert "max" in SWIFT_BUILTIN_FUNCTIONS
        assert "fatalError" in SWIFT_BUILTIN_FUNCTIONS

    def test_is_swift_builtin(self):
        """Should detect builtins."""
        assert is_swift_builtin("Int")
        assert is_swift_builtin("print")
        assert not is_swift_builtin("myFunction")
        assert not is_swift_builtin("func")

    def test_is_swift_builtin_type(self):
        """Should detect builtin types."""
        assert is_swift_builtin_type("Int")
        assert is_swift_builtin_type("String")
        assert not is_swift_builtin_type("print")

    def test_is_swift_builtin_function(self):
        """Should detect builtin functions."""
        assert is_swift_builtin_function("print")
        assert is_swift_builtin_function("abs")
        assert not is_swift_builtin_function("Int")


class TestSwiftOperators:
    """Tests for Swift operator detection."""

    def test_arithmetic_operators(self):
        """Should recognize arithmetic operators."""
        assert "+" in SWIFT_ALL_OPERATORS
        assert "-" in SWIFT_ALL_OPERATORS
        assert "*" in SWIFT_ALL_OPERATORS
        assert "/" in SWIFT_ALL_OPERATORS
        assert "%" in SWIFT_ALL_OPERATORS

    def test_overflow_operators(self):
        """Should recognize overflow operators."""
        assert "&+" in SWIFT_ALL_OPERATORS
        assert "&-" in SWIFT_ALL_OPERATORS
        assert "&*" in SWIFT_ALL_OPERATORS

    def test_comparison_operators(self):
        """Should recognize comparison operators."""
        assert "==" in SWIFT_ALL_OPERATORS
        assert "!=" in SWIFT_ALL_OPERATORS
        assert "<" in SWIFT_ALL_OPERATORS
        assert ">" in SWIFT_ALL_OPERATORS
        assert "===" in SWIFT_ALL_OPERATORS  # identity
        assert "!==" in SWIFT_ALL_OPERATORS

    def test_range_operators(self):
        """Should recognize range operators."""
        assert "..." in SWIFT_ALL_OPERATORS  # closed range
        assert "..<" in SWIFT_ALL_OPERATORS  # half-open range

    def test_optional_operators(self):
        """Should recognize optional operators."""
        assert "?" in SWIFT_ALL_OPERATORS
        assert "!" in SWIFT_ALL_OPERATORS
        assert "??" in SWIFT_ALL_OPERATORS  # nil coalescing

    def test_is_swift_operator(self):
        """Should detect operators."""
        assert is_swift_operator("+")
        assert is_swift_operator("??")
        assert is_swift_operator("->")
        assert not is_swift_operator("func")


class TestSwiftDelimiters:
    """Tests for Swift delimiter detection."""

    def test_delimiters(self):
        """Should recognize delimiters."""
        assert "(" in SWIFT_DELIMITERS
        assert ")" in SWIFT_DELIMITERS
        assert "{" in SWIFT_DELIMITERS
        assert "}" in SWIFT_DELIMITERS
        assert "[" in SWIFT_DELIMITERS
        assert "]" in SWIFT_DELIMITERS
        assert "," in SWIFT_DELIMITERS
        assert ":" in SWIFT_DELIMITERS

    def test_is_swift_delimiter(self):
        """Should detect delimiters."""
        assert is_swift_delimiter("(")
        assert is_swift_delimiter("{")
        assert not is_swift_delimiter("+")


class TestSwiftLiterals:
    """Tests for Swift literal detection."""

    def test_int_literals(self):
        """Should detect integer literals."""
        assert is_swift_int_literal("42")
        assert is_swift_int_literal("1_000_000")
        assert is_swift_int_literal("0xFF")
        assert is_swift_int_literal("0b1010")
        assert is_swift_int_literal("0o755")
        assert not is_swift_int_literal("3.14")
        assert not is_swift_int_literal("abc")

    def test_parse_int_literals(self):
        """Should parse integer literals."""
        assert parse_swift_int_literal("42") == 42
        assert parse_swift_int_literal("1_000") == 1000
        assert parse_swift_int_literal("0xFF") == 255
        assert parse_swift_int_literal("0b1010") == 10
        assert parse_swift_int_literal("0o10") == 8

    def test_float_literals(self):
        """Should detect float literals."""
        assert is_swift_float_literal("3.14")
        assert is_swift_float_literal("2.0e10")
        assert not is_swift_float_literal("42")
        assert not is_swift_float_literal("abc")

    def test_parse_float_literals(self):
        """Should parse float literals."""
        assert parse_swift_float_literal("3.14") == 3.14
        assert parse_swift_float_literal("2.0e10") == 2.0e10

    def test_string_literals(self):
        """Should detect string literals."""
        assert is_swift_string_literal('"hello"')
        assert is_swift_string_literal('"hello \\(name)"')
        assert not is_swift_string_literal("hello")

    def test_bool_literals(self):
        """Should detect boolean literals."""
        assert is_swift_bool_literal("true")
        assert is_swift_bool_literal("false")
        assert not is_swift_bool_literal("True")
        assert not is_swift_bool_literal("1")

    def test_nil_literal(self):
        """Should detect nil literal."""
        assert is_swift_nil_literal("nil")
        assert not is_swift_nil_literal("null")
        assert not is_swift_nil_literal("NULL")


class TestSwiftIdentifiers:
    """Tests for Swift identifier detection."""

    def test_valid_identifiers(self):
        """Should detect valid identifiers."""
        assert is_swift_identifier("myVar")
        assert is_swift_identifier("_private")
        assert is_swift_identifier("myFunction123")
        assert is_swift_identifier("camelCase")

    def test_invalid_identifiers(self):
        """Should reject invalid identifiers."""
        assert not is_swift_identifier("func")  # keyword
        assert not is_swift_identifier("class")  # keyword
        assert not is_swift_identifier("Int")  # builtin
        assert not is_swift_identifier("123abc")  # starts with digit


class TestSwiftTokenClassification:
    """Tests for Swift token classification."""

    def test_classify_keyword(self):
        """Should classify keywords."""
        result = classify_swift_token("func")
        assert result.category == TokenCategory.KEYWORD
        assert result.keyword_name == "func"

    def test_classify_builtin_type(self):
        """Should classify builtin types."""
        result = classify_swift_token("Int")
        assert result.category == TokenCategory.BUILTIN
        assert result.keyword_name == "Int"

    def test_classify_builtin_function(self):
        """Should classify builtin functions."""
        result = classify_swift_token("print")
        assert result.category == TokenCategory.BUILTIN

    def test_classify_int_literal(self):
        """Should classify integer literals."""
        result = classify_swift_token("42")
        assert result.category == TokenCategory.INT_LITERAL
        assert result.literal_value == 42

    def test_classify_float_literal(self):
        """Should classify float literals."""
        result = classify_swift_token("3.14")
        assert result.category == TokenCategory.FLOAT_LITERAL

    def test_classify_string_literal(self):
        """Should classify string literals."""
        result = classify_swift_token('"hello"')
        assert result.category == TokenCategory.STRING_LITERAL

    def test_classify_bool_literal(self):
        """Should classify boolean literals as keywords (they're in keyword set)."""
        # In Swift, true/false are keywords that happen to be literals
        result = classify_swift_token("true")
        assert result.category == TokenCategory.KEYWORD
        assert result.keyword_name == "true"

    def test_classify_nil_literal(self):
        """Should classify nil as keyword (it's in keyword set)."""
        # In Swift, nil is a keyword
        result = classify_swift_token("nil")
        assert result.category == TokenCategory.KEYWORD
        assert result.keyword_name == "nil"

    def test_classify_operator(self):
        """Should classify operators."""
        result = classify_swift_token("+")
        assert result.category == TokenCategory.OPERATOR

    def test_classify_delimiter(self):
        """Should classify delimiters."""
        result = classify_swift_token("(")
        assert result.category == TokenCategory.DELIMITER

    def test_classify_identifier(self):
        """Should classify identifiers."""
        result = classify_swift_token("myVariable")
        assert result.category == TokenCategory.IDENTIFIER

    def test_classify_whitespace(self):
        """Should classify whitespace."""
        result = classify_swift_token("   ")
        assert result.category == TokenCategory.WHITESPACE

    def test_classify_comment(self):
        """Should classify comments."""
        result = classify_swift_token("// comment")
        assert result.category == TokenCategory.COMMENT
