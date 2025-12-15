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
"""Tests for Go token classifier."""

import pytest

from core.token_classifier import TokenCategory
from core.token_classifier_go import (
    classify_go_token,
    is_go_keyword,
    is_go_builtin,
    is_go_operator,
    is_go_int_literal,
    is_go_float_literal,
    is_go_string_literal,
    is_go_identifier,
    parse_go_int_literal,
    parse_go_float_literal,
    get_go_keywords,
    get_go_builtins,
    GO_ALL_KEYWORDS,
    GO_BUILTIN_TYPES,
    GO_BUILTIN_FUNCTIONS,
)


class TestGoKeywordClassification:
    """Tests for Go keyword classification."""

    def test_control_keywords(self):
        """Should recognize control flow keywords."""
        keywords = ["if", "else", "for", "switch", "case", "return", "break"]
        for kw in keywords:
            assert is_go_keyword(kw), f"{kw} should be a keyword"
            result = classify_go_token(kw)
            assert result.category == TokenCategory.KEYWORD
            assert result.keyword_name == kw

    def test_definition_keywords(self):
        """Should recognize definition keywords."""
        keywords = ["func", "var", "const", "type", "struct", "interface"]
        for kw in keywords:
            assert is_go_keyword(kw), f"{kw} should be a keyword"
            result = classify_go_token(kw)
            assert result.category == TokenCategory.KEYWORD

    def test_special_keywords(self):
        """Should recognize special keywords."""
        keywords = ["defer", "go"]
        for kw in keywords:
            assert is_go_keyword(kw), f"{kw} should be a keyword"

    def test_non_keywords(self):
        """Should not classify non-keywords as keywords."""
        non_keywords = ["foo", "bar", "myFunc", "x"]
        for word in non_keywords:
            assert not is_go_keyword(word)


class TestGoBuiltinClassification:
    """Tests for Go builtin classification."""

    def test_builtin_types(self):
        """Should recognize builtin types."""
        types = ["int", "string", "bool", "float64", "error", "any"]
        for t in types:
            assert is_go_builtin(t), f"{t} should be a builtin"
            result = classify_go_token(t)
            assert result.category == TokenCategory.BUILTIN

    def test_builtin_functions(self):
        """Should recognize builtin functions."""
        funcs = ["len", "cap", "make", "new", "append", "panic", "recover"]
        for f in funcs:
            assert is_go_builtin(f), f"{f} should be a builtin"
            result = classify_go_token(f)
            assert result.category == TokenCategory.BUILTIN

    def test_bool_literals(self):
        """Should classify bool literals correctly."""
        result_true = classify_go_token("true")
        assert result_true.category == TokenCategory.BOOL_LITERAL
        assert result_true.literal_value is True

        result_false = classify_go_token("false")
        assert result_false.category == TokenCategory.BOOL_LITERAL
        assert result_false.literal_value is False

    def test_nil_literal(self):
        """Should classify nil correctly."""
        result = classify_go_token("nil")
        assert result.category == TokenCategory.NONE_LITERAL
        assert result.literal_value is None

    def test_iota(self):
        """Should classify iota as builtin."""
        result = classify_go_token("iota")
        assert result.category == TokenCategory.BUILTIN


class TestGoIntegerLiterals:
    """Tests for Go integer literal classification."""

    def test_decimal_integers(self):
        """Should recognize decimal integer literals."""
        literals = ["0", "42", "123", "1_000_000"]
        for lit in literals:
            assert is_go_int_literal(lit), f"{lit} should be int literal"
            result = classify_go_token(lit)
            assert result.category == TokenCategory.INT_LITERAL

    def test_binary_integers(self):
        """Should recognize binary integer literals."""
        literals = ["0b1010", "0B1111", "0b1010_1010"]
        for lit in literals:
            assert is_go_int_literal(lit), f"{lit} should be int literal"

    def test_octal_integers(self):
        """Should recognize octal integer literals."""
        literals = ["0o755", "0O644", "0755"]
        for lit in literals:
            assert is_go_int_literal(lit), f"{lit} should be int literal"

    def test_hex_integers(self):
        """Should recognize hex integer literals."""
        literals = ["0xFF", "0XAB", "0xDEAD_BEEF"]
        for lit in literals:
            assert is_go_int_literal(lit), f"{lit} should be int literal"

    def test_parse_decimal(self):
        """Should parse decimal integers correctly."""
        assert parse_go_int_literal("42") == 42
        assert parse_go_int_literal("1_000") == 1000

    def test_parse_binary(self):
        """Should parse binary integers correctly."""
        assert parse_go_int_literal("0b1010") == 10
        assert parse_go_int_literal("0B1111") == 15

    def test_parse_hex(self):
        """Should parse hex integers correctly."""
        assert parse_go_int_literal("0xFF") == 255
        assert parse_go_int_literal("0x10") == 16


class TestGoFloatLiterals:
    """Tests for Go float literal classification."""

    def test_basic_floats(self):
        """Should recognize basic float literals."""
        literals = ["3.14", "0.5", ".5", "1."]
        for lit in literals:
            assert is_go_float_literal(lit), f"{lit} should be float literal"
            result = classify_go_token(lit)
            assert result.category == TokenCategory.FLOAT_LITERAL

    def test_scientific_notation(self):
        """Should recognize scientific notation."""
        literals = ["1e10", "1.5e-3", "2.5E+10"]
        for lit in literals:
            assert is_go_float_literal(lit), f"{lit} should be float literal"

    def test_parse_floats(self):
        """Should parse floats correctly."""
        assert parse_go_float_literal("3.14") == pytest.approx(3.14)
        assert parse_go_float_literal("1e10") == pytest.approx(1e10)


class TestGoStringLiterals:
    """Tests for Go string literal classification."""

    def test_interpreted_strings(self):
        """Should recognize interpreted string literals."""
        literals = ['"hello"', '"world"', '""', '"line\\n"']
        for lit in literals:
            assert is_go_string_literal(lit), f"{lit} should be string literal"
            result = classify_go_token(lit)
            assert result.category == TokenCategory.STRING_LITERAL

    def test_raw_strings(self):
        """Should recognize raw string literals."""
        literals = ["`hello`", "`multi\nline`"]
        for lit in literals:
            assert is_go_string_literal(lit), f"{lit} should be string literal"


class TestGoOperatorClassification:
    """Tests for Go operator classification."""

    def test_arithmetic_operators(self):
        """Should recognize arithmetic operators."""
        ops = ["+", "-", "*", "/", "%"]
        for op in ops:
            assert is_go_operator(op), f"{op} should be operator"
            result = classify_go_token(op)
            assert result.category == TokenCategory.OPERATOR

    def test_comparison_operators(self):
        """Should recognize comparison operators."""
        ops = ["==", "!=", "<", ">", "<=", ">="]
        for op in ops:
            assert is_go_operator(op), f"{op} should be operator"

    def test_assignment_operators(self):
        """Should recognize assignment operators."""
        ops = ["=", ":=", "+=", "-=", "*=", "/="]
        for op in ops:
            assert is_go_operator(op), f"{op} should be operator"

    def test_channel_operator(self):
        """Should recognize channel operator."""
        assert is_go_operator("<-")


class TestGoIdentifierClassification:
    """Tests for Go identifier classification."""

    def test_valid_identifiers(self):
        """Should recognize valid identifiers."""
        identifiers = ["foo", "bar", "myFunc", "_private", "x1", "camelCase"]
        for ident in identifiers:
            assert is_go_identifier(ident), f"{ident} should be identifier"
            result = classify_go_token(ident)
            assert result.category == TokenCategory.IDENTIFIER

    def test_keywords_not_identifiers(self):
        """Keywords should not be identifiers."""
        for kw in GO_ALL_KEYWORDS:
            assert not is_go_identifier(kw)

    def test_builtins_not_identifiers(self):
        """Builtins should not be identifiers."""
        for builtin in GO_BUILTIN_TYPES:
            assert not is_go_identifier(builtin)


class TestGoDelimiterClassification:
    """Tests for Go delimiter classification."""

    def test_delimiters(self):
        """Should classify delimiters correctly."""
        delimiters = ["(", ")", "[", "]", "{", "}", ",", ";", ":"]
        for d in delimiters:
            result = classify_go_token(d)
            assert result.category == TokenCategory.DELIMITER


class TestGoHelperFunctions:
    """Tests for Go helper functions."""

    def test_get_keywords(self):
        """Should return all keywords."""
        keywords = get_go_keywords()
        assert "func" in keywords
        assert "if" in keywords
        assert "for" in keywords

    def test_get_builtins(self):
        """Should return all builtins."""
        builtins = get_go_builtins()
        assert "int" in builtins
        assert "len" in builtins
        assert "true" in builtins
