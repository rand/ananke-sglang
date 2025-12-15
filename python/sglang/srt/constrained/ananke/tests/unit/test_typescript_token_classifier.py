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
"""Unit tests for TypeScript token classifier.

Tests for the TypeScript token classification including:
- Keyword classification
- Type keyword classification
- Builtin classification
- Operator classification
- Delimiter classification
- Literal classification
- Identifier classification
- Completion suggestions
"""

import pytest

from core.token_classifier import (
    TokenCategory,
    get_language_keywords,
    get_language_builtins,
    classify_token_for_language,
    supported_classifier_languages,
)
from core.token_classifier_typescript import (
    classify_typescript_token,
    get_typescript_completions,
    is_typescript_keyword,
    is_typescript_builtin,
    is_typescript_operator,
    is_typescript_type_keyword,
    TYPESCRIPT_ALL_KEYWORDS,
    TYPESCRIPT_ALL_BUILTINS,
    TYPESCRIPT_ALL_OPERATORS,
    TYPESCRIPT_DELIMITERS,
    TYPESCRIPT_PRIMITIVE_TYPES,
    TYPESCRIPT_UTILITY_TYPES,
)


# ===========================================================================
# Registration Tests
# ===========================================================================


class TestTypeScriptClassifierRegistration:
    """Tests for TypeScript classifier registration."""

    def test_typescript_in_supported_languages(self):
        """TypeScript should be in supported classifier languages."""
        langs = supported_classifier_languages()
        assert "typescript" in langs

    def test_get_language_keywords(self):
        """Should return TypeScript keywords."""
        keywords = get_language_keywords("typescript")
        assert "const" in keywords
        assert "function" in keywords
        assert "interface" in keywords
        assert "type" in keywords

    def test_get_language_keywords_js(self):
        """Should return TypeScript keywords for 'js' alias."""
        keywords = get_language_keywords("js")
        assert "const" in keywords

    def test_get_language_builtins(self):
        """Should return TypeScript builtins."""
        builtins = get_language_builtins("typescript")
        assert "Array" in builtins
        assert "Object" in builtins
        assert "Promise" in builtins


# ===========================================================================
# Keyword Classification Tests
# ===========================================================================


class TestTypeScriptKeywordClassification:
    """Tests for keyword classification."""

    # Declaration keywords
    def test_classify_const(self):
        """Should classify 'const' as keyword."""
        result = classify_typescript_token("const")
        assert result.category == TokenCategory.KEYWORD
        assert result.keyword_name == "const"

    def test_classify_let(self):
        """Should classify 'let' as keyword."""
        result = classify_typescript_token("let")
        assert result.category == TokenCategory.KEYWORD
        assert result.keyword_name == "let"

    def test_classify_var(self):
        """Should classify 'var' as keyword."""
        result = classify_typescript_token("var")
        assert result.category == TokenCategory.KEYWORD

    def test_classify_function(self):
        """Should classify 'function' as keyword."""
        result = classify_typescript_token("function")
        assert result.category == TokenCategory.KEYWORD

    def test_classify_class(self):
        """Should classify 'class' as keyword."""
        result = classify_typescript_token("class")
        assert result.category == TokenCategory.KEYWORD

    def test_classify_interface(self):
        """Should classify 'interface' as keyword."""
        result = classify_typescript_token("interface")
        assert result.category == TokenCategory.KEYWORD

    def test_classify_type(self):
        """Should classify 'type' as keyword."""
        result = classify_typescript_token("type")
        assert result.category == TokenCategory.KEYWORD

    def test_classify_enum(self):
        """Should classify 'enum' as keyword."""
        result = classify_typescript_token("enum")
        assert result.category == TokenCategory.KEYWORD

    # Control flow keywords
    def test_classify_if(self):
        """Should classify 'if' as keyword."""
        result = classify_typescript_token("if")
        assert result.category == TokenCategory.KEYWORD

    def test_classify_else(self):
        """Should classify 'else' as keyword."""
        result = classify_typescript_token("else")
        assert result.category == TokenCategory.KEYWORD

    def test_classify_for(self):
        """Should classify 'for' as keyword."""
        result = classify_typescript_token("for")
        assert result.category == TokenCategory.KEYWORD

    def test_classify_while(self):
        """Should classify 'while' as keyword."""
        result = classify_typescript_token("while")
        assert result.category == TokenCategory.KEYWORD

    def test_classify_return(self):
        """Should classify 'return' as keyword."""
        result = classify_typescript_token("return")
        assert result.category == TokenCategory.KEYWORD

    # Modifier keywords
    def test_classify_async(self):
        """Should classify 'async' as keyword."""
        result = classify_typescript_token("async")
        assert result.category == TokenCategory.KEYWORD

    def test_classify_await(self):
        """Should classify 'await' as keyword."""
        result = classify_typescript_token("await")
        assert result.category == TokenCategory.KEYWORD

    def test_classify_export(self):
        """Should classify 'export' as keyword."""
        result = classify_typescript_token("export")
        assert result.category == TokenCategory.KEYWORD

    def test_classify_import(self):
        """Should classify 'import' as keyword."""
        result = classify_typescript_token("import")
        assert result.category == TokenCategory.KEYWORD

    # TypeScript-specific keywords
    def test_classify_abstract(self):
        """Should classify 'abstract' as keyword."""
        result = classify_typescript_token("abstract")
        assert result.category == TokenCategory.KEYWORD

    def test_classify_readonly(self):
        """Should classify 'readonly' as keyword."""
        result = classify_typescript_token("readonly")
        assert result.category == TokenCategory.KEYWORD

    def test_classify_implements(self):
        """Should classify 'implements' as keyword."""
        result = classify_typescript_token("implements")
        assert result.category == TokenCategory.KEYWORD

    def test_classify_keyof(self):
        """Should classify 'keyof' as keyword."""
        result = classify_typescript_token("keyof")
        assert result.category == TokenCategory.KEYWORD

    def test_classify_infer(self):
        """Should classify 'infer' as keyword."""
        result = classify_typescript_token("infer")
        assert result.category == TokenCategory.KEYWORD


# ===========================================================================
# Type Keyword Classification Tests
# ===========================================================================


class TestTypeScriptTypeKeywordClassification:
    """Tests for type keyword classification."""

    def test_classify_string(self):
        """Should classify 'string' as type keyword."""
        result = classify_typescript_token("string")
        assert result.category == TokenCategory.KEYWORD  # Type keywords are keywords
        assert result.keyword_name == "string"

    def test_classify_number(self):
        """Should classify 'number' as type keyword."""
        result = classify_typescript_token("number")
        assert result.category == TokenCategory.KEYWORD

    def test_classify_boolean(self):
        """Should classify 'boolean' as type keyword."""
        result = classify_typescript_token("boolean")
        assert result.category == TokenCategory.KEYWORD

    def test_classify_any(self):
        """Should classify 'any' as type keyword."""
        result = classify_typescript_token("any")
        assert result.category == TokenCategory.KEYWORD

    def test_classify_unknown(self):
        """Should classify 'unknown' as type keyword."""
        result = classify_typescript_token("unknown")
        assert result.category == TokenCategory.KEYWORD

    def test_classify_never(self):
        """Should classify 'never' as type keyword."""
        result = classify_typescript_token("never")
        assert result.category == TokenCategory.KEYWORD

    def test_classify_void(self):
        """Should classify 'void' as type keyword."""
        result = classify_typescript_token("void")
        assert result.category == TokenCategory.KEYWORD

    def test_classify_undefined(self):
        """Should classify 'undefined' as type keyword."""
        result = classify_typescript_token("undefined")
        assert result.category == TokenCategory.KEYWORD

    def test_classify_null(self):
        """Should classify 'null' as type keyword."""
        result = classify_typescript_token("null")
        assert result.category == TokenCategory.KEYWORD


# ===========================================================================
# Builtin Classification Tests
# ===========================================================================


class TestTypeScriptBuiltinClassification:
    """Tests for builtin classification."""

    def test_classify_array(self):
        """Should classify 'Array' as builtin."""
        result = classify_typescript_token("Array")
        assert result.category == TokenCategory.BUILTIN

    def test_classify_object(self):
        """Should classify 'Object' as builtin."""
        result = classify_typescript_token("Object")
        assert result.category == TokenCategory.BUILTIN

    def test_classify_string_class(self):
        """Should classify 'String' as builtin."""
        result = classify_typescript_token("String")
        assert result.category == TokenCategory.BUILTIN

    def test_classify_number_class(self):
        """Should classify 'Number' as builtin."""
        result = classify_typescript_token("Number")
        assert result.category == TokenCategory.BUILTIN

    def test_classify_promise(self):
        """Should classify 'Promise' as builtin."""
        result = classify_typescript_token("Promise")
        assert result.category == TokenCategory.BUILTIN

    def test_classify_map(self):
        """Should classify 'Map' as builtin."""
        result = classify_typescript_token("Map")
        assert result.category == TokenCategory.BUILTIN

    def test_classify_set(self):
        """Should classify 'Set' as builtin."""
        result = classify_typescript_token("Set")
        assert result.category == TokenCategory.BUILTIN

    def test_classify_date(self):
        """Should classify 'Date' as builtin."""
        result = classify_typescript_token("Date")
        assert result.category == TokenCategory.BUILTIN

    def test_classify_json(self):
        """Should classify 'JSON' as builtin."""
        result = classify_typescript_token("JSON")
        assert result.category == TokenCategory.BUILTIN

    def test_classify_console(self):
        """Should classify 'console' as builtin."""
        result = classify_typescript_token("console")
        assert result.category == TokenCategory.BUILTIN

    def test_classify_math(self):
        """Should classify 'Math' as builtin."""
        result = classify_typescript_token("Math")
        assert result.category == TokenCategory.BUILTIN

    # Utility types
    def test_classify_partial(self):
        """Should classify 'Partial' as builtin."""
        result = classify_typescript_token("Partial")
        assert result.category == TokenCategory.BUILTIN

    def test_classify_required(self):
        """Should classify 'Required' as builtin."""
        result = classify_typescript_token("Required")
        assert result.category == TokenCategory.BUILTIN

    def test_classify_readonly_util(self):
        """Should classify 'Readonly' as builtin."""
        result = classify_typescript_token("Readonly")
        assert result.category == TokenCategory.BUILTIN

    def test_classify_record(self):
        """Should classify 'Record' as builtin."""
        result = classify_typescript_token("Record")
        assert result.category == TokenCategory.BUILTIN

    def test_classify_pick(self):
        """Should classify 'Pick' as builtin."""
        result = classify_typescript_token("Pick")
        assert result.category == TokenCategory.BUILTIN

    def test_classify_omit(self):
        """Should classify 'Omit' as builtin."""
        result = classify_typescript_token("Omit")
        assert result.category == TokenCategory.BUILTIN


# ===========================================================================
# Operator Classification Tests
# ===========================================================================


class TestTypeScriptOperatorClassification:
    """Tests for operator classification."""

    # Arithmetic
    def test_classify_plus(self):
        """Should classify '+' as operator."""
        result = classify_typescript_token("+")
        assert result.category == TokenCategory.OPERATOR

    def test_classify_minus(self):
        """Should classify '-' as operator."""
        result = classify_typescript_token("-")
        assert result.category == TokenCategory.OPERATOR

    def test_classify_multiply(self):
        """Should classify '*' as operator."""
        result = classify_typescript_token("*")
        assert result.category == TokenCategory.OPERATOR

    def test_classify_divide(self):
        """Should classify '/' as operator."""
        result = classify_typescript_token("/")
        assert result.category == TokenCategory.OPERATOR

    # Comparison
    def test_classify_strict_equal(self):
        """Should classify '===' as operator."""
        result = classify_typescript_token("===")
        assert result.category == TokenCategory.OPERATOR

    def test_classify_not_strict_equal(self):
        """Should classify '!==' as operator."""
        result = classify_typescript_token("!==")
        assert result.category == TokenCategory.OPERATOR

    # Logical
    def test_classify_and(self):
        """Should classify '&&' as operator."""
        result = classify_typescript_token("&&")
        assert result.category == TokenCategory.OPERATOR

    def test_classify_or(self):
        """Should classify '||' as operator."""
        result = classify_typescript_token("||")
        assert result.category == TokenCategory.OPERATOR

    def test_classify_nullish(self):
        """Should classify '??' as operator."""
        result = classify_typescript_token("??")
        assert result.category == TokenCategory.OPERATOR

    # TypeScript-specific
    def test_classify_arrow(self):
        """Should classify '=>' as operator."""
        result = classify_typescript_token("=>")
        assert result.category == TokenCategory.OPERATOR

    def test_classify_optional_chain(self):
        """Should classify '?.' as operator."""
        result = classify_typescript_token("?.")
        assert result.category == TokenCategory.OPERATOR

    def test_classify_spread(self):
        """Should classify '...' as operator."""
        result = classify_typescript_token("...")
        assert result.category == TokenCategory.OPERATOR


# ===========================================================================
# Delimiter Classification Tests
# ===========================================================================


class TestTypeScriptDelimiterClassification:
    """Tests for delimiter classification."""

    def test_classify_paren_open(self):
        """Should classify '(' as delimiter."""
        result = classify_typescript_token("(")
        assert result.category == TokenCategory.DELIMITER

    def test_classify_paren_close(self):
        """Should classify ')' as delimiter."""
        result = classify_typescript_token(")")
        assert result.category == TokenCategory.DELIMITER

    def test_classify_bracket_open(self):
        """Should classify '[' as delimiter."""
        result = classify_typescript_token("[")
        assert result.category == TokenCategory.DELIMITER

    def test_classify_bracket_close(self):
        """Should classify ']' as delimiter."""
        result = classify_typescript_token("]")
        assert result.category == TokenCategory.DELIMITER

    def test_classify_brace_open(self):
        """Should classify '{' as delimiter."""
        result = classify_typescript_token("{")
        assert result.category == TokenCategory.DELIMITER

    def test_classify_brace_close(self):
        """Should classify '}' as delimiter."""
        result = classify_typescript_token("}")
        assert result.category == TokenCategory.DELIMITER

    def test_classify_semicolon(self):
        """Should classify ';' as delimiter."""
        result = classify_typescript_token(";")
        assert result.category == TokenCategory.DELIMITER

    def test_classify_comma(self):
        """Should classify ',' as delimiter."""
        result = classify_typescript_token(",")
        assert result.category == TokenCategory.DELIMITER

    def test_classify_colon(self):
        """Should classify ':' as delimiter."""
        result = classify_typescript_token(":")
        assert result.category == TokenCategory.DELIMITER


# ===========================================================================
# Literal Classification Tests
# ===========================================================================


class TestTypeScriptLiteralClassification:
    """Tests for literal classification."""

    # Boolean literals
    def test_classify_true(self):
        """Should classify 'true' as bool literal."""
        result = classify_typescript_token("true")
        assert result.category == TokenCategory.BOOL_LITERAL
        assert result.literal_value is True

    def test_classify_false(self):
        """Should classify 'false' as bool literal."""
        result = classify_typescript_token("false")
        assert result.category == TokenCategory.BOOL_LITERAL
        assert result.literal_value is False

    # Numeric literals
    def test_classify_integer(self):
        """Should classify integer as int literal."""
        result = classify_typescript_token("42")
        assert result.category == TokenCategory.INT_LITERAL
        assert result.literal_value == 42

    def test_classify_float(self):
        """Should classify float as float literal."""
        result = classify_typescript_token("3.14")
        assert result.category == TokenCategory.FLOAT_LITERAL

    def test_classify_hex(self):
        """Should classify hex number as int literal."""
        result = classify_typescript_token("0xFF")
        assert result.category == TokenCategory.INT_LITERAL

    # String literals
    def test_classify_single_quote_string(self):
        """Should classify single-quoted string."""
        result = classify_typescript_token("'hello'")
        assert result.category == TokenCategory.STRING_LITERAL

    def test_classify_double_quote_string(self):
        """Should classify double-quoted string."""
        result = classify_typescript_token('"hello"')
        assert result.category == TokenCategory.STRING_LITERAL

    def test_classify_template_string(self):
        """Should classify template string."""
        result = classify_typescript_token("`hello`")
        assert result.category == TokenCategory.STRING_LITERAL


# ===========================================================================
# Identifier Classification Tests
# ===========================================================================


class TestTypeScriptIdentifierClassification:
    """Tests for identifier classification."""

    def test_classify_simple_identifier(self):
        """Should classify simple identifier."""
        result = classify_typescript_token("myVariable")
        assert result.category == TokenCategory.IDENTIFIER

    def test_classify_underscore_identifier(self):
        """Should classify underscore identifier."""
        result = classify_typescript_token("_private")
        assert result.category == TokenCategory.IDENTIFIER

    def test_classify_dollar_identifier(self):
        """Should classify dollar identifier."""
        result = classify_typescript_token("$element")
        assert result.category == TokenCategory.IDENTIFIER

    def test_classify_camelcase(self):
        """Should classify camelCase identifier."""
        result = classify_typescript_token("myVariableName")
        assert result.category == TokenCategory.IDENTIFIER

    def test_classify_pascalcase(self):
        """Should classify PascalCase identifier."""
        result = classify_typescript_token("MyClassName")
        assert result.category == TokenCategory.IDENTIFIER


# ===========================================================================
# Completion Suggestion Tests
# ===========================================================================


class TestTypeScriptCompletions:
    """Tests for completion suggestions."""

    def test_completions_for_type_prefix(self):
        """Should suggest types for type-like prefix."""
        completions = get_typescript_completions("str", context="type")
        assert "string" in completions

    def test_completions_for_keyword_prefix(self):
        """Should suggest keywords for prefix."""
        completions = get_typescript_completions("con", context="value")
        assert "const" in completions

    def test_completions_for_builtin_prefix(self):
        """Should suggest builtins for prefix."""
        completions = get_typescript_completions("Arr", context="value")
        assert "Array" in completions

    def test_completions_empty_string(self):
        """Should return results for empty prefix."""
        completions = get_typescript_completions("")
        assert len(completions) > 0


# ===========================================================================
# Helper Function Tests
# ===========================================================================


class TestTypeScriptHelperFunctions:
    """Tests for helper functions."""

    def test_is_typescript_keyword(self):
        """Should identify keywords correctly."""
        assert is_typescript_keyword("const")
        assert is_typescript_keyword("function")
        assert is_typescript_keyword("interface")
        assert not is_typescript_keyword("myVariable")
        assert not is_typescript_keyword("Array")

    def test_is_typescript_builtin(self):
        """Should identify builtins correctly."""
        assert is_typescript_builtin("Array")
        assert is_typescript_builtin("Promise")
        assert is_typescript_builtin("console")
        assert not is_typescript_builtin("const")
        assert not is_typescript_builtin("myVariable")

    def test_is_typescript_operator(self):
        """Should identify operators correctly."""
        assert is_typescript_operator("+")
        assert is_typescript_operator("===")
        assert is_typescript_operator("=>")
        assert not is_typescript_operator("const")
        assert not is_typescript_operator("Array")

    def test_is_typescript_type_keyword(self):
        """Should identify type keywords correctly."""
        assert is_typescript_type_keyword("string")
        assert is_typescript_type_keyword("number")
        assert is_typescript_type_keyword("boolean")
        assert is_typescript_type_keyword("any")
        assert not is_typescript_type_keyword("const")


# ===========================================================================
# Constants Tests
# ===========================================================================


class TestTypeScriptConstants:
    """Tests for TypeScript constants."""

    def test_keywords_frozenset(self):
        """TYPESCRIPT_ALL_KEYWORDS should be a frozenset."""
        assert isinstance(TYPESCRIPT_ALL_KEYWORDS, frozenset)
        assert len(TYPESCRIPT_ALL_KEYWORDS) > 0

    def test_builtins_frozenset(self):
        """TYPESCRIPT_ALL_BUILTINS should be a frozenset."""
        assert isinstance(TYPESCRIPT_ALL_BUILTINS, frozenset)
        assert len(TYPESCRIPT_ALL_BUILTINS) > 0

    def test_operators_frozenset(self):
        """TYPESCRIPT_ALL_OPERATORS should be a frozenset."""
        assert isinstance(TYPESCRIPT_ALL_OPERATORS, frozenset)
        assert len(TYPESCRIPT_ALL_OPERATORS) > 0

    def test_delimiters_frozenset(self):
        """TYPESCRIPT_DELIMITERS should be a frozenset."""
        assert isinstance(TYPESCRIPT_DELIMITERS, frozenset)
        assert len(TYPESCRIPT_DELIMITERS) > 0

    def test_primitive_types_frozenset(self):
        """TYPESCRIPT_PRIMITIVE_TYPES should be a frozenset."""
        assert isinstance(TYPESCRIPT_PRIMITIVE_TYPES, frozenset)
        assert "string" in TYPESCRIPT_PRIMITIVE_TYPES
        assert "number" in TYPESCRIPT_PRIMITIVE_TYPES

    def test_utility_types_frozenset(self):
        """TYPESCRIPT_UTILITY_TYPES should be a frozenset."""
        assert isinstance(TYPESCRIPT_UTILITY_TYPES, frozenset)
        assert "Partial" in TYPESCRIPT_UTILITY_TYPES
        assert "Record" in TYPESCRIPT_UTILITY_TYPES


# ===========================================================================
# Integration with Main Classifier Tests
# ===========================================================================


class TestTypeScriptMainClassifierIntegration:
    """Tests for integration with main token classifier."""

    def test_classify_token_for_language_typescript(self):
        """Should dispatch to TypeScript classifier."""
        result = classify_token_for_language("const", "typescript")
        assert result[0] == TokenCategory.KEYWORD
        assert result[1] == "const"

    def test_classify_token_for_language_ts(self):
        """Should dispatch to TypeScript classifier for 'ts'."""
        result = classify_token_for_language("function", "ts")
        assert result[0] == TokenCategory.KEYWORD

    def test_classify_token_for_language_js(self):
        """Should dispatch to TypeScript classifier for 'js'."""
        result = classify_token_for_language("let", "js")
        assert result[0] == TokenCategory.KEYWORD
