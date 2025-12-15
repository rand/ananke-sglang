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
"""Kotlin-specific token classification.

This module provides token classification rules for the Kotlin programming
language, including keywords, operators, and type primitives.

Used by TokenClassifier when language="kotlin" for:
- Vocabulary classification during initialization
- Token mask computation during generation
- Keyword and builtin detection

References:
    - Kotlin Language Specification: https://kotlinlang.org/spec/
    - Kotlin Grammar: https://kotlinlang.org/docs/reference/grammar.html
"""

from __future__ import annotations

import re
from typing import Any, FrozenSet, Optional

from core.token_classifier import (
    TokenCategory,
    TokenClassification,
)


# =============================================================================
# Kotlin Keywords
# =============================================================================

# Hard keywords (reserved in all contexts)
KOTLIN_HARD_KEYWORDS: FrozenSet[str] = frozenset({
    "as",
    "as?",
    "break",
    "class",
    "continue",
    "do",
    "else",
    "false",
    "for",
    "fun",
    "if",
    "in",
    "!in",
    "interface",
    "is",
    "!is",
    "null",
    "object",
    "package",
    "return",
    "super",
    "this",
    "throw",
    "true",
    "try",
    "typealias",
    "typeof",
    "val",
    "var",
    "when",
    "while",
})

# Soft keywords (reserved only in specific contexts)
KOTLIN_SOFT_KEYWORDS: FrozenSet[str] = frozenset({
    "by",
    "catch",
    "constructor",
    "delegate",
    "dynamic",
    "field",
    "file",
    "finally",
    "get",
    "import",
    "init",
    "param",
    "property",
    "receiver",
    "set",
    "setparam",
    "value",
    "where",
})

# Modifier keywords
KOTLIN_MODIFIER_KEYWORDS: FrozenSet[str] = frozenset({
    "abstract",
    "actual",
    "annotation",
    "companion",
    "const",
    "crossinline",
    "data",
    "enum",
    "expect",
    "external",
    "final",
    "infix",
    "inline",
    "inner",
    "internal",
    "lateinit",
    "noinline",
    "open",
    "operator",
    "out",
    "override",
    "private",
    "protected",
    "public",
    "reified",
    "sealed",
    "suspend",
    "tailrec",
    "vararg",
})

# Special identifiers
KOTLIN_SPECIAL_IDENTIFIERS: FrozenSet[str] = frozenset({
    "it",       # implicit lambda parameter
    "field",    # backing field
})

# All keywords combined
KOTLIN_ALL_KEYWORDS: FrozenSet[str] = (
    KOTLIN_HARD_KEYWORDS
    | KOTLIN_SOFT_KEYWORDS
    | KOTLIN_MODIFIER_KEYWORDS
)


# =============================================================================
# Kotlin Builtin Types and Functions
# =============================================================================

# Primitive types
KOTLIN_PRIMITIVE_TYPES: FrozenSet[str] = frozenset({
    # Numeric types
    "Byte",
    "Short",
    "Int",
    "Long",
    "Float",
    "Double",
    # Boolean
    "Boolean",
    # Character
    "Char",
    # Unit (void-like)
    "Unit",
    # Nothing (bottom type)
    "Nothing",
})

# Common types
KOTLIN_COMMON_TYPES: FrozenSet[str] = frozenset({
    "String",
    "Any",
    "Array",
    "List",
    "MutableList",
    "Set",
    "MutableSet",
    "Map",
    "MutableMap",
    "Sequence",
    "Pair",
    "Triple",
    "IntArray",
    "LongArray",
    "FloatArray",
    "DoubleArray",
    "BooleanArray",
    "CharArray",
    "ByteArray",
    "ShortArray",
    "UInt",
    "ULong",
    "UByte",
    "UShort",
    "IntRange",
    "LongRange",
    "CharRange",
    "UIntRange",
    "ULongRange",
})

# All builtin types
KOTLIN_BUILTIN_TYPES: FrozenSet[str] = (
    KOTLIN_PRIMITIVE_TYPES
    | KOTLIN_COMMON_TYPES
)

# Builtin functions
KOTLIN_BUILTIN_FUNCTIONS: FrozenSet[str] = frozenset({
    # Standard library functions
    "println",
    "print",
    "readLine",
    "readln",
    # Collection functions
    "listOf",
    "mutableListOf",
    "setOf",
    "mutableSetOf",
    "mapOf",
    "mutableMapOf",
    "arrayOf",
    "intArrayOf",
    "longArrayOf",
    "floatArrayOf",
    "doubleArrayOf",
    "booleanArrayOf",
    "charArrayOf",
    "byteArrayOf",
    "shortArrayOf",
    "emptyList",
    "emptySet",
    "emptyMap",
    "emptyArray",
    "sequenceOf",
    # Pair/Triple
    "to",
    # Range functions
    "until",
    "downTo",
    "step",
    # Type functions
    "require",
    "requireNotNull",
    "check",
    "checkNotNull",
    "error",
    "TODO",
    # Scope functions
    "let",
    "run",
    "with",
    "apply",
    "also",
    "takeIf",
    "takeUnless",
    # Lazy
    "lazy",
    "lazyOf",
    # Comparison
    "maxOf",
    "minOf",
    "compareValues",
    "compareValuesBy",
    # Other
    "repeat",
    "buildString",
    "buildList",
    "buildSet",
    "buildMap",
})

# Coroutine-related
KOTLIN_COROUTINE_FUNCTIONS: FrozenSet[str] = frozenset({
    "launch",
    "async",
    "runBlocking",
    "coroutineScope",
    "supervisorScope",
    "withContext",
    "delay",
    "yield",
    "flow",
    "flowOf",
    "channelFlow",
    "callbackFlow",
    "collect",
    "emit",
})

# All builtins
KOTLIN_ALL_BUILTINS: FrozenSet[str] = (
    KOTLIN_BUILTIN_TYPES
    | KOTLIN_BUILTIN_FUNCTIONS
    | KOTLIN_COROUTINE_FUNCTIONS
    | KOTLIN_SPECIAL_IDENTIFIERS
)


# =============================================================================
# Kotlin Operators and Delimiters
# =============================================================================

# Arithmetic operators
KOTLIN_ARITHMETIC_OPS: FrozenSet[str] = frozenset({
    "+", "-", "*", "/", "%",
})

# Assignment operators
KOTLIN_ASSIGNMENT_OPS: FrozenSet[str] = frozenset({
    "=",
    "+=", "-=", "*=", "/=", "%=",
})

# Increment/decrement
KOTLIN_INCREMENT_OPS: FrozenSet[str] = frozenset({
    "++", "--",
})

# Comparison operators
KOTLIN_COMPARISON_OPS: FrozenSet[str] = frozenset({
    "==", "!=",
    "===", "!==",  # referential equality
    "<", ">", "<=", ">=",
})

# Logical operators
KOTLIN_LOGICAL_OPS: FrozenSet[str] = frozenset({
    "&&", "||", "!",
})

# Null safety operators
KOTLIN_NULL_SAFETY_OPS: FrozenSet[str] = frozenset({
    "?.",   # safe call
    "?:",   # elvis
    "!!",   # non-null assertion
})

# Range operators
KOTLIN_RANGE_OPS: FrozenSet[str] = frozenset({
    "..",       # range
    "..<",      # until (exclusive end)
})

# Special operators
KOTLIN_SPECIAL_OPS: FrozenSet[str] = frozenset({
    "->",    # lambda arrow
    "::",    # member reference
    "@",     # annotation
    ".",     # member access
    "?",     # nullable type
    "*",     # spread operator
    "in",    # containment (also keyword)
    "!in",   # not containment (also keyword)
    "is",    # type check (also keyword)
    "!is",   # negated type check (also keyword)
    "as",    # type cast (also keyword)
    "as?",   # safe cast (also keyword)
})

# All operators
KOTLIN_ALL_OPERATORS: FrozenSet[str] = (
    KOTLIN_ARITHMETIC_OPS
    | KOTLIN_ASSIGNMENT_OPS
    | KOTLIN_INCREMENT_OPS
    | KOTLIN_COMPARISON_OPS
    | KOTLIN_LOGICAL_OPS
    | KOTLIN_NULL_SAFETY_OPS
    | KOTLIN_RANGE_OPS
    | KOTLIN_SPECIAL_OPS
)

# Delimiters
KOTLIN_DELIMITERS: FrozenSet[str] = frozenset({
    "(", ")",
    "[", "]",
    "{", "}",
    "<", ">",  # also used for generics
    ",", ";", ":",
})


# =============================================================================
# Kotlin Literal Patterns
# =============================================================================

# Integer literal patterns
KOTLIN_INT_DECIMAL_PATTERN = re.compile(r"^[0-9][0-9_]*[uUlL]?$")
KOTLIN_INT_BINARY_PATTERN = re.compile(r"^0[bB][01_]+[uUlL]?$")
KOTLIN_INT_HEX_PATTERN = re.compile(r"^0[xX][0-9a-fA-F_]+[uUlL]?$")

# Float literal patterns
KOTLIN_FLOAT_PATTERN = re.compile(
    r"^[0-9][0-9_]*\.[0-9_]*([eE][+-]?[0-9_]+)?[fF]?$|"
    r"^[0-9][0-9_]*[eE][+-]?[0-9_]+[fF]?$|"
    r"^\.[0-9_]+([eE][+-]?[0-9_]+)?[fF]?$|"
    r"^[0-9][0-9_]*[fF]$"
)

# Character literal pattern
KOTLIN_CHAR_PATTERN = re.compile(r"^'(?:[^'\\]|\\[tbnr'\\\$]|\\u[0-9a-fA-F]{4})'$")

# String literal patterns
KOTLIN_STRING_PATTERN = re.compile(r'^"(?:[^"\\$]|\\.|\$[a-zA-Z_]|\$\{[^}]*\})*"$')
KOTLIN_RAW_STRING_PATTERN = re.compile(r'^"""[\s\S]*"""$')

# Identifier pattern
KOTLIN_IDENTIFIER_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
KOTLIN_BACKTICK_IDENTIFIER_PATTERN = re.compile(r"^`[^`]+`$")


# =============================================================================
# Classification Functions
# =============================================================================

def is_kotlin_keyword(text: str) -> bool:
    """Check if text is a Kotlin keyword."""
    return text in KOTLIN_ALL_KEYWORDS


def is_kotlin_builtin(text: str) -> bool:
    """Check if text is a Kotlin builtin identifier."""
    return text in KOTLIN_ALL_BUILTINS


def is_kotlin_builtin_type(text: str) -> bool:
    """Check if text is a Kotlin builtin type."""
    return text in KOTLIN_BUILTIN_TYPES


def is_kotlin_builtin_function(text: str) -> bool:
    """Check if text is a Kotlin builtin function."""
    return text in KOTLIN_BUILTIN_FUNCTIONS


def is_kotlin_operator(text: str) -> bool:
    """Check if text is a Kotlin operator."""
    return text in KOTLIN_ALL_OPERATORS


def is_kotlin_delimiter(text: str) -> bool:
    """Check if text is a Kotlin delimiter."""
    return text in KOTLIN_DELIMITERS


def is_kotlin_int_literal(text: str) -> bool:
    """Check if text is a Kotlin integer literal."""
    return bool(
        KOTLIN_INT_DECIMAL_PATTERN.match(text)
        or KOTLIN_INT_BINARY_PATTERN.match(text)
        or KOTLIN_INT_HEX_PATTERN.match(text)
    )


def is_kotlin_float_literal(text: str) -> bool:
    """Check if text is a Kotlin floating-point literal."""
    return bool(KOTLIN_FLOAT_PATTERN.match(text))


def is_kotlin_char_literal(text: str) -> bool:
    """Check if text is a Kotlin character literal."""
    return bool(KOTLIN_CHAR_PATTERN.match(text))


def is_kotlin_string_literal(text: str) -> bool:
    """Check if text is a Kotlin string literal."""
    return bool(
        KOTLIN_STRING_PATTERN.match(text)
        or KOTLIN_RAW_STRING_PATTERN.match(text)
    )


def is_kotlin_bool_literal(text: str) -> bool:
    """Check if text is a Kotlin boolean literal."""
    return text in ("true", "false")


def is_kotlin_null_literal(text: str) -> bool:
    """Check if text is Kotlin null."""
    return text == "null"


def is_kotlin_identifier(text: str) -> bool:
    """Check if text is a valid Kotlin identifier."""
    if text in KOTLIN_ALL_KEYWORDS or text in KOTLIN_ALL_BUILTINS:
        return False
    return bool(
        KOTLIN_IDENTIFIER_PATTERN.match(text)
        or KOTLIN_BACKTICK_IDENTIFIER_PATTERN.match(text)
    )


def parse_kotlin_int_literal(text: str) -> Optional[int]:
    """Parse a Kotlin integer literal to its value."""
    # Remove underscores and suffix
    clean = text.replace("_", "").rstrip("uUlL")

    try:
        if clean.startswith(("0b", "0B")):
            return int(clean[2:], 2)
        elif clean.startswith(("0x", "0X")):
            return int(clean[2:], 16)
        else:
            return int(clean)
    except ValueError:
        return None


def parse_kotlin_float_literal(text: str) -> Optional[float]:
    """Parse a Kotlin float literal to its value."""
    # Remove underscores and suffix
    clean = text.replace("_", "").rstrip("fF")

    try:
        return float(clean)
    except ValueError:
        return None


def classify_kotlin_token(text: str) -> TokenClassification:
    """Classify a single Kotlin token.

    Args:
        text: The token text to classify

    Returns:
        TokenClassification with category and metadata
    """
    # Strip whitespace for classification
    stripped = text.strip()

    if not stripped:
        return TokenClassification(
            token_id=-1,
            text=text,
            category=TokenCategory.WHITESPACE,
            is_complete=True,
        )

    # Check for keywords (before builtins)
    if stripped in KOTLIN_ALL_KEYWORDS:
        return TokenClassification(
            token_id=-1,
            text=text,
            category=TokenCategory.KEYWORD,
            keyword_name=stripped,
            is_complete=True,
        )

    # Check for boolean literals
    if stripped in ("true", "false"):
        return TokenClassification(
            token_id=-1,
            text=text,
            category=TokenCategory.BOOL_LITERAL,
            literal_value=stripped == "true",
            is_complete=True,
        )

    # Check for null
    if stripped == "null":
        return TokenClassification(
            token_id=-1,
            text=text,
            category=TokenCategory.NONE_LITERAL,
            literal_value=None,
            is_complete=True,
        )

    # Check for builtin types
    if stripped in KOTLIN_BUILTIN_TYPES:
        return TokenClassification(
            token_id=-1,
            text=text,
            category=TokenCategory.BUILTIN,
            keyword_name=stripped,
            is_complete=True,
        )

    # Check for builtin functions
    if stripped in KOTLIN_BUILTIN_FUNCTIONS or stripped in KOTLIN_COROUTINE_FUNCTIONS:
        return TokenClassification(
            token_id=-1,
            text=text,
            category=TokenCategory.BUILTIN,
            keyword_name=stripped,
            is_complete=True,
        )

    # Check for integer literals
    if is_kotlin_int_literal(stripped):
        return TokenClassification(
            token_id=-1,
            text=text,
            category=TokenCategory.INT_LITERAL,
            literal_value=parse_kotlin_int_literal(stripped),
            is_complete=True,
        )

    # Check for float literals
    if is_kotlin_float_literal(stripped):
        return TokenClassification(
            token_id=-1,
            text=text,
            category=TokenCategory.FLOAT_LITERAL,
            literal_value=parse_kotlin_float_literal(stripped),
            is_complete=True,
        )

    # Check for character literals
    if is_kotlin_char_literal(stripped):
        return TokenClassification(
            token_id=-1,
            text=text,
            category=TokenCategory.STRING_LITERAL,
            literal_value=stripped[1:-1],  # Remove quotes
            is_complete=True,
        )

    # Check for string literals
    if is_kotlin_string_literal(stripped):
        return TokenClassification(
            token_id=-1,
            text=text,
            category=TokenCategory.STRING_LITERAL,
            literal_value=stripped[1:-1] if stripped.startswith('"') else stripped[3:-3],
            is_complete=True,
        )

    # Check for operators
    if stripped in KOTLIN_ALL_OPERATORS:
        return TokenClassification(
            token_id=-1,
            text=text,
            category=TokenCategory.OPERATOR,
            is_complete=True,
        )

    # Check for delimiters
    if stripped in KOTLIN_DELIMITERS:
        return TokenClassification(
            token_id=-1,
            text=text,
            category=TokenCategory.DELIMITER,
            is_complete=True,
        )

    # Check for identifiers
    if is_kotlin_identifier(stripped):
        return TokenClassification(
            token_id=-1,
            text=text,
            category=TokenCategory.IDENTIFIER,
            is_complete=True,
        )

    # Check for comments
    if stripped.startswith("//") or stripped.startswith("/*"):
        return TokenClassification(
            token_id=-1,
            text=text,
            category=TokenCategory.COMMENT,
            is_complete=stripped.startswith("//") or stripped.endswith("*/"),
        )

    # Unknown or mixed
    return TokenClassification(
        token_id=-1,
        text=text,
        category=TokenCategory.UNKNOWN,
        is_complete=False,
    )


def get_kotlin_keywords() -> FrozenSet[str]:
    """Get all Kotlin keywords."""
    return KOTLIN_ALL_KEYWORDS


def get_kotlin_builtins() -> FrozenSet[str]:
    """Get all Kotlin builtin identifiers."""
    return KOTLIN_ALL_BUILTINS


def get_kotlin_operators() -> FrozenSet[str]:
    """Get all Kotlin operators."""
    return KOTLIN_ALL_OPERATORS


def get_kotlin_delimiters() -> FrozenSet[str]:
    """Get all Kotlin delimiters."""
    return KOTLIN_DELIMITERS
