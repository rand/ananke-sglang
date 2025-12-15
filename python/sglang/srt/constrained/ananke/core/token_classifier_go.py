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
"""Go-specific token classification.

This module provides token classification rules for the Go programming
language, including keywords, operators, and type primitives.

Used by TokenClassifier when language="go" for:
- Vocabulary classification during initialization
- Token mask computation during generation
- Keyword and builtin detection

References:
    - Go Language Specification: https://go.dev/ref/spec
    - Go Lexical Elements: https://go.dev/ref/spec#Lexical_elements
"""

from __future__ import annotations

import re
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

from core.token_classifier import (
    TokenCategory,
    TokenClassification,
)


# =============================================================================
# Go Keywords
# =============================================================================

# Control flow keywords
GO_CONTROL_KEYWORDS: FrozenSet[str] = frozenset({
    "if",
    "else",
    "for",
    "switch",
    "case",
    "default",
    "break",
    "continue",
    "goto",
    "return",
    "fallthrough",
    "select",
    "range",
})

# Definition keywords
GO_DEFINITION_KEYWORDS: FrozenSet[str] = frozenset({
    "func",
    "var",
    "const",
    "type",
    "struct",
    "interface",
    "package",
    "import",
    "map",
    "chan",
})

# Special keywords
GO_SPECIAL_KEYWORDS: FrozenSet[str] = frozenset({
    "defer",
    "go",
})

# All keywords combined
GO_ALL_KEYWORDS: FrozenSet[str] = (
    GO_CONTROL_KEYWORDS
    | GO_DEFINITION_KEYWORDS
    | GO_SPECIAL_KEYWORDS
)


# =============================================================================
# Go Builtin Identifiers
# =============================================================================

# Boolean and nil literals
GO_LITERAL_KEYWORDS: FrozenSet[str] = frozenset({
    "true",
    "false",
    "nil",
    "iota",
})

# Builtin types
GO_BUILTIN_TYPES: FrozenSet[str] = frozenset({
    # Boolean
    "bool",
    # String
    "string",
    # Signed integers
    "int",
    "int8",
    "int16",
    "int32",
    "int64",
    # Unsigned integers
    "uint",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "uintptr",
    # Aliases
    "byte",   # alias for uint8
    "rune",   # alias for int32
    # Floating point
    "float32",
    "float64",
    # Complex numbers
    "complex64",
    "complex128",
    # Special types
    "error",
    "any",         # alias for interface{}
    "comparable",  # type constraint
})

# Builtin functions
GO_BUILTIN_FUNCTIONS: FrozenSet[str] = frozenset({
    "append",
    "cap",
    "clear",
    "close",
    "complex",
    "copy",
    "delete",
    "imag",
    "len",
    "make",
    "max",
    "min",
    "new",
    "panic",
    "print",
    "println",
    "real",
    "recover",
})

# All builtins
GO_ALL_BUILTINS: FrozenSet[str] = (
    GO_LITERAL_KEYWORDS
    | GO_BUILTIN_TYPES
    | GO_BUILTIN_FUNCTIONS
)


# =============================================================================
# Go Operators and Delimiters
# =============================================================================

# Arithmetic operators
GO_ARITHMETIC_OPS: FrozenSet[str] = frozenset({
    "+", "-", "*", "/", "%",
})

# Bitwise operators
GO_BITWISE_OPS: FrozenSet[str] = frozenset({
    "&", "|", "^", "&^",  # &^ is bit clear
    "<<", ">>",
})

# Comparison operators
GO_COMPARISON_OPS: FrozenSet[str] = frozenset({
    "==", "!=",
    "<", ">", "<=", ">=",
})

# Logical operators
GO_LOGICAL_OPS: FrozenSet[str] = frozenset({
    "&&", "||", "!",
})

# Assignment operators
GO_ASSIGNMENT_OPS: FrozenSet[str] = frozenset({
    "=",
    ":=",  # short variable declaration
    "+=", "-=", "*=", "/=", "%=",
    "&=", "|=", "^=", "&^=",
    "<<=", ">>=",
    "++", "--",  # increment/decrement
})

# Special operators
GO_SPECIAL_OPS: FrozenSet[str] = frozenset({
    "<-",   # channel send/receive
    "...",  # variadic
    ".",    # selector
    "&",    # address of
    "*",    # pointer dereference (also multiplication)
})

# All operators
GO_ALL_OPERATORS: FrozenSet[str] = (
    GO_ARITHMETIC_OPS
    | GO_BITWISE_OPS
    | GO_COMPARISON_OPS
    | GO_LOGICAL_OPS
    | GO_ASSIGNMENT_OPS
    | GO_SPECIAL_OPS
)

# Delimiters
GO_DELIMITERS: FrozenSet[str] = frozenset({
    "(", ")",
    "[", "]",
    "{", "}",
    ",", ";", ":",
})


# =============================================================================
# Go Literal Patterns
# =============================================================================

# Integer literal patterns
GO_INT_DECIMAL_PATTERN = re.compile(r"^[0-9][0-9_]*$")
GO_INT_BINARY_PATTERN = re.compile(r"^0[bB][01_]+$")
GO_INT_OCTAL_PATTERN = re.compile(r"^0[oO]?[0-7_]+$")
GO_INT_HEX_PATTERN = re.compile(r"^0[xX][0-9a-fA-F_]+$")

# Float literal patterns
GO_FLOAT_PATTERN = re.compile(
    r"^[0-9][0-9_]*\.[0-9_]*([eE][+-]?[0-9_]+)?$|"
    r"^[0-9][0-9_]*[eE][+-]?[0-9_]+$|"
    r"^\.[0-9_]+([eE][+-]?[0-9_]+)?$|"
    r"^0[xX][0-9a-fA-F_]*\.[0-9a-fA-F_]*[pP][+-]?[0-9_]+$"
)

# Imaginary literal pattern
GO_IMAGINARY_PATTERN = re.compile(r"^[0-9][0-9_.eE+-]*i$")

# Rune (character) literal pattern
GO_RUNE_PATTERN = re.compile(r"^'(?:[^'\\]|\\[abfnrtv\\'\"]|\\x[0-9a-fA-F]{2}|\\u[0-9a-fA-F]{4}|\\U[0-9a-fA-F]{8}|\\[0-7]{3})'$")

# String literal patterns
GO_STRING_INTERPRETED_PATTERN = re.compile(r'^"(?:[^"\\]|\\.)*"$')
GO_STRING_RAW_PATTERN = re.compile(r"^`[^`]*`$")

# Identifier pattern
GO_IDENTIFIER_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


# =============================================================================
# Classification Functions
# =============================================================================

def is_go_keyword(text: str) -> bool:
    """Check if text is a Go keyword."""
    return text in GO_ALL_KEYWORDS


def is_go_builtin(text: str) -> bool:
    """Check if text is a Go builtin identifier."""
    return text in GO_ALL_BUILTINS


def is_go_builtin_type(text: str) -> bool:
    """Check if text is a Go builtin type."""
    return text in GO_BUILTIN_TYPES


def is_go_builtin_function(text: str) -> bool:
    """Check if text is a Go builtin function."""
    return text in GO_BUILTIN_FUNCTIONS


def is_go_operator(text: str) -> bool:
    """Check if text is a Go operator."""
    return text in GO_ALL_OPERATORS


def is_go_delimiter(text: str) -> bool:
    """Check if text is a Go delimiter."""
    return text in GO_DELIMITERS


def is_go_int_literal(text: str) -> bool:
    """Check if text is a Go integer literal."""
    return bool(
        GO_INT_DECIMAL_PATTERN.match(text)
        or GO_INT_BINARY_PATTERN.match(text)
        or GO_INT_OCTAL_PATTERN.match(text)
        or GO_INT_HEX_PATTERN.match(text)
    )


def is_go_float_literal(text: str) -> bool:
    """Check if text is a Go floating-point literal."""
    return bool(GO_FLOAT_PATTERN.match(text))


def is_go_imaginary_literal(text: str) -> bool:
    """Check if text is a Go imaginary literal."""
    return bool(GO_IMAGINARY_PATTERN.match(text))


def is_go_rune_literal(text: str) -> bool:
    """Check if text is a Go rune (character) literal."""
    return bool(GO_RUNE_PATTERN.match(text))


def is_go_string_literal(text: str) -> bool:
    """Check if text is a Go string literal."""
    return bool(
        GO_STRING_INTERPRETED_PATTERN.match(text)
        or GO_STRING_RAW_PATTERN.match(text)
    )


def is_go_bool_literal(text: str) -> bool:
    """Check if text is a Go boolean literal."""
    return text in ("true", "false")


def is_go_nil_literal(text: str) -> bool:
    """Check if text is Go nil."""
    return text == "nil"


def is_go_identifier(text: str) -> bool:
    """Check if text is a valid Go identifier."""
    if text in GO_ALL_KEYWORDS or text in GO_ALL_BUILTINS:
        return False
    return bool(GO_IDENTIFIER_PATTERN.match(text))


def parse_go_int_literal(text: str) -> Optional[int]:
    """Parse a Go integer literal to its value."""
    # Remove underscores
    clean = text.replace("_", "")

    try:
        if clean.startswith(("0b", "0B")):
            return int(clean[2:], 2)
        elif clean.startswith(("0o", "0O")):
            return int(clean[2:], 8)
        elif clean.startswith(("0x", "0X")):
            return int(clean[2:], 16)
        elif clean.startswith("0") and len(clean) > 1 and clean[1].isdigit():
            # Old-style octal
            return int(clean, 8)
        else:
            return int(clean)
    except ValueError:
        return None


def parse_go_float_literal(text: str) -> Optional[float]:
    """Parse a Go float literal to its value."""
    # Remove underscores and trailing i for imaginary
    clean = text.replace("_", "").rstrip("i")

    try:
        # Handle hex floats
        if clean.startswith(("0x", "0X")):
            return float.fromhex(clean)
        return float(clean)
    except ValueError:
        return None


def classify_go_token(text: str) -> TokenClassification:
    """Classify a single Go token.

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

    # Check for keywords
    if stripped in GO_ALL_KEYWORDS:
        return TokenClassification(
            token_id=-1,
            text=text,
            category=TokenCategory.KEYWORD,
            keyword_name=stripped,
            is_complete=True,
        )

    # Check for builtin types
    if stripped in GO_BUILTIN_TYPES:
        return TokenClassification(
            token_id=-1,
            text=text,
            category=TokenCategory.BUILTIN,
            keyword_name=stripped,
            is_complete=True,
        )

    # Check for builtin functions
    if stripped in GO_BUILTIN_FUNCTIONS:
        return TokenClassification(
            token_id=-1,
            text=text,
            category=TokenCategory.BUILTIN,
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

    # Check for nil
    if stripped == "nil":
        return TokenClassification(
            token_id=-1,
            text=text,
            category=TokenCategory.NONE_LITERAL,
            literal_value=None,
            is_complete=True,
        )

    # Check for iota
    if stripped == "iota":
        return TokenClassification(
            token_id=-1,
            text=text,
            category=TokenCategory.BUILTIN,
            keyword_name="iota",
            is_complete=True,
        )

    # Check for imaginary literals (before int/float)
    if is_go_imaginary_literal(stripped):
        value = parse_go_float_literal(stripped)
        return TokenClassification(
            token_id=-1,
            text=text,
            category=TokenCategory.FLOAT_LITERAL,  # Treat as float
            literal_value=complex(0, value) if value is not None else None,
            is_complete=True,
        )

    # Check for integer literals
    if is_go_int_literal(stripped):
        return TokenClassification(
            token_id=-1,
            text=text,
            category=TokenCategory.INT_LITERAL,
            literal_value=parse_go_int_literal(stripped),
            is_complete=True,
        )

    # Check for float literals
    if is_go_float_literal(stripped):
        return TokenClassification(
            token_id=-1,
            text=text,
            category=TokenCategory.FLOAT_LITERAL,
            literal_value=parse_go_float_literal(stripped),
            is_complete=True,
        )

    # Check for rune literals
    if is_go_rune_literal(stripped):
        return TokenClassification(
            token_id=-1,
            text=text,
            category=TokenCategory.STRING_LITERAL,  # Treat runes as strings
            literal_value=stripped[1:-1],  # Remove quotes
            is_complete=True,
        )

    # Check for string literals
    if is_go_string_literal(stripped):
        return TokenClassification(
            token_id=-1,
            text=text,
            category=TokenCategory.STRING_LITERAL,
            literal_value=stripped[1:-1],  # Remove quotes
            is_complete=True,
        )

    # Check for operators
    if stripped in GO_ALL_OPERATORS:
        return TokenClassification(
            token_id=-1,
            text=text,
            category=TokenCategory.OPERATOR,
            is_complete=True,
        )

    # Check for delimiters
    if stripped in GO_DELIMITERS:
        return TokenClassification(
            token_id=-1,
            text=text,
            category=TokenCategory.DELIMITER,
            is_complete=True,
        )

    # Check for identifiers
    if GO_IDENTIFIER_PATTERN.match(stripped):
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


def get_go_keywords() -> FrozenSet[str]:
    """Get all Go keywords."""
    return GO_ALL_KEYWORDS


def get_go_builtins() -> FrozenSet[str]:
    """Get all Go builtin identifiers."""
    return GO_ALL_BUILTINS


def get_go_operators() -> FrozenSet[str]:
    """Get all Go operators."""
    return GO_ALL_OPERATORS


def get_go_delimiters() -> FrozenSet[str]:
    """Get all Go delimiters."""
    return GO_DELIMITERS
