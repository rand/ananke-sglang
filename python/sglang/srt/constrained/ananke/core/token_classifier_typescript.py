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
"""TypeScript-specific token classification.

This module provides token classification rules for the TypeScript/JavaScript
programming language, including keywords, operators, type primitives, and
JSX support.

Used by TokenClassifier when language="typescript" for:
- Vocabulary classification during initialization
- Token mask computation during generation
- Keyword and builtin detection

References:
    - TypeScript Handbook: https://www.typescriptlang.org/docs/handbook/
    - ECMAScript Specification: https://tc39.es/ecma262/
"""

from __future__ import annotations

import re
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

from core.token_classifier import (
    TokenCategory,
    TokenClassification,
)


# =============================================================================
# TypeScript Keywords
# =============================================================================

# Declaration keywords
TYPESCRIPT_DECLARATION_KEYWORDS: FrozenSet[str] = frozenset({
    "const",
    "let",
    "var",
    "function",
    "class",
    "interface",
    "type",
    "enum",
    "namespace",
    "module",
    "declare",
    "abstract",
    "readonly",
})

# Control flow keywords
TYPESCRIPT_CONTROL_KEYWORDS: FrozenSet[str] = frozenset({
    "if",
    "else",
    "switch",
    "case",
    "default",
    "for",
    "while",
    "do",
    "break",
    "continue",
    "return",
    "throw",
    "try",
    "catch",
    "finally",
})

# Type-related keywords
TYPESCRIPT_TYPE_KEYWORDS: FrozenSet[str] = frozenset({
    "extends",
    "implements",
    "infer",
    "keyof",
    "typeof",
    "in",
    "is",
    "asserts",
    "as",
    "satisfies",
})

# Modifier keywords
TYPESCRIPT_MODIFIER_KEYWORDS: FrozenSet[str] = frozenset({
    "public",
    "private",
    "protected",
    "static",
    "override",
    "async",
    "await",
})

# Operator keywords
TYPESCRIPT_OPERATOR_KEYWORDS: FrozenSet[str] = frozenset({
    "new",
    "delete",
    "instanceof",
    "void",
    "yield",
})

# Import/export keywords
TYPESCRIPT_IMPORT_KEYWORDS: FrozenSet[str] = frozenset({
    "import",
    "export",
    "from",
    "require",
})

# Other keywords
TYPESCRIPT_OTHER_KEYWORDS: FrozenSet[str] = frozenset({
    "this",
    "super",
    "debugger",
    "with",
    "get",
    "set",
    "of",
})

# All TypeScript keywords
TYPESCRIPT_ALL_KEYWORDS: FrozenSet[str] = (
    TYPESCRIPT_DECLARATION_KEYWORDS |
    TYPESCRIPT_CONTROL_KEYWORDS |
    TYPESCRIPT_TYPE_KEYWORDS |
    TYPESCRIPT_MODIFIER_KEYWORDS |
    TYPESCRIPT_OPERATOR_KEYWORDS |
    TYPESCRIPT_IMPORT_KEYWORDS |
    TYPESCRIPT_OTHER_KEYWORDS
)


# =============================================================================
# TypeScript Type Keywords (primitive types)
# =============================================================================

TYPESCRIPT_PRIMITIVE_TYPES: FrozenSet[str] = frozenset({
    "string",
    "number",
    "boolean",
    "bigint",
    "symbol",
    "undefined",
    "null",
    "void",
    "object",
    "any",
    "unknown",
    "never",
})


# =============================================================================
# TypeScript Builtins
# =============================================================================

# Global objects
TYPESCRIPT_GLOBAL_OBJECTS: FrozenSet[str] = frozenset({
    "Array",
    "Object",
    "String",
    "Number",
    "Boolean",
    "Symbol",
    "BigInt",
    "Function",
    "Date",
    "RegExp",
    "Error",
    "Map",
    "Set",
    "WeakMap",
    "WeakSet",
    "Promise",
    "Proxy",
    "Reflect",
    "JSON",
    "Math",
    "console",
    "globalThis",
    "Intl",
})

# Error types
TYPESCRIPT_ERROR_TYPES: FrozenSet[str] = frozenset({
    "Error",
    "TypeError",
    "RangeError",
    "SyntaxError",
    "ReferenceError",
    "EvalError",
    "URIError",
    "AggregateError",
})

# Typed arrays
TYPESCRIPT_TYPED_ARRAYS: FrozenSet[str] = frozenset({
    "ArrayBuffer",
    "SharedArrayBuffer",
    "DataView",
    "Int8Array",
    "Uint8Array",
    "Uint8ClampedArray",
    "Int16Array",
    "Uint16Array",
    "Int32Array",
    "Uint32Array",
    "Float32Array",
    "Float64Array",
    "BigInt64Array",
    "BigUint64Array",
})

# TypeScript utility types
TYPESCRIPT_UTILITY_TYPES: FrozenSet[str] = frozenset({
    "Partial",
    "Required",
    "Readonly",
    "Record",
    "Pick",
    "Omit",
    "Exclude",
    "Extract",
    "NonNullable",
    "ReturnType",
    "Parameters",
    "ConstructorParameters",
    "InstanceType",
    "ThisType",
    "Awaited",
    "ThisParameterType",
    "OmitThisParameter",
    "Uppercase",
    "Lowercase",
    "Capitalize",
    "Uncapitalize",
})

# Global functions
TYPESCRIPT_GLOBAL_FUNCTIONS: FrozenSet[str] = frozenset({
    "parseInt",
    "parseFloat",
    "isNaN",
    "isFinite",
    "encodeURI",
    "decodeURI",
    "encodeURIComponent",
    "decodeURIComponent",
    "eval",
    "setTimeout",
    "setInterval",
    "clearTimeout",
    "clearInterval",
    "fetch",
    "atob",
    "btoa",
    "structuredClone",
    "queueMicrotask",
    "reportError",
})

# All builtins
TYPESCRIPT_ALL_BUILTINS: FrozenSet[str] = (
    TYPESCRIPT_GLOBAL_OBJECTS |
    TYPESCRIPT_ERROR_TYPES |
    TYPESCRIPT_TYPED_ARRAYS |
    TYPESCRIPT_UTILITY_TYPES |
    TYPESCRIPT_GLOBAL_FUNCTIONS
)


# =============================================================================
# TypeScript Operators
# =============================================================================

# Arithmetic operators
TYPESCRIPT_ARITHMETIC_OPS: FrozenSet[str] = frozenset({
    "+", "-", "*", "/", "%", "**",
})

# Assignment operators
TYPESCRIPT_ASSIGNMENT_OPS: FrozenSet[str] = frozenset({
    "=", "+=", "-=", "*=", "/=", "%=", "**=",
    "&=", "|=", "^=", "<<=", ">>=", ">>>=",
    "&&=", "||=", "??=",
})

# Comparison operators
TYPESCRIPT_COMPARISON_OPS: FrozenSet[str] = frozenset({
    "==", "!=", "===", "!==",
    "<", ">", "<=", ">=",
})

# Logical operators
TYPESCRIPT_LOGICAL_OPS: FrozenSet[str] = frozenset({
    "&&", "||", "??", "!",
})

# Bitwise operators
TYPESCRIPT_BITWISE_OPS: FrozenSet[str] = frozenset({
    "&", "|", "^", "~", "<<", ">>", ">>>",
})

# Type operators
TYPESCRIPT_TYPE_OPS: FrozenSet[str] = frozenset({
    "?", "=>", "...",
    # Note: ":" is classified as delimiter, not operator
})

# Member/optional chaining
TYPESCRIPT_MEMBER_OPS: FrozenSet[str] = frozenset({
    ".", "?.", "!.",
})

# All operators
TYPESCRIPT_ALL_OPERATORS: FrozenSet[str] = (
    TYPESCRIPT_ARITHMETIC_OPS |
    TYPESCRIPT_ASSIGNMENT_OPS |
    TYPESCRIPT_COMPARISON_OPS |
    TYPESCRIPT_LOGICAL_OPS |
    TYPESCRIPT_BITWISE_OPS |
    TYPESCRIPT_TYPE_OPS |
    TYPESCRIPT_MEMBER_OPS
)


# =============================================================================
# TypeScript Delimiters
# =============================================================================

TYPESCRIPT_DELIMITERS: FrozenSet[str] = frozenset({
    "(", ")", "[", "]", "{", "}",
    ",", ";", ".", "`", '"', "'",
    "<", ">",  # Also used for type parameters
    ":",  # Used in type annotations, object literals, ternary
})


# =============================================================================
# TypeScript Literals
# =============================================================================

TYPESCRIPT_BOOL_LITERALS: FrozenSet[str] = frozenset({
    "true",
    "false",
})

TYPESCRIPT_NULL_LITERALS: FrozenSet[str] = frozenset({
    "null",
    "undefined",
})


# =============================================================================
# Classification Functions
# =============================================================================

def classify_typescript_token(text: str) -> TokenClassification:
    """Classify a single TypeScript token.

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
    if stripped in TYPESCRIPT_ALL_KEYWORDS:
        return TokenClassification(
            token_id=-1,
            text=text,
            category=TokenCategory.KEYWORD,
            keyword_name=stripped,
            is_complete=True,
        )

    # Check for type keywords (primitives)
    if stripped in TYPESCRIPT_PRIMITIVE_TYPES:
        return TokenClassification(
            token_id=-1,
            text=text,
            category=TokenCategory.KEYWORD,
            keyword_name=stripped,
            is_complete=True,
        )

    # Check for boolean literals
    if stripped in TYPESCRIPT_BOOL_LITERALS:
        return TokenClassification(
            token_id=-1,
            text=text,
            category=TokenCategory.BOOL_LITERAL,
            literal_value=stripped == "true",
            is_complete=True,
        )

    # Check for null/undefined literals
    if stripped in TYPESCRIPT_NULL_LITERALS:
        return TokenClassification(
            token_id=-1,
            text=text,
            category=TokenCategory.NONE_LITERAL,
            literal_value=None,
            is_complete=True,
        )

    # Check for builtins
    if stripped in TYPESCRIPT_ALL_BUILTINS:
        return TokenClassification(
            token_id=-1,
            text=text,
            category=TokenCategory.BUILTIN,
            is_complete=True,
        )

    # Check for operators
    if stripped in TYPESCRIPT_ALL_OPERATORS:
        return TokenClassification(
            token_id=-1,
            text=text,
            category=TokenCategory.OPERATOR,
            is_complete=True,
        )

    # Check for delimiters
    if stripped in TYPESCRIPT_DELIMITERS:
        return TokenClassification(
            token_id=-1,
            text=text,
            category=TokenCategory.DELIMITER,
            is_complete=True,
        )

    # Check for numeric literals
    if _is_numeric_literal(stripped):
        if "." in stripped or "e" in stripped.lower():
            return TokenClassification(
                token_id=-1,
                text=text,
                category=TokenCategory.FLOAT_LITERAL,
                literal_value=float(stripped),
                is_complete=True,
            )
        return TokenClassification(
            token_id=-1,
            text=text,
            category=TokenCategory.INT_LITERAL,
            literal_value=_parse_int_literal(stripped),
            is_complete=True,
        )

    # Check for string literals
    if _is_string_literal(stripped):
        return TokenClassification(
            token_id=-1,
            text=text,
            category=TokenCategory.STRING_LITERAL,
            literal_value=_extract_string_value(stripped),
            is_complete=_is_complete_string(stripped),
        )

    # Check for template literals
    if stripped.startswith("`"):
        return TokenClassification(
            token_id=-1,
            text=text,
            category=TokenCategory.STRING_LITERAL,
            is_complete=stripped.endswith("`") and stripped.count("`") >= 2,
        )

    # Check for comment
    if stripped.startswith("//") or stripped.startswith("/*"):
        return TokenClassification(
            token_id=-1,
            text=text,
            category=TokenCategory.COMMENT,
            is_complete=stripped.startswith("//") or stripped.endswith("*/"),
        )

    # Check for valid identifier
    if _is_identifier(stripped):
        return TokenClassification(
            token_id=-1,
            text=text,
            category=TokenCategory.IDENTIFIER,
            is_complete=True,
        )

    return TokenClassification(
        token_id=-1,
        text=text,
        category=TokenCategory.UNKNOWN,
        is_complete=True,
    )


def _is_numeric_literal(s: str) -> bool:
    """Check if string is a numeric literal."""
    # Handle binary, octal, hex
    if s.startswith(("0b", "0B", "0o", "0O", "0x", "0X")):
        try:
            int(s, 0)
            return True
        except ValueError:
            return False

    # Handle BigInt suffix
    if s.endswith("n"):
        s = s[:-1]

    # Handle underscore separators
    s = s.replace("_", "")

    try:
        float(s)
        return True
    except ValueError:
        return False


def _parse_int_literal(s: str) -> int:
    """Parse an integer literal."""
    # Handle BigInt suffix
    if s.endswith("n"):
        s = s[:-1]

    # Handle underscore separators
    s = s.replace("_", "")

    # Use Python's int() with base 0 for auto-detection
    try:
        return int(s, 0)
    except ValueError:
        return int(float(s))


def _is_string_literal(s: str) -> bool:
    """Check if string starts like a string literal."""
    return s.startswith(("'", '"', "`"))


def _is_complete_string(s: str) -> bool:
    """Check if string literal is complete."""
    if len(s) < 2:
        return False

    quote = s[0]
    if quote not in ("'", '"', "`"):
        return False

    # Check for closing quote (not escaped)
    if s.endswith(quote):
        # Count trailing backslashes
        backslashes = 0
        for c in reversed(s[:-1]):
            if c == "\\":
                backslashes += 1
            else:
                break
        return backslashes % 2 == 0

    return False


def _extract_string_value(s: str) -> str:
    """Extract the value from a string literal."""
    if len(s) < 2:
        return ""

    quote = s[0]
    if _is_complete_string(s):
        return s[1:-1]
    return s[1:]


def _is_identifier(s: str) -> bool:
    """Check if string is a valid TypeScript identifier."""
    if not s:
        return False

    # First character must be letter, underscore, or $
    if not (s[0].isalpha() or s[0] in "_$"):
        return False

    # Rest can include digits
    for c in s[1:]:
        if not (c.isalnum() or c in "_$"):
            return False

    return True


# =============================================================================
# Token Context and Completions
# =============================================================================

def get_typescript_completions(prefix: str, context: Optional[str] = None) -> List[str]:
    """Get TypeScript completions for a prefix.

    Args:
        prefix: The prefix to complete
        context: Optional context hint ('type', 'expression', 'import', etc.)

    Returns:
        List of possible completions
    """
    completions: List[str] = []
    prefix_lower = prefix.lower()

    # Type context - suggest type keywords and utility types
    if context == "type":
        for t in TYPESCRIPT_PRIMITIVE_TYPES:
            if t.lower().startswith(prefix_lower):
                completions.append(t)
        for t in TYPESCRIPT_UTILITY_TYPES:
            if t.lower().startswith(prefix_lower):
                completions.append(t)
        for t in TYPESCRIPT_GLOBAL_OBJECTS:
            if t.lower().startswith(prefix_lower):
                completions.append(t)
        return completions

    # Import context
    if context == "import":
        for kw in ("import", "export", "from", "as", "default", "type"):
            if kw.startswith(prefix_lower):
                completions.append(kw)
        return completions

    # General expression context
    for kw in TYPESCRIPT_ALL_KEYWORDS:
        if kw.lower().startswith(prefix_lower):
            completions.append(kw)

    for builtin in TYPESCRIPT_ALL_BUILTINS:
        if builtin.lower().startswith(prefix_lower):
            completions.append(builtin)

    # Add common literals
    if "true".startswith(prefix_lower):
        completions.append("true")
    if "false".startswith(prefix_lower):
        completions.append("false")
    if "null".startswith(prefix_lower):
        completions.append("null")
    if "undefined".startswith(prefix_lower):
        completions.append("undefined")

    return completions


def is_typescript_keyword(text: str) -> bool:
    """Check if text is a TypeScript keyword."""
    return text in TYPESCRIPT_ALL_KEYWORDS or text in TYPESCRIPT_PRIMITIVE_TYPES


def is_typescript_builtin(text: str) -> bool:
    """Check if text is a TypeScript builtin."""
    return text in TYPESCRIPT_ALL_BUILTINS


def is_typescript_type_keyword(text: str) -> bool:
    """Check if text is a TypeScript type keyword."""
    return text in TYPESCRIPT_PRIMITIVE_TYPES or text in TYPESCRIPT_TYPE_KEYWORDS


def is_typescript_operator(text: str) -> bool:
    """Check if text is a TypeScript operator."""
    return text in TYPESCRIPT_ALL_OPERATORS


def get_keyword_category(text: str) -> Optional[str]:
    """Get the category of a TypeScript keyword."""
    if text in TYPESCRIPT_DECLARATION_KEYWORDS:
        return "declaration"
    if text in TYPESCRIPT_CONTROL_KEYWORDS:
        return "control"
    if text in TYPESCRIPT_TYPE_KEYWORDS:
        return "type"
    if text in TYPESCRIPT_MODIFIER_KEYWORDS:
        return "modifier"
    if text in TYPESCRIPT_OPERATOR_KEYWORDS:
        return "operator"
    if text in TYPESCRIPT_IMPORT_KEYWORDS:
        return "import"
    if text in TYPESCRIPT_OTHER_KEYWORDS:
        return "other"
    if text in TYPESCRIPT_PRIMITIVE_TYPES:
        return "primitive_type"
    return None
