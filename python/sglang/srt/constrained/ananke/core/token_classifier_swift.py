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
"""Swift-specific token classification.

This module provides token classification rules for the Swift programming
language, including keywords, operators, and type primitives.

Used by TokenClassifier when language="swift" for:
- Vocabulary classification during initialization
- Token mask computation during generation
- Keyword and builtin detection

References:
    - Swift Language Guide: https://docs.swift.org/swift-book/
    - Swift Grammar: https://docs.swift.org/swift-book/ReferenceManual/
"""

from __future__ import annotations

import re
from typing import Any, FrozenSet, Optional

from core.token_classifier import (
    TokenCategory,
    TokenClassification,
)


# =============================================================================
# Swift Keywords
# =============================================================================

# Declaration keywords
SWIFT_DECLARATION_KEYWORDS: FrozenSet[str] = frozenset({
    "associatedtype", "class", "deinit", "enum", "extension",
    "fileprivate", "func", "import", "init", "inout", "internal",
    "let", "open", "operator", "private", "precedencegroup", "protocol",
    "public", "rethrows", "static", "struct", "subscript", "typealias", "var",
})

# Statement keywords
SWIFT_STATEMENT_KEYWORDS: FrozenSet[str] = frozenset({
    "break", "case", "catch", "continue", "default", "defer", "do",
    "else", "fallthrough", "for", "guard", "if", "in", "repeat",
    "return", "switch", "throw", "throws", "try", "where", "while",
})

# Expression and type keywords
SWIFT_EXPRESSION_KEYWORDS: FrozenSet[str] = frozenset({
    "as", "Any", "catch", "false", "is", "nil", "rethrows",
    "self", "Self", "super", "throw", "throws", "true", "try",
})

# Context-specific keywords
SWIFT_CONTEXT_KEYWORDS: FrozenSet[str] = frozenset({
    "associativity", "convenience", "didSet", "dynamic", "final",
    "get", "indirect", "infix", "lazy", "left", "mutating",
    "none", "nonmutating", "optional", "override", "postfix", "prefix",
    "Protocol", "required", "right", "set", "some", "Type",
    "unowned", "weak", "willSet",
})

# Property wrapper and attribute keywords
SWIFT_ATTRIBUTE_KEYWORDS: FrozenSet[str] = frozenset({
    "available", "discardableResult", "dynamicCallable", "dynamicMemberLookup",
    "escaping", "frozen", "GKInspectable", "IBAction", "IBDesignable",
    "IBInspectable", "IBOutlet", "IBSegueAction", "inlinable", "main",
    "nonobjc", "NSApplicationMain", "NSCopying", "NSManaged", "objc",
    "objcMembers", "propertyWrapper", "resultBuilder", "testable",
    "UIApplicationMain", "unknown", "usableFromInline", "warn_unqualified_access",
})

# All keywords combined
SWIFT_ALL_KEYWORDS: FrozenSet[str] = (
    SWIFT_DECLARATION_KEYWORDS
    | SWIFT_STATEMENT_KEYWORDS
    | SWIFT_EXPRESSION_KEYWORDS
    | SWIFT_CONTEXT_KEYWORDS
)


# =============================================================================
# Swift Builtin Types and Functions
# =============================================================================

# Primitive types
SWIFT_PRIMITIVE_TYPES: FrozenSet[str] = frozenset({
    # Integer types
    "Int", "Int8", "Int16", "Int32", "Int64",
    "UInt", "UInt8", "UInt16", "UInt32", "UInt64",
    # Floating point
    "Float", "Double", "Float16", "Float80",
    # Boolean
    "Bool",
    # Character and String
    "Character", "String",
    # Special types
    "Void", "Never",
})

# Common types
SWIFT_COMMON_TYPES: FrozenSet[str] = frozenset({
    "Array", "Dictionary", "Set", "Optional",
    "Result", "Range", "ClosedRange", "PartialRangeFrom", "PartialRangeUpTo",
    "AnyObject", "AnyClass", "AnyHashable",
    "Sequence", "Collection", "RandomAccessCollection",
    "IteratorProtocol", "Comparable", "Equatable", "Hashable",
    "Codable", "Encodable", "Decodable",
    "Error", "LocalizedError",
    "ObservableObject", "Published",
    "Data", "Date", "URL", "UUID",
    "Binding", "State", "Environment", "EnvironmentObject",
    "View", "Text", "Image", "Button", "List", "NavigationView",
})

# All builtin types
SWIFT_BUILTIN_TYPES: FrozenSet[str] = (
    SWIFT_PRIMITIVE_TYPES
    | SWIFT_COMMON_TYPES
)

# Builtin functions
SWIFT_BUILTIN_FUNCTIONS: FrozenSet[str] = frozenset({
    "print", "debugPrint", "dump", "readLine",
    "abs", "min", "max", "stride",
    "zip", "sequence", "repeatElement",
    "type", "sizeof", "alignof", "strideof",
    "assert", "assertionFailure", "precondition", "preconditionFailure",
    "fatalError",
    "withUnsafePointer", "withUnsafeMutablePointer",
    "withUnsafeBytes", "withUnsafeMutableBytes",
})

# All builtins
SWIFT_ALL_BUILTINS: FrozenSet[str] = (
    SWIFT_BUILTIN_TYPES
    | SWIFT_BUILTIN_FUNCTIONS
)


# =============================================================================
# Swift Operators and Delimiters
# =============================================================================

# Arithmetic operators
SWIFT_ARITHMETIC_OPS: FrozenSet[str] = frozenset({
    "+", "-", "*", "/", "%",
    "&+", "&-", "&*",  # Overflow operators
})

# Assignment operators
SWIFT_ASSIGNMENT_OPS: FrozenSet[str] = frozenset({
    "=",
    "+=", "-=", "*=", "/=", "%=",
    "&=", "|=", "^=",
    "<<=", ">>=",
})

# Comparison operators
SWIFT_COMPARISON_OPS: FrozenSet[str] = frozenset({
    "==", "!=",
    "===", "!==",  # Identity comparison
    "<", ">", "<=", ">=",
    "~=",  # Pattern matching
})

# Logical operators
SWIFT_LOGICAL_OPS: FrozenSet[str] = frozenset({
    "&&", "||", "!",
})

# Bitwise operators
SWIFT_BITWISE_OPS: FrozenSet[str] = frozenset({
    "&", "|", "^", "~",
    "<<", ">>",
})

# Range operators
SWIFT_RANGE_OPS: FrozenSet[str] = frozenset({
    "...",   # Closed range
    "..<",   # Half-open range
})

# Optional operators
SWIFT_OPTIONAL_OPS: FrozenSet[str] = frozenset({
    "?",     # Optional chaining
    "!",     # Force unwrap
    "??",    # Nil coalescing
})

# Other operators
SWIFT_OTHER_OPS: FrozenSet[str] = frozenset({
    "->",    # Function return type
    ".",     # Member access
    "?.",    # Optional chaining
    "@",     # Attribute prefix
    "#",     # Compiler directive
})

# All operators
SWIFT_ALL_OPERATORS: FrozenSet[str] = (
    SWIFT_ARITHMETIC_OPS
    | SWIFT_ASSIGNMENT_OPS
    | SWIFT_COMPARISON_OPS
    | SWIFT_LOGICAL_OPS
    | SWIFT_BITWISE_OPS
    | SWIFT_RANGE_OPS
    | SWIFT_OPTIONAL_OPS
    | SWIFT_OTHER_OPS
)

# Delimiters
SWIFT_DELIMITERS: FrozenSet[str] = frozenset({
    "(", ")",
    "[", "]",
    "{", "}",
    "<", ">",
    ",", ";", ":",
})


# =============================================================================
# Swift Literal Patterns
# =============================================================================

# Integer literal patterns
SWIFT_INT_DECIMAL_PATTERN = re.compile(r"^[0-9][0-9_]*$")
SWIFT_INT_BINARY_PATTERN = re.compile(r"^0b[01_]+$")
SWIFT_INT_OCTAL_PATTERN = re.compile(r"^0o[0-7_]+$")
SWIFT_INT_HEX_PATTERN = re.compile(r"^0x[0-9a-fA-F_]+$")

# Float literal patterns
SWIFT_FLOAT_PATTERN = re.compile(
    r"^[0-9][0-9_]*\.[0-9_]*([eE][+-]?[0-9_]+)?$|"
    r"^[0-9][0-9_]*[eE][+-]?[0-9_]+$|"
    r"^0x[0-9a-fA-F_]*\.[0-9a-fA-F_]*[pP][+-]?[0-9_]+$"
)

# String literal patterns
SWIFT_STRING_PATTERN = re.compile(r'^"(?:[^"\\]|\\.)*"$')
SWIFT_MULTILINE_STRING_PATTERN = re.compile(r'^"""[\s\S]*"""$')

# Identifier pattern
SWIFT_IDENTIFIER_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
SWIFT_BACKTICK_IDENTIFIER_PATTERN = re.compile(r"^`[^`]+`$")


# =============================================================================
# Classification Functions
# =============================================================================

def is_swift_keyword(text: str) -> bool:
    """Check if text is a Swift keyword."""
    return text in SWIFT_ALL_KEYWORDS


def is_swift_builtin(text: str) -> bool:
    """Check if text is a Swift builtin identifier."""
    return text in SWIFT_ALL_BUILTINS


def is_swift_builtin_type(text: str) -> bool:
    """Check if text is a Swift builtin type."""
    return text in SWIFT_BUILTIN_TYPES


def is_swift_builtin_function(text: str) -> bool:
    """Check if text is a Swift builtin function."""
    return text in SWIFT_BUILTIN_FUNCTIONS


def is_swift_operator(text: str) -> bool:
    """Check if text is a Swift operator."""
    return text in SWIFT_ALL_OPERATORS


def is_swift_delimiter(text: str) -> bool:
    """Check if text is a Swift delimiter."""
    return text in SWIFT_DELIMITERS


def is_swift_int_literal(text: str) -> bool:
    """Check if text is a Swift integer literal."""
    return bool(
        SWIFT_INT_DECIMAL_PATTERN.match(text)
        or SWIFT_INT_BINARY_PATTERN.match(text)
        or SWIFT_INT_OCTAL_PATTERN.match(text)
        or SWIFT_INT_HEX_PATTERN.match(text)
    )


def is_swift_float_literal(text: str) -> bool:
    """Check if text is a Swift floating-point literal."""
    return bool(SWIFT_FLOAT_PATTERN.match(text))


def is_swift_string_literal(text: str) -> bool:
    """Check if text is a Swift string literal."""
    return bool(
        SWIFT_STRING_PATTERN.match(text)
        or SWIFT_MULTILINE_STRING_PATTERN.match(text)
    )


def is_swift_bool_literal(text: str) -> bool:
    """Check if text is a Swift boolean literal."""
    return text in ("true", "false")


def is_swift_nil_literal(text: str) -> bool:
    """Check if text is Swift nil."""
    return text == "nil"


def is_swift_identifier(text: str) -> bool:
    """Check if text is a valid Swift identifier."""
    if text in SWIFT_ALL_KEYWORDS or text in SWIFT_ALL_BUILTINS:
        return False
    return bool(
        SWIFT_IDENTIFIER_PATTERN.match(text)
        or SWIFT_BACKTICK_IDENTIFIER_PATTERN.match(text)
    )


def parse_swift_int_literal(text: str) -> Optional[int]:
    """Parse a Swift integer literal to its value."""
    clean = text.replace("_", "")

    try:
        if clean.startswith("0b"):
            return int(clean[2:], 2)
        elif clean.startswith("0o"):
            return int(clean[2:], 8)
        elif clean.startswith("0x"):
            return int(clean[2:], 16)
        else:
            return int(clean)
    except ValueError:
        return None


def parse_swift_float_literal(text: str) -> Optional[float]:
    """Parse a Swift float literal to its value."""
    clean = text.replace("_", "")

    try:
        if clean.startswith("0x"):
            return float.fromhex(clean)
        return float(clean)
    except ValueError:
        return None


def classify_swift_token(text: str) -> TokenClassification:
    """Classify a single Swift token.

    Args:
        text: The token text to classify

    Returns:
        TokenClassification with category and metadata
    """
    stripped = text.strip()

    if not stripped:
        return TokenClassification(
            token_id=-1,
            text=text,
            category=TokenCategory.WHITESPACE,
            is_complete=True,
        )

    # Check for keywords
    if stripped in SWIFT_ALL_KEYWORDS:
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

    # Check for nil
    if stripped == "nil":
        return TokenClassification(
            token_id=-1,
            text=text,
            category=TokenCategory.NONE_LITERAL,
            literal_value=None,
            is_complete=True,
        )

    # Check for builtin types
    if stripped in SWIFT_BUILTIN_TYPES:
        return TokenClassification(
            token_id=-1,
            text=text,
            category=TokenCategory.BUILTIN,
            keyword_name=stripped,
            is_complete=True,
        )

    # Check for builtin functions
    if stripped in SWIFT_BUILTIN_FUNCTIONS:
        return TokenClassification(
            token_id=-1,
            text=text,
            category=TokenCategory.BUILTIN,
            keyword_name=stripped,
            is_complete=True,
        )

    # Check for integer literals
    if is_swift_int_literal(stripped):
        return TokenClassification(
            token_id=-1,
            text=text,
            category=TokenCategory.INT_LITERAL,
            literal_value=parse_swift_int_literal(stripped),
            is_complete=True,
        )

    # Check for float literals
    if is_swift_float_literal(stripped):
        return TokenClassification(
            token_id=-1,
            text=text,
            category=TokenCategory.FLOAT_LITERAL,
            literal_value=parse_swift_float_literal(stripped),
            is_complete=True,
        )

    # Check for string literals
    if is_swift_string_literal(stripped):
        return TokenClassification(
            token_id=-1,
            text=text,
            category=TokenCategory.STRING_LITERAL,
            literal_value=stripped[1:-1] if stripped.startswith('"') else stripped[3:-3],
            is_complete=True,
        )

    # Check for operators
    if stripped in SWIFT_ALL_OPERATORS:
        return TokenClassification(
            token_id=-1,
            text=text,
            category=TokenCategory.OPERATOR,
            is_complete=True,
        )

    # Check for delimiters
    if stripped in SWIFT_DELIMITERS:
        return TokenClassification(
            token_id=-1,
            text=text,
            category=TokenCategory.DELIMITER,
            is_complete=True,
        )

    # Check for identifiers
    if is_swift_identifier(stripped):
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


def get_swift_keywords() -> FrozenSet[str]:
    """Get all Swift keywords."""
    return SWIFT_ALL_KEYWORDS


def get_swift_builtins() -> FrozenSet[str]:
    """Get all Swift builtin identifiers."""
    return SWIFT_ALL_BUILTINS


def get_swift_operators() -> FrozenSet[str]:
    """Get all Swift operators."""
    return SWIFT_ALL_OPERATORS


def get_swift_delimiters() -> FrozenSet[str]:
    """Get all Swift delimiters."""
    return SWIFT_DELIMITERS
