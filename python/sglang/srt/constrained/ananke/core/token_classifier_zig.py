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
"""Zig-specific token classification.

This module provides token classification rules for the Zig programming
language, including keywords, builtins, operators, and type primitives.

Used by TokenClassifier when language="zig" for:
- Vocabulary classification during initialization
- Token mask computation during generation
- Keyword and builtin detection

References:
    - Zig Language Reference: https://ziglang.org/documentation/master/
    - Zig Grammar: https://ziglang.org/documentation/master/#Grammar
"""

from __future__ import annotations

import re
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

from core.token_classifier import (
    TokenCategory,
    TokenClassification,
)


# =============================================================================
# Zig Keywords
# =============================================================================

# Control flow keywords
ZIG_CONTROL_KEYWORDS: FrozenSet[str] = frozenset({
    "if",
    "else",
    "for",
    "while",
    "switch",
    "break",
    "continue",
    "return",
    "unreachable",
})

# Error handling keywords
ZIG_ERROR_KEYWORDS: FrozenSet[str] = frozenset({
    "try",
    "catch",
    "orelse",
    "error",
})

# Definition keywords
ZIG_DEFINITION_KEYWORDS: FrozenSet[str] = frozenset({
    "fn",
    "const",
    "var",
    "pub",
    "extern",
    "export",
    "inline",
    "noinline",
    "comptime",
    "test",
    "struct",
    "enum",
    "union",
    "opaque",
    "packed",
    "threadlocal",
    "linksection",
    "usingnamespace",
})

# Memory/lifetime keywords
ZIG_MEMORY_KEYWORDS: FrozenSet[str] = frozenset({
    "defer",
    "errdefer",
    "nosuspend",
    "noalias",
    "volatile",
    "allowzero",
    "align",
    "addrspace",
})

# Async keywords
ZIG_ASYNC_KEYWORDS: FrozenSet[str] = frozenset({
    "async",
    "await",
    "suspend",
    "resume",
    "anyframe",
})

# Type-related keywords
ZIG_TYPE_KEYWORDS: FrozenSet[str] = frozenset({
    "type",
    "anytype",
    "anyopaque",
    "anyerror",
})

# Operator keywords (words that act as operators)
ZIG_OPERATOR_KEYWORDS: FrozenSet[str] = frozenset({
    "and",
    "or",
})

# Assembly keywords
ZIG_ASM_KEYWORDS: FrozenSet[str] = frozenset({
    "asm",
    "callconv",
})

# All Zig keywords
ZIG_ALL_KEYWORDS: FrozenSet[str] = (
    ZIG_CONTROL_KEYWORDS |
    ZIG_ERROR_KEYWORDS |
    ZIG_DEFINITION_KEYWORDS |
    ZIG_MEMORY_KEYWORDS |
    ZIG_ASYNC_KEYWORDS |
    ZIG_TYPE_KEYWORDS |
    ZIG_OPERATOR_KEYWORDS |
    ZIG_ASM_KEYWORDS
)


# =============================================================================
# Zig Builtins (@functions)
# =============================================================================

ZIG_BUILTINS: FrozenSet[str] = frozenset({
    # Type introspection
    "@TypeOf",
    "@typeInfo",
    "@typeName",
    "@Type",
    "@This",
    "@hasDecl",
    "@hasField",
    "@field",

    # Size and alignment
    "@sizeOf",
    "@alignOf",
    "@bitSizeOf",
    "@offsetOf",

    # Casting
    "@as",
    "@intCast",
    "@floatCast",
    "@ptrCast",
    "@alignCast",
    "@constCast",
    "@volatileCast",
    "@enumFromInt",
    "@intFromEnum",
    "@errorCast",
    "@errorFromInt",
    "@intFromError",
    "@intFromPtr",
    "@ptrFromInt",
    "@intFromBool",
    "@intFromFloat",
    "@floatFromInt",
    "@truncate",
    "@bitCast",
    "@addrSpaceCast",

    # Math operations
    "@addWithOverflow",
    "@subWithOverflow",
    "@mulWithOverflow",
    "@shlWithOverflow",
    "@shlExact",
    "@shrExact",
    "@divExact",
    "@divFloor",
    "@divTrunc",
    "@mod",
    "@rem",
    "@min",
    "@max",
    "@clz",
    "@ctz",
    "@popCount",
    "@byteSwap",
    "@bitReverse",
    "@abs",
    "@sqrt",
    "@log",
    "@log2",
    "@log10",
    "@exp",
    "@exp2",
    "@floor",
    "@ceil",
    "@round",
    "@fabs",
    "@mulAdd",

    # Memory operations
    "@memcpy",
    "@memset",
    "@shuffle",
    "@select",
    "@splat",
    "@reduce",

    # Atomic operations
    "@atomicLoad",
    "@atomicStore",
    "@atomicRmw",
    "@cmpxchgStrong",
    "@cmpxchgWeak",
    "@fence",

    # Compilation control
    "@compileError",
    "@compileLog",
    "@setEvalBranchQuota",
    "@setFloatMode",
    "@setRuntimeSafety",
    "@setAlignStack",

    # Debug and runtime
    "@panic",
    "@breakpoint",
    "@returnAddress",
    "@frameAddress",
    "@src",
    "@errorReturnTrace",

    # Import and embedding
    "@import",
    "@embedFile",
    "@cImport",
    "@cInclude",
    "@cDefine",
    "@cUndef",

    # Vector/SIMD
    "@Vector",

    # Misc
    "@call",
    "@tagName",
    "@errorName",
    "@unionInit",
    "@prefetch",
    "@wasmMemorySize",
    "@wasmMemoryGrow",

    # Deprecated but still recognized
    "@bitOffsetOf",
    "@export",
    "@extern",
})


# =============================================================================
# Zig Primitive Types
# =============================================================================

ZIG_PRIMITIVE_TYPES: FrozenSet[str] = frozenset({
    # Signed integers
    "i8", "i16", "i32", "i64", "i128", "isize",

    # Unsigned integers
    "u8", "u16", "u32", "u64", "u128", "usize",

    # Floats
    "f16", "f32", "f64", "f80", "f128",

    # Special types
    "bool",
    "void",
    "noreturn",
    "type",
    "anytype",
    "anyopaque",
    "anyerror",
    "anyframe",
    "comptime_int",
    "comptime_float",

    # C interop types
    "c_char",
    "c_short",
    "c_ushort",
    "c_int",
    "c_uint",
    "c_long",
    "c_ulong",
    "c_longlong",
    "c_ulonglong",
    "c_longdouble",
})


# =============================================================================
# Zig Operators
# =============================================================================

ZIG_OPERATORS: FrozenSet[str] = frozenset({
    # Arithmetic
    "+", "-", "*", "/", "%",
    "+%", "-%", "*%",  # Wrapping operations
    "+|", "-|", "*|",  # Saturating operations

    # Comparison
    "==", "!=", "<", ">", "<=", ">=",

    # Bitwise
    "&", "|", "^", "~",
    "<<", ">>",
    "<<|",  # Saturating shift

    # Assignment
    "=",
    "+=", "-=", "*=", "/=", "%=",
    "+%=", "-%=", "*%=",
    "+|=", "-|=", "*|=",
    "&=", "|=", "^=",
    "<<=", ">>=",
    "<<|=",

    # Pointer
    ".*",  # Pointer dereference
    ".?",  # Optional unwrap
    ".@",  # Field access by name

    # Misc
    "++",  # Array concatenation (comptime)
    "**",  # Array multiplication (comptime)
    "||",  # Merging error sets
    "=>",  # Switch prong
    "->",  # Anonymous struct field type
    "..",  # Range (in for loops)
    "...",  # Sentinel terminator
    "?",   # Optional type
    "!",   # Error union type / unwrap
})


# =============================================================================
# Zig Delimiters
# =============================================================================

ZIG_DELIMITERS: FrozenSet[str] = frozenset({
    "(", ")",
    "[", "]",
    "{", "}",
    ",", ":",
    ";",
    ".",
    "@",  # Builtin prefix
    "\\",  # Multiline string continuation
})


# =============================================================================
# Zig Literals
# =============================================================================

ZIG_BOOL_LITERALS: FrozenSet[str] = frozenset({
    "true",
    "false",
})

ZIG_NULL_LITERALS: FrozenSet[str] = frozenset({
    "null",
    "undefined",
})


# =============================================================================
# Standard Library Common Names
# =============================================================================

ZIG_STD_COMMON: FrozenSet[str] = frozenset({
    # Common std imports
    "std",
    "mem",
    "fmt",
    "debug",
    "io",
    "fs",
    "os",
    "heap",
    "math",
    "json",
    "testing",

    # Common types
    "Allocator",
    "ArrayList",
    "HashMap",
    "AutoHashMap",
    "StringHashMap",
    "BoundedArray",

    # Common functions
    "print",
    "eql",
    "len",
    "init",
    "deinit",
    "append",
    "items",
    "get",
    "put",
    "remove",
})


# =============================================================================
# Token Classification Functions
# =============================================================================

def classify_zig_token(text: str) -> Tuple[TokenCategory, Optional[str], Optional[Any]]:
    """Classify a token for Zig.

    Args:
        text: The token text to classify

    Returns:
        Tuple of (category, keyword_name, literal_value)
    """
    stripped = text.strip()

    if not stripped:
        return (TokenCategory.WHITESPACE, None, None)

    # Comments
    if stripped.startswith("//"):
        return (TokenCategory.COMMENT, None, None)

    # String literals
    if is_zig_string(stripped):
        return (TokenCategory.STRING_LITERAL, None, parse_zig_string(stripped))

    # Character literals
    if is_zig_char(stripped):
        return (TokenCategory.STRING_LITERAL, None, parse_zig_char(stripped))

    # Boolean literals
    if stripped in ZIG_BOOL_LITERALS:
        return (TokenCategory.BOOL_LITERAL, None, stripped == "true")

    # Null/undefined literals
    if stripped in ZIG_NULL_LITERALS:
        return (TokenCategory.NONE_LITERAL, None, None)

    # Numeric literals
    num_result = classify_zig_numeric(stripped)
    if num_result is not None:
        return num_result

    # Builtins (@functions)
    if stripped.startswith("@"):
        if stripped in ZIG_BUILTINS:
            return (TokenCategory.BUILTIN, stripped, None)
        # Might be partial builtin or field access
        return (TokenCategory.BUILTIN, None, None)

    # Keywords
    if stripped in ZIG_ALL_KEYWORDS:
        return (TokenCategory.KEYWORD, stripped, None)

    # Primitive types
    if stripped in ZIG_PRIMITIVE_TYPES:
        return (TokenCategory.BUILTIN, None, None)

    # Operators
    if stripped in ZIG_OPERATORS:
        return (TokenCategory.OPERATOR, None, None)

    # Delimiters
    if stripped in ZIG_DELIMITERS:
        return (TokenCategory.DELIMITER, None, None)

    # Arbitrary-width integer types (u3, i17, etc.)
    if re.match(r"^[iu]\d+$", stripped):
        bits = int(stripped[1:])
        if 0 < bits <= 65535:
            return (TokenCategory.BUILTIN, None, None)

    # Identifier
    if is_zig_identifier(stripped):
        return (TokenCategory.IDENTIFIER, None, None)

    # Unknown
    return (TokenCategory.UNKNOWN, None, None)


def is_zig_string(text: str) -> bool:
    """Check if text is a Zig string literal."""
    # Regular strings
    if text.startswith('"'):
        return True
    # Multiline strings
    if text.startswith("\\\\"):
        return True
    return False


def parse_zig_string(text: str) -> Optional[str]:
    """Parse a Zig string literal's value."""
    if not text.startswith('"'):
        return None

    # Check for complete string
    if len(text) >= 2 and text.endswith('"'):
        # Simple parsing - would need full escape handling
        return text[1:-1]
    return None


def is_zig_char(text: str) -> bool:
    """Check if text is a Zig character literal."""
    return text.startswith("'") and len(text) >= 2


def parse_zig_char(text: str) -> Optional[str]:
    """Parse a Zig character literal's value."""
    if not text.startswith("'"):
        return None
    if len(text) >= 3 and text.endswith("'"):
        return text[1:-1]
    return None


def classify_zig_numeric(text: str) -> Optional[Tuple[TokenCategory, Optional[str], Optional[Any]]]:
    """Classify and parse a Zig numeric literal.

    Zig numeric formats:
    - Decimal: 123, 123_456
    - Hex: 0x1A, 0x1A_2B
    - Octal: 0o17, 0o1_7
    - Binary: 0b1010, 0b10_10
    - Float: 1.0, 1.0e10, 0x1.0p10
    """
    # Remove underscores for parsing
    clean = text.replace("_", "")

    # Hex float: 0x1.0p10
    if clean.startswith("0x") and ("." in clean or "p" in clean.lower()):
        try:
            value = float.fromhex(clean)
            return (TokenCategory.FLOAT_LITERAL, None, value)
        except ValueError:
            pass

    # Hex integer
    if clean.startswith("0x"):
        try:
            value = int(clean, 16)
            return (TokenCategory.INT_LITERAL, None, value)
        except ValueError:
            # Might be partial
            if re.match(r"^0x[0-9a-fA-F]*$", clean):
                return (TokenCategory.INT_LITERAL, None, None)
            return None

    # Octal integer
    if clean.startswith("0o"):
        try:
            value = int(clean, 8)
            return (TokenCategory.INT_LITERAL, None, value)
        except ValueError:
            if re.match(r"^0o[0-7]*$", clean):
                return (TokenCategory.INT_LITERAL, None, None)
            return None

    # Binary integer
    if clean.startswith("0b"):
        try:
            value = int(clean, 2)
            return (TokenCategory.INT_LITERAL, None, value)
        except ValueError:
            if re.match(r"^0b[01]*$", clean):
                return (TokenCategory.INT_LITERAL, None, None)
            return None

    # Decimal float
    if "." in clean or "e" in clean.lower():
        try:
            value = float(clean)
            return (TokenCategory.FLOAT_LITERAL, None, value)
        except ValueError:
            if re.match(r"^[0-9]*\.?[0-9]*([eE][+-]?[0-9]*)?$", clean):
                return (TokenCategory.FLOAT_LITERAL, None, None)
            return None

    # Decimal integer
    if clean.isdigit() or (clean and clean[0].isdigit()):
        try:
            value = int(clean)
            return (TokenCategory.INT_LITERAL, None, value)
        except ValueError:
            if re.match(r"^[0-9]+$", clean):
                return (TokenCategory.INT_LITERAL, None, None)
            return None

    return None


def is_zig_identifier(text: str) -> bool:
    """Check if text is a valid Zig identifier."""
    if not text:
        return False

    # Can't start with digit
    if text[0].isdigit():
        return False

    # Can be @"identifier" for reserved words
    if text.startswith('@"') and text.endswith('"'):
        return True

    # Standard identifier: alphanumeric + underscore
    return all(c.isalnum() or c == "_" for c in text)


def get_zig_keyword_category(keyword: str) -> str:
    """Get the category of a Zig keyword.

    Returns one of: 'control', 'error', 'definition', 'memory',
    'async', 'type', 'operator', 'asm'
    """
    if keyword in ZIG_CONTROL_KEYWORDS:
        return "control"
    if keyword in ZIG_ERROR_KEYWORDS:
        return "error"
    if keyword in ZIG_DEFINITION_KEYWORDS:
        return "definition"
    if keyword in ZIG_MEMORY_KEYWORDS:
        return "memory"
    if keyword in ZIG_ASYNC_KEYWORDS:
        return "async"
    if keyword in ZIG_TYPE_KEYWORDS:
        return "type"
    if keyword in ZIG_OPERATOR_KEYWORDS:
        return "operator"
    if keyword in ZIG_ASM_KEYWORDS:
        return "asm"
    return "unknown"


# =============================================================================
# Integration with main TokenClassifier
# =============================================================================

def get_zig_keywords() -> FrozenSet[str]:
    """Get all Zig keywords."""
    return ZIG_ALL_KEYWORDS


def get_zig_builtins() -> FrozenSet[str]:
    """Get all Zig builtins."""
    return ZIG_BUILTINS


def get_zig_operators() -> FrozenSet[str]:
    """Get all Zig operators."""
    return ZIG_OPERATORS


def get_zig_delimiters() -> FrozenSet[str]:
    """Get all Zig delimiters."""
    return ZIG_DELIMITERS


def get_zig_primitive_types() -> FrozenSet[str]:
    """Get all Zig primitive type names."""
    return ZIG_PRIMITIVE_TYPES
