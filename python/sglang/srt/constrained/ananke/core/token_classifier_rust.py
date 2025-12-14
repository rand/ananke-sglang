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
"""Rust-specific token classification.

This module provides token classification rules for the Rust programming
language, including keywords, macros, operators, and type primitives.

Used by TokenClassifier when language="rust" for:
- Vocabulary classification during initialization
- Token mask computation during generation
- Keyword and macro detection

References:
    - Rust Reference: https://doc.rust-lang.org/reference/
    - Rust Lexical Structure: https://doc.rust-lang.org/reference/lexical-structure.html
"""

from __future__ import annotations

import re
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

from core.token_classifier import (
    TokenCategory,
    TokenClassification,
)


# =============================================================================
# Rust Keywords
# =============================================================================

# Control flow keywords
RUST_CONTROL_KEYWORDS: FrozenSet[str] = frozenset({
    "if",
    "else",
    "match",
    "loop",
    "while",
    "for",
    "break",
    "continue",
    "return",
})

# Definition keywords
RUST_DEFINITION_KEYWORDS: FrozenSet[str] = frozenset({
    "fn",
    "let",
    "const",
    "static",
    "struct",
    "enum",
    "union",
    "trait",
    "impl",
    "type",
    "mod",
    "use",
    "pub",
    "crate",
    "extern",
    "macro_rules",
})

# Type-related keywords
RUST_TYPE_KEYWORDS: FrozenSet[str] = frozenset({
    "where",
    "dyn",
    "Self",
    "self",
    "super",
})

# Memory/ownership keywords
RUST_MEMORY_KEYWORDS: FrozenSet[str] = frozenset({
    "mut",
    "ref",
    "move",
    "box",
})

# Safety keywords
RUST_SAFETY_KEYWORDS: FrozenSet[str] = frozenset({
    "unsafe",
    "safe",
})

# Async keywords
RUST_ASYNC_KEYWORDS: FrozenSet[str] = frozenset({
    "async",
    "await",
})

# Operator keywords
RUST_OPERATOR_KEYWORDS: FrozenSet[str] = frozenset({
    "as",
    "in",
})

# Reserved keywords (for future use)
RUST_RESERVED_KEYWORDS: FrozenSet[str] = frozenset({
    "abstract",
    "become",
    "do",
    "final",
    "macro",
    "override",
    "priv",
    "try",
    "typeof",
    "unsized",
    "virtual",
    "yield",
})

# All Rust keywords
RUST_ALL_KEYWORDS: FrozenSet[str] = (
    RUST_CONTROL_KEYWORDS |
    RUST_DEFINITION_KEYWORDS |
    RUST_TYPE_KEYWORDS |
    RUST_MEMORY_KEYWORDS |
    RUST_SAFETY_KEYWORDS |
    RUST_ASYNC_KEYWORDS |
    RUST_OPERATOR_KEYWORDS |
    RUST_RESERVED_KEYWORDS
)


# =============================================================================
# Rust Macros (standard library)
# =============================================================================

RUST_STD_MACROS: FrozenSet[str] = frozenset({
    # Printing
    "print!",
    "println!",
    "eprint!",
    "eprintln!",
    "write!",
    "writeln!",
    "format!",
    "format_args!",

    # Assertions
    "assert!",
    "assert_eq!",
    "assert_ne!",
    "debug_assert!",
    "debug_assert_eq!",
    "debug_assert_ne!",

    # Panics
    "panic!",
    "todo!",
    "unimplemented!",
    "unreachable!",

    # Collections
    "vec!",
    "hashmap!",

    # Debugging
    "dbg!",

    # Configuration
    "cfg!",
    "env!",
    "option_env!",
    "concat!",
    "stringify!",
    "include!",
    "include_str!",
    "include_bytes!",

    # Module/file info
    "file!",
    "line!",
    "column!",
    "module_path!",

    # Compile-time
    "compile_error!",

    # Threading
    "thread_local!",

    # Testing
    "matches!",

    # Derive macros (common)
    "derive",
})

# Macro names without the ! suffix (for partial matching)
RUST_STD_MACRO_NAMES: FrozenSet[str] = frozenset({
    name.rstrip("!") for name in RUST_STD_MACROS if name.endswith("!")
})


# =============================================================================
# Rust Primitive Types
# =============================================================================

RUST_PRIMITIVE_TYPES: FrozenSet[str] = frozenset({
    # Signed integers
    "i8", "i16", "i32", "i64", "i128", "isize",

    # Unsigned integers
    "u8", "u16", "u32", "u64", "u128", "usize",

    # Floats
    "f32", "f64",

    # Boolean
    "bool",

    # Character
    "char",

    # String slice
    "str",

    # Unit type
    "()",

    # Never type
    "!",
})


# =============================================================================
# Rust Standard Library Types
# =============================================================================

RUST_STD_TYPES: FrozenSet[str] = frozenset({
    # Smart pointers
    "Box",
    "Rc",
    "Arc",
    "Cell",
    "RefCell",
    "Mutex",
    "RwLock",
    "Cow",

    # Option and Result
    "Option",
    "Some",
    "None",
    "Result",
    "Ok",
    "Err",

    # Collections
    "Vec",
    "VecDeque",
    "LinkedList",
    "HashMap",
    "HashSet",
    "BTreeMap",
    "BTreeSet",
    "BinaryHeap",

    # Strings
    "String",
    "CString",
    "OsString",
    "PathBuf",
    "Chars",

    # Iterators
    "Iterator",
    "IntoIterator",
    "Iter",
    "IterMut",

    # IO
    "Read",
    "Write",
    "Seek",
    "BufRead",
    "BufReader",
    "BufWriter",
    "File",
    "Stdin",
    "Stdout",
    "Stderr",

    # Common types
    "Range",
    "RangeInclusive",
    "Duration",
    "Instant",
    "SystemTime",
    "Path",
    "Pin",
    "PhantomData",
})


# =============================================================================
# Rust Common Traits
# =============================================================================

RUST_COMMON_TRAITS: FrozenSet[str] = frozenset({
    # Core traits
    "Clone",
    "Copy",
    "Debug",
    "Default",
    "Display",

    # Comparison
    "Eq",
    "PartialEq",
    "Ord",
    "PartialOrd",
    "Hash",

    # Conversion
    "From",
    "Into",
    "TryFrom",
    "TryInto",
    "AsRef",
    "AsMut",

    # Memory
    "Drop",
    "Sized",
    "Send",
    "Sync",
    "Unpin",

    # Operators
    "Add",
    "Sub",
    "Mul",
    "Div",
    "Rem",
    "Neg",
    "Not",
    "BitAnd",
    "BitOr",
    "BitXor",
    "Shl",
    "Shr",
    "Index",
    "IndexMut",
    "Deref",
    "DerefMut",

    # Functions
    "Fn",
    "FnMut",
    "FnOnce",

    # Iteration
    "Iterator",
    "IntoIterator",
    "ExactSizeIterator",
    "DoubleEndedIterator",
    "Extend",
    "FromIterator",

    # IO
    "Read",
    "Write",
    "Seek",
    "BufRead",

    # Async
    "Future",
    "Stream",
})


# =============================================================================
# Rust Operators
# =============================================================================

RUST_OPERATORS: FrozenSet[str] = frozenset({
    # Arithmetic
    "+", "-", "*", "/", "%",

    # Comparison
    "==", "!=", "<", ">", "<=", ">=",

    # Logical
    "&&", "||", "!",

    # Bitwise
    "&", "|", "^", "~",
    "<<", ">>",

    # Assignment
    "=",
    "+=", "-=", "*=", "/=", "%=",
    "&=", "|=", "^=",
    "<<=", ">>=",

    # Reference/Dereference
    "&",  # Borrow (context-dependent)
    "*",  # Dereference (context-dependent)

    # Range
    "..",
    "..=",

    # Path
    "::",

    # Field access
    ".",

    # Error propagation
    "?",

    # Pattern binding
    "@",

    # Fat arrow (closures, match arms)
    "=>",

    # Thin arrow (return type)
    "->",

    # Type ascription
    ":",
})


# =============================================================================
# Rust Delimiters
# =============================================================================

RUST_DELIMITERS: FrozenSet[str] = frozenset({
    "(", ")",
    "[", "]",
    "{", "}",
    "<", ">",  # Also used as operators
    ",",
    ";",
    "#",  # Attributes
    "$",  # Macro metavariables
    "'",  # Lifetimes
})


# =============================================================================
# Rust Literals
# =============================================================================

RUST_BOOL_LITERALS: FrozenSet[str] = frozenset({
    "true",
    "false",
})


# =============================================================================
# Rust Attributes
# =============================================================================

RUST_COMMON_ATTRIBUTES: FrozenSet[str] = frozenset({
    # Derive
    "derive",

    # Testing
    "test",
    "bench",
    "ignore",
    "should_panic",

    # Configuration
    "cfg",
    "cfg_attr",

    # Code generation
    "inline",
    "cold",
    "track_caller",

    # Documentation
    "doc",
    "deprecated",

    # Lints
    "allow",
    "warn",
    "deny",
    "forbid",

    # Linkage
    "link",
    "link_name",
    "no_mangle",
    "export_name",

    # Misc
    "must_use",
    "non_exhaustive",
    "repr",
    "path",
    "macro_use",
    "macro_export",
    "global_allocator",

    # Feature flags
    "feature",
})


# =============================================================================
# Rust Lifetimes
# =============================================================================

RUST_COMMON_LIFETIMES: FrozenSet[str] = frozenset({
    "'static",
    "'_",
    "'a",
    "'b",
    "'c",
})


# =============================================================================
# Token Classification Functions
# =============================================================================

def classify_rust_token(text: str) -> Tuple[TokenCategory, Optional[str], Optional[Any]]:
    """Classify a token for Rust.

    Args:
        text: The token text to classify

    Returns:
        Tuple of (category, keyword_name, literal_value)
    """
    stripped = text.strip()

    if not stripped:
        return (TokenCategory.WHITESPACE, None, None)

    # Comments
    if stripped.startswith("//") or stripped.startswith("/*"):
        return (TokenCategory.COMMENT, None, None)

    # Raw string literals
    if is_rust_raw_string(stripped):
        return (TokenCategory.STRING_LITERAL, None, parse_rust_raw_string(stripped))

    # Regular string literals
    if is_rust_string(stripped):
        return (TokenCategory.STRING_LITERAL, None, parse_rust_string(stripped))

    # Byte string literals
    if is_rust_byte_string(stripped):
        return (TokenCategory.STRING_LITERAL, None, parse_rust_byte_string(stripped))

    # Character literals
    if is_rust_char(stripped):
        return (TokenCategory.STRING_LITERAL, None, parse_rust_char(stripped))

    # Byte literals
    if is_rust_byte(stripped):
        return (TokenCategory.INT_LITERAL, None, parse_rust_byte(stripped))

    # Boolean literals
    if stripped in RUST_BOOL_LITERALS:
        return (TokenCategory.BOOL_LITERAL, None, stripped == "true")

    # Numeric literals
    num_result = classify_rust_numeric(stripped)
    if num_result is not None:
        return num_result

    # Macros (ending with !)
    if stripped.endswith("!") and len(stripped) > 1:
        macro_name = stripped[:-1]
        if is_rust_identifier(macro_name) or stripped in RUST_STD_MACROS:
            return (TokenCategory.BUILTIN, stripped, None)

    # Lifetimes
    if stripped.startswith("'") and len(stripped) > 1:
        return (TokenCategory.BUILTIN, stripped, None)

    # Keywords
    if stripped in RUST_ALL_KEYWORDS:
        return (TokenCategory.KEYWORD, stripped, None)

    # Primitive types
    if stripped in RUST_PRIMITIVE_TYPES:
        return (TokenCategory.BUILTIN, None, None)

    # Standard library types
    if stripped in RUST_STD_TYPES:
        return (TokenCategory.BUILTIN, None, None)

    # Common traits
    if stripped in RUST_COMMON_TRAITS:
        return (TokenCategory.BUILTIN, None, None)

    # Operators
    if stripped in RUST_OPERATORS:
        return (TokenCategory.OPERATOR, None, None)

    # Delimiters
    if stripped in RUST_DELIMITERS:
        return (TokenCategory.DELIMITER, None, None)

    # Attributes
    if stripped.startswith("#[") or stripped.startswith("#!["):
        return (TokenCategory.BUILTIN, None, None)

    # Identifier
    if is_rust_identifier(stripped):
        return (TokenCategory.IDENTIFIER, None, None)

    # Unknown
    return (TokenCategory.UNKNOWN, None, None)


def is_rust_string(text: str) -> bool:
    """Check if text is a Rust string literal."""
    return text.startswith('"') and not text.startswith('b"')


def parse_rust_string(text: str) -> Optional[str]:
    """Parse a Rust string literal's value."""
    if not text.startswith('"'):
        return None

    if len(text) >= 2 and text.endswith('"'):
        return text[1:-1]
    return None


def is_rust_raw_string(text: str) -> bool:
    """Check if text is a Rust raw string literal."""
    if text.startswith('r#'):
        return True
    if text.startswith('r"'):
        return True
    if text.startswith('br#') or text.startswith('br"'):
        return True
    return False


def parse_rust_raw_string(text: str) -> Optional[str]:
    """Parse a Rust raw string literal's value."""
    # Count # symbols
    if text.startswith('r'):
        idx = 1
    elif text.startswith('br'):
        idx = 2
    else:
        return None

    hash_count = 0
    while idx < len(text) and text[idx] == '#':
        hash_count += 1
        idx += 1

    if idx >= len(text) or text[idx] != '"':
        return None

    # Find closing delimiter
    end_delim = '"' + '#' * hash_count
    end_idx = text.find(end_delim, idx + 1)
    if end_idx != -1:
        return text[idx + 1:end_idx]
    return None


def is_rust_byte_string(text: str) -> bool:
    """Check if text is a Rust byte string literal."""
    return text.startswith('b"')


def parse_rust_byte_string(text: str) -> Optional[bytes]:
    """Parse a Rust byte string literal's value."""
    if not text.startswith('b"'):
        return None

    if len(text) >= 3 and text.endswith('"'):
        try:
            return text[2:-1].encode('utf-8')
        except UnicodeError:
            return None
    return None


def is_rust_char(text: str) -> bool:
    """Check if text is a Rust character literal."""
    return text.startswith("'") and not text.startswith("'static") and len(text) >= 3


def parse_rust_char(text: str) -> Optional[str]:
    """Parse a Rust character literal's value."""
    if not text.startswith("'"):
        return None
    if len(text) >= 3 and text.endswith("'"):
        return text[1:-1]
    return None


def is_rust_byte(text: str) -> bool:
    """Check if text is a Rust byte literal."""
    return text.startswith("b'")


def parse_rust_byte(text: str) -> Optional[int]:
    """Parse a Rust byte literal's value."""
    if not text.startswith("b'"):
        return None
    if len(text) >= 4 and text.endswith("'"):
        char = text[2:-1]
        if len(char) == 1:
            return ord(char)
    return None


def classify_rust_numeric(text: str) -> Optional[Tuple[TokenCategory, Optional[str], Optional[Any]]]:
    """Classify and parse a Rust numeric literal.

    Rust numeric formats:
    - Decimal: 123, 123_456, 123i32, 123usize
    - Hex: 0x1A, 0x1A_2B
    - Octal: 0o17, 0o1_7
    - Binary: 0b1010, 0b10_10
    - Float: 1.0, 1.0e10, 1.0f32
    """
    # Extract type suffix if present
    suffix_match = re.match(r'^(.+?)(i8|i16|i32|i64|i128|isize|u8|u16|u32|u64|u128|usize|f32|f64)$', text)
    if suffix_match:
        num_part = suffix_match.group(1)
        # Recurse without suffix
        result = classify_rust_numeric(num_part)
        return result

    # Remove underscores for parsing
    clean = text.replace("_", "")

    # Hex integer
    if clean.startswith("0x"):
        try:
            value = int(clean, 16)
            return (TokenCategory.INT_LITERAL, None, value)
        except ValueError:
            if re.match(r'^0x[0-9a-fA-F]*$', clean):
                return (TokenCategory.INT_LITERAL, None, None)
            return None

    # Octal integer
    if clean.startswith("0o"):
        try:
            value = int(clean, 8)
            return (TokenCategory.INT_LITERAL, None, value)
        except ValueError:
            if re.match(r'^0o[0-7]*$', clean):
                return (TokenCategory.INT_LITERAL, None, None)
            return None

    # Binary integer
    if clean.startswith("0b"):
        try:
            value = int(clean, 2)
            return (TokenCategory.INT_LITERAL, None, value)
        except ValueError:
            if re.match(r'^0b[01]*$', clean):
                return (TokenCategory.INT_LITERAL, None, None)
            return None

    # Float (check before integer)
    if "." in clean or "e" in clean.lower():
        try:
            value = float(clean)
            return (TokenCategory.FLOAT_LITERAL, None, value)
        except ValueError:
            if re.match(r'^[0-9]*\.?[0-9]*([eE][+-]?[0-9]*)?$', clean):
                return (TokenCategory.FLOAT_LITERAL, None, None)
            return None

    # Decimal integer
    if clean.isdigit() or (clean and clean[0].isdigit()):
        try:
            value = int(clean)
            return (TokenCategory.INT_LITERAL, None, value)
        except ValueError:
            if re.match(r'^[0-9]+$', clean):
                return (TokenCategory.INT_LITERAL, None, None)
            return None

    return None


def is_rust_identifier(text: str) -> bool:
    """Check if text is a valid Rust identifier."""
    if not text:
        return False

    # Can't start with digit
    if text[0].isdigit():
        return False

    # Raw identifier r#keyword
    if text.startswith("r#"):
        return is_rust_identifier(text[2:])

    # Standard identifier: alphanumeric + underscore
    # First char must be alphabetic or underscore
    if not (text[0].isalpha() or text[0] == "_"):
        return False

    return all(c.isalnum() or c == "_" for c in text)


def get_rust_keyword_category(keyword: str) -> str:
    """Get the category of a Rust keyword.

    Returns one of: 'control', 'definition', 'type', 'memory',
    'safety', 'async', 'operator', 'reserved'
    """
    if keyword in RUST_CONTROL_KEYWORDS:
        return "control"
    if keyword in RUST_DEFINITION_KEYWORDS:
        return "definition"
    if keyword in RUST_TYPE_KEYWORDS:
        return "type"
    if keyword in RUST_MEMORY_KEYWORDS:
        return "memory"
    if keyword in RUST_SAFETY_KEYWORDS:
        return "safety"
    if keyword in RUST_ASYNC_KEYWORDS:
        return "async"
    if keyword in RUST_OPERATOR_KEYWORDS:
        return "operator"
    if keyword in RUST_RESERVED_KEYWORDS:
        return "reserved"
    return "unknown"


# =============================================================================
# Integration with main TokenClassifier
# =============================================================================

def get_rust_keywords() -> FrozenSet[str]:
    """Get all Rust keywords."""
    return RUST_ALL_KEYWORDS


def get_rust_macros() -> FrozenSet[str]:
    """Get all standard Rust macros."""
    return RUST_STD_MACROS


def get_rust_operators() -> FrozenSet[str]:
    """Get all Rust operators."""
    return RUST_OPERATORS


def get_rust_delimiters() -> FrozenSet[str]:
    """Get all Rust delimiters."""
    return RUST_DELIMITERS


def get_rust_primitive_types() -> FrozenSet[str]:
    """Get all Rust primitive type names."""
    return RUST_PRIMITIVE_TYPES


def get_rust_std_types() -> FrozenSet[str]:
    """Get all Rust standard library types."""
    return RUST_STD_TYPES


def get_rust_common_traits() -> FrozenSet[str]:
    """Get all common Rust traits."""
    return RUST_COMMON_TRAITS


def get_rust_attributes() -> FrozenSet[str]:
    """Get common Rust attributes."""
    return RUST_COMMON_ATTRIBUTES
