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
"""Token classifier for vocabulary-wide token categorization.

This module provides TokenClassifier, a shared infrastructure for classifying
all tokens in a vocabulary. Classification is performed once at initialization,
enabling O(1) lookups during token mask computation.

Key optimization: Instead of calling tokenizer.decode() per token in token_mask(),
classify the entire vocabulary once and store results in lookup tables.

Usage:
    classifier = TokenClassifier(tokenizer, language="python")
    classifier.initialize()

    # O(1) lookups
    int_tokens = classifier.by_category(TokenCategory.INT_LITERAL)
    keyword_tokens = classifier.by_keyword("import")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import torch


class TokenCategory(Enum):
    """Categories of tokens based on syntactic role.

    Categories are mutually exclusive - each token belongs to exactly one.
    The classification is based on what a token would represent if it
    appeared at the start of an expression.
    """

    # Literals
    INT_LITERAL = auto()       # 0, 1, 42, 0x1F
    FLOAT_LITERAL = auto()     # 1.0, .5, 3e10
    STRING_LITERAL = auto()    # "foo", 'bar', f"...", b'...'
    BOOL_LITERAL = auto()      # True, False
    NONE_LITERAL = auto()      # None

    # Structural
    KEYWORD = auto()           # if, for, def, class, import, etc.
    OPERATOR = auto()          # +, -, *, =, ==, etc.
    DELIMITER = auto()         # (, ), [, ], {, }, ,, :, ;

    # Names
    IDENTIFIER = auto()        # variable/function names
    BUILTIN = auto()           # len, print, int, str, etc.

    # Special
    WHITESPACE = auto()        # spaces, newlines, tabs
    COMMENT = auto()           # # comment
    MIXED = auto()             # token contains multiple categories (rare)
    UNKNOWN = auto()           # cannot classify


@dataclass(frozen=True)
class TokenClassification:
    """Classification of a single token.

    Attributes:
        token_id: The token ID in the vocabulary
        text: The decoded text of the token
        category: Primary syntactic category
        literal_value: Parsed value if this is a complete literal
        keyword_name: The keyword if category is KEYWORD
        is_complete: Whether this is a complete syntactic unit
    """

    token_id: int
    text: str
    category: TokenCategory
    literal_value: Optional[Any] = None
    keyword_name: Optional[str] = None
    is_complete: bool = True


# Python keywords by category
PYTHON_CONTROL_KEYWORDS = frozenset({
    "if", "else", "elif", "for", "while", "break", "continue",
    "try", "except", "finally", "raise", "with", "as", "pass",
    "return", "yield", "async", "await",
})

PYTHON_DEFINITION_KEYWORDS = frozenset({
    "def", "class", "lambda", "global", "nonlocal",
})

PYTHON_IMPORT_KEYWORDS = frozenset({
    "import", "from",
})

PYTHON_OPERATOR_KEYWORDS = frozenset({
    "and", "or", "not", "in", "is",
})

PYTHON_ALL_KEYWORDS = (
    PYTHON_CONTROL_KEYWORDS |
    PYTHON_DEFINITION_KEYWORDS |
    PYTHON_IMPORT_KEYWORDS |
    PYTHON_OPERATOR_KEYWORDS
)

# Python builtins (functions and types)
PYTHON_BUILTINS = frozenset({
    # Types
    "int", "float", "str", "bool", "list", "dict", "set", "tuple",
    "bytes", "bytearray", "complex", "frozenset", "object", "type",
    # Functions
    "len", "print", "range", "enumerate", "zip", "map", "filter",
    "sorted", "reversed", "sum", "min", "max", "abs", "round",
    "open", "input", "repr", "hash", "id", "isinstance", "issubclass",
    "hasattr", "getattr", "setattr", "delattr", "callable", "iter", "next",
    "any", "all", "ord", "chr", "hex", "bin", "oct", "format",
    "super", "classmethod", "staticmethod", "property",
    # Exceptions
    "Exception", "ValueError", "TypeError", "KeyError", "IndexError",
    "RuntimeError", "StopIteration", "AttributeError", "ImportError",
})

# Python operators (single and multi-character)
PYTHON_OPERATORS = frozenset({
    "+", "-", "*", "/", "//", "%", "**", "@",
    "=", "==", "!=", "<", ">", "<=", ">=", "<>",
    "&", "|", "^", "~", "<<", ">>",
    "+=", "-=", "*=", "/=", "//=", "%=", "**=", "@=",
    "&=", "|=", "^=", ">>=", "<<=",
    "->", ":=",
})

# Python delimiters
PYTHON_DELIMITERS = frozenset({
    "(", ")", "[", "]", "{", "}",
    ",", ":", ";", ".", "...",
})


class TokenClassifier:
    """Classifies all tokens in a vocabulary for efficient lookup.

    The classifier performs full vocabulary classification once during
    initialization, then provides O(1) lookups by category or keyword.

    This is the key optimization for precise token masking - instead of
    calling tokenizer.decode() during every token_mask() call, we
    precompute all classifications upfront.

    Example:
        >>> classifier = TokenClassifier(tokenizer)
        >>> classifier.initialize()
        >>>
        >>> # Get all int literal tokens
        >>> int_tokens = classifier.by_category(TokenCategory.INT_LITERAL)
        >>>
        >>> # Get tokens that start "import"
        >>> import_tokens = classifier.by_keyword("import")
        >>>
        >>> # Create a mask blocking string literals
        >>> mask = classifier.create_mask(
        ...     vocab_size, device,
        ...     block_categories={TokenCategory.STRING_LITERAL}
        ... )
    """

    def __init__(
        self,
        tokenizer: Any,
        language: str = "python",
        vocab_size: Optional[int] = None,
    ) -> None:
        """Initialize the classifier.

        Args:
            tokenizer: The model's tokenizer
            language: Programming language for classification rules
            vocab_size: Override vocab size (defaults to tokenizer's)
        """
        self._tokenizer = tokenizer
        self._language = language
        self._vocab_size = vocab_size or getattr(tokenizer, "vocab_size", 32000)

        # Classification storage (populated by initialize())
        self._classifications: List[TokenClassification] = []
        self._by_category: Dict[TokenCategory, Set[int]] = {
            cat: set() for cat in TokenCategory
        }
        self._by_keyword: Dict[str, Set[int]] = {}
        self._by_builtin: Dict[str, Set[int]] = {}

        # Cached masks
        self._category_masks: Dict[TokenCategory, torch.Tensor] = {}

        self._initialized = False

    @property
    def initialized(self) -> bool:
        """Check if classifier has been initialized."""
        return self._initialized

    @property
    def vocab_size(self) -> int:
        """Get the vocabulary size."""
        return self._vocab_size

    def initialize(self) -> None:
        """Classify the entire vocabulary.

        This is an expensive operation (O(vocab_size)) but only
        happens once. All subsequent lookups are O(1).
        """
        if self._initialized:
            return

        self._classifications = []

        for token_id in range(self._vocab_size):
            classification = self._classify_token(token_id)
            self._classifications.append(classification)

            # Index by category
            self._by_category[classification.category].add(token_id)

            # Index by keyword
            if classification.keyword_name:
                kw = classification.keyword_name
                if kw not in self._by_keyword:
                    self._by_keyword[kw] = set()
                self._by_keyword[kw].add(token_id)

            # Index by builtin
            if classification.category == TokenCategory.BUILTIN:
                name = classification.text.strip()
                if name not in self._by_builtin:
                    self._by_builtin[name] = set()
                self._by_builtin[name].add(token_id)

        self._initialized = True

    def _classify_token(self, token_id: int) -> TokenClassification:
        """Classify a single token.

        Args:
            token_id: The token ID to classify

        Returns:
            TokenClassification for this token
        """
        # Decode the token
        try:
            text = self._tokenizer.decode([token_id])
        except Exception:
            return TokenClassification(
                token_id=token_id,
                text="",
                category=TokenCategory.UNKNOWN,
            )

        stripped = text.strip()

        # Empty or whitespace
        if not stripped:
            return TokenClassification(
                token_id=token_id,
                text=text,
                category=TokenCategory.WHITESPACE,
            )

        # Comment
        if stripped.startswith("#"):
            return TokenClassification(
                token_id=token_id,
                text=text,
                category=TokenCategory.COMMENT,
            )

        # String literal
        if self._is_string_literal(stripped):
            return TokenClassification(
                token_id=token_id,
                text=text,
                category=TokenCategory.STRING_LITERAL,
                literal_value=self._parse_string_literal(stripped),
                is_complete=self._is_complete_string(stripped),
            )

        # Boolean literal
        if stripped in ("True", "False"):
            return TokenClassification(
                token_id=token_id,
                text=text,
                category=TokenCategory.BOOL_LITERAL,
                literal_value=(stripped == "True"),
            )

        # None literal
        if stripped == "None":
            return TokenClassification(
                token_id=token_id,
                text=text,
                category=TokenCategory.NONE_LITERAL,
                literal_value=None,
            )

        # Numeric literals
        num_result = self._classify_numeric(stripped)
        if num_result is not None:
            category, value = num_result
            return TokenClassification(
                token_id=token_id,
                text=text,
                category=category,
                literal_value=value,
            )

        # Keywords
        if stripped in PYTHON_ALL_KEYWORDS:
            return TokenClassification(
                token_id=token_id,
                text=text,
                category=TokenCategory.KEYWORD,
                keyword_name=stripped,
            )

        # Builtins
        if stripped in PYTHON_BUILTINS:
            return TokenClassification(
                token_id=token_id,
                text=text,
                category=TokenCategory.BUILTIN,
            )

        # Operators
        if stripped in PYTHON_OPERATORS or self._is_operator_prefix(stripped):
            return TokenClassification(
                token_id=token_id,
                text=text,
                category=TokenCategory.OPERATOR,
            )

        # Delimiters
        if stripped in PYTHON_DELIMITERS:
            return TokenClassification(
                token_id=token_id,
                text=text,
                category=TokenCategory.DELIMITER,
            )

        # Identifier
        if self._is_identifier(stripped):
            return TokenClassification(
                token_id=token_id,
                text=text,
                category=TokenCategory.IDENTIFIER,
            )

        # Mixed or unknown
        return TokenClassification(
            token_id=token_id,
            text=text,
            category=TokenCategory.MIXED if len(stripped) > 1 else TokenCategory.UNKNOWN,
        )

    def _is_string_literal(self, text: str) -> bool:
        """Check if text is a string literal."""
        # Check for string prefixes (f, r, b, u, fr, rf, br, rb)
        prefixes = ("f'", 'f"', "r'", 'r"', "b'", 'b"', "u'", 'u"',
                    "fr'", 'fr"', "rf'", 'rf"', "br'", 'br"', "rb'", 'rb"',
                    "F'", 'F"', "R'", 'R"', "B'", 'B"', "U'", 'U"',
                    "'", '"', '"""', "'''")
        return any(text.startswith(p) for p in prefixes)

    def _is_complete_string(self, text: str) -> bool:
        """Check if string literal is complete (properly closed)."""
        if text.startswith('"""') or text.startswith("'''"):
            quote = text[:3]
            return len(text) >= 6 and text.endswith(quote)

        # Strip prefix
        stripped = text
        while stripped and stripped[0] in "frbFRBu":
            stripped = stripped[1:]

        if not stripped:
            return False

        quote = stripped[0]
        if quote not in ('"', "'"):
            return False

        # Check for matching close quote
        return len(stripped) >= 2 and stripped.endswith(quote) and not stripped.endswith("\\" + quote)

    def _parse_string_literal(self, text: str) -> Optional[str]:
        """Parse a string literal's value."""
        try:
            # Use ast.literal_eval for safety
            import ast
            return ast.literal_eval(text)
        except Exception:
            return None

    def _classify_numeric(self, text: str) -> Optional[Tuple[TokenCategory, Any]]:
        """Classify numeric literals.

        Returns:
            Tuple of (category, parsed_value) or None if not numeric
        """
        # Integer patterns
        int_patterns = [
            (r"^0[xX][0-9a-fA-F]+$", 16),  # Hex
            (r"^0[oO][0-7]+$", 8),          # Octal
            (r"^0[bB][01]+$", 2),           # Binary
            (r"^[0-9]+$", 10),               # Decimal
        ]

        for pattern, base in int_patterns:
            if re.match(pattern, text):
                try:
                    value = int(text, base) if base != 10 else int(text)
                    return (TokenCategory.INT_LITERAL, value)
                except ValueError:
                    pass

        # Float patterns
        float_pattern = r"^[0-9]*\.?[0-9]+([eE][+-]?[0-9]+)?$"
        if re.match(float_pattern, text) and ("." in text or "e" in text.lower()):
            try:
                value = float(text)
                return (TokenCategory.FLOAT_LITERAL, value)
            except ValueError:
                pass

        # Partial numeric (starts with digit but not complete)
        if text and text[0].isdigit():
            # Could be start of a number
            return (TokenCategory.INT_LITERAL, None)

        return None

    def _is_operator_prefix(self, text: str) -> bool:
        """Check if text is a prefix of any operator."""
        return any(op.startswith(text) for op in PYTHON_OPERATORS)

    def _is_identifier(self, text: str) -> bool:
        """Check if text is a valid identifier."""
        if not text:
            return False
        if not (text[0].isalpha() or text[0] == '_'):
            return False
        return all(c.isalnum() or c == '_' for c in text)

    # ==================== Query Methods ====================

    def get_classification(self, token_id: int) -> TokenClassification:
        """Get classification for a specific token.

        Args:
            token_id: The token ID

        Returns:
            TokenClassification for this token
        """
        if not self._initialized:
            raise RuntimeError("TokenClassifier not initialized. Call initialize() first.")

        if token_id < 0 or token_id >= len(self._classifications):
            raise ValueError(f"Token ID {token_id} out of range")

        return self._classifications[token_id]

    def by_category(self, category: TokenCategory) -> FrozenSet[int]:
        """Get all token IDs of a category.

        Args:
            category: The token category

        Returns:
            Frozen set of token IDs
        """
        if not self._initialized:
            raise RuntimeError("TokenClassifier not initialized. Call initialize() first.")

        return frozenset(self._by_category.get(category, set()))

    def by_keyword(self, keyword: str) -> FrozenSet[int]:
        """Get all token IDs that are/start a keyword.

        Args:
            keyword: The keyword name (e.g., "import", "if")

        Returns:
            Frozen set of token IDs
        """
        if not self._initialized:
            raise RuntimeError("TokenClassifier not initialized. Call initialize() first.")

        return frozenset(self._by_keyword.get(keyword, set()))

    def by_builtin(self, name: str) -> FrozenSet[int]:
        """Get all token IDs that are/start a builtin name.

        Args:
            name: The builtin name (e.g., "len", "print")

        Returns:
            Frozen set of token IDs
        """
        if not self._initialized:
            raise RuntimeError("TokenClassifier not initialized. Call initialize() first.")

        return frozenset(self._by_builtin.get(name, set()))

    def all_keywords(self) -> FrozenSet[str]:
        """Get all keywords that have associated tokens."""
        return frozenset(self._by_keyword.keys())

    def all_builtins(self) -> FrozenSet[str]:
        """Get all builtins that have associated tokens."""
        return frozenset(self._by_builtin.keys())

    # ==================== Mask Creation ====================

    def create_mask(
        self,
        vocab_size: int,
        device: str = "cpu",
        *,
        allow_categories: Optional[Set[TokenCategory]] = None,
        block_categories: Optional[Set[TokenCategory]] = None,
        allow_keywords: Optional[Set[str]] = None,
        block_keywords: Optional[Set[str]] = None,
    ) -> torch.Tensor:
        """Create a boolean mask based on token categories.

        Args:
            vocab_size: Size of the vocabulary
            device: PyTorch device for the mask
            allow_categories: Only allow these categories (whitelist)
            block_categories: Block these categories (blacklist)
            allow_keywords: Only allow these keywords
            block_keywords: Block these keywords

        Returns:
            Boolean tensor of shape (vocab_size,)
        """
        if not self._initialized:
            raise RuntimeError("TokenClassifier not initialized. Call initialize() first.")

        # Start with all True (allow all)
        mask = torch.ones(vocab_size, dtype=torch.bool, device=device)

        # Apply category filtering
        if allow_categories is not None:
            # Whitelist mode: block everything not in allow_categories
            for cat in TokenCategory:
                if cat not in allow_categories:
                    for token_id in self._by_category.get(cat, set()):
                        if token_id < vocab_size:
                            mask[token_id] = False

        if block_categories is not None:
            # Blacklist mode: block specific categories
            for cat in block_categories:
                for token_id in self._by_category.get(cat, set()):
                    if token_id < vocab_size:
                        mask[token_id] = False

        # Apply keyword filtering
        if allow_keywords is not None:
            # Block all keywords not in allow_keywords
            for kw, tokens in self._by_keyword.items():
                if kw not in allow_keywords:
                    for token_id in tokens:
                        if token_id < vocab_size:
                            mask[token_id] = False

        if block_keywords is not None:
            # Block specific keywords
            for kw in block_keywords:
                for token_id in self._by_keyword.get(kw, set()):
                    if token_id < vocab_size:
                        mask[token_id] = False

        return mask

    def get_category_mask(
        self,
        category: TokenCategory,
        vocab_size: int,
        device: str = "cpu",
    ) -> torch.Tensor:
        """Get a cached mask for tokens of a specific category.

        Args:
            category: The token category
            vocab_size: Size of the vocabulary
            device: PyTorch device

        Returns:
            Boolean tensor where True indicates token is in category
        """
        cache_key = (category, vocab_size, device)

        # Check cache (simplified - full impl would handle device properly)
        if category not in self._category_masks:
            mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
            for token_id in self._by_category.get(category, set()):
                if token_id < vocab_size:
                    mask[token_id] = True
            self._category_masks[category] = mask

        return self._category_masks[category].to(device)

    # ==================== Statistics ====================

    def get_statistics(self) -> Dict[str, Any]:
        """Get classification statistics.

        Returns:
            Dictionary with counts per category
        """
        if not self._initialized:
            return {"initialized": False}

        stats = {
            "initialized": True,
            "vocab_size": self._vocab_size,
            "categories": {
                cat.name: len(tokens)
                for cat, tokens in self._by_category.items()
            },
            "keyword_count": len(self._by_keyword),
            "builtin_count": len(self._by_builtin),
        }

        return stats


# Global classifier cache
_classifier_cache: Dict[int, TokenClassifier] = {}


def get_or_create_classifier(
    tokenizer: Any,
    language: str = "python",
) -> TokenClassifier:
    """Get or create a TokenClassifier for a tokenizer.

    Uses a global cache keyed by tokenizer ID to avoid
    re-classifying the same vocabulary multiple times.

    Args:
        tokenizer: The model's tokenizer
        language: Programming language

    Returns:
        Initialized TokenClassifier
    """
    key = id(tokenizer)

    if key not in _classifier_cache:
        classifier = TokenClassifier(tokenizer, language)
        classifier.initialize()
        _classifier_cache[key] = classifier

    return _classifier_cache[key]


def clear_classifier_cache() -> None:
    """Clear the global classifier cache."""
    _classifier_cache.clear()


# ===========================================================================
# Language-Specific Classification Dispatch
# ===========================================================================


def get_language_keywords(language: str) -> FrozenSet[str]:
    """Get all keywords for a programming language.

    Args:
        language: Language name ("python", "zig", "rust", "typescript", "go")

    Returns:
        Frozen set of keyword strings
    """
    if language == "python" or language == "py":
        return PYTHON_ALL_KEYWORDS
    elif language == "zig":
        from core.token_classifier_zig import ZIG_ALL_KEYWORDS
        return ZIG_ALL_KEYWORDS
    elif language == "rust" or language == "rs":
        from core.token_classifier_rust import RUST_ALL_KEYWORDS
        return RUST_ALL_KEYWORDS
    elif language in ("typescript", "ts", "javascript", "js"):
        from core.token_classifier_typescript import TYPESCRIPT_ALL_KEYWORDS
        return TYPESCRIPT_ALL_KEYWORDS
    elif language == "go":
        from core.token_classifier_go import GO_ALL_KEYWORDS
        return GO_ALL_KEYWORDS
    elif language in ("kotlin", "kt"):
        from core.token_classifier_kotlin import KOTLIN_ALL_KEYWORDS
        return KOTLIN_ALL_KEYWORDS
    elif language == "swift":
        from core.token_classifier_swift import SWIFT_ALL_KEYWORDS
        return SWIFT_ALL_KEYWORDS
    else:
        return PYTHON_ALL_KEYWORDS  # Default to Python


def get_language_builtins(language: str) -> FrozenSet[str]:
    """Get all builtins for a programming language.

    Args:
        language: Language name ("python", "zig", "rust", "typescript", "go")

    Returns:
        Frozen set of builtin strings
    """
    if language == "python" or language == "py":
        return PYTHON_BUILTINS
    elif language == "zig":
        from core.token_classifier_zig import ZIG_BUILTINS
        return ZIG_BUILTINS
    elif language == "rust" or language == "rs":
        from core.token_classifier_rust import RUST_COMMON_MACROS
        return RUST_COMMON_MACROS
    elif language in ("typescript", "ts", "javascript", "js"):
        from core.token_classifier_typescript import TYPESCRIPT_ALL_BUILTINS
        return TYPESCRIPT_ALL_BUILTINS
    elif language == "go":
        from core.token_classifier_go import get_go_builtins
        return frozenset(get_go_builtins())
    elif language in ("kotlin", "kt"):
        from core.token_classifier_kotlin import KOTLIN_ALL_BUILTINS
        return KOTLIN_ALL_BUILTINS
    elif language == "swift":
        from core.token_classifier_swift import SWIFT_ALL_BUILTINS
        return SWIFT_ALL_BUILTINS
    else:
        return PYTHON_BUILTINS  # Default to Python


def classify_token_for_language(
    text: str,
    language: str = "python"
) -> Tuple[TokenCategory, Optional[str], Optional[Any]]:
    """Classify a single token using language-specific rules.

    This function dispatches to the appropriate language classifier.

    Args:
        text: The token text to classify
        language: Programming language ("python", "zig", "rust", "typescript", "go")

    Returns:
        Tuple of (category, keyword_name or None, literal_value or None)
    """
    stripped = text.strip()

    if language == "zig":
        from core.token_classifier_zig import classify_zig_token
        return classify_zig_token(stripped)
    elif language == "rust" or language == "rs":
        from core.token_classifier_rust import classify_rust_token
        return classify_rust_token(stripped)
    elif language in ("typescript", "ts", "javascript", "js"):
        from core.token_classifier_typescript import classify_typescript_token
        result = classify_typescript_token(stripped)
        return (result.category, result.keyword_name, result.literal_value)
    elif language == "go":
        from core.token_classifier_go import classify_go_token
        result = classify_go_token(stripped)
        return (result.category, result.keyword_name, result.literal_value)
    elif language in ("kotlin", "kt"):
        from core.token_classifier_kotlin import classify_kotlin_token
        result = classify_kotlin_token(stripped)
        return (result.category, result.keyword_name, result.literal_value)
    elif language == "swift":
        from core.token_classifier_swift import classify_swift_token
        result = classify_swift_token(stripped)
        return (result.category, result.keyword_name, result.literal_value)
    else:
        # Python (default)
        return _classify_python_token(stripped)


def _classify_python_token(text: str) -> Tuple[TokenCategory, Optional[str], Optional[Any]]:
    """Classify a Python token.

    Args:
        text: The stripped token text

    Returns:
        Tuple of (category, keyword_name or None, literal_value or None)
    """
    # Empty
    if not text:
        return (TokenCategory.WHITESPACE, None, None)

    # Comment
    if text.startswith("#"):
        return (TokenCategory.COMMENT, None, None)

    # Keywords
    if text in PYTHON_ALL_KEYWORDS:
        return (TokenCategory.KEYWORD, text, None)

    # Boolean literals
    if text in ("True", "False"):
        return (TokenCategory.BOOL_LITERAL, None, text == "True")

    # None literal
    if text == "None":
        return (TokenCategory.NONE_LITERAL, None, None)

    # Builtins
    if text in PYTHON_BUILTINS:
        return (TokenCategory.BUILTIN, None, None)

    # Operators
    if text in PYTHON_OPERATORS:
        return (TokenCategory.OPERATOR, None, None)

    # Delimiters
    if text in PYTHON_DELIMITERS:
        return (TokenCategory.DELIMITER, None, None)

    # Numeric literals (simplified check)
    if text and text[0].isdigit():
        if "." in text or "e" in text.lower():
            try:
                return (TokenCategory.FLOAT_LITERAL, None, float(text))
            except ValueError:
                pass
        else:
            try:
                return (TokenCategory.INT_LITERAL, None, int(text, 0))
            except ValueError:
                pass

    # String literal check (simplified)
    if text.startswith(('"', "'", 'f"', "f'", 'r"', "r'", 'b"', "b'")):
        return (TokenCategory.STRING_LITERAL, None, None)

    # Identifier
    if text.isidentifier():
        return (TokenCategory.IDENTIFIER, None, None)

    return (TokenCategory.UNKNOWN, None, None)


def supported_classifier_languages() -> List[str]:
    """Get list of languages with token classifier support.

    Returns:
        List of supported language names
    """
    return ["python", "zig", "rust", "typescript", "go", "kotlin", "swift"]
