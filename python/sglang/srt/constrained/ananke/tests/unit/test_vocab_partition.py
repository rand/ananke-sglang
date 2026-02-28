# Copyright Rand Arete @ Ananke 2025
# Licensed under the Apache License, Version 2.0
"""Unit tests for VocabPartition FFI wrapper.

Tests the Zig-based vocabulary classification system that provides
O(1) token category lookup after one-time vocabulary classification.
"""

import pytest
import torch

# Check numpy availability
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

requires_numpy = pytest.mark.skipif(not HAS_NUMPY, reason="NumPy required")

# Import from the ananke package
try:
    from zig.ffi import (
        VocabPartition,
        is_native_available,
        TYPE_CATEGORY_INTEGER,
        TYPE_CATEGORY_FLOAT,
        TYPE_CATEGORY_STRING,
        TYPE_CATEGORY_BOOLEAN,
        TYPE_CATEGORY_NONE_NULL,
        TYPE_CATEGORY_IDENTIFIER,
        TYPE_CATEGORY_KEYWORD,
        TYPE_CATEGORY_BUILTIN,
        TYPE_CATEGORY_DELIMITER,
        TYPE_CATEGORY_WHITESPACE,
        TYPE_CATEGORY_UNKNOWN,
    )
    from core.token_classifier import (
        get_best_classifier,
        is_native_classifier_available,
        NativeTokenClassifier,
        TokenCategory,
    )
except ImportError:
    from sglang.srt.constrained.ananke.zig.ffi import (
        VocabPartition,
        is_native_available,
        TYPE_CATEGORY_INTEGER,
        TYPE_CATEGORY_FLOAT,
        TYPE_CATEGORY_STRING,
        TYPE_CATEGORY_BOOLEAN,
        TYPE_CATEGORY_NONE_NULL,
        TYPE_CATEGORY_IDENTIFIER,
        TYPE_CATEGORY_KEYWORD,
        TYPE_CATEGORY_BUILTIN,
        TYPE_CATEGORY_DELIMITER,
        TYPE_CATEGORY_WHITESPACE,
        TYPE_CATEGORY_UNKNOWN,
    )
    from sglang.srt.constrained.ananke.core.token_classifier import (
        get_best_classifier,
        is_native_classifier_available,
        NativeTokenClassifier,
        TokenCategory,
    )


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self, vocab: dict):
        """Initialize with token -> text mapping."""
        self._vocab = vocab
        self.vocab_size = len(vocab)

    def decode(self, token_ids):
        return "".join(self._vocab.get(tid, "") for tid in token_ids)

    def get_vocab(self):
        return {v: k for k, v in self._vocab.items()}


@pytest.fixture
def simple_tokenizer():
    """Create a simple tokenizer with common Python tokens."""
    vocab = {
        0: "def",
        1: "class",
        2: "if",
        3: "return",
        4: "True",
        5: "False",
        6: "None",
        7: "print",
        8: "len",
        9: "42",
        10: "3.14",
        11: '"hello"',
        12: "foo",
        13: "bar",
        14: "+",
        15: "-",
        16: "(",
        17: ")",
        18: ",",
        19: " ",
    }
    return MockTokenizer(vocab)


# =============================================================================
# VocabPartition Tests
# =============================================================================


class TestVocabPartition:
    """Tests for VocabPartition wrapper."""

    def test_creation(self):
        """Test partition can be created."""
        partition = VocabPartition(vocab_size=100, language="python")
        assert partition.vocab_size == 100

    def test_classify_vocabulary(self, simple_tokenizer):
        """Test vocabulary classification."""
        partition = VocabPartition(
            vocab_size=simple_tokenizer.vocab_size,
            language="python",
        )

        # Get token strings
        token_strings = [
            simple_tokenizer.decode([i]) for i in range(simple_tokenizer.vocab_size)
        ]

        # Classify
        partition.classify_vocabulary(token_strings)

        # Verify classifications (these work regardless of native/Python)
        # Keywords
        assert partition.get_category(0) == TYPE_CATEGORY_KEYWORD  # def
        assert partition.get_category(1) == TYPE_CATEGORY_KEYWORD  # class
        assert partition.get_category(2) == TYPE_CATEGORY_KEYWORD  # if
        assert partition.get_category(3) == TYPE_CATEGORY_KEYWORD  # return

        # Boolean literals
        assert partition.get_category(4) == TYPE_CATEGORY_BOOLEAN  # True
        assert partition.get_category(5) == TYPE_CATEGORY_BOOLEAN  # False

        # None literal
        assert partition.get_category(6) == TYPE_CATEGORY_NONE_NULL  # None

        # Builtins
        assert partition.get_category(7) == TYPE_CATEGORY_BUILTIN  # print
        assert partition.get_category(8) == TYPE_CATEGORY_BUILTIN  # len

        # Integer
        assert partition.get_category(9) == TYPE_CATEGORY_INTEGER  # 42

        # Float
        assert partition.get_category(10) == TYPE_CATEGORY_FLOAT  # 3.14

        # String
        assert partition.get_category(11) == TYPE_CATEGORY_STRING  # "hello"

        # Identifiers
        assert partition.get_category(12) == TYPE_CATEGORY_IDENTIFIER  # foo
        assert partition.get_category(13) == TYPE_CATEGORY_IDENTIFIER  # bar

    @requires_numpy
    def test_get_category_mask(self, simple_tokenizer):
        """Test getting precomputed category mask."""
        partition = VocabPartition(
            vocab_size=simple_tokenizer.vocab_size,
            language="python",
        )

        token_strings = [
            simple_tokenizer.decode([i]) for i in range(simple_tokenizer.vocab_size)
        ]
        partition.classify_vocabulary(token_strings)

        # Get keyword mask
        mask, popcount = partition.get_category_mask(TYPE_CATEGORY_KEYWORD)

        # Should have 4 keywords: def, class, if, return
        assert popcount == 4

    @requires_numpy
    def test_compute_type_mask(self, simple_tokenizer):
        """Test computing combined type mask."""
        partition = VocabPartition(
            vocab_size=simple_tokenizer.vocab_size,
            language="python",
        )

        token_strings = [
            simple_tokenizer.decode([i]) for i in range(simple_tokenizer.vocab_size)
        ]
        partition.classify_vocabulary(token_strings)

        # Get mask for literals (int, float, string, bool, none)
        categories = [
            TYPE_CATEGORY_INTEGER,
            TYPE_CATEGORY_FLOAT,
            TYPE_CATEGORY_STRING,
            TYPE_CATEGORY_BOOLEAN,
            TYPE_CATEGORY_NONE_NULL,
        ]
        mask, popcount = partition.compute_type_mask(categories)

        # Should have: 42, 3.14, "hello", True, False, None = 6 tokens
        assert popcount == 6

    @requires_numpy
    def test_to_bool_tensor(self, simple_tokenizer):
        """Test converting packed mask to tensor."""
        partition = VocabPartition(
            vocab_size=simple_tokenizer.vocab_size,
            language="python",
        )

        token_strings = [
            simple_tokenizer.decode([i]) for i in range(simple_tokenizer.vocab_size)
        ]
        partition.classify_vocabulary(token_strings)

        mask, _ = partition.get_category_mask(TYPE_CATEGORY_KEYWORD)
        tensor = partition.to_bool_tensor(mask, device="cpu")

        assert tensor.shape == (simple_tokenizer.vocab_size,)
        assert tensor.dtype == torch.bool

        # Keywords at positions 0, 1, 2, 3
        assert tensor[0].item() is True
        assert tensor[1].item() is True
        assert tensor[2].item() is True
        assert tensor[3].item() is True

        # Non-keywords should be False
        assert tensor[4].item() is False  # True (boolean)
        assert tensor[9].item() is False  # 42 (integer)

    def test_is_in_category(self, simple_tokenizer):
        """Test checking token category membership."""
        partition = VocabPartition(
            vocab_size=simple_tokenizer.vocab_size,
            language="python",
        )

        token_strings = [
            simple_tokenizer.decode([i]) for i in range(simple_tokenizer.vocab_size)
        ]
        partition.classify_vocabulary(token_strings)

        assert partition.is_in_category(0, TYPE_CATEGORY_KEYWORD) == True
        assert partition.is_in_category(0, TYPE_CATEGORY_INTEGER) == False
        assert partition.is_in_category(9, TYPE_CATEGORY_INTEGER) == True
        assert partition.is_in_category(9, TYPE_CATEGORY_KEYWORD) == False


# =============================================================================
# NativeTokenClassifier Tests
# =============================================================================


class TestNativeTokenClassifier:
    """Tests for NativeTokenClassifier wrapper."""

    def test_creation(self, simple_tokenizer):
        """Test creating NativeTokenClassifier."""
        partition = VocabPartition(
            vocab_size=simple_tokenizer.vocab_size,
            language="python",
        )

        token_strings = [
            simple_tokenizer.decode([i]) for i in range(simple_tokenizer.vocab_size)
        ]
        partition.classify_vocabulary(token_strings)

        classifier = NativeTokenClassifier(partition, simple_tokenizer, "python")
        assert classifier.initialized is True
        assert classifier.vocab_size == simple_tokenizer.vocab_size

    @requires_numpy
    def test_get_category_mask(self, simple_tokenizer):
        """Test getting category mask through classifier interface."""
        partition = VocabPartition(
            vocab_size=simple_tokenizer.vocab_size,
            language="python",
        )

        token_strings = [
            simple_tokenizer.decode([i]) for i in range(simple_tokenizer.vocab_size)
        ]
        partition.classify_vocabulary(token_strings)

        classifier = NativeTokenClassifier(partition, simple_tokenizer, "python")
        mask = classifier.get_category_mask(
            TokenCategory.KEYWORD,
            simple_tokenizer.vocab_size,
            device="cpu",
        )

        assert mask.shape == (simple_tokenizer.vocab_size,)
        assert mask.dtype == torch.bool

    @requires_numpy
    def test_create_mask_with_allow_categories(self, simple_tokenizer):
        """Test creating mask with allow_categories."""
        partition = VocabPartition(
            vocab_size=simple_tokenizer.vocab_size,
            language="python",
        )

        token_strings = [
            simple_tokenizer.decode([i]) for i in range(simple_tokenizer.vocab_size)
        ]
        partition.classify_vocabulary(token_strings)

        classifier = NativeTokenClassifier(partition, simple_tokenizer, "python")
        mask = classifier.create_mask(
            simple_tokenizer.vocab_size,
            device="cpu",
            allow_categories={TokenCategory.INT_LITERAL, TokenCategory.FLOAT_LITERAL},
        )

        # Only 42 and 3.14 should be allowed
        assert mask[9].item() is True  # 42
        assert mask[10].item() is True  # 3.14
        assert mask[0].item() is False  # def


# =============================================================================
# get_best_classifier Tests
# =============================================================================


class TestGetBestClassifier:
    """Tests for get_best_classifier function."""

    def test_returns_classifier(self, simple_tokenizer):
        """Test that get_best_classifier returns a working classifier."""
        classifier = get_best_classifier(
            simple_tokenizer,
            language="python",
            prefer_native=True,
        )

        assert classifier.initialized is True
        assert classifier.vocab_size == simple_tokenizer.vocab_size

    def test_python_fallback(self, simple_tokenizer):
        """Test that Python fallback works when native is disabled."""
        classifier = get_best_classifier(
            simple_tokenizer,
            language="python",
            prefer_native=False,
        )

        assert classifier.initialized is True

    def test_is_native_available(self):
        """Test is_native_classifier_available function."""
        # Should return bool without crashing
        result = is_native_classifier_available()
        assert isinstance(result, bool)


# =============================================================================
# Language Support Tests
# =============================================================================


class TestLanguageSupport:
    """Tests for multi-language classification."""

    @pytest.fixture
    def typescript_tokenizer(self):
        """Create a tokenizer with TypeScript tokens."""
        vocab = {
            0: "const",
            1: "let",
            2: "interface",
            3: "async",
            4: "await",
            5: "true",
            6: "false",
            7: "null",
            8: "undefined",
            9: "function",
        }
        return MockTokenizer(vocab)

    def test_typescript_keywords(self, typescript_tokenizer):
        """Test TypeScript keyword classification."""
        partition = VocabPartition(
            vocab_size=typescript_tokenizer.vocab_size,
            language="typescript",
        )

        token_strings = [
            typescript_tokenizer.decode([i])
            for i in range(typescript_tokenizer.vocab_size)
        ]
        partition.classify_vocabulary(token_strings)

        # TypeScript keywords (unambiguous)
        assert partition.get_category(0) == TYPE_CATEGORY_KEYWORD  # const
        assert partition.get_category(1) == TYPE_CATEGORY_KEYWORD  # let
        assert partition.get_category(2) == TYPE_CATEGORY_KEYWORD  # interface
        assert partition.get_category(9) == TYPE_CATEGORY_KEYWORD  # function

        # Booleans
        assert partition.get_category(5) == TYPE_CATEGORY_BOOLEAN  # true
        assert partition.get_category(6) == TYPE_CATEGORY_BOOLEAN  # false

        # null/undefined: These are classified as KEYWORD by Python fallback
        # but as NONE_NULL by native Zig classifier. Both are valid.
        null_cat = partition.get_category(7)
        undefined_cat = partition.get_category(8)
        assert null_cat in (TYPE_CATEGORY_NONE_NULL, TYPE_CATEGORY_KEYWORD)
        assert undefined_cat in (TYPE_CATEGORY_NONE_NULL, TYPE_CATEGORY_KEYWORD)

    @pytest.fixture
    def rust_tokenizer(self):
        """Create a tokenizer with Rust tokens."""
        vocab = {
            0: "fn",
            1: "let",
            2: "mut",
            3: "impl",
            4: "struct",
            5: "true",
            6: "false",
        }
        return MockTokenizer(vocab)

    def test_rust_keywords(self, rust_tokenizer):
        """Test Rust keyword classification."""
        partition = VocabPartition(
            vocab_size=rust_tokenizer.vocab_size,
            language="rust",
        )

        token_strings = [
            rust_tokenizer.decode([i]) for i in range(rust_tokenizer.vocab_size)
        ]
        partition.classify_vocabulary(token_strings)

        # Rust keywords
        assert partition.get_category(0) == TYPE_CATEGORY_KEYWORD  # fn
        assert partition.get_category(1) == TYPE_CATEGORY_KEYWORD  # let
        assert partition.get_category(2) == TYPE_CATEGORY_KEYWORD  # mut
        assert partition.get_category(3) == TYPE_CATEGORY_KEYWORD  # impl
        assert partition.get_category(4) == TYPE_CATEGORY_KEYWORD  # struct
