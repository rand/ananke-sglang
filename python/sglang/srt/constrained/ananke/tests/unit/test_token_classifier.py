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
"""Unit tests for TokenClassifier."""

import pytest
import torch
from typing import Dict, List

from core.token_classifier import (
    TokenCategory,
    TokenClassification,
    TokenClassifier,
    get_or_create_classifier,
    clear_classifier_cache,
    PYTHON_ALL_KEYWORDS,
    PYTHON_CONTROL_KEYWORDS,
    PYTHON_DEFINITION_KEYWORDS,
    PYTHON_IMPORT_KEYWORDS,
    PYTHON_BUILTINS,
    PYTHON_OPERATORS,
    PYTHON_DELIMITERS,
)


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self, vocab: Dict[int, str]):
        """Initialize with a vocabulary mapping.

        Args:
            vocab: Mapping from token ID to token text
        """
        self._vocab = vocab
        self.vocab_size = len(vocab)

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        return "".join(self._vocab.get(tid, "") for tid in token_ids)


@pytest.fixture
def small_tokenizer():
    """A small mock tokenizer for testing."""
    vocab = {
        0: " ",       # whitespace
        1: "\n",      # newline
        2: "42",      # int literal
        3: "3.14",    # float literal
        4: '"hello"', # string literal
        5: "True",    # bool literal
        6: "False",   # bool literal
        7: "None",    # none literal
        8: "if",      # keyword
        9: "def",     # keyword
        10: "import", # keyword
        11: "return", # keyword
        12: "+",      # operator
        13: "=",      # operator
        14: "==",     # operator
        15: "(",      # delimiter
        16: ")",      # delimiter
        17: "[",      # delimiter
        18: "]",      # delimiter
        19: "foo",    # identifier
        20: "bar",    # identifier
        21: "_var",   # identifier
        22: "len",    # builtin
        23: "print",  # builtin
        24: "int",    # builtin
        25: "#comment",  # comment
        26: "0x1F",   # hex int
        27: "1e10",   # scientific float
        28: "'single'",  # single quote string
        29: "for",    # keyword
        30: "while",  # keyword
        31: "class",  # keyword
    }
    return MockTokenizer(vocab)


@pytest.fixture
def classifier(small_tokenizer):
    """An initialized classifier for testing."""
    clf = TokenClassifier(small_tokenizer)
    clf.initialize()
    return clf


class TestTokenCategory:
    """Tests for TokenCategory enum."""

    def test_all_categories_distinct(self):
        """All category values should be distinct."""
        values = [cat.value for cat in TokenCategory]
        assert len(values) == len(set(values))

    def test_expected_categories_exist(self):
        """Expected categories should exist."""
        assert TokenCategory.INT_LITERAL is not None
        assert TokenCategory.FLOAT_LITERAL is not None
        assert TokenCategory.STRING_LITERAL is not None
        assert TokenCategory.BOOL_LITERAL is not None
        assert TokenCategory.NONE_LITERAL is not None
        assert TokenCategory.KEYWORD is not None
        assert TokenCategory.IDENTIFIER is not None
        assert TokenCategory.OPERATOR is not None
        assert TokenCategory.DELIMITER is not None


class TestTokenClassification:
    """Tests for TokenClassification dataclass."""

    def test_frozen(self):
        """TokenClassification should be frozen."""
        tc = TokenClassification(
            token_id=0,
            text="test",
            category=TokenCategory.IDENTIFIER,
        )
        with pytest.raises(AttributeError):
            tc.text = "other"

    def test_equality(self):
        """Equal classifications should compare equal."""
        tc1 = TokenClassification(0, "test", TokenCategory.IDENTIFIER)
        tc2 = TokenClassification(0, "test", TokenCategory.IDENTIFIER)
        assert tc1 == tc2

    def test_with_literal_value(self):
        """Classification can store literal value."""
        tc = TokenClassification(
            token_id=0,
            text="42",
            category=TokenCategory.INT_LITERAL,
            literal_value=42,
        )
        assert tc.literal_value == 42

    def test_with_keyword_name(self):
        """Classification can store keyword name."""
        tc = TokenClassification(
            token_id=0,
            text="if",
            category=TokenCategory.KEYWORD,
            keyword_name="if",
        )
        assert tc.keyword_name == "if"


class TestTokenClassifier:
    """Tests for TokenClassifier."""

    def test_not_initialized_raises(self, small_tokenizer):
        """Accessing classifications before initialize() raises."""
        clf = TokenClassifier(small_tokenizer)
        with pytest.raises(RuntimeError, match="not initialized"):
            clf.by_category(TokenCategory.INT_LITERAL)

    def test_initialize_success(self, small_tokenizer):
        """initialize() should complete successfully."""
        clf = TokenClassifier(small_tokenizer)
        clf.initialize()
        assert clf.initialized is True

    def test_double_initialize_no_op(self, small_tokenizer):
        """Double initialize should be a no-op."""
        clf = TokenClassifier(small_tokenizer)
        clf.initialize()
        clf.initialize()  # Should not raise
        assert clf.initialized is True

    def test_vocab_size(self, classifier):
        """vocab_size should match tokenizer."""
        assert classifier.vocab_size == 32


class TestClassificationAccuracy:
    """Tests for classification accuracy."""

    def test_whitespace_classification(self, classifier):
        """Whitespace tokens should be classified correctly."""
        whitespace_tokens = classifier.by_category(TokenCategory.WHITESPACE)
        assert 0 in whitespace_tokens  # " "
        assert 1 in whitespace_tokens  # "\n"

    def test_int_literal_classification(self, classifier):
        """Int literals should be classified correctly."""
        int_tokens = classifier.by_category(TokenCategory.INT_LITERAL)
        assert 2 in int_tokens  # "42"
        assert 26 in int_tokens  # "0x1F"

    def test_float_literal_classification(self, classifier):
        """Float literals should be classified correctly."""
        float_tokens = classifier.by_category(TokenCategory.FLOAT_LITERAL)
        assert 3 in float_tokens  # "3.14"
        assert 27 in float_tokens  # "1e10"

    def test_string_literal_classification(self, classifier):
        """String literals should be classified correctly."""
        string_tokens = classifier.by_category(TokenCategory.STRING_LITERAL)
        assert 4 in string_tokens  # '"hello"'
        assert 28 in string_tokens  # "'single'"

    def test_bool_literal_classification(self, classifier):
        """Bool literals should be classified correctly."""
        bool_tokens = classifier.by_category(TokenCategory.BOOL_LITERAL)
        assert 5 in bool_tokens  # "True"
        assert 6 in bool_tokens  # "False"

    def test_none_literal_classification(self, classifier):
        """None literal should be classified correctly."""
        none_tokens = classifier.by_category(TokenCategory.NONE_LITERAL)
        assert 7 in none_tokens  # "None"

    def test_keyword_classification(self, classifier):
        """Keywords should be classified correctly."""
        keyword_tokens = classifier.by_category(TokenCategory.KEYWORD)
        assert 8 in keyword_tokens   # "if"
        assert 9 in keyword_tokens   # "def"
        assert 10 in keyword_tokens  # "import"
        assert 11 in keyword_tokens  # "return"
        assert 29 in keyword_tokens  # "for"
        assert 30 in keyword_tokens  # "while"
        assert 31 in keyword_tokens  # "class"

    def test_operator_classification(self, classifier):
        """Operators should be classified correctly."""
        operator_tokens = classifier.by_category(TokenCategory.OPERATOR)
        assert 12 in operator_tokens  # "+"
        assert 13 in operator_tokens  # "="
        assert 14 in operator_tokens  # "=="

    def test_delimiter_classification(self, classifier):
        """Delimiters should be classified correctly."""
        delimiter_tokens = classifier.by_category(TokenCategory.DELIMITER)
        assert 15 in delimiter_tokens  # "("
        assert 16 in delimiter_tokens  # ")"
        assert 17 in delimiter_tokens  # "["
        assert 18 in delimiter_tokens  # "]"

    def test_identifier_classification(self, classifier):
        """Identifiers should be classified correctly."""
        identifier_tokens = classifier.by_category(TokenCategory.IDENTIFIER)
        assert 19 in identifier_tokens  # "foo"
        assert 20 in identifier_tokens  # "bar"
        assert 21 in identifier_tokens  # "_var"

    def test_builtin_classification(self, classifier):
        """Builtins should be classified correctly."""
        builtin_tokens = classifier.by_category(TokenCategory.BUILTIN)
        assert 22 in builtin_tokens  # "len"
        assert 23 in builtin_tokens  # "print"
        assert 24 in builtin_tokens  # "int"

    def test_comment_classification(self, classifier):
        """Comments should be classified correctly."""
        comment_tokens = classifier.by_category(TokenCategory.COMMENT)
        assert 25 in comment_tokens  # "#comment"


class TestByKeywordLookup:
    """Tests for by_keyword() lookups."""

    def test_by_keyword_if(self, classifier):
        """by_keyword('if') should return if tokens."""
        if_tokens = classifier.by_keyword("if")
        assert 8 in if_tokens

    def test_by_keyword_import(self, classifier):
        """by_keyword('import') should return import tokens."""
        import_tokens = classifier.by_keyword("import")
        assert 10 in import_tokens

    def test_by_keyword_nonexistent(self, classifier):
        """by_keyword() for nonexistent keyword returns empty."""
        tokens = classifier.by_keyword("nonexistent")
        assert len(tokens) == 0

    def test_all_keywords(self, classifier):
        """all_keywords() should return registered keywords."""
        keywords = classifier.all_keywords()
        assert "if" in keywords
        assert "def" in keywords
        assert "import" in keywords


class TestByBuiltinLookup:
    """Tests for by_builtin() lookups."""

    def test_by_builtin_len(self, classifier):
        """by_builtin('len') should return len tokens."""
        len_tokens = classifier.by_builtin("len")
        assert 22 in len_tokens

    def test_by_builtin_print(self, classifier):
        """by_builtin('print') should return print tokens."""
        print_tokens = classifier.by_builtin("print")
        assert 23 in print_tokens

    def test_all_builtins(self, classifier):
        """all_builtins() should return registered builtins."""
        builtins = classifier.all_builtins()
        assert "len" in builtins
        assert "print" in builtins
        assert "int" in builtins


class TestGetClassification:
    """Tests for get_classification()."""

    def test_get_classification_valid(self, classifier):
        """get_classification() should return correct classification."""
        tc = classifier.get_classification(8)  # "if"
        assert tc.token_id == 8
        assert tc.text == "if"
        assert tc.category == TokenCategory.KEYWORD
        assert tc.keyword_name == "if"

    def test_get_classification_out_of_range(self, classifier):
        """get_classification() should raise for out-of-range token."""
        with pytest.raises(ValueError):
            classifier.get_classification(1000)

    def test_get_classification_negative(self, classifier):
        """get_classification() should raise for negative token."""
        with pytest.raises(ValueError):
            classifier.get_classification(-1)


class TestMaskCreation:
    """Tests for create_mask()."""

    def test_create_mask_all_true(self, classifier):
        """Default mask should be all True."""
        mask = classifier.create_mask(32, "cpu")
        assert mask.shape == (32,)
        assert mask.all()

    def test_create_mask_block_category(self, classifier):
        """Blocking a category should set those tokens to False."""
        mask = classifier.create_mask(
            32, "cpu",
            block_categories={TokenCategory.STRING_LITERAL}
        )
        # String tokens should be blocked
        assert mask[4] is False or mask[4].item() is False  # '"hello"'
        assert mask[28] is False or mask[28].item() is False  # "'single'"
        # Other tokens should be allowed
        assert mask[2] is True or mask[2].item() is True  # "42"

    def test_create_mask_block_multiple_categories(self, classifier):
        """Blocking multiple categories should work."""
        mask = classifier.create_mask(
            32, "cpu",
            block_categories={
                TokenCategory.STRING_LITERAL,
                TokenCategory.FLOAT_LITERAL,
            }
        )
        assert mask[4] is False or mask[4].item() is False   # string
        assert mask[3] is False or mask[3].item() is False   # float
        assert mask[2] is True or mask[2].item() is True     # int (allowed)

    def test_create_mask_block_keyword(self, classifier):
        """Blocking specific keywords should work."""
        mask = classifier.create_mask(
            32, "cpu",
            block_keywords={"import", "from"}
        )
        assert mask[10] is False or mask[10].item() is False  # "import"
        assert mask[8] is True or mask[8].item() is True      # "if" (allowed)

    def test_create_mask_allow_categories(self, classifier):
        """Allow categories should only permit those categories."""
        mask = classifier.create_mask(
            32, "cpu",
            allow_categories={TokenCategory.KEYWORD}
        )
        # Keywords allowed
        assert mask[8] is True or mask[8].item() is True   # "if"
        assert mask[9] is True or mask[9].item() is True   # "def"
        # Non-keywords blocked
        assert mask[2] is False or mask[2].item() is False  # "42"
        assert mask[19] is False or mask[19].item() is False  # "foo"


class TestCategoryMask:
    """Tests for get_category_mask()."""

    def test_category_mask_int(self, classifier):
        """get_category_mask() for INT_LITERAL should work."""
        mask = classifier.get_category_mask(TokenCategory.INT_LITERAL, 32, "cpu")
        assert mask[2] is True or mask[2].item() is True   # "42"
        assert mask[26] is True or mask[26].item() is True  # "0x1F"
        assert mask[3] is False or mask[3].item() is False  # "3.14" (float)

    def test_category_mask_cached(self, classifier):
        """Category masks should be cached."""
        mask1 = classifier.get_category_mask(TokenCategory.KEYWORD, 32, "cpu")
        mask2 = classifier.get_category_mask(TokenCategory.KEYWORD, 32, "cpu")
        # Should be the same object (cached)
        assert torch.equal(mask1, mask2)


class TestStatistics:
    """Tests for get_statistics()."""

    def test_statistics_before_init(self, small_tokenizer):
        """get_statistics() before init should show not initialized."""
        clf = TokenClassifier(small_tokenizer)
        stats = clf.get_statistics()
        assert stats["initialized"] is False

    def test_statistics_after_init(self, classifier):
        """get_statistics() after init should show counts."""
        stats = classifier.get_statistics()
        assert stats["initialized"] is True
        assert stats["vocab_size"] == 32
        assert "categories" in stats
        assert stats["categories"]["KEYWORD"] > 0
        assert stats["categories"]["INT_LITERAL"] > 0


class TestGlobalCache:
    """Tests for the global classifier cache."""

    def test_get_or_create_classifier(self, small_tokenizer):
        """get_or_create_classifier() should cache classifiers."""
        clear_classifier_cache()

        clf1 = get_or_create_classifier(small_tokenizer)
        clf2 = get_or_create_classifier(small_tokenizer)

        # Should be the same object
        assert clf1 is clf2
        assert clf1.initialized

    def test_clear_cache(self, small_tokenizer):
        """clear_classifier_cache() should clear the cache."""
        clear_classifier_cache()

        clf1 = get_or_create_classifier(small_tokenizer)
        clear_classifier_cache()
        clf2 = get_or_create_classifier(small_tokenizer)

        # Should be different objects after cache clear
        assert clf1 is not clf2


class TestPythonKeywordSets:
    """Tests for Python keyword sets."""

    def test_control_keywords(self):
        """Control keywords should include if, for, while, etc."""
        assert "if" in PYTHON_CONTROL_KEYWORDS
        assert "for" in PYTHON_CONTROL_KEYWORDS
        assert "while" in PYTHON_CONTROL_KEYWORDS
        assert "break" in PYTHON_CONTROL_KEYWORDS
        assert "continue" in PYTHON_CONTROL_KEYWORDS
        assert "return" in PYTHON_CONTROL_KEYWORDS

    def test_definition_keywords(self):
        """Definition keywords should include def, class, lambda."""
        assert "def" in PYTHON_DEFINITION_KEYWORDS
        assert "class" in PYTHON_DEFINITION_KEYWORDS
        assert "lambda" in PYTHON_DEFINITION_KEYWORDS

    def test_import_keywords(self):
        """Import keywords should include import, from."""
        assert "import" in PYTHON_IMPORT_KEYWORDS
        assert "from" in PYTHON_IMPORT_KEYWORDS

    def test_all_keywords_union(self):
        """All keywords should be union of subcategories."""
        assert PYTHON_ALL_KEYWORDS >= PYTHON_CONTROL_KEYWORDS
        assert PYTHON_ALL_KEYWORDS >= PYTHON_DEFINITION_KEYWORDS
        assert PYTHON_ALL_KEYWORDS >= PYTHON_IMPORT_KEYWORDS


class TestPythonBuiltins:
    """Tests for Python builtin sets."""

    def test_type_builtins(self):
        """Type builtins should include int, str, etc."""
        assert "int" in PYTHON_BUILTINS
        assert "str" in PYTHON_BUILTINS
        assert "list" in PYTHON_BUILTINS
        assert "dict" in PYTHON_BUILTINS

    def test_function_builtins(self):
        """Function builtins should include len, print, etc."""
        assert "len" in PYTHON_BUILTINS
        assert "print" in PYTHON_BUILTINS
        assert "range" in PYTHON_BUILTINS


class TestPythonOperatorsAndDelimiters:
    """Tests for Python operator and delimiter sets."""

    def test_operators(self):
        """Common operators should be included."""
        assert "+" in PYTHON_OPERATORS
        assert "-" in PYTHON_OPERATORS
        assert "=" in PYTHON_OPERATORS
        assert "==" in PYTHON_OPERATORS
        assert "->" in PYTHON_OPERATORS

    def test_delimiters(self):
        """Common delimiters should be included."""
        assert "(" in PYTHON_DELIMITERS
        assert ")" in PYTHON_DELIMITERS
        assert "[" in PYTHON_DELIMITERS
        assert "]" in PYTHON_DELIMITERS
        assert "," in PYTHON_DELIMITERS
        assert ":" in PYTHON_DELIMITERS
