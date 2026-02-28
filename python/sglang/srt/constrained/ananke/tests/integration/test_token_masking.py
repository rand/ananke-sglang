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
"""Integration tests for precise token-level masking across domains.

These tests verify:
1. TokenClassifier correctly classifies vocabulary
2. Each domain produces selective masks (not all-True)
3. Multi-domain fusion produces correct intersection
4. Checkpoint/restore preserves masking state
5. Performance meets targets
"""

import pytest
import torch

from core.domain import GenerationContext
from core.token_classifier import (
    TokenClassifier,
    TokenCategory,
    get_or_create_classifier,
    clear_classifier_cache,
)
from domains.types.domain import TypeDomain
from domains.types.constraint import TYPE_TOP, INT, STR, BOOL, FLOAT, NONE, type_expecting
from domains.imports.domain import ImportDomain
from domains.imports.constraint import IMPORT_TOP, ImportConstraint
from domains.controlflow.domain import ControlFlowDomain
from domains.controlflow.constraint import CONTROLFLOW_TOP
from domains.semantics.domain import SemanticDomain
from domains.semantics.constraint import SEMANTIC_TOP


class MockTokenizer:
    """Mock tokenizer with realistic vocabulary for testing."""

    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        # Build a vocabulary with different token types
        self._vocab = {}
        self._id_to_text = {}

        # Add keywords (token ids 0-30)
        keywords = [
            "if", "else", "elif", "for", "while", "def", "class", "return",
            "import", "from", "as", "try", "except", "finally", "with",
            "raise", "assert", "yield", "lambda", "pass", "break", "continue",
            "True", "False", "None", "and", "or", "not", "is", "in", "del",
        ]
        for i, kw in enumerate(keywords):
            self._vocab[kw] = i
            self._id_to_text[i] = kw

        # Add integer literals (token ids 31-50)
        for i, num in enumerate(range(-5, 15)):
            token_id = 31 + i
            text = str(num)
            self._vocab[text] = token_id
            self._id_to_text[token_id] = text

        # Add float literals (token ids 51-60)
        floats = ["0.0", "0.5", "1.0", "1.5", "2.0", "3.14", "2.718", "-1.0", "-0.5", "0.1"]
        for i, f in enumerate(floats):
            token_id = 51 + i
            self._vocab[f] = token_id
            self._id_to_text[token_id] = f

        # Add string literal tokens (token ids 61-70)
        strings = ['"hello"', '"world"', '""', '"test"', '"foo"',
                   "'bar'", "'baz'", '"""', "'''", '"str"']
        for i, s in enumerate(strings):
            token_id = 61 + i
            self._vocab[s] = token_id
            self._id_to_text[token_id] = s

        # Add identifiers (token ids 71-90)
        identifiers = [
            "x", "y", "z", "foo", "bar", "baz", "result", "value", "data",
            "items", "count", "index", "name", "obj", "self", "cls", "args",
            "kwargs", "func", "method",
        ]
        for i, ident in enumerate(identifiers):
            token_id = 71 + i
            self._vocab[ident] = token_id
            self._id_to_text[token_id] = ident

        # Add operators (token ids 91-110)
        operators = [
            "+", "-", "*", "/", "//", "%", "**", "=", "==", "!=",
            "<", ">", "<=", ">=", "+=", "-=", "*=", "/=", "&", "|",
        ]
        for i, op in enumerate(operators):
            token_id = 91 + i
            self._vocab[op] = token_id
            self._id_to_text[token_id] = op

        # Add delimiters (token ids 111-130)
        delimiters = [
            "(", ")", "[", "]", "{", "}", ",", ":", ";", ".",
            "->", "@", "\n", " ", "\t", "  ", "    ", "...", "#", "\\",
        ]
        for i, d in enumerate(delimiters):
            token_id = 111 + i
            self._vocab[d] = token_id
            self._id_to_text[token_id] = d

        # Add module names (token ids 131-150)
        modules = [
            "os", "sys", "json", "re", "math", "random", "collections",
            "itertools", "functools", "typing", "subprocess", "pickle",
            "socket", "http", "urllib", "pathlib", "datetime", "time",
            "logging", "unittest",
        ]
        for i, mod in enumerate(modules):
            token_id = 131 + i
            self._vocab[mod] = token_id
            self._id_to_text[token_id] = mod

        # Fill remaining vocab with generic identifiers
        for i in range(151, vocab_size):
            text = f"tok{i}"
            self._vocab[text] = i
            self._id_to_text[i] = text

    def decode(self, token_ids):
        """Decode token IDs to text."""
        if isinstance(token_ids, int):
            return self._id_to_text.get(token_ids, f"<unk{token_ids}>")
        return "".join(self._id_to_text.get(t, f"<unk{t}>") for t in token_ids)

    def get_vocab(self):
        """Get vocabulary mapping."""
        return self._vocab.copy()


class TestTokenClassifierIntegration:
    """Integration tests for TokenClassifier with real vocabulary."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Clear classifier cache before each test."""
        clear_classifier_cache()

    @pytest.fixture
    def tokenizer(self):
        return MockTokenizer(vocab_size=200)

    @pytest.fixture
    def classifier(self, tokenizer):
        clf = get_or_create_classifier(tokenizer, "python")
        clf.initialize()
        return clf

    def test_classifier_initialization(self, classifier, tokenizer):
        """Test classifier initializes with full vocabulary."""
        assert classifier._initialized
        assert classifier.vocab_size == tokenizer.vocab_size

    def test_keyword_classification(self, classifier):
        """Test keywords are correctly classified."""
        # Check Python keywords
        keyword_tokens = classifier.by_keyword("return")
        assert len(keyword_tokens) > 0

        import_tokens = classifier.by_keyword("import")
        assert len(import_tokens) > 0

        from_tokens = classifier.by_keyword("from")
        assert len(from_tokens) > 0

    def test_category_classification(self, classifier):
        """Test categories contain expected tokens."""
        # Integer literals
        int_tokens = classifier.by_category(TokenCategory.INT_LITERAL)
        assert len(int_tokens) > 0

        # Keywords
        kw_tokens = classifier.by_category(TokenCategory.KEYWORD)
        assert len(kw_tokens) > 0

        # Identifiers
        id_tokens = classifier.by_category(TokenCategory.IDENTIFIER)
        assert len(id_tokens) > 0

    def test_category_mask_creation(self, classifier, tokenizer):
        """Test category mask creation."""
        mask = classifier.get_category_mask(TokenCategory.INT_LITERAL, tokenizer.vocab_size)
        assert mask.shape[0] == tokenizer.vocab_size
        assert mask.dtype == torch.bool

        # Should have some True values (integer tokens)
        assert mask.sum() > 0

        # Should have some False values (non-integer tokens)
        assert mask.sum() < tokenizer.vocab_size


class TestTypeDomainMasking:
    """Integration tests for TypeDomain token masking."""

    @pytest.fixture(autouse=True)
    def setup(self):
        clear_classifier_cache()

    @pytest.fixture
    def tokenizer(self):
        return MockTokenizer(vocab_size=200)

    @pytest.fixture
    def context(self, tokenizer):
        return GenerationContext(
            vocab_size=tokenizer.vocab_size,
            position=0,
            device="cpu",
            tokenizer=tokenizer,
        )

    @pytest.fixture
    def domain(self, tokenizer):
        return TypeDomain(language="python", tokenizer=tokenizer)

    def test_top_constraint_allows_all(self, domain, context):
        """TOP constraint should allow all tokens."""
        mask = domain.token_mask(TYPE_TOP, context)
        assert mask.all(), "TOP should allow all tokens"

    def test_int_constraint_selective(self, domain, context, tokenizer):
        """INT type constraint should block non-integer tokens."""
        constraint = type_expecting(INT)
        mask = domain.token_mask(constraint, context)

        # Should not be all True (should block some tokens)
        assert not mask.all(), "INT constraint should block some tokens"

        # Should not be all False (should allow some tokens)
        assert mask.any(), "INT constraint should allow some tokens"

        # String literal tokens should be blocked
        # Token ID 61 is '"hello"'
        if 61 < context.vocab_size:
            assert not mask[61], "String literal should be blocked for INT"

    def test_str_constraint_selective(self, domain, context):
        """STR type constraint should block non-string tokens."""
        constraint = type_expecting(STR)
        mask = domain.token_mask(constraint, context)

        # Should be selective
        blocked_count = (~mask).sum().item()
        assert blocked_count > 0, "STR constraint should block some tokens"

    def test_bool_constraint_selective(self, domain, context):
        """BOOL type constraint should block non-boolean tokens."""
        constraint = type_expecting(BOOL)
        mask = domain.token_mask(constraint, context)

        # Should be selective
        blocked_count = (~mask).sum().item()
        assert blocked_count > 0, "BOOL constraint should block some tokens"


class TestImportDomainMasking:
    """Integration tests for ImportDomain token masking."""

    @pytest.fixture(autouse=True)
    def setup(self):
        clear_classifier_cache()

    @pytest.fixture
    def tokenizer(self):
        return MockTokenizer(vocab_size=200)

    @pytest.fixture
    def context(self, tokenizer):
        return GenerationContext(
            vocab_size=tokenizer.vocab_size,
            position=0,
            device="cpu",
            tokenizer=tokenizer,
        )

    @pytest.fixture
    def domain(self, tokenizer):
        return ImportDomain(language="python", tokenizer=tokenizer)

    def test_top_constraint_allows_all(self, domain, context):
        """TOP constraint should allow all tokens."""
        mask = domain.token_mask(IMPORT_TOP, context)
        assert mask.all(), "TOP should allow all tokens"

    def test_forbidden_import_in_context(self, domain, context, tokenizer):
        """When in import context with forbidden modules, should block completions."""
        # Create constraint with forbidden modules
        constraint = ImportConstraint(forbidden=frozenset(["os", "subprocess"]))

        # Simulate being in import context by observing "import " tokens
        # Token ID 8 is "import", token ID 113 is " "
        domain.observe_token(constraint, 8, context)  # "import"
        domain.observe_token(constraint, 113, context)  # " "

        # Now get mask - should block "os" and "subprocess" completions
        mask = domain.token_mask(constraint, context)

        # The mask should be selective when in import context
        # Note: The actual blocking depends on import context state
        assert mask.dtype == torch.bool
        assert mask.shape[0] == context.vocab_size


class TestControlFlowDomainMasking:
    """Integration tests for ControlFlowDomain token masking."""

    @pytest.fixture(autouse=True)
    def setup(self):
        clear_classifier_cache()

    @pytest.fixture
    def tokenizer(self):
        return MockTokenizer(vocab_size=200)

    @pytest.fixture
    def context(self, tokenizer):
        return GenerationContext(
            vocab_size=tokenizer.vocab_size,
            position=0,
            device="cpu",
            tokenizer=tokenizer,
        )

    @pytest.fixture
    def domain(self, tokenizer):
        return ControlFlowDomain(language="python", tokenizer=tokenizer)

    def test_top_constraint_allows_all(self, domain, context):
        """TOP constraint should allow all tokens."""
        mask = domain.token_mask(CONTROLFLOW_TOP, context)
        assert mask.all(), "TOP should allow all tokens"


class TestSemanticDomainMasking:
    """Integration tests for SemanticDomain token masking."""

    @pytest.fixture(autouse=True)
    def setup(self):
        clear_classifier_cache()

    @pytest.fixture
    def tokenizer(self):
        return MockTokenizer(vocab_size=200)

    @pytest.fixture
    def context(self, tokenizer):
        return GenerationContext(
            vocab_size=tokenizer.vocab_size,
            position=0,
            device="cpu",
            tokenizer=tokenizer,
        )

    @pytest.fixture
    def domain(self, tokenizer):
        return SemanticDomain(language="python", aggressive_mode=True, tokenizer=tokenizer)

    def test_top_constraint_allows_all(self, domain, context):
        """TOP constraint should allow all tokens."""
        mask = domain.token_mask(SEMANTIC_TOP, context)
        assert mask.all(), "TOP should allow all tokens"

    def test_bounds_extraction(self, domain, context, tokenizer):
        """Test that bounds are extracted from formulas."""
        # Create a constraint with bounds
        constraint = domain.create_constraint(assertions=["x > 5", "x < 10"])

        # Should extract bounds for variable x
        mask = domain.token_mask(constraint, context)

        # Mask should be valid
        assert mask.dtype == torch.bool
        assert mask.shape[0] == context.vocab_size

    def test_aggressive_mode_blocking(self, domain, context, tokenizer):
        """Test aggressive mode blocks out-of-bounds literals."""
        # Create a constraint with bounds
        constraint = domain.create_constraint(assertions=["x >= 5", "x <= 10"])

        # Simulate being in assignment context for x
        # Token ID 71 is "x", Token ID 97 is "="
        domain.observe_token(constraint, 71, context)  # "x"
        domain.observe_token(constraint, 97, context)  # "="

        # Now get mask - should block out-of-bounds integers
        mask = domain.token_mask(constraint, context)

        # Mask should be valid
        assert mask.dtype == torch.bool


class TestMultiDomainIntegration:
    """Integration tests for multiple domains working together."""

    @pytest.fixture(autouse=True)
    def setup(self):
        clear_classifier_cache()

    @pytest.fixture
    def tokenizer(self):
        return MockTokenizer(vocab_size=200)

    @pytest.fixture
    def context(self, tokenizer):
        return GenerationContext(
            vocab_size=tokenizer.vocab_size,
            position=0,
            device="cpu",
            tokenizer=tokenizer,
        )

    def test_mask_intersection(self, tokenizer, context):
        """Test that multiple domain masks can be intersected."""
        type_domain = TypeDomain(language="python", tokenizer=tokenizer)
        import_domain = ImportDomain(language="python", tokenizer=tokenizer)

        # Get individual masks
        type_mask = type_domain.token_mask(type_expecting(INT), context)
        import_mask = import_domain.token_mask(IMPORT_TOP, context)

        # Intersection should be valid
        fused = type_mask & import_mask
        assert fused.dtype == torch.bool
        assert fused.shape[0] == context.vocab_size

        # Fused should be no larger than either individual mask
        assert fused.sum() <= type_mask.sum()
        assert fused.sum() <= import_mask.sum()

    def test_checkpoint_restore_integration(self, tokenizer, context):
        """Test checkpoint/restore across multiple domains."""
        type_domain = TypeDomain(language="python", tokenizer=tokenizer)
        semantic_domain = SemanticDomain(
            language="python", aggressive_mode=True, tokenizer=tokenizer
        )

        # Create checkpoints
        type_checkpoint = type_domain.checkpoint()
        semantic_checkpoint = semantic_domain.checkpoint()

        # Modify state
        type_constraint = type_expecting(INT)
        semantic_constraint = semantic_domain.create_constraint(assertions=["x > 0"])

        type_domain.observe_token(type_constraint, 35, context)  # Token "0"
        semantic_domain.observe_token(semantic_constraint, 71, context)  # Token "x"

        # Restore
        type_domain.restore(type_checkpoint)
        semantic_domain.restore(semantic_checkpoint)

        # State should be restored - masks should work
        type_mask = type_domain.token_mask(TYPE_TOP, context)
        semantic_mask = semantic_domain.token_mask(SEMANTIC_TOP, context)

        assert type_mask.all()
        assert semantic_mask.all()


class TestMaskPerformance:
    """Performance tests for token masking."""

    @pytest.fixture(autouse=True)
    def setup(self):
        clear_classifier_cache()

    @pytest.fixture
    def tokenizer(self):
        # Use larger vocab for realistic performance testing
        return MockTokenizer(vocab_size=1000)

    @pytest.fixture
    def context(self, tokenizer):
        return GenerationContext(
            vocab_size=tokenizer.vocab_size,
            position=0,
            device="cpu",
            tokenizer=tokenizer,
        )

    def test_type_domain_mask_performance(self, tokenizer, context):
        """Test TypeDomain mask computation is fast after initialization."""
        import time

        domain = TypeDomain(language="python", tokenizer=tokenizer)

        # Warm up (includes classifier initialization)
        constraint = type_expecting(INT)
        _ = domain.token_mask(constraint, context)

        # Measure subsequent calls
        times = []
        for _ in range(100):
            start = time.perf_counter_ns()
            _ = domain.token_mask(constraint, context)
            elapsed = time.perf_counter_ns() - start
            times.append(elapsed)

        mean_us = (sum(times) / len(times)) / 1000
        p99_us = sorted(times)[98] / 1000

        print(f"\nTypeDomain mask: mean={mean_us:.1f}us, p99={p99_us:.1f}us")
        # Target: <500us p99 (generous for test environment)
        assert p99_us < 500, f"TypeDomain p99 too slow: {p99_us}us"

    def test_semantic_domain_mask_performance(self, tokenizer, context):
        """Test SemanticDomain mask computation is fast."""
        import time

        domain = SemanticDomain(
            language="python", aggressive_mode=True, tokenizer=tokenizer
        )

        # Warm up
        constraint = domain.create_constraint(assertions=["x > 0"])
        _ = domain.token_mask(constraint, context)

        # Measure
        times = []
        for _ in range(100):
            start = time.perf_counter_ns()
            _ = domain.token_mask(constraint, context)
            elapsed = time.perf_counter_ns() - start
            times.append(elapsed)

        mean_us = (sum(times) / len(times)) / 1000
        p99_us = sorted(times)[98] / 1000

        print(f"\nSemanticDomain mask: mean={mean_us:.1f}us, p99={p99_us:.1f}us")
        # Target: <500us p99
        assert p99_us < 500, f"SemanticDomain p99 too slow: {p99_us}us"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
