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
"""Tests for speculative decoding support in AnankeGrammar.

Tests the verify_draft_tokens functionality for speculative decoding,
which verifies draft model tokens against constraints.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

# Setup import paths
_TEST_DIR = Path(__file__).parent
_ANANKE_ROOT = _TEST_DIR.parent.parent
if str(_ANANKE_ROOT) not in sys.path:
    sys.path.insert(0, str(_ANANKE_ROOT))

from backend.grammar import AnankeGrammar
from core.unified import UnifiedConstraint, UNIFIED_TOP, UNIFIED_BOTTOM
from core.domain import GenerationContext
from domains.types.constraint import TYPE_TOP, TYPE_BOTTOM, TypeConstraint


class MockDomain:
    """Mock domain for testing."""

    def __init__(self, valid_tokens=None):
        self.valid_tokens = valid_tokens or set(range(1000))

    def observe_token(self, constraint, token, context):
        return constraint

    def token_mask(self, constraint, context):
        mask = torch.zeros(1000, dtype=torch.bool)
        for t in self.valid_tokens:
            if t < 1000:
                mask[t] = True
        return mask

    def checkpoint(self):
        return {}

    def restore(self, checkpoint):
        pass


class MockConstrainedDomain:
    """Mock domain that respects non-TOP constraints."""

    def __init__(self, valid_tokens=None):
        self.valid_tokens = valid_tokens or set(range(1000))

    def observe_token(self, constraint, token, context):
        return constraint

    def token_mask(self, constraint, context):
        # Always compute mask based on valid_tokens (not the constraint)
        # This simulates a domain that has its own token restrictions
        mask = torch.zeros(1000, dtype=torch.bool)
        for t in self.valid_tokens:
            if t < 1000:
                mask[t] = True
        return mask

    def checkpoint(self):
        return {}

    def restore(self, checkpoint):
        pass


class MockSyntaxGrammar:
    """Mock syntax grammar for testing."""

    def __init__(self, valid_tokens=None):
        self.valid_tokens = valid_tokens or set(range(1000))
        self.finished = False
        self._accepted_tokens = []

    def accept_token(self, token):
        self._accepted_tokens.append(token)
        if token not in self.valid_tokens:
            self.finished = True

    def fill_vocab_mask(self, vocab_mask, idx):
        # Set all valid tokens in the mask
        for token in self.valid_tokens:
            word_idx = token // 32
            bit_idx = token % 32
            if word_idx < vocab_mask.shape[1]:
                vocab_mask[idx, word_idx] |= (1 << bit_idx)

    def is_terminated(self):
        return self.finished

    def copy(self):
        new_grammar = MockSyntaxGrammar(self.valid_tokens.copy())
        new_grammar._accepted_tokens = self._accepted_tokens.copy()
        new_grammar.finished = self.finished
        return new_grammar


class TestVerifyDraftTokens:
    """Tests for verify_draft_tokens method."""

    def test_empty_draft_returns_zero(self):
        """Test that empty draft sequence returns 0."""
        grammar = AnankeGrammar(
            syntax_grammar=None,
            domains={},
            vocab_size=1000,
            device="cpu",
        )

        num_valid, rejection = grammar.verify_draft_tokens([])

        assert num_valid == 0
        assert rejection is None

    def test_all_tokens_valid(self):
        """Test when all draft tokens are valid."""
        syntax = MockSyntaxGrammar(valid_tokens=set(range(100)))
        grammar = AnankeGrammar(
            syntax_grammar=syntax,
            domains={},
            vocab_size=1000,
            device="cpu",
        )

        draft_tokens = [1, 2, 3, 4, 5]
        num_valid, rejection = grammar.verify_draft_tokens(draft_tokens)

        assert num_valid == 5
        assert rejection is None

    def test_some_tokens_invalid(self):
        """Test when some draft tokens are invalid."""
        # Only tokens 0-49 are valid
        syntax = MockSyntaxGrammar(valid_tokens=set(range(50)))
        grammar = AnankeGrammar(
            syntax_grammar=syntax,
            domains={},
            vocab_size=1000,
            device="cpu",
        )

        # Token 60 is invalid
        draft_tokens = [1, 2, 3, 60, 5]
        num_valid, rejection = grammar.verify_draft_tokens(draft_tokens)

        assert num_valid == 3
        assert rejection is not None

    def test_first_token_invalid(self):
        """Test when first token is invalid."""
        syntax = MockSyntaxGrammar(valid_tokens={10, 20, 30})
        grammar = AnankeGrammar(
            syntax_grammar=syntax,
            domains={},
            vocab_size=1000,
            device="cpu",
        )

        # Token 1 is not in valid set
        draft_tokens = [1, 2, 3]
        num_valid, rejection = grammar.verify_draft_tokens(draft_tokens)

        assert num_valid == 0
        assert rejection is not None

    def test_state_restored_after_verification(self):
        """Test that grammar state is restored after verification."""
        syntax = MockSyntaxGrammar(valid_tokens=set(range(100)))
        grammar = AnankeGrammar(
            syntax_grammar=syntax,
            domains={},
            vocab_size=1000,
            device="cpu",
        )

        # Save initial state
        initial_position = grammar.context.position
        initial_finished = grammar.finished

        # Verify some tokens
        draft_tokens = [1, 2, 3, 4, 5]
        grammar.verify_draft_tokens(draft_tokens)

        # State should be restored
        assert grammar.context.position == initial_position
        assert grammar.finished == initial_finished

    def test_domain_constraint_validation(self):
        """Test that domain constraints are checked."""
        # Domain only allows tokens 0-29
        domain = MockConstrainedDomain(valid_tokens=set(range(30)))

        # Create grammar with non-TOP type constraint
        non_top_constraint = UnifiedConstraint(
            types=TypeConstraint(expected_type=None),  # Non-TOP constraint
        )
        grammar = AnankeGrammar(
            syntax_grammar=None,
            domains={"types": domain},
            constraint=non_top_constraint,
            vocab_size=1000,
            device="cpu",
        )

        # Token 50 violates domain constraint
        draft_tokens = [1, 2, 50, 4]
        num_valid, rejection = grammar.verify_draft_tokens(draft_tokens)

        assert num_valid == 2  # First two are valid
        assert rejection is not None


class TestVerifyDraftBatch:
    """Tests for verify_draft_batch method."""

    def test_batch_verification(self):
        """Test batch verification of multiple sequences."""
        syntax = MockSyntaxGrammar(valid_tokens=set(range(50)))
        grammar = AnankeGrammar(
            syntax_grammar=syntax,
            domains={},
            vocab_size=1000,
            device="cpu",
        )

        draft_sequences = [
            [1, 2, 3],      # All valid
            [1, 100, 3],    # Second invalid
            [100, 1, 2],    # First invalid
        ]

        results = grammar.verify_draft_batch(draft_sequences)

        assert len(results) == 3
        assert results[0][0] == 3  # All valid
        assert results[1][0] == 1  # One valid
        assert results[2][0] == 0  # None valid


class TestSpeculativeStats:
    """Tests for speculative decoding statistics."""

    def test_stats_with_syntax_grammar(self):
        """Test stats when syntax grammar is available."""
        syntax = MockSyntaxGrammar()
        grammar = AnankeGrammar(
            syntax_grammar=syntax,
            domains={"types": MockDomain()},
            vocab_size=1000,
            device="cpu",
        )

        stats = grammar.get_speculative_stats()

        assert stats["supported"] is True
        assert stats["syntax_grammar_available"] is True
        assert "types" in stats["active_domains"]

    def test_stats_without_syntax_grammar(self):
        """Test stats when syntax grammar is not available."""
        grammar = AnankeGrammar(
            syntax_grammar=None,
            domains={},
            vocab_size=1000,
            device="cpu",
        )

        stats = grammar.get_speculative_stats()

        assert stats["supported"] is True
        assert stats["syntax_grammar_available"] is False


class TestIsTokenValidForDraft:
    """Tests for _is_token_valid_for_draft method."""

    def test_valid_token(self):
        """Test that valid token returns True."""
        syntax = MockSyntaxGrammar(valid_tokens=set(range(100)))
        grammar = AnankeGrammar(
            syntax_grammar=syntax,
            domains={},
            vocab_size=1000,
            device="cpu",
        )

        assert grammar._is_token_valid_for_draft(50) is True

    def test_invalid_token(self):
        """Test that invalid token returns False."""
        syntax = MockSyntaxGrammar(valid_tokens=set(range(50)))
        grammar = AnankeGrammar(
            syntax_grammar=syntax,
            domains={},
            vocab_size=1000,
            device="cpu",
        )

        assert grammar._is_token_valid_for_draft(100) is False

    def test_domain_invalid_token(self):
        """Test that domain-invalid token returns False."""
        domain = MockConstrainedDomain(valid_tokens=set(range(30)))

        # Create grammar with non-TOP type constraint
        non_top_constraint = UnifiedConstraint(
            types=TypeConstraint(expected_type=None),  # Non-TOP constraint
        )
        grammar = AnankeGrammar(
            syntax_grammar=None,
            domains={"types": domain},
            constraint=non_top_constraint,
            vocab_size=1000,
            device="cpu",
        )

        # Token 50 is not in domain's valid set
        assert grammar._is_token_valid_for_draft(50) is False

    def test_no_constraints_all_valid(self):
        """Test that tokens are valid when no constraints."""
        grammar = AnankeGrammar(
            syntax_grammar=None,
            domains={},
            vocab_size=1000,
            device="cpu",
        )

        # Any token should be valid
        assert grammar._is_token_valid_for_draft(500) is True


class TestStatePreservation:
    """Tests for state preservation during speculative verification."""

    def test_constraint_state_preserved(self):
        """Test that constraint state is preserved."""
        grammar = AnankeGrammar(
            syntax_grammar=None,
            domains={},
            vocab_size=1000,
            device="cpu",
        )

        initial_constraint = grammar.constraint

        grammar.verify_draft_tokens([1, 2, 3])

        assert grammar.constraint == initial_constraint

    def test_context_state_preserved(self):
        """Test that context state is preserved."""
        grammar = AnankeGrammar(
            syntax_grammar=None,
            domains={},
            vocab_size=1000,
            device="cpu",
        )

        initial_position = grammar.context.position
        initial_tokens = grammar.context.generated_tokens.copy()

        grammar.verify_draft_tokens([1, 2, 3])

        assert grammar.context.position == initial_position
        assert grammar.context.generated_tokens == initial_tokens

    def test_mask_cache_preserved(self):
        """Test that mask cache is preserved."""
        grammar = AnankeGrammar(
            syntax_grammar=None,
            domains={},
            vocab_size=1000,
            device="cpu",
        )

        # Pre-populate cache
        grammar._domain_mask_cache["test"] = torch.ones(100)
        initial_cache = grammar._domain_mask_cache.copy()

        grammar.verify_draft_tokens([1, 2, 3])

        assert list(grammar._domain_mask_cache.keys()) == list(initial_cache.keys())


class TestEdgeCases:
    """Tests for edge cases in speculative decoding."""

    def test_single_token_draft(self):
        """Test verification of single token draft."""
        syntax = MockSyntaxGrammar(valid_tokens=set(range(100)))
        grammar = AnankeGrammar(
            syntax_grammar=syntax,
            domains={},
            vocab_size=1000,
            device="cpu",
        )

        num_valid, rejection = grammar.verify_draft_tokens([50])

        assert num_valid == 1
        assert rejection is None

    def test_large_token_ids(self):
        """Test handling of large token IDs."""
        grammar = AnankeGrammar(
            syntax_grammar=None,
            domains={},
            vocab_size=100000,
            device="cpu",
        )

        # Large token ID should work
        num_valid, rejection = grammar.verify_draft_tokens([99999])

        assert num_valid == 1

    def test_error_handling_in_domain_check(self):
        """Test that errors in domain check don't crash."""
        class BrokenDomain:
            def observe_token(self, c, t, ctx):
                return c
            def token_mask(self, c, ctx):
                raise RuntimeError("Domain error")
            def checkpoint(self):
                return {}
            def restore(self, cp):
                pass

        # Use "types" as a valid domain name with a broken implementation
        # and a non-TOP constraint to trigger domain checking
        non_top_constraint = UnifiedConstraint(
            types=TypeConstraint(expected_type=None),  # Non-TOP constraint
        )
        grammar = AnankeGrammar(
            syntax_grammar=None,
            domains={"types": BrokenDomain()},
            constraint=non_top_constraint,
            vocab_size=1000,
            device="cpu",
        )

        # Should not crash, and should assume valid (soundness)
        num_valid, rejection = grammar.verify_draft_tokens([1, 2, 3])

        # All should be considered valid due to error handling
        assert num_valid == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
