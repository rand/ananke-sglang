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
"""Property-based tests for speculative decoding soundness.

These tests verify that verify_draft_tokens maintains:
1. State preservation - grammar state restored after verification
2. Soundness - valid tokens are never incorrectly rejected
3. Monotonicity - accepting tokens maintains constraint validity
4. Commutativity with accept_token - verify then accept == accept only for valid

Key Invariant: Verification NEVER permanently mutates grammar state.
"""

import pytest
import sys
from pathlib import Path
from hypothesis import given, settings, assume, strategies as st, HealthCheck
import torch

# Add the ananke package root to sys.path for standalone testing
_PROPERTY_DIR = Path(__file__).parent
_TESTS_DIR = _PROPERTY_DIR.parent
_ANANKE_ROOT = _TESTS_DIR.parent
if str(_ANANKE_ROOT) not in sys.path:
    sys.path.insert(0, str(_ANANKE_ROOT))

from backend.grammar import AnankeGrammar
from core.unified import UnifiedConstraint, UNIFIED_TOP
from core.domain import GenerationContext
from domains.types.constraint import TypeConstraint


# =============================================================================
# Test Fixtures and Mocks
# =============================================================================

class MockSyntaxGrammar:
    """Mock syntax grammar with controllable valid token set."""

    def __init__(self, valid_tokens=None, vocab_size=1000):
        self.valid_tokens = valid_tokens if valid_tokens is not None else set(range(vocab_size))
        self.vocab_size = vocab_size
        self._accepted_tokens = []
        self.finished = False

    def accept_token(self, token):
        self._accepted_tokens.append(token)
        if token not in self.valid_tokens:
            self.finished = True

    def fill_vocab_mask(self, vocab_mask, idx):
        for token in self.valid_tokens:
            word_idx = token // 32
            bit_idx = token % 32
            if word_idx < vocab_mask.shape[1]:
                vocab_mask[idx, word_idx] |= (1 << bit_idx)

    def is_terminated(self):
        return self.finished

    def copy(self):
        new_grammar = MockSyntaxGrammar(
            valid_tokens=self.valid_tokens.copy(),
            vocab_size=self.vocab_size,
        )
        new_grammar._accepted_tokens = self._accepted_tokens.copy()
        new_grammar.finished = self.finished
        return new_grammar


class MockDomain:
    """Mock domain with controllable valid token set."""

    def __init__(self, valid_tokens=None, vocab_size=1000):
        self.valid_tokens = valid_tokens if valid_tokens is not None else set(range(vocab_size))
        self.vocab_size = vocab_size

    def observe_token(self, constraint, token, context):
        return constraint

    def token_mask(self, constraint, context):
        mask = torch.zeros(self.vocab_size, dtype=torch.bool)
        for t in self.valid_tokens:
            if t < self.vocab_size:
                mask[t] = True
        return mask

    def checkpoint(self):
        return {"valid_tokens": self.valid_tokens.copy()}

    def restore(self, checkpoint):
        self.valid_tokens = checkpoint["valid_tokens"]


# =============================================================================
# Strategies for generating test inputs
# =============================================================================

@st.composite
def valid_token_set(draw, vocab_size=1000, min_size=10, max_size=500):
    """Generate a set of valid token IDs."""
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    return set(draw(st.lists(
        st.integers(min_value=0, max_value=vocab_size - 1),
        min_size=size,
        max_size=size,
        unique=True,
    )))


@st.composite
def draft_token_sequence(draw, vocab_size=1000, min_length=1, max_length=20):
    """Generate a draft token sequence."""
    length = draw(st.integers(min_value=min_length, max_value=max_length))
    return draw(st.lists(
        st.integers(min_value=0, max_value=vocab_size - 1),
        min_size=length,
        max_size=length,
    ))


@st.composite
def valid_draft_sequence(draw, valid_tokens):
    """Generate a draft sequence using only valid tokens."""
    assume(len(valid_tokens) > 0)
    tokens = list(valid_tokens)
    length = draw(st.integers(min_value=1, max_value=min(20, len(tokens))))
    return [draw(st.sampled_from(tokens)) for _ in range(length)]


@st.composite
def mixed_draft_sequence(draw, valid_tokens, vocab_size=1000):
    """Generate a draft sequence with some invalid tokens."""
    assume(len(valid_tokens) > 0)
    length = draw(st.integers(min_value=2, max_value=10))
    invalid_position = draw(st.integers(min_value=0, max_value=length - 1))

    tokens = list(valid_tokens)
    invalid_tokens = list(set(range(vocab_size)) - valid_tokens)
    assume(len(invalid_tokens) > 0)

    sequence = []
    for i in range(length):
        if i == invalid_position:
            sequence.append(draw(st.sampled_from(invalid_tokens)))
        else:
            sequence.append(draw(st.sampled_from(tokens)))

    return sequence, invalid_position


# =============================================================================
# State Preservation Property Tests
# =============================================================================

class TestStatePreservationProperties:
    """Property tests for state preservation after verify_draft_tokens."""

    @given(
        draft_tokens=draft_token_sequence(vocab_size=500, max_length=10),
    )
    @settings(max_examples=50)
    def test_context_position_preserved(self, draft_tokens):
        """Context position should be unchanged after verification."""
        grammar = AnankeGrammar(
            syntax_grammar=None,
            domains={},
            vocab_size=500,
            device="cpu",
        )

        initial_position = grammar.context.position

        grammar.verify_draft_tokens(draft_tokens)

        assert grammar.context.position == initial_position, \
            "Context position was modified by verify_draft_tokens"

    @given(
        draft_tokens=draft_token_sequence(vocab_size=500, max_length=10),
    )
    @settings(max_examples=50)
    def test_generated_tokens_preserved(self, draft_tokens):
        """Generated tokens list should be unchanged after verification."""
        grammar = AnankeGrammar(
            syntax_grammar=None,
            domains={},
            vocab_size=500,
            device="cpu",
        )

        initial_tokens = grammar.context.generated_tokens.copy()

        grammar.verify_draft_tokens(draft_tokens)

        assert grammar.context.generated_tokens == initial_tokens, \
            "Generated tokens were modified by verify_draft_tokens"

    @given(
        draft_tokens=draft_token_sequence(vocab_size=500, max_length=10),
    )
    @settings(max_examples=50)
    def test_finished_flag_preserved(self, draft_tokens):
        """Finished flag should be unchanged after verification."""
        grammar = AnankeGrammar(
            syntax_grammar=None,
            domains={},
            vocab_size=500,
            device="cpu",
        )

        initial_finished = grammar.finished

        grammar.verify_draft_tokens(draft_tokens)

        assert grammar.finished == initial_finished, \
            "Finished flag was modified by verify_draft_tokens"

    @given(
        draft_tokens=draft_token_sequence(vocab_size=500, max_length=10),
    )
    @settings(max_examples=30)
    def test_constraint_preserved(self, draft_tokens):
        """Constraint object should be unchanged after verification."""
        grammar = AnankeGrammar(
            syntax_grammar=None,
            domains={},
            vocab_size=500,
            device="cpu",
        )

        initial_constraint = grammar.constraint

        grammar.verify_draft_tokens(draft_tokens)

        assert grammar.constraint == initial_constraint, \
            "Constraint was modified by verify_draft_tokens"

    @given(
        valid_tokens=valid_token_set(vocab_size=500, min_size=50, max_size=200),
        draft_tokens=draft_token_sequence(vocab_size=500, max_length=5),
    )
    @settings(max_examples=30)
    def test_syntax_grammar_state_preserved(self, valid_tokens, draft_tokens):
        """Syntax grammar state should be restored after verification."""
        syntax = MockSyntaxGrammar(valid_tokens=valid_tokens, vocab_size=500)
        grammar = AnankeGrammar(
            syntax_grammar=syntax,
            domains={},
            vocab_size=500,
            device="cpu",
        )

        initial_accepted = syntax._accepted_tokens.copy()
        initial_finished = syntax.finished

        grammar.verify_draft_tokens(draft_tokens)

        # State should be restored
        assert grammar.syntax_grammar._accepted_tokens == initial_accepted, \
            "Syntax grammar accepted tokens were modified"
        assert grammar.syntax_grammar.finished == initial_finished, \
            "Syntax grammar finished flag was modified"


# =============================================================================
# Soundness Property Tests
# =============================================================================

class TestVerificationSoundnessProperties:
    """Property tests for verification soundness.

    Key Property: Valid tokens should never be incorrectly rejected.
    """

    @given(
        valid_tokens=valid_token_set(vocab_size=500, min_size=50, max_size=200),
        data=st.data(),
    )
    @settings(max_examples=50)
    def test_all_valid_tokens_accepted(self, valid_tokens, data):
        """Sequence of only valid tokens should all be accepted."""
        assume(len(valid_tokens) >= 5)

        syntax = MockSyntaxGrammar(valid_tokens=valid_tokens, vocab_size=500)
        grammar = AnankeGrammar(
            syntax_grammar=syntax,
            domains={},
            vocab_size=500,
            device="cpu",
        )

        # Generate sequence from valid tokens only
        draft_tokens = data.draw(valid_draft_sequence(valid_tokens))

        num_valid, rejection = grammar.verify_draft_tokens(draft_tokens)

        assert num_valid == len(draft_tokens), \
            f"Expected {len(draft_tokens)} valid, got {num_valid}"
        assert rejection is None, \
            "All-valid sequence should not have rejection constraint"

    @given(
        valid_tokens=valid_token_set(vocab_size=500, min_size=50, max_size=200),
        data=st.data(),
    )
    @settings(max_examples=50)
    def test_invalid_detected_at_correct_position(self, valid_tokens, data):
        """First invalid token should be detected at correct position."""
        assume(len(valid_tokens) >= 5)
        invalid_tokens = set(range(500)) - valid_tokens
        assume(len(invalid_tokens) >= 5)

        syntax = MockSyntaxGrammar(valid_tokens=valid_tokens, vocab_size=500)
        grammar = AnankeGrammar(
            syntax_grammar=syntax,
            domains={},
            vocab_size=500,
            device="cpu",
        )

        # Generate mixed sequence with known invalid position
        result = data.draw(mixed_draft_sequence(valid_tokens, vocab_size=500))
        draft_tokens, invalid_position = result

        num_valid, rejection = grammar.verify_draft_tokens(draft_tokens)

        # Should stop at or before invalid position
        assert num_valid <= invalid_position, \
            f"Expected to stop at/before position {invalid_position}, got {num_valid}"

    @given(
        draft_tokens=st.lists(st.integers(min_value=0, max_value=499), min_size=1, max_size=10),
    )
    @settings(max_examples=50)
    def test_no_constraints_all_valid(self, draft_tokens):
        """With no constraints, all tokens should be valid."""
        grammar = AnankeGrammar(
            syntax_grammar=None,
            domains={},
            vocab_size=500,
            device="cpu",
        )

        num_valid, rejection = grammar.verify_draft_tokens(draft_tokens)

        assert num_valid == len(draft_tokens), \
            "No constraints should mean all tokens valid"
        assert rejection is None


# =============================================================================
# Monotonicity Property Tests
# =============================================================================

class TestMonotonicityProperties:
    """Property tests for monotonicity of verification.

    Accepting more tokens should not increase the set of valid continuations.
    """

    @given(
        valid_tokens=valid_token_set(vocab_size=500, min_size=50, max_size=200),
        data=st.data(),
    )
    @settings(max_examples=30)
    def test_prefix_valid_count_monotonic(self, valid_tokens, data):
        """Longer sequence should have valid count >= valid count of any prefix."""
        assume(len(valid_tokens) >= 5)

        syntax = MockSyntaxGrammar(valid_tokens=valid_tokens, vocab_size=500)
        grammar = AnankeGrammar(
            syntax_grammar=syntax,
            domains={},
            vocab_size=500,
            device="cpu",
        )

        draft_tokens = data.draw(draft_token_sequence(vocab_size=500, min_length=3, max_length=10))

        # Verify full sequence
        full_valid, _ = grammar.verify_draft_tokens(draft_tokens)

        # Verify all prefixes
        for i in range(1, len(draft_tokens)):
            prefix_valid, _ = grammar.verify_draft_tokens(draft_tokens[:i])
            # If prefix has all valid, full should have at least that many
            if prefix_valid == i:
                assert full_valid >= i, \
                    f"Full sequence valid count {full_valid} < prefix count {prefix_valid}"


# =============================================================================
# Empty and Edge Case Property Tests
# =============================================================================

class TestEdgeCaseProperties:
    """Property tests for edge cases."""

    @settings(max_examples=20)
    @given(vocab_size=st.integers(min_value=100, max_value=10000))
    def test_empty_sequence_returns_zero(self, vocab_size):
        """Empty draft sequence should return 0."""
        grammar = AnankeGrammar(
            syntax_grammar=None,
            domains={},
            vocab_size=vocab_size,
            device="cpu",
        )

        num_valid, rejection = grammar.verify_draft_tokens([])

        assert num_valid == 0
        assert rejection is None

    @settings(max_examples=20)
    @given(token=st.integers(min_value=0, max_value=999))
    def test_single_token_verification(self, token):
        """Single token verification should work."""
        grammar = AnankeGrammar(
            syntax_grammar=None,
            domains={},
            vocab_size=1000,
            device="cpu",
        )

        num_valid, rejection = grammar.verify_draft_tokens([token])

        assert num_valid in (0, 1)

    @given(
        vocab_size=st.integers(min_value=1000, max_value=100000),
        token=st.integers(min_value=0, max_value=99999),
    )
    @settings(max_examples=20)
    def test_large_vocab_support(self, vocab_size, token):
        """Large vocabulary sizes should be supported."""
        assume(token < vocab_size)

        grammar = AnankeGrammar(
            syntax_grammar=None,
            domains={},
            vocab_size=vocab_size,
            device="cpu",
        )

        num_valid, rejection = grammar.verify_draft_tokens([token])

        assert num_valid in (0, 1)


# =============================================================================
# Batch Verification Property Tests
# =============================================================================

class TestBatchVerificationProperties:
    """Property tests for verify_draft_batch."""

    @given(
        sequences=st.lists(
            draft_token_sequence(vocab_size=500, max_length=5),
            min_size=1,
            max_size=5,
        ),
    )
    @settings(max_examples=30)
    def test_batch_returns_correct_count(self, sequences):
        """Batch verification should return result for each sequence."""
        grammar = AnankeGrammar(
            syntax_grammar=None,
            domains={},
            vocab_size=500,
            device="cpu",
        )

        results = grammar.verify_draft_batch(sequences)

        assert len(results) == len(sequences), \
            "Batch should return one result per sequence"

    @given(
        sequences=st.lists(
            draft_token_sequence(vocab_size=500, max_length=5),
            min_size=1,
            max_size=5,
        ),
    )
    @settings(max_examples=30)
    def test_batch_matches_individual(self, sequences):
        """Batch results should match individual verification results."""
        grammar = AnankeGrammar(
            syntax_grammar=None,
            domains={},
            vocab_size=500,
            device="cpu",
        )

        batch_results = grammar.verify_draft_batch(sequences)

        for i, seq in enumerate(sequences):
            individual_result = grammar.verify_draft_tokens(seq)
            assert batch_results[i][0] == individual_result[0], \
                f"Batch result {i} doesn't match individual"


# =============================================================================
# Error Handling Property Tests
# =============================================================================

class TestErrorHandlingProperties:
    """Property tests for error handling soundness.

    Key Property: Errors should be handled gracefully, defaulting to PERMISSIVE (soundness).
    """

    @given(
        draft_tokens=draft_token_sequence(vocab_size=500, max_length=5),
    )
    @settings(max_examples=30)
    def test_broken_domain_handled_gracefully(self, draft_tokens):
        """Broken domain should not crash verification."""

        class BrokenDomain:
            def observe_token(self, c, t, ctx):
                return c
            def token_mask(self, c, ctx):
                raise RuntimeError("Domain error")
            def checkpoint(self):
                return {}
            def restore(self, cp):
                pass

        # Use non-TOP constraint to trigger domain checking
        non_top_constraint = UnifiedConstraint(
            types=TypeConstraint(expected_type=None),
        )
        grammar = AnankeGrammar(
            syntax_grammar=None,
            domains={"types": BrokenDomain()},
            constraint=non_top_constraint,
            vocab_size=500,
            device="cpu",
        )

        # Should not crash - should handle gracefully
        num_valid, rejection = grammar.verify_draft_tokens(draft_tokens)

        # Should default to permissive (soundness)
        assert num_valid >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
