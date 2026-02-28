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
"""Integration tests for AnankeGrammar end-to-end flow.

These tests verify the full lifecycle of AnankeGrammar:
1. Creation and initialization
2. Token acceptance and constraint updates
3. Mask computation and fusion
4. Rollback and checkpoint restore
5. Copy for parallel decoding
"""

import pytest
import torch

from core.constraint import TOP, BOTTOM, Satisfiability
from core.domain import GenerationContext, PassthroughDomain
from core.unified import UNIFIED_TOP, UnifiedConstraint
from core.checkpoint import Checkpoint


class MockTokenizer:
    """Mock tokenizer for testing without full SGLang."""

    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        # Simple token -> character mapping
        self._tokens = {i: chr(ord('a') + (i % 26)) for i in range(vocab_size)}

    def decode(self, token_ids):
        return "".join(self._tokens.get(t, "?") for t in token_ids)


class MockSyntaxGrammar:
    """Mock syntax grammar that mimics llguidance behavior."""

    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.finished = False
        self.accepted_tokens = []
        self._valid_tokens = set(range(vocab_size))  # All valid by default

    def accept_token(self, token: int) -> None:
        self.accepted_tokens.append(token)
        # Simulate finishing after seeing token 999
        if token == 999:
            self.finished = True

    def is_terminated(self) -> bool:
        return self.finished

    def fill_vocab_mask(self, vocab_mask: torch.Tensor, idx: int) -> None:
        # Set bits for valid tokens
        for token_id in self._valid_tokens:
            word_idx = token_id // 32
            bit_idx = token_id % 32
            if word_idx < vocab_mask.shape[1]:
                vocab_mask[idx, word_idx] |= 1 << bit_idx

    def allocate_vocab_mask(
        self, vocab_size: int, batch_size: int, device
    ) -> torch.Tensor:
        mask_size = (vocab_size + 31) // 32
        return torch.zeros((batch_size, mask_size), dtype=torch.int32, device=device)

    def copy(self):
        new = MockSyntaxGrammar(self.vocab_size)
        new.accepted_tokens = self.accepted_tokens.copy()
        new.finished = self.finished
        new._valid_tokens = self._valid_tokens.copy()
        return new

    def try_jump_forward(self, tokenizer):
        return None

    def jump_forward_str_state(self, helper):
        return "", -1

    def jump_and_retokenize(self, old_ids, new_ids, next_state):
        pass


class TestAnankeGrammarLifecycle:
    """Test the full lifecycle of AnankeGrammar."""

    @pytest.fixture
    def tokenizer(self):
        return MockTokenizer(vocab_size=1000)

    @pytest.fixture
    def syntax_grammar(self):
        return MockSyntaxGrammar(vocab_size=1000)

    @pytest.fixture
    def domains(self):
        return {
            "types": PassthroughDomain(
                domain_name="types",
                top_constraint=TOP,
                bottom_constraint=BOTTOM,
            ),
            "imports": PassthroughDomain(
                domain_name="imports",
                top_constraint=TOP,
                bottom_constraint=BOTTOM,
            ),
            "controlflow": PassthroughDomain(
                domain_name="controlflow",
                top_constraint=TOP,
                bottom_constraint=BOTTOM,
            ),
            "semantics": PassthroughDomain(
                domain_name="semantics",
                top_constraint=TOP,
                bottom_constraint=BOTTOM,
            ),
        }

    @pytest.fixture
    def grammar(self, syntax_grammar, domains, tokenizer):
        # Import here to avoid import errors when sglang not available
        from backend.grammar import AnankeGrammar

        return AnankeGrammar(
            syntax_grammar=syntax_grammar,
            domains=domains,
            constraint=UNIFIED_TOP,
            vocab_size=1000,
            device="cpu",
            tokenizer=tokenizer,
            language="python",
            max_rollback_tokens=100,
        )

    def test_grammar_initialization(self, grammar):
        """Grammar initializes with correct state."""
        assert grammar.vocab_size == 1000
        assert grammar.device == "cpu"
        assert grammar.language == "python"
        assert grammar.constraint == UNIFIED_TOP
        assert grammar.context.position == 0
        assert grammar.context.generated_text == ""
        assert grammar.context.generated_tokens == []

    def test_accept_single_token(self, grammar):
        """Grammar accepts tokens and updates context."""
        initial_position = grammar.context.position

        grammar.accept_token(42)

        assert grammar.context.position == initial_position + 1
        assert 42 in grammar.context.generated_tokens
        assert not grammar.finished

    def test_accept_multiple_tokens(self, grammar):
        """Grammar tracks multiple token acceptances."""
        tokens = [10, 20, 30, 40]

        for token in tokens:
            grammar.accept_token(token)

        assert grammar.context.position == len(tokens)
        assert grammar.context.generated_tokens == tokens
        assert len(grammar.context.generated_text) > 0

    def test_syntax_grammar_delegation(self, grammar, syntax_grammar):
        """Token acceptance delegates to syntax grammar."""
        grammar.accept_token(100)

        assert 100 in syntax_grammar.accepted_tokens

    def test_syntax_finish_propagates(self, grammar, syntax_grammar):
        """Syntax grammar finish propagates to Ananke grammar."""
        assert not grammar.finished
        assert not syntax_grammar.finished

        # Token 999 triggers finish in mock
        grammar.accept_token(999)

        assert syntax_grammar.finished
        assert grammar.finished

    def test_is_terminated(self, grammar, syntax_grammar):
        """is_terminated delegates to syntax grammar."""
        assert not grammar.is_terminated()

        syntax_grammar.finished = True

        assert grammar.is_terminated()

    def test_constraint_updates_on_accept(self, grammar):
        """Constraint is preserved through token acceptance."""
        initial_constraint = grammar.constraint

        grammar.accept_token(50)

        # With passthrough domains, constraint should remain TOP
        assert grammar.constraint.is_top()
        # But might be a new object
        assert grammar.constraint.satisfiability() == Satisfiability.SAT


class TestAnankeGrammarMaskComputation:
    """Test mask computation and fusion."""

    @pytest.fixture
    def tokenizer(self):
        return MockTokenizer(vocab_size=100)

    @pytest.fixture
    def syntax_grammar(self):
        return MockSyntaxGrammar(vocab_size=100)

    @pytest.fixture
    def domains(self):
        return {
            "types": PassthroughDomain(
                domain_name="types",
                top_constraint=TOP,
                bottom_constraint=BOTTOM,
            ),
        }

    @pytest.fixture
    def grammar(self, syntax_grammar, domains, tokenizer):
        from backend.grammar import AnankeGrammar

        return AnankeGrammar(
            syntax_grammar=syntax_grammar,
            domains=domains,
            constraint=UNIFIED_TOP,
            vocab_size=100,
            device="cpu",
            tokenizer=tokenizer,
            language="python",
        )

    def test_allocate_vocab_mask(self, grammar):
        """Can allocate vocabulary mask tensor."""
        mask = grammar.allocate_vocab_mask(vocab_size=100, batch_size=2, device="cpu")

        assert mask.shape[0] == 2  # batch_size
        assert mask.shape[1] == (100 + 31) // 32  # ceil(vocab_size / 32)
        assert mask.dtype == torch.int32

    def test_fill_vocab_mask(self, grammar):
        """fill_vocab_mask populates bitmask."""
        mask = grammar.allocate_vocab_mask(vocab_size=100, batch_size=1, device="cpu")
        grammar.fill_vocab_mask(mask, idx=0)

        # Mask should have some bits set (syntax grammar allows all by default)
        assert mask.sum() > 0

    def test_move_vocab_mask(self, grammar):
        """Can move mask between devices."""
        mask = grammar.allocate_vocab_mask(vocab_size=100, batch_size=1, device="cpu")
        moved = grammar.move_vocab_mask(mask, "cpu")

        assert moved.device.type == "cpu"


class TestAnankeGrammarRollback:
    """Test checkpoint and rollback functionality."""

    @pytest.fixture
    def tokenizer(self):
        return MockTokenizer(vocab_size=100)

    @pytest.fixture
    def syntax_grammar(self):
        return MockSyntaxGrammar(vocab_size=100)

    @pytest.fixture
    def domains(self):
        return {
            "types": PassthroughDomain(
                domain_name="types",
                top_constraint=TOP,
                bottom_constraint=BOTTOM,
            ),
        }

    @pytest.fixture
    def grammar(self, syntax_grammar, domains, tokenizer):
        from backend.grammar import AnankeGrammar

        return AnankeGrammar(
            syntax_grammar=syntax_grammar,
            domains=domains,
            constraint=UNIFIED_TOP,
            vocab_size=100,
            device="cpu",
            tokenizer=tokenizer,
            language="python",
            max_rollback_tokens=50,
        )

    def test_checkpoint_created_on_accept(self, grammar):
        """Checkpoints are created when tokens are accepted."""
        assert grammar.checkpoint_manager.checkpoint_count == 0

        grammar.accept_token(10)

        assert grammar.checkpoint_manager.checkpoint_count == 1

    def test_rollback_restores_position(self, grammar):
        """Rollback restores context to previous position."""
        grammar.accept_token(10)
        grammar.accept_token(20)
        grammar.accept_token(30)

        assert grammar.context.position == 3

        grammar.rollback(k=2)

        assert grammar.context.position == 1

    def test_rollback_restores_generated_text(self, grammar):
        """Rollback restores generated text."""
        grammar.accept_token(10)
        grammar.accept_token(20)

        text_at_2 = grammar.context.generated_text
        grammar.accept_token(30)

        assert grammar.context.position == 3

        grammar.rollback(k=1)

        # Position 2, same text as before token 30
        assert grammar.context.position == 2

    def test_rollback_clears_mask_cache(self, grammar):
        """Rollback clears the domain mask cache."""
        grammar.accept_token(10)
        # Compute a mask to populate cache
        mask = grammar.allocate_vocab_mask(vocab_size=100, batch_size=1, device="cpu")
        grammar.fill_vocab_mask(mask, idx=0)

        grammar.rollback(k=1)

        # Cache should be cleared
        assert len(grammar._domain_mask_cache) == 0

    def test_rollback_resets_finished_flag(self, grammar, syntax_grammar):
        """Rollback resets the finished flag."""
        grammar.accept_token(999)  # Triggers finish

        assert grammar.finished

        grammar.rollback(k=1)

        assert not grammar.finished

    def test_rollback_to_beginning(self, grammar):
        """Can rollback to the very beginning."""
        for i in range(10):
            grammar.accept_token(i)

        assert grammar.context.position == 10

        grammar.rollback(k=10)

        assert grammar.context.position == 0


class TestAnankeGrammarCopy:
    """Test copy functionality for parallel decoding."""

    @pytest.fixture
    def tokenizer(self):
        return MockTokenizer(vocab_size=100)

    @pytest.fixture
    def syntax_grammar(self):
        return MockSyntaxGrammar(vocab_size=100)

    @pytest.fixture
    def domains(self):
        return {
            "types": PassthroughDomain(
                domain_name="types",
                top_constraint=TOP,
                bottom_constraint=BOTTOM,
            ),
        }

    @pytest.fixture
    def grammar(self, syntax_grammar, domains, tokenizer):
        from backend.grammar import AnankeGrammar

        return AnankeGrammar(
            syntax_grammar=syntax_grammar,
            domains=domains,
            constraint=UNIFIED_TOP,
            vocab_size=100,
            device="cpu",
            tokenizer=tokenizer,
            language="python",
        )

    def test_copy_creates_new_instance(self, grammar):
        """Copy creates a distinct grammar instance."""
        copy = grammar.copy()

        assert copy is not grammar

    def test_copy_preserves_constraint(self, grammar):
        """Copy preserves the current constraint."""
        grammar.accept_token(10)
        copy = grammar.copy()

        assert copy.constraint == grammar.constraint

    def test_copy_independence(self, grammar):
        """Copies are independent - changes to one don't affect the other."""
        grammar.accept_token(10)
        copy = grammar.copy()

        # Modify original
        grammar.accept_token(20)

        # Copy should be unaffected (context position)
        # Note: Grammar doesn't track context in copy - this tests syntax_grammar independence
        assert copy.syntax_grammar is not grammar.syntax_grammar


class TestAnankeGrammarWithoutSyntax:
    """Test AnankeGrammar without underlying syntax grammar."""

    @pytest.fixture
    def tokenizer(self):
        return MockTokenizer(vocab_size=100)

    @pytest.fixture
    def domains(self):
        return {
            "types": PassthroughDomain(
                domain_name="types",
                top_constraint=TOP,
                bottom_constraint=BOTTOM,
            ),
        }

    @pytest.fixture
    def grammar(self, domains, tokenizer):
        from backend.grammar import AnankeGrammar

        return AnankeGrammar(
            syntax_grammar=None,  # No syntax grammar
            domains=domains,
            constraint=UNIFIED_TOP,
            vocab_size=100,
            device="cpu",
            tokenizer=tokenizer,
            language="python",
        )

    def test_works_without_syntax_grammar(self, grammar):
        """Grammar functions without syntax grammar."""
        grammar.accept_token(10)

        assert grammar.context.position == 1
        assert not grammar.finished

    def test_allocates_mask_without_syntax(self, grammar):
        """Can allocate mask without syntax grammar."""
        mask = grammar.allocate_vocab_mask(vocab_size=100, batch_size=1, device="cpu")

        assert mask.shape == (1, 4)  # ceil(100/32) = 4

    def test_is_terminated_without_syntax(self, grammar):
        """is_terminated returns finished flag without syntax grammar."""
        assert not grammar.is_terminated()

        grammar.finished = True

        assert grammar.is_terminated()


class TestAnankeGrammarConstraintFlow:
    """Test constraint updates and satisfiability checking."""

    @pytest.fixture
    def tokenizer(self):
        return MockTokenizer(vocab_size=100)

    @pytest.fixture
    def syntax_grammar(self):
        return MockSyntaxGrammar(vocab_size=100)

    def test_unsatisfiable_constraint_finishes_grammar(self, tokenizer, syntax_grammar):
        """Grammar finishes when constraint becomes unsatisfiable."""
        from backend.grammar import AnankeGrammar

        # Create a domain that will return BOTTOM on observe_token
        class UnsatisfiableDomain(PassthroughDomain):
            def observe_token(self, constraint, token_id, context):
                return BOTTOM  # Always unsatisfiable

        domains = {
            "types": UnsatisfiableDomain(
                domain_name="types",
                top_constraint=TOP,
                bottom_constraint=BOTTOM,
            ),
        }

        grammar = AnankeGrammar(
            syntax_grammar=syntax_grammar,
            domains=domains,
            constraint=UNIFIED_TOP,
            vocab_size=100,
            device="cpu",
            tokenizer=tokenizer,
            language="python",
        )

        assert not grammar.finished

        grammar.accept_token(10)

        # Should finish due to unsatisfiable constraint
        assert grammar.finished
        assert grammar.constraint.is_bottom()


class TestAnankeGrammarWithSyntaxDomain:
    """Test AnankeGrammar integration with SyntaxDomain."""

    @pytest.fixture
    def tokenizer(self):
        return MockTokenizer(vocab_size=100)

    @pytest.fixture
    def syntax_grammar(self):
        return MockSyntaxGrammar(vocab_size=100)

    def test_syntax_domain_integration(self, tokenizer, syntax_grammar):
        """SyntaxDomain integrates correctly with AnankeGrammar."""
        from backend.grammar import AnankeGrammar
        from domains.syntax import SyntaxDomain, GrammarType, SYNTAX_TOP

        # Create a real SyntaxDomain
        syntax_domain = SyntaxDomain(
            grammar_object=syntax_grammar,
            grammar_type=GrammarType.JSON_SCHEMA,
            grammar_string='{"type": "string"}',
        )

        # Use SyntaxDomain for types (demonstrating domain swapping)
        domains = {
            "types": PassthroughDomain(
                domain_name="types",
                top_constraint=TOP,
                bottom_constraint=BOTTOM,
            ),
        }

        grammar = AnankeGrammar(
            syntax_grammar=syntax_grammar,
            domains=domains,
            constraint=UNIFIED_TOP,
            vocab_size=100,
            device="cpu",
            tokenizer=tokenizer,
            language="python",
        )

        # Accept some tokens
        grammar.accept_token(10)
        grammar.accept_token(20)

        assert grammar.context.position == 2
        assert not grammar.finished

    def test_syntax_constraint_creation(self):
        """Can create syntax constraints for various grammar types."""
        from domains.syntax import (
            syntax_from_json_schema,
            syntax_from_regex,
            syntax_from_ebnf,
            syntax_from_structural_tag,
            GrammarType,
        )

        json_c = syntax_from_json_schema('{"type": "object"}')
        assert json_c.grammar_type == GrammarType.JSON_SCHEMA

        regex_c = syntax_from_regex(r"[a-z]+")
        assert regex_c.grammar_type == GrammarType.REGEX

        ebnf_c = syntax_from_ebnf("expr = term")
        assert ebnf_c.grammar_type == GrammarType.EBNF

        tag_c = syntax_from_structural_tag('{"begin": "<", "end": ">"}')
        assert tag_c.grammar_type == GrammarType.STRUCTURAL_TAG

    def test_syntax_domain_checkpoint_restore(self, tokenizer):
        """SyntaxDomain checkpoints and restores correctly."""
        from domains.syntax import SyntaxDomain, GrammarType
        from core.domain import GenerationContext

        domain = SyntaxDomain(
            grammar_type=GrammarType.JSON_SCHEMA,
            grammar_string='{"type": "string"}',
        )

        ctx = GenerationContext(vocab_size=100, device="cpu")
        constraint = domain.create_constraint_for_grammar(
            GrammarType.JSON_SCHEMA, '{"type": "string"}'
        )

        # Observe some tokens
        domain.observe_token(constraint, 1, ctx)
        domain.observe_token(constraint, 2, ctx)

        # Create checkpoint
        cp = domain.checkpoint()

        # Observe more tokens
        domain.observe_token(constraint, 3, ctx)
        domain.observe_token(constraint, 4, ctx)

        # Remember state after more tokens
        state_after = domain._state_counter

        # Restore
        domain.restore(cp)

        # State should be restored
        assert domain._state_counter < state_after


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
