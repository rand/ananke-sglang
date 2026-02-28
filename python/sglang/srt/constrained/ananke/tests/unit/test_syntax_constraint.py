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
"""Unit tests for SyntaxConstraint and syntax domain."""

import pytest
import torch

from core.constraint import Satisfiability
from domains.syntax.constraint import (
    SYNTAX_BOTTOM,
    SYNTAX_TOP,
    GrammarType,
    SyntaxConstraint,
    syntax_from_ebnf,
    syntax_from_json_schema,
    syntax_from_regex,
    syntax_from_structural_tag,
)
from domains.syntax.domain import SyntaxDomain
from core.domain import GenerationContext


class TestSyntaxConstraintBasics:
    """Basic tests for SyntaxConstraint."""

    def test_syntax_top_is_top(self):
        """SYNTAX_TOP is the top element."""
        assert SYNTAX_TOP.is_top()
        assert not SYNTAX_TOP.is_bottom()

    def test_syntax_bottom_is_bottom(self):
        """SYNTAX_BOTTOM is the bottom element."""
        assert SYNTAX_BOTTOM.is_bottom()
        assert not SYNTAX_BOTTOM.is_top()

    def test_syntax_top_satisfiability(self):
        """SYNTAX_TOP is satisfiable."""
        assert SYNTAX_TOP.satisfiability() == Satisfiability.SAT

    def test_syntax_bottom_satisfiability(self):
        """SYNTAX_BOTTOM is unsatisfiable."""
        assert SYNTAX_BOTTOM.satisfiability() == Satisfiability.UNSAT

    def test_create_json_constraint(self):
        """Can create constraint from JSON schema."""
        schema = '{"type": "object", "properties": {"name": {"type": "string"}}}'
        constraint = syntax_from_json_schema(schema)

        assert constraint.grammar_type == GrammarType.JSON_SCHEMA
        assert constraint.grammar_string == schema
        assert not constraint.is_top()
        assert not constraint.is_bottom()

    def test_create_regex_constraint(self):
        """Can create constraint from regex."""
        pattern = r"[a-z]+@[a-z]+\.[a-z]+"
        constraint = syntax_from_regex(pattern)

        assert constraint.grammar_type == GrammarType.REGEX
        assert constraint.grammar_string == pattern

    def test_create_ebnf_constraint(self):
        """Can create constraint from EBNF grammar."""
        grammar = "expr = term (('+' | '-') term)*"
        constraint = syntax_from_ebnf(grammar)

        assert constraint.grammar_type == GrammarType.EBNF
        assert constraint.grammar_string == grammar

    def test_create_structural_tag_constraint(self):
        """Can create constraint from structural tag."""
        tag = '{"begin": "<code>", "end": "</code>"}'
        constraint = syntax_from_structural_tag(tag)

        assert constraint.grammar_type == GrammarType.STRUCTURAL_TAG
        assert constraint.grammar_string == tag


class TestSyntaxConstraintSemilattice:
    """Tests for semilattice laws."""

    def test_meet_identity_top(self):
        """c ⊓ ⊤ = c (identity law)."""
        schema = '{"type": "string"}'
        c = syntax_from_json_schema(schema)

        result = c.meet(SYNTAX_TOP)

        assert result == c

    def test_meet_identity_top_reverse(self):
        """⊤ ⊓ c = c (identity law, reversed)."""
        schema = '{"type": "string"}'
        c = syntax_from_json_schema(schema)

        result = SYNTAX_TOP.meet(c)

        assert result == c

    def test_meet_annihilation_bottom(self):
        """c ⊓ ⊥ = ⊥ (annihilation law)."""
        c = syntax_from_json_schema('{"type": "string"}')

        result = c.meet(SYNTAX_BOTTOM)

        assert result == SYNTAX_BOTTOM

    def test_meet_annihilation_bottom_reverse(self):
        """⊥ ⊓ c = ⊥ (annihilation law, reversed)."""
        c = syntax_from_json_schema('{"type": "string"}')

        result = SYNTAX_BOTTOM.meet(c)

        assert result == SYNTAX_BOTTOM

    def test_meet_idempotence(self):
        """c ⊓ c = c (idempotence)."""
        c = syntax_from_json_schema('{"type": "string"}')

        result = c.meet(c)

        assert result == c

    def test_meet_commutativity(self):
        """c₁ ⊓ c₂ = c₂ ⊓ c₁ (commutativity)."""
        c1 = syntax_from_json_schema('{"type": "string"}')
        c2 = syntax_from_json_schema('{"type": "number"}')

        result1 = c1.meet(c2)
        result2 = c2.meet(c1)

        assert result1 == result2

    def test_meet_same_grammar_merges(self):
        """Meet of same grammar returns merged constraint."""
        schema = '{"type": "object"}'
        c1 = syntax_from_json_schema(schema)
        c2 = syntax_from_json_schema(schema)

        result = c1.meet(c2)

        # Since both have same state_hash (0), result equals both
        assert result == c1
        assert result == c2

    def test_meet_same_grammar_different_state(self):
        """Meet of same grammar with different state returns merged."""
        schema = '{"type": "object"}'
        c1 = syntax_from_json_schema(schema).with_state_hash(10)
        c2 = syntax_from_json_schema(schema).with_state_hash(20)

        result = c1.meet(c2)

        # Should take max state_hash
        assert result.state_hash == 20
        assert result.grammar_string == schema

    def test_meet_same_grammar_complete_state(self):
        """Meet of same grammar with completion state merges correctly."""
        schema = '{"type": "object"}'
        c1 = syntax_from_json_schema(schema).with_complete(True)
        c2 = syntax_from_json_schema(schema).with_complete(False)

        result = c1.meet(c2)

        # Both must be complete for result to be complete
        assert not result.is_complete

    def test_meet_different_grammars_returns_bottom(self):
        """Meet of different grammars returns BOTTOM (conservative)."""
        c1 = syntax_from_json_schema('{"type": "string"}')
        c2 = syntax_from_json_schema('{"type": "number"}')

        result = c1.meet(c2)

        assert result == SYNTAX_BOTTOM

    def test_meet_different_types_returns_bottom(self):
        """Meet of different grammar types returns BOTTOM."""
        c1 = syntax_from_json_schema('{"type": "string"}')
        c2 = syntax_from_regex(r".*")

        result = c1.meet(c2)

        assert result == SYNTAX_BOTTOM


class TestSyntaxConstraintState:
    """Tests for state management."""

    def test_with_state_hash(self):
        """Can update state hash."""
        c = syntax_from_json_schema('{"type": "string"}')

        updated = c.with_state_hash(42)

        assert updated.state_hash == 42
        assert updated.grammar_string == c.grammar_string
        assert updated.grammar_type == c.grammar_type

    def test_with_complete(self):
        """Can mark constraint as complete."""
        c = syntax_from_json_schema('{"type": "string"}')
        assert not c.is_complete

        completed = c.with_complete(True)

        assert completed.is_complete
        assert completed.grammar_string == c.grammar_string

    def test_equality(self):
        """Constraints are equal if all fields match."""
        c1 = syntax_from_json_schema('{"type": "string"}')
        c2 = syntax_from_json_schema('{"type": "string"}')

        assert c1 == c2

    def test_inequality_different_state(self):
        """Constraints with different state are not equal."""
        c1 = syntax_from_json_schema('{"type": "string"}')
        c2 = c1.with_state_hash(100)

        assert c1 != c2

    def test_hash_consistency(self):
        """Equal constraints have equal hashes."""
        c1 = syntax_from_json_schema('{"type": "string"}')
        c2 = syntax_from_json_schema('{"type": "string"}')

        assert hash(c1) == hash(c2)


class TestSyntaxDomain:
    """Tests for SyntaxDomain."""

    def test_domain_name(self):
        """Domain name is 'syntax'."""
        domain = SyntaxDomain()

        assert domain.name == "syntax"

    def test_domain_top_bottom(self):
        """Domain has correct TOP and BOTTOM."""
        domain = SyntaxDomain()

        assert domain.top == SYNTAX_TOP
        assert domain.bottom == SYNTAX_BOTTOM

    def test_token_mask_all_true(self):
        """Token mask returns all-True (actual masking via grammar object)."""
        domain = SyntaxDomain()
        ctx = GenerationContext(vocab_size=100, device="cpu")
        constraint = syntax_from_json_schema('{"type": "string"}')

        mask = domain.token_mask(constraint, ctx)

        assert mask.shape == (100,)
        assert mask.dtype == torch.bool
        assert mask.all()

    def test_observe_token_updates_state(self):
        """observe_token updates state hash."""
        domain = SyntaxDomain(
            grammar_type=GrammarType.JSON_SCHEMA,
            grammar_string='{"type": "string"}',
        )
        ctx = GenerationContext(vocab_size=100)
        constraint = domain.create_constraint_for_grammar(
            GrammarType.JSON_SCHEMA, '{"type": "string"}'
        )
        initial_hash = constraint.state_hash

        updated = domain.observe_token(constraint, token_id=42, context=ctx)

        assert updated.state_hash != initial_hash

    def test_observe_token_bottom_stays_bottom(self):
        """observe_token on BOTTOM returns BOTTOM."""
        domain = SyntaxDomain()
        ctx = GenerationContext(vocab_size=100)

        result = domain.observe_token(SYNTAX_BOTTOM, token_id=42, context=ctx)

        assert result == SYNTAX_BOTTOM

    def test_observe_token_top_stays_top(self):
        """observe_token on TOP returns TOP."""
        domain = SyntaxDomain()
        ctx = GenerationContext(vocab_size=100)

        result = domain.observe_token(SYNTAX_TOP, token_id=42, context=ctx)

        assert result == SYNTAX_TOP

    def test_checkpoint_restore(self):
        """Can checkpoint and restore state."""
        domain = SyntaxDomain(
            grammar_type=GrammarType.JSON_SCHEMA,
            grammar_string='{"type": "string"}',
        )
        ctx = GenerationContext(vocab_size=100)
        constraint = domain.create_constraint_for_grammar(
            GrammarType.JSON_SCHEMA, '{"type": "string"}'
        )

        # Observe some tokens
        domain.observe_token(constraint, 1, ctx)
        domain.observe_token(constraint, 2, ctx)

        cp = domain.checkpoint()

        # Observe more tokens
        domain.observe_token(constraint, 3, ctx)
        domain.observe_token(constraint, 4, ctx)

        current_counter = domain._state_counter

        # Restore
        domain.restore(cp)

        assert domain._state_counter < current_counter

    def test_satisfiability_delegates(self):
        """satisfiability delegates to constraint."""
        domain = SyntaxDomain()

        assert domain.satisfiability(SYNTAX_TOP) == Satisfiability.SAT
        assert domain.satisfiability(SYNTAX_BOTTOM) == Satisfiability.UNSAT

    def test_create_constraint_for_grammar(self):
        """Can create constraint for grammar."""
        domain = SyntaxDomain()
        schema = '{"type": "object"}'

        constraint = domain.create_constraint_for_grammar(
            GrammarType.JSON_SCHEMA, schema
        )

        assert constraint.grammar_type == GrammarType.JSON_SCHEMA
        assert constraint.grammar_string == schema
        assert constraint.state_hash == 0


class TestSyntaxConstraintRepr:
    """Tests for string representation."""

    def test_repr_top(self):
        """SYNTAX_TOP has correct repr."""
        assert repr(SYNTAX_TOP) == "SYNTAX_TOP"

    def test_repr_bottom(self):
        """SYNTAX_BOTTOM has correct repr."""
        assert repr(SYNTAX_BOTTOM) == "SYNTAX_BOTTOM"

    def test_repr_constraint(self):
        """Regular constraint has informative repr."""
        c = syntax_from_json_schema('{"type": "string"}')
        r = repr(c)

        assert "SyntaxConstraint" in r
        assert "JSON_SCHEMA" in r


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
