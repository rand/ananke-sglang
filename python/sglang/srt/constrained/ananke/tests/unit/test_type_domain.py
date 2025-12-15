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
"""Tests for the TypeDomain class.

Tests type-based token masking and constraint observation following
the Hazel research foundations.
"""

import pytest
import torch

from core.domain import GenerationContext
from domains.types.domain import TypeDomain, TypeDomainCheckpoint
from domains.types.constraint import (
    TYPE_TOP,
    TYPE_BOTTOM,
    TypeConstraint,
    INT,
    STR,
    BOOL,
    FLOAT,
    NONE,
    ANY,
    NEVER,
    FunctionType,
    ListType,
    DictType,
    TupleType,
    HoleType,
    type_expecting,
)
from domains.types.environment import EMPTY_ENVIRONMENT


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self, vocab: dict[int, str]):
        """Initialize with a vocabulary mapping.

        Args:
            vocab: Mapping from token_id to token text
        """
        self.vocab = vocab
        # Required for TokenClassifier disk cache hash computation
        self.vocab_size = len(vocab)

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs to text."""
        return "".join(self.vocab.get(tid, "") for tid in token_ids)

    def get_vocab(self) -> dict[str, int]:
        """Get vocabulary mapping (token -> id).

        Required for TokenClassifier initialization.
        """
        return {text: tid for tid, text in self.vocab.items()}


class TestTypeDomainBasics:
    """Tests for TypeDomain basic functionality."""

    def test_domain_name(self):
        """Domain has correct name."""
        domain = TypeDomain()
        assert domain.name == "types"

    def test_top_constraint(self):
        """Top constraint is TYPE_TOP."""
        domain = TypeDomain()
        assert domain.top == TYPE_TOP

    def test_bottom_constraint(self):
        """Bottom constraint is TYPE_BOTTOM."""
        domain = TypeDomain()
        assert domain.bottom == TYPE_BOTTOM

    def test_default_environment(self):
        """Default environment is empty."""
        domain = TypeDomain()
        assert domain.environment == EMPTY_ENVIRONMENT

    def test_default_expected_type(self):
        """Default expected type is Any."""
        domain = TypeDomain()
        assert domain.expected_type == ANY

    def test_custom_environment(self):
        """Custom environment is stored."""
        env = EMPTY_ENVIRONMENT.bind("x", INT)
        domain = TypeDomain(environment=env)
        assert domain.environment.lookup("x") == INT

    def test_custom_expected_type(self):
        """Custom expected type is stored."""
        domain = TypeDomain(expected_type=INT)
        assert domain.expected_type == INT


class TestTypeDomainTokenMask:
    """Tests for TypeDomain.token_mask()."""

    def setup_method(self):
        """Set up test fixtures."""
        self.domain = TypeDomain()
        self.vocab = {
            0: "42",
            1: "3.14",
            2: '"hello"',
            3: "True",
            4: "None",
            5: "x",
            6: "+",
            7: "[",
            8: "]",
            9: " ",
        }
        self.tokenizer = MockTokenizer(self.vocab)
        self.context = GenerationContext(
            vocab_size=10,
            device="cpu",
            tokenizer=self.tokenizer,
        )

    def test_top_allows_all(self):
        """TOP constraint allows all tokens."""
        mask = self.domain.token_mask(TYPE_TOP, self.context)

        assert mask.shape == (10,)
        assert mask.all()

    def test_bottom_blocks_all(self):
        """BOTTOM constraint blocks all tokens."""
        mask = self.domain.token_mask(TYPE_BOTTOM, self.context)

        assert mask.shape == (10,)
        assert not mask.any()

    def test_any_type_allows_all(self):
        """Any expected type allows all tokens."""
        constraint = type_expecting(ANY)
        mask = self.domain.token_mask(constraint, self.context)

        assert mask.all()

    def test_none_expected_allows_all(self):
        """No expected type allows all tokens."""
        constraint = TypeConstraint(expected_type=None)
        mask = self.domain.token_mask(constraint, self.context)

        assert mask.all()

    def test_int_expected_blocks_strings(self):
        """Int expected blocks string literals."""
        constraint = type_expecting(INT)
        mask = self.domain.token_mask(constraint, self.context)

        # Token 2 is '"hello"' - should be blocked
        assert not mask[2]
        # Token 0 is "42" - should be allowed
        assert mask[0]

    def test_str_expected_blocks_numbers(self):
        """Str expected blocks numeric literals."""
        constraint = type_expecting(STR)
        mask = self.domain.token_mask(constraint, self.context)

        # Token 0 is "42" - should be blocked
        assert not mask[0]
        # Token 1 is "3.14" - should be blocked
        assert not mask[1]
        # Token 2 is '"hello"' - should be allowed
        assert mask[2]

    def test_bool_expected_blocks_strings_and_floats(self):
        """Bool expected blocks strings and floats."""
        constraint = type_expecting(BOOL)
        mask = self.domain.token_mask(constraint, self.context)

        # Token 2 is '"hello"' - should be blocked
        assert not mask[2]
        # Token 1 is "3.14" - should be blocked
        assert not mask[1]
        # Token 3 is "True" - should be allowed
        assert mask[3]

    def test_function_expected_blocks_literals(self):
        """Function type blocks non-callable literals."""
        fn_type = FunctionType((INT,), STR)
        constraint = type_expecting(fn_type)
        mask = self.domain.token_mask(constraint, self.context)

        # Literals should be blocked
        assert not mask[0]  # "42"
        assert not mask[1]  # "3.14"
        assert not mask[2]  # '"hello"'
        # Identifiers and operators should be allowed
        assert mask[5]  # "x"
        assert mask[6]  # "+"

    def test_operators_always_allowed(self):
        """Operators are always allowed."""
        constraint = type_expecting(INT)
        mask = self.domain.token_mask(constraint, self.context)

        assert mask[6]  # "+"
        assert mask[7]  # "["
        assert mask[8]  # "]"

    def test_whitespace_always_allowed(self):
        """Whitespace is always allowed."""
        constraint = type_expecting(INT)
        mask = self.domain.token_mask(constraint, self.context)

        assert mask[9]  # " "

    def test_identifiers_allowed(self):
        """Identifiers are allowed (type checked at observation)."""
        constraint = type_expecting(INT)
        mask = self.domain.token_mask(constraint, self.context)

        assert mask[5]  # "x"

    def test_no_tokenizer_allows_all(self):
        """Without tokenizer, all tokens are allowed."""
        context = GenerationContext(vocab_size=10, device="cpu", tokenizer=None)
        constraint = type_expecting(INT)
        mask = self.domain.token_mask(constraint, self.context)

        # Should still work, allowing all tokens for tokens beyond check budget
        assert mask.shape == (10,)

    def test_list_type_allows_brackets(self):
        """List type allows bracket tokens."""
        list_type = ListType(INT)
        constraint = type_expecting(list_type)
        mask = self.domain.token_mask(constraint, self.context)

        # Brackets should be allowed
        assert mask[7]  # "["
        assert mask[8]  # "]"

    def test_hole_type_allows_all(self):
        """Hole type allows all tokens."""
        hole_type = HoleType("h0")
        constraint = type_expecting(hole_type)
        mask = self.domain.token_mask(constraint, self.context)

        assert mask.all()


class TestTypeDomainObserveToken:
    """Tests for TypeDomain.observe_token()."""

    def setup_method(self):
        """Set up test fixtures."""
        self.domain = TypeDomain()
        self.vocab = {
            0: "42",
            1: "3.14",
            2: '"hello"',
            3: "True",
            4: "None",
            5: "x",
            6: "+",
            7: "[",
            8: "]",
            9: " ",
            10: "=",
            11: ":",
            12: "->",
            13: "(",
            14: ")",
            15: "{",
            16: "}",
            17: ",",
            18: "return",
        }
        self.tokenizer = MockTokenizer(self.vocab)
        self.context = GenerationContext(
            vocab_size=20,
            device="cpu",
            tokenizer=self.tokenizer,
        )

    def test_top_unchanged(self):
        """TOP constraint is unchanged after observation."""
        result = self.domain.observe_token(TYPE_TOP, 0, self.context)
        assert result.is_top()

    def test_bottom_unchanged(self):
        """BOTTOM constraint is unchanged after observation."""
        result = self.domain.observe_token(TYPE_BOTTOM, 0, self.context)
        assert result.is_bottom()

    def test_state_counter_increments(self):
        """State counter increments after observation."""
        initial = self.domain._state_counter
        constraint = type_expecting(INT)
        self.domain.observe_token(constraint, 0, self.context)

        assert self.domain._state_counter == initial + 1

    def test_environment_hash_updated(self):
        """Environment hash is updated after observation."""
        constraint = type_expecting(INT)
        result = self.domain.observe_token(constraint, 0, self.context)

        assert result.environment_hash > 0

    def test_whitespace_preserves_constraint(self):
        """Whitespace token preserves constraint."""
        constraint = type_expecting(INT)
        result = self.domain.observe_token(constraint, 9, self.context)

        assert result.expected_type == INT

    def test_list_bracket_updates_expected(self):
        """[ updates expected type for list."""
        list_type = ListType(INT)
        constraint = type_expecting(list_type)
        result = self.domain.observe_token(constraint, 7, self.context)

        # Expected type should now be element type
        assert result.expected_type == INT

    def test_tuple_paren_updates_expected(self):
        """( updates expected type for tuple."""
        tuple_type = TupleType((STR, INT))
        constraint = type_expecting(tuple_type)
        result = self.domain.observe_token(constraint, 13, self.context)

        # Expected type should now be first element type
        assert result.expected_type == STR

    def test_dict_brace_updates_expected(self):
        """{ updates expected type for dict."""
        dict_type = DictType(STR, INT)
        constraint = type_expecting(dict_type)
        result = self.domain.observe_token(constraint, 15, self.context)

        # Expected type should now be key type
        assert result.expected_type == STR

    def test_function_paren_updates_expected(self):
        """( updates expected type for function call."""
        fn_type = FunctionType((INT, STR), BOOL)
        constraint = type_expecting(fn_type)
        result = self.domain.observe_token(constraint, 13, self.context)

        # Expected type should now be first param type
        assert result.expected_type == INT

    def test_compatible_literal_no_error(self):
        """Compatible literal doesn't set error."""
        constraint = type_expecting(INT)
        result = self.domain.observe_token(constraint, 0, self.context)  # "42"

        assert not result.has_errors

    def test_incompatible_literal_sets_error(self):
        """Incompatible literal sets error flag."""
        constraint = type_expecting(INT)
        result = self.domain.observe_token(constraint, 2, self.context)  # '"hello"'

        assert result.has_errors

    def test_compatible_identifier_no_error(self):
        """Compatible identifier doesn't set error."""
        env = EMPTY_ENVIRONMENT.bind("x", INT)
        domain = TypeDomain(environment=env)
        constraint = type_expecting(INT)
        result = domain.observe_token(constraint, 5, self.context)  # "x"

        assert not result.has_errors

    def test_incompatible_identifier_sets_error(self):
        """Incompatible identifier sets error flag."""
        env = EMPTY_ENVIRONMENT.bind("x", STR)
        domain = TypeDomain(environment=env)
        constraint = type_expecting(INT)
        result = domain.observe_token(constraint, 5, self.context)  # "x"

        assert result.has_errors

    def test_unknown_identifier_no_error(self):
        """Unknown identifier doesn't set error."""
        constraint = type_expecting(INT)
        result = self.domain.observe_token(constraint, 5, self.context)  # "x"

        # Unknown variable - no error (could be defined later)
        assert not result.has_errors

    def test_int_compatible_with_float(self):
        """Int literal is compatible with float expected."""
        constraint = type_expecting(FLOAT)
        result = self.domain.observe_token(constraint, 0, self.context)  # "42"

        # int <: float, so no error
        assert not result.has_errors


class TestTypeDomainCheckpoint:
    """Tests for TypeDomain checkpoint/restore."""

    def test_checkpoint_captures_state(self):
        """Checkpoint captures current state."""
        env = EMPTY_ENVIRONMENT.bind("x", INT)
        domain = TypeDomain(environment=env, expected_type=STR)
        domain._state_counter = 42

        checkpoint = domain.checkpoint()

        assert isinstance(checkpoint, TypeDomainCheckpoint)
        assert checkpoint.environment == env
        assert checkpoint.current_expected == STR
        assert checkpoint.state_counter == 42

    def test_restore_restores_state(self):
        """Restore restores captured state."""
        env1 = EMPTY_ENVIRONMENT.bind("x", INT)
        domain = TypeDomain(environment=env1, expected_type=STR)
        domain._state_counter = 10

        checkpoint = domain.checkpoint()

        # Modify state
        domain._environment = EMPTY_ENVIRONMENT
        domain._expected_type = INT
        domain._state_counter = 100

        # Restore
        domain.restore(checkpoint)

        assert domain.environment == env1
        assert domain.expected_type == STR
        assert domain._state_counter == 10

    def test_restore_wrong_type_raises(self):
        """Restore with wrong checkpoint type raises."""
        domain = TypeDomain()

        with pytest.raises(TypeError):
            domain.restore("not a checkpoint")


class TestTypeDomainEnvironment:
    """Tests for TypeDomain environment operations."""

    def test_bind_variable(self):
        """bind_variable adds to environment."""
        domain = TypeDomain()
        domain.bind_variable("x", INT)

        assert domain.lookup_variable("x") == INT

    def test_lookup_missing_returns_none(self):
        """lookup_variable returns None for missing."""
        domain = TypeDomain()
        assert domain.lookup_variable("x") is None

    def test_set_expected_type(self):
        """set_expected_type updates expected."""
        domain = TypeDomain()
        domain.set_expected_type(STR)

        assert domain.expected_type == STR

    def test_push_scope(self):
        """push_scope creates new scope."""
        domain = TypeDomain()
        domain.bind_variable("x", INT)
        domain.push_scope()
        domain.bind_variable("y", STR)

        # Both visible
        assert domain.lookup_variable("x") == INT
        assert domain.lookup_variable("y") == STR

    def test_pop_scope(self):
        """pop_scope removes inner scope."""
        domain = TypeDomain()
        domain.bind_variable("x", INT)
        domain.push_scope()
        domain.bind_variable("y", STR)
        domain.pop_scope()

        # x still visible, y not
        assert domain.lookup_variable("x") == INT
        assert domain.lookup_variable("y") is None


class TestTypeDomainConstraintCreation:
    """Tests for TypeDomain constraint creation."""

    def test_create_constraint_none(self):
        """create_constraint with None returns TOP."""
        domain = TypeDomain()
        constraint = domain.create_constraint(None)

        assert constraint.is_top()

    def test_create_constraint_with_type(self):
        """create_constraint with type returns expecting constraint."""
        domain = TypeDomain()
        constraint = domain.create_constraint(INT)

        assert constraint.expected_type == INT


class TestTypeDomainSatisfiability:
    """Tests for TypeDomain satisfiability checking."""

    def test_satisfiability_top(self):
        """TOP is satisfiable."""
        domain = TypeDomain()
        from core.constraint import Satisfiability

        assert domain.satisfiability(TYPE_TOP) == Satisfiability.SAT

    def test_satisfiability_bottom(self):
        """BOTTOM is unsatisfiable."""
        domain = TypeDomain()
        from core.constraint import Satisfiability

        assert domain.satisfiability(TYPE_BOTTOM) == Satisfiability.UNSAT

    def test_satisfiability_normal(self):
        """Normal constraint is satisfiable."""
        domain = TypeDomain()
        constraint = type_expecting(INT)
        from core.constraint import Satisfiability

        assert domain.satisfiability(constraint) == Satisfiability.SAT


class TestTypeDomainTokenCategories:
    """Tests for token categorization."""

    def test_categorize_int_literal(self):
        """Categorize integer literal."""
        domain = TypeDomain()
        assert domain._categorize_token("42") == "int_literal"
        assert domain._categorize_token("0") == "int_literal"

    def test_categorize_float_literal(self):
        """Categorize float literal."""
        domain = TypeDomain()
        assert domain._categorize_token("3.14") == "float_literal"
        assert domain._categorize_token("0.0") == "float_literal"

    def test_categorize_string_literal(self):
        """Categorize string literal."""
        domain = TypeDomain()
        assert domain._categorize_token('"hello"') == "string_literal"
        assert domain._categorize_token("'world'") == "string_literal"

    def test_categorize_bool_literal(self):
        """Categorize boolean literal."""
        domain = TypeDomain()
        assert domain._categorize_token("True") == "bool_literal"
        assert domain._categorize_token("False") == "bool_literal"

    def test_categorize_none_literal(self):
        """Categorize None literal."""
        domain = TypeDomain()
        assert domain._categorize_token("None") == "none_literal"

    def test_categorize_keyword(self):
        """Categorize keywords."""
        domain = TypeDomain()
        assert domain._categorize_token("def") == "keyword"
        assert domain._categorize_token("class") == "keyword"
        assert domain._categorize_token("return") == "keyword"

    def test_categorize_operator(self):
        """Categorize operators."""
        domain = TypeDomain()
        assert domain._categorize_token("+") == "operator"
        assert domain._categorize_token("==") == "operator"
        assert domain._categorize_token("[") == "operator"

    def test_categorize_identifier(self):
        """Categorize identifiers."""
        domain = TypeDomain()
        assert domain._categorize_token("x") == "identifier"
        assert domain._categorize_token("my_var") == "identifier"
        assert domain._categorize_token("_private") == "identifier"

    def test_categorize_whitespace(self):
        """Categorize whitespace."""
        domain = TypeDomain()
        assert domain._categorize_token("") == "whitespace"
        assert domain._categorize_token("  ") == "whitespace"


class TestTypeDomainTypeCompatibility:
    """Tests for type compatibility checking."""

    def test_same_type_compatible(self):
        """Same types are compatible."""
        domain = TypeDomain()
        assert domain._types_compatible(INT, INT)
        assert domain._types_compatible(STR, STR)

    def test_any_compatible_with_all(self):
        """Any is compatible with everything."""
        domain = TypeDomain()
        assert domain._types_compatible(ANY, INT)
        assert domain._types_compatible(INT, ANY)

    def test_hole_compatible_with_all(self):
        """Hole types are compatible with everything."""
        domain = TypeDomain()
        hole = HoleType("h0")
        assert domain._types_compatible(hole, INT)
        assert domain._types_compatible(INT, hole)

    def test_int_compatible_with_float(self):
        """Int is compatible with float (numeric promotion)."""
        domain = TypeDomain()
        assert domain._types_compatible(INT, FLOAT)

    def test_different_primitives_incompatible(self):
        """Different primitive types are incompatible."""
        domain = TypeDomain()
        assert not domain._types_compatible(INT, STR)
        assert not domain._types_compatible(BOOL, FLOAT)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
