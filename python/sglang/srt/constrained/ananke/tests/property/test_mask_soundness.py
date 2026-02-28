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
"""Property-based tests for token mask soundness using Hypothesis.

These tests verify that token masking is SOUND:
- If a token would definitely violate a constraint, it should be blocked
- If we're uncertain, we should allow the token (conservative)

The key invariant is: NEVER block a valid token.
It's acceptable to allow invalid tokens (completeness sacrifice for soundness).

Tests cover:
1. TOP constraints allow all tokens
2. BOTTOM constraints block all tokens
3. Type constraints block obvious type violations
4. Meet of masks is subset of each operand
5. Checkpoint/restore preserves mask behavior
"""

import pytest
from hypothesis import given, settings, assume, strategies as st, HealthCheck
import torch

from core.domain import GenerationContext
from core.token_classifier import (
    TokenClassifier,
    TokenCategory,
    get_or_create_classifier,
    clear_classifier_cache,
)
from domains.types.domain import TypeDomain
from domains.types.constraint import (
    TYPE_TOP,
    TYPE_BOTTOM,
    INT,
    STR,
    BOOL,
    FLOAT,
    NONE,
    ANY,
    type_expecting,
)
from domains.imports.domain import ImportDomain
from domains.imports.constraint import IMPORT_TOP, IMPORT_BOTTOM, ImportConstraint
from domains.controlflow.domain import ControlFlowDomain
from domains.controlflow.constraint import CONTROLFLOW_TOP, CONTROLFLOW_BOTTOM
from domains.semantics.domain import SemanticDomain
from domains.semantics.constraint import SEMANTIC_TOP, SEMANTIC_BOTTOM


class MockTokenizer:
    """Mock tokenizer for property tests."""

    def __init__(self, vocab_size: int = 200):
        self.vocab_size = vocab_size
        self._id_to_text = {}

        # Simple vocab with known patterns
        keywords = ["if", "else", "for", "while", "def", "return", "import", "from",
                    "True", "False", "None", "and", "or", "not", "break", "continue"]
        for i, kw in enumerate(keywords):
            self._id_to_text[i] = kw

        # Integers 16-35
        for i in range(20):
            self._id_to_text[16 + i] = str(i - 5)

        # Floats 36-45
        for i, f in enumerate(["0.0", "1.0", "2.5", "3.14", "-1.0", "0.5", "1.5", "2.0", "9.9", "-0.1"]):
            self._id_to_text[36 + i] = f

        # Strings 46-55
        for i, s in enumerate(['"a"', '"b"', '""', '"test"', '"x"', "'y'", "'z'", "'''", '"""', '"str"']):
            self._id_to_text[46 + i] = s

        # Identifiers 56-75
        for i, ident in enumerate("xyzabcdefghijklmnopqr"):
            self._id_to_text[56 + i] = ident

        # Fill rest
        for i in range(76, vocab_size):
            self._id_to_text[i] = f"t{i}"

    def decode(self, token_ids):
        if isinstance(token_ids, int):
            return self._id_to_text.get(token_ids, f"<{token_ids}>")
        return "".join(self._id_to_text.get(t, f"<{t}>") for t in token_ids)


# Strategies for generating test inputs
@st.composite
def primitive_type(draw):
    """Generate a primitive type."""
    return draw(st.sampled_from([INT, STR, BOOL, FLOAT, NONE]))


@st.composite
def type_constraint(draw):
    """Generate a type constraint."""
    if draw(st.booleans()):
        return TYPE_TOP
    return type_expecting(draw(primitive_type()))


@st.composite
def import_constraint(draw):
    """Generate an import constraint."""
    if draw(st.booleans()):
        return IMPORT_TOP
    # Generate some forbidden modules
    modules = ["os", "sys", "subprocess", "pickle", "socket"]
    forbidden = frozenset(draw(st.lists(st.sampled_from(modules), max_size=3)))
    return ImportConstraint(forbidden=forbidden)


@st.composite
def token_id(draw, max_id: int = 199):
    """Generate a valid token ID."""
    return draw(st.integers(min_value=0, max_value=max_id))


class TestTopConstraintSoundness:
    """Property tests for TOP constraints."""

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

    @given(tok=token_id())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_type_top_allows_all(self, tok, tokenizer, context):
        """TYPE_TOP should allow all tokens."""
        domain = TypeDomain(language="python", tokenizer=tokenizer)
        mask = domain.token_mask(TYPE_TOP, context)
        assume(tok < context.vocab_size)
        assert mask[tok], f"TYPE_TOP blocked token {tok}"

    @given(tok=token_id())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_import_top_allows_all(self, tok, tokenizer, context):
        """IMPORT_TOP should allow all tokens."""
        domain = ImportDomain(language="python", tokenizer=tokenizer)
        mask = domain.token_mask(IMPORT_TOP, context)
        assume(tok < context.vocab_size)
        assert mask[tok], f"IMPORT_TOP blocked token {tok}"

    @given(tok=token_id())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_controlflow_top_allows_all(self, tok, tokenizer, context):
        """CONTROLFLOW_TOP should allow all tokens."""
        domain = ControlFlowDomain(language="python", tokenizer=tokenizer)
        mask = domain.token_mask(CONTROLFLOW_TOP, context)
        assume(tok < context.vocab_size)
        assert mask[tok], f"CONTROLFLOW_TOP blocked token {tok}"

    @given(tok=token_id())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_semantic_top_allows_all(self, tok, tokenizer, context):
        """SEMANTIC_TOP should allow all tokens."""
        domain = SemanticDomain(language="python", tokenizer=tokenizer)
        mask = domain.token_mask(SEMANTIC_TOP, context)
        assume(tok < context.vocab_size)
        assert mask[tok], f"SEMANTIC_TOP blocked token {tok}"


class TestBottomConstraintSoundness:
    """Property tests for BOTTOM constraints."""

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

    @given(tok=token_id())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_type_bottom_blocks_all(self, tok, tokenizer, context):
        """TYPE_BOTTOM should block all tokens."""
        domain = TypeDomain(language="python", tokenizer=tokenizer)
        mask = domain.token_mask(TYPE_BOTTOM, context)
        assume(tok < context.vocab_size)
        assert not mask[tok], f"TYPE_BOTTOM allowed token {tok}"

    @given(tok=token_id())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_import_bottom_blocks_all(self, tok, tokenizer, context):
        """IMPORT_BOTTOM should block all tokens."""
        domain = ImportDomain(language="python", tokenizer=tokenizer)
        mask = domain.token_mask(IMPORT_BOTTOM, context)
        assume(tok < context.vocab_size)
        assert not mask[tok], f"IMPORT_BOTTOM allowed token {tok}"

    @given(tok=token_id())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_controlflow_bottom_blocks_all(self, tok, tokenizer, context):
        """CONTROLFLOW_BOTTOM should block all tokens."""
        domain = ControlFlowDomain(language="python", tokenizer=tokenizer)
        mask = domain.token_mask(CONTROLFLOW_BOTTOM, context)
        assume(tok < context.vocab_size)
        assert not mask[tok], f"CONTROLFLOW_BOTTOM allowed token {tok}"

    @given(tok=token_id())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_semantic_bottom_blocks_all(self, tok, tokenizer, context):
        """SEMANTIC_BOTTOM should block all tokens."""
        domain = SemanticDomain(language="python", tokenizer=tokenizer)
        mask = domain.token_mask(SEMANTIC_BOTTOM, context)
        assume(tok < context.vocab_size)
        assert not mask[tok], f"SEMANTIC_BOTTOM allowed token {tok}"


class TestMaskMeetSoundness:
    """Property tests for mask intersection (meet operation)."""

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

    @given(c1=type_constraint(), c2=type_constraint(), tok=token_id())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_meet_is_subset(self, c1, c2, tok, tokenizer, context):
        """Meet of masks should be subset of each operand."""
        domain = TypeDomain(language="python", tokenizer=tokenizer)

        mask1 = domain.token_mask(c1, context)
        mask2 = domain.token_mask(c2, context)
        fused = mask1 & mask2

        assume(tok < context.vocab_size)

        # If fused allows, both must allow
        if fused[tok]:
            assert mask1[tok], f"Fused allowed but mask1 blocked token {tok}"
            assert mask2[tok], f"Fused allowed but mask2 blocked token {tok}"

    @given(c=type_constraint(), tok=token_id())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_meet_with_top_identity(self, c, tok, tokenizer, context):
        """Meeting with TOP should return original mask."""
        domain = TypeDomain(language="python", tokenizer=tokenizer)

        mask = domain.token_mask(c, context)
        top_mask = domain.token_mask(TYPE_TOP, context)
        fused = mask & top_mask

        assume(tok < context.vocab_size)
        assert fused[tok] == mask[tok], f"Meet with TOP changed mask at {tok}"


class TestTypeConstraintSoundness:
    """Property tests for type-specific masking soundness."""

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

    def test_int_constraint_allows_integers(self, tokenizer, context):
        """INT constraint should allow integer literal tokens."""
        domain = TypeDomain(language="python", tokenizer=tokenizer)
        constraint = type_expecting(INT)
        mask = domain.token_mask(constraint, context)

        # Token IDs 16-35 are integers in our mock tokenizer
        # At least some should be allowed
        int_token_ids = list(range(16, 36))
        allowed_ints = [t for t in int_token_ids if t < context.vocab_size and mask[t]]

        assert len(allowed_ints) > 0, "INT constraint should allow some integer tokens"

    def test_str_constraint_allows_strings(self, tokenizer, context):
        """STR constraint should allow string literal tokens."""
        domain = TypeDomain(language="python", tokenizer=tokenizer)
        constraint = type_expecting(STR)
        mask = domain.token_mask(constraint, context)

        # Token IDs 46-55 are strings in our mock tokenizer
        str_token_ids = list(range(46, 56))
        allowed_strs = [t for t in str_token_ids if t < context.vocab_size and mask[t]]

        assert len(allowed_strs) > 0, "STR constraint should allow some string tokens"

    def test_bool_constraint_allows_booleans(self, tokenizer, context):
        """BOOL constraint should allow True/False tokens."""
        domain = TypeDomain(language="python", tokenizer=tokenizer)
        constraint = type_expecting(BOOL)
        mask = domain.token_mask(constraint, context)

        # Token IDs 8, 9 are True, False in our mock tokenizer
        bool_token_ids = [8, 9]
        allowed_bools = [t for t in bool_token_ids if t < context.vocab_size and mask[t]]

        assert len(allowed_bools) > 0, "BOOL constraint should allow True/False tokens"

    def test_none_constraint_allows_none(self, tokenizer, context):
        """NONE constraint should allow None token."""
        domain = TypeDomain(language="python", tokenizer=tokenizer)
        constraint = type_expecting(NONE)
        mask = domain.token_mask(constraint, context)

        # Token ID 10 is None in our mock tokenizer
        none_token_id = 10
        if none_token_id < context.vocab_size:
            assert mask[none_token_id], "NONE constraint should allow None token"


class TestCheckpointRestoreSoundness:
    """Property tests for checkpoint/restore preserving mask soundness."""

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

    @given(c=type_constraint())
    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_type_checkpoint_restore_preserves_mask(self, c, tokenizer, context):
        """Checkpoint/restore should preserve mask computation."""
        domain = TypeDomain(language="python", tokenizer=tokenizer)

        # Get mask before checkpoint
        mask_before = domain.token_mask(c, context)

        # Checkpoint
        checkpoint = domain.checkpoint()

        # Modify state
        domain.observe_token(c, 35, context)

        # Restore
        domain.restore(checkpoint)

        # Get mask after restore
        mask_after = domain.token_mask(c, context)

        # Masks should be equal
        assert torch.equal(mask_before, mask_after), "Checkpoint/restore changed mask"

    @given(c=import_constraint())
    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_import_checkpoint_restore_preserves_mask(self, c, tokenizer, context):
        """Checkpoint/restore should preserve mask computation for imports."""
        domain = ImportDomain(language="python", tokenizer=tokenizer)

        mask_before = domain.token_mask(c, context)
        checkpoint = domain.checkpoint()
        domain.observe_token(c, 6, context)  # "import"
        domain.restore(checkpoint)
        mask_after = domain.token_mask(c, context)

        assert torch.equal(mask_before, mask_after), "Import checkpoint/restore changed mask"


class TestSemanticBoundsSoundness:
    """Property tests for semantic bounds-based masking."""

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

    @given(lo=st.integers(min_value=-10, max_value=10),
           hi=st.integers(min_value=-10, max_value=20))
    @settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_bounds_extraction(self, lo, hi, tokenizer, context):
        """Bounds should be correctly extracted from assertions."""
        assume(lo < hi)  # Valid bounds

        domain = SemanticDomain(
            language="python",
            aggressive_mode=True,
            tokenizer=tokenizer,
        )

        constraint = domain.create_constraint(
            assertions=[f"x >= {lo}", f"x <= {hi}"]
        )

        # Update variable bounds
        domain._update_variable_bounds(constraint)

        # Check bounds were extracted
        if "x" in domain._variable_bounds:
            bounds = domain._variable_bounds["x"]
            if bounds.lower is not None:
                assert bounds.lower <= lo or bounds.lower == lo
            if bounds.upper is not None:
                assert bounds.upper >= hi or bounds.upper == hi


# Import context-aware types for property tests
from domains.semantics.domain import (
    ExpressionContext,
    ContextConfidence,
    BoundsConfidence,
    BlockingLevel,
    ExpressionState,
    VariableBounds,
)


class TestContextAwareSoundness:
    """Property tests for context-aware bounds checking soundness.

    Key invariant: SOUNDNESS OVER COMPLETENESS
    - NEVER block a valid token (false positive NOT OK)
    - May allow invalid tokens (false negative OK)
    - Low confidence -> PERMISSIVE (allow all)
    """

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

    @given(
        context_conf=st.sampled_from(list(ContextConfidence)),
        bounds_conf=st.sampled_from(list(BoundsConfidence)),
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_decision_matrix_soundness(self, context_conf, bounds_conf, tokenizer, context):
        """Decision matrix should be sound: low confidence -> PERMISSIVE."""
        domain = SemanticDomain(
            language="python",
            aggressive_mode=True,
            tokenizer=tokenizer,
        )

        level = domain._compute_blocking_level(context_conf, bounds_conf)

        # Soundness property: low confidence on either side -> PERMISSIVE
        if context_conf in (ContextConfidence.LOW, ContextConfidence.NONE):
            assert level == BlockingLevel.PERMISSIVE, \
                f"Low context confidence {context_conf} should be PERMISSIVE, got {level}"

        if bounds_conf in (BoundsConfidence.LOW, BoundsConfidence.UNKNOWN):
            assert level == BlockingLevel.PERMISSIVE, \
                f"Low bounds confidence {bounds_conf} should be PERMISSIVE, got {level}"

        # AGGRESSIVE only with HIGH confidence on both sides
        if level == BlockingLevel.AGGRESSIVE:
            assert context_conf == ContextConfidence.HIGH, \
                f"AGGRESSIVE requires HIGH context confidence, got {context_conf}"
            assert bounds_conf == BoundsConfidence.HIGH, \
                f"AGGRESSIVE requires HIGH bounds confidence, got {bounds_conf}"

    @given(tok=token_id())
    @settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_permissive_mode_allows_all(self, tok, tokenizer, context):
        """PERMISSIVE blocking level should allow all tokens."""
        domain = SemanticDomain(
            language="python",
            aggressive_mode=True,
            tokenizer=tokenizer,
        )

        # Set up low confidence state
        domain._expression_state.context = ExpressionContext.FUNCTION_CALL
        domain._expression_state.context_confidence = ContextConfidence.NONE

        # Create constraint with bounds
        constraint = domain.create_constraint(assertions=["x >= 0", "x <= 10"])
        domain._update_variable_bounds(constraint)
        domain._expression_state.target_variable = "x"

        mask = domain.token_mask(constraint, context)

        assume(tok < context.vocab_size)
        # With NONE confidence, should be PERMISSIVE and allow all
        assert mask[tok], f"PERMISSIVE mode blocked token {tok}"

    @given(
        lower=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
        upper=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
        smt_uncertain=st.booleans(),
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_effective_confidence_downgrade(self, lower, upper, smt_uncertain, tokenizer, context):
        """SMT uncertainty should downgrade confidence."""
        assume(lower < upper)

        bounds = VariableBounds(
            lower=lower,
            upper=upper,
            confidence=BoundsConfidence.HIGH,
            smt_uncertain=smt_uncertain,
        )

        effective = bounds.effective_confidence()

        if smt_uncertain:
            # Should be downgraded
            assert effective != BoundsConfidence.HIGH, \
                "SMT uncertainty should downgrade HIGH confidence"
        else:
            # Should be unchanged
            assert effective == BoundsConfidence.HIGH, \
                "No uncertainty should keep HIGH confidence"

    @given(
        value=st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
        lower=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
        upper=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_bounds_contains_correctness(self, value, lower, upper, tokenizer, context):
        """bounds.contains() should be mathematically correct."""
        assume(lower < upper)

        bounds = VariableBounds(lower=lower, upper=upper)
        result = bounds.contains(value)

        expected = lower <= value <= upper
        assert result == expected, \
            f"contains({value}) for [{lower}, {upper}] should be {expected}, got {result}"

    @given(
        value=st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
        lower=st.floats(min_value=0.01, max_value=100, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_clearly_violated_soundness(self, value, lower, tokenizer, context):
        """is_clearly_violated() should only return True for obvious violations."""
        bounds = VariableBounds(lower=lower, upper=lower * 10)

        if bounds.is_clearly_violated(value):
            # If we say it's clearly violated, it must be out of bounds
            assert not bounds.contains(value), \
                f"is_clearly_violated({value}) but value is in bounds [{lower}, {lower*10}]"

        # But in-bounds values should NOT be clearly violated
        if bounds.contains(value):
            assert not bounds.is_clearly_violated(value), \
                f"In-bounds value {value} was marked as clearly violated"


class TestExpressionStateSoundness:
    """Property tests for expression state machine soundness."""

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

    @given(
        paren_count=st.integers(min_value=0, max_value=10),
        bracket_count=st.integers(min_value=0, max_value=10),
        brace_count=st.integers(min_value=0, max_value=10),
    )
    @settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_nesting_depth_invariants(self, paren_count, bracket_count, brace_count, tokenizer, context):
        """Nesting depths should never go negative."""
        domain = SemanticDomain(
            language="python",
            aggressive_mode=True,
            tokenizer=tokenizer,
        )

        # Open brackets
        for _ in range(paren_count):
            domain._update_expression_state("(")
        for _ in range(bracket_count):
            domain._update_expression_state("[")
        for _ in range(brace_count):
            domain._update_expression_state("{")

        state = domain._expression_state
        assert state.paren_depth == paren_count
        assert state.bracket_depth == bracket_count
        assert state.brace_depth == brace_count

        # Close more than opened (should clamp to 0)
        for _ in range(paren_count + 5):
            domain._update_expression_state(")")

        assert domain._expression_state.paren_depth >= 0, \
            "paren_depth went negative"

    @given(var_name=st.from_regex(r"[a-zA-Z_][a-zA-Z0-9_]*", fullmatch=True))
    @settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_assignment_context_captures_variable(self, var_name, tokenizer, context):
        """Assignment context should capture the variable being assigned."""
        assume(len(var_name) > 0 and len(var_name) < 20)

        domain = SemanticDomain(
            language="python",
            aggressive_mode=True,
            tokenizer=tokenizer,
        )

        domain._token_buffer = var_name
        domain._update_expression_state("=")

        state = domain._expression_state
        assert state.target_variable == var_name, \
            f"Expected target_variable={var_name}, got {state.target_variable}"
        assert state.context == ExpressionContext.SIMPLE_ASSIGNMENT_RHS
        assert state.context_confidence == ContextConfidence.HIGH

    def test_statement_terminator_resets_state(self, tokenizer, context):
        """Statement terminators should reset expression state."""
        domain = SemanticDomain(
            language="python",
            aggressive_mode=True,
            tokenizer=tokenizer,
        )

        # Set up assignment context
        domain._token_buffer = "x"
        domain._update_expression_state("=")

        # Verify state is set
        assert domain._expression_state.context == ExpressionContext.SIMPLE_ASSIGNMENT_RHS

        # Terminate with newline
        domain._update_expression_state("\n")

        # State should be reset
        state = domain._expression_state
        assert state.context == ExpressionContext.NONE
        assert state.context_confidence == ContextConfidence.NONE
        assert state.target_variable is None


class TestCheckpointRestoreExpressionState:
    """Property tests for checkpoint/restore of expression state."""

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

    @given(
        context_type=st.sampled_from(list(ExpressionContext)),
        confidence=st.sampled_from(list(ContextConfidence)),
        paren_depth=st.integers(min_value=0, max_value=5),
        tokens_since=st.integers(min_value=0, max_value=10),
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_checkpoint_preserves_expression_state(
        self, context_type, confidence, paren_depth, tokens_since, tokenizer, context
    ):
        """Checkpoint should preserve all expression state fields."""
        domain = SemanticDomain(
            language="python",
            aggressive_mode=True,
            tokenizer=tokenizer,
        )

        # Set up state
        domain._expression_state.context = context_type
        domain._expression_state.context_confidence = confidence
        domain._expression_state.paren_depth = paren_depth
        domain._expression_state.tokens_since_assignment = tokens_since
        domain._expression_state.target_variable = "test_var"

        # Checkpoint
        checkpoint = domain.checkpoint()

        # Modify state
        domain._expression_state.context = ExpressionContext.NONE
        domain._expression_state.context_confidence = ContextConfidence.NONE
        domain._expression_state.paren_depth = 0
        domain._expression_state.target_variable = None

        # Restore
        domain.restore(checkpoint)

        # Verify restoration
        state = domain._expression_state
        assert state.context == context_type
        assert state.context_confidence == confidence
        assert state.paren_depth == paren_depth
        assert state.tokens_since_assignment == tokens_since
        assert state.target_variable == "test_var"

    @given(
        lower=st.floats(min_value=-100, max_value=0, allow_nan=False, allow_infinity=False),
        upper=st.floats(min_value=1, max_value=100, allow_nan=False, allow_infinity=False),
        confidence=st.sampled_from(list(BoundsConfidence)),
        smt_uncertain=st.booleans(),
    )
    @settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_checkpoint_preserves_bounds_confidence(
        self, lower, upper, confidence, smt_uncertain, tokenizer, context
    ):
        """Checkpoint should preserve bounds with confidence fields."""
        domain = SemanticDomain(
            language="python",
            aggressive_mode=True,
            tokenizer=tokenizer,
        )

        # Set up bounds with confidence
        domain._variable_bounds["x"] = VariableBounds(
            lower=lower,
            upper=upper,
            confidence=confidence,
            source="test",
            smt_uncertain=smt_uncertain,
        )

        # Checkpoint
        checkpoint = domain.checkpoint()

        # Modify
        domain._variable_bounds.clear()

        # Restore
        domain.restore(checkpoint)

        # Verify
        assert "x" in domain._variable_bounds
        bounds = domain._variable_bounds["x"]
        assert bounds.lower == lower
        assert bounds.upper == upper
        assert bounds.confidence == confidence
        assert bounds.smt_uncertain == smt_uncertain


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
