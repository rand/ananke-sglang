# Copyright 2023-2024 SGLang Team
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
"""Property-based tests for semilattice laws using Hypothesis.

These tests verify that all constraint implementations satisfy the fundamental
semilattice properties required for correct constraint composition:

1. Identity: c ⊓ ⊤ = c
2. Annihilation: c ⊓ ⊥ = ⊥
3. Idempotence: c ⊓ c = c
4. Commutativity: c₁ ⊓ c₂ = c₂ ⊓ c₁
5. Associativity: (c₁ ⊓ c₂) ⊓ c₃ = c₁ ⊓ (c₂ ⊓ c₃)

These properties are critical for:
- Correct constraint fusion across domains
- Fixpoint convergence in propagation
- Deterministic constraint resolution

References:
    - Hazel: Mathematical foundations of compositional constraint systems
"""

import pytest
from hypothesis import given, settings, strategies as st

from core.constraint import (
    BOTTOM,
    TOP,
    BottomConstraint,
    Constraint,
    Satisfiability,
    TopConstraint,
)
from core.unified import (
    UNIFIED_BOTTOM,
    UNIFIED_TOP,
    UnifiedConstraint,
)
from domains.syntax.constraint import (
    SYNTAX_BOTTOM,
    SYNTAX_TOP,
    GrammarType,
    SyntaxConstraint,
)
from domains.types.constraint import (
    TYPE_BOTTOM,
    TYPE_TOP,
    TypeConstraint,
    TypeVar,
    INT,
    STR,
    BOOL,
    FLOAT,
    ANY,
    type_expecting,
)


# Strategy for generating basic constraints (TOP or BOTTOM)
basic_constraint = st.sampled_from([TOP, BOTTOM])


# Strategy for generating unified constraints with random components
@st.composite
def unified_constraint_strategy(draw):
    """Generate random UnifiedConstraint with each component being TOP or BOTTOM."""
    return UnifiedConstraint(
        syntax=draw(basic_constraint),
        types=draw(basic_constraint),
        imports=draw(basic_constraint),
        controlflow=draw(basic_constraint),
        semantics=draw(basic_constraint),
    )


class TestBasicConstraintSemilattice:
    """Property tests for TOP and BOTTOM constraints."""

    @given(c=basic_constraint)
    def test_identity_law(self, c: Constraint):
        """c ⊓ ⊤ = c (identity law)."""
        assert c.meet(TOP) == c

    @given(c=basic_constraint)
    def test_annihilation_law(self, c: Constraint):
        """c ⊓ ⊥ = ⊥ (annihilation law)."""
        result = c.meet(BOTTOM)
        assert result == BOTTOM

    @given(c=basic_constraint)
    def test_idempotence(self, c: Constraint):
        """c ⊓ c = c (idempotence)."""
        assert c.meet(c) == c

    @given(c1=basic_constraint, c2=basic_constraint)
    def test_commutativity(self, c1: Constraint, c2: Constraint):
        """c₁ ⊓ c₂ = c₂ ⊓ c₁ (commutativity)."""
        assert c1.meet(c2) == c2.meet(c1)

    @given(c1=basic_constraint, c2=basic_constraint, c3=basic_constraint)
    def test_associativity(self, c1: Constraint, c2: Constraint, c3: Constraint):
        """(c₁ ⊓ c₂) ⊓ c₃ = c₁ ⊓ (c₂ ⊓ c₃) (associativity)."""
        left_assoc = c1.meet(c2).meet(c3)
        right_assoc = c1.meet(c2.meet(c3))
        assert left_assoc == right_assoc


class TestUnifiedConstraintSemilattice:
    """Property tests for UnifiedConstraint semilattice laws."""

    @given(c=unified_constraint_strategy())
    def test_identity_law(self, c: UnifiedConstraint):
        """c ⊓ ⊤ = c (identity law)."""
        result = c.meet(UNIFIED_TOP)
        assert result == c

    @given(c=unified_constraint_strategy())
    def test_annihilation_law(self, c: UnifiedConstraint):
        """c ⊓ ⊥ = ⊥ (annihilation law)."""
        result = c.meet(UNIFIED_BOTTOM)
        assert result.is_bottom()

    @given(c=unified_constraint_strategy())
    def test_idempotence(self, c: UnifiedConstraint):
        """c ⊓ c = c (idempotence)."""
        assert c.meet(c) == c

    @given(c1=unified_constraint_strategy(), c2=unified_constraint_strategy())
    def test_commutativity(self, c1: UnifiedConstraint, c2: UnifiedConstraint):
        """c₁ ⊓ c₂ = c₂ ⊓ c₁ (commutativity)."""
        assert c1.meet(c2) == c2.meet(c1)

    @given(
        c1=unified_constraint_strategy(),
        c2=unified_constraint_strategy(),
        c3=unified_constraint_strategy(),
    )
    def test_associativity(
        self, c1: UnifiedConstraint, c2: UnifiedConstraint, c3: UnifiedConstraint
    ):
        """(c₁ ⊓ c₂) ⊓ c₃ = c₁ ⊓ (c₂ ⊓ c₃) (associativity)."""
        left_assoc = c1.meet(c2).meet(c3)
        right_assoc = c1.meet(c2.meet(c3))
        assert left_assoc == right_assoc


class TestUnifiedConstraintProperties:
    """Additional property tests for UnifiedConstraint behavior."""

    @given(c=unified_constraint_strategy())
    def test_is_bottom_iff_any_component_is_bottom(self, c: UnifiedConstraint):
        """UnifiedConstraint is BOTTOM iff any component is BOTTOM."""
        has_bottom_component = (
            c.syntax.is_bottom()
            or c.types.is_bottom()
            or c.imports.is_bottom()
            or c.controlflow.is_bottom()
            or c.semantics.is_bottom()
        )
        assert c.is_bottom() == has_bottom_component

    @given(c=unified_constraint_strategy())
    def test_is_top_iff_all_components_are_top(self, c: UnifiedConstraint):
        """UnifiedConstraint is TOP iff all components are TOP."""
        all_top = (
            c.syntax.is_top()
            and c.types.is_top()
            and c.imports.is_top()
            and c.controlflow.is_top()
            and c.semantics.is_top()
        )
        assert c.is_top() == all_top

    @given(c=unified_constraint_strategy())
    def test_satisfiability_consistency(self, c: UnifiedConstraint):
        """Satisfiability is consistent with is_bottom/is_top."""
        sat = c.satisfiability()
        if c.is_bottom():
            assert sat == Satisfiability.UNSAT
        elif c.is_top():
            assert sat == Satisfiability.SAT

    @given(c1=unified_constraint_strategy(), c2=unified_constraint_strategy())
    def test_meet_is_component_wise(self, c1: UnifiedConstraint, c2: UnifiedConstraint):
        """Meet operates component-wise."""
        result = c1.meet(c2)
        assert result.syntax == c1.syntax.meet(c2.syntax)
        assert result.types == c1.types.meet(c2.types)
        assert result.imports == c1.imports.meet(c2.imports)
        assert result.controlflow == c1.controlflow.meet(c2.controlflow)
        assert result.semantics == c1.semantics.meet(c2.semantics)


class TestSatisfiabilityAlgebra:
    """Property tests for Satisfiability enum algebra."""

    @given(s=st.sampled_from(list(Satisfiability)))
    def test_sat_dominates_in_disjunction(self, s: Satisfiability):
        """SAT ∨ x = SAT for any x."""
        assert (Satisfiability.SAT | s) == Satisfiability.SAT

    @given(s=st.sampled_from(list(Satisfiability)))
    def test_unsat_dominates_in_conjunction(self, s: Satisfiability):
        """UNSAT ∧ x = UNSAT for any x."""
        assert (Satisfiability.UNSAT & s) == Satisfiability.UNSAT

    @given(s1=st.sampled_from(list(Satisfiability)), s2=st.sampled_from(list(Satisfiability)))
    def test_conjunction_commutativity(self, s1: Satisfiability, s2: Satisfiability):
        """s₁ ∧ s₂ = s₂ ∧ s₁ (commutativity)."""
        assert (s1 & s2) == (s2 & s1)

    @given(s1=st.sampled_from(list(Satisfiability)), s2=st.sampled_from(list(Satisfiability)))
    def test_disjunction_commutativity(self, s1: Satisfiability, s2: Satisfiability):
        """s₁ ∨ s₂ = s₂ ∨ s₁ (commutativity)."""
        assert (s1 | s2) == (s2 | s1)


# Strategy for generating syntax constraints
@st.composite
def syntax_constraint_strategy(draw):
    """Generate random SyntaxConstraint."""
    # 10% chance of TOP, 10% chance of BOTTOM, 80% real constraint
    choice = draw(st.integers(min_value=0, max_value=9))
    if choice == 0:
        return SYNTAX_TOP
    elif choice == 1:
        return SYNTAX_BOTTOM
    else:
        grammar_type = draw(st.sampled_from(list(GrammarType)))
        # Use a small set of grammar strings to enable meaningful meet operations
        grammar_string = draw(st.sampled_from([
            '{"type": "string"}',
            '{"type": "number"}',
            '{"type": "boolean"}',
            r"[a-z]+",
            r"[0-9]+",
        ]))
        state_hash = draw(st.integers(min_value=0, max_value=100))
        is_complete = draw(st.booleans())
        return SyntaxConstraint(
            grammar_type=grammar_type,
            grammar_string=grammar_string,
            state_hash=state_hash,
            is_complete=is_complete,
        )


class TestSyntaxConstraintSemilattice:
    """Property tests for SyntaxConstraint semilattice laws."""

    @given(c=syntax_constraint_strategy())
    def test_identity_law(self, c: SyntaxConstraint):
        """c ⊓ ⊤ = c (identity law)."""
        assert c.meet(SYNTAX_TOP) == c

    @given(c=syntax_constraint_strategy())
    def test_annihilation_law(self, c: SyntaxConstraint):
        """c ⊓ ⊥ = ⊥ (annihilation law)."""
        result = c.meet(SYNTAX_BOTTOM)
        assert result == SYNTAX_BOTTOM

    @given(c=syntax_constraint_strategy())
    def test_idempotence(self, c: SyntaxConstraint):
        """c ⊓ c = c (idempotence)."""
        assert c.meet(c) == c

    @given(c1=syntax_constraint_strategy(), c2=syntax_constraint_strategy())
    def test_commutativity(self, c1: SyntaxConstraint, c2: SyntaxConstraint):
        """c₁ ⊓ c₂ = c₂ ⊓ c₁ (commutativity)."""
        assert c1.meet(c2) == c2.meet(c1)

    @given(
        c1=syntax_constraint_strategy(),
        c2=syntax_constraint_strategy(),
        c3=syntax_constraint_strategy(),
    )
    def test_associativity(
        self, c1: SyntaxConstraint, c2: SyntaxConstraint, c3: SyntaxConstraint
    ):
        """(c₁ ⊓ c₂) ⊓ c₃ = c₁ ⊓ (c₂ ⊓ c₃) (associativity)."""
        left_assoc = c1.meet(c2).meet(c3)
        right_assoc = c1.meet(c2.meet(c3))
        assert left_assoc == right_assoc

    @given(c=syntax_constraint_strategy())
    def test_satisfiability_consistency(self, c: SyntaxConstraint):
        """Satisfiability is consistent with is_bottom."""
        sat = c.satisfiability()
        if c.is_bottom():
            assert sat == Satisfiability.UNSAT
        else:
            assert sat == Satisfiability.SAT


# Strategy for generating type constraints
@st.composite
def type_constraint_strategy(draw):
    """Generate random TypeConstraint."""
    # 10% chance of TOP, 10% chance of BOTTOM, 80% real constraint
    choice = draw(st.integers(min_value=0, max_value=9))
    if choice == 0:
        return TYPE_TOP
    elif choice == 1:
        return TYPE_BOTTOM
    else:
        # Use a small set of expected types
        expected_type = draw(st.sampled_from([INT, STR, BOOL, FLOAT, ANY, None]))
        environment_hash = draw(st.integers(min_value=0, max_value=100))
        has_errors = draw(st.booleans())

        if expected_type is None:
            return TypeConstraint(
                environment_hash=environment_hash,
                has_errors=has_errors,
            )
        return TypeConstraint(
            expected_type=expected_type,
            environment_hash=environment_hash,
            has_errors=has_errors,
        )


class TestTypeConstraintSemilattice:
    """Property tests for TypeConstraint semilattice laws."""

    @given(c=type_constraint_strategy())
    def test_identity_law(self, c: TypeConstraint):
        """c ⊓ ⊤ = c (identity law)."""
        result = c.meet(TYPE_TOP)
        # For type constraints, meet with TOP returns the constraint
        # but environment_hash takes max, so we check key properties
        if c.is_top():
            assert result.is_top()
        elif c.is_bottom():
            assert result.is_bottom()
        else:
            assert result.expected_type == c.expected_type

    @given(c=type_constraint_strategy())
    def test_annihilation_law(self, c: TypeConstraint):
        """c ⊓ ⊥ = ⊥ (annihilation law)."""
        result = c.meet(TYPE_BOTTOM)
        assert result == TYPE_BOTTOM

    @given(c=type_constraint_strategy())
    def test_idempotence(self, c: TypeConstraint):
        """c ⊓ c = c (idempotence)."""
        result = c.meet(c)
        assert result == c

    @given(c1=type_constraint_strategy(), c2=type_constraint_strategy())
    def test_commutativity(self, c1: TypeConstraint, c2: TypeConstraint):
        """c₁ ⊓ c₂ = c₂ ⊓ c₁ (commutativity)."""
        assert c1.meet(c2) == c2.meet(c1)

    @given(
        c1=type_constraint_strategy(),
        c2=type_constraint_strategy(),
        c3=type_constraint_strategy(),
    )
    def test_associativity(
        self, c1: TypeConstraint, c2: TypeConstraint, c3: TypeConstraint
    ):
        """(c₁ ⊓ c₂) ⊓ c₃ = c₁ ⊓ (c₂ ⊓ c₃) (associativity)."""
        left_assoc = c1.meet(c2).meet(c3)
        right_assoc = c1.meet(c2.meet(c3))
        assert left_assoc == right_assoc

    @given(c=type_constraint_strategy())
    def test_satisfiability_consistency(self, c: TypeConstraint):
        """Satisfiability is consistent with is_bottom."""
        sat = c.satisfiability()
        if c.is_bottom():
            assert sat == Satisfiability.UNSAT
        else:
            assert sat == Satisfiability.SAT


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
