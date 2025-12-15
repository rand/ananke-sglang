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
from domains.imports.constraint import (
    IMPORT_TOP,
    IMPORT_BOTTOM,
    ImportConstraint,
    ModuleSpec,
)
from domains.controlflow.constraint import (
    CONTROLFLOW_TOP,
    CONTROLFLOW_BOTTOM,
    ControlFlowConstraint,
    CodePoint,
    TerminationRequirement,
)
from domains.semantics.constraint import (
    SEMANTIC_TOP,
    SEMANTIC_BOTTOM,
    SemanticConstraint,
    SMTFormula,
    FormulaKind,
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
        """c ⊓ ⊤ = c (identity law).

        The meet() implementation returns self directly when other is TOP,
        so this should be exact equality (not just expected_type match).
        """
        result = c.meet(TYPE_TOP)
        assert result == c, f"Identity law violated: {c}.meet(TYPE_TOP) = {result} != {c}"

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


# ===========================================================================
# ImportConstraint Strategies and Tests
# ===========================================================================


@st.composite
def module_spec_strategy(draw):
    """Generate random ModuleSpec."""
    name = draw(st.sampled_from(["numpy", "pandas", "torch", "json", "os", "sys"]))
    version = draw(st.sampled_from([None, ">=1.0.0", "==2.0.0", "<3.0"]))
    alias = draw(st.sampled_from([None, "np", "pd", "th"]))
    return ModuleSpec(name=name, version=version, alias=alias)


@st.composite
def import_constraint_strategy(draw):
    """Generate random ImportConstraint."""
    # 10% chance of TOP, 10% chance of BOTTOM, 80% real constraint
    choice = draw(st.integers(min_value=0, max_value=9))
    if choice == 0:
        return IMPORT_TOP
    elif choice == 1:
        return IMPORT_BOTTOM
    else:
        # Generate non-conflicting required/forbidden sets
        # Use a small set of module names
        all_names = ["numpy", "pandas", "torch", "json", "os", "sys"]

        # Draw required modules (as ModuleSpecs)
        num_required = draw(st.integers(min_value=0, max_value=2))
        required_names = draw(st.lists(
            st.sampled_from(all_names),
            min_size=0,
            max_size=num_required,
            unique=True,
        ))
        required = frozenset(ModuleSpec(name=n) for n in required_names)

        # Draw forbidden modules (must not conflict with required)
        remaining_names = [n for n in all_names if n not in required_names]
        num_forbidden = draw(st.integers(min_value=0, max_value=2))
        forbidden = frozenset(draw(st.lists(
            st.sampled_from(remaining_names) if remaining_names else st.just("_none_"),
            min_size=0,
            max_size=min(num_forbidden, len(remaining_names)),
            unique=True,
        )))
        # Remove placeholder if present
        forbidden = frozenset(f for f in forbidden if f != "_none_")

        return ImportConstraint(required=required, forbidden=forbidden)


class TestImportConstraintSemilattice:
    """Property tests for ImportConstraint semilattice laws."""

    @given(c=import_constraint_strategy())
    def test_identity_law(self, c: ImportConstraint):
        """c ⊓ ⊤ = c (identity law)."""
        result = c.meet(IMPORT_TOP)
        assert result == c

    @given(c=import_constraint_strategy())
    def test_annihilation_law(self, c: ImportConstraint):
        """c ⊓ ⊥ = ⊥ (annihilation law)."""
        result = c.meet(IMPORT_BOTTOM)
        assert result == IMPORT_BOTTOM

    @given(c=import_constraint_strategy())
    def test_idempotence(self, c: ImportConstraint):
        """c ⊓ c = c (idempotence)."""
        result = c.meet(c)
        assert result == c

    @given(c1=import_constraint_strategy(), c2=import_constraint_strategy())
    def test_commutativity(self, c1: ImportConstraint, c2: ImportConstraint):
        """c₁ ⊓ c₂ = c₂ ⊓ c₁ (commutativity)."""
        assert c1.meet(c2) == c2.meet(c1)

    @given(
        c1=import_constraint_strategy(),
        c2=import_constraint_strategy(),
        c3=import_constraint_strategy(),
    )
    def test_associativity(
        self, c1: ImportConstraint, c2: ImportConstraint, c3: ImportConstraint
    ):
        """(c₁ ⊓ c₂) ⊓ c₃ = c₁ ⊓ (c₂ ⊓ c₃) (associativity)."""
        left_assoc = c1.meet(c2).meet(c3)
        right_assoc = c1.meet(c2.meet(c3))
        assert left_assoc == right_assoc


class TestImportConstraintProperties:
    """Additional property tests for ImportConstraint behavior."""

    @given(c=import_constraint_strategy())
    def test_satisfiability_consistency(self, c: ImportConstraint):
        """Satisfiability is consistent with is_bottom."""
        sat = c.satisfiability()
        if c.is_bottom():
            assert sat == Satisfiability.UNSAT
        else:
            assert sat == Satisfiability.SAT

    @given(c1=import_constraint_strategy(), c2=import_constraint_strategy())
    def test_meet_preserves_required(self, c1: ImportConstraint, c2: ImportConstraint):
        """Meet should include all required from both constraints."""
        result = c1.meet(c2)
        if not result.is_bottom():
            # All required from c1 and c2 should be in result
            for spec in c1.required:
                assert spec in result.required or result.is_bottom()
            for spec in c2.required:
                assert spec in result.required or result.is_bottom()


# ===========================================================================
# ControlFlowConstraint Strategies and Tests
# ===========================================================================


@st.composite
def code_point_strategy(draw):
    """Generate random CodePoint."""
    label = draw(st.sampled_from(["entry", "exit", "loop_start", "loop_end", "branch"]))
    index = draw(st.integers(min_value=0, max_value=10))
    kind = draw(st.sampled_from([None, "function", "loop", "conditional"]))
    line = draw(st.sampled_from([None, 10, 20, 30, 40]))
    return CodePoint(label=f"{label}_{index}", kind=kind, line=line)


@st.composite
def controlflow_constraint_strategy(draw):
    """Generate random ControlFlowConstraint."""
    # 10% chance of TOP, 10% chance of BOTTOM, 80% real constraint
    choice = draw(st.integers(min_value=0, max_value=9))
    if choice == 0:
        return CONTROLFLOW_TOP
    elif choice == 1:
        return CONTROLFLOW_BOTTOM
    else:
        # Generate unique code points for must_reach
        num_must_reach = draw(st.integers(min_value=0, max_value=3))
        must_reach_points = []
        for i in range(num_must_reach):
            # Use unique labels based on index to ensure uniqueness
            label = draw(st.sampled_from(["entry", "exit", "loop"]))
            must_reach_points.append(CodePoint(
                label=f"must_{label}_{i}",
                kind=draw(st.sampled_from([None, "function", "loop"])),
                line=draw(st.sampled_from([None, 10, 20])),
            ))
        must_reach = frozenset(must_reach_points)

        # Generate unique code points for must_not_reach (disjoint from must_reach)
        num_must_not = draw(st.integers(min_value=0, max_value=3))
        must_not_points = []
        for i in range(num_must_not):
            # Use distinct label prefix to ensure disjointness
            label = draw(st.sampled_from(["forbidden", "blocked", "avoid"]))
            must_not_points.append(CodePoint(
                label=f"not_{label}_{i}",
                kind=draw(st.sampled_from([None, "function", "loop"])),
                line=draw(st.sampled_from([None, 30, 40])),
            ))
        must_not_reach = frozenset(must_not_points)

        termination = draw(st.sampled_from(list(TerminationRequirement)))

        return ControlFlowConstraint(
            must_reach=must_reach,
            must_not_reach=must_not_reach,
            termination=termination,
        )


class TestControlFlowConstraintSemilattice:
    """Property tests for ControlFlowConstraint semilattice laws."""

    @given(c=controlflow_constraint_strategy())
    def test_identity_law(self, c: ControlFlowConstraint):
        """c ⊓ ⊤ = c (identity law)."""
        result = c.meet(CONTROLFLOW_TOP)
        assert result == c

    @given(c=controlflow_constraint_strategy())
    def test_annihilation_law(self, c: ControlFlowConstraint):
        """c ⊓ ⊥ = ⊥ (annihilation law)."""
        result = c.meet(CONTROLFLOW_BOTTOM)
        assert result == CONTROLFLOW_BOTTOM

    @given(c=controlflow_constraint_strategy())
    def test_idempotence(self, c: ControlFlowConstraint):
        """c ⊓ c = c (idempotence)."""
        result = c.meet(c)
        assert result == c

    @given(c1=controlflow_constraint_strategy(), c2=controlflow_constraint_strategy())
    def test_commutativity(
        self, c1: ControlFlowConstraint, c2: ControlFlowConstraint
    ):
        """c₁ ⊓ c₂ = c₂ ⊓ c₁ (commutativity)."""
        assert c1.meet(c2) == c2.meet(c1)

    @given(
        c1=controlflow_constraint_strategy(),
        c2=controlflow_constraint_strategy(),
        c3=controlflow_constraint_strategy(),
    )
    def test_associativity(
        self,
        c1: ControlFlowConstraint,
        c2: ControlFlowConstraint,
        c3: ControlFlowConstraint,
    ):
        """(c₁ ⊓ c₂) ⊓ c₃ = c₁ ⊓ (c₂ ⊓ c₃) (associativity)."""
        left_assoc = c1.meet(c2).meet(c3)
        right_assoc = c1.meet(c2.meet(c3))
        assert left_assoc == right_assoc


class TestControlFlowConstraintProperties:
    """Additional property tests for ControlFlowConstraint behavior."""

    @given(c=controlflow_constraint_strategy())
    def test_satisfiability_consistency(self, c: ControlFlowConstraint):
        """Satisfiability is consistent with is_bottom."""
        sat = c.satisfiability()
        if c.is_bottom():
            assert sat == Satisfiability.UNSAT
        else:
            assert sat == Satisfiability.SAT

    @given(c1=controlflow_constraint_strategy(), c2=controlflow_constraint_strategy())
    def test_meet_detects_conflicts(
        self, c1: ControlFlowConstraint, c2: ControlFlowConstraint
    ):
        """Meet should detect when must_reach and must_not_reach conflict."""
        result = c1.meet(c2)
        if not result.is_bottom() and not result.is_top():
            # No overlap between must_reach and must_not_reach
            assert not (result.must_reach & result.must_not_reach)


# ===========================================================================
# SemanticConstraint Strategies and Tests
# ===========================================================================


@st.composite
def smt_formula_strategy(draw):
    """Generate random SMTFormula."""
    expr = draw(st.sampled_from([
        "x > 0",
        "x < 10",
        "y >= 0",
        "y <= 100",
        "x + y > 0",
        "true",
    ]))
    kind = draw(st.sampled_from(list(FormulaKind)))
    return SMTFormula(expression=expr, kind=kind)


@st.composite
def semantic_constraint_strategy(draw):
    """Generate random SemanticConstraint."""
    # 10% chance of TOP, 10% chance of BOTTOM, 80% real constraint
    choice = draw(st.integers(min_value=0, max_value=9))
    if choice == 0:
        return SEMANTIC_TOP
    elif choice == 1:
        return SEMANTIC_BOTTOM
    else:
        # Generate some formulas
        num_formulas = draw(st.integers(min_value=0, max_value=3))
        formulas = frozenset(
            draw(smt_formula_strategy()) for _ in range(num_formulas)
        )

        num_assumptions = draw(st.integers(min_value=0, max_value=2))
        assumptions = frozenset(
            draw(smt_formula_strategy()) for _ in range(num_assumptions)
        )

        return SemanticConstraint(formulas=formulas, assumptions=assumptions)


class TestSemanticConstraintSemilattice:
    """Property tests for SemanticConstraint semilattice laws."""

    @given(c=semantic_constraint_strategy())
    def test_identity_law(self, c: SemanticConstraint):
        """c ⊓ ⊤ = c (identity law)."""
        result = c.meet(SEMANTIC_TOP)
        assert result == c

    @given(c=semantic_constraint_strategy())
    def test_annihilation_law(self, c: SemanticConstraint):
        """c ⊓ ⊥ = ⊥ (annihilation law)."""
        result = c.meet(SEMANTIC_BOTTOM)
        assert result == SEMANTIC_BOTTOM

    @given(c=semantic_constraint_strategy())
    def test_idempotence(self, c: SemanticConstraint):
        """c ⊓ c = c (idempotence)."""
        result = c.meet(c)
        assert result == c

    @given(c1=semantic_constraint_strategy(), c2=semantic_constraint_strategy())
    def test_commutativity(self, c1: SemanticConstraint, c2: SemanticConstraint):
        """c₁ ⊓ c₂ = c₂ ⊓ c₁ (commutativity)."""
        assert c1.meet(c2) == c2.meet(c1)

    @given(
        c1=semantic_constraint_strategy(),
        c2=semantic_constraint_strategy(),
        c3=semantic_constraint_strategy(),
    )
    def test_associativity(
        self, c1: SemanticConstraint, c2: SemanticConstraint, c3: SemanticConstraint
    ):
        """(c₁ ⊓ c₂) ⊓ c₃ = c₁ ⊓ (c₂ ⊓ c₃) (associativity)."""
        left_assoc = c1.meet(c2).meet(c3)
        right_assoc = c1.meet(c2.meet(c3))
        assert left_assoc == right_assoc


class TestSemanticConstraintProperties:
    """Additional property tests for SemanticConstraint behavior."""

    @given(c=semantic_constraint_strategy())
    def test_satisfiability_consistency(self, c: SemanticConstraint):
        """Satisfiability is consistent with is_bottom/is_top."""
        sat = c.satisfiability()
        if c.is_bottom():
            assert sat == Satisfiability.UNSAT
        elif c.is_top():
            assert sat == Satisfiability.SAT

    @given(c1=semantic_constraint_strategy(), c2=semantic_constraint_strategy())
    def test_meet_is_union_of_formulas(
        self, c1: SemanticConstraint, c2: SemanticConstraint
    ):
        """Meet should combine formulas as union."""
        result = c1.meet(c2)
        if not result.is_bottom() and not result.is_top():
            # Result formulas should contain all from both
            if not c1.is_top() and not c1.is_bottom():
                for formula in c1.formulas:
                    assert formula in result.formulas
            if not c2.is_top() and not c2.is_bottom():
                for formula in c2.formulas:
                    assert formula in result.formulas


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
