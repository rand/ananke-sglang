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
"""Tests for the Semantic Domain.

Tests cover:
- SemanticConstraint semilattice laws
- SMTFormula creation and properties
- SMT solver integration
- Formula extraction
- SemanticDomain functionality
"""

from __future__ import annotations

import pytest
import torch
from typing import List

try:
    from ...domains.semantics import (
        SemanticConstraint,
        SMTFormula,
        FormulaKind,
        SEMANTIC_TOP,
        SEMANTIC_BOTTOM,
        semantic_assertion,
        semantic_precondition,
        semantic_postcondition,
        SemanticDomain,
        SemanticDomainCheckpoint,
        VariableBounds,
        ExpressionState,
        ExpressionContext,
        ContextConfidence,
        BoundsConfidence,
        BlockingLevel,
        SMTSolver,
        SimpleSMTSolver,
        IncrementalSMTSolver,
        SMTResult,
        SMTModel,
        SMTCheckResult,
        is_z3_available,
        create_smt_solver,
        PythonAssertExtractor,
        DocstringContractExtractor,
        TypeAnnotationExtractor,
        CompositeExtractor,
        extract_formulas,
        extract_assertions,
        extract_contracts,
    )
    from ...core.constraint import Satisfiability
    from ...core.domain import GenerationContext
except ImportError:
    from domains.semantics import (
        SemanticConstraint,
        SMTFormula,
        FormulaKind,
        SEMANTIC_TOP,
        SEMANTIC_BOTTOM,
        semantic_assertion,
        semantic_precondition,
        semantic_postcondition,
        SemanticDomain,
        SemanticDomainCheckpoint,
        VariableBounds,
        ExpressionState,
        ExpressionContext,
        ContextConfidence,
        BoundsConfidence,
        BlockingLevel,
        SMTSolver,
        SimpleSMTSolver,
        IncrementalSMTSolver,
        SMTResult,
        SMTModel,
        SMTCheckResult,
        is_z3_available,
        create_smt_solver,
        PythonAssertExtractor,
        DocstringContractExtractor,
        TypeAnnotationExtractor,
        CompositeExtractor,
        extract_formulas,
        extract_assertions,
        extract_contracts,
    )
    from core.constraint import Satisfiability
    from core.domain import GenerationContext


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def generation_context() -> GenerationContext:
    """Create a generation context for testing."""
    return GenerationContext(
        vocab_size=100,
        device=torch.device("cpu"),
        tokenizer=None,
        generated_text="",
        position=0,
    )


@pytest.fixture
def python_domain() -> SemanticDomain:
    """Create a Python semantic domain."""
    return SemanticDomain(language="python", use_z3=False)


@pytest.fixture
def simple_solver() -> IncrementalSMTSolver:
    """Create a simple SMT solver."""
    return create_smt_solver(use_z3=False)


# =============================================================================
# SMTFormula Tests
# =============================================================================


class TestSMTFormula:
    """Tests for SMTFormula dataclass."""

    def test_create_simple(self) -> None:
        """Test creating simple SMTFormula."""
        formula = SMTFormula(expression="x > 0")
        assert formula.expression == "x > 0"
        assert formula.kind == FormulaKind.ASSERTION
        assert formula.source is None
        assert formula.name is None

    def test_create_full(self) -> None:
        """Test creating SMTFormula with all fields."""
        formula = SMTFormula(
            expression="x > 0",
            kind=FormulaKind.PRECONDITION,
            source="func.py:10",
            name="positive_x",
        )
        assert formula.expression == "x > 0"
        assert formula.kind == FormulaKind.PRECONDITION
        assert formula.source == "func.py:10"
        assert formula.name == "positive_x"

    def test_equality(self) -> None:
        """Test SMTFormula equality."""
        f1 = SMTFormula(expression="x > 0", kind=FormulaKind.ASSERTION)
        f2 = SMTFormula(expression="x > 0", kind=FormulaKind.ASSERTION)
        f3 = SMTFormula(expression="x < 0", kind=FormulaKind.ASSERTION)

        assert f1 == f2
        assert f1 != f3

    def test_hashable(self) -> None:
        """Test SMTFormula is hashable."""
        formula = SMTFormula(expression="x > 0")
        s = {formula}
        assert formula in s

    def test_negate(self) -> None:
        """Test formula negation."""
        formula = SMTFormula(expression="x > 0", name="positive")
        negated = formula.negate()

        assert "not" in negated.expression.lower()
        assert "x > 0" in negated.expression

    def test_repr(self) -> None:
        """Test string representation."""
        formula = SMTFormula(expression="x > 0", kind=FormulaKind.PRECONDITION)
        assert "x > 0" in repr(formula)
        assert "PRECONDITION" in repr(formula)


# =============================================================================
# SemanticConstraint Tests
# =============================================================================


class TestSemanticConstraint:
    """Tests for SemanticConstraint."""

    def test_top_is_top(self) -> None:
        """Test TOP constraint is_top."""
        assert SEMANTIC_TOP.is_top()
        assert not SEMANTIC_TOP.is_bottom()

    def test_bottom_is_bottom(self) -> None:
        """Test BOTTOM constraint is_bottom."""
        assert SEMANTIC_BOTTOM.is_bottom()
        assert not SEMANTIC_BOTTOM.is_top()

    def test_empty_constraint_satisfiable(self) -> None:
        """Test empty constraint is satisfiable."""
        c = SemanticConstraint()
        assert c.satisfiability() == Satisfiability.SAT
        assert not c.is_top()
        assert not c.is_bottom()

    def test_add_assertion(self) -> None:
        """Test adding assertion."""
        c = semantic_assertion("x > 0", "positive_x")
        assert c.formula_count() == 1
        assert len(c.get_assertions()) == 1

    def test_add_precondition(self) -> None:
        """Test adding precondition."""
        c = semantic_precondition("n >= 0", "non_negative")
        assert c.formula_count() == 1
        assert len(c.get_preconditions()) == 1

    def test_add_postcondition(self) -> None:
        """Test adding postcondition."""
        c = semantic_postcondition("result > input", "increased")
        assert c.formula_count() == 1
        assert len(c.get_postconditions()) == 1

    def test_meet_with_top(self) -> None:
        """Test meet with TOP returns other constraint."""
        c = semantic_assertion("x > 0")
        assert c.meet(SEMANTIC_TOP) == c
        assert SEMANTIC_TOP.meet(c) == c

    def test_meet_with_bottom(self) -> None:
        """Test meet with BOTTOM returns BOTTOM."""
        c = semantic_assertion("x > 0")
        assert c.meet(SEMANTIC_BOTTOM).is_bottom()
        assert SEMANTIC_BOTTOM.meet(c).is_bottom()

    def test_meet_combines_formulas(self) -> None:
        """Test meet combines formula sets."""
        c1 = semantic_assertion("x > 0")
        c2 = semantic_assertion("y > 0")
        result = c1.meet(c2)

        assert result.formula_count() == 2

    def test_add_multiple_formulas(self) -> None:
        """Test adding multiple formulas."""
        c = (
            SemanticConstraint()
            .add_assertion("x > 0")
            .add_assertion("x < 10")
            .add_precondition("n >= 0")
        )

        assert c.formula_count() == 3
        assert len(c.get_assertions()) == 2
        assert len(c.get_preconditions()) == 1

    def test_add_assumption(self) -> None:
        """Test adding assumption."""
        c = SemanticConstraint().add_assumption("positive_mode")
        assert c.assumption_count() == 1

    def test_get_formulas_by_kind(self) -> None:
        """Test filtering formulas by kind."""
        c = (
            SemanticConstraint()
            .add_assertion("a1")
            .add_assertion("a2")
            .add_precondition("p1")
            .add_postcondition("post1")
        )

        assertions = c.get_formulas_by_kind(FormulaKind.ASSERTION)
        assert len(assertions) == 2

        preconditions = c.get_formulas_by_kind(FormulaKind.PRECONDITION)
        assert len(preconditions) == 1


class TestSemanticConstraintSemilattice:
    """Verify semilattice laws for SemanticConstraint."""

    def test_idempotent(self) -> None:
        """Test c.meet(c) == c."""
        c = semantic_assertion("x > 0").meet(semantic_assertion("y > 0"))
        assert c.meet(c) == c

    def test_commutative(self) -> None:
        """Test c1.meet(c2) == c2.meet(c1)."""
        c1 = semantic_assertion("x > 0")
        c2 = semantic_assertion("y > 0")

        r1 = c1.meet(c2)
        r2 = c2.meet(c1)

        assert r1.formulas == r2.formulas

    def test_associative(self) -> None:
        """Test (c1.meet(c2)).meet(c3) == c1.meet(c2.meet(c3))."""
        c1 = semantic_assertion("x > 0")
        c2 = semantic_assertion("y > 0")
        c3 = semantic_assertion("z > 0")

        left = (c1.meet(c2)).meet(c3)
        right = c1.meet(c2.meet(c3))

        assert left.formulas == right.formulas

    def test_top_identity(self) -> None:
        """Test c.meet(TOP) == c."""
        c = semantic_assertion("x > 0")
        assert c.meet(SEMANTIC_TOP) == c

    def test_bottom_absorbing(self) -> None:
        """Test c.meet(BOTTOM) == BOTTOM."""
        c = semantic_assertion("x > 0")
        assert c.meet(SEMANTIC_BOTTOM).is_bottom()


# =============================================================================
# SMT Solver Tests
# =============================================================================


class TestSMTResult:
    """Tests for SMTResult enum."""

    def test_result_values(self) -> None:
        """Test SMTResult values exist."""
        assert SMTResult.SAT is not None
        assert SMTResult.UNSAT is not None
        assert SMTResult.UNKNOWN is not None
        assert SMTResult.TIMEOUT is not None
        assert SMTResult.ERROR is not None


class TestSMTCheckResult:
    """Tests for SMTCheckResult."""

    def test_sat_result(self) -> None:
        """Test SAT result."""
        result = SMTCheckResult(result=SMTResult.SAT)
        assert result.is_sat
        assert not result.is_unsat
        assert not result.is_unknown

    def test_unsat_result(self) -> None:
        """Test UNSAT result."""
        result = SMTCheckResult(result=SMTResult.UNSAT)
        assert result.is_unsat
        assert not result.is_sat
        assert not result.is_unknown

    def test_unknown_result(self) -> None:
        """Test UNKNOWN result."""
        result = SMTCheckResult(result=SMTResult.UNKNOWN)
        assert result.is_unknown
        assert not result.is_sat
        assert not result.is_unsat


class TestSimpleSMTSolver:
    """Tests for SimpleSMTSolver."""

    def test_empty_formulas_sat(self) -> None:
        """Test empty formula set is SAT."""
        solver = SimpleSMTSolver()
        result = solver.check([])
        assert result.is_sat

    def test_trivially_false_unsat(self) -> None:
        """Test trivially false formula is UNSAT."""
        solver = SimpleSMTSolver()
        formula = SMTFormula(expression="false")
        result = solver.check([formula])
        assert result.is_unsat

    def test_simple_contradiction_unsat(self) -> None:
        """Test simple contradiction is UNSAT."""
        solver = SimpleSMTSolver()
        f1 = SMTFormula(expression="x > 0")
        f2 = SMTFormula(expression="not (x > 0)")
        result = solver.check([f1, f2])
        assert result.is_unsat

    def test_non_trivial_unknown(self) -> None:
        """Test non-trivial formula returns UNKNOWN."""
        solver = SimpleSMTSolver()
        formula = SMTFormula(expression="x > 0 and y < 10")
        result = solver.check([formula])
        assert result.is_unknown


class TestIncrementalSMTSolver:
    """Tests for IncrementalSMTSolver."""

    def test_create_without_z3(self, simple_solver: IncrementalSMTSolver) -> None:
        """Test creating solver without Z3."""
        assert not simple_solver.using_z3

    def test_push_pop(self, simple_solver: IncrementalSMTSolver) -> None:
        """Test push/pop operations."""
        initial_depth = simple_solver.depth()

        simple_solver.push()
        assert simple_solver.depth() == initial_depth + 1

        simple_solver.pop()
        assert simple_solver.depth() == initial_depth

    def test_add_formula(self, simple_solver: IncrementalSMTSolver) -> None:
        """Test adding formulas."""
        formula = SMTFormula(expression="x > 0")
        simple_solver.add_formula(formula)
        # Should not raise

    def test_check_constraint(self, simple_solver: IncrementalSMTSolver) -> None:
        """Test checking constraint."""
        constraint = semantic_assertion("x > 0")
        result = simple_solver.check_constraint(constraint)
        assert result.result in [SMTResult.SAT, SMTResult.UNSAT, SMTResult.UNKNOWN]

    def test_reset(self, simple_solver: IncrementalSMTSolver) -> None:
        """Test reset operation."""
        simple_solver.push()
        simple_solver.push()
        simple_solver.reset()
        assert simple_solver.depth() == 1  # Back to initial


# =============================================================================
# Formula Extractor Tests
# =============================================================================


class TestPythonAssertExtractor:
    """Tests for PythonAssertExtractor."""

    def test_simple_assert(self) -> None:
        """Test extracting simple assert."""
        extractor = PythonAssertExtractor()
        source = "assert x > 0\n"
        result = extractor.extract(source)

        assert len(result.formulas) == 1
        assert result.formulas[0].expression == "x > 0"
        assert result.formulas[0].kind == FormulaKind.ASSERTION

    def test_assert_with_message(self) -> None:
        """Test extracting assert with message."""
        extractor = PythonAssertExtractor()
        source = 'assert x > 0, "x must be positive"\n'
        result = extractor.extract(source)

        assert len(result.formulas) == 1
        assert result.formulas[0].name == "x must be positive"

    def test_multiple_asserts(self) -> None:
        """Test extracting multiple asserts."""
        extractor = PythonAssertExtractor()
        source = """
assert x > 0
assert y < 10
assert z != 0
"""
        result = extractor.extract(source)
        assert len(result.formulas) == 3


class TestDocstringContractExtractor:
    """Tests for DocstringContractExtractor."""

    def test_precondition(self) -> None:
        """Test extracting precondition."""
        extractor = DocstringContractExtractor()
        source = ":pre: x > 0\n"
        result = extractor.extract(source)

        assert len(result.formulas) == 1
        assert result.formulas[0].kind == FormulaKind.PRECONDITION

    def test_postcondition(self) -> None:
        """Test extracting postcondition."""
        extractor = DocstringContractExtractor()
        source = ":post: result > input\n"
        result = extractor.extract(source)

        assert len(result.formulas) == 1
        assert result.formulas[0].kind == FormulaKind.POSTCONDITION

    def test_invariant(self) -> None:
        """Test extracting invariant."""
        extractor = DocstringContractExtractor()
        source = ":invariant: count >= 0\n"
        result = extractor.extract(source)

        assert len(result.formulas) == 1
        assert result.formulas[0].kind == FormulaKind.INVARIANT

    def test_requires_format(self) -> None:
        """Test 'Requires:' format."""
        extractor = DocstringContractExtractor()
        source = "Requires: n >= 0\n"
        result = extractor.extract(source)

        assert len(result.formulas) == 1
        assert result.formulas[0].kind == FormulaKind.PRECONDITION

    def test_ensures_format(self) -> None:
        """Test 'Ensures:' format."""
        extractor = DocstringContractExtractor()
        source = "Ensures: result is not None\n"
        result = extractor.extract(source)

        assert len(result.formulas) == 1
        assert result.formulas[0].kind == FormulaKind.POSTCONDITION


class TestTypeAnnotationExtractor:
    """Tests for TypeAnnotationExtractor."""

    def test_literal_annotation(self) -> None:
        """Test extracting Literal annotation."""
        extractor = TypeAnnotationExtractor()
        source = "x: Literal[1, 2, 3]"
        result = extractor.extract(source)

        assert len(result.formulas) == 1
        assert "x" in result.formulas[0].expression
        assert "1, 2, 3" in result.formulas[0].expression

    def test_gt_annotation(self) -> None:
        """Test extracting Gt annotation."""
        extractor = TypeAnnotationExtractor()
        source = "x: Annotated[int, Gt(0)]"
        result = extractor.extract(source)

        assert len(result.formulas) == 1
        assert ">" in result.formulas[0].expression


class TestCompositeExtractor:
    """Tests for CompositeExtractor."""

    def test_combines_all_extractors(self) -> None:
        """Test composite extractor uses all sub-extractors."""
        extractor = CompositeExtractor()
        source = """
assert x > 0
:pre: n >= 0
"""
        result = extractor.extract(source)

        # Should find both the assert and the precondition
        assert len(result.formulas) >= 2


class TestExtractorConvenienceFunctions:
    """Tests for convenience extraction functions."""

    def test_extract_formulas(self) -> None:
        """Test extract_formulas function."""
        source = "assert x > 0\n"
        formulas = extract_formulas(source)
        assert len(formulas) >= 1

    def test_extract_assertions(self) -> None:
        """Test extract_assertions function."""
        source = "assert x > 0\nassert y < 10\n"
        formulas = extract_assertions(source)
        assert len(formulas) == 2

    def test_extract_contracts(self) -> None:
        """Test extract_contracts function."""
        source = ":pre: x > 0\n:post: result > 0\n"
        formulas = extract_contracts(source)
        assert len(formulas) == 2


# =============================================================================
# SemanticDomain Tests
# =============================================================================


class TestSemanticDomain:
    """Tests for SemanticDomain."""

    def test_domain_name(self, python_domain: SemanticDomain) -> None:
        """Test domain name is 'semantics'."""
        assert python_domain.name == "semantics"

    def test_domain_language(self, python_domain: SemanticDomain) -> None:
        """Test domain language."""
        assert python_domain.language == "python"

    def test_top_and_bottom(self, python_domain: SemanticDomain) -> None:
        """Test domain provides TOP and BOTTOM."""
        assert python_domain.top.is_top()
        assert python_domain.bottom.is_bottom()

    def test_create_constraint_empty(self, python_domain: SemanticDomain) -> None:
        """Test creating empty constraint returns TOP."""
        c = python_domain.create_constraint()
        assert c.is_top()

    def test_create_constraint_with_assertions(self, python_domain: SemanticDomain) -> None:
        """Test creating constraint with assertions."""
        c = python_domain.create_constraint(assertions=["x > 0", "y < 10"])
        assert c.formula_count() == 2

    def test_create_constraint_with_all(self, python_domain: SemanticDomain) -> None:
        """Test creating constraint with all types."""
        c = python_domain.create_constraint(
            assertions=["x > 0"],
            preconditions=["n >= 0"],
            postconditions=["result > 0"],
            assumptions=["positive_mode"],
        )
        assert c.formula_count() == 3  # assertions + pre + post
        assert c.assumption_count() == 1


class TestSemanticDomainTokenMask:
    """Tests for SemanticDomain.token_mask()."""

    def test_top_allows_all(
        self,
        python_domain: SemanticDomain,
        generation_context: GenerationContext,
    ) -> None:
        """Test TOP constraint allows all tokens."""
        mask = python_domain.token_mask(SEMANTIC_TOP, generation_context)
        assert mask.all()

    def test_bottom_blocks_all(
        self,
        python_domain: SemanticDomain,
        generation_context: GenerationContext,
    ) -> None:
        """Test BOTTOM constraint blocks all tokens."""
        mask = python_domain.token_mask(SEMANTIC_BOTTOM, generation_context)
        assert not mask.any()

    def test_regular_constraint_conservative(
        self,
        python_domain: SemanticDomain,
        generation_context: GenerationContext,
    ) -> None:
        """Test regular constraint currently allows all (conservative)."""
        c = semantic_assertion("x > 0")
        mask = python_domain.token_mask(c, generation_context)
        # Current implementation is conservative
        assert mask.all()


class TestSemanticDomainCheckpoint:
    """Tests for SemanticDomain checkpoint/restore."""

    def test_checkpoint_preserves_state(self, python_domain: SemanticDomain) -> None:
        """Test checkpoint preserves domain state."""
        python_domain._accumulated_formulas.append(
            SMTFormula(expression="x > 0")
        )
        python_domain._token_buffer = "test"

        checkpoint = python_domain.checkpoint()

        assert len(checkpoint.formulas) == 1
        assert checkpoint.token_buffer == "test"

    def test_restore_reverts_state(self, python_domain: SemanticDomain) -> None:
        """Test restore reverts domain state."""
        python_domain._accumulated_formulas.append(
            SMTFormula(expression="before")
        )
        checkpoint = python_domain.checkpoint()

        python_domain._accumulated_formulas.append(
            SMTFormula(expression="after")
        )
        assert python_domain.formula_count == 2

        python_domain.restore(checkpoint)
        assert python_domain.formula_count == 1

    def test_restore_wrong_type_raises(self, python_domain: SemanticDomain) -> None:
        """Test restore with wrong type raises TypeError."""
        with pytest.raises(TypeError):
            python_domain.restore("not a checkpoint")  # type: ignore

    def test_satisfiability_check(self, python_domain: SemanticDomain) -> None:
        """Test satisfiability returns correct status."""
        c = semantic_assertion("x > 0")
        # Simple solver returns UNKNOWN for non-trivial formulas
        result = python_domain.satisfiability(c)
        assert result in [Satisfiability.SAT, Satisfiability.UNKNOWN]

        assert python_domain.satisfiability(SEMANTIC_TOP) == Satisfiability.SAT
        assert python_domain.satisfiability(SEMANTIC_BOTTOM) == Satisfiability.UNSAT


class TestSemanticDomainSolverOperations:
    """Tests for SemanticDomain solver operations."""

    def test_push_pop(self, python_domain: SemanticDomain) -> None:
        """Test push/pop operations."""
        python_domain.push()
        python_domain.pop()
        # Should not raise

    def test_reset(self, python_domain: SemanticDomain) -> None:
        """Test reset operation."""
        python_domain._accumulated_formulas.append(
            SMTFormula(expression="x > 0")
        )
        python_domain._token_buffer = "test"

        python_domain.reset()

        assert python_domain.formula_count == 0
        assert python_domain._token_buffer == ""


# =============================================================================
# Integration Tests
# =============================================================================


class TestSemanticDomainIntegration:
    """Integration tests for Semantic Domain."""

    def test_check_satisfiability(self, python_domain: SemanticDomain) -> None:
        """Test check_satisfiability method."""
        constraint = python_domain.create_constraint(
            assertions=["x > 0", "x < 10"]
        )
        result = python_domain.check_satisfiability(constraint)
        # Should return a valid result
        assert result.result in [SMTResult.SAT, SMTResult.UNSAT, SMTResult.UNKNOWN]

    def test_checkpoint_restore_cycle(self, python_domain: SemanticDomain) -> None:
        """Test multiple checkpoint/restore cycles."""
        python_domain._accumulated_formulas.append(
            SMTFormula(expression="f1")
        )
        cp1 = python_domain.checkpoint()

        python_domain._accumulated_formulas.append(
            SMTFormula(expression="f2")
        )
        cp2 = python_domain.checkpoint()

        python_domain._accumulated_formulas.append(
            SMTFormula(expression="f3")
        )

        # Restore to cp2
        python_domain.restore(cp2)
        assert python_domain.formula_count == 2

        # Restore to cp1
        python_domain.restore(cp1)
        assert python_domain.formula_count == 1


# =============================================================================
# Context-Aware Bounds Checking Tests
# =============================================================================


class TestExpressionContext:
    """Tests for ExpressionContext enum."""

    def test_context_values(self) -> None:
        """Test all context values exist."""
        assert ExpressionContext.NONE is not None
        assert ExpressionContext.SIMPLE_ASSIGNMENT_RHS is not None
        assert ExpressionContext.COMPOUND_EXPR is not None
        assert ExpressionContext.FUNCTION_CALL is not None
        assert ExpressionContext.SUBSCRIPT is not None
        assert ExpressionContext.CONDITIONAL is not None
        assert ExpressionContext.LIST_LITERAL is not None
        assert ExpressionContext.DICT_LITERAL is not None
        assert ExpressionContext.ATTRIBUTE_ACCESS is not None


class TestContextConfidence:
    """Tests for ContextConfidence enum."""

    def test_confidence_ordering(self) -> None:
        """Test confidence level ordering."""
        # HIGH should be most confident (lowest value)
        assert ContextConfidence.HIGH.value < ContextConfidence.MEDIUM.value
        assert ContextConfidence.MEDIUM.value < ContextConfidence.LOW.value
        assert ContextConfidence.LOW.value < ContextConfidence.NONE.value


class TestBoundsConfidence:
    """Tests for BoundsConfidence enum."""

    def test_bounds_confidence_values(self) -> None:
        """Test bounds confidence values."""
        assert BoundsConfidence.HIGH is not None
        assert BoundsConfidence.MEDIUM is not None
        assert BoundsConfidence.LOW is not None
        assert BoundsConfidence.UNKNOWN is not None


class TestBlockingLevel:
    """Tests for BlockingLevel enum."""

    def test_blocking_level_values(self) -> None:
        """Test blocking level values."""
        assert BlockingLevel.AGGRESSIVE is not None
        assert BlockingLevel.CONSERVATIVE is not None
        assert BlockingLevel.PERMISSIVE is not None


class TestExpressionState:
    """Tests for ExpressionState dataclass."""

    def test_default_state(self) -> None:
        """Test default ExpressionState values."""
        state = ExpressionState()
        assert state.context == ExpressionContext.NONE
        assert state.context_confidence == ContextConfidence.NONE
        assert state.target_variable is None
        assert state.paren_depth == 0
        assert state.bracket_depth == 0
        assert state.brace_depth == 0
        assert state.tokens_since_assignment == 0

    def test_is_direct_literal_position(self) -> None:
        """Test is_direct_literal_position method."""
        # Direct assignment position
        state = ExpressionState(
            context=ExpressionContext.SIMPLE_ASSIGNMENT_RHS,
            context_confidence=ContextConfidence.HIGH,
            tokens_since_assignment=1,
        )
        assert state.is_direct_literal_position()

        # Too many tokens since assignment
        state.tokens_since_assignment = 5
        assert not state.is_direct_literal_position()

        # Inside parentheses
        state.tokens_since_assignment = 1
        state.paren_depth = 1
        assert not state.is_direct_literal_position()

        # Wrong context
        state.paren_depth = 0
        state.context = ExpressionContext.COMPOUND_EXPR
        assert not state.is_direct_literal_position()

    def test_copy(self) -> None:
        """Test ExpressionState copy method."""
        original = ExpressionState(
            context=ExpressionContext.FUNCTION_CALL,
            context_confidence=ContextConfidence.NONE,
            target_variable="x",
            paren_depth=2,
            tokens_since_assignment=5,
        )
        copy = original.copy()

        assert copy.context == original.context
        assert copy.context_confidence == original.context_confidence
        assert copy.target_variable == original.target_variable
        assert copy.paren_depth == original.paren_depth
        assert copy.tokens_since_assignment == original.tokens_since_assignment

        # Ensure it's a new object
        copy.paren_depth = 0
        assert original.paren_depth == 2


class TestVariableBoundsConfidence:
    """Tests for VariableBounds confidence features."""

    def test_effective_confidence_no_uncertainty(self) -> None:
        """Test effective_confidence without SMT uncertainty."""
        bounds = VariableBounds(
            lower=0,
            upper=100,
            confidence=BoundsConfidence.HIGH,
            smt_uncertain=False,
        )
        assert bounds.effective_confidence() == BoundsConfidence.HIGH

    def test_effective_confidence_with_uncertainty(self) -> None:
        """Test effective_confidence with SMT uncertainty."""
        # HIGH -> MEDIUM when uncertain
        bounds = VariableBounds(
            lower=0,
            confidence=BoundsConfidence.HIGH,
            smt_uncertain=True,
        )
        assert bounds.effective_confidence() == BoundsConfidence.MEDIUM

        # MEDIUM -> LOW when uncertain
        bounds.confidence = BoundsConfidence.MEDIUM
        assert bounds.effective_confidence() == BoundsConfidence.LOW

        # LOW -> UNKNOWN when uncertain
        bounds.confidence = BoundsConfidence.LOW
        assert bounds.effective_confidence() == BoundsConfidence.UNKNOWN

    def test_is_clearly_violated(self) -> None:
        """Test is_clearly_violated for CONSERVATIVE mode."""
        # Positive lower bound, negative value
        bounds = VariableBounds(lower=5, upper=100)
        assert bounds.is_clearly_violated(-1)
        assert not bounds.is_clearly_violated(50)

        # Value far out of range
        bounds = VariableBounds(lower=10, upper=20)
        assert bounds.is_clearly_violated(-5)  # < lower - |lower|
        assert bounds.is_clearly_violated(50)  # > upper + |upper|
        assert not bounds.is_clearly_violated(5)  # Close but not clearly wrong


class TestBlockingLevelComputation:
    """Tests for _compute_blocking_level decision matrix."""

    def test_high_high_aggressive(self, python_domain: SemanticDomain) -> None:
        """Test HIGH context + HIGH bounds = AGGRESSIVE."""
        level = python_domain._compute_blocking_level(
            ContextConfidence.HIGH,
            BoundsConfidence.HIGH,
        )
        assert level == BlockingLevel.AGGRESSIVE

    def test_high_medium_conservative(self, python_domain: SemanticDomain) -> None:
        """Test HIGH context + MEDIUM bounds = CONSERVATIVE."""
        level = python_domain._compute_blocking_level(
            ContextConfidence.HIGH,
            BoundsConfidence.MEDIUM,
        )
        assert level == BlockingLevel.CONSERVATIVE

    def test_medium_high_conservative(self, python_domain: SemanticDomain) -> None:
        """Test MEDIUM context + HIGH bounds = CONSERVATIVE."""
        level = python_domain._compute_blocking_level(
            ContextConfidence.MEDIUM,
            BoundsConfidence.HIGH,
        )
        assert level == BlockingLevel.CONSERVATIVE

    def test_medium_medium_conservative(self, python_domain: SemanticDomain) -> None:
        """Test MEDIUM context + MEDIUM bounds = CONSERVATIVE."""
        level = python_domain._compute_blocking_level(
            ContextConfidence.MEDIUM,
            BoundsConfidence.MEDIUM,
        )
        assert level == BlockingLevel.CONSERVATIVE

    def test_low_anything_permissive(self, python_domain: SemanticDomain) -> None:
        """Test LOW context = PERMISSIVE (soundness)."""
        for bounds_conf in BoundsConfidence:
            level = python_domain._compute_blocking_level(
                ContextConfidence.LOW,
                bounds_conf,
            )
            assert level == BlockingLevel.PERMISSIVE

    def test_none_context_permissive(self, python_domain: SemanticDomain) -> None:
        """Test NONE context = PERMISSIVE (soundness)."""
        for bounds_conf in BoundsConfidence:
            level = python_domain._compute_blocking_level(
                ContextConfidence.NONE,
                bounds_conf,
            )
            assert level == BlockingLevel.PERMISSIVE

    def test_low_bounds_permissive(self, python_domain: SemanticDomain) -> None:
        """Test LOW bounds = PERMISSIVE (soundness)."""
        for context_conf in [ContextConfidence.HIGH, ContextConfidence.MEDIUM]:
            level = python_domain._compute_blocking_level(
                context_conf,
                BoundsConfidence.LOW,
            )
            assert level == BlockingLevel.PERMISSIVE

    def test_unknown_bounds_permissive(self, python_domain: SemanticDomain) -> None:
        """Test UNKNOWN bounds = PERMISSIVE (soundness)."""
        for context_conf in [ContextConfidence.HIGH, ContextConfidence.MEDIUM]:
            level = python_domain._compute_blocking_level(
                context_conf,
                BoundsConfidence.UNKNOWN,
            )
            assert level == BlockingLevel.PERMISSIVE


class TestExpressionStateTransitions:
    """Tests for expression state machine transitions."""

    def test_assignment_operator_transition(
        self, python_domain: SemanticDomain
    ) -> None:
        """Test transition on seeing = operator."""
        # Set up buffer with variable name
        python_domain._token_buffer = "x"
        python_domain._update_expression_state("=")

        state = python_domain._expression_state
        assert state.context == ExpressionContext.SIMPLE_ASSIGNMENT_RHS
        assert state.context_confidence == ContextConfidence.HIGH
        assert state.target_variable == "x"

    def test_function_call_transition(self, python_domain: SemanticDomain) -> None:
        """Test transition to FUNCTION_CALL on func(."""
        # Start in assignment context
        python_domain._token_buffer = "x"
        python_domain._update_expression_state("=")
        python_domain._token_buffer += " ="

        # Simulate "func("
        python_domain._token_buffer += " func"
        python_domain._update_expression_state("(")

        state = python_domain._expression_state
        assert state.context == ExpressionContext.FUNCTION_CALL
        assert state.context_confidence == ContextConfidence.NONE

    def test_list_literal_transition(self, python_domain: SemanticDomain) -> None:
        """Test transition to LIST_LITERAL on [."""
        # Start in assignment context
        python_domain._token_buffer = "x"
        python_domain._update_expression_state("=")
        python_domain._token_buffer += " ="

        # Simulate "["
        python_domain._update_expression_state("[")

        state = python_domain._expression_state
        assert state.context == ExpressionContext.LIST_LITERAL
        assert state.context_confidence == ContextConfidence.MEDIUM

    def test_statement_terminator_reset(self, python_domain: SemanticDomain) -> None:
        """Test state reset on statement terminator."""
        # Set up assignment context
        python_domain._token_buffer = "x"
        python_domain._update_expression_state("=")
        python_domain._token_buffer += " = 5"

        # Terminate statement
        python_domain._update_expression_state("\n")

        state = python_domain._expression_state
        assert state.context == ExpressionContext.NONE
        assert state.context_confidence == ContextConfidence.NONE
        assert state.target_variable is None

    def test_operator_downgrades_confidence(
        self, python_domain: SemanticDomain
    ) -> None:
        """Test that operators downgrade to COMPOUND_EXPR."""
        # Start in simple assignment
        python_domain._token_buffer = "x"
        python_domain._update_expression_state("=")
        python_domain._token_buffer += " = 5"

        # See an operator
        python_domain._update_expression_state("+")

        state = python_domain._expression_state
        assert state.context == ExpressionContext.COMPOUND_EXPR
        assert state.context_confidence == ContextConfidence.LOW

    def test_conditional_transition(self, python_domain: SemanticDomain) -> None:
        """Test transition to CONDITIONAL on 'if' keyword."""
        # Start in simple assignment
        python_domain._token_buffer = "x"
        python_domain._update_expression_state("=")
        python_domain._token_buffer += " = 5"

        # See 'if' keyword
        python_domain._update_expression_state("if")

        state = python_domain._expression_state
        assert state.context == ExpressionContext.CONDITIONAL
        assert state.context_confidence == ContextConfidence.MEDIUM


class TestCheckpointRestoreExpressionState:
    """Tests for checkpoint/restore of expression state."""

    def test_checkpoint_preserves_expression_state(
        self, python_domain: SemanticDomain
    ) -> None:
        """Test that checkpoint preserves expression state."""
        # Set up expression state
        python_domain._token_buffer = "x"
        python_domain._update_expression_state("=")
        python_domain._token_buffer += " ="

        checkpoint = python_domain.checkpoint()

        assert checkpoint.expression_state is not None
        assert checkpoint.expression_state.context == ExpressionContext.SIMPLE_ASSIGNMENT_RHS
        assert checkpoint.expression_state.target_variable == "x"

    def test_restore_reverts_expression_state(
        self, python_domain: SemanticDomain
    ) -> None:
        """Test that restore reverts expression state."""
        # Set up initial state
        python_domain._token_buffer = "x"
        python_domain._update_expression_state("=")
        python_domain._token_buffer += " ="

        checkpoint = python_domain.checkpoint()

        # Modify state
        python_domain._update_expression_state("+")
        assert python_domain._expression_state.context == ExpressionContext.COMPOUND_EXPR

        # Restore
        python_domain.restore(checkpoint)
        assert python_domain._expression_state.context == ExpressionContext.SIMPLE_ASSIGNMENT_RHS
        assert python_domain._expression_state.target_variable == "x"


# =============================================================================
# Z3 Formula Parsing Tests
# =============================================================================


class TestZ3FormulaParsing:
    """Tests for enhanced Z3 formula parsing."""

    @pytest.fixture
    def z3_solver(self):
        """Create a Z3 solver if available."""
        try:
            from domains.semantics.smt import Z3Solver, is_z3_available
            if not is_z3_available():
                pytest.skip("Z3 not available")
            return Z3Solver(timeout_ms=1000)
        except ImportError:
            pytest.skip("Z3 not available")

    def test_parse_boolean_literals(self, z3_solver) -> None:
        """Test parsing boolean literals."""
        import z3

        true_formula = z3_solver._parse_formula("true")
        assert z3.is_true(true_formula)

        false_formula = z3_solver._parse_formula("false")
        assert z3.is_false(false_formula)

        # Case insensitive
        true_upper = z3_solver._parse_formula("TRUE")
        assert z3.is_true(true_upper)

    def test_parse_integer_literals(self, z3_solver) -> None:
        """Test parsing integer literals."""
        import z3

        five = z3_solver._parse_formula("5")
        assert z3.is_int_value(five)
        assert five.as_long() == 5

        negative = z3_solver._parse_formula("-10")
        # Note: This may create a variable or expression
        # The parser handles unary minus

    def test_parse_comparison_operators(self, z3_solver) -> None:
        """Test parsing comparison operators."""
        # x > 5
        formula = z3_solver._parse_formula("x > 5")
        assert formula is not None

        # x <= 10
        formula = z3_solver._parse_formula("x <= 10")
        assert formula is not None

        # x == y
        formula = z3_solver._parse_formula("x == y")
        assert formula is not None

        # x != 0
        formula = z3_solver._parse_formula("x != 0")
        assert formula is not None

    def test_parse_arithmetic_operators(self, z3_solver) -> None:
        """Test parsing arithmetic operators."""
        # x + 5
        formula = z3_solver._parse_formula("x + 5")
        assert formula is not None

        # x - y
        formula = z3_solver._parse_formula("x - y")
        assert formula is not None

        # x * 2
        formula = z3_solver._parse_formula("x * 2")
        assert formula is not None

        # x / 3
        formula = z3_solver._parse_formula("x / 3")
        assert formula is not None

    def test_parse_logical_operators(self, z3_solver) -> None:
        """Test parsing logical operators."""
        # x > 0 and x < 10
        formula = z3_solver._parse_formula("x > 0 and x < 10")
        assert formula is not None

        # x == 0 or y == 0
        formula = z3_solver._parse_formula("x == 0 or y == 0")
        assert formula is not None

        # not x > 5
        formula = z3_solver._parse_formula("not x > 5")
        assert formula is not None

    def test_parse_parentheses(self, z3_solver) -> None:
        """Test parsing parenthesized expressions."""
        # (x + 5) > 10
        formula = z3_solver._parse_formula("(x + 5) > 10")
        assert formula is not None

        # (x > 0) and (y > 0)
        formula = z3_solver._parse_formula("(x > 0) and (y > 0)")
        assert formula is not None

    def test_parse_complex_expression(self, z3_solver) -> None:
        """Test parsing complex expressions."""
        # (x + y) * 2 >= 10 and z < 5
        formula = z3_solver._parse_formula("(x + y) * 2 >= 10 and z < 5")
        assert formula is not None

    def test_satisfiability_simple_sat(self, z3_solver) -> None:
        """Test satisfiability of simple SAT formula."""
        from domains.semantics.constraint import SMTFormula, FormulaKind
        from domains.semantics.smt import SMTResult

        formula = SMTFormula(expression="x > 5", kind=FormulaKind.ASSERTION)
        result = z3_solver.check([formula])

        assert result.result == SMTResult.SAT
        assert result.model is not None

    def test_satisfiability_simple_unsat(self, z3_solver) -> None:
        """Test satisfiability of simple UNSAT formula."""
        from domains.semantics.constraint import SMTFormula, FormulaKind
        from domains.semantics.smt import SMTResult

        # x > 5 and x < 3 is UNSAT
        f1 = SMTFormula(expression="x > 5", kind=FormulaKind.ASSERTION)
        f2 = SMTFormula(expression="x < 3", kind=FormulaKind.ASSERTION)
        result = z3_solver.check([f1, f2])

        assert result.result == SMTResult.UNSAT

    def test_satisfiability_complex_sat(self, z3_solver) -> None:
        """Test satisfiability of complex SAT formula."""
        from domains.semantics.constraint import SMTFormula, FormulaKind
        from domains.semantics.smt import SMTResult

        # x >= 0 and y >= 0 and x + y == 10
        f1 = SMTFormula(expression="x >= 0", kind=FormulaKind.ASSERTION)
        f2 = SMTFormula(expression="y >= 0", kind=FormulaKind.ASSERTION)
        f3 = SMTFormula(expression="x + y == 10", kind=FormulaKind.ASSERTION)
        result = z3_solver.check([f1, f2, f3])

        assert result.result == SMTResult.SAT
        assert result.model is not None

    def test_formula_caching(self, z3_solver) -> None:
        """Test that parsed formulas are cached."""
        # Parse the same formula twice
        formula1 = z3_solver._parse_formula("x > 5")
        formula2 = z3_solver._parse_formula("x > 5")

        # Should be the same object (cached)
        assert formula1 is formula2

    def test_parse_failure_fallback(self, z3_solver) -> None:
        """Test that parse failure falls back to boolean variable."""
        # Unparseable expression should create a boolean variable
        formula = z3_solver._parse_formula("this is not a valid expression!!!")
        assert formula is not None  # Falls back to Bool variable

    def test_variable_type_inference(self, z3_solver) -> None:
        """Test variable type inference from names."""
        import z3

        # is_valid should be Bool
        z3_solver._parse_formula("is_valid == true")
        assert z3.is_bool(z3_solver._variables.get("is_valid", None))

        # x should be Int (default)
        z3_solver.reset()
        z3_solver._parse_formula("x > 5")
        assert z3.is_int(z3_solver._variables.get("x", None))

    def test_reset_clears_cache(self, z3_solver) -> None:
        """Test that reset clears the formula cache."""
        z3_solver._parse_formula("x > 5")
        assert len(z3_solver._formula_cache) > 0

        z3_solver.reset()
        assert len(z3_solver._formula_cache) == 0
        assert len(z3_solver._variables) == 0
