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
