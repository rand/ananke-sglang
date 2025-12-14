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
"""Unit tests for SMT solver timeout handling.

Tests for proper timeout detection, graceful degradation,
and error recovery in the SMT solver integration.
"""

import pytest
import torch

from domains.semantics.smt import (
    SMTResult,
    SMTCheckResult,
    SMTFormula,
    SimpleSMTSolver,
    Z3Solver,
    IncrementalSMTSolver,
    create_smt_solver,
    create_timeout_triggering_formulas,
    is_z3_available,
)
from domains.semantics.constraint import (
    SemanticConstraint,
    FormulaKind,
    SEMANTIC_TOP,
    SEMANTIC_BOTTOM,
)


# ===========================================================================
# SMTCheckResult Property Tests
# ===========================================================================


class TestSMTCheckResultProperties:
    """Tests for SMTCheckResult convenience properties."""

    def test_is_sat(self):
        """is_sat should return True only for SAT result."""
        assert SMTCheckResult(result=SMTResult.SAT).is_sat
        assert not SMTCheckResult(result=SMTResult.UNSAT).is_sat
        assert not SMTCheckResult(result=SMTResult.UNKNOWN).is_sat
        assert not SMTCheckResult(result=SMTResult.TIMEOUT).is_sat
        assert not SMTCheckResult(result=SMTResult.ERROR).is_sat

    def test_is_unsat(self):
        """is_unsat should return True only for UNSAT result."""
        assert not SMTCheckResult(result=SMTResult.SAT).is_unsat
        assert SMTCheckResult(result=SMTResult.UNSAT).is_unsat
        assert not SMTCheckResult(result=SMTResult.UNKNOWN).is_unsat
        assert not SMTCheckResult(result=SMTResult.TIMEOUT).is_unsat
        assert not SMTCheckResult(result=SMTResult.ERROR).is_unsat

    def test_is_unknown(self):
        """is_unknown should return True only for UNKNOWN result."""
        assert not SMTCheckResult(result=SMTResult.SAT).is_unknown
        assert not SMTCheckResult(result=SMTResult.UNSAT).is_unknown
        assert SMTCheckResult(result=SMTResult.UNKNOWN).is_unknown
        assert not SMTCheckResult(result=SMTResult.TIMEOUT).is_unknown
        assert not SMTCheckResult(result=SMTResult.ERROR).is_unknown

    def test_is_timeout(self):
        """is_timeout should return True only for TIMEOUT result."""
        assert not SMTCheckResult(result=SMTResult.SAT).is_timeout
        assert not SMTCheckResult(result=SMTResult.UNSAT).is_timeout
        assert not SMTCheckResult(result=SMTResult.UNKNOWN).is_timeout
        assert SMTCheckResult(result=SMTResult.TIMEOUT).is_timeout
        assert not SMTCheckResult(result=SMTResult.ERROR).is_timeout

    def test_is_error(self):
        """is_error should return True only for ERROR result."""
        assert not SMTCheckResult(result=SMTResult.SAT).is_error
        assert not SMTCheckResult(result=SMTResult.UNSAT).is_error
        assert not SMTCheckResult(result=SMTResult.UNKNOWN).is_error
        assert not SMTCheckResult(result=SMTResult.TIMEOUT).is_error
        assert SMTCheckResult(result=SMTResult.ERROR).is_error


# ===========================================================================
# SimpleSMTSolver Tests
# ===========================================================================


class TestSimpleSMTSolver:
    """Tests for SimpleSMTSolver (fallback solver)."""

    @pytest.fixture
    def solver(self):
        return SimpleSMTSolver()

    def test_empty_formulas_is_sat(self, solver):
        """Empty formula list should be SAT."""
        result = solver.check([])
        assert result.is_sat

    def test_true_formula_is_sat(self, solver):
        """'true' formula should be SAT."""
        formulas = [SMTFormula(expression="true", kind=FormulaKind.ASSERTION)]
        result = solver.check(formulas)
        # Simple solver may return SAT or UNKNOWN for non-trivial
        assert result.result in (SMTResult.SAT, SMTResult.UNKNOWN)

    def test_false_formula_is_unsat(self, solver):
        """'false' formula should be UNSAT."""
        formulas = [SMTFormula(expression="false", kind=FormulaKind.ASSERTION)]
        result = solver.check(formulas)
        assert result.is_unsat

    def test_complex_formula_is_unknown(self, solver):
        """Complex formulas should return UNKNOWN."""
        formulas = [
            SMTFormula(expression="x > 0", kind=FormulaKind.ASSERTION),
            SMTFormula(expression="x < 10", kind=FormulaKind.ASSERTION),
        ]
        result = solver.check(formulas)
        # Simple solver can't handle complex formulas
        assert result.result in (SMTResult.SAT, SMTResult.UNKNOWN)


# ===========================================================================
# Z3Solver Tests
# ===========================================================================


@pytest.mark.skipif(not is_z3_available(), reason="Z3 not available")
class TestZ3SolverBasic:
    """Basic tests for Z3Solver (when available)."""

    def test_can_create(self):
        """Should be able to create Z3Solver."""
        solver = Z3Solver()
        assert solver is not None

    def test_sat_formula(self):
        """Should return SAT for satisfiable formulas."""
        solver = Z3Solver()
        formulas = [SMTFormula(expression="true", kind=FormulaKind.ASSERTION)]
        result = solver.check(formulas)
        assert result.is_sat

    def test_unsat_formula(self):
        """Should return UNSAT for unsatisfiable formulas."""
        solver = Z3Solver()
        formulas = [SMTFormula(expression="false", kind=FormulaKind.ASSERTION)]
        result = solver.check(formulas)
        assert result.is_unsat


@pytest.mark.skipif(not is_z3_available(), reason="Z3 not available")
class TestZ3TimeoutHandling:
    """Tests for Z3 timeout handling."""

    def test_short_timeout_solver(self):
        """Z3Solver should accept timeout parameter."""
        solver = Z3Solver(timeout_ms=100)
        assert solver is not None

    def test_normal_check_does_not_timeout(self):
        """Simple checks should not timeout."""
        solver = Z3Solver(timeout_ms=5000)
        formulas = [SMTFormula(expression="true", kind=FormulaKind.ASSERTION)]
        result = solver.check(formulas)
        assert not result.is_timeout
        assert not result.is_error

    def test_timeout_has_error_message(self):
        """Timeout result should include error message."""
        # Create a result as if it were a timeout
        result = SMTCheckResult(result=SMTResult.TIMEOUT, error="Solver timeout: timeout")
        assert result.is_timeout
        assert "timeout" in result.error.lower()


# ===========================================================================
# IncrementalSMTSolver Tests
# ===========================================================================


class TestIncrementalSMTSolver:
    """Tests for IncrementalSMTSolver."""

    def test_without_z3(self):
        """Should work without Z3 (fallback to SimpleSMTSolver)."""
        solver = IncrementalSMTSolver(use_z3=False)
        assert not solver.using_z3

    @pytest.mark.skipif(not is_z3_available(), reason="Z3 not available")
    def test_with_z3_when_available(self):
        """Should use Z3 when available and requested."""
        solver = IncrementalSMTSolver(use_z3=True)
        assert solver.using_z3

    def test_push_pop_stack(self):
        """Push/pop should maintain stack correctly."""
        solver = IncrementalSMTSolver(use_z3=False)

        assert solver.depth() == 1
        solver.push()
        assert solver.depth() == 2
        solver.push()
        assert solver.depth() == 3
        solver.pop()
        assert solver.depth() == 2
        solver.pop()
        assert solver.depth() == 1

    def test_reset_clears_stack(self):
        """Reset should return to initial state."""
        solver = IncrementalSMTSolver(use_z3=False)

        solver.push()
        solver.push()
        solver.add_formula(SMTFormula(expression="x > 0", kind=FormulaKind.ASSERTION))

        solver.reset()

        assert solver.depth() == 1

    def test_check_constraint(self):
        """Should check SemanticConstraint directly."""
        solver = IncrementalSMTSolver(use_z3=False)

        constraint = SemanticConstraint(
            formulas=frozenset([SMTFormula(expression="x > 0", kind=FormulaKind.ASSERTION)]),
        )

        result = solver.check_constraint(constraint)
        assert result is not None
        # SimpleSMTSolver returns UNKNOWN for complex formulas
        assert result.result in (SMTResult.SAT, SMTResult.UNKNOWN)


# ===========================================================================
# Graceful Degradation Tests
# ===========================================================================


class TestGracefulDegradation:
    """Tests for graceful degradation when Z3 unavailable."""

    def test_create_solver_without_z3(self):
        """create_smt_solver should work without Z3."""
        solver = create_smt_solver(use_z3=False)
        assert not solver.using_z3

    def test_fallback_solver_handles_all_results(self):
        """Fallback solver should never raise exceptions."""
        solver = create_smt_solver(use_z3=False)

        # Empty formulas
        result1 = solver.check()
        assert result1 is not None

        # Simple formulas
        solver.add_formula(SMTFormula(expression="x > 0", kind=FormulaKind.ASSERTION))
        result2 = solver.check()
        assert result2 is not None

        # After reset
        solver.reset()
        result3 = solver.check()
        assert result3 is not None


# ===========================================================================
# Error Recovery Tests
# ===========================================================================


class TestSMTErrorRecovery:
    """Tests for error recovery in SMT solvers."""

    def test_reset_after_formulas(self):
        """Reset should clear all added formulas."""
        solver = create_smt_solver(use_z3=False)

        solver.add_formula(SMTFormula(expression="false", kind=FormulaKind.ASSERTION))
        result_before = solver.check()
        assert result_before.is_unsat

        solver.reset()
        result_after = solver.check()
        assert result_after.is_sat  # Empty is SAT

    def test_checkpoint_via_push_pop(self):
        """Push/pop should act as checkpoint/restore."""
        solver = create_smt_solver(use_z3=False)

        # Add base formula
        solver.add_formula(SMTFormula(expression="true", kind=FormulaKind.ASSERTION))

        # Checkpoint
        solver.push()

        # Add contradicting formula
        solver.add_formula(SMTFormula(expression="false", kind=FormulaKind.ASSERTION))
        result_with_contradiction = solver.check()
        assert result_with_contradiction.is_unsat

        # Restore checkpoint
        solver.pop()

        # Should be SAT again (without the contradiction)
        result_after_restore = solver.check()
        # Note: Simple solver may still return SAT or UNKNOWN
        assert not result_after_restore.is_unsat


# ===========================================================================
# Factory Function Tests
# ===========================================================================


class TestCreateSMTSolver:
    """Tests for create_smt_solver factory."""

    def test_creates_incremental_solver(self):
        """Should create IncrementalSMTSolver."""
        solver = create_smt_solver()
        assert isinstance(solver, IncrementalSMTSolver)

    def test_respects_use_z3_false(self):
        """Should not use Z3 when use_z3=False."""
        solver = create_smt_solver(use_z3=False)
        assert not solver.using_z3

    @pytest.mark.skipif(not is_z3_available(), reason="Z3 not available")
    def test_uses_z3_when_available(self):
        """Should use Z3 when available and requested."""
        solver = create_smt_solver(use_z3=True)
        assert solver.using_z3

    def test_timeout_parameter(self):
        """Should accept timeout parameter."""
        solver = create_smt_solver(timeout_ms=100)
        assert solver is not None


# ===========================================================================
# Timeout Triggering Formulas Tests
# ===========================================================================


class TestTimeoutTriggeringFormulas:
    """Tests for create_timeout_triggering_formulas helper."""

    def test_creates_formulas(self):
        """Should create list of formulas."""
        formulas = create_timeout_triggering_formulas(count=10)
        assert isinstance(formulas, list)
        assert all(isinstance(f, SMTFormula) for f in formulas)

    def test_default_count(self):
        """Default count should create many formulas."""
        formulas = create_timeout_triggering_formulas()
        assert len(formulas) > 50

    def test_custom_count(self):
        """Custom count should affect formula count."""
        formulas_small = create_timeout_triggering_formulas(count=5)
        formulas_large = create_timeout_triggering_formulas(count=50)
        assert len(formulas_large) > len(formulas_small)


# ===========================================================================
# SemanticConstraint Integration Tests
# ===========================================================================


class TestSemanticConstraintWithSolver:
    """Tests for SemanticConstraint with SMT solver."""

    def test_top_constraint_is_sat(self):
        """TOP constraint should always be SAT."""
        solver = create_smt_solver(use_z3=False)
        result = solver.check_constraint(SEMANTIC_TOP)
        # TOP has no formulas, should be SAT
        assert result.is_sat

    def test_constraint_with_formulas(self):
        """Constraint with formulas should be checkable."""
        solver = create_smt_solver(use_z3=False)

        constraint = SemanticConstraint(
            formulas=frozenset([
                SMTFormula(expression="x > 0", kind=FormulaKind.ASSERTION),
                SMTFormula(expression="x < 10", kind=FormulaKind.ASSERTION),
            ]),
        )

        result = solver.check_constraint(constraint)
        assert result is not None
        # Result depends on solver capability


# ===========================================================================
# SMTResult Enum Tests
# ===========================================================================


class TestSMTResultEnum:
    """Tests for SMTResult enum values."""

    def test_all_values_exist(self):
        """All expected result values should exist."""
        assert SMTResult.SAT is not None
        assert SMTResult.UNSAT is not None
        assert SMTResult.UNKNOWN is not None
        assert SMTResult.TIMEOUT is not None
        assert SMTResult.ERROR is not None

    def test_values_are_distinct(self):
        """All result values should be distinct."""
        values = [
            SMTResult.SAT,
            SMTResult.UNSAT,
            SMTResult.UNKNOWN,
            SMTResult.TIMEOUT,
            SMTResult.ERROR,
        ]
        assert len(set(values)) == 5
