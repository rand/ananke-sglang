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
"""SMT solver integration for semantic constraints.

Provides integration with Z3 (when available) for:
- Satisfiability checking of formula sets
- Incremental solving with push/pop
- Model extraction for debugging
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple

from .constraint import SMTFormula, FormulaKind, SemanticConstraint

# Try to import Z3
_z3_available = False
try:
    import z3
    _z3_available = True
except ImportError:
    z3 = None  # type: ignore


class SMTResult(Enum):
    """Result of SMT satisfiability check."""

    SAT = auto()  # Satisfiable
    UNSAT = auto()  # Unsatisfiable
    UNKNOWN = auto()  # Could not determine
    TIMEOUT = auto()  # Solver timed out
    ERROR = auto()  # Error during solving


@dataclass
class SMTModel:
    """A satisfying model for SMT formulas.

    Attributes:
        variables: Map from variable name to value
        raw_model: The raw Z3 model (if using Z3)
    """

    variables: Dict[str, Any] = field(default_factory=dict)
    raw_model: Optional[Any] = None

    def get(self, name: str) -> Optional[Any]:
        """Get the value of a variable."""
        return self.variables.get(name)

    def __repr__(self) -> str:
        return f"SMTModel({self.variables})"


@dataclass
class SMTCheckResult:
    """Result of SMT satisfiability check.

    Attributes:
        result: The satisfiability result
        model: The model if SAT (optional)
        unsat_core: The unsatisfiable core if UNSAT (optional)
        error: Error message if ERROR
    """

    result: SMTResult
    model: Optional[SMTModel] = None
    unsat_core: Optional[List[str]] = None
    error: Optional[str] = None

    @property
    def is_sat(self) -> bool:
        """Check if result is SAT."""
        return self.result == SMTResult.SAT

    @property
    def is_unsat(self) -> bool:
        """Check if result is UNSAT."""
        return self.result == SMTResult.UNSAT

    @property
    def is_unknown(self) -> bool:
        """Check if result is UNKNOWN."""
        return self.result == SMTResult.UNKNOWN


def is_z3_available() -> bool:
    """Check if Z3 is available."""
    return _z3_available


class SMTSolver:
    """Abstract SMT solver interface."""

    def check(self, formulas: List[SMTFormula], assumptions: Optional[List[SMTFormula]] = None) -> SMTCheckResult:
        """Check satisfiability of formulas.

        Args:
            formulas: The formulas to check
            assumptions: Optional assumptions (context)

        Returns:
            SMTCheckResult with result and optional model/core
        """
        raise NotImplementedError

    def push(self) -> None:
        """Push solver state for incremental solving."""
        raise NotImplementedError

    def pop(self) -> None:
        """Pop solver state."""
        raise NotImplementedError

    def reset(self) -> None:
        """Reset solver state."""
        raise NotImplementedError


class SimpleSMTSolver(SMTSolver):
    """Simple SMT solver using pattern matching.

    This is a fallback solver when Z3 is not available.
    It handles simple cases like trivially true/false formulas.
    """

    def check(self, formulas: List[SMTFormula], assumptions: Optional[List[SMTFormula]] = None) -> SMTCheckResult:
        """Check satisfiability using simple pattern matching."""
        if not formulas:
            return SMTCheckResult(result=SMTResult.SAT)

        # Check for trivial cases
        for formula in formulas:
            expr = formula.expression.strip().lower()

            # Trivially false
            if expr in ("false", "0", ""):
                return SMTCheckResult(result=SMTResult.UNSAT)

            # Look for simple contradictions
            if self._is_contradiction(expr, formulas):
                return SMTCheckResult(result=SMTResult.UNSAT)

        # For anything non-trivial, return UNKNOWN
        return SMTCheckResult(result=SMTResult.UNKNOWN)

    def _is_contradiction(self, expr: str, formulas: List[SMTFormula]) -> bool:
        """Check for simple contradictions."""
        # Check for x and not x patterns
        for other in formulas:
            other_expr = other.expression.strip().lower()
            if f"not ({expr})" == other_expr or f"not({expr})" == other_expr:
                return True
            if expr == f"not ({other_expr})" or expr == f"not({other_expr})":
                return True
        return False

    def push(self) -> None:
        """Push state (no-op for simple solver)."""
        pass

    def pop(self) -> None:
        """Pop state (no-op for simple solver)."""
        pass

    def reset(self) -> None:
        """Reset state (no-op for simple solver)."""
        pass


class Z3Solver(SMTSolver):
    """Z3-based SMT solver.

    Provides full SMT solving capabilities using Z3.
    """

    def __init__(self, timeout_ms: int = 5000):
        """Initialize the Z3 solver.

        Args:
            timeout_ms: Timeout in milliseconds
        """
        if not _z3_available:
            raise RuntimeError("Z3 is not available. Install z3-solver package.")

        self._solver = z3.Solver()
        self._solver.set("timeout", timeout_ms)
        self._variables: Dict[str, Any] = {}

    def check(self, formulas: List[SMTFormula], assumptions: Optional[List[SMTFormula]] = None) -> SMTCheckResult:
        """Check satisfiability using Z3."""
        try:
            # Parse and add formulas
            z3_formulas = []
            for formula in formulas:
                z3_formula = self._parse_formula(formula.expression)
                if z3_formula is not None:
                    z3_formulas.append(z3_formula)
                    self._solver.add(z3_formula)

            # Parse and add assumptions
            z3_assumptions = []
            if assumptions:
                for assumption in assumptions:
                    z3_assumption = self._parse_formula(assumption.expression)
                    if z3_assumption is not None:
                        z3_assumptions.append(z3_assumption)

            # Check satisfiability
            if z3_assumptions:
                result = self._solver.check(*z3_assumptions)
            else:
                result = self._solver.check()

            # Process result
            if result == z3.sat:
                model = self._extract_model()
                return SMTCheckResult(result=SMTResult.SAT, model=model)
            elif result == z3.unsat:
                core = self._extract_unsat_core()
                return SMTCheckResult(result=SMTResult.UNSAT, unsat_core=core)
            else:
                return SMTCheckResult(result=SMTResult.UNKNOWN)

        except Exception as e:
            return SMTCheckResult(result=SMTResult.ERROR, error=str(e))

    def _parse_formula(self, expression: str) -> Optional[Any]:
        """Parse a formula expression into Z3 format.

        This is a simplified parser - full implementation would
        support more operators and types.
        """
        expr = expression.strip()

        # Handle simple cases
        if expr.lower() == "true":
            return z3.BoolVal(True)
        if expr.lower() == "false":
            return z3.BoolVal(False)

        # For now, create a boolean variable for unknown expressions
        # Full implementation would parse arithmetic, comparisons, etc.
        if expr not in self._variables:
            self._variables[expr] = z3.Bool(expr)
        return self._variables[expr]

    def _extract_model(self) -> SMTModel:
        """Extract model from SAT result."""
        z3_model = self._solver.model()
        variables = {}

        for decl in z3_model.decls():
            name = str(decl.name())
            value = z3_model[decl]
            # Convert Z3 value to Python
            if z3.is_bool(value):
                variables[name] = z3.is_true(value)
            elif z3.is_int(value):
                variables[name] = value.as_long()
            elif z3.is_real(value):
                variables[name] = float(value.as_decimal(6))
            else:
                variables[name] = str(value)

        return SMTModel(variables=variables, raw_model=z3_model)

    def _extract_unsat_core(self) -> Optional[List[str]]:
        """Extract unsatisfiable core."""
        try:
            core = self._solver.unsat_core()
            return [str(c) for c in core]
        except Exception:
            return None

    def push(self) -> None:
        """Push solver state."""
        self._solver.push()

    def pop(self) -> None:
        """Pop solver state."""
        self._solver.pop()

    def reset(self) -> None:
        """Reset solver state."""
        self._solver.reset()
        self._variables.clear()


class IncrementalSMTSolver:
    """Incremental SMT solver with push/pop stack.

    Maintains a stack of solver states for efficient
    constraint checking during generation.
    """

    def __init__(self, use_z3: bool = True, timeout_ms: int = 5000):
        """Initialize the incremental solver.

        Args:
            use_z3: Whether to use Z3 (if available)
            timeout_ms: Timeout for Z3 solver
        """
        if use_z3 and _z3_available:
            self._solver: SMTSolver = Z3Solver(timeout_ms)
            self._using_z3 = True
        else:
            self._solver = SimpleSMTSolver()
            self._using_z3 = False

        self._formula_stack: List[List[SMTFormula]] = [[]]
        self._assumption_stack: List[List[SMTFormula]] = [[]]

    @property
    def using_z3(self) -> bool:
        """Check if using Z3."""
        return self._using_z3

    def add_formula(self, formula: SMTFormula) -> None:
        """Add a formula at the current level."""
        self._formula_stack[-1].append(formula)

    def add_assumption(self, assumption: SMTFormula) -> None:
        """Add an assumption at the current level."""
        self._assumption_stack[-1].append(assumption)

    def check(self) -> SMTCheckResult:
        """Check satisfiability of all formulas."""
        all_formulas = [f for level in self._formula_stack for f in level]
        all_assumptions = [a for level in self._assumption_stack for a in level]
        return self._solver.check(all_formulas, all_assumptions)

    def check_constraint(self, constraint: SemanticConstraint) -> SMTCheckResult:
        """Check satisfiability of a SemanticConstraint."""
        formulas = list(constraint.formulas)
        assumptions = list(constraint.assumptions)
        return self._solver.check(formulas, assumptions)

    def push(self) -> None:
        """Push a new level onto the stack."""
        self._formula_stack.append([])
        self._assumption_stack.append([])
        self._solver.push()

    def pop(self) -> None:
        """Pop the top level from the stack."""
        if len(self._formula_stack) > 1:
            self._formula_stack.pop()
            self._assumption_stack.pop()
            self._solver.pop()

    def reset(self) -> None:
        """Reset to initial state."""
        self._formula_stack = [[]]
        self._assumption_stack = [[]]
        self._solver.reset()

    def depth(self) -> int:
        """Get the current stack depth."""
        return len(self._formula_stack)


def create_smt_solver(use_z3: bool = True, timeout_ms: int = 5000) -> IncrementalSMTSolver:
    """Factory function to create an SMT solver.

    Args:
        use_z3: Whether to use Z3 (if available)
        timeout_ms: Timeout for Z3 solver

    Returns:
        Configured IncrementalSMTSolver
    """
    return IncrementalSMTSolver(use_z3=use_z3, timeout_ms=timeout_ms)
