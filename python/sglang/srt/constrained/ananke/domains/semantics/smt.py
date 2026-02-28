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

    @property
    def is_timeout(self) -> bool:
        """Check if result is TIMEOUT."""
        return self.result == SMTResult.TIMEOUT

    @property
    def is_error(self) -> bool:
        """Check if result is ERROR."""
        return self.result == SMTResult.ERROR


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

    Supports formula parsing for:
    - Arithmetic: +, -, *, /, //, %, **
    - Comparison: <, <=, >, >=, ==, !=
    - Logical: and, or, not, implies (->)
    - Variables with type inference (Int, Real, Bool)
    - Parentheses for grouping
    """

    # Operator precedence (higher = binds tighter)
    _PRECEDENCE = {
        'or': 1,
        'and': 2,
        'not': 3,
        '->': 4,  # implies
        '==': 5, '!=': 5, '<': 5, '<=': 5, '>': 5, '>=': 5,
        '+': 6, '-': 6,
        '*': 7, '/': 7, '//': 7, '%': 7,
        '**': 8,
        'unary-': 9,  # unary minus
    }

    # Tokens that are operators
    _BINARY_OPS = {'or', 'and', '->', '==', '!=', '<', '<=', '>', '>=',
                   '+', '-', '*', '/', '//', '%', '**'}
    _UNARY_OPS = {'not', '-'}

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
        self._formula_cache: Dict[str, Any] = {}  # Cache for parsed formulas

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
                # Z3 returned unknown - check if it was a timeout
                reason = self._solver.reason_unknown()
                if reason in ("timeout", "canceled"):
                    return SMTCheckResult(
                        result=SMTResult.TIMEOUT,
                        error=f"Solver timeout: {reason}",
                    )
                return SMTCheckResult(result=SMTResult.UNKNOWN)

        except Exception as e:
            # Check for timeout-related exceptions
            error_msg = str(e).lower()
            if "timeout" in error_msg or "canceled" in error_msg:
                return SMTCheckResult(
                    result=SMTResult.TIMEOUT,
                    error=str(e),
                )
            return SMTCheckResult(result=SMTResult.ERROR, error=str(e))

    def _parse_formula(self, expression: str) -> Optional[Any]:
        """Parse a formula expression into Z3 format.

        Supports:
        - Boolean literals: true, false
        - Numeric literals: integers and floats
        - Variables: auto-typed based on context
        - Arithmetic: +, -, *, /, //, %, **
        - Comparison: <, <=, >, >=, ==, !=
        - Logical: and, or, not, implies (->)
        - Parentheses for grouping

        Args:
            expression: The expression string to parse

        Returns:
            Z3 expression or None if parsing fails
        """
        expr = expression.strip()

        # Check cache first
        if expr in self._formula_cache:
            return self._formula_cache[expr]

        try:
            result = self._parse_expr(expr)
            self._formula_cache[expr] = result
            return result
        except Exception:
            # On parse failure, fall back to boolean variable
            # This preserves soundness - we don't block based on unparseable formulas
            if expr not in self._variables:
                self._variables[expr] = z3.Bool(expr)
            return self._variables[expr]

    def _tokenize(self, expr: str) -> List[str]:
        """Tokenize an expression string.

        Args:
            expr: Expression string

        Returns:
            List of tokens
        """
        import re
        # Pattern matches: multi-char ops, numbers, identifiers, single chars
        pattern = r'(->|//|\*\*|<=|>=|==|!=|and|or|not|\d+\.?\d*|[a-zA-Z_]\w*|[+\-*/%<>()!])'
        tokens = re.findall(pattern, expr, re.IGNORECASE)
        return tokens

    def _parse_expr(self, expr: str) -> Any:
        """Parse an expression using precedence climbing.

        Args:
            expr: Expression string

        Returns:
            Z3 expression
        """
        tokens = self._tokenize(expr)
        pos = [0]  # Mutable position counter

        def peek() -> Optional[str]:
            if pos[0] < len(tokens):
                return tokens[pos[0]]
            return None

        def consume() -> str:
            tok = tokens[pos[0]]
            pos[0] += 1
            return tok

        def parse_atom() -> Any:
            """Parse an atomic expression (literal, variable, or parenthesized expr)."""
            tok = peek()
            if tok is None:
                raise ValueError("Unexpected end of expression")

            # Unary operators
            if tok.lower() == 'not':
                consume()
                operand = parse_atom()
                return z3.Not(operand)
            if tok == '-' and (pos[0] == 0 or tokens[pos[0]-1] in self._BINARY_OPS or tokens[pos[0]-1] == '('):
                consume()
                operand = parse_atom()
                return -operand

            # Parentheses
            if tok == '(':
                consume()
                inner = parse_binary(0)
                if peek() == ')':
                    consume()
                return inner

            consume()

            # Boolean literals
            if tok.lower() == 'true':
                return z3.BoolVal(True)
            if tok.lower() == 'false':
                return z3.BoolVal(False)

            # Numeric literals
            if tok.replace('.', '', 1).replace('-', '', 1).isdigit():
                if '.' in tok:
                    return z3.RealVal(tok)
                else:
                    return z3.IntVal(int(tok))

            # Variables - infer type from name or default to Int
            return self._get_or_create_variable(tok)

        def parse_binary(min_prec: int) -> Any:
            """Parse binary expressions with precedence climbing."""
            left = parse_atom()

            while True:
                op = peek()
                if op is None:
                    break

                # Normalize operator
                op_lower = op.lower()
                if op_lower not in self._PRECEDENCE:
                    break

                prec = self._PRECEDENCE[op_lower]
                if prec < min_prec:
                    break

                consume()
                # Right-associative for ** and ->, left-associative otherwise
                next_min_prec = prec + 1 if op_lower not in ('**', '->') else prec
                right = parse_binary(next_min_prec)

                left = self._apply_binary_op(op_lower, left, right)

            return left

        return parse_binary(0)

    def _get_or_create_variable(self, name: str) -> Any:
        """Get or create a Z3 variable.

        Infers type from variable name conventions:
        - Names starting with 'b' or 'is_' -> Bool
        - Names starting with 'f' or containing 'float' -> Real
        - Default -> Int

        Args:
            name: Variable name

        Returns:
            Z3 variable
        """
        if name in self._variables:
            return self._variables[name]

        name_lower = name.lower()

        # Type inference heuristics
        if name_lower.startswith('is_') or name_lower.startswith('has_'):
            var = z3.Bool(name)
        elif name_lower.startswith('b') and len(name) > 1 and not name[1].isalpha():
            var = z3.Bool(name)
        elif 'float' in name_lower or 'real' in name_lower:
            var = z3.Real(name)
        elif name_lower.startswith('f') and len(name) > 1 and not name[1].isalpha():
            var = z3.Real(name)
        else:
            # Default to Int for most numeric variables
            var = z3.Int(name)

        self._variables[name] = var
        return var

    def _apply_binary_op(self, op: str, left: Any, right: Any) -> Any:
        """Apply a binary operator to two operands.

        Handles type coercion for mixed Int/Real operations.

        Args:
            op: Operator string
            left: Left operand
            right: Right operand

        Returns:
            Z3 expression
        """
        # Coerce types if needed for arithmetic
        if op in ('+', '-', '*', '/', '//', '%', '**'):
            left, right = self._coerce_arithmetic(left, right)

        # Apply operator
        if op == 'and':
            return z3.And(left, right)
        elif op == 'or':
            return z3.Or(left, right)
        elif op == '->':
            return z3.Implies(left, right)
        elif op == '==':
            return left == right
        elif op == '!=':
            return left != right
        elif op == '<':
            return left < right
        elif op == '<=':
            return left <= right
        elif op == '>':
            return left > right
        elif op == '>=':
            return left >= right
        elif op == '+':
            return left + right
        elif op == '-':
            return left - right
        elif op == '*':
            return left * right
        elif op == '/':
            # Z3 uses / for real division
            return left / right
        elif op == '//':
            # Integer division - convert to Div for ints
            if z3.is_int(left) and z3.is_int(right):
                return left / right  # Z3 Int / Int is floor division
            else:
                return z3.ToInt(left / right)
        elif op == '%':
            return left % right
        elif op == '**':
            # Z3 doesn't have direct power, approximate with constraints
            # For now, return a fresh variable (conservative)
            return self._get_or_create_variable(f"pow_{id(left)}_{id(right)}")

        # Fallback - should not reach here
        raise ValueError(f"Unknown operator: {op}")

    def _coerce_arithmetic(self, left: Any, right: Any) -> Tuple[Any, Any]:
        """Coerce operands for arithmetic operations.

        If one is Real and one is Int, convert Int to Real.

        Args:
            left: Left operand
            right: Right operand

        Returns:
            Coerced (left, right) tuple
        """
        left_is_real = z3.is_real(left) and not z3.is_int(left)
        right_is_real = z3.is_real(right) and not z3.is_int(right)

        if left_is_real and z3.is_int(right):
            right = z3.ToReal(right)
        elif right_is_real and z3.is_int(left):
            left = z3.ToReal(left)

        return left, right

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
        self._formula_cache.clear()


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


def create_timeout_triggering_formulas(count: int = 100) -> List[SMTFormula]:
    """Create a set of formulas likely to trigger solver timeout.

    This is a helper for testing timeout handling. Creates a set of
    variables with constraints that require exponential search.

    Args:
        count: Number of boolean variables to create

    Returns:
        List of SMTFormulas that are hard to solve
    """
    formulas = []

    # Create many interrelated boolean variables
    # This creates a satisfiability problem that's hard for the solver
    for i in range(count):
        # Each variable depends on many others
        if i > 0:
            # XOR-like constraints are hard
            expr = f"(x{i} != x{i-1})"
            formulas.append(SMTFormula(expression=expr, kind=FormulaKind.ASSERTION))

        # Add arbitrary constraints
        if i > 5:
            expr = f"(x{i} or x{i-3} or x{i-5})"
            formulas.append(SMTFormula(expression=expr, kind=FormulaKind.ASSERTION))

    return formulas
