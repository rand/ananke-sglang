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
"""Semantic domain for SMT-based constraint tracking.

The SemanticDomain tracks:
- Assertions extracted from generated code
- Pre/post conditions from contracts
- Invariants that must be maintained

It uses SMT solving (via Z3 when available) to check
satisfiability of accumulated constraints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

import torch

try:
    from ...core.constraint import Satisfiability
    from ...core.domain import ConstraintDomain, GenerationContext
    from .constraint import (
        SEMANTIC_TOP,
        SEMANTIC_BOTTOM,
        SemanticConstraint,
        SMTFormula,
        FormulaKind,
    )
    from .smt import (
        IncrementalSMTSolver,
        SMTResult,
        SMTCheckResult,
        create_smt_solver,
        is_z3_available,
    )
    from .extractors import (
        CompositeExtractor,
        PythonAssertExtractor,
        extract_formulas,
    )
except ImportError:
    from core.constraint import Satisfiability
    from core.domain import ConstraintDomain, GenerationContext
    from domains.semantics.constraint import (
        SEMANTIC_TOP,
        SEMANTIC_BOTTOM,
        SemanticConstraint,
        SMTFormula,
        FormulaKind,
    )
    from domains.semantics.smt import (
        IncrementalSMTSolver,
        SMTResult,
        SMTCheckResult,
        create_smt_solver,
        is_z3_available,
    )
    from domains.semantics.extractors import (
        CompositeExtractor,
        PythonAssertExtractor,
        extract_formulas,
    )


@dataclass
class SemanticDomainCheckpoint:
    """Checkpoint for SemanticDomain state.

    Attributes:
        formulas: Set of accumulated formulas
        token_buffer: Current token buffer
        solver_depth: Solver stack depth
    """

    formulas: List[SMTFormula]
    token_buffer: str
    solver_depth: int


class SemanticDomain(ConstraintDomain[SemanticConstraint]):
    """Semantic domain for SMT-based constraint tracking.

    The semantic domain:
    1. Extracts semantic constraints from generated code
    2. Accumulates assertions, contracts, and invariants
    3. Uses SMT solving to check satisfiability
    4. Provides token masks based on semantic validity

    Example:
        >>> domain = SemanticDomain(language="python")
        >>> constraint = domain.create_constraint(
        ...     assertions=["x > 0", "x < 10"],
        ... )
        >>> # As code is generated, constraints are validated
    """

    def __init__(self, language: str = "python", use_z3: bool = True):
        """Initialize the semantic domain.

        Args:
            language: Programming language (affects extraction)
            use_z3: Whether to use Z3 for SMT solving
        """
        self._language = language
        self._use_z3 = use_z3 and is_z3_available()
        self._solver = create_smt_solver(use_z3=self._use_z3)
        self._extractor = CompositeExtractor()
        self._accumulated_formulas: List[SMTFormula] = []
        self._token_buffer = ""

    @property
    def name(self) -> str:
        """Return the domain name."""
        return "semantics"

    @property
    def top(self) -> SemanticConstraint:
        """Return the TOP constraint (no restrictions)."""
        return SEMANTIC_TOP

    @property
    def bottom(self) -> SemanticConstraint:
        """Return the BOTTOM constraint (unsatisfiable)."""
        return SEMANTIC_BOTTOM

    @property
    def language(self) -> str:
        """Return the target language."""
        return self._language

    @property
    def using_z3(self) -> bool:
        """Check if using Z3."""
        return self._use_z3

    @property
    def formula_count(self) -> int:
        """Get the number of accumulated formulas."""
        return len(self._accumulated_formulas)

    def token_mask(
        self,
        constraint: SemanticConstraint,
        context: GenerationContext,
    ) -> torch.Tensor:
        """Compute a token mask based on semantic constraints.

        Current implementation is conservative (allows all tokens).
        A full implementation would:
        - Predict constraint implications of each token
        - Block tokens that would make constraints unsatisfiable

        Args:
            constraint: Current semantic constraint
            context: Generation context

        Returns:
            Boolean tensor of valid tokens
        """
        # Handle TOP/BOTTOM
        if constraint.is_top():
            return torch.ones(context.vocab_size, dtype=torch.bool, device=context.device)
        if constraint.is_bottom():
            return torch.zeros(context.vocab_size, dtype=torch.bool, device=context.device)

        # Conservative: allow all tokens
        # Full implementation would use SMT solver to check implications
        return torch.ones(context.vocab_size, dtype=torch.bool, device=context.device)

    def observe_token(
        self,
        constraint: SemanticConstraint,
        token_id: int,
        context: GenerationContext,
    ) -> SemanticConstraint:
        """Update the semantic constraint after observing a token.

        Extracts semantic formulas from newly generated code
        and validates against the current constraint.

        Args:
            constraint: Current semantic constraint
            token_id: The generated token
            context: Generation context

        Returns:
            Updated semantic constraint
        """
        if constraint.is_top() or constraint.is_bottom():
            return constraint

        # Get token text
        token_text = ""
        if context.tokenizer is not None:
            try:
                token_text = context.tokenizer.decode([token_id])
            except Exception:
                pass

        # Accumulate token text
        self._token_buffer += token_text

        # Check for complete statements and extract formulas
        new_formulas = self._extract_new_formulas()

        # Add new formulas to constraint
        for formula in new_formulas:
            self._accumulated_formulas.append(formula)
            constraint = self._add_formula_to_constraint(constraint, formula)

        # Check satisfiability
        if constraint.formula_count() > 0:
            result = self._solver.check_constraint(constraint)
            if result.is_unsat:
                return SEMANTIC_BOTTOM

        return constraint

    def _extract_new_formulas(self) -> List[SMTFormula]:
        """Extract new formulas from the token buffer."""
        new_formulas: List[SMTFormula] = []

        # Check for complete statements (newline terminated)
        if "\n" not in self._token_buffer:
            return new_formulas

        lines = self._token_buffer.split("\n")
        for line in lines[:-1]:  # Process complete lines
            formulas = extract_formulas(line)
            new_formulas.extend(formulas)

        # Keep last incomplete line in buffer
        self._token_buffer = lines[-1]

        return new_formulas

    def _add_formula_to_constraint(
        self,
        constraint: SemanticConstraint,
        formula: SMTFormula,
    ) -> SemanticConstraint:
        """Add a formula to a constraint."""
        if formula.kind == FormulaKind.ASSERTION:
            return constraint.add_assertion(formula.expression, formula.name)
        elif formula.kind == FormulaKind.PRECONDITION:
            return constraint.add_precondition(formula.expression, formula.name)
        elif formula.kind == FormulaKind.POSTCONDITION:
            return constraint.add_postcondition(formula.expression, formula.name)
        elif formula.kind == FormulaKind.INVARIANT:
            return constraint.add_invariant(formula.expression, formula.name)
        elif formula.kind == FormulaKind.ASSUMPTION:
            return constraint.add_assumption(formula.expression, formula.name)
        else:
            return constraint.add_assertion(formula.expression, formula.name)

    def check_satisfiability(self, constraint: SemanticConstraint) -> SMTCheckResult:
        """Check satisfiability of a constraint.

        Args:
            constraint: The constraint to check

        Returns:
            SMTCheckResult with detailed result
        """
        return self._solver.check_constraint(constraint)

    def checkpoint(self) -> SemanticDomainCheckpoint:
        """Create a checkpoint of the current state.

        Returns:
            Checkpoint for restoration
        """
        return SemanticDomainCheckpoint(
            formulas=self._accumulated_formulas.copy(),
            token_buffer=self._token_buffer,
            solver_depth=self._solver.depth(),
        )

    def restore(self, checkpoint: Any) -> None:
        """Restore state from a checkpoint.

        Args:
            checkpoint: Previously created checkpoint
        """
        if not isinstance(checkpoint, SemanticDomainCheckpoint):
            raise TypeError(
                f"Expected SemanticDomainCheckpoint, got {type(checkpoint).__name__}"
            )

        self._accumulated_formulas = checkpoint.formulas.copy()
        self._token_buffer = checkpoint.token_buffer

        # Restore solver state
        while self._solver.depth() > checkpoint.solver_depth:
            self._solver.pop()
        while self._solver.depth() < checkpoint.solver_depth:
            self._solver.push()

    def satisfiability(self, constraint: SemanticConstraint) -> Satisfiability:
        """Check satisfiability of a semantic constraint.

        Args:
            constraint: The constraint to check

        Returns:
            Satisfiability status
        """
        if constraint.is_top():
            return Satisfiability.SAT
        if constraint.is_bottom():
            return Satisfiability.UNSAT

        result = self._solver.check_constraint(constraint)
        if result.is_sat:
            return Satisfiability.SAT
        elif result.is_unsat:
            return Satisfiability.UNSAT
        else:
            return Satisfiability.UNKNOWN

    def create_constraint(
        self,
        assertions: Optional[List[str]] = None,
        preconditions: Optional[List[str]] = None,
        postconditions: Optional[List[str]] = None,
        assumptions: Optional[List[str]] = None,
    ) -> SemanticConstraint:
        """Create a semantic constraint.

        Args:
            assertions: List of assertion expressions
            preconditions: List of precondition expressions
            postconditions: List of postcondition expressions
            assumptions: List of assumption expressions

        Returns:
            New SemanticConstraint
        """
        if (assertions is None and preconditions is None and
            postconditions is None and assumptions is None):
            return SEMANTIC_TOP

        constraint = SemanticConstraint()

        for expr in (assertions or []):
            constraint = constraint.add_assertion(expr)

        for expr in (preconditions or []):
            constraint = constraint.add_precondition(expr)

        for expr in (postconditions or []):
            constraint = constraint.add_postcondition(expr)

        for expr in (assumptions or []):
            constraint = constraint.add_assumption(expr)

        return constraint

    def push(self) -> None:
        """Push solver state for incremental solving."""
        self._solver.push()

    def pop(self) -> None:
        """Pop solver state."""
        self._solver.pop()

    def reset(self) -> None:
        """Reset domain state."""
        self._accumulated_formulas.clear()
        self._token_buffer = ""
        self._solver.reset()
