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

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import torch

try:
    from ...core.constraint import Satisfiability
    from ...core.domain import ConstraintDomain, GenerationContext
    from ...core.token_classifier import (
        TokenClassifier,
        TokenCategory,
        get_or_create_classifier,
    )
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
    from core.token_classifier import (
        TokenClassifier,
        TokenCategory,
        get_or_create_classifier,
    )
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


# Pattern for extracting bounds from simple comparison formulas
# Matches: var > num, var >= num, var < num, var <= num, var == num
_BOUND_PATTERN = re.compile(
    r"^\s*(\w+(?:\.\w+)*)\s*(>|>=|<|<=|==)\s*(-?\d+(?:\.\d+)?)\s*$"
)

# Reverse pattern: num < var, num <= var, etc.
_REVERSE_BOUND_PATTERN = re.compile(
    r"^\s*(-?\d+(?:\.\d+)?)\s*(>|>=|<|<=|==)\s*(\w+(?:\.\w+)*)\s*$"
)

# Compound pattern: var > num and var < num (extracts both bounds)
_COMPOUND_BOUND_PATTERN = re.compile(
    r"^\s*(\w+(?:\.\w+)*)\s*(>|>=)\s*(-?\d+(?:\.\d+)?)\s+and\s+\1\s*(<|<=)\s*(-?\d+(?:\.\d+)?)\s*$"
)

# Chained comparison: num <= var < num (Python-style)
_CHAINED_BOUND_PATTERN = re.compile(
    r"^\s*(-?\d+(?:\.\d+)?)\s*(<=?)\s*(\w+(?:\.\w+)*)\s*(<|<=)\s*(-?\d+(?:\.\d+)?)\s*$"
)


# =============================================================================
# Context-Aware Bounds Checking Types
# =============================================================================


class ExpressionContext(Enum):
    """Syntactic position within an expression.

    Tracks where we are in the current expression to determine
    whether bounds checking should be aggressive or permissive.
    """

    NONE = auto()                    # Not in a relevant context
    SIMPLE_ASSIGNMENT_RHS = auto()   # Direct literal position: x = <here>
    COMPOUND_EXPR = auto()           # Inside x + y, binary ops
    FUNCTION_CALL = auto()           # Inside function call parentheses
    SUBSCRIPT = auto()               # Inside [...]
    CONDITIONAL = auto()             # Inside ternary ... if ... else ...
    LIST_LITERAL = auto()            # Inside [1, 2, 3]
    DICT_LITERAL = auto()            # Inside {k: v}
    ATTRIBUTE_ACCESS = auto()        # After .


class ContextConfidence(Enum):
    """Confidence level in the syntactic context determination.

    Higher confidence means we have more certainty about what
    syntactic form we're generating.
    """

    HIGH = auto()    # Simple literal assignment: x = 5
    MEDIUM = auto()  # Conditional or structured: x = (5 if cond else 10)
    LOW = auto()     # Complex expression: x = y + z
    NONE = auto()    # Unknown context or function call: x = calculate()


class BoundsConfidence(Enum):
    """Confidence level in extracted bounds.

    Different sources of bounds have different reliability levels.
    """

    HIGH = auto()    # From explicit assert/require: assert x > 0
    MEDIUM = auto()  # From simple comparison in precondition
    LOW = auto()     # From transitive inference or complex expression
    UNKNOWN = auto() # SMT returned UNKNOWN, bounds are uncertain


class BlockingLevel(Enum):
    """How aggressively to block tokens based on bounds.

    Determined by combining context and bounds confidence.
    """

    AGGRESSIVE = auto()   # Block all out-of-bounds violations
    CONSERVATIVE = auto() # Block only obvious violations
    PERMISSIVE = auto()   # Allow all tokens (soundness preservation)


@dataclass
class ExpressionState:
    """Tracks the current state of expression parsing.

    Maintained token-by-token to understand syntactic position.
    This enables context-aware decisions about bounds checking.
    """

    context: ExpressionContext = ExpressionContext.NONE
    context_confidence: ContextConfidence = ContextConfidence.NONE
    target_variable: Optional[str] = None  # Variable being assigned
    paren_depth: int = 0                   # Nesting depth of ()
    bracket_depth: int = 0                 # Nesting depth of []
    brace_depth: int = 0                   # Nesting depth of {}
    tokens_since_assignment: int = 0       # Tokens since `=`

    def is_direct_literal_position(self) -> bool:
        """Check if we're at a position where a literal directly assigns.

        Returns True only when we're immediately after `=` with no
        intervening operators, parentheses, or other syntax.
        """
        return (
            self.context == ExpressionContext.SIMPLE_ASSIGNMENT_RHS
            and self.paren_depth == 0
            and self.bracket_depth == 0
            and self.brace_depth == 0
            and self.tokens_since_assignment <= 1  # Allow leading whitespace
        )

    def copy(self) -> "ExpressionState":
        """Create a copy of this state."""
        return ExpressionState(
            context=self.context,
            context_confidence=self.context_confidence,
            target_variable=self.target_variable,
            paren_depth=self.paren_depth,
            bracket_depth=self.bracket_depth,
            brace_depth=self.brace_depth,
            tokens_since_assignment=self.tokens_since_assignment,
        )


@dataclass
class VariableBounds:
    """Bounds for a variable extracted from constraints.

    Attributes:
        lower: Lower bound (None if unbounded)
        upper: Upper bound (None if unbounded)
        is_float: Whether the bound is a float
        confidence: Confidence level in these bounds
        source: Source of the bounds (for debugging)
        smt_uncertain: Whether SMT returned UNKNOWN for these bounds
    """

    lower: Optional[float] = None
    upper: Optional[float] = None
    is_float: bool = False
    confidence: BoundsConfidence = BoundsConfidence.MEDIUM
    source: Optional[str] = None
    smt_uncertain: bool = False

    def contains(self, value: float) -> bool:
        """Check if a value is within bounds.

        Args:
            value: The value to check

        Returns:
            True if value is within bounds
        """
        if self.lower is not None and value < self.lower:
            return False
        if self.upper is not None and value > self.upper:
            return False
        return True

    def effective_confidence(self) -> BoundsConfidence:
        """Get effective confidence accounting for SMT uncertainty.

        When SMT returns UNKNOWN, we downgrade confidence by one level
        to reflect the uncertainty in the bounds.

        Returns:
            Effective confidence level
        """
        if not self.smt_uncertain:
            return self.confidence

        # Downgrade confidence when SMT is uncertain
        if self.confidence == BoundsConfidence.HIGH:
            return BoundsConfidence.MEDIUM
        elif self.confidence == BoundsConfidence.MEDIUM:
            return BoundsConfidence.LOW
        else:
            return BoundsConfidence.UNKNOWN

    def is_clearly_violated(self, value: float) -> bool:
        """Check if a value clearly violates bounds (for CONSERVATIVE mode).

        More permissive than `contains()` to avoid false positives, but
        still catches values that are unambiguously out of range.

        Args:
            value: The value to check

        Returns:
            True if value clearly violates bounds
        """
        # Negative value with non-negative lower bound
        if self.lower is not None and self.lower >= 0 and value < 0:
            return True
        # Positive value with non-positive upper bound
        if self.upper is not None and self.upper <= 0 and value > 0:
            return True
        # Value below lower bound (with small margin for rounding)
        if self.lower is not None and value < self.lower:
            margin = max(1.0, abs(self.lower) * 0.1)
            if value < self.lower - margin:
                return True
        # Value above upper bound (with small margin for rounding)
        if self.upper is not None and value > self.upper:
            margin = max(1.0, abs(self.upper) * 0.1)
            if value > self.upper + margin:
                return True
        return False


@dataclass
class SemanticDomainCheckpoint:
    """Checkpoint for SemanticDomain state.

    Attributes:
        formulas: Set of accumulated formulas
        token_buffer: Current token buffer
        solver_depth: Solver stack depth
        variable_bounds: Extracted variable bounds
        expression_state: Current expression parsing state
        assignment_context: Current assignment target (deprecated, use expression_state)
    """

    formulas: List[SMTFormula]
    token_buffer: str
    solver_depth: int
    variable_bounds: Dict[str, VariableBounds] = field(default_factory=dict)
    expression_state: Optional[ExpressionState] = None
    assignment_context: Optional[str] = None  # Kept for backwards compatibility


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

    def __init__(
        self,
        language: str = "python",
        use_z3: bool = True,
        aggressive_mode: bool = True,
        tokenizer: Optional[Any] = None,
    ):
        """Initialize the semantic domain.

        Args:
            language: Programming language (affects extraction)
            use_z3: Whether to use Z3 for SMT solving
            aggressive_mode: Whether to aggressively block out-of-bounds literals
            tokenizer: Optional tokenizer for precise masking
        """
        self._language = language
        self._use_z3 = use_z3 and is_z3_available()
        self._solver = create_smt_solver(use_z3=self._use_z3)
        self._extractor = CompositeExtractor()
        self._accumulated_formulas: List[SMTFormula] = []
        self._token_buffer = ""

        # Aggressive mode state (legacy - kept for backwards compatibility)
        self._aggressive_mode = aggressive_mode
        self._variable_bounds: Dict[str, VariableBounds] = {}
        self._assignment_context: Optional[str] = None

        # Context-aware bounds checking state
        self._expression_state = ExpressionState()

        # Lazy-initialized classifier
        self._tokenizer = tokenizer
        self._classifier: Optional[TokenClassifier] = None

    def _ensure_classifier_initialized(self, context: GenerationContext) -> None:
        """Ensure classifier is initialized.

        Args:
            context: Generation context with tokenizer
        """
        tokenizer = context.tokenizer or self._tokenizer
        if tokenizer is None:
            return

        if self._classifier is None:
            self._classifier = get_or_create_classifier(tokenizer, self._language)

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

        Uses context-aware graduated blocking:
        - AGGRESSIVE: Block all out-of-bounds literals (high confidence)
        - CONSERVATIVE: Block only obvious violations (medium confidence)
        - PERMISSIVE: Allow all tokens (low confidence, soundness)

        Performance target: <50μs typical, <200μs worst case.

        Args:
            constraint: Current semantic constraint
            context: Generation context

        Returns:
            Boolean tensor of valid tokens
        """
        # Handle TOP/BOTTOM
        if constraint.is_top():
            return context.create_mask(fill_value=True)
        if constraint.is_bottom():
            return context.create_mask(fill_value=False)

        # If aggressive mode is off or no formulas, return all-True
        if not self._aggressive_mode or constraint.formula_count() == 0:
            return context.create_mask(fill_value=True)

        # Ensure classifier is initialized
        self._ensure_classifier_initialized(context)

        # Create base mask (all True)
        mask = context.create_mask(fill_value=True)

        # Extract bounds from constraint formulas
        self._update_variable_bounds(constraint)

        # Determine target variable from expression state (preferred) or legacy context
        target_var = self._expression_state.target_variable or self._assignment_context

        # If we're in an assignment context for a bounded variable, apply graduated blocking
        if target_var and target_var in self._variable_bounds:
            bounds = self._variable_bounds[target_var]

            # Compute blocking level from combined confidence
            blocking_level = self._get_current_blocking_level(target_var)

            # Apply graduated blocking
            mask = self._apply_graduated_blocking(mask, bounds, blocking_level, context)

        return mask

    def _apply_graduated_blocking(
        self,
        mask: torch.Tensor,
        bounds: VariableBounds,
        blocking_level: BlockingLevel,
        context: GenerationContext,
    ) -> torch.Tensor:
        """Apply graduated blocking based on confidence level.

        Args:
            mask: Current mask to modify
            bounds: Variable bounds
            blocking_level: How aggressively to block
            context: Generation context

        Returns:
            Modified mask
        """
        # PERMISSIVE: Allow everything (soundness preservation)
        if blocking_level == BlockingLevel.PERMISSIVE:
            return mask

        # AGGRESSIVE or CONSERVATIVE: Block based on level
        if self._classifier is None:
            return mask

        # Check integer literals
        for token_id in self._classifier.by_category(TokenCategory.INT_LITERAL):
            if token_id >= context.vocab_size:
                continue

            tc = self._classifier.get_classification(token_id)
            if tc.literal_value is not None:
                try:
                    value = float(tc.literal_value)
                    should_block = self._should_block_value(
                        value, bounds, blocking_level
                    )
                    if should_block:
                        mask[token_id] = False
                except (TypeError, ValueError):
                    pass

        # Check float literals if bounds allow floats
        if bounds.is_float:
            for token_id in self._classifier.by_category(TokenCategory.FLOAT_LITERAL):
                if token_id >= context.vocab_size:
                    continue

                tc = self._classifier.get_classification(token_id)
                if tc.literal_value is not None:
                    try:
                        value = float(tc.literal_value)
                        should_block = self._should_block_value(
                            value, bounds, blocking_level
                        )
                        if should_block:
                            mask[token_id] = False
                    except (TypeError, ValueError):
                        pass

        return mask

    def _should_block_value(
        self,
        value: float,
        bounds: VariableBounds,
        blocking_level: BlockingLevel,
    ) -> bool:
        """Determine if a value should be blocked based on blocking level.

        Args:
            value: The numeric value to check
            bounds: The variable bounds
            blocking_level: How aggressively to block

        Returns:
            True if the value should be blocked
        """
        if blocking_level == BlockingLevel.AGGRESSIVE:
            # Block all out-of-bounds values
            return not bounds.contains(value)
        elif blocking_level == BlockingLevel.CONSERVATIVE:
            # Block only obvious violations
            return bounds.is_clearly_violated(value)
        else:
            # PERMISSIVE: Never block
            return False

    def _update_variable_bounds(self, constraint: SemanticConstraint) -> None:
        """Update variable bounds from constraint formulas.

        Sets bounds confidence based on formula kind:
        - ASSERTION, PRECONDITION, INVARIANT -> HIGH confidence
        - POSTCONDITION, ASSUMPTION -> MEDIUM confidence
        - Other/transitive -> LOW confidence

        Args:
            constraint: The semantic constraint
        """
        for formula in constraint.formulas:
            # Determine confidence based on formula kind
            confidence = self._formula_kind_to_confidence(formula.kind)

            bounds = self._extract_bounds_from_formula(formula.expression, confidence)
            if bounds:
                var_name, bound_info = bounds
                if var_name not in self._variable_bounds:
                    self._variable_bounds[var_name] = VariableBounds(
                        confidence=bound_info.confidence,
                        source=formula.name,
                    )

                existing = self._variable_bounds[var_name]
                if bound_info.lower is not None:
                    if existing.lower is None or bound_info.lower > existing.lower:
                        existing.lower = bound_info.lower
                if bound_info.upper is not None:
                    if existing.upper is None or bound_info.upper < existing.upper:
                        existing.upper = bound_info.upper
                if bound_info.is_float:
                    existing.is_float = True
                # Use highest confidence from any source
                if bound_info.confidence.value < existing.confidence.value:
                    existing.confidence = bound_info.confidence
                    existing.source = formula.name

    def _formula_kind_to_confidence(self, kind: FormulaKind) -> BoundsConfidence:
        """Map formula kind to bounds confidence.

        Args:
            kind: The formula kind

        Returns:
            Appropriate bounds confidence level
        """
        # High confidence: explicit assertions and preconditions
        if kind in (FormulaKind.ASSERTION, FormulaKind.PRECONDITION, FormulaKind.INVARIANT):
            return BoundsConfidence.HIGH
        # Medium confidence: postconditions and assumptions
        elif kind in (FormulaKind.POSTCONDITION, FormulaKind.ASSUMPTION):
            return BoundsConfidence.MEDIUM
        # Default to medium for unknown kinds
        return BoundsConfidence.MEDIUM

    def _extract_bounds_from_formula(
        self,
        expression: str,
        confidence: BoundsConfidence = BoundsConfidence.MEDIUM,
    ) -> Optional[Tuple[str, VariableBounds]]:
        """Extract bounds from comparison formulas.

        Handles patterns like:
        - x > 5, x >= 10, y < 100 (simple)
        - 0 <= x < 10 (chained comparison)
        - x >= 0 and x < 100 (compound)
        - result >= 0 (dotted names like self.balance)

        Args:
            expression: The formula expression
            confidence: Confidence level for extracted bounds

        Returns:
            Tuple of (variable_name, bounds) or None if not a bound pattern
        """
        # Try compound pattern: var >= num and var < num
        match = _COMPOUND_BOUND_PATTERN.match(expression)
        if match:
            var, lower_op, lower_str, upper_op, upper_str = match.groups()
            lower_num = float(lower_str)
            upper_num = float(upper_str)
            is_float = '.' in lower_str or '.' in upper_str
            bounds = VariableBounds(is_float=is_float, confidence=confidence)
            # Lower bound
            if lower_op == '>':
                bounds.lower = lower_num + (0.0 if is_float else 1.0)
            else:  # >=
                bounds.lower = lower_num
            # Upper bound
            if upper_op == '<':
                bounds.upper = upper_num - (0.0 if is_float else 1.0)
            else:  # <=
                bounds.upper = upper_num
            return (var, bounds)

        # Try chained comparison: num <= var < num
        match = _CHAINED_BOUND_PATTERN.match(expression)
        if match:
            lower_str, lower_op, var, upper_op, upper_str = match.groups()
            lower_num = float(lower_str)
            upper_num = float(upper_str)
            is_float = '.' in lower_str or '.' in upper_str
            bounds = VariableBounds(is_float=is_float, confidence=confidence)
            if lower_op == '<':
                bounds.lower = lower_num + (0.0 if is_float else 1.0)
            else:  # <=
                bounds.lower = lower_num
            if upper_op == '<':
                bounds.upper = upper_num - (0.0 if is_float else 1.0)
            else:  # <=
                bounds.upper = upper_num
            return (var, bounds)

        # Try direct pattern: var op num
        match = _BOUND_PATTERN.match(expression)
        if match:
            var, op, num_str = match.groups()
            num = float(num_str)
            is_float = '.' in num_str
            return self._make_bounds(var, op, num, is_float, confidence)

        # Try reverse pattern: num op var
        match = _REVERSE_BOUND_PATTERN.match(expression)
        if match:
            num_str, op, var = match.groups()
            num = float(num_str)
            is_float = '.' in num_str
            # Reverse the operator
            reverse_ops = {'>': '<', '>=': '<=', '<': '>', '<=': '>=', '==': '=='}
            reversed_op = reverse_ops.get(op, op)
            return self._make_bounds(var, reversed_op, num, is_float, confidence)

        return None

    def _make_bounds(
        self,
        var: str,
        op: str,
        num: float,
        is_float: bool,
        confidence: BoundsConfidence = BoundsConfidence.MEDIUM,
    ) -> Tuple[str, VariableBounds]:
        """Create a VariableBounds from an operator and value.

        Args:
            var: Variable name
            op: Comparison operator
            num: Numeric value
            is_float: Whether the value is a float
            confidence: Confidence level for these bounds

        Returns:
            Tuple of (variable_name, bounds)
        """
        bounds = VariableBounds(is_float=is_float, confidence=confidence)

        if op == '>':
            # x > 5 means x >= 6 (for integers)
            bounds.lower = num + (0.0 if is_float else 1.0)
        elif op == '>=':
            bounds.lower = num
        elif op == '<':
            # x < 10 means x <= 9 (for integers)
            bounds.upper = num - (0.0 if is_float else 1.0)
        elif op == '<=':
            bounds.upper = num
        elif op == '==':
            bounds.lower = num
            bounds.upper = num

        return (var, bounds)

    def _block_out_of_bounds_literals(
        self,
        mask: torch.Tensor,
        bounds: VariableBounds,
        context: GenerationContext,
    ) -> torch.Tensor:
        """Block numeric literal tokens that are out of bounds.

        Args:
            mask: Current mask to modify
            bounds: The variable bounds
            context: Generation context

        Returns:
            Modified mask
        """
        if self._classifier is None:
            return mask

        # Check integer literals
        for token_id in self._classifier.by_category(TokenCategory.INT_LITERAL):
            if token_id >= context.vocab_size:
                continue

            tc = self._classifier.get_classification(token_id)
            if tc.literal_value is not None:
                try:
                    value = float(tc.literal_value)
                    if not bounds.contains(value):
                        mask[token_id] = False
                except (TypeError, ValueError):
                    pass

        # Check float literals if bounds allow floats
        if bounds.is_float:
            for token_id in self._classifier.by_category(TokenCategory.FLOAT_LITERAL):
                if token_id >= context.vocab_size:
                    continue

                tc = self._classifier.get_classification(token_id)
                if tc.literal_value is not None:
                    try:
                        value = float(tc.literal_value)
                        if not bounds.contains(value):
                            mask[token_id] = False
                    except (TypeError, ValueError):
                        pass

        return mask

    def _update_assignment_context(self, token_text: str) -> None:
        """Update assignment context based on observed token.

        Detects when we're about to assign to a variable so we can
        enforce bounds on the RHS.

        Args:
            token_text: The token text

        Note:
            This is the legacy method. New code should use _update_expression_state.
        """
        stripped = token_text.strip()

        # Check for assignment operator
        if stripped == "=":
            # The variable before = becomes our assignment target
            # Look back in buffer for identifier
            words = self._token_buffer.strip().split()
            if words:
                last_word = words[-1].strip()
                if last_word.isidentifier():
                    self._assignment_context = last_word
        elif stripped in ("\n", ";", ",", ")"):
            # End of statement, clear assignment context
            self._assignment_context = None
        elif stripped and not stripped.isspace() and self._assignment_context is None:
            # We're not in assignment context and saw a non-space token
            # Keep looking for next assignment
            pass

    def _update_expression_state(self, token_text: str) -> None:
        """Update expression state based on observed token.

        Implements a state machine that tracks:
        1. When we enter an assignment (LHS to RHS transition)
        2. What kind of expression we're in (simple vs complex)
        3. Nesting depth for parentheses/brackets/braces
        4. How far we are from the assignment operator

        This enables context-aware bounds checking decisions.

        Args:
            token_text: The decoded token text
        """
        stripped = token_text.strip()
        state = self._expression_state

        # Check for newline before stripping (stripping removes \n)
        is_newline = "\n" in token_text

        # Track statement terminators - reset state
        # Check newlines first before stripping removes them
        if is_newline or stripped == ";":
            # End of statement, but only if we're not inside nested structure
            if state.paren_depth == 0 and state.bracket_depth == 0 and state.brace_depth == 0:
                state.context = ExpressionContext.NONE
                state.context_confidence = ContextConfidence.NONE
                state.target_variable = None
                state.tokens_since_assignment = 0
            return  # Early return, no other processing needed

        # Track nesting depth
        if stripped == "(":
            state.paren_depth += 1
            # If we just entered parens after =, could be grouping or function call
            if state.context == ExpressionContext.SIMPLE_ASSIGNMENT_RHS:
                # Check if previous token was an identifier (function call)
                words = self._token_buffer.strip().split()
                if words and words[-1].strip().isidentifier():
                    state.context = ExpressionContext.FUNCTION_CALL
                    state.context_confidence = ContextConfidence.NONE
                else:
                    state.context = ExpressionContext.COMPOUND_EXPR
                    state.context_confidence = ContextConfidence.MEDIUM
        elif stripped == ")":
            state.paren_depth = max(0, state.paren_depth - 1)
        elif stripped == "[":
            state.bracket_depth += 1
            if state.context == ExpressionContext.SIMPLE_ASSIGNMENT_RHS:
                state.context = ExpressionContext.LIST_LITERAL
                state.context_confidence = ContextConfidence.MEDIUM
        elif stripped == "]":
            state.bracket_depth = max(0, state.bracket_depth - 1)
        elif stripped == "{":
            state.brace_depth += 1
            if state.context == ExpressionContext.SIMPLE_ASSIGNMENT_RHS:
                state.context = ExpressionContext.DICT_LITERAL
                state.context_confidence = ContextConfidence.MEDIUM
        elif stripped == "}":
            state.brace_depth = max(0, state.brace_depth - 1)

        # Track assignment context
        elif stripped == "=":
            # Check for == (equality test, not assignment)
            if self._token_buffer.rstrip().endswith(("=", "!", "<", ">")):
                # This is ==, !=, <=, >= - not an assignment
                if state.context == ExpressionContext.SIMPLE_ASSIGNMENT_RHS:
                    state.context = ExpressionContext.COMPOUND_EXPR
                    state.context_confidence = ContextConfidence.LOW
            else:
                # This is an assignment operator
                words = self._token_buffer.strip().split()
                if words:
                    last_word = words[-1].strip()
                    if last_word.isidentifier():
                        state.target_variable = last_word
                        state.context = ExpressionContext.SIMPLE_ASSIGNMENT_RHS
                        state.context_confidence = ContextConfidence.HIGH
                        state.tokens_since_assignment = 0

        # Track operators that indicate complex expressions
        elif stripped in ("+", "-", "*", "/", "//", "%", "**", "&", "|", "^", "~"):
            if state.context == ExpressionContext.SIMPLE_ASSIGNMENT_RHS:
                state.context = ExpressionContext.COMPOUND_EXPR
                state.context_confidence = ContextConfidence.LOW

        # Track 'if' keyword (ternary conditional)
        elif stripped == "if":
            if state.context in (ExpressionContext.SIMPLE_ASSIGNMENT_RHS,
                                 ExpressionContext.COMPOUND_EXPR):
                state.context = ExpressionContext.CONDITIONAL
                state.context_confidence = ContextConfidence.MEDIUM

        # Track attribute access
        elif stripped == ".":
            if state.context == ExpressionContext.SIMPLE_ASSIGNMENT_RHS:
                state.context = ExpressionContext.ATTRIBUTE_ACCESS
                state.context_confidence = ContextConfidence.LOW

        # Increment tokens since assignment (for non-whitespace tokens)
        if stripped and state.context == ExpressionContext.SIMPLE_ASSIGNMENT_RHS:
            state.tokens_since_assignment += 1

    def _compute_blocking_level(
        self,
        context_conf: ContextConfidence,
        bounds_conf: BoundsConfidence,
    ) -> BlockingLevel:
        """Determine blocking level from context and bounds confidence.

        Implements the decision matrix:

                                 Bounds Confidence
                                 HIGH      MEDIUM    LOW/UNKNOWN
            Context     HIGH     AGGR      AGGR      PERM
            Confidence  MEDIUM   AGGR      CONS      PERM
                        LOW/NONE CONS      PERM      PERM

        Updated to be more assertive when bounds confidence is HIGH,
        since HIGH bounds come from explicit assertions/preconditions
        and are reliable. CONSERVATIVE mode now catches more violations
        (uses 10% margin instead of 2x margin).

        Args:
            context_conf: Confidence in the syntactic context
            bounds_conf: Confidence in the variable bounds

        Returns:
            BlockingLevel indicating how aggressively to block
        """
        # LOW or UNKNOWN bounds confidence -> mostly PERMISSIVE
        if bounds_conf in (BoundsConfidence.LOW, BoundsConfidence.UNKNOWN):
            return BlockingLevel.PERMISSIVE

        # NONE context confidence -> PERMISSIVE (don't know what we're generating)
        if context_conf == ContextConfidence.NONE:
            return BlockingLevel.PERMISSIVE

        # HIGH bounds from explicit assertions/preconditions
        if bounds_conf == BoundsConfidence.HIGH:
            # HIGH or MEDIUM context -> AGGRESSIVE (bounds are reliable)
            if context_conf in (ContextConfidence.HIGH, ContextConfidence.MEDIUM):
                return BlockingLevel.AGGRESSIVE
            # LOW context -> CONSERVATIVE (bounds are good, but context uncertain)
            return BlockingLevel.CONSERVATIVE

        # MEDIUM bounds
        if bounds_conf == BoundsConfidence.MEDIUM:
            if context_conf == ContextConfidence.HIGH:
                return BlockingLevel.AGGRESSIVE
            if context_conf == ContextConfidence.MEDIUM:
                return BlockingLevel.CONSERVATIVE
            # LOW context + MEDIUM bounds -> PERMISSIVE
            return BlockingLevel.PERMISSIVE

        return BlockingLevel.CONSERVATIVE

    def _get_current_blocking_level(self, var_name: str) -> BlockingLevel:
        """Get the blocking level for a variable based on current state.

        Combines the expression state confidence with the variable's
        bounds confidence to determine appropriate blocking level.

        Args:
            var_name: The variable name to check

        Returns:
            BlockingLevel for this variable in current context
        """
        # Get context confidence from expression state
        context_conf = self._expression_state.context_confidence

        # Get bounds confidence for this variable
        if var_name not in self._variable_bounds:
            return BlockingLevel.PERMISSIVE

        bounds = self._variable_bounds[var_name]
        bounds_conf = bounds.effective_confidence()

        return self._compute_blocking_level(context_conf, bounds_conf)

    def observe_token(
        self,
        constraint: SemanticConstraint,
        token_id: int,
        context: GenerationContext,
    ) -> SemanticConstraint:
        """Update the semantic constraint after observing a token.

        Extracts semantic formulas from newly generated code
        and validates against the current constraint. Also updates
        assignment context for aggressive bounds checking.

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

        # Update expression state for context-aware bounds checking
        self._update_expression_state(token_text)

        # Update legacy assignment context for backwards compatibility
        if self._aggressive_mode:
            self._update_assignment_context(token_text)

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
            variable_bounds={
                k: VariableBounds(
                    lower=v.lower,
                    upper=v.upper,
                    is_float=v.is_float,
                    confidence=v.confidence,
                    source=v.source,
                    smt_uncertain=v.smt_uncertain,
                )
                for k, v in self._variable_bounds.items()
            },
            expression_state=self._expression_state.copy(),
            assignment_context=self._assignment_context,
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

        # Restore variable bounds with full confidence tracking
        self._variable_bounds = {
            k: VariableBounds(
                lower=v.lower,
                upper=v.upper,
                is_float=v.is_float,
                confidence=v.confidence,
                source=v.source,
                smt_uncertain=v.smt_uncertain,
            )
            for k, v in checkpoint.variable_bounds.items()
        }

        # Restore expression state (with fallback for old checkpoints)
        if checkpoint.expression_state is not None:
            self._expression_state = checkpoint.expression_state.copy()
        else:
            self._expression_state = ExpressionState()

        # Restore legacy assignment context
        self._assignment_context = checkpoint.assignment_context

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
        self._variable_bounds.clear()
        self._assignment_context = None
        self._expression_state = ExpressionState()

    def add_semantic_constraint(
        self,
        kind: str,
        expression: str,
        scope: Optional[str] = None,
        variables: Optional[List[str]] = None,
    ) -> SemanticConstraint:
        """Add a semantic constraint from external specification.

        Args:
            kind: Constraint kind ("precondition", "postcondition", "invariant", "assertion", "assume")
            expression: Boolean expression in target language
            scope: Where this constraint applies
            variables: Free variables in expression

        Returns:
            Updated SemanticConstraint
        """
        constraint = self.create_constraint()

        if kind == "precondition":
            constraint = constraint.add_precondition(expression, scope)
        elif kind == "postcondition":
            constraint = constraint.add_postcondition(expression, scope)
        elif kind == "invariant":
            constraint = constraint.add_invariant(expression, scope)
        elif kind == "assertion":
            constraint = constraint.add_assertion(expression, scope)
        elif kind == "assume" or kind == "assumption":
            constraint = constraint.add_assumption(expression, scope)
        else:
            # Default to assertion
            constraint = constraint.add_assertion(expression, scope)

        return constraint

    def set_bounds(
        self,
        variable: str,
        lower: Optional[float] = None,
        upper: Optional[float] = None,
        is_float: bool = False,
    ) -> None:
        """Set bounds for a variable directly.

        This is useful when bounds are known from external context
        (e.g., from type annotations or explicit constraints).

        Args:
            variable: Variable name
            lower: Lower bound (inclusive)
            upper: Upper bound (inclusive)
            is_float: Whether the variable is a float
        """
        self._variable_bounds[variable] = VariableBounds(
            lower=lower,
            upper=upper,
            is_float=is_float,
            confidence=BoundsConfidence.HIGH,
            source="external",
        )

    def inject_context(self, spec: Any) -> None:
        """Inject context from a ConstraintSpec.

        Called when a cached grammar object needs fresh context.
        This re-seeds the semantic state with data from the spec.

        Args:
            spec: A ConstraintSpec object (typed as Any to avoid circular import)
        """
        # Import locally to avoid circular dependency
        # Try relative import first, fall back to absolute
        try:
            from ...spec.constraint_spec import ConstraintSpec
        except ImportError:
            try:
                from spec.constraint_spec import ConstraintSpec
            except ImportError:
                # If we can't import, check by class name
                if spec.__class__.__name__ != "ConstraintSpec":
                    return
                ConstraintSpec = spec.__class__

        if not isinstance(spec, ConstraintSpec):
            return

        # Reset state
        self.reset()

        # Add semantic constraints from spec
        for sc in spec.semantic_constraints:
            # Create formula from constraint
            formula_kind = self._map_constraint_kind(sc.kind)
            formula = SMTFormula(
                expression=sc.expression,
                kind=formula_kind,
                name=sc.scope,
            )
            self._accumulated_formulas.append(formula)

            # Extract bounds from the formula
            bounds = self._extract_bounds_from_formula(
                sc.expression,
                self._formula_kind_to_confidence(formula_kind),
            )
            if bounds:
                var_name, bound_info = bounds
                if var_name not in self._variable_bounds:
                    self._variable_bounds[var_name] = bound_info
                else:
                    # Merge bounds
                    existing = self._variable_bounds[var_name]
                    if bound_info.lower is not None:
                        if existing.lower is None or bound_info.lower > existing.lower:
                            existing.lower = bound_info.lower
                    if bound_info.upper is not None:
                        if existing.upper is None or bound_info.upper < existing.upper:
                            existing.upper = bound_info.upper

    def _map_constraint_kind(self, kind_str: str) -> FormulaKind:
        """Map constraint kind string to FormulaKind enum.

        Args:
            kind_str: Kind string from ConstraintSpec

        Returns:
            Corresponding FormulaKind
        """
        kind_map = {
            "precondition": FormulaKind.PRECONDITION,
            "postcondition": FormulaKind.POSTCONDITION,
            "invariant": FormulaKind.INVARIANT,
            "assertion": FormulaKind.ASSERTION,
            "assume": FormulaKind.ASSUMPTION,
            "assumption": FormulaKind.ASSUMPTION,
        }
        return kind_map.get(kind_str.lower(), FormulaKind.ASSERTION)

    def seed_constraints(
        self,
        constraints: List[Tuple[str, str, Optional[str], List[str]]],
    ) -> None:
        """Seed semantic constraints from a list of tuples.

        This is the primary method for initializing semantic context from
        a ConstraintSpec's semantic_constraints.

        Args:
            constraints: List of (kind, expression, scope, variables) tuples
                Note: variables parameter is accepted but not stored in SMTFormula
        """
        for kind, expression, scope, variables in constraints:
            formula_kind = self._map_constraint_kind(kind)
            formula = SMTFormula(
                expression=expression,
                kind=formula_kind,
                name=scope,
            )
            self._accumulated_formulas.append(formula)

            # Extract and store bounds
            bounds = self._extract_bounds_from_formula(
                expression,
                self._formula_kind_to_confidence(formula_kind),
            )
            if bounds:
                var_name, bound_info = bounds
                self._variable_bounds[var_name] = bound_info
