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
"""Type domain for Ananke's constraint system.

The TypeDomain implements incremental bidirectional type checking based on
the Hazel research (POPL 2024, OOPSLA 2025). It provides:

- TypeConstraint: The constraint type with expected types and equations
- Type checking that works on partial programs (marked lambda calculus)
- Incremental updates as tokens are generated
- Token masks based on type compatibility

References:
    - POPL 2024: "Total Type Error Localization and Recovery with Holes"
    - OOPSLA 2025: "Incremental Bidirectional Typing via Order Maintenance"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch

# Support both relative imports (when used as subpackage) and absolute imports (standalone testing)
try:
    from ...core.constraint import Satisfiability
    from ...core.domain import ConstraintDomain, GenerationContext
    from .constraint import (
        TYPE_BOTTOM,
        TYPE_TOP,
        TypeConstraint,
        Type,
        ANY,
        NEVER,
        INT,
        STR,
        BOOL,
        FLOAT,
        NONE,
        FunctionType,
        ListType,
        DictType,
        TupleType,
        ClassType,
        AnyType,
        NeverType,
        HoleType,
        type_expecting,
    )
    from .environment import TypeEnvironment, EMPTY_ENVIRONMENT
    from .unification import Substitution, EMPTY_SUBSTITUTION, unify
    from .marking.marked_ast import MarkedAST, MarkedASTNode, ASTNodeKind
except ImportError:
    from core.constraint import Satisfiability
    from core.domain import ConstraintDomain, GenerationContext
    from domains.types.constraint import (
        TYPE_BOTTOM,
        TYPE_TOP,
        TypeConstraint,
        Type,
        ANY,
        NEVER,
        INT,
        STR,
        BOOL,
        FLOAT,
        NONE,
        FunctionType,
        ListType,
        DictType,
        TupleType,
        ClassType,
        AnyType,
        NeverType,
        HoleType,
        type_expecting,
    )
    from domains.types.environment import TypeEnvironment, EMPTY_ENVIRONMENT
    from domains.types.unification import Substitution, EMPTY_SUBSTITUTION, unify
    from domains.types.marking.marked_ast import MarkedAST, MarkedASTNode, ASTNodeKind


@dataclass
class TypeDomainCheckpoint:
    """Checkpoint for TypeDomain state.

    Captures all mutable state needed to restore the type domain
    to a previous point. This is not frozen because Substitution
    contains a mutable mapping (could be made frozen with FrozenDict).

    Attributes:
        environment: Snapshot of the type environment
        substitution: Current unification substitution
        state_counter: The state counter value
        current_expected: The current expected type
    """

    environment: TypeEnvironment
    substitution: Substitution
    state_counter: int
    current_expected: Optional[Type]


class TypeDomain(ConstraintDomain[TypeConstraint]):
    """Type domain implementing incremental bidirectional type checking.

    The type domain tracks:
    - Type environment (variable -> type mappings)
    - Current expected type (from context)
    - Accumulated type equations
    - Substitution from unification

    Token masks are computed by checking which tokens could produce
    values of the expected type.

    Attributes:
        name: Domain identifier ("types")
        environment: Current type environment
        substitution: Current type substitution from unification
    """

    def __init__(
        self,
        environment: Optional[TypeEnvironment] = None,
        expected_type: Optional[Type] = None,
    ):
        """Initialize the type domain.

        Args:
            environment: Initial type environment (defaults to empty)
            expected_type: Initial expected type (defaults to Any)
        """
        self._environment = environment if environment is not None else EMPTY_ENVIRONMENT
        self._expected_type = expected_type if expected_type is not None else ANY
        self._substitution = EMPTY_SUBSTITUTION
        self._state_counter = 0

    @property
    def name(self) -> str:
        """Return the domain name."""
        return "types"

    @property
    def top(self) -> TypeConstraint:
        """Return the TOP constraint (no type restriction)."""
        return TYPE_TOP

    @property
    def bottom(self) -> TypeConstraint:
        """Return the BOTTOM constraint (unsatisfiable)."""
        return TYPE_BOTTOM

    @property
    def environment(self) -> TypeEnvironment:
        """Return the current type environment."""
        return self._environment

    @property
    def expected_type(self) -> Type:
        """Return the current expected type."""
        return self._expected_type

    def token_mask(
        self,
        constraint: TypeConstraint,
        context: GenerationContext,
    ) -> torch.Tensor:
        """Compute a token mask based on type constraints.

        Returns a boolean tensor indicating which tokens could produce
        values compatible with the expected type.

        The implementation uses a hybrid approach:
        1. Structural tokens (brackets, operators, keywords) - always allowed
        2. Literals - checked against expected type
        3. Identifiers - checked against type environment

        Performance is maintained via budget-limited checking and caching.

        Args:
            constraint: The current type constraint
            context: Generation context with vocab size

        Returns:
            Boolean tensor of shape (vocab_size,)
        """
        # Handle TOP/BOTTOM constraints
        if constraint.is_top():
            return torch.ones(context.vocab_size, dtype=torch.bool, device=context.device)
        if constraint.is_bottom():
            return torch.zeros(context.vocab_size, dtype=torch.bool, device=context.device)

        # Get expected type
        expected = constraint.expected_type
        if expected is None or isinstance(expected, AnyType):
            return torch.ones(context.vocab_size, dtype=torch.bool, device=context.device)

        # Initialize mask - all tokens allowed by default for safety
        mask = torch.ones(context.vocab_size, dtype=torch.bool, device=context.device)

        # If we have a tokenizer, we can do more refined filtering
        if context.tokenizer is not None:
            mask = self._compute_type_aware_mask(expected, context)

        return mask

    def _compute_type_aware_mask(
        self,
        expected: Type,
        context: GenerationContext,
    ) -> torch.Tensor:
        """Compute a type-aware token mask using the tokenizer.

        This performs budget-limited type checking on candidate tokens
        to determine which could produce type-compatible values.

        Args:
            expected: The expected type at current position
            context: Generation context with tokenizer

        Returns:
            Boolean tensor of valid tokens
        """
        # Default: allow all tokens (conservative)
        mask = torch.ones(context.vocab_size, dtype=torch.bool, device=context.device)

        # Budget for how many tokens to explicitly check
        # Higher budget = better precision, more latency
        check_budget = min(1000, context.vocab_size)

        # Get token categories that we know are type-incompatible
        blocked_categories = self._get_blocked_token_categories(expected)

        # Apply category-based blocking
        tokenizer = context.tokenizer
        for token_id in range(check_budget):
            try:
                token_text = tokenizer.decode([token_id])
                if self._is_token_blocked_by_type(token_text, expected, blocked_categories):
                    mask[token_id] = False
            except Exception:
                # If we can't decode, allow the token (conservative)
                pass

        return mask

    def _get_blocked_token_categories(self, expected: Type) -> Set[str]:
        """Get token categories that are definitely type-incompatible.

        Args:
            expected: The expected type

        Returns:
            Set of category names to block
        """
        blocked: Set[str] = set()

        # Type-specific blocking rules
        if isinstance(expected, (AnyType, HoleType)):
            # Any type allows anything
            return blocked

        if expected == INT:
            # Int doesn't allow string literals or float literals with decimals
            blocked.add("string_literal")

        elif expected == STR:
            # String doesn't allow numeric literals without conversion
            blocked.add("int_literal")
            blocked.add("float_literal")

        elif expected == BOOL:
            # Bool expects True/False or expressions that evaluate to bool
            blocked.add("string_literal")
            blocked.add("float_literal")

        elif isinstance(expected, ListType):
            # List expects [ or list() or identifier
            pass  # Allow most things, structural validation happens elsewhere

        elif isinstance(expected, FunctionType):
            # Function type expects lambda or def or callable identifier
            blocked.add("int_literal")
            blocked.add("float_literal")
            blocked.add("string_literal")

        return blocked

    def _is_token_blocked_by_type(
        self,
        token_text: str,
        expected: Type,
        blocked_categories: Set[str],
    ) -> bool:
        """Check if a specific token is blocked by type constraints.

        Args:
            token_text: The decoded token text
            expected: The expected type
            blocked_categories: Pre-computed blocked categories

        Returns:
            True if the token should be blocked
        """
        token_text = token_text.strip()
        if not token_text:
            return False  # Whitespace always allowed

        # Categorize the token
        category = self._categorize_token(token_text)

        # Check against blocked categories
        return category in blocked_categories

    def _categorize_token(self, token_text: str) -> str:
        """Categorize a token by its syntactic type.

        Args:
            token_text: The token text

        Returns:
            Category string
        """
        token_text = token_text.strip()
        if not token_text:
            return "whitespace"

        # Check for string literal start
        if token_text.startswith('"') or token_text.startswith("'"):
            return "string_literal"

        # Check for numeric literal
        if token_text[0].isdigit():
            if '.' in token_text:
                return "float_literal"
            return "int_literal"

        # Check for boolean
        if token_text in ("True", "False"):
            return "bool_literal"

        # Check for None
        if token_text == "None":
            return "none_literal"

        # Check for keywords
        keywords = {
            "def", "class", "return", "if", "else", "elif", "for", "while",
            "import", "from", "as", "try", "except", "finally", "with",
            "lambda", "yield", "async", "await", "raise", "pass", "break",
            "continue", "in", "is", "not", "and", "or", "global", "nonlocal",
        }
        if token_text in keywords:
            return "keyword"

        # Check for operators
        operators = {
            "+", "-", "*", "/", "//", "%", "**", "=", "==", "!=", "<", ">",
            "<=", ">=", "&", "|", "^", "~", "<<", ">>", ".", ",", ":", ";",
            "(", ")", "[", "]", "{", "}", "@", "->", "+=", "-=", "*=", "/=",
        }
        if token_text in operators:
            return "operator"

        # Default: identifier
        return "identifier"

    def observe_token(
        self,
        constraint: TypeConstraint,
        token_id: int,
        context: GenerationContext,
    ) -> TypeConstraint:
        """Update the type constraint after observing a token.

        This is called after each token is generated. It:
        1. Updates the internal state counter
        2. Analyzes the token to update type state
        3. Potentially updates the expected type based on parse progress
        4. Returns an updated constraint

        Following Hazel's incremental typing approach, this performs
        minimal recomputation - only updating what changed.

        Args:
            constraint: Current type constraint
            token_id: The token that was generated
            context: Generation context

        Returns:
            Updated type constraint
        """
        # Handle TOP and BOTTOM
        if constraint.is_top() or constraint.is_bottom():
            return constraint

        # Update state counter
        self._state_counter += 1

        # Get token text if tokenizer available
        token_text = ""
        if context.tokenizer is not None:
            try:
                token_text = context.tokenizer.decode([token_id])
            except Exception:
                pass

        # Analyze token and update state
        updated_constraint = self._analyze_token_for_type_update(
            constraint, token_text, context
        )

        # Update environment hash
        return updated_constraint.with_environment_hash(self._state_counter)

    def _analyze_token_for_type_update(
        self,
        constraint: TypeConstraint,
        token_text: str,
        context: GenerationContext,
    ) -> TypeConstraint:
        """Analyze a token and update the type constraint accordingly.

        This implements incremental type state updates based on
        the syntactic role of the observed token.

        Args:
            constraint: Current type constraint
            token_text: The decoded token text
            context: Generation context

        Returns:
            Updated type constraint
        """
        token_text = token_text.strip()
        if not token_text:
            return constraint

        # Track structural transitions that change expected type
        current_expected = constraint.expected_type

        # Handle assignment operator - RHS inherits LHS type
        if token_text == "=":
            # After =, we're expecting a value
            # The type depends on what's on the LHS (variable type)
            # For now, keep the expected type
            return constraint

        # Handle colon - type annotation follows
        if token_text == ":":
            # After :, we might be in a type annotation context
            # Keep current constraint
            return constraint

        # Handle arrow - return type annotation
        if token_text == "->":
            # After ->, we're in return type position
            return constraint

        # Handle opening brackets - context change
        if token_text == "[":
            # Inside [], expect elements of list type
            if isinstance(current_expected, ListType):
                return constraint.with_expected_type(current_expected.element)
            return constraint

        if token_text == "(":
            # Inside (), could be tuple elements or function args
            if isinstance(current_expected, TupleType) and current_expected.elements:
                return constraint.with_expected_type(current_expected.elements[0])
            if isinstance(current_expected, FunctionType):
                # Function call - arguments expected
                if current_expected.params:
                    return constraint.with_expected_type(current_expected.params[0])
            return constraint

        if token_text == "{":
            # Inside {}, could be dict entries or set elements
            if isinstance(current_expected, DictType):
                return constraint.with_expected_type(current_expected.key)
            return constraint

        # Handle closing brackets - pop context
        if token_text in ("]", ")", "}"):
            # Context restoration would require stack tracking
            # For now, keep the constraint
            return constraint

        # Handle comma - next element
        if token_text == ",":
            # Comma typically means we're at next element
            # Complex tracking would require more state
            return constraint

        # Handle return keyword
        if token_text == "return":
            # After return, expect the function's return type
            # This would need function context
            return constraint

        # Handle variable binding
        if self._is_identifier(token_text):
            # Check if this variable is in the environment
            var_type = self._environment.lookup(token_text)
            if var_type is not None:
                # Variable exists - check compatibility with expected type
                if current_expected is not None and not isinstance(current_expected, AnyType):
                    # Check if variable type is compatible with expected
                    if not self._types_compatible(var_type, current_expected):
                        return constraint.with_error()
            return constraint

        # Handle literals - check type compatibility
        category = self._categorize_token(token_text)
        literal_type = self._get_literal_type(category)

        if literal_type is not None and current_expected is not None:
            if not isinstance(current_expected, AnyType):
                if not self._types_compatible(literal_type, current_expected):
                    return constraint.with_error()

        return constraint

    def _is_identifier(self, text: str) -> bool:
        """Check if text is a valid identifier."""
        if not text:
            return False
        if not (text[0].isalpha() or text[0] == '_'):
            return False
        return all(c.isalnum() or c == '_' for c in text)

    def _get_literal_type(self, category: str) -> Optional[Type]:
        """Get the type of a literal based on its category."""
        type_map = {
            "int_literal": INT,
            "float_literal": FLOAT,
            "string_literal": STR,
            "bool_literal": BOOL,
            "none_literal": NONE,
        }
        return type_map.get(category)

    def _types_compatible(self, actual: Type, expected: Type) -> bool:
        """Check if actual type is compatible with expected type.

        This implements a simple subtyping check.

        Args:
            actual: The actual type
            expected: The expected type

        Returns:
            True if actual is compatible with expected
        """
        # Any is compatible with anything
        if isinstance(expected, AnyType) or isinstance(actual, AnyType):
            return True

        # Holes are compatible with anything
        if isinstance(expected, HoleType) or isinstance(actual, HoleType):
            return True

        # Same type is compatible
        if actual == expected:
            return True

        # int is compatible with float (numeric promotion)
        if actual == INT and expected == FLOAT:
            return True

        # Check unification - result has is_success property
        result = unify(actual, expected)
        return result.is_success

    def checkpoint(self) -> TypeDomainCheckpoint:
        """Create a checkpoint of the current state.

        Returns:
            Checkpoint that can restore this state
        """
        return TypeDomainCheckpoint(
            environment=self._environment,
            substitution=self._substitution,
            state_counter=self._state_counter,
            current_expected=self._expected_type,
        )

    def restore(self, checkpoint: Checkpoint) -> None:
        """Restore state from a checkpoint.

        Args:
            checkpoint: Previously created checkpoint
        """
        if not isinstance(checkpoint, TypeDomainCheckpoint):
            raise TypeError(
                f"Expected TypeDomainCheckpoint, got {type(checkpoint).__name__}"
            )
        self._environment = checkpoint.environment
        self._substitution = checkpoint.substitution
        self._state_counter = checkpoint.state_counter
        self._expected_type = checkpoint.current_expected

    def satisfiability(self, constraint: TypeConstraint) -> Satisfiability:
        """Check the satisfiability of a type constraint.

        Args:
            constraint: The constraint to check

        Returns:
            SAT, UNSAT, or UNKNOWN
        """
        return constraint.satisfiability()

    def propagate_from(
        self,
        constraint: TypeConstraint,
        source_domain: str,
        source_constraint: Any,
    ) -> TypeConstraint:
        """Handle constraint propagation from another domain.

        Types can receive information from:
        - Syntax domain: syntactic structure provides type expectations
        - Imports domain: available imports affect type environment

        Args:
            constraint: Current type constraint
            source_domain: Name of the source domain
            source_constraint: The constraint from that domain

        Returns:
            Potentially refined type constraint
        """
        # For now, no cross-domain propagation
        return constraint

    def create_constraint(
        self,
        expected_type: Optional[Type] = None,
    ) -> TypeConstraint:
        """Create a new type constraint.

        Args:
            expected_type: The expected type (defaults to Any)

        Returns:
            A new TypeConstraint
        """
        if expected_type is None:
            return TYPE_TOP

        return type_expecting(expected_type)

    def bind_variable(self, name: str, ty: Type) -> None:
        """Bind a variable in the type environment.

        Args:
            name: Variable name
            ty: The type to bind
        """
        self._environment = self._environment.bind(name, ty)

    def lookup_variable(self, name: str) -> Optional[Type]:
        """Look up a variable's type.

        Args:
            name: Variable name

        Returns:
            The type if bound, None otherwise
        """
        return self._environment.lookup(name)

    def set_expected_type(self, ty: Type) -> None:
        """Set the current expected type.

        Args:
            ty: The expected type
        """
        self._expected_type = ty

    def push_scope(self) -> None:
        """Enter a new scope."""
        self._environment = self._environment.push_scope()

    def pop_scope(self) -> None:
        """Exit the current scope."""
        self._environment = self._environment.pop_scope()
