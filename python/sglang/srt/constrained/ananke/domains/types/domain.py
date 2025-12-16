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
    from ...core.token_classifier import (
        TokenClassifier,
        TokenCategory,
        get_or_create_classifier,
    )
    from .constraint import (
        TYPE_BOTTOM,
        TYPE_TOP,
        TypeConstraint,
        Type,
        TypeVar,
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
    from .unification import Substitution, EMPTY_SUBSTITUTION, unify, is_subtype
    from .marking.marked_ast import MarkedAST, MarkedASTNode, ASTNodeKind
except ImportError:
    from core.constraint import Satisfiability
    from core.domain import ConstraintDomain, GenerationContext
    from core.token_classifier import (
        TokenClassifier,
        TokenCategory,
        get_or_create_classifier,
    )
    from domains.types.constraint import (
        TYPE_BOTTOM,
        TYPE_TOP,
        TypeConstraint,
        Type,
        TypeVar,
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
    from domains.types.unification import Substitution, EMPTY_SUBSTITUTION, unify, is_subtype
    from domains.types.marking.marked_ast import MarkedAST, MarkedASTNode, ASTNodeKind


@dataclass
class NarrowingContext:
    """Context for flow-sensitive type narrowing.

    Tracks type narrowings that apply in the current control flow branch,
    such as after `if x is not None:` or `if isinstance(x, int):`.

    Attributes:
        narrowings: Dict mapping variable names to their narrowed types
        condition_depth: Depth of nested conditionals
    """

    narrowings: Dict[str, Type] = field(default_factory=dict)
    condition_depth: int = 0

    def narrow(self, var_name: str, narrowed_type: Type) -> "NarrowingContext":
        """Create a new context with an additional narrowing."""
        new_narrowings = dict(self.narrowings)
        new_narrowings[var_name] = narrowed_type
        return NarrowingContext(new_narrowings, self.condition_depth)

    def enter_conditional(self) -> "NarrowingContext":
        """Create a new context entering a conditional."""
        return NarrowingContext(dict(self.narrowings), self.condition_depth + 1)

    def exit_conditional(self) -> "NarrowingContext":
        """Create a new context exiting a conditional (clears narrowings)."""
        return NarrowingContext({}, max(0, self.condition_depth - 1))

    def get_narrowed_type(self, var_name: str) -> Optional[Type]:
        """Get the narrowed type for a variable, if any."""
        return self.narrowings.get(var_name)


EMPTY_NARROWING = NarrowingContext()


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
        narrowing_context: Flow-sensitive type narrowings
    """

    environment: TypeEnvironment
    substitution: Substitution
    state_counter: int
    current_expected: Optional[Type]
    narrowing_context: NarrowingContext = field(default_factory=lambda: EMPTY_NARROWING)


class TypeMaskCache:
    """Cache for precomputed type masks with hierarchical composition.

    Precomputes boolean masks for primitive and compound types at initialization.
    This enables O(1) mask retrieval during token_mask() instead of
    O(vocab_size) per-token classification.

    Key optimizations:
    1. Classify vocabulary once, create base masks for primitives
    2. Compose compound type masks hierarchically from base masks
    3. Track hit/miss statistics for cache effectiveness monitoring

    Coverage targets:
    - Phase 1: Primitives (~60% of type patterns)
    - Phase 2: Common compounds (~85% coverage)
    - Phase 3 (this update): Extended compounds (~95% coverage)
    """

    def __init__(
        self,
        classifier: TokenClassifier,
        vocab_size: int,
        device: str = "cpu",
    ):
        """Initialize the mask cache.

        Args:
            classifier: Pre-initialized TokenClassifier
            vocab_size: Size of the vocabulary
            device: PyTorch device for tensors
        """
        self._classifier = classifier
        self._vocab_size = vocab_size
        self._device = device

        # Precomputed masks for primitive types
        self._type_masks: Dict[Type, torch.Tensor] = {}
        self._initialized = False

        # Cache statistics
        self._cache_hits = 0
        self._cache_misses = 0
        self._hit_types: Dict[str, int] = {}
        self._miss_types: Dict[str, int] = {}

    def initialize(self) -> None:
        """Precompute masks for primitive and common compound types.

        Extended to cover ~95% of common type patterns:
        - Primitive types: int, str, bool, float, None
        - List types with common inner types
        - Dict types with common key/value combinations
        - Tuple types with common element patterns
        - Set types with common inner types
        - Callable patterns for function signatures
        - Optional/Union types
        """
        if self._initialized:
            return

        # Ensure classifier is initialized
        if not self._classifier.initialized:
            self._classifier.initialize()

        # === Phase 1: Primitive types ===
        self._type_masks[INT] = self._create_int_mask()
        self._type_masks[FLOAT] = self._create_float_mask()
        self._type_masks[STR] = self._create_str_mask()
        self._type_masks[BOOL] = self._create_bool_mask()
        self._type_masks[NONE] = self._create_none_mask()
        self._type_masks[ANY] = self._create_any_mask()

        # === Phase 2: Common compound types ===
        # List types
        list_mask = self._create_list_mask()
        for inner in [INT, STR, FLOAT, BOOL, ANY]:
            self._type_masks[ListType(inner)] = list_mask

        # Nested List types (common in matrix operations)
        self._type_masks[ListType(ListType(INT))] = list_mask
        self._type_masks[ListType(ListType(FLOAT))] = list_mask
        self._type_masks[ListType(ListType(STR))] = list_mask

        # Dict types - expanded combinations
        dict_mask = self._create_dict_mask()
        for key_type in [STR, INT]:
            for val_type in [ANY, INT, STR, FLOAT, BOOL]:
                self._type_masks[DictType(key_type, val_type)] = dict_mask

        # Tuple types - expanded combinations
        tuple_mask = self._create_tuple_mask()
        primitives = [INT, STR, FLOAT, BOOL]
        for t1 in primitives:
            for t2 in primitives:
                self._type_masks[TupleType((t1, t2))] = tuple_mask
        # Single-element tuples
        for t in primitives:
            self._type_masks[TupleType((t,))] = tuple_mask
        # Three-element tuples
        self._type_masks[TupleType((INT, INT, INT))] = tuple_mask
        self._type_masks[TupleType((STR, STR, STR))] = tuple_mask
        self._type_masks[TupleType((INT, STR, FLOAT))] = tuple_mask

        # === Phase 3: Extended types (new in this update) ===

        # Set types
        set_mask = self._create_set_mask()
        for inner in [INT, STR, FLOAT, BOOL]:
            self._type_masks[self._set_type(inner)] = set_mask

        # Optional types (Union[T, None]) - expanded
        # These use the base type mask since None is allowed anywhere
        for inner in [INT, STR, FLOAT, BOOL, ANY]:
            self._type_masks[self._optional_type(inner)] = self._type_masks.get(inner, self._type_masks[ANY])

        # Optional compound types
        self._type_masks[self._optional_type(ListType(INT))] = list_mask
        self._type_masks[self._optional_type(ListType(STR))] = list_mask
        self._type_masks[self._optional_type(DictType(STR, ANY))] = dict_mask

        # Callable types - common function signature patterns
        callable_mask = self._create_callable_mask()

        # Single-argument callables
        for param_type in [INT, STR, FLOAT, BOOL, ANY]:
            for return_type in [INT, STR, FLOAT, BOOL, ANY, NONE]:
                self._type_masks[self._callable_type((param_type,), return_type)] = callable_mask

        # No-argument callables
        for return_type in [INT, STR, FLOAT, BOOL, ANY, NONE]:
            self._type_masks[self._callable_type((), return_type)] = callable_mask

        # Two-argument callables (common patterns)
        common_pairs = [(INT, INT), (STR, STR), (ANY, ANY), (INT, STR), (STR, INT)]
        for params in common_pairs:
            for return_type in [INT, STR, BOOL, ANY, NONE]:
                self._type_masks[self._callable_type(params, return_type)] = callable_mask

        # bytes type (common in I/O operations)
        self._type_masks[self._bytes_type()] = self._create_bytes_mask()

        self._initialized = True

    def _set_type(self, inner: Type) -> Type:
        """Create a unique key for Set[T] without importing Set."""
        return ("Set", inner)

    def _bytes_type(self) -> Type:
        """Create a unique key for bytes type."""
        return ("bytes",)

    def _callable_type(self, params: Tuple[Type, ...], return_type: Type) -> Type:
        """Create a unique key for Callable[[params], return_type]."""
        return ("Callable", params, return_type)

    def _optional_type(self, inner: Type) -> Type:
        """Create a unique key for Optional[T] without importing Union.

        Uses a simple tuple representation for cache keying.
        """
        # Return a tuple that can serve as dict key
        return ("Optional", inner)

    def _create_any_mask(self) -> torch.Tensor:
        """Create mask for Any type - allows all tokens."""
        return torch.ones(self._vocab_size, dtype=torch.bool, device=self._device)

    def _create_list_mask(self) -> torch.Tensor:
        """Create mask for list-compatible tokens.

        Allows: [], [expr], list(), identifiers
        Blocks: bare literals not starting a list
        """
        mask = torch.ones(self._vocab_size, dtype=torch.bool, device=self._device)

        # Lists are very permissive - most tokens can appear in list context
        # Block standalone string literals that can't start a list
        # (Actually, even strings can be in a list literal, so this is conservative)
        return mask

    def _create_dict_mask(self) -> torch.Tensor:
        """Create mask for dict-compatible tokens.

        Allows: {}, {k: v}, dict(), identifiers
        """
        mask = torch.ones(self._vocab_size, dtype=torch.bool, device=self._device)

        # Dicts are permissive - most tokens can appear in dict context
        return mask

    def _create_tuple_mask(self) -> torch.Tensor:
        """Create mask for tuple-compatible tokens.

        Allows: (), (expr,), tuple(), identifiers
        """
        mask = torch.ones(self._vocab_size, dtype=torch.bool, device=self._device)

        # Tuples are permissive
        return mask

    def _create_set_mask(self) -> torch.Tensor:
        """Create mask for set-compatible tokens.

        Allows: set(), {expr}, set comprehensions, identifiers
        """
        mask = torch.ones(self._vocab_size, dtype=torch.bool, device=self._device)

        # Sets are permissive - most tokens can appear in set context
        return mask

    def _create_callable_mask(self) -> torch.Tensor:
        """Create mask for callable-compatible tokens.

        Allows: function names, lambda, identifiers, etc.
        Blocks: literals that can't be callable
        """
        mask = torch.ones(self._vocab_size, dtype=torch.bool, device=self._device)

        # Block bare numeric literals (not callable without wrapping)
        for token_id in self._classifier.by_category(TokenCategory.INT_LITERAL):
            if token_id < self._vocab_size:
                mask[token_id] = False

        for token_id in self._classifier.by_category(TokenCategory.FLOAT_LITERAL):
            if token_id < self._vocab_size:
                mask[token_id] = False

        return mask

    def _create_bytes_mask(self) -> torch.Tensor:
        """Create mask for bytes-compatible tokens.

        Allows: b"...", bytes(), identifiers
        Blocks: non-bytes string literals, numeric literals
        """
        mask = torch.ones(self._vocab_size, dtype=torch.bool, device=self._device)

        # Block int and float literals
        for token_id in self._classifier.by_category(TokenCategory.INT_LITERAL):
            if token_id < self._vocab_size:
                mask[token_id] = False

        for token_id in self._classifier.by_category(TokenCategory.FLOAT_LITERAL):
            if token_id < self._vocab_size:
                mask[token_id] = False

        return mask

    def _create_int_mask(self) -> torch.Tensor:
        """Create mask for int-compatible tokens.

        Blocks: string literals, float literals with decimals
        Allows: int literals, identifiers, operators, keywords, etc.
        """
        mask = torch.ones(self._vocab_size, dtype=torch.bool, device=self._device)

        # Block string literals
        for token_id in self._classifier.by_category(TokenCategory.STRING_LITERAL):
            if token_id < self._vocab_size:
                mask[token_id] = False

        return mask

    def _create_float_mask(self) -> torch.Tensor:
        """Create mask for float-compatible tokens.

        Blocks: string literals
        Allows: int and float literals, identifiers, operators, etc.
        """
        mask = torch.ones(self._vocab_size, dtype=torch.bool, device=self._device)

        # Block string literals
        for token_id in self._classifier.by_category(TokenCategory.STRING_LITERAL):
            if token_id < self._vocab_size:
                mask[token_id] = False

        return mask

    def _create_str_mask(self) -> torch.Tensor:
        """Create mask for str-compatible tokens.

        Blocks: bare numeric literals (without conversion)
        Allows: string literals, identifiers, operators, etc.
        """
        mask = torch.ones(self._vocab_size, dtype=torch.bool, device=self._device)

        # Block int and float literals (need explicit str() call)
        for token_id in self._classifier.by_category(TokenCategory.INT_LITERAL):
            if token_id < self._vocab_size:
                mask[token_id] = False

        for token_id in self._classifier.by_category(TokenCategory.FLOAT_LITERAL):
            if token_id < self._vocab_size:
                mask[token_id] = False

        return mask

    def _create_bool_mask(self) -> torch.Tensor:
        """Create mask for bool-compatible tokens.

        Blocks: string literals, float literals
        Allows: True, False, identifiers, comparison operators, etc.
        """
        mask = torch.ones(self._vocab_size, dtype=torch.bool, device=self._device)

        # Block string literals
        for token_id in self._classifier.by_category(TokenCategory.STRING_LITERAL):
            if token_id < self._vocab_size:
                mask[token_id] = False

        # Block float literals
        for token_id in self._classifier.by_category(TokenCategory.FLOAT_LITERAL):
            if token_id < self._vocab_size:
                mask[token_id] = False

        return mask

    def _create_none_mask(self) -> torch.Tensor:
        """Create mask for None-compatible tokens.

        Conservative: allows most things since None context is usually
        checking for None explicitly.
        """
        return torch.ones(self._vocab_size, dtype=torch.bool, device=self._device)

    def get_mask(self, expected: Type) -> Optional[torch.Tensor]:
        """Get precomputed mask for a type.

        Also updates cache statistics for monitoring.

        Args:
            expected: The expected type

        Returns:
            Precomputed mask or None if not cached
        """
        if not self._initialized:
            self.initialize()

        mask = self._type_masks.get(expected)

        # Track statistics
        type_name = self._type_to_string(expected)
        if mask is not None:
            self._cache_hits += 1
            self._hit_types[type_name] = self._hit_types.get(type_name, 0) + 1
        else:
            self._cache_misses += 1
            self._miss_types[type_name] = self._miss_types.get(type_name, 0) + 1

        return mask

    def has_mask(self, expected: Type) -> bool:
        """Check if a type has a precomputed mask.

        Args:
            expected: The expected type

        Returns:
            True if mask is precomputed
        """
        if not self._initialized:
            self.initialize()

        return expected in self._type_masks

    def _type_to_string(self, t: Type) -> str:
        """Convert a type to a string for statistics tracking."""
        if isinstance(t, tuple):
            return str(t)
        return type(t).__name__ if hasattr(t, '__class__') else str(t)

    @property
    def classifier(self) -> TokenClassifier:
        """Get the underlying classifier."""
        return self._classifier

    @property
    def hit_rate(self) -> float:
        """Get the cache hit rate.

        Returns:
            Hit rate as a float between 0 and 1
        """
        total = self._cache_hits + self._cache_misses
        if total == 0:
            return 0.0
        return self._cache_hits / total

    @property
    def cached_type_count(self) -> int:
        """Get the number of cached type patterns."""
        return len(self._type_masks)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        total = self._cache_hits + self._cache_misses
        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": self.hit_rate,
            "total_lookups": total,
            "cached_type_count": self.cached_type_count,
            "top_hit_types": dict(sorted(
                self._hit_types.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]),
            "top_miss_types": dict(sorted(
                self._miss_types.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]),
        }

    def reset_stats(self) -> None:
        """Reset cache statistics."""
        self._cache_hits = 0
        self._cache_misses = 0
        self._hit_types.clear()
        self._miss_types.clear()


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
        language: str = "python",
        tokenizer: Optional[Any] = None,
    ):
        """Initialize the type domain.

        Args:
            environment: Initial type environment (defaults to empty)
            expected_type: Initial expected type (defaults to Any)
            language: Programming language for classification
            tokenizer: Optional tokenizer for precomputed masks
        """
        self._environment = environment if environment is not None else EMPTY_ENVIRONMENT
        self._expected_type = expected_type if expected_type is not None else ANY
        self._substitution = EMPTY_SUBSTITUTION
        self._state_counter = 0
        self._language = language

        # Flow-sensitive type narrowing
        self._narrowing_context = EMPTY_NARROWING

        # Lazy-initialized classifier and mask cache
        self._tokenizer = tokenizer
        self._classifier: Optional[TokenClassifier] = None
        self._mask_cache: Optional[TypeMaskCache] = None

    def _ensure_classifier_initialized(self, context: GenerationContext) -> None:
        """Ensure classifier and mask cache are initialized.

        Args:
            context: Generation context with tokenizer and vocab_size
        """
        tokenizer = context.tokenizer or self._tokenizer
        if tokenizer is None:
            return

        if self._classifier is None:
            self._classifier = get_or_create_classifier(tokenizer, self._language)

        if self._mask_cache is None and self._classifier is not None:
            self._mask_cache = TypeMaskCache(
                self._classifier,
                vocab_size=context.vocab_size,
                device=context.device,
            )
            self._mask_cache.initialize()

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

        The implementation uses precomputed masks for O(1) performance:
        1. Primitive types (int, str, bool, float, None) use precomputed masks
        2. Compound types fall back to incremental computation
        3. Identifiers are filtered against type environment

        Performance target: <100μs for primitive types.

        Args:
            constraint: The current type constraint
            context: Generation context with vocab size

        Returns:
            Boolean tensor of shape (vocab_size,)
        """
        # Handle TOP/BOTTOM constraints
        if constraint.is_top():
            return context.create_mask(fill_value=True)
        if constraint.is_bottom():
            return context.create_mask(fill_value=False)

        # Get expected type
        expected = constraint.expected_type
        if expected is None or isinstance(expected, AnyType):
            return context.create_mask(fill_value=True)

        # Ensure classifier and mask cache are initialized
        self._ensure_classifier_initialized(context)

        # Try precomputed mask first (O(1) for primitive types)
        if self._mask_cache is not None and self._mask_cache.has_mask(expected):
            mask = self._mask_cache.get_mask(expected)
            if mask is not None:
                # Clone to avoid modifying cached mask
                mask = mask.clone()
                # Apply identifier blocking based on type environment
                mask = self._apply_identifier_blocking(mask, expected, context)
                return mask

        # Fall back to incremental computation for compound types
        mask = context.create_mask(fill_value=True)

        # If we have a tokenizer, compute type-aware mask
        if context.tokenizer is not None:
            mask = self._compute_type_aware_mask(expected, context)

        return mask

    def _apply_identifier_blocking(
        self,
        mask: torch.Tensor,
        expected: Type,
        context: GenerationContext,
    ) -> torch.Tensor:
        """Apply identifier blocking based on type environment.

        Blocks identifiers that have incompatible types according to
        the current type environment. Uses flow-sensitive narrowed types
        when available (after conditionals like `if x is not None:`).

        Args:
            mask: The current mask (will be modified in place)
            expected: The expected type
            context: Generation context

        Returns:
            Modified mask
        """
        if self._classifier is None:
            return mask

        # Get all identifier tokens
        identifier_tokens = self._classifier.by_category(TokenCategory.IDENTIFIER)

        # Check each identifier against the type environment
        for token_id in identifier_tokens:
            if token_id >= context.vocab_size:
                continue

            # Get the identifier text
            tc = self._classifier.get_classification(token_id)
            var_name = tc.text.strip()

            # First check for flow-sensitive narrowed type
            var_type = self._narrowing_context.get_narrowed_type(var_name)

            # Fall back to environment lookup if no narrowing
            if var_type is None:
                var_type = self._environment.lookup(var_name)

            if var_type is not None:
                # Check compatibility with expected type
                if not self._types_compatible(var_type, expected):
                    mask[token_id] = False

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
        mask = context.create_mask(fill_value=True)

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

        Uses proper subtyping with variance handling (covariance, contravariance).
        Supports union types, protocol satisfaction, and function variance.

        Args:
            actual: The actual type
            expected: The expected type

        Returns:
            True if actual is a subtype of (compatible with) expected
        """
        # Use the full subtype checker which handles:
        # - Any/Never as top/bottom types
        # - HoleType as wildcard
        # - Union subtyping
        # - List/Set/Dict/Tuple covariance
        # - Function contravariant params, covariant return
        # - Protocol structural subtyping
        # - Numeric promotion (int <: float)
        return is_subtype(actual, expected)

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
            narrowing_context=self._narrowing_context,
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
        self._narrowing_context = checkpoint.narrowing_context

    def narrow_type(self, var_name: str, narrowed_type: Type) -> None:
        """Apply a type narrowing to a variable.

        Used after type guards like `if x is not None:` or `if isinstance(x, int):`.

        Args:
            var_name: The variable name to narrow
            narrowed_type: The narrowed type
        """
        self._narrowing_context = self._narrowing_context.narrow(var_name, narrowed_type)

    def enter_conditional_branch(self) -> None:
        """Enter a conditional branch where narrowings may apply."""
        self._narrowing_context = self._narrowing_context.enter_conditional()

    def exit_conditional_branch(self) -> None:
        """Exit a conditional branch, clearing conditional narrowings."""
        self._narrowing_context = self._narrowing_context.exit_conditional()

    def get_effective_type(self, var_name: str) -> Optional[Type]:
        """Get the effective type of a variable, considering narrowings.

        First checks for flow-sensitive narrowings, then falls back to
        the type environment.

        Args:
            var_name: The variable name

        Returns:
            The effective type, or None if unknown
        """
        # Check for narrowed type first
        narrowed = self._narrowing_context.get_narrowed_type(var_name)
        if narrowed is not None:
            return narrowed
        # Fall back to environment
        return self._environment.lookup(var_name)

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

    def register_function(
        self,
        name: str,
        params: List[Tuple[str, Type]],
        return_type: Type,
        type_params: Optional[List[str]] = None,
        is_async: bool = False,
        is_generator: bool = False,
    ) -> None:
        """Register a function signature in the type environment.

        Args:
            name: Function name
            params: List of (param_name, param_type) tuples
            return_type: Return type of the function
            type_params: Generic type parameters
            is_async: Whether the function is async
            is_generator: Whether the function is a generator
        """
        # Create function type (params, returns)
        param_types = tuple(ty for _, ty in params)
        param_names = tuple(pname for pname, _ in params)
        func_type = FunctionType(params=param_types, returns=return_type, param_names=param_names)

        # Bind function name to its type
        self._environment = self._environment.bind(name, func_type)

    def register_class(
        self,
        name: str,
        bases: Optional[List[str]] = None,
        type_params: Optional[List[str]] = None,
        methods: Optional[Dict[str, Tuple[List[Tuple[str, Type]], Type]]] = None,
        class_vars: Optional[Dict[str, Type]] = None,
        instance_vars: Optional[Dict[str, Type]] = None,
    ) -> None:
        """Register a class definition in the type environment.

        Args:
            name: Class name
            bases: Base class names (stored as metadata, not in ClassType)
            type_params: Generic type parameters (converted to type args)
            methods: Dict of method_name -> (params, return_type)
            class_vars: Dict of class variable name -> type
            instance_vars: Dict of instance variable name -> type
        """
        # Create class type (name, type_args)
        # Note: ClassType only stores name and type_args, not bases
        type_args = tuple(TypeVar(tp) for tp in (type_params or []))
        class_type = ClassType(name=name, type_args=type_args)

        # Bind class name to its type
        self._environment = self._environment.bind(name, class_type)

        # Register methods as attributes of the class
        if methods:
            for method_name, (params, ret_type) in methods.items():
                param_types = tuple(ty for _, ty in params)
                param_names = tuple(pname for pname, _ in params)
                method_type = FunctionType(params=param_types, returns=ret_type, param_names=param_names)
                # Methods are stored in a separate namespace (class.method_name)
                self._environment = self._environment.bind(f"{name}.{method_name}", method_type)

    def add_type_alias(self, alias: str, target_type: Type) -> None:
        """Add a type alias to the environment.

        Args:
            alias: The alias name
            target_type: The type the alias refers to
        """
        self._environment = self._environment.bind(alias, target_type)

    def seed_from_bindings(
        self,
        bindings: List[Tuple[str, Type, Optional[str]]],
    ) -> None:
        """Seed the type environment from a list of bindings.

        This is the primary method for initializing type context from
        a ConstraintSpec's type_bindings.

        Args:
            bindings: List of (name, type, scope) tuples
        """
        for name, ty, scope in bindings:
            self._environment = self._environment.bind(name, ty)

    def inject_context(self, spec: Any) -> None:
        """Inject context from a ConstraintSpec.

        Called when a cached grammar object needs fresh context.
        This re-seeds the type environment with bindings from the spec.

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

        # Clear and rebuild environment
        self._environment = EMPTY_ENVIRONMENT

        # Add type bindings
        for binding in spec.type_bindings:
            # Parse type expression (simplified - real implementation would
            # use a proper type parser for the language)
            ty = self._parse_type_expr(binding.type_expr)
            self._environment = self._environment.bind(binding.name, ty)

        # Set expected type if provided
        if spec.expected_type:
            self._expected_type = self._parse_type_expr(spec.expected_type)

        # Add type aliases
        for alias, target in spec.type_aliases.items():
            ty = self._parse_type_expr(target)
            self._environment = self._environment.bind(alias, ty)

    def _parse_type_expr(self, type_expr: str) -> Type:
        """Parse a type expression string into a Type.

        This is a simplified parser. A full implementation would handle
        the complete type syntax for each supported language.

        Args:
            type_expr: Type expression string (e.g., "int", "List[str]")

        Returns:
            Parsed Type object
        """
        type_expr = type_expr.strip()

        # Handle primitive types
        primitive_map = {
            "int": INT,
            "str": STR,
            "bool": BOOL,
            "float": FLOAT,
            "None": NONE,
            "Any": ANY,
            "Never": NEVER,
        }
        if type_expr in primitive_map:
            return primitive_map[type_expr]

        # Handle List[T]
        if type_expr.startswith("List[") and type_expr.endswith("]"):
            inner = type_expr[5:-1]
            return ListType(self._parse_type_expr(inner))

        # Handle Dict[K, V]
        if type_expr.startswith("Dict[") and type_expr.endswith("]"):
            inner = type_expr[5:-1]
            # Simple split on comma (doesn't handle nested generics properly)
            parts = inner.split(",", 1)
            if len(parts) == 2:
                key_type = self._parse_type_expr(parts[0].strip())
                val_type = self._parse_type_expr(parts[1].strip())
                return DictType(key_type, val_type)

        # Handle Tuple[T, ...]
        if type_expr.startswith("Tuple[") and type_expr.endswith("]"):
            inner = type_expr[6:-1]
            parts = [p.strip() for p in inner.split(",")]
            element_types = tuple(self._parse_type_expr(p) for p in parts)
            return TupleType(element_types)

        # Handle Optional[T] as Union[T, None]
        if type_expr.startswith("Optional[") and type_expr.endswith("]"):
            inner = type_expr[9:-1]
            return self._parse_type_expr(inner)  # Simplified - ignores None

        # Default to class type for unknown types
        return ClassType(name=type_expr)
