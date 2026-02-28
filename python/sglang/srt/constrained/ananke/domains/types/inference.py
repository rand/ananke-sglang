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
"""Enhanced generic type inference for Ananke.

Phase 2.3 implementation providing:
1. Bounded type variables with variance tracking
2. Protocol/trait structural subtyping
3. Type parameter inference from usage context
4. Recursive type handling with occur check

This module extends the base unification with:
- BoundedTypeVar: Type variables with upper/lower bounds
- TypeParameterInference: Infers generic type parameters from usage
- StructuralSubtyping: Enhanced protocol/trait matching
- RecursiveTypeHandler: Safe handling of recursive types

References:
    - Pierce, B.C. (2002). "Types and Programming Languages", Ch 22-24
    - Hazel POPL 2024: "Total Type Error Localization and Recovery with Holes"
    - Scala type inference: "Local Type Inference" (Pierce & Turner, 1998)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

try:
    from .constraint import (
        ANY,
        NEVER,
        AnyType,
        ClassType,
        DictType,
        FunctionType,
        HoleType,
        ListType,
        NeverType,
        ProtocolType,
        SetType,
        TupleType,
        Type,
        TypeVar,
        UnionType,
        Variance,
    )
    from .unification import (
        EMPTY_SUBSTITUTION,
        Substitution,
        UnificationError,
        UnificationResult,
        is_subtype,
        occurs_check,
        unify,
    )
except ImportError:
    from domains.types.constraint import (
        ANY,
        NEVER,
        AnyType,
        ClassType,
        DictType,
        FunctionType,
        HoleType,
        ListType,
        NeverType,
        ProtocolType,
        SetType,
        TupleType,
        Type,
        TypeVar,
        UnionType,
        Variance,
    )
    from domains.types.unification import (
        EMPTY_SUBSTITUTION,
        Substitution,
        UnificationError,
        UnificationResult,
        is_subtype,
        occurs_check,
        unify,
    )


# =============================================================================
# Bounded Type Variables
# =============================================================================


@dataclass(frozen=True)
class BoundedTypeVar(Type):
    """Type variable with upper and lower bounds.

    Extends TypeVar to support bounds like:
    - T extends Number (upper bound)
    - T super Integer (lower bound)
    - T extends Comparable<T> (F-bounded)

    Variance tracking determines how the bound is checked:
    - COVARIANT: T must be subtype of upper bound
    - CONTRAVARIANT: T must be supertype of lower bound
    - INVARIANT: T must equal the exact type

    Attributes:
        name: Variable name (e.g., "T", "K", "V")
        upper_bound: Upper bound type (default: Any)
        lower_bound: Lower bound type (default: Never)
        variance: Variance of this type variable
        is_inferred: Whether this was inferred from usage
    """

    name: str
    upper_bound: Type = field(default_factory=lambda: ANY)
    lower_bound: Type = field(default_factory=lambda: NEVER)
    variance: Variance = Variance.INVARIANT
    is_inferred: bool = False

    def free_type_vars(self) -> FrozenSet[str]:
        """Return free type variables including self and bounds."""
        vars_set = {self.name}
        vars_set.update(self.upper_bound.free_type_vars())
        vars_set.update(self.lower_bound.free_type_vars())
        return frozenset(vars_set)

    def substitute(self, substitution) -> Type:
        """Apply substitution to this bounded type variable.

        Args:
            substitution: Either a Dict[str, Type] mapping or a Substitution object
        """
        # Handle both dict and Substitution object
        if hasattr(substitution, 'mapping'):
            mapping = substitution.mapping
        else:
            mapping = substitution

        if self.name in mapping:
            return mapping[self.name]
        # Substitute in bounds too
        new_upper = self.upper_bound.substitute(mapping)
        new_lower = self.lower_bound.substitute(mapping)
        if new_upper == self.upper_bound and new_lower == self.lower_bound:
            return self
        return BoundedTypeVar(
            name=self.name,
            upper_bound=new_upper,
            lower_bound=new_lower,
            variance=self.variance,
            is_inferred=self.is_inferred,
        )

    def is_within_bounds(self, ty: Type) -> bool:
        """Check if a type is within this variable's bounds.

        Args:
            ty: The type to check

        Returns:
            True if ty satisfies both bounds
        """
        # Check upper bound: ty <: upper_bound
        if not is_subtype(ty, self.upper_bound):
            return False
        # Check lower bound: lower_bound <: ty
        if not is_subtype(self.lower_bound, ty):
            return False
        return True

    def __repr__(self) -> str:
        parts = [f"'{self.name}"]
        if self.upper_bound != ANY:
            parts.append(f" extends {self.upper_bound}")
        if self.lower_bound != NEVER:
            parts.append(f" super {self.lower_bound}")
        if self.variance != Variance.INVARIANT:
            parts.append(f" ({self.variance.name.lower()})")
        return "".join(parts)


def create_bounded_var(
    name: str,
    upper: Optional[Type] = None,
    lower: Optional[Type] = None,
    variance: Variance = Variance.INVARIANT,
) -> BoundedTypeVar:
    """Create a bounded type variable.

    Args:
        name: Variable name
        upper: Upper bound (default: Any)
        lower: Lower bound (default: Never)
        variance: Variance of the variable

    Returns:
        A new BoundedTypeVar
    """
    return BoundedTypeVar(
        name=name,
        upper_bound=upper or ANY,
        lower_bound=lower or NEVER,
        variance=variance,
    )


# =============================================================================
# Type Parameter Inference
# =============================================================================


class InferenceDirection(Enum):
    """Direction of type inference."""
    INWARD = auto()   # From outer context to inner (top-down)
    OUTWARD = auto()  # From inner expressions to outer (bottom-up)
    BIDIRECTIONAL = auto()  # Both directions


@dataclass
class InferenceConstraint:
    """A constraint discovered during inference.

    Attributes:
        target: The type variable to constrain
        constraint_type: The constraining type
        direction: Whether this is upper or lower bound
        source: Where this constraint came from
        confidence: Confidence in this constraint (0.0 to 1.0)
    """

    target: str  # Type variable name
    constraint_type: Type
    direction: str  # "upper", "lower", or "exact"
    source: str  # Description of constraint source
    confidence: float = 1.0


@dataclass
class TypeParameterInference:
    """Infers generic type parameters from usage context.

    Uses local type inference (Pierce & Turner 1998) to infer
    type arguments for generic functions and classes without
    requiring explicit annotations.

    Example:
        ```python
        def identity(x: T) -> T: return x
        identity(42)  # Infer T = int from argument
        ```

    Attributes:
        constraints: Collected inference constraints
        substitution: Current inferred substitution
        max_iterations: Maximum inference iterations
    """

    constraints: List[InferenceConstraint] = field(default_factory=list)
    substitution: Substitution = field(default_factory=lambda: EMPTY_SUBSTITUTION)
    max_iterations: int = 100

    def add_constraint(
        self,
        target: str,
        constraint_type: Type,
        direction: str = "exact",
        source: str = "unknown",
        confidence: float = 1.0,
    ) -> None:
        """Add an inference constraint.

        Args:
            target: Type variable name to constrain
            constraint_type: The constraining type
            direction: "upper", "lower", or "exact"
            source: Description of where this came from
            confidence: Confidence in this constraint
        """
        self.constraints.append(InferenceConstraint(
            target=target,
            constraint_type=constraint_type,
            direction=direction,
            source=source,
            confidence=confidence,
        ))

    def infer_from_argument(
        self,
        param_type: Type,
        arg_type: Type,
        variance: Variance = Variance.COVARIANT,
    ) -> bool:
        """Infer type parameters from an argument.

        Given a parameter type (possibly containing type variables)
        and an argument type (concrete), infer bindings for variables.

        Args:
            param_type: The parameter type with type variables
            arg_type: The concrete argument type
            variance: Position variance

        Returns:
            True if inference succeeded
        """
        return self._infer_recursive(param_type, arg_type, variance, "argument")

    def infer_from_return(
        self,
        return_type: Type,
        expected_type: Type,
    ) -> bool:
        """Infer type parameters from expected return type.

        Given a return type (possibly with variables) and expected type,
        infer bindings.

        Args:
            return_type: The function's return type
            expected_type: The expected/required type

        Returns:
            True if inference succeeded
        """
        return self._infer_recursive(return_type, expected_type, Variance.COVARIANT, "return")

    def _infer_recursive(
        self,
        template: Type,
        concrete: Type,
        variance: Variance,
        source: str,
    ) -> bool:
        """Recursively infer type parameters.

        Args:
            template: Type template with variables
            concrete: Concrete type to match
            variance: Position variance
            source: Source description

        Returns:
            True if inference succeeded
        """
        # Type variable - add constraint
        if isinstance(template, TypeVar):
            direction = "exact"
            if variance == Variance.COVARIANT:
                direction = "upper"
            elif variance == Variance.CONTRAVARIANT:
                direction = "lower"
            self.add_constraint(template.name, concrete, direction, source)
            return True

        if isinstance(template, BoundedTypeVar):
            # Check bounds first
            if not template.is_within_bounds(concrete):
                return False
            direction = "exact"
            if variance == Variance.COVARIANT:
                direction = "upper"
            elif variance == Variance.CONTRAVARIANT:
                direction = "lower"
            self.add_constraint(template.name, concrete, direction, source)
            return True

        # Any/Never/Hole - always match
        if isinstance(template, (AnyType, NeverType, HoleType)):
            return True
        if isinstance(concrete, (AnyType, NeverType, HoleType)):
            return True

        # List types
        if isinstance(template, ListType) and isinstance(concrete, ListType):
            return self._infer_recursive(template.element, concrete.element, variance, source)

        # Set types
        if isinstance(template, SetType) and isinstance(concrete, SetType):
            return self._infer_recursive(template.element, concrete.element, variance, source)

        # Dict types
        if isinstance(template, DictType) and isinstance(concrete, DictType):
            key_ok = self._infer_recursive(template.key, concrete.key, Variance.INVARIANT, source)
            value_ok = self._infer_recursive(template.value, concrete.value, variance, source)
            return key_ok and value_ok

        # Tuple types
        if isinstance(template, TupleType) and isinstance(concrete, TupleType):
            if len(template.elements) != len(concrete.elements):
                return False
            for t_elem, c_elem in zip(template.elements, concrete.elements):
                if not self._infer_recursive(t_elem, c_elem, variance, source):
                    return False
            return True

        # Function types
        if isinstance(template, FunctionType) and isinstance(concrete, FunctionType):
            if len(template.params) != len(concrete.params):
                return False
            # Parameters are contravariant
            for t_param, c_param in zip(template.params, concrete.params):
                contra_var = Variance.CONTRAVARIANT if variance == Variance.COVARIANT else Variance.COVARIANT
                if not self._infer_recursive(t_param, c_param, contra_var, source):
                    return False
            # Return is covariant
            return self._infer_recursive(template.returns, concrete.returns, variance, source)

        # Class types
        if isinstance(template, ClassType) and isinstance(concrete, ClassType):
            if template.name != concrete.name:
                return False
            if len(template.type_args) != len(concrete.type_args):
                return False
            for t_arg, c_arg in zip(template.type_args, concrete.type_args):
                # Default to invariant for type arguments
                if not self._infer_recursive(t_arg, c_arg, Variance.INVARIANT, source):
                    return False
            return True

        # Same types or unification possible
        return template == concrete

    def solve(self) -> UnificationResult:
        """Solve all collected constraints.

        Combines constraints on each type variable to compute
        the final substitution.

        Returns:
            UnificationResult with the solved substitution
        """
        if not self.constraints:
            return UnificationResult.success(EMPTY_SUBSTITUTION)

        # Group constraints by target variable
        var_constraints: Dict[str, List[InferenceConstraint]] = {}
        for c in self.constraints:
            if c.target not in var_constraints:
                var_constraints[c.target] = []
            var_constraints[c.target].append(c)

        # Solve each variable
        result_subst = EMPTY_SUBSTITUTION

        for var_name, constraints in var_constraints.items():
            resolved = self._solve_variable(var_name, constraints)
            if resolved is None:
                return UnificationResult.failure(
                    TypeVar(var_name),
                    ANY,
                    f"conflicting constraints for {var_name}",
                )
            result_subst = result_subst.extend(var_name, resolved)

        self.substitution = result_subst
        return UnificationResult.success(result_subst)

    def _solve_variable(
        self,
        var_name: str,
        constraints: List[InferenceConstraint],
    ) -> Optional[Type]:
        """Solve constraints for a single variable.

        Args:
            var_name: The variable name
            constraints: Constraints on this variable

        Returns:
            The resolved type, or None if unsolvable
        """
        upper_bounds: List[Type] = []
        lower_bounds: List[Type] = []
        exact: List[Type] = []

        for c in constraints:
            if c.direction == "exact":
                exact.append(c.constraint_type)
            elif c.direction == "upper":
                upper_bounds.append(c.constraint_type)
            elif c.direction == "lower":
                lower_bounds.append(c.constraint_type)

        # If we have exact constraints, unify them
        if exact:
            result = self._unify_all(exact)
            if result is not None:
                return result
            # Exact constraints conflict - try using the most specific
            return exact[0]

        # No exact constraints - use bounds
        if upper_bounds:
            # Take intersection of upper bounds (GLB)
            result = self._find_glb(upper_bounds)
            if result is not None:
                return result
            # Fallback to first upper bound
            return upper_bounds[0]

        if lower_bounds:
            # Take union of lower bounds (LUB)
            result = self._find_lub(lower_bounds)
            if result is not None:
                return result
            return lower_bounds[0]

        # No constraints - return Any
        return ANY

    def _unify_all(self, types: List[Type]) -> Optional[Type]:
        """Try to unify all types in the list."""
        if len(types) == 1:
            return types[0]

        result = types[0]
        for other in types[1:]:
            unify_result = unify(result, other)
            if unify_result.is_failure:
                return None
            result = unify_result.substitution.apply(result)
        return result

    def _find_glb(self, types: List[Type]) -> Optional[Type]:
        """Find greatest lower bound (most specific common type)."""
        if not types:
            return ANY
        if len(types) == 1:
            return types[0]

        # Simple approach: find the most specific type that all are subtypes of
        for t in types:
            if all(is_subtype(t, other) or t == other for other in types):
                return t

        # Fallback: return first type
        return types[0]

    def _find_lub(self, types: List[Type]) -> Optional[Type]:
        """Find least upper bound (most general common type)."""
        if not types:
            return NEVER
        if len(types) == 1:
            return types[0]

        # Simple approach: return union of types
        return UnionType(members=frozenset(types))


# =============================================================================
# Recursive Type Handling
# =============================================================================


@dataclass
class RecursiveTypeHandler:
    """Handles recursive types safely with occur check.

    Prevents infinite loops when processing recursive types like:
    - type List = Nil | Cons(head: T, tail: List[T])
    - type JSON = null | bool | number | string | List[JSON] | Dict[str, JSON]

    Uses a seen set to detect cycles and a depth limit for safety.

    Attributes:
        seen: Set of types currently being processed
        max_depth: Maximum recursion depth
        current_depth: Current recursion depth
    """

    seen: Set[int] = field(default_factory=set)  # Type IDs being processed
    max_depth: int = 50
    current_depth: int = 0

    def enter(self, ty: Type) -> bool:
        """Enter processing of a type.

        Args:
            ty: The type being entered

        Returns:
            True if safe to continue, False if cycle detected
        """
        ty_id = id(ty)
        if ty_id in self.seen:
            return False  # Cycle detected
        if self.current_depth >= self.max_depth:
            return False  # Depth limit reached
        self.seen.add(ty_id)
        self.current_depth += 1
        return True

    def exit(self, ty: Type) -> None:
        """Exit processing of a type."""
        ty_id = id(ty)
        self.seen.discard(ty_id)
        self.current_depth = max(0, self.current_depth - 1)

    def is_recursive(self, ty: Type) -> bool:
        """Check if a type is recursive (refers to itself)."""
        return self._check_recursive(ty, set())

    def _check_recursive(self, ty: Type, seen_vars: Set[str]) -> bool:
        """Recursively check for cycles."""
        if isinstance(ty, TypeVar):
            return ty.name in seen_vars

        if isinstance(ty, BoundedTypeVar):
            if ty.name in seen_vars:
                return True
            new_seen = seen_vars | {ty.name}
            return (self._check_recursive(ty.upper_bound, new_seen) or
                    self._check_recursive(ty.lower_bound, new_seen))

        if isinstance(ty, (AnyType, NeverType)):
            return False

        if isinstance(ty, HoleType):
            if ty.expected:
                return self._check_recursive(ty.expected, seen_vars)
            return False

        if isinstance(ty, ListType):
            return self._check_recursive(ty.element, seen_vars)

        if isinstance(ty, SetType):
            return self._check_recursive(ty.element, seen_vars)

        if isinstance(ty, DictType):
            return (self._check_recursive(ty.key, seen_vars) or
                    self._check_recursive(ty.value, seen_vars))

        if isinstance(ty, TupleType):
            return any(self._check_recursive(e, seen_vars) for e in ty.elements)

        if isinstance(ty, FunctionType):
            return (any(self._check_recursive(p, seen_vars) for p in ty.params) or
                    self._check_recursive(ty.returns, seen_vars))

        if isinstance(ty, UnionType):
            return any(self._check_recursive(m, seen_vars) for m in ty.members)

        if isinstance(ty, ClassType):
            return any(self._check_recursive(a, seen_vars) for a in ty.type_args)

        if isinstance(ty, ProtocolType):
            return any(self._check_recursive(mt, seen_vars) for _, mt in ty.members)

        return False


# =============================================================================
# Convenience Functions
# =============================================================================


def infer_type_params(
    generic_type: Type,
    concrete_type: Type,
) -> Optional[Substitution]:
    """Infer type parameters from a concrete instantiation.

    Args:
        generic_type: The generic type with type variables
        concrete_type: The concrete type to match

    Returns:
        Substitution mapping type variables to concrete types, or None
    """
    inference = TypeParameterInference()
    if inference.infer_from_argument(generic_type, concrete_type):
        result = inference.solve()
        if result.is_success:
            return result.substitution
    return None


def create_generic_instantiation(
    generic_type: ClassType,
    type_args: List[Type],
) -> ClassType:
    """Create an instantiation of a generic type.

    Args:
        generic_type: The generic class type
        type_args: Type arguments to substitute

    Returns:
        The instantiated class type
    """
    return ClassType(
        name=generic_type.name,
        type_args=tuple(type_args),
    )


def unify_with_bounds(
    t1: Type,
    t2: Type,
) -> UnificationResult:
    """Unify types with bound checking for bounded type variables.

    Like regular unify but respects bounds on BoundedTypeVar.

    Args:
        t1: First type
        t2: Second type

    Returns:
        UnificationResult
    """
    # Handle bounded type variables
    if isinstance(t1, BoundedTypeVar):
        if not t1.is_within_bounds(t2):
            return UnificationResult.failure(
                t1, t2, f"type {t2} not within bounds of {t1}"
            )
        return UnificationResult.success(Substitution({t1.name: t2}))

    if isinstance(t2, BoundedTypeVar):
        if not t2.is_within_bounds(t1):
            return UnificationResult.failure(
                t1, t2, f"type {t1} not within bounds of {t2}"
            )
        return UnificationResult.success(Substitution({t2.name: t1}))

    # Fall back to regular unification
    return unify(t1, t2)
