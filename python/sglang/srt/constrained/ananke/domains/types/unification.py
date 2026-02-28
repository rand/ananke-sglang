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
"""Type unification algorithm for Ananke.

Implements Robinson's unification algorithm with occurs check for type inference.
Supports all types in the Ananke type hierarchy including holes.

The unification algorithm finds a substitution (mapping from type variables to types)
that makes two types equal, if one exists.

References:
    - Robinson, J.A. (1965). "A Machine-Oriented Logic Based on the Resolution Principle"
    - Pierce, B.C. (2002). "Types and Programming Languages", Chapter 22
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Tuple

# Support both relative imports (when used as subpackage) and absolute imports (standalone testing)
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
        TypeEquation,
        TypeVar,
        UnionType,
        Variance,
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
        TypeEquation,
        TypeVar,
        UnionType,
        Variance,
    )


@dataclass
class Substitution:
    """A type substitution mapping type variable names to types.

    Substitutions are the result of unification - they tell us what
    concrete types should replace type variables.

    Attributes:
        mapping: Dictionary from variable names to their bound types
    """

    mapping: Dict[str, Type] = field(default_factory=dict)

    def apply(self, ty: Type) -> Type:
        """Apply this substitution to a type.

        Recursively replaces all type variables in the given type
        with their bound values from this substitution.

        Args:
            ty: The type to apply substitution to

        Returns:
            A new type with all substitutions applied
        """
        return ty.substitute(self.mapping)

    def apply_to_equation(self, eq: TypeEquation) -> TypeEquation:
        """Apply this substitution to a type equation."""
        return TypeEquation(
            self.apply(eq.lhs), self.apply(eq.rhs), eq.origin
        )

    def compose(self, other: Substitution) -> Substitution:
        """Compose two substitutions (self ∘ other).

        Applying the composed substitution is equivalent to first applying
        `other`, then applying `self`.

        Args:
            other: The substitution to compose with

        Returns:
            A new substitution representing self ∘ other
        """
        # Apply self to all types in other's mapping
        new_mapping = {k: self.apply(v) for k, v in other.mapping.items()}
        # Add self's mappings (they take precedence for new variables)
        for k, v in self.mapping.items():
            if k not in new_mapping:
                new_mapping[k] = v
        return Substitution(new_mapping)

    def extend(self, var: str, ty: Type) -> Substitution:
        """Extend this substitution with a new binding.

        Args:
            var: The type variable name to bind
            ty: The type to bind it to

        Returns:
            A new substitution with the additional binding
        """
        new_mapping = dict(self.mapping)
        new_mapping[var] = ty
        return Substitution(new_mapping)

    def __repr__(self) -> str:
        if not self.mapping:
            return "Substitution({})"
        items = ", ".join(f"'{k} → {v}" for k, v in sorted(self.mapping.items()))
        return f"Substitution({{{items}}})"


# Empty substitution singleton
EMPTY_SUBSTITUTION = Substitution()


def occurs_check(var: TypeVar, ty: Type) -> bool:
    """Check if a type variable occurs in a type.

    The occurs check prevents infinite types like α = List[α].
    If α occurs in τ, then α cannot be unified with τ.

    Args:
        var: The type variable to look for
        ty: The type to search in

    Returns:
        True if var occurs in ty, False otherwise
    """
    if isinstance(ty, TypeVar):
        return ty.name == var.name

    if isinstance(ty, (AnyType, NeverType)):
        return False

    if isinstance(ty, HoleType):
        if ty.expected:
            return occurs_check(var, ty.expected)
        return False

    if isinstance(ty, ListType):
        return occurs_check(var, ty.element)

    if isinstance(ty, SetType):
        return occurs_check(var, ty.element)

    if isinstance(ty, DictType):
        return occurs_check(var, ty.key) or occurs_check(var, ty.value)

    if isinstance(ty, TupleType):
        return any(occurs_check(var, elem) for elem in ty.elements)

    if isinstance(ty, FunctionType):
        return any(occurs_check(var, p) for p in ty.params) or occurs_check(
            var, ty.returns
        )

    if isinstance(ty, UnionType):
        return any(occurs_check(var, m) for m in ty.members)

    if isinstance(ty, ClassType):
        return any(occurs_check(var, arg) for arg in ty.type_args)

    if isinstance(ty, ProtocolType):
        return any(occurs_check(var, member_ty) for _, member_ty in ty.members)

    # Primitive types and other base types don't contain variables
    return False


@dataclass
class UnificationError:
    """Represents a failure in unification.

    Attributes:
        lhs: The left-hand side type that failed to unify
        rhs: The right-hand side type that failed to unify
        reason: Human-readable explanation of the failure
    """

    lhs: Type
    rhs: Type
    reason: str

    def __repr__(self) -> str:
        return f"UnificationError({self.lhs} ≠ {self.rhs}: {self.reason})"


@dataclass
class UnificationResult:
    """Result of a unification attempt.

    Either contains a successful substitution or an error.

    Attributes:
        substitution: The resulting substitution if successful
        error: The error if unification failed
    """

    substitution: Optional[Substitution] = None
    error: Optional[UnificationError] = None

    @property
    def is_success(self) -> bool:
        """True if unification succeeded."""
        return self.substitution is not None

    @property
    def is_failure(self) -> bool:
        """True if unification failed."""
        return self.error is not None

    @staticmethod
    def success(subst: Substitution) -> UnificationResult:
        """Create a successful result."""
        return UnificationResult(substitution=subst)

    @staticmethod
    def failure(lhs: Type, rhs: Type, reason: str) -> UnificationResult:
        """Create a failure result."""
        return UnificationResult(error=UnificationError(lhs, rhs, reason))

    def __repr__(self) -> str:
        if self.is_success:
            return f"UnificationResult.success({self.substitution})"
        return f"UnificationResult.failure({self.error})"


def is_subtype(sub: Type, sup: Type) -> bool:
    """Check if sub is a subtype of sup (sub <: sup).

    This implements subtype checking with proper variance handling.
    Uses a permissive approach - defaults to True on uncertainty.

    Args:
        sub: The potential subtype
        sup: The potential supertype

    Returns:
        True if sub is a subtype of sup
    """
    # Same type
    if sub == sup:
        return True

    # Any is both top and wildcard
    if isinstance(sup, AnyType) or isinstance(sub, AnyType):
        return True

    # Never is bottom type - subtype of everything
    if isinstance(sub, NeverType):
        return True

    # HoleType acts as wildcard
    if isinstance(sub, HoleType) or isinstance(sup, HoleType):
        return True

    # TypeVar - requires unification context, be permissive
    if isinstance(sub, TypeVar) or isinstance(sup, TypeVar):
        return True

    # int <: float (numeric promotion)
    from .constraint import INT, FLOAT
    if sub == INT and sup == FLOAT:
        return True

    # Union subtyping: sub <: Union if sub <: any member
    if isinstance(sup, UnionType):
        return any(is_subtype(sub, member) for member in sup.members)

    # Union subtyping: Union <: sup if all members <: sup
    if isinstance(sub, UnionType):
        return all(is_subtype(member, sup) for member in sub.members)

    # List covariance: List[T] <: List[U] if T <: U
    if isinstance(sub, ListType) and isinstance(sup, ListType):
        return is_subtype(sub.element, sup.element)

    # Set covariance (read-only)
    if isinstance(sub, SetType) and isinstance(sup, SetType):
        return is_subtype(sub.element, sup.element)

    # Dict covariance in value (when read-only)
    if isinstance(sub, DictType) and isinstance(sup, DictType):
        return is_subtype(sub.key, sup.key) and is_subtype(sub.value, sup.value)

    # Tuple covariance
    if isinstance(sub, TupleType) and isinstance(sup, TupleType):
        if len(sub.elements) != len(sup.elements):
            return False
        return all(is_subtype(s, u) for s, u in zip(sub.elements, sup.elements))

    # Function contravariant params, covariant return
    if isinstance(sub, FunctionType) and isinstance(sup, FunctionType):
        if len(sub.params) != len(sup.params):
            return False
        # Contravariant: sup_param <: sub_param
        params_ok = all(
            is_subtype(sup_p, sub_p)
            for sub_p, sup_p in zip(sub.params, sup.params)
        )
        # Covariant: sub_ret <: sup_ret
        return_ok = is_subtype(sub.returns, sup.returns)
        return params_ok and return_ok

    # Class nominal subtyping (same name)
    if isinstance(sub, ClassType) and isinstance(sup, ClassType):
        if sub.name != sup.name:
            return False
        if len(sub.type_args) != len(sup.type_args):
            return False
        # Default to invariant for generic args
        return all(a1 == a2 for a1, a2 in zip(sub.type_args, sup.type_args))

    # Protocol structural subtyping
    if isinstance(sup, ProtocolType):
        return _satisfies_protocol(sub, sup)

    # Default: not a subtype (conservative)
    return False


def _satisfies_protocol(ty: Type, protocol: ProtocolType) -> bool:
    """Check if a type satisfies a protocol's structural requirements.

    A type satisfies a protocol if it has all the required members
    with compatible types (structural subtyping / duck typing).

    Args:
        ty: The type to check
        protocol: The protocol to check against

    Returns:
        True if ty satisfies the protocol
    """
    # Any and Hole satisfy all protocols
    if isinstance(ty, (AnyType, HoleType)):
        return True

    # TypeVar - be permissive (would need bounds for precision)
    if isinstance(ty, TypeVar):
        return True

    # Protocols satisfy protocols if they have all required members
    if isinstance(ty, ProtocolType):
        ty_members = dict(ty.members)
        for name, expected_type in protocol.members:
            if name not in ty_members:
                return False
            # Check member type compatibility (covariant for methods/properties)
            if not is_subtype(ty_members[name], expected_type):
                return False
        return True

    # ClassType satisfies protocol if it has all members
    # (In practice, would need class member info; be permissive)
    if isinstance(ty, ClassType):
        # Without member info, be permissive - assume it might satisfy
        return True

    # FunctionType can satisfy single-method protocols (callable)
    if isinstance(ty, FunctionType):
        # Check for __call__ protocol
        call_type = protocol.get_member("__call__")
        if call_type is not None and len(protocol.members) == 1:
            return is_subtype(ty, call_type)

    # Default: doesn't satisfy
    return False


def unify_with_variance(
    t1: Type,
    t2: Type,
    variance: Variance = Variance.INVARIANT,
) -> UnificationResult:
    """Unify two types with variance handling.

    Variance affects whether we accept subtypes in certain positions:
    - COVARIANT: t1 <: t2 is acceptable (output position)
    - CONTRAVARIANT: t2 <: t1 is acceptable (input position)
    - INVARIANT: exact match required
    - BIVARIANT: either direction acceptable

    We still perform unification to resolve type variables, but variance
    determines which type mismatches are acceptable.

    Args:
        t1: First type (actual)
        t2: Second type (expected)
        variance: The variance of this position

    Returns:
        UnificationResult
    """
    # Always try unification first to resolve type variables
    result = unify(t1, t2)
    if result.is_success:
        return result

    # For invariant, require exact unification (already failed)
    if variance == Variance.INVARIANT:
        return result

    # For covariant, t1 must be subtype of t2
    if variance == Variance.COVARIANT:
        if is_subtype(t1, t2):
            return UnificationResult.success(EMPTY_SUBSTITUTION)
        return result

    # For contravariant, t2 must be subtype of t1
    if variance == Variance.CONTRAVARIANT:
        if is_subtype(t2, t1):
            return UnificationResult.success(EMPTY_SUBSTITUTION)
        return result

    # Bivariant - accept if either direction works
    if variance == Variance.BIVARIANT:
        if is_subtype(t1, t2) or is_subtype(t2, t1):
            return UnificationResult.success(EMPTY_SUBSTITUTION)
        return result

    return result


def unify(t1: Type, t2: Type) -> UnificationResult:
    """Unify two types, finding a substitution that makes them equal.

    This is Robinson's unification algorithm with occurs check,
    extended with variance handling for function types.

    Args:
        t1: First type
        t2: Second type

    Returns:
        UnificationResult containing either a substitution or an error
    """
    # Same type - trivially unifiable
    if t1 == t2:
        return UnificationResult.success(EMPTY_SUBSTITUTION)

    # AnyType unifies with anything
    if isinstance(t1, AnyType):
        return UnificationResult.success(EMPTY_SUBSTITUTION)
    if isinstance(t2, AnyType):
        return UnificationResult.success(EMPTY_SUBSTITUTION)

    # NeverType unifies with anything (bottom is subtype of all)
    if isinstance(t1, NeverType):
        return UnificationResult.success(EMPTY_SUBSTITUTION)
    if isinstance(t2, NeverType):
        return UnificationResult.success(EMPTY_SUBSTITUTION)

    # HoleType unifies with anything (holes are placeholders)
    if isinstance(t1, HoleType):
        return UnificationResult.success(EMPTY_SUBSTITUTION)
    if isinstance(t2, HoleType):
        return UnificationResult.success(EMPTY_SUBSTITUTION)

    # Type variable on left - bind it
    if isinstance(t1, TypeVar):
        if occurs_check(t1, t2):
            return UnificationResult.failure(
                t1, t2, f"infinite type: {t1.name} occurs in {t2}"
            )
        return UnificationResult.success(Substitution({t1.name: t2}))

    # Type variable on right - bind it
    if isinstance(t2, TypeVar):
        if occurs_check(t2, t1):
            return UnificationResult.failure(
                t1, t2, f"infinite type: {t2.name} occurs in {t1}"
            )
        return UnificationResult.success(Substitution({t2.name: t1}))

    # Function types - unify with proper variance
    if isinstance(t1, FunctionType) and isinstance(t2, FunctionType):
        if len(t1.params) != len(t2.params):
            return UnificationResult.failure(
                t1,
                t2,
                f"function arity mismatch: {len(t1.params)} vs {len(t2.params)}",
            )

        result_subst = EMPTY_SUBSTITUTION

        # Unify parameters with contravariance
        for p1, p2 in zip(t1.params, t2.params):
            p1_subst = result_subst.apply(p1)
            p2_subst = result_subst.apply(p2)
            # Contravariant: swap order for subtype check
            param_result = unify_with_variance(p2_subst, p1_subst, Variance.CONTRAVARIANT)
            if param_result.is_failure:
                # Fall back to invariant unification
                param_result = unify(p1_subst, p2_subst)
                if param_result.is_failure:
                    return param_result
            result_subst = param_result.substitution.compose(result_subst)

        # Unify return types with covariance
        ret1_subst = result_subst.apply(t1.returns)
        ret2_subst = result_subst.apply(t2.returns)
        ret_result = unify_with_variance(ret1_subst, ret2_subst, Variance.COVARIANT)
        if ret_result.is_failure:
            return ret_result

        return UnificationResult.success(ret_result.substitution.compose(result_subst))

    # List types - unify elements
    if isinstance(t1, ListType) and isinstance(t2, ListType):
        return unify(t1.element, t2.element)

    # Set types - unify elements
    if isinstance(t1, SetType) and isinstance(t2, SetType):
        return unify(t1.element, t2.element)

    # Dict types - unify key and value
    if isinstance(t1, DictType) and isinstance(t2, DictType):
        key_result = unify(t1.key, t2.key)
        if key_result.is_failure:
            return key_result
        value1 = key_result.substitution.apply(t1.value)
        value2 = key_result.substitution.apply(t2.value)
        value_result = unify(value1, value2)
        if value_result.is_failure:
            return value_result
        return UnificationResult.success(
            value_result.substitution.compose(key_result.substitution)
        )

    # Tuple types - unify element-wise
    if isinstance(t1, TupleType) and isinstance(t2, TupleType):
        if len(t1.elements) != len(t2.elements):
            return UnificationResult.failure(
                t1,
                t2,
                f"tuple length mismatch: {len(t1.elements)} vs {len(t2.elements)}",
            )

        result_subst = EMPTY_SUBSTITUTION
        for e1, e2 in zip(t1.elements, t2.elements):
            e1_subst = result_subst.apply(e1)
            e2_subst = result_subst.apply(e2)
            elem_result = unify(e1_subst, e2_subst)
            if elem_result.is_failure:
                return elem_result
            result_subst = elem_result.substitution.compose(result_subst)

        return UnificationResult.success(result_subst)

    # Class types - must have same name, unify type args
    if isinstance(t1, ClassType) and isinstance(t2, ClassType):
        if t1.name != t2.name:
            return UnificationResult.failure(
                t1, t2, f"class name mismatch: {t1.name} vs {t2.name}"
            )
        if len(t1.type_args) != len(t2.type_args):
            return UnificationResult.failure(
                t1,
                t2,
                f"type argument count mismatch: {len(t1.type_args)} vs {len(t2.type_args)}",
            )

        result_subst = EMPTY_SUBSTITUTION
        for a1, a2 in zip(t1.type_args, t2.type_args):
            a1_subst = result_subst.apply(a1)
            a2_subst = result_subst.apply(a2)
            arg_result = unify(a1_subst, a2_subst)
            if arg_result.is_failure:
                return arg_result
            result_subst = arg_result.substitution.compose(result_subst)

        return UnificationResult.success(result_subst)

    # Protocol types - structural subtyping
    if isinstance(t1, ProtocolType) and isinstance(t2, ProtocolType):
        # Both protocols: unify member-by-member
        t1_members = dict(t1.members)
        t2_members = dict(t2.members)

        result_subst = EMPTY_SUBSTITUTION

        # Check all members in t2 exist in t1 with compatible types
        for name, ty2 in t2_members.items():
            if name not in t1_members:
                return UnificationResult.failure(
                    t1, t2, f"protocol missing member: {name}"
                )
            ty1 = t1_members[name]
            ty1_subst = result_subst.apply(ty1)
            ty2_subst = result_subst.apply(ty2)
            member_result = unify(ty1_subst, ty2_subst)
            if member_result.is_failure:
                return member_result
            result_subst = member_result.substitution.compose(result_subst)

        return UnificationResult.success(result_subst)

    # Protocol against other types - use structural subtype check
    if isinstance(t2, ProtocolType):
        if _satisfies_protocol(t1, t2):
            return UnificationResult.success(EMPTY_SUBSTITUTION)
        return UnificationResult.failure(
            t1, t2, f"type does not satisfy protocol {t2.name}"
        )

    if isinstance(t1, ProtocolType):
        if _satisfies_protocol(t2, t1):
            return UnificationResult.success(EMPTY_SUBSTITUTION)
        return UnificationResult.failure(
            t1, t2, f"type does not satisfy protocol {t1.name}"
        )

    # Union types - more complex, require structural matching
    # For now, unions only unify if they have the same members
    if isinstance(t1, UnionType) and isinstance(t2, UnionType):
        if t1.members == t2.members:
            return UnificationResult.success(EMPTY_SUBSTITUTION)
        return UnificationResult.failure(
            t1, t2, "union types have different members"
        )

    # Type mismatch - cannot unify
    return UnificationResult.failure(
        t1, t2, f"incompatible types: {type(t1).__name__} vs {type(t2).__name__}"
    )


def solve_equations(
    equations: List[TypeEquation],
) -> UnificationResult:
    """Solve a system of type equations.

    Uses a worklist algorithm to solve all equations, composing
    substitutions as we go.

    Args:
        equations: List of type equations to solve

    Returns:
        UnificationResult containing the combined substitution or an error
    """
    worklist = list(equations)
    result_subst = EMPTY_SUBSTITUTION

    while worklist:
        eq = worklist.pop()

        # Apply current substitution to the equation
        lhs = result_subst.apply(eq.lhs)
        rhs = result_subst.apply(eq.rhs)

        # Unify the types
        unify_result = unify(lhs, rhs)
        if unify_result.is_failure:
            return unify_result

        # Compose the new substitution with the accumulated result
        result_subst = unify_result.substitution.compose(result_subst)

    return UnificationResult.success(result_subst)


def unify_types(types: List[Type]) -> UnificationResult:
    """Unify a list of types to find their common type.

    All types in the list must be unifiable with each other.

    Args:
        types: List of types to unify

    Returns:
        UnificationResult with substitution that makes all types equal
    """
    if not types:
        return UnificationResult.success(EMPTY_SUBSTITUTION)
    if len(types) == 1:
        return UnificationResult.success(EMPTY_SUBSTITUTION)

    result_subst = EMPTY_SUBSTITUTION
    first = types[0]

    for other in types[1:]:
        first_subst = result_subst.apply(first)
        other_subst = result_subst.apply(other)
        unify_result = unify(first_subst, other_subst)
        if unify_result.is_failure:
            return unify_result
        result_subst = unify_result.substitution.compose(result_subst)
        first = result_subst.apply(first)

    return UnificationResult.success(result_subst)
