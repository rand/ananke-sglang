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
"""Subtype checking (subsumption) for bidirectional typing.

Subsumption checks whether one type is a subtype of another (A <: B).
This is used in analysis mode to check if a synthesized type matches
the expected type from context.

The subsumption relation handles:
- Top type (Any) is a supertype of everything
- Bottom type (Never) is a subtype of everything
- Holes satisfy any constraint (for partial programs)
- Structural subtyping for compound types
- Variance for function types (contravariant params, covariant returns)

References:
    - Pierce (2002). "Types and Programming Languages", Chapter 15
    - Dunfield & Krishnaswami (2019). "Bidirectional Typing"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

# Support both relative imports (when used as subpackage) and absolute imports (standalone testing)
try:
    from ..constraint import (
        ANY,
        NEVER,
        AnyType,
        ClassType,
        DictType,
        FunctionType,
        HoleType,
        ListType,
        NeverType,
        SetType,
        TupleType,
        Type,
        TypeVar,
        UnionType,
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
        SetType,
        TupleType,
        Type,
        TypeVar,
        UnionType,
    )


@dataclass
class SubsumptionResult:
    """Result of a subsumption check.

    Attributes:
        success: True if the subtype relation holds
        reason: Explanation if the check failed
    """

    success: bool
    reason: Optional[str] = None

    @staticmethod
    def ok() -> SubsumptionResult:
        """Create a successful result."""
        return SubsumptionResult(success=True)

    @staticmethod
    def fail(reason: str) -> SubsumptionResult:
        """Create a failure result."""
        return SubsumptionResult(success=False, reason=reason)


def subsumes(sub: Type, super_: Type) -> bool:
    """Check if sub is a subtype of super_ (sub <: super_).

    This is the main entry point for subtype checking.

    Args:
        sub: The potential subtype
        super_: The potential supertype

    Returns:
        True if sub <: super_
    """
    return check_subsumption(sub, super_).success


def check_subsumption(sub: Type, super_: Type) -> SubsumptionResult:
    """Check subsumption with detailed result.

    Args:
        sub: The potential subtype
        super_: The potential supertype

    Returns:
        SubsumptionResult with success status and reason
    """
    # Reflexivity: A <: A
    if sub == super_:
        return SubsumptionResult.ok()

    # Top: A <: Any for all A
    if isinstance(super_, AnyType):
        return SubsumptionResult.ok()

    # Bottom: Never <: A for all A
    if isinstance(sub, NeverType):
        return SubsumptionResult.ok()

    # Holes: Holes satisfy any constraint (for partial programs)
    if isinstance(sub, HoleType) or isinstance(super_, HoleType):
        return SubsumptionResult.ok()

    # Type variables: check if same variable
    if isinstance(sub, TypeVar) and isinstance(super_, TypeVar):
        if sub.name == super_.name:
            return SubsumptionResult.ok()
        return SubsumptionResult.fail(
            f"type variable '{sub.name}' is not '{super_.name}'"
        )

    # TypeVar on one side only - fails (unless already handled by hole/any)
    if isinstance(sub, TypeVar):
        return SubsumptionResult.fail(
            f"type variable '{sub.name}' does not subsume {super_}"
        )
    if isinstance(super_, TypeVar):
        return SubsumptionResult.fail(
            f"{sub} does not subsume type variable '{super_.name}'"
        )

    # Function types: contravariant in params, covariant in return
    if isinstance(sub, FunctionType) and isinstance(super_, FunctionType):
        return _check_function_subsumption(sub, super_)

    # List types: covariant (for simplicity, should be invariant for mutability)
    if isinstance(sub, ListType) and isinstance(super_, ListType):
        result = check_subsumption(sub.element, super_.element)
        if not result.success:
            return SubsumptionResult.fail(
                f"List[{sub.element}] is not a subtype of List[{super_.element}]: {result.reason}"
            )
        return SubsumptionResult.ok()

    # Set types: covariant (same caveat as list)
    if isinstance(sub, SetType) and isinstance(super_, SetType):
        result = check_subsumption(sub.element, super_.element)
        if not result.success:
            return SubsumptionResult.fail(
                f"Set[{sub.element}] is not a subtype of Set[{super_.element}]: {result.reason}"
            )
        return SubsumptionResult.ok()

    # Dict types: invariant in key, covariant in value (simplification)
    if isinstance(sub, DictType) and isinstance(super_, DictType):
        key_result = check_subsumption(sub.key, super_.key)
        if not key_result.success:
            return SubsumptionResult.fail(
                f"Dict key type mismatch: {key_result.reason}"
            )
        value_result = check_subsumption(sub.value, super_.value)
        if not value_result.success:
            return SubsumptionResult.fail(
                f"Dict value type mismatch: {value_result.reason}"
            )
        return SubsumptionResult.ok()

    # Tuple types: element-wise covariance
    if isinstance(sub, TupleType) and isinstance(super_, TupleType):
        if len(sub.elements) != len(super_.elements):
            return SubsumptionResult.fail(
                f"Tuple length mismatch: {len(sub.elements)} vs {len(super_.elements)}"
            )
        for i, (sub_elem, super_elem) in enumerate(
            zip(sub.elements, super_.elements)
        ):
            result = check_subsumption(sub_elem, super_elem)
            if not result.success:
                return SubsumptionResult.fail(
                    f"Tuple element {i} type mismatch: {result.reason}"
                )
        return SubsumptionResult.ok()

    # Union types: sub <: (A | B) if sub <: A or sub <: B
    if isinstance(super_, UnionType):
        for member in super_.members:
            if check_subsumption(sub, member).success:
                return SubsumptionResult.ok()
        return SubsumptionResult.fail(
            f"{sub} is not a subtype of any member of {super_}"
        )

    # Union types: (A | B) <: super if A <: super and B <: super
    if isinstance(sub, UnionType):
        for member in sub.members:
            result = check_subsumption(member, super_)
            if not result.success:
                return SubsumptionResult.fail(
                    f"Union member {member} is not a subtype of {super_}: {result.reason}"
                )
        return SubsumptionResult.ok()

    # Class types: same name, check type arguments
    if isinstance(sub, ClassType) and isinstance(super_, ClassType):
        if sub.name != super_.name:
            return SubsumptionResult.fail(
                f"Class '{sub.name}' is not '{super_.name}'"
            )
        if len(sub.type_args) != len(super_.type_args):
            return SubsumptionResult.fail(
                f"Type argument count mismatch: {len(sub.type_args)} vs {len(super_.type_args)}"
            )
        # Invariant type arguments (simplification)
        for i, (sub_arg, super_arg) in enumerate(
            zip(sub.type_args, super_.type_args)
        ):
            if sub_arg != super_arg:
                return SubsumptionResult.fail(
                    f"Type argument {i} mismatch: {sub_arg} vs {super_arg}"
                )
        return SubsumptionResult.ok()

    # Different type constructors - no subsumption
    return SubsumptionResult.fail(
        f"{type(sub).__name__} is not a subtype of {type(super_).__name__}"
    )


def _check_function_subsumption(
    sub: FunctionType, super_: FunctionType
) -> SubsumptionResult:
    """Check function type subsumption.

    Functions are contravariant in parameter types and covariant in return type.
    (A -> B) <: (C -> D) if C <: A and B <: D

    Args:
        sub: The subtype function
        super_: The supertype function

    Returns:
        SubsumptionResult
    """
    # Check arity
    if len(sub.params) != len(super_.params):
        return SubsumptionResult.fail(
            f"Function arity mismatch: {len(sub.params)} vs {len(super_.params)}"
        )

    # Contravariant parameters: super_param <: sub_param
    for i, (sub_param, super_param) in enumerate(
        zip(sub.params, super_.params)
    ):
        result = check_subsumption(super_param, sub_param)
        if not result.success:
            return SubsumptionResult.fail(
                f"Parameter {i} contravariance violation: {result.reason}"
            )

    # Covariant return: sub_return <: super_return
    result = check_subsumption(sub.returns, super_.returns)
    if not result.success:
        return SubsumptionResult.fail(
            f"Return type covariance violation: {result.reason}"
        )

    return SubsumptionResult.ok()


def is_assignable(source: Type, target: Type) -> bool:
    """Check if a value of source type can be assigned to a target type.

    This is typically the same as subsumption: source <: target.

    Args:
        source: The type of the value being assigned
        target: The type of the assignment target

    Returns:
        True if assignment is valid
    """
    return subsumes(source, target)


def join(t1: Type, t2: Type) -> Type:
    """Compute the join (least upper bound) of two types.

    The join is the smallest type that both t1 and t2 are subtypes of.

    Args:
        t1: First type
        t2: Second type

    Returns:
        The join type
    """
    # Same type
    if t1 == t2:
        return t1

    # Any absorbs everything
    if isinstance(t1, AnyType) or isinstance(t2, AnyType):
        return ANY

    # Never is identity for join
    if isinstance(t1, NeverType):
        return t2
    if isinstance(t2, NeverType):
        return t1

    # Holes join to Any (conservative)
    if isinstance(t1, HoleType) or isinstance(t2, HoleType):
        return ANY

    # If one subsumes the other, return the supertype
    if subsumes(t1, t2):
        return t2
    if subsumes(t2, t1):
        return t1

    # Otherwise, create a union
    return UnionType(frozenset({t1, t2}))


def meet(t1: Type, t2: Type) -> Type:
    """Compute the meet (greatest lower bound) of two types.

    The meet is the largest type that is a subtype of both t1 and t2.

    Args:
        t1: First type
        t2: Second type

    Returns:
        The meet type (may be Never if no common subtype)
    """
    # Same type
    if t1 == t2:
        return t1

    # Any is identity for meet
    if isinstance(t1, AnyType):
        return t2
    if isinstance(t2, AnyType):
        return t1

    # Never absorbs everything in meet
    if isinstance(t1, NeverType) or isinstance(t2, NeverType):
        return NEVER

    # If one subsumes the other, return the subtype
    if subsumes(t1, t2):
        return t1
    if subsumes(t2, t1):
        return t2

    # No common subtype
    return NEVER
