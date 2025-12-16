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
"""Type constraint representing type system requirements.

This module defines the type hierarchy and TypeConstraint for Ananke's type domain.
The type system is based on Hazel research (POPL 2024, OOPSLA 2025) with:
- Full type representation including holes and marks
- TypeConstraint as a bounded meet-semilattice
- Support for incremental bidirectional type checking

References:
    - POPL 2024: "Total Type Error Localization and Recovery with Holes"
    - OOPSLA 2025: "Incremental Bidirectional Typing via Order Maintenance"
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple, Union


# =============================================================================
# Variance
# =============================================================================


class Variance(Enum):
    """Variance of a type parameter.

    Variance describes how subtyping of a generic type relates to
    subtyping of its type arguments:

    - COVARIANT (+): List[Cat] <: List[Animal] if Cat <: Animal
    - CONTRAVARIANT (-): Consumer[Animal] <: Consumer[Cat] if Cat <: Animal
    - INVARIANT (=): Container[Cat] is unrelated to Container[Animal]
    - BIVARIANT (±): Accepts both (rare, usually indicates design issue)

    Examples:
        - Return types are covariant: fn() -> Cat <: fn() -> Animal
        - Parameter types are contravariant: fn(Animal) <: fn(Cat)
        - Mutable containers are invariant: List[Cat] NOT <: List[Animal]
        - Read-only sequences are covariant: Sequence[Cat] <: Sequence[Animal]
    """

    COVARIANT = auto()      # Read-only, output position
    CONTRAVARIANT = auto()  # Write-only, input position
    INVARIANT = auto()      # Read-write, exact match
    BIVARIANT = auto()      # Both (unsafe, for compatibility)

# Support both relative imports (when used as subpackage) and absolute imports (standalone testing)
try:
    from ...core.constraint import Constraint, Satisfiability
except ImportError:
    from core.constraint import Constraint, Satisfiability


# =============================================================================
# Type Representation Hierarchy
# =============================================================================


class Type(ABC):
    """Abstract base class for all types.

    Types form a hierarchy supporting:
    - Primitive types (int, str, bool, float, None)
    - Compound types (function, list, dict, tuple, union)
    - Type variables (for unification)
    - Special types (Any, Never, Hole)

    Note: Subclasses using @dataclass(frozen=True) get __eq__ and __hash__
    automatically. The free_type_vars() and substitute() methods have default
    implementations that work for simple types without type variables. Override
    them in subclasses that contain nested types or type variables.
    """

    def free_type_vars(self) -> FrozenSet[str]:
        """Return the set of free type variable names in this type.

        Default implementation returns empty set. Override in subclasses
        that contain type variables or nested types.
        """
        return frozenset()

    def substitute(self, substitution: Dict[str, "Type"]) -> "Type":
        """Apply a substitution to this type.

        Default implementation returns self unchanged. Override in subclasses
        that contain type variables or nested types.
        """
        return self

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        pass

    @abstractmethod
    def __hash__(self) -> int:
        pass


@dataclass(frozen=True, slots=True)
class TypeVar(Type):
    """A type variable (unification variable).

    Type variables are placeholders that can be unified with concrete types.
    They have a unique name and optionally track their origin.

    Attributes:
        name: Unique identifier for this type variable
        origin: Optional description of where this variable originated
    """

    name: str
    origin: Optional[str] = None

    def free_type_vars(self) -> FrozenSet[str]:
        return frozenset({self.name})

    def substitute(self, substitution: Dict[str, Type]) -> Type:
        if self.name in substitution:
            return substitution[self.name]
        return self

    def __repr__(self) -> str:
        return f"'{self.name}"


@dataclass(frozen=True, slots=True)
class PrimitiveType(Type):
    """A primitive/base type.

    Attributes:
        name: The type name (e.g., "int", "str", "bool", "float", "None")
    """

    name: str

    def free_type_vars(self) -> FrozenSet[str]:
        return frozenset()

    def substitute(self, substitution: Dict[str, Type]) -> Type:
        return self

    def __repr__(self) -> str:
        return self.name


# Primitive type singletons
INT = PrimitiveType("int")
STR = PrimitiveType("str")
BOOL = PrimitiveType("bool")
FLOAT = PrimitiveType("float")
NONE = PrimitiveType("None")
BYTES = PrimitiveType("bytes")


@dataclass(frozen=True, slots=True)
class FunctionType(Type):
    """A function type with parameter types and return type.

    Supports both positional and keyword parameters.

    Attributes:
        params: Tuple of parameter types (positional)
        returns: The return type
        param_names: Optional names for keyword parameters
    """

    params: Tuple[Type, ...]
    returns: Type
    param_names: Optional[Tuple[str, ...]] = None

    def free_type_vars(self) -> FrozenSet[str]:
        vars_set: FrozenSet[str] = frozenset()
        for p in self.params:
            vars_set = vars_set | p.free_type_vars()
        return vars_set | self.returns.free_type_vars()

    def substitute(self, substitution: Dict[str, Type]) -> Type:
        new_params = tuple(p.substitute(substitution) for p in self.params)
        new_returns = self.returns.substitute(substitution)
        return FunctionType(new_params, new_returns, self.param_names)

    def __repr__(self) -> str:
        if not self.params:
            return f"() -> {self.returns}"
        params_str = ", ".join(repr(p) for p in self.params)
        return f"({params_str}) -> {self.returns}"


@dataclass(frozen=True, slots=True)
class ListType(Type):
    """A homogeneous list type.

    Attributes:
        element: The type of list elements
    """

    element: Type

    def free_type_vars(self) -> FrozenSet[str]:
        return self.element.free_type_vars()

    def substitute(self, substitution: Dict[str, Type]) -> Type:
        return ListType(self.element.substitute(substitution))

    def __repr__(self) -> str:
        return f"List[{self.element}]"


@dataclass(frozen=True, slots=True)
class DictType(Type):
    """A dictionary type with key and value types.

    Attributes:
        key: The type of dictionary keys
        value: The type of dictionary values
    """

    key: Type
    value: Type

    def free_type_vars(self) -> FrozenSet[str]:
        return self.key.free_type_vars() | self.value.free_type_vars()

    def substitute(self, substitution: Dict[str, Type]) -> Type:
        return DictType(
            self.key.substitute(substitution), self.value.substitute(substitution)
        )

    def __repr__(self) -> str:
        return f"Dict[{self.key}, {self.value}]"


@dataclass(frozen=True, slots=True)
class TupleType(Type):
    """A tuple type with fixed element types.

    Attributes:
        elements: The types of each tuple element
    """

    elements: Tuple[Type, ...]

    def free_type_vars(self) -> FrozenSet[str]:
        vars_set: FrozenSet[str] = frozenset()
        for e in self.elements:
            vars_set = vars_set | e.free_type_vars()
        return vars_set

    def substitute(self, substitution: Dict[str, Type]) -> Type:
        return TupleType(tuple(e.substitute(substitution) for e in self.elements))

    def __repr__(self) -> str:
        if not self.elements:
            return "Tuple[()]"
        elems_str = ", ".join(repr(e) for e in self.elements)
        return f"Tuple[{elems_str}]"


@dataclass(frozen=True, slots=True)
class SetType(Type):
    """A homogeneous set type.

    Attributes:
        element: The type of set elements
    """

    element: Type

    def free_type_vars(self) -> FrozenSet[str]:
        return self.element.free_type_vars()

    def substitute(self, substitution: Dict[str, Type]) -> Type:
        return SetType(self.element.substitute(substitution))

    def __repr__(self) -> str:
        return f"Set[{self.element}]"


@dataclass(frozen=True)
class UnionType(Type):
    """A union of types (T1 | T2 | ...).

    The members are stored as a frozenset to ensure canonical representation
    and proper equality semantics.

    Attributes:
        members: The set of types in the union
    """

    members: FrozenSet[Type]

    def free_type_vars(self) -> FrozenSet[str]:
        vars_set: FrozenSet[str] = frozenset()
        for m in self.members:
            vars_set = vars_set | m.free_type_vars()
        return vars_set

    def substitute(self, substitution: Dict[str, Type]) -> Type:
        return UnionType(
            frozenset(m.substitute(substitution) for m in self.members)
        )

    def __repr__(self) -> str:
        # Sort for consistent representation
        sorted_members = sorted(self.members, key=repr)
        return " | ".join(repr(m) for m in sorted_members)

    def __hash__(self) -> int:
        return hash(self.members)


def OptionalType(inner: Type) -> UnionType:
    """Create an optional type (T | None).

    This is a convenience function that creates a union with None.
    """
    return UnionType(frozenset({inner, NONE}))


@dataclass(frozen=True, slots=True)
class ClassType(Type):
    """A nominal class/object type.

    Supports generic classes with type parameters.

    Attributes:
        name: The class name (fully qualified)
        type_args: Type arguments for generic classes
    """

    name: str
    type_args: Tuple[Type, ...] = ()

    def free_type_vars(self) -> FrozenSet[str]:
        vars_set: FrozenSet[str] = frozenset()
        for arg in self.type_args:
            vars_set = vars_set | arg.free_type_vars()
        return vars_set

    def substitute(self, substitution: Dict[str, Type]) -> Type:
        if not self.type_args:
            return self
        new_args = tuple(arg.substitute(substitution) for arg in self.type_args)
        return ClassType(self.name, new_args)

    def __repr__(self) -> str:
        if not self.type_args:
            return self.name
        args_str = ", ".join(repr(a) for a in self.type_args)
        return f"{self.name}[{args_str}]"


@dataclass(frozen=True)
class ProtocolType(Type):
    """A structural protocol/interface type for duck typing.

    Unlike ClassType which uses nominal subtyping (requires explicit inheritance),
    ProtocolType uses structural subtyping: any type with the required members
    is considered a subtype.

    Used for:
    - Python's Protocol (typing.Protocol)
    - TypeScript interfaces
    - Go interfaces
    - Rust traits (when used as bounds)

    Attributes:
        name: The protocol name (for error messages)
        members: Dict of member_name -> member_type (methods, properties)
        type_params: Type parameters with their variances
    """

    name: str
    members: FrozenSet[Tuple[str, Type]]  # FrozenSet for hashability
    type_params: Tuple[Tuple[str, Variance], ...] = ()

    def free_type_vars(self) -> FrozenSet[str]:
        vars_set: FrozenSet[str] = frozenset()
        for _, member_type in self.members:
            vars_set = vars_set | member_type.free_type_vars()
        return vars_set

    def substitute(self, substitution: Dict[str, Type]) -> Type:
        new_members = frozenset(
            (name, ty.substitute(substitution)) for name, ty in self.members
        )
        return ProtocolType(self.name, new_members, self.type_params)

    def get_member(self, name: str) -> Optional[Type]:
        """Look up a member by name."""
        for member_name, member_type in self.members:
            if member_name == name:
                return member_type
        return None

    def __repr__(self) -> str:
        if not self.members:
            return f"Protocol[{self.name}]"
        members_str = ", ".join(f"{n}: {t}" for n, t in sorted(self.members))
        return f"Protocol[{self.name}]{{{members_str}}}"

    def __hash__(self) -> int:
        return hash((self.name, self.members, self.type_params))


@dataclass(frozen=True, slots=True)
class AnyType(Type):
    """The top type (⊤) - any value has this type.

    AnyType is compatible with all other types. It represents
    unconstrained or unknown types.
    """

    def free_type_vars(self) -> FrozenSet[str]:
        return frozenset()

    def substitute(self, substitution: Dict[str, Type]) -> Type:
        return self

    def __repr__(self) -> str:
        return "Any"


# Singleton instance
ANY = AnyType()


@dataclass(frozen=True, slots=True)
class NeverType(Type):
    """The bottom type (⊥) - no value has this type.

    NeverType represents impossible types, such as the return type
    of a function that never returns.
    """

    def free_type_vars(self) -> FrozenSet[str]:
        return frozenset()

    def substitute(self, substitution: Dict[str, Type]) -> Type:
        return self

    def __repr__(self) -> str:
        return "Never"


# Singleton instance
NEVER = NeverType()


@dataclass(frozen=True, slots=True)
class HoleType(Type):
    """A typed hole placeholder (from Hazel).

    Represents an unknown part of a type that will be filled in later.
    Holes can unify with any type, enabling partial type inference.

    Attributes:
        hole_id: Unique identifier for this hole
        expected: Optional expected type (from context)
    """

    hole_id: str
    expected: Optional[Type] = None

    def free_type_vars(self) -> FrozenSet[str]:
        if self.expected:
            return self.expected.free_type_vars()
        return frozenset()

    def substitute(self, substitution: Dict[str, Type]) -> Type:
        if self.expected:
            return HoleType(self.hole_id, self.expected.substitute(substitution))
        return self

    def __repr__(self) -> str:
        if self.expected:
            return f"?{self.hole_id}:{self.expected}"
        return f"?{self.hole_id}"


# =============================================================================
# Type Equations for Unification
# =============================================================================


@dataclass(frozen=True, slots=True)
class TypeEquation:
    """An equation between two types for unification.

    Represents the constraint that lhs and rhs must be equal types.

    Attributes:
        lhs: Left-hand side type
        rhs: Right-hand side type
        origin: Description of where this equation came from
    """

    lhs: Type
    rhs: Type
    origin: Optional[str] = None

    def __repr__(self) -> str:
        return f"{self.lhs} = {self.rhs}"


# =============================================================================
# Type Constraint
# =============================================================================


@dataclass(frozen=True, slots=True)
class TypeConstraint(Constraint["TypeConstraint"]):
    """Constraint representing type system requirements.

    TypeConstraint tracks:
    - expected_type: The type expected at the current position
    - equations: Type equations that must be satisfied
    - environment_hash: Hash of the type environment (for caching)
    - has_errors: Whether type errors have been detected

    The constraint forms a semilattice where:
    - TOP (TYPE_TOP) represents no type constraint
    - BOTTOM (TYPE_BOTTOM) represents unsatisfiable type
    - meet() combines type requirements

    Attributes:
        expected_type: The expected type at current position (None for TOP)
        equations: Frozen set of type equations
        environment_hash: Hash of the current type environment
        has_errors: True if type errors have been detected
        _is_top: True if this is the TOP element
        _is_bottom: True if this is the BOTTOM element
    """

    expected_type: Optional[Type] = None
    equations: FrozenSet[TypeEquation] = field(default_factory=frozenset)
    environment_hash: int = 0
    has_errors: bool = False
    _is_top: bool = False
    _is_bottom: bool = False

    def meet(self, other: TypeConstraint) -> TypeConstraint:
        """Compute the meet of two type constraints.

        Meet combines type requirements:
        - If either is BOTTOM, return BOTTOM
        - If either is TOP, return the other
        - Otherwise, combine expected types and equations

        Args:
            other: The constraint to combine with

        Returns:
            The conjunction of both constraints
        """
        # Handle BOTTOM (annihilation)
        if self._is_bottom or other._is_bottom:
            return TYPE_BOTTOM

        # Handle TOP (identity)
        if self._is_top:
            return other
        if other._is_top:
            return self

        # Combine constraints
        # For expected_type: if both have one, they must be compatible
        combined_expected: Optional[Type] = None
        combined_errors = self.has_errors or other.has_errors

        if self.expected_type is None:
            combined_expected = other.expected_type
        elif other.expected_type is None:
            combined_expected = self.expected_type
        elif self.expected_type == other.expected_type:
            combined_expected = self.expected_type
        else:
            # Different expected types - check for Any
            if isinstance(self.expected_type, AnyType):
                combined_expected = other.expected_type
            elif isinstance(other.expected_type, AnyType):
                combined_expected = self.expected_type
            else:
                # Incompatible types - this is unsatisfiable
                # Return BOTTOM for commutativity (same result regardless of order)
                return TYPE_BOTTOM

        # Combine equations
        combined_equations = self.equations | other.equations

        # Combine environment hash (take max for commutativity)
        combined_env_hash = max(self.environment_hash, other.environment_hash)

        return TypeConstraint(
            expected_type=combined_expected,
            equations=combined_equations,
            environment_hash=combined_env_hash,
            has_errors=combined_errors,
        )

    def is_top(self) -> bool:
        """Check if this is the TOP (unconstrained) element."""
        return self._is_top

    def is_bottom(self) -> bool:
        """Check if this is the BOTTOM (unsatisfiable) element."""
        return self._is_bottom

    def satisfiability(self) -> Satisfiability:
        """Determine the satisfiability status.

        Returns:
            SAT if the constraint can be satisfied
            UNSAT if the constraint cannot be satisfied
        """
        if self._is_bottom:
            return Satisfiability.UNSAT
        if self.has_errors:
            # Has errors but might still be satisfiable in some sense
            return Satisfiability.SAT
        return Satisfiability.SAT

    def with_expected_type(self, expected: Type) -> TypeConstraint:
        """Create a new constraint with updated expected type."""
        return TypeConstraint(
            expected_type=expected,
            equations=self.equations,
            environment_hash=self.environment_hash,
            has_errors=self.has_errors,
            _is_top=False,
            _is_bottom=False,
        )

    def with_equation(self, equation: TypeEquation) -> TypeConstraint:
        """Create a new constraint with an additional equation."""
        return TypeConstraint(
            expected_type=self.expected_type,
            equations=self.equations | {equation},
            environment_hash=self.environment_hash,
            has_errors=self.has_errors,
            _is_top=False,
            _is_bottom=False,
        )

    def with_environment_hash(self, env_hash: int) -> TypeConstraint:
        """Create a new constraint with updated environment hash."""
        return TypeConstraint(
            expected_type=self.expected_type,
            equations=self.equations,
            environment_hash=env_hash,
            has_errors=self.has_errors,
            _is_top=self._is_top,
            _is_bottom=self._is_bottom,
        )

    def with_error(self) -> TypeConstraint:
        """Create a new constraint marked as having errors."""
        return TypeConstraint(
            expected_type=self.expected_type,
            equations=self.equations,
            environment_hash=self.environment_hash,
            has_errors=True,
            _is_top=False,
            _is_bottom=False,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TypeConstraint):
            return NotImplemented
        return (
            self._is_top == other._is_top
            and self._is_bottom == other._is_bottom
            and self.expected_type == other.expected_type
            and self.equations == other.equations
            and self.environment_hash == other.environment_hash
            and self.has_errors == other.has_errors
        )

    def __hash__(self) -> int:
        if self._is_top:
            return hash("TYPE_TOP")
        if self._is_bottom:
            return hash("TYPE_BOTTOM")
        return hash(
            (
                self.expected_type,
                self.equations,
                self.environment_hash,
                self.has_errors,
            )
        )

    def __repr__(self) -> str:
        if self._is_top:
            return "TYPE_TOP"
        if self._is_bottom:
            return "TYPE_BOTTOM"
        parts = []
        if self.expected_type:
            parts.append(f"expected={self.expected_type}")
        if self.equations:
            parts.append(f"equations={len(self.equations)}")
        if self.has_errors:
            parts.append("has_errors")
        return f"TypeConstraint({', '.join(parts)})"


# Singleton instances
TYPE_TOP = TypeConstraint(_is_top=True)
TYPE_BOTTOM = TypeConstraint(_is_bottom=True)


# =============================================================================
# Factory Functions
# =============================================================================


def type_expecting(expected: Type) -> TypeConstraint:
    """Create a type constraint expecting a specific type."""
    return TypeConstraint(expected_type=expected)


def type_with_equation(lhs: Type, rhs: Type, origin: Optional[str] = None) -> TypeConstraint:
    """Create a type constraint with a single equation."""
    return TypeConstraint(equations=frozenset({TypeEquation(lhs, rhs, origin)}))
