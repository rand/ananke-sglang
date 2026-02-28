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
"""Extended type representations for systems-level languages.

This module provides shared type abstractions used by Zig, Rust, and other
systems programming languages that need:

- Pointer types with mutability and const qualifiers
- Slice types (views into contiguous memory)
- Fixed-length array types
- Result/error union types
- Reference types with lifetime annotations

These types extend the base type hierarchy in constraint.py to support
languages with explicit memory management and ownership semantics.

References:
    - Zig Language Reference: https://ziglang.org/documentation/
    - Rust Reference: https://doc.rust-lang.org/reference/
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from domains.types.constraint import Type as TypeBase
else:
    from domains.types.constraint import Type as TypeBase

# Type alias for clarity
Type = TypeBase


@dataclass(frozen=True, slots=True)
class PointerType(Type):
    """Pointer to a type.

    Represents:
    - Zig: *T, *const T, *volatile T
    - Rust: *const T, *mut T (raw pointers)

    For Rust references (&T, &mut T), use ReferenceType instead.

    Attributes:
        pointee: The type being pointed to
        is_const: Whether the pointee is const (cannot be modified through pointer)
        is_mutable: Whether the pointer itself can be reassigned
        is_volatile: Whether accesses through this pointer are volatile
    """

    pointee: Type
    is_const: bool = False
    is_mutable: bool = False
    is_volatile: bool = False

    def free_type_vars(self) -> FrozenSet[str]:
        return self.pointee.free_type_vars()

    def substitute(self, substitution: Dict[str, Type]) -> "PointerType":
        return PointerType(
            pointee=self.pointee.substitute(substitution),
            is_const=self.is_const,
            is_mutable=self.is_mutable,
            is_volatile=self.is_volatile,
        )

    def __str__(self) -> str:
        qualifiers = []
        if self.is_const:
            qualifiers.append("const")
        if self.is_volatile:
            qualifiers.append("volatile")
        qual_str = " ".join(qualifiers)
        if qual_str:
            return f"*{qual_str} {self.pointee}"
        return f"*{self.pointee}"


@dataclass(frozen=True, slots=True)
class ReferenceType(Type):
    """Reference to a type (Rust-style).

    Represents Rust references with borrowing semantics:
    - &T: Shared reference (immutable borrow)
    - &mut T: Mutable reference (exclusive borrow)
    - &'a T: Reference with explicit lifetime

    Attributes:
        referent: The type being referenced
        is_mutable: Whether this is a mutable reference (&mut T)
        lifetime: Optional lifetime annotation ('a, 'static, etc.)
    """

    referent: Type
    is_mutable: bool = False
    lifetime: Optional[str] = None

    def free_type_vars(self) -> FrozenSet[str]:
        return self.referent.free_type_vars()

    def substitute(self, substitution: Dict[str, Type]) -> "ReferenceType":
        return ReferenceType(
            referent=self.referent.substitute(substitution),
            is_mutable=self.is_mutable,
            lifetime=self.lifetime,
        )

    def __str__(self) -> str:
        lifetime_str = f"'{self.lifetime} " if self.lifetime else ""
        mut_str = "mut " if self.is_mutable else ""
        return f"&{lifetime_str}{mut_str}{self.referent}"


@dataclass(frozen=True, slots=True)
class SliceType(Type):
    """Slice type - a view into contiguous memory.

    Represents:
    - Zig: []T, []const T, [:sentinel]T
    - Rust: &[T], &mut [T]

    A slice is a pointer + length pair that provides bounds-checked
    access to a contiguous sequence of elements.

    Attributes:
        element: The element type
        is_const: Whether elements are const (Zig)
        is_mutable: Whether elements can be modified (Rust)
        sentinel: Optional sentinel value marking the end (Zig)
    """

    element: Type
    is_const: bool = False
    is_mutable: bool = False
    sentinel: Optional[Any] = None

    def free_type_vars(self) -> FrozenSet[str]:
        return self.element.free_type_vars()

    def substitute(self, substitution: Dict[str, Type]) -> "SliceType":
        return SliceType(
            element=self.element.substitute(substitution),
            is_const=self.is_const,
            is_mutable=self.is_mutable,
            sentinel=self.sentinel,
        )

    def __str__(self) -> str:
        if self.sentinel is not None:
            return f"[:{self.sentinel}]{self.element}"
        const_str = "const " if self.is_const else ""
        mut_str = "mut " if self.is_mutable else ""
        return f"[]{const_str}{mut_str}{self.element}"


@dataclass(frozen=True, slots=True)
class ArrayType(Type):
    """Fixed-length array type.

    Represents:
    - Zig: [N]T, [N:sentinel]T
    - Rust: [T; N]

    Arrays have a compile-time known length and store elements
    contiguously in memory.

    Attributes:
        element: The element type
        length: The number of elements (None if length is a comptime expression)
        sentinel: Optional sentinel value after the array (Zig)
    """

    element: Type
    length: Optional[int] = None
    sentinel: Optional[Any] = None

    def free_type_vars(self) -> FrozenSet[str]:
        return self.element.free_type_vars()

    def substitute(self, substitution: Dict[str, Type]) -> "ArrayType":
        return ArrayType(
            element=self.element.substitute(substitution),
            length=self.length,
            sentinel=self.sentinel,
        )

    def __str__(self) -> str:
        len_str = str(self.length) if self.length is not None else "_"
        if self.sentinel is not None:
            return f"[{len_str}:{self.sentinel}]{self.element}"
        return f"[{len_str}]{self.element}"


@dataclass(frozen=True, slots=True)
class ErrorSetType(Type):
    """Error set type (Zig-specific).

    Represents a set of possible error values:
    - error{OutOfMemory, InvalidArgument}
    - anyerror (the superset of all errors)

    Attributes:
        error_names: The set of error names, or None for anyerror
        is_anyerror: Whether this represents anyerror
    """

    error_names: Optional[FrozenSet[str]] = None
    is_anyerror: bool = False

    def free_type_vars(self) -> FrozenSet[str]:
        return frozenset()

    def substitute(self, substitution: Dict[str, Type]) -> "ErrorSetType":
        return self  # No type vars to substitute

    def __str__(self) -> str:
        if self.is_anyerror:
            return "anyerror"
        if self.error_names:
            names = ", ".join(sorted(self.error_names))
            return f"error{{{names}}}"
        return "error{}"


@dataclass(frozen=True, slots=True)
class ResultType(Type):
    """Result/error union type.

    Represents:
    - Zig: E!T (error union - either an error from E or a value of T)
    - Rust: Result<T, E> (either Ok(T) or Err(E))

    Attributes:
        ok_type: The success type (T)
        error_type: The error type (E)
    """

    ok_type: Type
    error_type: Type

    def free_type_vars(self) -> FrozenSet[str]:
        return self.ok_type.free_type_vars() | self.error_type.free_type_vars()

    def substitute(self, substitution: Dict[str, Type]) -> "ResultType":
        return ResultType(
            ok_type=self.ok_type.substitute(substitution),
            error_type=self.error_type.substitute(substitution),
        )

    def __str__(self) -> str:
        return f"{self.error_type}!{self.ok_type}"


@dataclass(frozen=True, slots=True)
class OptionalType(Type):
    """Optional/nullable type.

    Represents:
    - Zig: ?T (either null or a value of T)
    - Rust: Option<T> (either None or Some(T))

    Note: Python's Optional[T] is represented as UnionType({T, None}).
    This type is for languages with first-class optional types.

    Attributes:
        inner: The wrapped type
    """

    inner: Type

    def free_type_vars(self) -> FrozenSet[str]:
        return self.inner.free_type_vars()

    def substitute(self, substitution: Dict[str, Type]) -> "OptionalType":
        return OptionalType(inner=self.inner.substitute(substitution))

    def __str__(self) -> str:
        return f"?{self.inner}"


@dataclass(frozen=True, slots=True)
class ManyPointerType(Type):
    """Many-item pointer type (Zig-specific).

    Represents Zig's [*]T - a pointer to an unknown number of items.
    Unlike slices, many-pointers don't carry length information.

    Attributes:
        pointee: The element type
        is_const: Whether elements are const
        sentinel: Optional sentinel marking the end
        alignment: Optional alignment override
    """

    pointee: Type
    is_const: bool = False
    sentinel: Optional[Any] = None
    alignment: Optional[int] = None

    def free_type_vars(self) -> FrozenSet[str]:
        return self.pointee.free_type_vars()

    def substitute(self, substitution: Dict[str, Type]) -> "ManyPointerType":
        return ManyPointerType(
            pointee=self.pointee.substitute(substitution),
            is_const=self.is_const,
            sentinel=self.sentinel,
            alignment=self.alignment,
        )

    def __str__(self) -> str:
        parts = ["[*"]
        if self.sentinel is not None:
            parts.append(f":{self.sentinel}")
        parts.append("]")
        if self.is_const:
            parts.append("const ")
        parts.append(str(self.pointee))
        return "".join(parts)


@dataclass(frozen=True, slots=True)
class CPointerType(Type):
    """C-compatible pointer type (Zig-specific).

    Represents Zig's [*c]T - a pointer with C ABI compatibility.
    Can be null and doesn't carry length information.

    Attributes:
        pointee: The element type
        is_const: Whether elements are const
    """

    pointee: Type
    is_const: bool = False

    def free_type_vars(self) -> FrozenSet[str]:
        return self.pointee.free_type_vars()

    def substitute(self, substitution: Dict[str, Type]) -> "CPointerType":
        return CPointerType(
            pointee=self.pointee.substitute(substitution),
            is_const=self.is_const,
        )

    def __str__(self) -> str:
        const_str = "const " if self.is_const else ""
        return f"[*c]{const_str}{self.pointee}"


@dataclass(frozen=True, slots=True)
class ComptimeType(Type):
    """Comptime-known type wrapper (Zig-specific).

    Wraps a type that is known at compile time, enabling
    compile-time evaluation and type manipulation.

    Attributes:
        inner: The underlying type
        comptime_value: The compile-time value if known
    """

    inner: Type
    comptime_value: Optional[Any] = None

    def free_type_vars(self) -> FrozenSet[str]:
        return self.inner.free_type_vars()

    def substitute(self, substitution: Dict[str, Type]) -> "ComptimeType":
        return ComptimeType(
            inner=self.inner.substitute(substitution),
            comptime_value=self.comptime_value,
        )

    def __str__(self) -> str:
        return f"comptime {self.inner}"


@dataclass(frozen=True, slots=True)
class LifetimeBound(Type):
    """Lifetime bound for type parameters (Rust-specific).

    Represents lifetime constraints like 'a, 'static, or 'b: 'a.

    Attributes:
        name: The lifetime name (without the apostrophe)
        outlives: Lifetimes that this lifetime must outlive
    """

    name: str
    outlives: FrozenSet[str] = frozenset()

    def free_type_vars(self) -> FrozenSet[str]:
        return frozenset()  # Lifetimes are not type variables

    def substitute(self, substitution: Dict[str, Type]) -> "LifetimeBound":
        return self  # No type vars to substitute

    def __str__(self) -> str:
        if self.outlives:
            bounds = " + ".join(f"'{b}" for b in sorted(self.outlives))
            return f"'{self.name}: {bounds}"
        return f"'{self.name}"


@dataclass(frozen=True, slots=True)
class TraitBoundType(Type):
    """Type with trait bounds (Rust-specific).

    Represents generic type parameters with constraints:
    - T: Clone + Debug
    - T: Iterator<Item = u32>
    - T: 'static + Send

    Attributes:
        type_param: The type parameter name
        trait_bounds: Set of trait names this type must implement
        lifetime_bounds: Set of lifetime bounds
        associated_types: Mapping of associated type names to their types
    """

    type_param: str
    trait_bounds: FrozenSet[str] = frozenset()
    lifetime_bounds: FrozenSet[str] = frozenset()
    associated_types: Tuple[Tuple[str, Type], ...] = ()

    def free_type_vars(self) -> FrozenSet[str]:
        # The type parameter itself is a free variable
        result = frozenset({self.type_param})
        # Plus any free vars in associated types
        for _, assoc_type in self.associated_types:
            result = result | assoc_type.free_type_vars()
        return result

    def substitute(self, substitution: Dict[str, Type]) -> Type:
        # If this type param is in the substitution, return the substituted type
        if self.type_param in substitution:
            return substitution[self.type_param]
        # Otherwise, substitute in associated types
        new_assocs = tuple(
            (name, t.substitute(substitution))
            for name, t in self.associated_types
        )
        return TraitBoundType(
            type_param=self.type_param,
            trait_bounds=self.trait_bounds,
            lifetime_bounds=self.lifetime_bounds,
            associated_types=new_assocs,
        )

    def __str__(self) -> str:
        bounds = []
        bounds.extend(sorted(self.trait_bounds))
        bounds.extend(f"'{lt}" for lt in sorted(self.lifetime_bounds))
        if bounds:
            return f"{self.type_param}: {' + '.join(bounds)}"
        return self.type_param


# Convenience constructors for common patterns


def pointer_to(t: Type, const: bool = False) -> PointerType:
    """Create a pointer type to T."""
    return PointerType(pointee=t, is_const=const)


def slice_of(t: Type, const: bool = False) -> SliceType:
    """Create a slice type of T."""
    return SliceType(element=t, is_const=const)


def array_of(t: Type, length: int) -> ArrayType:
    """Create an array type [N]T."""
    return ArrayType(element=t, length=length)


def optional(t: Type) -> OptionalType:
    """Create an optional type ?T."""
    return OptionalType(inner=t)


def error_union(ok: Type, err: Type) -> ResultType:
    """Create an error union type E!T."""
    return ResultType(ok_type=ok, error_type=err)


def reference_to(t: Type, mutable: bool = False, lifetime: Optional[str] = None) -> ReferenceType:
    """Create a reference type &T or &mut T."""
    return ReferenceType(referent=t, is_mutable=mutable, lifetime=lifetime)


# Sentinel values for common patterns
ANYERROR = ErrorSetType(is_anyerror=True)
