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
"""Rust type system implementation.

This module implements the Rust type system with ownership tracking.
Key features:

- Primitive types: i8-i128, u8-u128, f32, f64, bool, char, str
- Reference types: &T, &mut T, &'a T (with lifetimes)
- Smart pointers: Box<T>, Rc<T>, Arc<T>, Cow<T>
- Option and Result types
- Generic types with trait bounds
- Slice types: &[T], &mut [T]
- Tuple and array types
- Function types

References:
    - The Rust Reference: https://doc.rust-lang.org/reference/
    - Rust Nomicon: https://doc.rust-lang.org/nomicon/
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple as PyTuple

from domains.types.constraint import (
    Type,
    TypeVar,
    PrimitiveType,
    FunctionType,
    TupleType,
    AnyType,
    NeverType,
    HoleType,
    ANY,
    NEVER,
)
from domains.types.extended_types import (
    ReferenceType,
    SliceType,
    ArrayType,
    TraitBoundType,
)
from domains.types.languages.base import (
    LanguageTypeSystem,
    TypeSystemCapabilities,
    TypeParseError,
    LiteralInfo,
    LiteralKind,
)


# =============================================================================
# Rust Primitive Types
# =============================================================================

# Signed integers
RUST_I8 = PrimitiveType("i8")
RUST_I16 = PrimitiveType("i16")
RUST_I32 = PrimitiveType("i32")
RUST_I64 = PrimitiveType("i64")
RUST_I128 = PrimitiveType("i128")
RUST_ISIZE = PrimitiveType("isize")

# Unsigned integers
RUST_U8 = PrimitiveType("u8")
RUST_U16 = PrimitiveType("u16")
RUST_U32 = PrimitiveType("u32")
RUST_U64 = PrimitiveType("u64")
RUST_U128 = PrimitiveType("u128")
RUST_USIZE = PrimitiveType("usize")

# Floats
RUST_F32 = PrimitiveType("f32")
RUST_F64 = PrimitiveType("f64")

# Other primitives
RUST_BOOL = PrimitiveType("bool")
RUST_CHAR = PrimitiveType("char")
RUST_STR = PrimitiveType("str")

# Unit type ()
RUST_UNIT = TupleType(())

# Never type !
RUST_NEVER = NeverType()


# =============================================================================
# Rust-Specific Type Classes
# =============================================================================

@dataclass(frozen=True, slots=True)
class RustReferenceType(Type):
    """Rust reference type &T or &mut T.

    Represents borrowed data with optional lifetime annotation.
    """
    referent: Type
    is_mutable: bool = False
    lifetime: Optional[str] = None

    def free_type_vars(self) -> FrozenSet[str]:
        return self.referent.free_type_vars()

    def substitute(self, substitution: Dict[str, Type]) -> "RustReferenceType":
        return RustReferenceType(
            referent=self.referent.substitute(substitution),
            is_mutable=self.is_mutable,
            lifetime=self.lifetime,
        )

    def __str__(self) -> str:
        lifetime_str = f"'{self.lifetime} " if self.lifetime else ""
        mut_str = "mut " if self.is_mutable else ""
        return f"&{lifetime_str}{mut_str}{self.referent}"


@dataclass(frozen=True, slots=True)
class RustSliceType(Type):
    """Rust slice type &[T] or &mut [T]."""
    element: Type
    is_mutable: bool = False
    lifetime: Optional[str] = None

    def free_type_vars(self) -> FrozenSet[str]:
        return self.element.free_type_vars()

    def substitute(self, substitution: Dict[str, Type]) -> "RustSliceType":
        return RustSliceType(
            element=self.element.substitute(substitution),
            is_mutable=self.is_mutable,
            lifetime=self.lifetime,
        )

    def __str__(self) -> str:
        lifetime_str = f"'{self.lifetime} " if self.lifetime else ""
        mut_str = "mut " if self.is_mutable else ""
        return f"&{lifetime_str}{mut_str}[{self.element}]"


@dataclass(frozen=True, slots=True)
class RustArrayType(Type):
    """Rust array type [T; N]."""
    element: Type
    length: int

    def free_type_vars(self) -> FrozenSet[str]:
        return self.element.free_type_vars()

    def substitute(self, substitution: Dict[str, Type]) -> "RustArrayType":
        return RustArrayType(
            element=self.element.substitute(substitution),
            length=self.length,
        )

    def __str__(self) -> str:
        return f"[{self.element}; {self.length}]"


@dataclass(frozen=True, slots=True)
class RustBoxType(Type):
    """Box<T> - owned heap allocation."""
    inner: Type

    def free_type_vars(self) -> FrozenSet[str]:
        return self.inner.free_type_vars()

    def substitute(self, substitution: Dict[str, Type]) -> "RustBoxType":
        return RustBoxType(inner=self.inner.substitute(substitution))

    def __str__(self) -> str:
        return f"Box<{self.inner}>"


@dataclass(frozen=True, slots=True)
class RustRcType(Type):
    """Rc<T> or Arc<T> - reference counted pointer."""
    inner: Type
    is_arc: bool = False

    def free_type_vars(self) -> FrozenSet[str]:
        return self.inner.free_type_vars()

    def substitute(self, substitution: Dict[str, Type]) -> "RustRcType":
        return RustRcType(
            inner=self.inner.substitute(substitution),
            is_arc=self.is_arc,
        )

    def __str__(self) -> str:
        wrapper = "Arc" if self.is_arc else "Rc"
        return f"{wrapper}<{self.inner}>"


@dataclass(frozen=True, slots=True)
class RustCowType(Type):
    """Cow<'a, T> - clone on write."""
    inner: Type
    lifetime: Optional[str] = None

    def free_type_vars(self) -> FrozenSet[str]:
        return self.inner.free_type_vars()

    def substitute(self, substitution: Dict[str, Type]) -> "RustCowType":
        return RustCowType(
            inner=self.inner.substitute(substitution),
            lifetime=self.lifetime,
        )

    def __str__(self) -> str:
        lifetime_str = f"'{self.lifetime}, " if self.lifetime else ""
        return f"Cow<{lifetime_str}{self.inner}>"


@dataclass(frozen=True, slots=True)
class RustOptionType(Type):
    """Option<T>."""
    inner: Type

    def free_type_vars(self) -> FrozenSet[str]:
        return self.inner.free_type_vars()

    def substitute(self, substitution: Dict[str, Type]) -> "RustOptionType":
        return RustOptionType(inner=self.inner.substitute(substitution))

    def __str__(self) -> str:
        return f"Option<{self.inner}>"


@dataclass(frozen=True, slots=True)
class RustResultType(Type):
    """Result<T, E>."""
    ok_type: Type
    err_type: Type

    def free_type_vars(self) -> FrozenSet[str]:
        return self.ok_type.free_type_vars() | self.err_type.free_type_vars()

    def substitute(self, substitution: Dict[str, Type]) -> "RustResultType":
        return RustResultType(
            ok_type=self.ok_type.substitute(substitution),
            err_type=self.err_type.substitute(substitution),
        )

    def __str__(self) -> str:
        return f"Result<{self.ok_type}, {self.err_type}>"


@dataclass(frozen=True, slots=True)
class RustVecType(Type):
    """Vec<T> - growable array."""
    element: Type

    def free_type_vars(self) -> FrozenSet[str]:
        return self.element.free_type_vars()

    def substitute(self, substitution: Dict[str, Type]) -> "RustVecType":
        return RustVecType(element=self.element.substitute(substitution))

    def __str__(self) -> str:
        return f"Vec<{self.element}>"


@dataclass(frozen=True, slots=True)
class RustStringType(Type):
    """String type (owned string)."""

    def free_type_vars(self) -> FrozenSet[str]:
        return frozenset()

    def substitute(self, substitution: Dict[str, Type]) -> "RustStringType":
        return self

    def __str__(self) -> str:
        return "String"


@dataclass(frozen=True, slots=True)
class RustHashMapType(Type):
    """HashMap<K, V>."""
    key: Type
    value: Type

    def free_type_vars(self) -> FrozenSet[str]:
        return self.key.free_type_vars() | self.value.free_type_vars()

    def substitute(self, substitution: Dict[str, Type]) -> "RustHashMapType":
        return RustHashMapType(
            key=self.key.substitute(substitution),
            value=self.value.substitute(substitution),
        )

    def __str__(self) -> str:
        return f"HashMap<{self.key}, {self.value}>"


@dataclass(frozen=True, slots=True)
class RustTraitBound(Type):
    """Type parameter with trait bounds: T: Clone + Debug."""
    type_param: str
    trait_bounds: FrozenSet[str] = frozenset()
    lifetime_bounds: FrozenSet[str] = frozenset()

    def free_type_vars(self) -> FrozenSet[str]:
        # The type parameter is a free variable
        return frozenset({self.type_param}) if self.type_param else frozenset()

    def substitute(self, substitution: Dict[str, Type]) -> Type:
        # If this type param is in the substitution, return the substituted type
        if self.type_param in substitution:
            return substitution[self.type_param]
        return self

    def __str__(self) -> str:
        bounds = []
        bounds.extend(f"'{lt}" for lt in sorted(self.lifetime_bounds))
        bounds.extend(sorted(self.trait_bounds))
        if bounds:
            return f"{self.type_param}: {' + '.join(bounds)}"
        return self.type_param


@dataclass(frozen=True, slots=True)
class RustGenericType(Type):
    """Generic type with type parameters."""
    name: str
    type_params: PyTuple[Type, ...] = ()
    lifetime_params: PyTuple[str, ...] = ()

    def free_type_vars(self) -> FrozenSet[str]:
        result: FrozenSet[str] = frozenset()
        for tp in self.type_params:
            result = result | tp.free_type_vars()
        return result

    def substitute(self, substitution: Dict[str, Type]) -> "RustGenericType":
        new_params = tuple(tp.substitute(substitution) for tp in self.type_params)
        return RustGenericType(
            name=self.name,
            type_params=new_params,
            lifetime_params=self.lifetime_params,
        )

    def __str__(self) -> str:
        params = []
        params.extend(f"'{lt}" for lt in self.lifetime_params)
        params.extend(str(t) for t in self.type_params)
        if params:
            return f"{self.name}<{', '.join(params)}>"
        return self.name


@dataclass(frozen=True, slots=True)
class RustFunctionType(Type):
    """Rust function type fn(T) -> R or Fn/FnMut/FnOnce traits."""
    params: PyTuple[Type, ...]
    return_type: Type
    is_unsafe: bool = False
    is_extern: bool = False
    abi: Optional[str] = None  # "C", "system", etc.
    trait_kind: Optional[str] = None  # "Fn", "FnMut", "FnOnce"

    def free_type_vars(self) -> FrozenSet[str]:
        result: FrozenSet[str] = frozenset()
        for param in self.params:
            result = result | param.free_type_vars()
        result = result | self.return_type.free_type_vars()
        return result

    def substitute(self, substitution: Dict[str, Type]) -> "RustFunctionType":
        new_params = tuple(p.substitute(substitution) for p in self.params)
        new_return = self.return_type.substitute(substitution)
        return RustFunctionType(
            params=new_params,
            return_type=new_return,
            is_unsafe=self.is_unsafe,
            is_extern=self.is_extern,
            abi=self.abi,
            trait_kind=self.trait_kind,
        )

    def __str__(self) -> str:
        if self.trait_kind:
            params = ", ".join(str(p) for p in self.params)
            return f"{self.trait_kind}({params}) -> {self.return_type}"

        unsafe_str = "unsafe " if self.is_unsafe else ""
        extern_str = ""
        if self.is_extern:
            abi_str = f'"{self.abi}" ' if self.abi else ""
            extern_str = f"extern {abi_str}"

        params = ", ".join(str(p) for p in self.params)
        return f"{unsafe_str}{extern_str}fn({params}) -> {self.return_type}"


@dataclass(frozen=True, slots=True)
class RustRawPointerType(Type):
    """Raw pointer *const T or *mut T."""
    pointee: Type
    is_mutable: bool = False

    def free_type_vars(self) -> FrozenSet[str]:
        return self.pointee.free_type_vars()

    def substitute(self, substitution: Dict[str, Type]) -> "RustRawPointerType":
        return RustRawPointerType(
            pointee=self.pointee.substitute(substitution),
            is_mutable=self.is_mutable,
        )

    def __str__(self) -> str:
        mut_str = "mut" if self.is_mutable else "const"
        return f"*{mut_str} {self.pointee}"


@dataclass(frozen=True, slots=True)
class RustDynTraitType(Type):
    """dyn Trait type (trait object)."""
    trait_name: str
    lifetime: Optional[str] = None

    def free_type_vars(self) -> FrozenSet[str]:
        return frozenset()

    def substitute(self, substitution: Dict[str, Type]) -> "RustDynTraitType":
        return self

    def __str__(self) -> str:
        lifetime_str = f" + '{self.lifetime}" if self.lifetime else ""
        return f"dyn {self.trait_name}{lifetime_str}"


@dataclass(frozen=True, slots=True)
class RustImplTraitType(Type):
    """impl Trait type (existential type)."""
    trait_name: str

    def free_type_vars(self) -> FrozenSet[str]:
        return frozenset()

    def substitute(self, substitution: Dict[str, Type]) -> "RustImplTraitType":
        return self

    def __str__(self) -> str:
        return f"impl {self.trait_name}"


# =============================================================================
# Rust Type System Implementation
# =============================================================================

class RustTypeSystem(LanguageTypeSystem):
    """Rust type system with ownership tracking.

    Implements type parsing, inference, and checking for the Rust language.
    Key features include:

    - Reference and borrow checking
    - Lifetime annotations
    - Trait bounds on generics
    - Smart pointer types
    - Option/Result type handling
    """

    def __init__(self) -> None:
        """Initialize the Rust type system."""
        self._builtin_types = self._build_builtin_types()
        self._builtin_functions = self._build_builtin_functions()

    @property
    def name(self) -> str:
        return "rust"

    @property
    def capabilities(self) -> TypeSystemCapabilities:
        return TypeSystemCapabilities(
            supports_generics=True,
            supports_union_types=True,
            supports_optional_types=True,
            supports_type_inference=True,
            supports_protocols=True,  # Traits
            supports_variance=True,
            supports_overloading=False,
            supports_ownership=True,
            supports_comptime=False,
            supports_error_unions=False,
            supports_lifetime_bounds=True,
            supports_sentinels=False,
            supports_allocators=False,
        )

    def _build_builtin_types(self) -> Dict[str, Type]:
        """Build the dictionary of Rust builtin types."""
        return {
            # Signed integers
            "i8": RUST_I8,
            "i16": RUST_I16,
            "i32": RUST_I32,
            "i64": RUST_I64,
            "i128": RUST_I128,
            "isize": RUST_ISIZE,
            # Unsigned integers
            "u8": RUST_U8,
            "u16": RUST_U16,
            "u32": RUST_U32,
            "u64": RUST_U64,
            "u128": RUST_U128,
            "usize": RUST_USIZE,
            # Floats
            "f32": RUST_F32,
            "f64": RUST_F64,
            # Other
            "bool": RUST_BOOL,
            "char": RUST_CHAR,
            "str": RUST_STR,
            "()": RUST_UNIT,
            "!": RUST_NEVER,
            # Common types
            "String": RustStringType(),
            "Self": RustGenericType(name="Self"),
        }

    def _build_builtin_functions(self) -> Dict[str, FunctionType]:
        """Build dictionary of Rust std function signatures."""
        return {
            # Conversion traits
            "From::from": RustFunctionType((ANY,), ANY),
            "Into::into": RustFunctionType((ANY,), ANY),
            "TryFrom::try_from": RustFunctionType((ANY,), RustResultType(ANY, ANY)),
            "TryInto::try_into": RustFunctionType((ANY,), RustResultType(ANY, ANY)),
            # Clone
            "Clone::clone": RustFunctionType((RustReferenceType(ANY),), ANY),
            "Clone::clone_from": RustFunctionType((RustReferenceType(ANY, is_mutable=True), RustReferenceType(ANY)), RUST_UNIT),
            # Default
            "Default::default": RustFunctionType((), ANY),
            # Debug/Display (simplified)
            "format!": RustFunctionType((RUST_STR,), RustStringType()),
            "println!": RustFunctionType((RUST_STR,), RUST_UNIT),
            "print!": RustFunctionType((RUST_STR,), RUST_UNIT),
            "eprintln!": RustFunctionType((RUST_STR,), RUST_UNIT),
            "eprint!": RustFunctionType((RUST_STR,), RUST_UNIT),
            "dbg!": RustFunctionType((ANY,), ANY),
            # Vec operations
            "Vec::new": RustFunctionType((), RustVecType(ANY)),
            "Vec::with_capacity": RustFunctionType((RUST_USIZE,), RustVecType(ANY)),
            "Vec::push": RustFunctionType((RustReferenceType(ANY, is_mutable=True), ANY), RUST_UNIT),
            "Vec::pop": RustFunctionType((RustReferenceType(ANY, is_mutable=True),), RustOptionType(ANY)),
            "Vec::len": RustFunctionType((RustReferenceType(ANY),), RUST_USIZE),
            "Vec::is_empty": RustFunctionType((RustReferenceType(ANY),), RUST_BOOL),
            # Option operations
            "Option::unwrap": RustFunctionType((RustOptionType(ANY),), ANY),
            "Option::expect": RustFunctionType((RustOptionType(ANY), RustReferenceType(RUST_STR)), ANY),
            "Option::unwrap_or": RustFunctionType((RustOptionType(ANY), ANY), ANY),
            "Option::map": RustFunctionType((RustOptionType(ANY), RustFunctionType((ANY,), ANY)), RustOptionType(ANY)),
            "Option::is_some": RustFunctionType((RustReferenceType(RustOptionType(ANY)),), RUST_BOOL),
            "Option::is_none": RustFunctionType((RustReferenceType(RustOptionType(ANY)),), RUST_BOOL),
            # Result operations
            "Result::unwrap": RustFunctionType((RustResultType(ANY, ANY),), ANY),
            "Result::expect": RustFunctionType((RustResultType(ANY, ANY), RustReferenceType(RUST_STR)), ANY),
            "Result::ok": RustFunctionType((RustResultType(ANY, ANY),), RustOptionType(ANY)),
            "Result::err": RustFunctionType((RustResultType(ANY, ANY),), RustOptionType(ANY)),
            "Result::is_ok": RustFunctionType((RustReferenceType(RustResultType(ANY, ANY)),), RUST_BOOL),
            "Result::is_err": RustFunctionType((RustReferenceType(RustResultType(ANY, ANY)),), RUST_BOOL),
            # Memory
            "std::mem::drop": RustFunctionType((ANY,), RUST_UNIT),
            "std::mem::size_of": RustFunctionType((), RUST_USIZE),
            "std::mem::align_of": RustFunctionType((), RUST_USIZE),
            "std::mem::replace": RustFunctionType((RustReferenceType(ANY, is_mutable=True), ANY), ANY),
            "std::mem::swap": RustFunctionType((RustReferenceType(ANY, is_mutable=True), RustReferenceType(ANY, is_mutable=True)), RUST_UNIT),
        }

    def parse_type_annotation(self, annotation: str) -> Type:
        """Parse a Rust type annotation string.

        Supports:
        - Primitives: i32, u8, f64, bool, char, str
        - References: &T, &mut T, &'a T
        - Raw pointers: *const T, *mut T
        - Slices: &[T], &mut [T]
        - Arrays: [T; N]
        - Tuples: (T, U, V)
        - Option/Result: Option<T>, Result<T, E>
        - Smart pointers: Box<T>, Rc<T>, Arc<T>, Vec<T>
        - Generics: Type<T>, Type<T: Bound>
        - Function types: fn(T) -> R, Fn(T) -> R
        - Trait objects: dyn Trait, impl Trait

        Args:
            annotation: The type annotation string

        Returns:
            The parsed Type

        Raises:
            TypeParseError: If parsing fails
        """
        annotation = annotation.strip()

        if not annotation:
            raise TypeParseError(annotation, "Empty annotation")

        return self._parse_type(annotation)

    def _parse_type(self, s: str) -> Type:
        """Parse a type string recursively."""
        s = s.strip()

        # Check for reference: &T, &mut T, &'a T
        if s.startswith("&"):
            return self._parse_reference(s[1:])

        # Check for raw pointer: *const T, *mut T
        if s.startswith("*"):
            return self._parse_raw_pointer(s[1:])

        # Check for slice reference: &[T]
        # Already handled by _parse_reference

        # Check for array: [T; N]
        if s.startswith("[") and ";" in s:
            return self._parse_array(s)

        # Check for tuple: (T, U, V)
        if s.startswith("("):
            return self._parse_tuple(s)

        # Check for function: fn(...) -> T
        if s.startswith("fn(") or s.startswith("fn ("):
            return self._parse_function_type(s)

        # Check for unsafe/extern fn
        if s.startswith("unsafe ") or s.startswith("extern "):
            return self._parse_function_type(s)

        # Check for dyn Trait
        if s.startswith("dyn "):
            return self._parse_dyn_trait(s[4:])

        # Check for impl Trait
        if s.startswith("impl "):
            return self._parse_impl_trait(s[5:])

        # Check for Fn/FnMut/FnOnce traits
        if s.startswith(("Fn(", "FnMut(", "FnOnce(")):
            return self._parse_fn_trait(s)

        # Check for generic types with <>
        if "<" in s:
            return self._parse_generic(s)

        # Check for never type
        if s == "!":
            return RUST_NEVER

        # Check builtins
        if s in self._builtin_types:
            return self._builtin_types[s]

        # Check for lifetime alone (for trait bounds)
        if s.startswith("'"):
            return RustTraitBound(type_param="", lifetime_bounds=frozenset({s[1:]}))

        # Assume named type (struct, enum, etc.)
        if self._is_valid_identifier(s):
            return RustGenericType(name=s)

        raise TypeParseError(s, f"Cannot parse type: {s}")

    def _parse_reference(self, rest: str) -> Type:
        """Parse &T, &mut T, &'a T, &'a mut T."""
        rest = rest.strip()
        lifetime = None
        is_mutable = False

        # Check for lifetime
        if rest.startswith("'"):
            # Find end of lifetime
            space_pos = rest.find(" ")
            if space_pos > 0:
                lifetime = rest[1:space_pos]
                rest = rest[space_pos + 1:].strip()
            else:
                # Lifetime only - invalid
                raise TypeParseError(rest, "Expected type after lifetime")

        # Check for mut
        if rest.startswith("mut "):
            is_mutable = True
            rest = rest[4:].strip()

        # Check for slice: [T]
        if rest.startswith("[") and not (";" in rest):
            # Parse slice element type
            bracket_end = self._find_matching_bracket(rest, 0, "[", "]")
            element_str = rest[1:bracket_end].strip()
            element = self._parse_type(element_str) if element_str else ANY
            return RustSliceType(element=element, is_mutable=is_mutable, lifetime=lifetime)

        # Regular reference
        referent = self._parse_type(rest)
        return RustReferenceType(referent=referent, is_mutable=is_mutable, lifetime=lifetime)

    def _parse_raw_pointer(self, rest: str) -> Type:
        """Parse *const T, *mut T."""
        rest = rest.strip()

        if rest.startswith("const "):
            pointee = self._parse_type(rest[6:])
            return RustRawPointerType(pointee=pointee, is_mutable=False)
        elif rest.startswith("mut "):
            pointee = self._parse_type(rest[4:])
            return RustRawPointerType(pointee=pointee, is_mutable=True)
        else:
            raise TypeParseError(rest, "Expected 'const' or 'mut' after *")

    def _parse_array(self, s: str) -> RustArrayType:
        """Parse [T; N]."""
        # Find the semicolon
        bracket_end = self._find_matching_bracket(s, 0, "[", "]")
        content = s[1:bracket_end]

        semi_pos = content.rfind(";")
        if semi_pos < 0:
            raise TypeParseError(s, "Expected ';' in array type")

        element_str = content[:semi_pos].strip()
        length_str = content[semi_pos + 1:].strip()

        element = self._parse_type(element_str)

        try:
            length = int(length_str)
        except ValueError:
            # Might be a const generic - treat as unknown length
            length = 0

        return RustArrayType(element=element, length=length)

    def _parse_tuple(self, s: str) -> TupleType:
        """Parse (T, U, V)."""
        if s == "()":
            return RUST_UNIT

        # Find matching paren
        paren_end = self._find_matching_bracket(s, 0, "(", ")")
        content = s[1:paren_end].strip()

        if not content:
            return RUST_UNIT

        # Split by commas
        parts = self._split_by_comma(content)
        elements = tuple(self._parse_type(p.strip()) for p in parts if p.strip())

        return TupleType(elements)

    def _parse_function_type(self, s: str) -> RustFunctionType:
        """Parse fn(T) -> R, unsafe fn(...), extern "C" fn(...)."""
        is_unsafe = False
        is_extern = False
        abi = None

        # Parse qualifiers
        if s.startswith("unsafe "):
            is_unsafe = True
            s = s[7:].strip()

        if s.startswith("extern "):
            is_extern = True
            s = s[7:].strip()

            # Check for ABI string
            if s.startswith('"'):
                end_quote = s.index('"', 1)
                abi = s[1:end_quote]
                s = s[end_quote + 1:].strip()

        # Now parse fn(params) -> return
        if not s.startswith("fn"):
            raise TypeParseError(s, "Expected 'fn'")

        s = s[2:].strip()

        # Parse params
        paren_start = s.index("(")
        paren_end = self._find_matching_bracket(s, paren_start, "(", ")")
        params_str = s[paren_start + 1:paren_end].strip()

        params: List[Type] = []
        if params_str:
            for p in self._split_by_comma(params_str):
                p = p.strip()
                if p:
                    params.append(self._parse_type(p))

        # Parse return type
        rest = s[paren_end + 1:].strip()
        if rest.startswith("->"):
            return_type = self._parse_type(rest[2:].strip())
        else:
            return_type = RUST_UNIT

        return RustFunctionType(
            params=tuple(params),
            return_type=return_type,
            is_unsafe=is_unsafe,
            is_extern=is_extern,
            abi=abi,
        )

    def _parse_fn_trait(self, s: str) -> RustFunctionType:
        """Parse Fn(T) -> R, FnMut(T) -> R, FnOnce(T) -> R."""
        # Determine trait kind
        if s.startswith("FnOnce"):
            trait_kind = "FnOnce"
            s = s[6:]
        elif s.startswith("FnMut"):
            trait_kind = "FnMut"
            s = s[5:]
        else:  # Fn
            trait_kind = "Fn"
            s = s[2:]

        # Parse (params) -> return
        paren_end = self._find_matching_bracket(s, 0, "(", ")")
        params_str = s[1:paren_end].strip()

        params: List[Type] = []
        if params_str:
            for p in self._split_by_comma(params_str):
                p = p.strip()
                if p:
                    params.append(self._parse_type(p))

        rest = s[paren_end + 1:].strip()
        if rest.startswith("->"):
            return_type = self._parse_type(rest[2:].strip())
        else:
            return_type = RUST_UNIT

        return RustFunctionType(
            params=tuple(params),
            return_type=return_type,
            trait_kind=trait_kind,
        )

    def _parse_dyn_trait(self, s: str) -> RustDynTraitType:
        """Parse dyn Trait + 'a."""
        # Check for lifetime bound
        parts = s.split("+")
        trait_name = parts[0].strip()
        lifetime = None

        for part in parts[1:]:
            part = part.strip()
            if part.startswith("'"):
                lifetime = part[1:]

        return RustDynTraitType(trait_name=trait_name, lifetime=lifetime)

    def _parse_impl_trait(self, s: str) -> RustImplTraitType:
        """Parse impl Trait."""
        return RustImplTraitType(trait_name=s.strip())

    def _parse_generic(self, s: str) -> Type:
        """Parse Type<T, U>, Option<T>, Result<T, E>, etc."""
        angle_pos = s.index("<")
        name = s[:angle_pos].strip()
        angle_end = self._find_matching_bracket(s, angle_pos, "<", ">")
        args_str = s[angle_pos + 1:angle_end].strip()

        # Parse type arguments
        args = self._split_by_comma(args_str)
        type_params: List[Type] = []
        lifetime_params: List[str] = []

        for arg in args:
            arg = arg.strip()
            if not arg:
                continue
            if arg.startswith("'"):
                lifetime_params.append(arg[1:])
            else:
                type_params.append(self._parse_type(arg))

        # Handle special types
        if name == "Option" and len(type_params) == 1:
            return RustOptionType(inner=type_params[0])

        if name == "Result" and len(type_params) == 2:
            return RustResultType(ok_type=type_params[0], err_type=type_params[1])

        if name == "Vec" and len(type_params) == 1:
            return RustVecType(element=type_params[0])

        if name == "Box" and len(type_params) == 1:
            return RustBoxType(inner=type_params[0])

        if name == "Rc" and len(type_params) == 1:
            return RustRcType(inner=type_params[0], is_arc=False)

        if name == "Arc" and len(type_params) == 1:
            return RustRcType(inner=type_params[0], is_arc=True)

        if name == "Cow" and len(type_params) >= 1:
            lifetime = lifetime_params[0] if lifetime_params else None
            return RustCowType(inner=type_params[0], lifetime=lifetime)

        if name == "HashMap" and len(type_params) == 2:
            return RustHashMapType(key=type_params[0], value=type_params[1])

        # Generic type
        return RustGenericType(
            name=name,
            type_params=tuple(type_params),
            lifetime_params=tuple(lifetime_params),
        )

    def _find_matching_bracket(self, s: str, start: int, open_b: str, close_b: str) -> int:
        """Find the matching closing bracket."""
        depth = 1
        for i in range(start + 1, len(s)):
            if s[i] == open_b:
                depth += 1
            elif s[i] == close_b:
                depth -= 1
                if depth == 0:
                    return i
        raise TypeParseError(s, f"Unmatched {open_b}")

    def _split_by_comma(self, s: str) -> List[str]:
        """Split by comma, respecting nested brackets."""
        parts = []
        current = []
        depth = 0

        for c in s:
            if c in "<([":
                depth += 1
                current.append(c)
            elif c in ">)]":
                depth -= 1
                current.append(c)
            elif c == "," and depth == 0:
                parts.append("".join(current))
                current = []
            else:
                current.append(c)

        if current:
            parts.append("".join(current))

        return parts

    def _is_valid_identifier(self, s: str) -> bool:
        """Check if s is a valid Rust identifier."""
        if not s:
            return False
        if s[0].isdigit():
            return False
        return all(c.isalnum() or c == "_" or c == ":" for c in s)

    def infer_literal_type(self, literal: LiteralInfo) -> Type:
        """Infer the type of a Rust literal.

        Integer literals default to i32, float literals to f64.
        """
        if literal.kind == LiteralKind.INTEGER:
            # Check for suffix
            if literal.text:
                text = literal.text.lower()
                if text.endswith("i8"):
                    return RUST_I8
                if text.endswith("i16"):
                    return RUST_I16
                if text.endswith("i32"):
                    return RUST_I32
                if text.endswith("i64"):
                    return RUST_I64
                if text.endswith("i128"):
                    return RUST_I128
                if text.endswith("isize"):
                    return RUST_ISIZE
                if text.endswith("u8"):
                    return RUST_U8
                if text.endswith("u16"):
                    return RUST_U16
                if text.endswith("u32"):
                    return RUST_U32
                if text.endswith("u64"):
                    return RUST_U64
                if text.endswith("u128"):
                    return RUST_U128
                if text.endswith("usize"):
                    return RUST_USIZE
            return RUST_I32  # Default

        elif literal.kind == LiteralKind.FLOAT:
            if literal.text:
                text = literal.text.lower()
                if text.endswith("f32"):
                    return RUST_F32
            return RUST_F64  # Default

        elif literal.kind == LiteralKind.STRING:
            # String literals are &str
            return RustReferenceType(referent=RUST_STR, lifetime="static")

        elif literal.kind == LiteralKind.BOOLEAN:
            return RUST_BOOL

        elif literal.kind == LiteralKind.CHARACTER:
            return RUST_CHAR

        else:
            return ANY

    def get_builtin_types(self) -> Dict[str, Type]:
        """Return Rust builtin types."""
        return self._builtin_types.copy()

    def get_builtin_functions(self) -> Dict[str, FunctionType]:
        """Return Rust builtin function signatures."""
        return self._builtin_functions.copy()

    def check_assignable(self, source: Type, target: Type) -> bool:
        """Check if source type is assignable to target type.

        Rust assignability includes:
        - Move semantics for owned types
        - Borrow rules for references
        - Deref coercion
        - Lifetime compatibility
        """
        # Handle special types
        if isinstance(target, AnyType) or isinstance(source, AnyType):
            return True

        if isinstance(source, NeverType):
            return True

        if isinstance(target, NeverType):
            return False

        # Handle holes
        if isinstance(source, HoleType) or isinstance(target, HoleType):
            return True

        # Same type
        if source == target:
            return True

        # Unit type
        if target == RUST_UNIT:
            return source == RUST_UNIT

        # Reference coercion: &mut T -> &T
        if isinstance(source, RustReferenceType) and isinstance(target, RustReferenceType):
            if source.is_mutable and not target.is_mutable:
                # Can coerce &mut T to &T
                return self.check_assignable(source.referent, target.referent)
            if source.is_mutable == target.is_mutable:
                return self.check_assignable(source.referent, target.referent)

        # Slice coercion: &mut [T] -> &[T]
        if isinstance(source, RustSliceType) and isinstance(target, RustSliceType):
            if source.is_mutable and not target.is_mutable:
                return self.check_assignable(source.element, target.element)
            if source.is_mutable == target.is_mutable:
                return self.check_assignable(source.element, target.element)

        # Array to slice coercion: &[T; N] -> &[T]
        if isinstance(source, RustReferenceType) and isinstance(target, RustSliceType):
            if isinstance(source.referent, RustArrayType):
                return self.check_assignable(source.referent.element, target.element)

        # Deref coercion: Box<T> -> &T, etc.
        if isinstance(source, RustBoxType) and isinstance(target, RustReferenceType):
            if not target.is_mutable:
                return self.check_assignable(source.inner, target.referent)

        # String to &str coercion
        if isinstance(source, RustStringType) and isinstance(target, RustReferenceType):
            if isinstance(target.referent, PrimitiveType) and target.referent.name == "str":
                return not target.is_mutable

        # Vec to slice coercion: &Vec<T> -> &[T]
        if isinstance(source, RustReferenceType) and isinstance(target, RustSliceType):
            if isinstance(source.referent, RustVecType):
                return self.check_assignable(source.referent.element, target.element)

        # Option coercion: T -> Option<T> (via Some)
        if isinstance(target, RustOptionType):
            return self.check_assignable(source, target.inner)

        # Result coercion: T -> Result<T, E> (via Ok)
        if isinstance(target, RustResultType):
            return self.check_assignable(source, target.ok_type)

        # Generic type matching
        if isinstance(source, RustGenericType) and isinstance(target, RustGenericType):
            if source.name != target.name:
                return False
            if len(source.type_params) != len(target.type_params):
                return False
            return all(
                self.check_assignable(s, t)
                for s, t in zip(source.type_params, target.type_params)
            )

        # Trait bound checking
        if isinstance(target, RustTraitBound):
            return self._check_trait_bound_satisfied(source, target)

        if isinstance(source, RustTraitBound):
            # A type parameter with bounds can be assigned to a compatible bound
            if isinstance(target, RustTraitBound):
                return self._check_bounds_compatible(source, target)
            # Type parameter can be assigned to any type that could satisfy it
            return True

        # dyn Trait handling
        if isinstance(target, RustDynTraitType):
            return self._check_dyn_trait_assignable(source, target)

        # impl Trait handling
        if isinstance(target, RustImplTraitType):
            return self._check_impl_trait_assignable(source, target)

        # Numeric type promotion (Rust is strict, but allow for inference)
        # In practice, Rust doesn't do implicit numeric coercion

        return False

    def _check_trait_bound_satisfied(self, source: Type, bound: RustTraitBound) -> bool:
        """Check if a concrete type satisfies trait bounds.

        Args:
            source: The concrete type to check
            bound: The trait bound to satisfy

        Returns:
            True if source satisfies all trait bounds
        """
        # Check each required trait
        for trait in bound.trait_bounds:
            if not self._type_implements_trait(source, trait):
                return False

        # Lifetime bounds are checked separately (simplified here)
        # In full Rust, this would involve lifetime analysis

        return True

    def _check_bounds_compatible(self, source: RustTraitBound, target: RustTraitBound) -> bool:
        """Check if source bounds are compatible with target bounds.

        Source bounds must be at least as restrictive as target bounds
        for the assignment to be valid.

        Args:
            source: The source trait bound
            target: The target trait bound

        Returns:
            True if bounds are compatible
        """
        # Source must have all traits that target requires
        if not target.trait_bounds.issubset(source.trait_bounds):
            return False

        # Source must have all lifetime bounds that target requires
        if not target.lifetime_bounds.issubset(source.lifetime_bounds):
            return False

        return True

    def _type_implements_trait(self, typ: Type, trait: str) -> bool:
        """Check if a type implements a specific trait.

        This is a heuristic implementation since we don't have full
        trait impl tracking.

        Args:
            typ: The type to check
            trait: The trait name to check for

        Returns:
            True if the type likely implements the trait
        """
        # Auto traits (marker traits)
        auto_traits = {"Send", "Sync", "Unpin"}
        if trait in auto_traits:
            # Most types implement auto traits by default
            return self._type_is_auto_trait_safe(typ, trait)

        # Common traits with known implementations
        trait_impls = {
            # Clone is implemented by most types
            "Clone": lambda t: not isinstance(t, RustReferenceType) or not t.is_mutable,
            # Copy is stricter - only primitives and simple types
            "Copy": lambda t: isinstance(t, PrimitiveType) or (
                isinstance(t, TupleType) and all(
                    self._type_implements_trait(e, "Copy") for e in t.elements
                )
            ),
            # Debug is implemented by most types
            "Debug": lambda t: True,
            # Display is implemented by strings and primitives
            "Display": lambda t: isinstance(t, (PrimitiveType, RustStringType)),
            # Default
            "Default": lambda t: isinstance(t, PrimitiveType) or isinstance(t, RustOptionType),
            # Sized - most types are sized
            "Sized": lambda t: not isinstance(t, (RustSliceType,)) or isinstance(t, RustReferenceType),
            # Eq and PartialEq
            "Eq": lambda t: isinstance(t, PrimitiveType) and t.name not in ("f32", "f64"),
            "PartialEq": lambda t: True,  # Most types
            # Ord and PartialOrd
            "Ord": lambda t: isinstance(t, PrimitiveType) and t.name not in ("f32", "f64"),
            "PartialOrd": lambda t: isinstance(t, PrimitiveType),
            # Hash
            "Hash": lambda t: isinstance(t, PrimitiveType) and t.name not in ("f32", "f64"),
            # Iterator trait
            "Iterator": lambda t: isinstance(t, (RustVecType, RustSliceType)),
            # Into/From traits
            "Into": lambda t: True,  # All types implement Into<Self>
            "From": lambda t: True,  # Many types implement From
            # Error trait
            "Error": lambda t: (
                isinstance(t, RustGenericType) and "Error" in (t.name or "")
            ),
            # AsRef and AsMut
            "AsRef": lambda t: True,
            "AsMut": lambda t: isinstance(t, RustReferenceType) and t.is_mutable,
            # Deref
            "Deref": lambda t: isinstance(t, (RustBoxType, RustRcType, RustArcType, RustStringType)),
            "DerefMut": lambda t: isinstance(t, RustBoxType),
        }

        if trait in trait_impls:
            return trait_impls[trait](typ)

        # For unknown traits, be conservative and return False
        return False

    def _type_is_auto_trait_safe(self, typ: Type, trait: str) -> bool:
        """Check if a type is safe for auto traits (Send, Sync, Unpin).

        Args:
            typ: The type to check
            trait: The auto trait name

        Returns:
            True if the type is safe for the auto trait
        """
        if trait == "Send":
            # Rc is not Send (thread-unsafe reference counting)
            if isinstance(typ, RustRcType):
                return False
            # Raw pointers are not Send by default
            if isinstance(typ, RustRawPointerType):
                return False
            # Cell/RefCell are not Sync but are Send
            return True

        if trait == "Sync":
            # Rc is not Sync
            if isinstance(typ, RustRcType):
                return False
            # Cell/RefCell are not Sync
            if isinstance(typ, RustGenericType):
                if typ.name in ("Cell", "RefCell", "UnsafeCell"):
                    return False
            return True

        if trait == "Unpin":
            # Most types are Unpin, except self-referential types
            return True

        return True

    def _check_dyn_trait_assignable(self, source: Type, target: RustDynTraitType) -> bool:
        """Check if source can be assigned to dyn Trait.

        Args:
            source: The source type
            target: The dyn Trait type

        Returns:
            True if source implements the required trait
        """
        # The trait_name might contain multiple traits separated by +
        trait_names = [t.strip() for t in target.trait_name.split("+")]
        for trait in trait_names:
            if not self._type_implements_trait(source, trait):
                return False
        return True

    def _check_impl_trait_assignable(self, source: Type, target: RustImplTraitType) -> bool:
        """Check if source can be assigned to impl Trait.

        Args:
            source: The source type
            target: The impl Trait type

        Returns:
            True if source implements the required trait
        """
        # The trait_name might contain multiple traits separated by +
        trait_names = [t.strip() for t in target.trait_name.split("+")]
        for trait in trait_names:
            if not self._type_implements_trait(source, trait):
                return False
        return True

    def format_type(self, typ: Type) -> str:
        """Format a type as Rust syntax."""
        if isinstance(typ, PrimitiveType):
            return typ.name

        if isinstance(typ, AnyType):
            return "_"  # Rust uses _ for inferred types

        if isinstance(typ, NeverType):
            return "!"

        if isinstance(typ, HoleType):
            return f"/* hole {typ.hole_id} */"

        if isinstance(typ, TupleType):
            if not typ.elements:
                return "()"
            elems = ", ".join(self.format_type(e) for e in typ.elements)
            return f"({elems})"

        if isinstance(typ, RustReferenceType):
            lifetime_str = f"'{typ.lifetime} " if typ.lifetime else ""
            mut_str = "mut " if typ.is_mutable else ""
            return f"&{lifetime_str}{mut_str}{self.format_type(typ.referent)}"

        if isinstance(typ, RustSliceType):
            lifetime_str = f"'{typ.lifetime} " if typ.lifetime else ""
            mut_str = "mut " if typ.is_mutable else ""
            return f"&{lifetime_str}{mut_str}[{self.format_type(typ.element)}]"

        if isinstance(typ, RustArrayType):
            return f"[{self.format_type(typ.element)}; {typ.length}]"

        if isinstance(typ, RustBoxType):
            return f"Box<{self.format_type(typ.inner)}>"

        if isinstance(typ, RustRcType):
            wrapper = "Arc" if typ.is_arc else "Rc"
            return f"{wrapper}<{self.format_type(typ.inner)}>"

        if isinstance(typ, RustOptionType):
            return f"Option<{self.format_type(typ.inner)}>"

        if isinstance(typ, RustResultType):
            return f"Result<{self.format_type(typ.ok_type)}, {self.format_type(typ.err_type)}>"

        if isinstance(typ, RustVecType):
            return f"Vec<{self.format_type(typ.element)}>"

        if isinstance(typ, RustStringType):
            return "String"

        if isinstance(typ, RustHashMapType):
            return f"HashMap<{self.format_type(typ.key)}, {self.format_type(typ.value)}>"

        if isinstance(typ, RustFunctionType):
            params = ", ".join(self.format_type(p) for p in typ.params)
            ret = self.format_type(typ.return_type)
            if typ.trait_kind:
                return f"{typ.trait_kind}({params}) -> {ret}"
            unsafe_str = "unsafe " if typ.is_unsafe else ""
            extern_str = f'extern "{typ.abi}" ' if typ.is_extern and typ.abi else ("extern " if typ.is_extern else "")
            return f"{unsafe_str}{extern_str}fn({params}) -> {ret}"

        if isinstance(typ, RustRawPointerType):
            mut_str = "mut" if typ.is_mutable else "const"
            return f"*{mut_str} {self.format_type(typ.pointee)}"

        if isinstance(typ, RustDynTraitType):
            lifetime_str = f" + '{typ.lifetime}" if typ.lifetime else ""
            return f"dyn {typ.trait_name}{lifetime_str}"

        if isinstance(typ, RustImplTraitType):
            return f"impl {typ.trait_name}"

        if isinstance(typ, RustGenericType):
            if typ.type_params or typ.lifetime_params:
                params = []
                params.extend(f"'{lt}" for lt in typ.lifetime_params)
                params.extend(self.format_type(t) for t in typ.type_params)
                return f"{typ.name}<{', '.join(params)}>"
            return typ.name

        if isinstance(typ, RustTraitBound):
            bounds = []
            bounds.extend(f"'{lt}" for lt in sorted(typ.lifetime_bounds))
            bounds.extend(sorted(typ.trait_bounds))
            if bounds:
                return f"{typ.type_param}: {' + '.join(bounds)}"
            return typ.type_param

        return str(typ)

    def get_common_imports(self) -> Dict[str, Type]:
        """Return commonly imported types from std."""
        return {
            "std::vec::Vec": RustVecType(ANY),
            "std::string::String": RustStringType(),
            "std::collections::HashMap": RustHashMapType(ANY, ANY),
            "std::collections::HashSet": RustGenericType(name="HashSet", type_params=(ANY,)),
            "std::collections::BTreeMap": RustGenericType(name="BTreeMap", type_params=(ANY, ANY)),
            "std::collections::VecDeque": RustGenericType(name="VecDeque", type_params=(ANY,)),
            "std::rc::Rc": RustRcType(ANY, is_arc=False),
            "std::sync::Arc": RustRcType(ANY, is_arc=True),
            "std::sync::Mutex": RustGenericType(name="Mutex", type_params=(ANY,)),
            "std::sync::RwLock": RustGenericType(name="RwLock", type_params=(ANY,)),
            "std::cell::Cell": RustGenericType(name="Cell", type_params=(ANY,)),
            "std::cell::RefCell": RustGenericType(name="RefCell", type_params=(ANY,)),
            "std::borrow::Cow": RustCowType(ANY),
            "std::io::Result": RustResultType(ANY, RustGenericType(name="std::io::Error")),
            "std::fmt::Result": RustResultType(RUST_UNIT, RustGenericType(name="std::fmt::Error")),
            "std::path::Path": RustGenericType(name="Path"),
            "std::path::PathBuf": RustGenericType(name="PathBuf"),
            "std::fs::File": RustGenericType(name="File"),
            "std::time::Duration": RustGenericType(name="Duration"),
            "std::time::Instant": RustGenericType(name="Instant"),
        }

    def normalize_type(self, typ: Type) -> Type:
        """Normalize Rust type to canonical form."""
        # Most Rust types are already in canonical form
        return typ
