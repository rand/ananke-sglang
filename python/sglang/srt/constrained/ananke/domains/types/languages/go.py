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
"""Go type system implementation.

This module implements the Go type system for constrained decoding,
supporting Go's type hierarchy, structural interfaces, and generics (Go 1.18+).

Features:
- Primitive types: bool, string, int*, uint*, float*, complex*
- Composite types: arrays, slices, maps, structs, pointers, channels
- Function types with multiple returns
- Interface types (structural subtyping)
- Generic types with type parameters (Go 1.18+)
- Error type handling

References:
    - Go Language Specification: https://go.dev/ref/spec
    - Go Type System: https://go.dev/ref/spec#Types
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    FrozenSet,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    Union,
)

from domains.types.constraint import (
    Type,
    PrimitiveType,
    FunctionType as BaseFunctionType,
    AnyType,
    NeverType,
)
from domains.types.languages.base import (
    LanguageTypeSystem,
    TypeSystemCapabilities,
    TypeParseError,
    LiteralInfo,
    LiteralKind,
)


# =============================================================================
# Go Primitive Types
# =============================================================================

# Boolean
GO_BOOL = PrimitiveType("bool")

# String
GO_STRING = PrimitiveType("string")

# Signed integers
GO_INT = PrimitiveType("int")
GO_INT8 = PrimitiveType("int8")
GO_INT16 = PrimitiveType("int16")
GO_INT32 = PrimitiveType("int32")
GO_INT64 = PrimitiveType("int64")

# Unsigned integers
GO_UINT = PrimitiveType("uint")
GO_UINT8 = PrimitiveType("uint8")
GO_UINT16 = PrimitiveType("uint16")
GO_UINT32 = PrimitiveType("uint32")
GO_UINT64 = PrimitiveType("uint64")
GO_UINTPTR = PrimitiveType("uintptr")

# Aliases
GO_BYTE = GO_UINT8  # byte is alias for uint8
GO_RUNE = GO_INT32  # rune is alias for int32

# Floating point
GO_FLOAT32 = PrimitiveType("float32")
GO_FLOAT64 = PrimitiveType("float64")

# Complex numbers
GO_COMPLEX64 = PrimitiveType("complex64")
GO_COMPLEX128 = PrimitiveType("complex128")

# Special types
GO_ANY = AnyType()  # any is alias for interface{}
GO_COMPARABLE = PrimitiveType("comparable")  # type constraint
GO_ERROR = PrimitiveType("error")  # error interface

# Untyped constants
GO_UNTYPED_BOOL = PrimitiveType("untyped bool")
GO_UNTYPED_INT = PrimitiveType("untyped int")
GO_UNTYPED_RUNE = PrimitiveType("untyped rune")
GO_UNTYPED_FLOAT = PrimitiveType("untyped float")
GO_UNTYPED_COMPLEX = PrimitiveType("untyped complex")
GO_UNTYPED_STRING = PrimitiveType("untyped string")
GO_UNTYPED_NIL = PrimitiveType("untyped nil")


# Primitive type lookup
GO_PRIMITIVES: Dict[str, Type] = {
    "bool": GO_BOOL,
    "string": GO_STRING,
    "int": GO_INT,
    "int8": GO_INT8,
    "int16": GO_INT16,
    "int32": GO_INT32,
    "int64": GO_INT64,
    "uint": GO_UINT,
    "uint8": GO_UINT8,
    "uint16": GO_UINT16,
    "uint32": GO_UINT32,
    "uint64": GO_UINT64,
    "uintptr": GO_UINTPTR,
    "byte": GO_BYTE,
    "rune": GO_RUNE,
    "float32": GO_FLOAT32,
    "float64": GO_FLOAT64,
    "complex64": GO_COMPLEX64,
    "complex128": GO_COMPLEX128,
    "any": GO_ANY,
    "comparable": GO_COMPARABLE,
    "error": GO_ERROR,
}


# =============================================================================
# Go Composite Types
# =============================================================================

@dataclass(frozen=True)
class GoArrayType(Type):
    """Go fixed-size array type: [N]T"""
    length: int
    element: Type

    def __str__(self) -> str:
        return f"[{self.length}]{self.element}"


@dataclass(frozen=True)
class GoSliceType(Type):
    """Go slice type: []T"""
    element: Type

    def __str__(self) -> str:
        return f"[]{self.element}"


@dataclass(frozen=True)
class GoMapType(Type):
    """Go map type: map[K]V"""
    key: Type
    value: Type

    def __str__(self) -> str:
        return f"map[{self.key}]{self.value}"


@dataclass(frozen=True)
class GoPointerType(Type):
    """Go pointer type: *T"""
    pointee: Type

    def __str__(self) -> str:
        return f"*{self.pointee}"


@dataclass(frozen=True)
class GoChannelType(Type):
    """Go channel type: chan T, chan<- T, <-chan T"""
    element: Type
    direction: Literal["bidirectional", "send", "receive"] = "bidirectional"

    def __str__(self) -> str:
        if self.direction == "send":
            return f"chan<- {self.element}"
        elif self.direction == "receive":
            return f"<-chan {self.element}"
        else:
            return f"chan {self.element}"


@dataclass(frozen=True)
class GoFunctionType(Type):
    """Go function type: func(params) (returns)

    Supports multiple return values.
    """
    parameters: Tuple[Tuple[str, Type], ...]  # (name, type) pairs
    returns: Tuple[Type, ...]  # Multiple return values
    variadic: bool = False

    def __str__(self) -> str:
        params = []
        for i, (name, typ) in enumerate(self.parameters):
            if self.variadic and i == len(self.parameters) - 1:
                params.append(f"{name} ...{typ}" if name else f"...{typ}")
            else:
                params.append(f"{name} {typ}" if name else str(typ))

        param_str = ", ".join(params)

        if len(self.returns) == 0:
            return f"func({param_str})"
        elif len(self.returns) == 1:
            return f"func({param_str}) {self.returns[0]}"
        else:
            return_str = ", ".join(str(r) for r in self.returns)
            return f"func({param_str}) ({return_str})"


@dataclass(frozen=True)
class GoMethodType(Type):
    """Go method type with receiver."""
    receiver: Tuple[str, Type]  # (name, type)
    function: GoFunctionType

    def __str__(self) -> str:
        recv_name, recv_type = self.receiver
        func_str = str(self.function)
        # Insert receiver after 'func'
        return func_str.replace("func(", f"func ({recv_name} {recv_type}) (", 1)


@dataclass(frozen=True)
class GoInterfaceType(Type):
    """Go interface type with method set and embedded interfaces."""
    name: Optional[str]
    methods: Tuple[Tuple[str, GoFunctionType], ...]
    embedded: Tuple[Type, ...] = ()

    def __str__(self) -> str:
        if self.name:
            return self.name
        if not self.methods and not self.embedded:
            return "interface{}"
        return "interface{...}"


@dataclass(frozen=True)
class GoStructType(Type):
    """Go struct type with fields."""
    name: Optional[str]
    fields: Tuple[Tuple[str, Type, str], ...]  # (name, type, tag)
    embedded: Tuple[Type, ...] = ()

    def __str__(self) -> str:
        if self.name:
            return self.name
        return "struct{...}"

    def get_field(self, name: str) -> Optional[Type]:
        """Get field type by name."""
        for field_name, field_type, _ in self.fields:
            if field_name == name:
                return field_type
        return None


@dataclass(frozen=True)
class GoTypeParameter(Type):
    """Go type parameter (Go 1.18+): T, K comparable"""
    name: str
    constraint: Optional[Type] = None  # interface constraint

    def __str__(self) -> str:
        if self.constraint:
            return f"{self.name} {self.constraint}"
        return self.name


@dataclass(frozen=True)
class GoGenericType(Type):
    """Go generic instantiation: List[int]"""
    base: str
    type_args: Tuple[Type, ...]

    def __str__(self) -> str:
        args = ", ".join(str(a) for a in self.type_args)
        return f"{self.base}[{args}]"


@dataclass(frozen=True)
class GoNamedType(Type):
    """Go named type (type alias or definition)."""
    name: str
    underlying: Optional[Type] = None

    def __str__(self) -> str:
        return self.name


# =============================================================================
# Go Type System
# =============================================================================

class GoTypeSystem(LanguageTypeSystem):
    """Go type system implementation."""

    def __init__(self):
        self._type_cache: Dict[str, Type] = {}

    @property
    def name(self) -> str:
        return "go"

    @property
    def capabilities(self) -> TypeSystemCapabilities:
        return TypeSystemCapabilities(
            supports_generics=True,           # Go 1.18+
            supports_union_types=False,       # No union types
            supports_optional_types=False,    # Use pointers
            supports_type_inference=True,     # := short declaration
            supports_protocols=True,          # Structural interfaces
            supports_variance=False,          # No variance
            supports_overloading=False,       # No overloading
            supports_ownership=False,         # GC-managed
            supports_comptime=False,
            supports_error_unions=False,      # Uses (T, error)
            supports_lifetime_bounds=False,
            supports_sentinels=False,
            supports_allocators=False,
        )

    def parse_type_annotation(self, annotation: str) -> Type:
        """Parse a Go type annotation string."""
        annotation = annotation.strip()

        if not annotation:
            raise TypeParseError("Empty type annotation")

        # Check cache
        if annotation in self._type_cache:
            return self._type_cache[annotation]

        result = self._parse_type(annotation)
        self._type_cache[annotation] = result
        return result

    def _parse_type(self, text: str) -> Type:
        """Internal type parsing."""
        text = text.strip()

        # Check primitives
        if text in GO_PRIMITIVES:
            return GO_PRIMITIVES[text]

        # Pointer type: *T
        if text.startswith("*"):
            inner = self._parse_type(text[1:])
            return GoPointerType(inner)

        # Slice type: []T
        if text.startswith("[]"):
            inner = self._parse_type(text[2:])
            return GoSliceType(inner)

        # Array type: [N]T
        array_match = re.match(r"^\[(\d+)\](.+)$", text)
        if array_match:
            length = int(array_match.group(1))
            element = self._parse_type(array_match.group(2))
            return GoArrayType(length, element)

        # Map type: map[K]V
        if text.startswith("map["):
            # Find the matching bracket for key type
            bracket_count = 1
            i = 4  # Start after "map["
            while i < len(text) and bracket_count > 0:
                if text[i] == "[":
                    bracket_count += 1
                elif text[i] == "]":
                    bracket_count -= 1
                i += 1
            if bracket_count == 0:
                key_str = text[4:i-1]
                value_str = text[i:]
                key = self._parse_type(key_str)
                value = self._parse_type(value_str)
                return GoMapType(key, value)

        # Channel types
        if text.startswith("<-chan "):
            element = self._parse_type(text[7:])
            return GoChannelType(element, "receive")
        if text.startswith("chan<- "):
            element = self._parse_type(text[7:])
            return GoChannelType(element, "send")
        if text.startswith("chan "):
            element = self._parse_type(text[5:])
            return GoChannelType(element, "bidirectional")

        # Function type: func(params) returns
        if text.startswith("func"):
            return self._parse_function_type(text)

        # Interface type
        if text.startswith("interface{"):
            return self._parse_interface_type(text)

        # Struct type
        if text.startswith("struct{"):
            return self._parse_struct_type(text)

        # Generic type: Name[T1, T2]
        if "[" in text and text.endswith("]"):
            bracket_idx = text.index("[")
            base = text[:bracket_idx]
            args_str = text[bracket_idx+1:-1]
            args = self._parse_type_args(args_str)
            return GoGenericType(base, tuple(args))

        # Named type or type parameter
        if re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", text):
            return GoNamedType(text)

        # Qualified type: package.Type
        if "." in text:
            return GoNamedType(text)

        raise TypeParseError(f"Cannot parse Go type: {text}")

    def _parse_function_type(self, text: str) -> GoFunctionType:
        """Parse a function type."""
        # Simple parsing - find params and returns
        # func(params) returns or func(params) (returns)
        text = text[4:].strip()  # Remove "func"

        if not text.startswith("("):
            raise TypeParseError(f"Invalid function type: {text}")

        # Find matching paren for params
        params_end = self._find_matching_paren(text, 0)
        params_str = text[1:params_end]

        rest = text[params_end+1:].strip()

        # Parse parameters
        params = self._parse_params(params_str)

        # Parse returns
        returns: Tuple[Type, ...] = ()
        if rest:
            if rest.startswith("("):
                # Multiple returns
                returns_end = self._find_matching_paren(rest, 0)
                returns_str = rest[1:returns_end]
                returns = tuple(self._parse_type(t.strip())
                               for t in self._split_type_list(returns_str) if t.strip())
            else:
                # Single return
                returns = (self._parse_type(rest),)

        # Check for variadic
        variadic = params_str.endswith("...")

        return GoFunctionType(tuple(params), returns, variadic)

    def _parse_interface_type(self, text: str) -> GoInterfaceType:
        """Parse an interface type."""
        if text == "interface{}":
            return GoInterfaceType(None, ())

        # For complex interfaces, return a placeholder
        return GoInterfaceType(None, ())

    def _parse_struct_type(self, text: str) -> GoStructType:
        """Parse a struct type."""
        if text == "struct{}":
            return GoStructType(None, ())

        # For complex structs, return a placeholder
        return GoStructType(None, ())

    def _parse_params(self, params_str: str) -> List[Tuple[str, Type]]:
        """Parse function parameters."""
        if not params_str.strip():
            return []

        params = []
        for part in self._split_type_list(params_str):
            part = part.strip()
            if not part:
                continue

            # Handle variadic
            is_variadic = "..." in part
            part = part.replace("...", "")

            # Split name and type
            parts = part.split()
            if len(parts) >= 2:
                name = parts[0]
                type_str = " ".join(parts[1:])
                typ = self._parse_type(type_str)
                params.append((name, typ))
            elif len(parts) == 1:
                # Just type, no name
                typ = self._parse_type(parts[0])
                params.append(("", typ))

        return params

    def _parse_type_args(self, args_str: str) -> List[Type]:
        """Parse generic type arguments."""
        args = []
        for arg in self._split_type_list(args_str):
            arg = arg.strip()
            if arg:
                args.append(self._parse_type(arg))
        return args

    def _split_type_list(self, text: str) -> List[str]:
        """Split a comma-separated type list respecting brackets."""
        parts = []
        current = []
        depth = 0

        for char in text:
            if char in "([{":
                depth += 1
                current.append(char)
            elif char in ")]}":
                depth -= 1
                current.append(char)
            elif char == "," and depth == 0:
                parts.append("".join(current))
                current = []
            else:
                current.append(char)

        if current:
            parts.append("".join(current))

        return parts

    def _find_matching_paren(self, text: str, start: int) -> int:
        """Find matching closing paren."""
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "(":
                depth += 1
            elif text[i] == ")":
                depth -= 1
                if depth == 0:
                    return i
        return len(text)

    def infer_literal_type(self, literal: LiteralInfo) -> Type:
        """Infer the type of a literal value."""
        if literal.kind == LiteralKind.INTEGER:
            return GO_UNTYPED_INT
        elif literal.kind == LiteralKind.FLOAT:
            return GO_UNTYPED_FLOAT
        elif literal.kind == LiteralKind.STRING:
            return GO_UNTYPED_STRING
        elif literal.kind == LiteralKind.BOOLEAN:
            return GO_UNTYPED_BOOL
        elif literal.kind == LiteralKind.NONE:
            return GO_UNTYPED_NIL
        else:
            return GO_ANY

    def check_assignable(self, source: Type, target: Type) -> bool:
        """Check if source type is assignable to target type."""
        # Same type
        if source == target:
            return True

        # Any accepts anything
        if isinstance(target, AnyType):
            return True

        # Never is assignable to nothing
        if isinstance(source, NeverType):
            return True

        # Untyped constants are assignable to compatible types
        if isinstance(source, PrimitiveType) and source.name.startswith("untyped"):
            return self._check_untyped_assignable(source, target)

        # Interface assignability (structural)
        if isinstance(target, GoInterfaceType):
            return self._check_interface_assignable(source, target)

        # Pointer types
        if isinstance(source, GoPointerType) and isinstance(target, GoPointerType):
            return source.pointee == target.pointee

        # Slice types
        if isinstance(source, GoSliceType) and isinstance(target, GoSliceType):
            return source.element == target.element

        # Array types
        if isinstance(source, GoArrayType) and isinstance(target, GoArrayType):
            return source.length == target.length and source.element == target.element

        # Map types
        if isinstance(source, GoMapType) and isinstance(target, GoMapType):
            return source.key == target.key and source.value == target.value

        # Channel types
        if isinstance(source, GoChannelType) and isinstance(target, GoChannelType):
            if source.element != target.element:
                return False
            # Bidirectional can be assigned to directional
            if source.direction == "bidirectional":
                return True
            return source.direction == target.direction

        # Named types are only assignable if identical
        if isinstance(source, GoNamedType) and isinstance(target, GoNamedType):
            return source.name == target.name

        return False

    def _check_untyped_assignable(self, source: PrimitiveType, target: Type) -> bool:
        """Check if an untyped constant is assignable to a type."""
        if source == GO_UNTYPED_INT:
            return target in (
                GO_INT, GO_INT8, GO_INT16, GO_INT32, GO_INT64,
                GO_UINT, GO_UINT8, GO_UINT16, GO_UINT32, GO_UINT64, GO_UINTPTR,
                GO_FLOAT32, GO_FLOAT64, GO_COMPLEX64, GO_COMPLEX128,
                GO_BYTE, GO_RUNE,
            )
        elif source == GO_UNTYPED_FLOAT:
            return target in (GO_FLOAT32, GO_FLOAT64, GO_COMPLEX64, GO_COMPLEX128)
        elif source == GO_UNTYPED_COMPLEX:
            return target in (GO_COMPLEX64, GO_COMPLEX128)
        elif source == GO_UNTYPED_STRING:
            return target == GO_STRING
        elif source == GO_UNTYPED_BOOL:
            return target == GO_BOOL
        elif source == GO_UNTYPED_RUNE:
            return target in (GO_RUNE, GO_INT32)
        elif source == GO_UNTYPED_NIL:
            # nil is assignable to pointers, slices, maps, channels, functions, interfaces
            return isinstance(target, (
                GoPointerType, GoSliceType, GoMapType,
                GoChannelType, GoFunctionType, GoInterfaceType,
            ))
        return False

    def _check_interface_assignable(self, source: Type, target: GoInterfaceType) -> bool:
        """Check if source implements target interface.

        Go interface satisfaction is structural - a type implements an interface
        if it has all the methods declared by the interface.

        Args:
            source: The source type to check
            target: The target interface type

        Returns:
            True if source satisfies the target interface
        """
        # Empty interface (interface{} or any) accepts anything
        if not target.methods and not target.embedded:
            return True

        # Get all required methods from target interface (including embedded)
        required_methods = self._get_interface_methods(target)

        # Interface to interface assignment
        if isinstance(source, GoInterfaceType):
            source_methods = self._get_interface_methods(source)
            # Source interface must have all methods of target interface
            for name, sig in required_methods.items():
                if name not in source_methods:
                    return False
                # Check method signature compatibility
                if not self._check_method_compatible(source_methods[name], sig):
                    return False
            return True

        # Pointer types can satisfy interfaces
        if isinstance(source, GoPointerType):
            # Try the pointee type
            return self._check_interface_assignable(source.pointee, target)

        # Named types can have methods
        if isinstance(source, GoNamedType):
            # Named types may have methods defined elsewhere
            # Without full method tracking, we check if this is a known type
            # that satisfies common interfaces
            return self._check_named_type_interface(source, target, required_methods)

        # Struct types can have methods
        if isinstance(source, GoStructType):
            # Structs with embedded types may satisfy interfaces through embedding
            return self._check_struct_interface(source, target, required_methods)

        # Other types generally don't satisfy non-empty interfaces
        return False

    def _get_interface_methods(self, iface: GoInterfaceType) -> Dict[str, GoFunctionType]:
        """Get all methods from an interface, including embedded interfaces.

        Args:
            iface: The interface to get methods from

        Returns:
            Dict mapping method names to their function types
        """
        methods: Dict[str, GoFunctionType] = {}

        # Add direct methods
        for name, sig in iface.methods:
            methods[name] = sig

        # Add methods from embedded interfaces
        for embedded in iface.embedded:
            if isinstance(embedded, GoInterfaceType):
                embedded_methods = self._get_interface_methods(embedded)
                methods.update(embedded_methods)

        return methods

    def _check_method_compatible(self, source_sig: GoFunctionType, target_sig: GoFunctionType) -> bool:
        """Check if source method signature is compatible with target.

        For Go, method signatures must match exactly (no covariance/contravariance).

        Args:
            source_sig: The source method signature
            target_sig: The target method signature

        Returns:
            True if signatures are compatible
        """
        # Check parameter count
        if len(source_sig.params) != len(target_sig.params):
            return False

        # Check return type count
        if len(source_sig.returns) != len(target_sig.returns):
            return False

        # Check parameter types
        for src_param, tgt_param in zip(source_sig.params, target_sig.params):
            if src_param != tgt_param:
                return False

        # Check return types
        for src_ret, tgt_ret in zip(source_sig.returns, target_sig.returns):
            if src_ret != tgt_ret:
                return False

        return True

    def _check_named_type_interface(
        self,
        source: GoNamedType,
        target: GoInterfaceType,
        required_methods: Dict[str, GoFunctionType],
    ) -> bool:
        """Check if a named type satisfies an interface.

        This is a heuristic check since we don't have full method tracking.

        Args:
            source: The named type
            target: The target interface
            required_methods: Methods required by the interface

        Returns:
            True if the named type likely satisfies the interface
        """
        # Check common interface patterns
        source_name = source.name.lower() if source.name else ""

        # error interface - any type named "error" or ending in "Error"
        if target.name == "error" or (len(required_methods) == 1 and "Error" in required_methods):
            if source_name == "error" or source_name.endswith("error"):
                return True

        # Stringer interface (fmt.Stringer)
        if len(required_methods) == 1 and "String" in required_methods:
            # Common types that implement Stringer
            stringer_types = {
                "time", "duration", "ip", "url", "regexp",
                "int", "float", "rat", "complex",  # big package types
            }
            if any(s in source_name for s in stringer_types):
                return True
            # Types explicitly named as stringers or with string representation
            if "stringer" in source_name or "string" in source_name:
                return True

        # io.Reader interface
        if len(required_methods) == 1 and "Read" in required_methods:
            reader_types = {"reader", "file", "buffer", "conn", "response"}
            if any(r in source_name for r in reader_types):
                return True

        # io.Writer interface
        if len(required_methods) == 1 and "Write" in required_methods:
            writer_types = {"writer", "file", "buffer", "conn", "response"}
            if any(w in source_name for w in writer_types):
                return True

        # io.Closer interface
        if len(required_methods) == 1 and "Close" in required_methods:
            closer_types = {"file", "conn", "client", "response", "reader", "writer"}
            if any(c in source_name for c in closer_types):
                return True

        # For unknown interfaces, be conservative
        return False

    def _check_struct_interface(
        self,
        source: GoStructType,
        target: GoInterfaceType,
        required_methods: Dict[str, GoFunctionType],
    ) -> bool:
        """Check if a struct type satisfies an interface.

        Structs can satisfy interfaces through:
        1. Direct methods (not tracked here)
        2. Embedded types that satisfy the interface

        Args:
            source: The struct type
            target: The target interface
            required_methods: Methods required by the interface

        Returns:
            True if the struct satisfies the interface
        """
        # Check embedded types
        for field in source.fields:
            _, field_type, _ = field
            if isinstance(field_type, GoInterfaceType):
                # Embedded interface directly satisfies target if it's a superset
                if self._check_interface_assignable(field_type, target):
                    return True
            elif isinstance(field_type, GoNamedType):
                # Embedded named type might satisfy interface
                if self._check_interface_assignable(field_type, target):
                    return True

        return False

    def format_type(self, typ: Type) -> str:
        """Format a type for display."""
        return str(typ)

    def get_builtin_types(self) -> Dict[str, Type]:
        """Get all builtin types."""
        return GO_PRIMITIVES.copy()

    def get_builtin_functions(self) -> Dict[str, BaseFunctionType]:
        """Get builtin functions with their types."""
        # Simplified - full implementation would have complete signatures
        return {
            "len": BaseFunctionType((GO_ANY,), GO_INT),
            "cap": BaseFunctionType((GO_ANY,), GO_INT),
            "make": BaseFunctionType((GO_ANY,), GO_ANY),
            "new": BaseFunctionType((GO_ANY,), GO_ANY),
            "append": BaseFunctionType((GO_ANY, GO_ANY), GO_ANY),
            "copy": BaseFunctionType((GO_ANY, GO_ANY), GO_INT),
            "delete": BaseFunctionType((GO_ANY, GO_ANY), PrimitiveType("void")),
            "close": BaseFunctionType((GO_ANY,), PrimitiveType("void")),
            "panic": BaseFunctionType((GO_ANY,), NeverType()),
            "recover": BaseFunctionType((), GO_ANY),
            "print": BaseFunctionType((GO_ANY,), PrimitiveType("void")),
            "println": BaseFunctionType((GO_ANY,), PrimitiveType("void")),
            "real": BaseFunctionType((GO_COMPLEX128,), GO_FLOAT64),
            "imag": BaseFunctionType((GO_COMPLEX128,), GO_FLOAT64),
            "complex": BaseFunctionType((GO_FLOAT64, GO_FLOAT64), GO_COMPLEX128),
        }

    def lub(self, types: List[Type]) -> Type:
        """Compute least upper bound of types."""
        if not types:
            return NeverType()
        if len(types) == 1:
            return types[0]

        # Remove duplicates
        unique = list(dict.fromkeys(types))
        if len(unique) == 1:
            return unique[0]

        # If all are same primitive, return that
        if all(t == unique[0] for t in unique):
            return unique[0]

        # For Go, the LUB is typically interface{} / any
        return GO_ANY

    def glb(self, types: List[Type]) -> Type:
        """Compute greatest lower bound of types."""
        if not types:
            return GO_ANY
        if len(types) == 1:
            return types[0]

        # Remove duplicates
        unique = list(dict.fromkeys(types))
        if len(unique) == 1:
            return unique[0]

        # If all are same, return that
        if all(t == unique[0] for t in unique):
            return unique[0]

        # For Go, the GLB doesn't really exist - return Never
        return NeverType()
