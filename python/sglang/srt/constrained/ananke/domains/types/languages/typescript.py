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
"""TypeScript type system implementation.

This module implements the TypeScript type system with structural typing.
Key features:

- Primitive types: string, number, boolean, bigint, symbol, undefined, null, void, object
- Special types: any, unknown, never
- Literal types: 'hello', 42, true
- Array types: T[], Array<T>, ReadonlyArray<T>
- Tuple types: [T, U, V], [T, ...U[]]
- Object types: { key: Type }, { [key: string]: Type }
- Function types: (x: T) => R
- Union/Intersection types: A | B, A & B
- Generic types: Type<T, U>
- Conditional types: T extends U ? X : Y
- Mapped types: { [K in Keys]: Type }
- Utility types: Partial, Required, Pick, Omit, Record, etc.

TypeScript's key distinguishing feature is **structural typing** - types are
compatible based on their structure, not their identity/name.

References:
    - TypeScript Handbook: https://www.typescriptlang.org/docs/handbook/
    - TypeScript Language Specification
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, NamedTuple, Optional, Tuple as PyTuple, Union

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
from domains.types.languages.base import (
    LanguageTypeSystem,
    TypeSystemCapabilities,
    TypeParseError,
    LiteralInfo,
    LiteralKind,
)


# =============================================================================
# Helper Types
# =============================================================================


class TSParameter(NamedTuple):
    """TypeScript function parameter with named access.

    Provides cleaner access to function parameters than raw tuples.

    Example:
        >>> func_type = TSFunctionType(...)
        >>> param = func_type.get_parameter(0)
        >>> print(param.name, param.type, param.optional)
    """
    name: str
    type: Type
    optional: bool = False


# =============================================================================
# TypeScript Primitive Types
# =============================================================================

TS_STRING = PrimitiveType("string")
TS_NUMBER = PrimitiveType("number")
TS_BOOLEAN = PrimitiveType("boolean")
TS_BIGINT = PrimitiveType("bigint")
TS_SYMBOL = PrimitiveType("symbol")
TS_UNDEFINED = PrimitiveType("undefined")
TS_NULL = PrimitiveType("null")
TS_VOID = PrimitiveType("void")
TS_OBJECT = PrimitiveType("object")
TS_UNKNOWN = PrimitiveType("unknown")

# Use the standard ANY and NEVER from constraint module
# but also provide TS-prefixed aliases for consistency
TS_ANY = ANY
TS_NEVER = NEVER


# =============================================================================
# TypeScript-Specific Type Classes
# =============================================================================

@dataclass(frozen=True, slots=True)
class TSLiteralType(Type):
    """TypeScript literal type: 'hello', 42, true.

    Literal types are exact value types that can be widened to their base type.
    """
    value: Union[str, int, float, bool]
    kind: str  # 'string' | 'number' | 'boolean'

    def free_type_vars(self) -> FrozenSet[str]:
        return frozenset()

    def substitute(self, substitution: Dict[str, Type]) -> "TSLiteralType":
        return self

    def __str__(self) -> str:
        if self.kind == "string":
            return f"'{self.value}'"
        return str(self.value).lower() if self.kind == "boolean" else str(self.value)


@dataclass(frozen=True, slots=True)
class TSArrayType(Type):
    """TypeScript array type: T[] or Array<T>."""
    element: Type
    is_readonly: bool = False

    def free_type_vars(self) -> FrozenSet[str]:
        return self.element.free_type_vars()

    def substitute(self, substitution: Dict[str, Type]) -> "TSArrayType":
        return TSArrayType(
            element=self.element.substitute(substitution),
            is_readonly=self.is_readonly,
        )

    def __str__(self) -> str:
        if self.is_readonly:
            return f"readonly {self.element}[]"
        return f"{self.element}[]"


@dataclass(frozen=True, slots=True)
class TSTupleType(Type):
    """TypeScript tuple type: [T, U, V]."""
    elements: PyTuple[Type, ...]
    rest_element: Optional[Type] = None
    is_readonly: bool = False

    def free_type_vars(self) -> FrozenSet[str]:
        result = frozenset()
        for elem in self.elements:
            result = result | elem.free_type_vars()
        if self.rest_element:
            result = result | self.rest_element.free_type_vars()
        return result

    def substitute(self, substitution: Dict[str, Type]) -> "TSTupleType":
        return TSTupleType(
            elements=tuple(e.substitute(substitution) for e in self.elements),
            rest_element=self.rest_element.substitute(substitution) if self.rest_element else None,
            is_readonly=self.is_readonly,
        )

    def __str__(self) -> str:
        parts = [str(e) for e in self.elements]
        if self.rest_element:
            parts.append(f"...{self.rest_element}[]")
        prefix = "readonly " if self.is_readonly else ""
        return f"{prefix}[{', '.join(parts)}]"


@dataclass(frozen=True, slots=True)
class TSObjectType(Type):
    """TypeScript object type: { key: Type, ... }."""
    properties: PyTuple[PyTuple[str, Type], ...]  # (name, type) pairs
    optional_properties: FrozenSet[str] = frozenset()
    index_signature: Optional[PyTuple[Type, Type]] = None  # (key_type, value_type)
    call_signature: Optional["TSFunctionType"] = None
    construct_signature: Optional["TSFunctionType"] = None

    def free_type_vars(self) -> FrozenSet[str]:
        result = frozenset()
        for _, typ in self.properties:
            result = result | typ.free_type_vars()
        if self.index_signature:
            result = result | self.index_signature[0].free_type_vars()
            result = result | self.index_signature[1].free_type_vars()
        if self.call_signature:
            result = result | self.call_signature.free_type_vars()
        return result

    def substitute(self, substitution: Dict[str, Type]) -> "TSObjectType":
        return TSObjectType(
            properties=tuple(
                (name, typ.substitute(substitution))
                for name, typ in self.properties
            ),
            optional_properties=self.optional_properties,
            index_signature=(
                (self.index_signature[0].substitute(substitution),
                 self.index_signature[1].substitute(substitution))
                if self.index_signature else None
            ),
            call_signature=(
                self.call_signature.substitute(substitution)
                if self.call_signature else None
            ),
            construct_signature=(
                self.construct_signature.substitute(substitution)
                if self.construct_signature else None
            ),
        )

    def get_property(self, name: str, default: Optional[Type] = None) -> Optional[Type]:
        """Get the type of a property by name.

        Args:
            name: Property name to look up
            default: Value to return if property not found

        Returns:
            The property type, or default if not found
        """
        for prop_name, prop_type in self.properties:
            if prop_name == name:
                return prop_type
        return default

    def __contains__(self, name: str) -> bool:
        """Check if a property exists: 'name' in obj_type."""
        return any(prop_name == name for prop_name, _ in self.properties)

    def __getitem__(self, name: str) -> Type:
        """Get property type by name: obj_type['name'].

        Raises:
            KeyError: If property doesn't exist
        """
        for prop_name, prop_type in self.properties:
            if prop_name == name:
                return prop_type
        raise KeyError(f"Property '{name}' not found in object type")

    def property_names(self) -> FrozenSet[str]:
        """Return all property names."""
        return frozenset(name for name, _ in self.properties)

    def __str__(self) -> str:
        parts = []
        for name, typ in self.properties:
            optional = "?" if name in self.optional_properties else ""
            parts.append(f"{name}{optional}: {typ}")
        if self.index_signature:
            key_type, value_type = self.index_signature
            parts.append(f"[key: {key_type}]: {value_type}")
        return "{ " + ", ".join(parts) + " }"


@dataclass(frozen=True, slots=True)
class TSFunctionType(Type):
    """TypeScript function type: (x: T, y?: U, ...rest: V[]) => R."""
    parameters: PyTuple[PyTuple[str, Type, bool], ...]  # (name, type, optional)
    return_type: Type
    type_parameters: PyTuple["TSTypeParameter", ...] = ()
    rest_parameter: Optional[PyTuple[str, Type]] = None

    def free_type_vars(self) -> FrozenSet[str]:
        result = frozenset()
        for _, typ, _ in self.parameters:
            result = result | typ.free_type_vars()
        result = result | self.return_type.free_type_vars()
        if self.rest_parameter:
            result = result | self.rest_parameter[1].free_type_vars()
        return result

    def substitute(self, substitution: Dict[str, Type]) -> "TSFunctionType":
        return TSFunctionType(
            parameters=tuple(
                (name, typ.substitute(substitution), opt)
                for name, typ, opt in self.parameters
            ),
            return_type=self.return_type.substitute(substitution),
            type_parameters=self.type_parameters,
            rest_parameter=(
                (self.rest_parameter[0], self.rest_parameter[1].substitute(substitution))
                if self.rest_parameter else None
            ),
        )

    def __str__(self) -> str:
        type_params = ""
        if self.type_parameters:
            type_params = "<" + ", ".join(str(tp) for tp in self.type_parameters) + ">"

        params = []
        for name, typ, optional in self.parameters:
            opt_mark = "?" if optional else ""
            params.append(f"{name}{opt_mark}: {typ}")
        if self.rest_parameter:
            name, typ = self.rest_parameter
            params.append(f"...{name}: {typ}[]")

        return f"{type_params}({', '.join(params)}) => {self.return_type}"

    def get_parameter(self, index: int) -> TSParameter:
        """Get a parameter by index as a TSParameter named tuple.

        Args:
            index: Parameter index (0-based)

        Returns:
            TSParameter with name, type, and optional fields

        Raises:
            IndexError: If index is out of bounds
        """
        if index < 0 or index >= len(self.parameters):
            raise IndexError(f"Parameter index {index} out of range (0-{len(self.parameters)-1})")
        name, typ, optional = self.parameters[index]
        return TSParameter(name=name, type=typ, optional=optional)

    def get_parameter_by_name(self, name: str) -> Optional[TSParameter]:
        """Get a parameter by name.

        Args:
            name: Parameter name to look up

        Returns:
            TSParameter if found, None otherwise
        """
        for param_name, typ, optional in self.parameters:
            if param_name == name:
                return TSParameter(name=param_name, type=typ, optional=optional)
        return None

    def iter_parameters(self):
        """Iterate over parameters as TSParameter named tuples.

        Yields:
            TSParameter for each parameter
        """
        for name, typ, optional in self.parameters:
            yield TSParameter(name=name, type=typ, optional=optional)

    @property
    def required_parameters(self) -> List[TSParameter]:
        """Return list of required (non-optional) parameters."""
        return [
            TSParameter(name=name, type=typ, optional=False)
            for name, typ, optional in self.parameters
            if not optional
        ]

    @property
    def optional_parameters(self) -> List[TSParameter]:
        """Return list of optional parameters."""
        return [
            TSParameter(name=name, type=typ, optional=True)
            for name, typ, optional in self.parameters
            if optional
        ]


@dataclass(frozen=True, slots=True)
class TSTypeParameter(Type):
    """TypeScript type parameter: T extends U = Default."""
    name: str
    constraint: Optional[Type] = None
    default: Optional[Type] = None

    def free_type_vars(self) -> FrozenSet[str]:
        result = frozenset({self.name})
        if self.constraint:
            result = result | self.constraint.free_type_vars()
        if self.default:
            result = result | self.default.free_type_vars()
        return result

    def substitute(self, substitution: Dict[str, Type]) -> "TSTypeParameter":
        if self.name in substitution:
            # This type parameter is being substituted - return as is
            return self
        return TSTypeParameter(
            name=self.name,
            constraint=self.constraint.substitute(substitution) if self.constraint else None,
            default=self.default.substitute(substitution) if self.default else None,
        )

    def __str__(self) -> str:
        result = self.name
        if self.constraint:
            result += f" extends {self.constraint}"
        if self.default:
            result += f" = {self.default}"
        return result


@dataclass(frozen=True, slots=True)
class TSUnionType(Type):
    """TypeScript union type: A | B | C."""
    members: FrozenSet[Type]

    def free_type_vars(self) -> FrozenSet[str]:
        result = frozenset()
        for member in self.members:
            result = result | member.free_type_vars()
        return result

    def substitute(self, substitution: Dict[str, Type]) -> "TSUnionType":
        return TSUnionType(
            members=frozenset(m.substitute(substitution) for m in self.members)
        )

    def __str__(self) -> str:
        return " | ".join(str(m) for m in sorted(self.members, key=str))


@dataclass(frozen=True, slots=True)
class TSIntersectionType(Type):
    """TypeScript intersection type: A & B & C."""
    members: FrozenSet[Type]

    def free_type_vars(self) -> FrozenSet[str]:
        result = frozenset()
        for member in self.members:
            result = result | member.free_type_vars()
        return result

    def substitute(self, substitution: Dict[str, Type]) -> "TSIntersectionType":
        return TSIntersectionType(
            members=frozenset(m.substitute(substitution) for m in self.members)
        )

    def __str__(self) -> str:
        return " & ".join(str(m) for m in sorted(self.members, key=str))


@dataclass(frozen=True, slots=True)
class TSConditionalType(Type):
    """TypeScript conditional type: T extends U ? X : Y."""
    check_type: Type
    extends_type: Type
    true_type: Type
    false_type: Type

    def free_type_vars(self) -> FrozenSet[str]:
        return (
            self.check_type.free_type_vars() |
            self.extends_type.free_type_vars() |
            self.true_type.free_type_vars() |
            self.false_type.free_type_vars()
        )

    def substitute(self, substitution: Dict[str, Type]) -> "TSConditionalType":
        return TSConditionalType(
            check_type=self.check_type.substitute(substitution),
            extends_type=self.extends_type.substitute(substitution),
            true_type=self.true_type.substitute(substitution),
            false_type=self.false_type.substitute(substitution),
        )

    def __str__(self) -> str:
        return f"{self.check_type} extends {self.extends_type} ? {self.true_type} : {self.false_type}"


@dataclass(frozen=True, slots=True)
class TSMappedType(Type):
    """TypeScript mapped type: { [K in Keys]: Type }."""
    key_param: str
    key_constraint: Type
    value_type: Type
    readonly_modifier: Optional[str] = None  # '+' | '-' | None
    optional_modifier: Optional[str] = None  # '+' | '-' | None

    def free_type_vars(self) -> FrozenSet[str]:
        return self.key_constraint.free_type_vars() | self.value_type.free_type_vars()

    def substitute(self, substitution: Dict[str, Type]) -> "TSMappedType":
        return TSMappedType(
            key_param=self.key_param,
            key_constraint=self.key_constraint.substitute(substitution),
            value_type=self.value_type.substitute(substitution),
            readonly_modifier=self.readonly_modifier,
            optional_modifier=self.optional_modifier,
        )

    def __str__(self) -> str:
        readonly = f"{self.readonly_modifier}readonly " if self.readonly_modifier else ""
        optional = f"{self.optional_modifier}?" if self.optional_modifier else ""
        return f"{{ {readonly}[{self.key_param} in {self.key_constraint}]{optional}: {self.value_type} }}"


@dataclass(frozen=True, slots=True)
class TSIndexedAccessType(Type):
    """TypeScript indexed access type: T[K]."""
    object_type: Type
    index_type: Type

    def free_type_vars(self) -> FrozenSet[str]:
        return self.object_type.free_type_vars() | self.index_type.free_type_vars()

    def substitute(self, substitution: Dict[str, Type]) -> "TSIndexedAccessType":
        return TSIndexedAccessType(
            object_type=self.object_type.substitute(substitution),
            index_type=self.index_type.substitute(substitution),
        )

    def __str__(self) -> str:
        return f"{self.object_type}[{self.index_type}]"


@dataclass(frozen=True, slots=True)
class TSKeyofType(Type):
    """TypeScript keyof type: keyof T."""
    target: Type

    def free_type_vars(self) -> FrozenSet[str]:
        return self.target.free_type_vars()

    def substitute(self, substitution: Dict[str, Type]) -> "TSKeyofType":
        return TSKeyofType(target=self.target.substitute(substitution))

    def __str__(self) -> str:
        return f"keyof {self.target}"


@dataclass(frozen=True, slots=True)
class TSTypeofType(Type):
    """TypeScript typeof type: typeof x."""
    target: str  # Variable name

    def free_type_vars(self) -> FrozenSet[str]:
        return frozenset()

    def substitute(self, substitution: Dict[str, Type]) -> "TSTypeofType":
        return self

    def __str__(self) -> str:
        return f"typeof {self.target}"


@dataclass(frozen=True, slots=True)
class TSTemplateLiteralType(Type):
    """TypeScript template literal type: `${string}-${number}`."""
    parts: PyTuple[Union[str, Type], ...]

    def free_type_vars(self) -> FrozenSet[str]:
        result = frozenset()
        for part in self.parts:
            if isinstance(part, Type):
                result = result | part.free_type_vars()
        return result

    def substitute(self, substitution: Dict[str, Type]) -> "TSTemplateLiteralType":
        new_parts = []
        for part in self.parts:
            if isinstance(part, Type):
                new_parts.append(part.substitute(substitution))
            else:
                new_parts.append(part)
        return TSTemplateLiteralType(parts=tuple(new_parts))

    def __str__(self) -> str:
        result = "`"
        for part in self.parts:
            if isinstance(part, str):
                result += part
            else:
                result += f"${{{part}}}"
        result += "`"
        return result


@dataclass(frozen=True, slots=True)
class TSTypeReference(Type):
    """Reference to a named type: Array<T>, Promise<T>, MyType."""
    name: str
    type_arguments: PyTuple[Type, ...] = ()

    def free_type_vars(self) -> FrozenSet[str]:
        result = frozenset()
        for arg in self.type_arguments:
            result = result | arg.free_type_vars()
        return result

    def substitute(self, substitution: Dict[str, Type]) -> "TSTypeReference":
        return TSTypeReference(
            name=self.name,
            type_arguments=tuple(arg.substitute(substitution) for arg in self.type_arguments),
        )

    def __str__(self) -> str:
        if self.type_arguments:
            args = ", ".join(str(arg) for arg in self.type_arguments)
            return f"{self.name}<{args}>"
        return self.name


@dataclass(frozen=True, slots=True)
class TSInferType(Type):
    """TypeScript infer type in conditional types: infer U."""
    name: str

    def free_type_vars(self) -> FrozenSet[str]:
        return frozenset({self.name})

    def substitute(self, substitution: Dict[str, Type]) -> Type:
        if self.name in substitution:
            return substitution[self.name]
        return self

    def __str__(self) -> str:
        return f"infer {self.name}"


# =============================================================================
# TypeScript Type System Implementation
# =============================================================================

class TypeScriptTypeSystem(LanguageTypeSystem):
    """TypeScript type system with structural typing.

    TypeScript uses structural typing - types are compatible if their
    structures are compatible, not based on type identity/names.
    """

    def __init__(self):
        self._builtin_types = self._build_builtin_types()
        self._builtin_functions = self._build_builtin_functions()

    @property
    def name(self) -> str:
        return "typescript"

    @property
    def capabilities(self) -> TypeSystemCapabilities:
        return TypeSystemCapabilities(
            supports_generics=True,
            supports_union_types=True,
            supports_optional_types=True,
            supports_type_inference=True,
            supports_protocols=True,  # Structural typing is like protocols
            supports_variance=True,   # Explicit variance in TS 4.7+
            supports_overloading=True,
            supports_ownership=False,
            supports_comptime=False,
            supports_error_unions=False,
            supports_lifetime_bounds=False,
            supports_sentinels=False,
            supports_allocators=False,
        )

    def _build_builtin_types(self) -> Dict[str, Type]:
        """Build the dictionary of TypeScript builtin types."""
        return {
            # Primitives
            "string": TS_STRING,
            "number": TS_NUMBER,
            "boolean": TS_BOOLEAN,
            "bigint": TS_BIGINT,
            "symbol": TS_SYMBOL,
            "undefined": TS_UNDEFINED,
            "null": TS_NULL,
            "void": TS_VOID,
            "object": TS_OBJECT,
            # Special types
            "any": ANY,
            "unknown": TS_UNKNOWN,
            "never": NEVER,
            # Boolean literals
            "true": TSLiteralType(True, "boolean"),
            "false": TSLiteralType(False, "boolean"),
            # Common type constructors (as references)
            "Array": TSTypeReference("Array"),
            "ReadonlyArray": TSTypeReference("ReadonlyArray"),
            "Promise": TSTypeReference("Promise"),
            "Map": TSTypeReference("Map"),
            "Set": TSTypeReference("Set"),
            "WeakMap": TSTypeReference("WeakMap"),
            "WeakSet": TSTypeReference("WeakSet"),
            # Utility types
            "Partial": TSTypeReference("Partial"),
            "Required": TSTypeReference("Required"),
            "Readonly": TSTypeReference("Readonly"),
            "Record": TSTypeReference("Record"),
            "Pick": TSTypeReference("Pick"),
            "Omit": TSTypeReference("Omit"),
            "Exclude": TSTypeReference("Exclude"),
            "Extract": TSTypeReference("Extract"),
            "NonNullable": TSTypeReference("NonNullable"),
            "ReturnType": TSTypeReference("ReturnType"),
            "Parameters": TSTypeReference("Parameters"),
            "ConstructorParameters": TSTypeReference("ConstructorParameters"),
            "InstanceType": TSTypeReference("InstanceType"),
            "ThisType": TSTypeReference("ThisType"),
            "Awaited": TSTypeReference("Awaited"),
            # Error types
            "Error": TSTypeReference("Error"),
            "TypeError": TSTypeReference("TypeError"),
            "RangeError": TSTypeReference("RangeError"),
            "SyntaxError": TSTypeReference("SyntaxError"),
            "ReferenceError": TSTypeReference("ReferenceError"),
            # Other common types
            "Date": TSTypeReference("Date"),
            "RegExp": TSTypeReference("RegExp"),
            "Function": TSTypeReference("Function"),
            "Object": TSTypeReference("Object"),
            "String": TSTypeReference("String"),
            "Number": TSTypeReference("Number"),
            "Boolean": TSTypeReference("Boolean"),
            "Symbol": TSTypeReference("Symbol"),
            "BigInt": TSTypeReference("BigInt"),
        }

    def _build_builtin_functions(self) -> Dict[str, TSFunctionType]:
        """Build dictionary of TypeScript builtin function signatures."""
        return {
            # Console methods
            "console.log": TSFunctionType(
                parameters=(),
                return_type=TS_VOID,
                rest_parameter=("args", ANY),
            ),
            "console.error": TSFunctionType(
                parameters=(),
                return_type=TS_VOID,
                rest_parameter=("args", ANY),
            ),
            "console.warn": TSFunctionType(
                parameters=(),
                return_type=TS_VOID,
                rest_parameter=("args", ANY),
            ),
            # Parsing functions
            "parseInt": TSFunctionType(
                parameters=(
                    ("string", TS_STRING, False),
                    ("radix", TS_NUMBER, True),
                ),
                return_type=TS_NUMBER,
            ),
            "parseFloat": TSFunctionType(
                parameters=(("string", TS_STRING, False),),
                return_type=TS_NUMBER,
            ),
            # Type checking
            "isNaN": TSFunctionType(
                parameters=(("number", TS_NUMBER, False),),
                return_type=TS_BOOLEAN,
            ),
            "isFinite": TSFunctionType(
                parameters=(("number", TS_NUMBER, False),),
                return_type=TS_BOOLEAN,
            ),
            # JSON
            "JSON.parse": TSFunctionType(
                parameters=(("text", TS_STRING, False),),
                return_type=ANY,
            ),
            "JSON.stringify": TSFunctionType(
                parameters=(("value", ANY, False),),
                return_type=TS_STRING,
            ),
            # Array static methods
            "Array.isArray": TSFunctionType(
                parameters=(("arg", ANY, False),),
                return_type=TS_BOOLEAN,
            ),
            "Array.from": TSFunctionType(
                parameters=(("iterable", ANY, False),),
                return_type=TSArrayType(ANY),
            ),
            # Object static methods
            "Object.keys": TSFunctionType(
                parameters=(("obj", ANY, False),),
                return_type=TSArrayType(TS_STRING),
            ),
            "Object.values": TSFunctionType(
                parameters=(("obj", ANY, False),),
                return_type=TSArrayType(ANY),
            ),
            "Object.entries": TSFunctionType(
                parameters=(("obj", ANY, False),),
                return_type=TSArrayType(TSTupleType((TS_STRING, ANY))),
            ),
            "Object.assign": TSFunctionType(
                parameters=(("target", ANY, False),),
                return_type=ANY,
                rest_parameter=("sources", ANY),
            ),
            # Promise
            "Promise.resolve": TSFunctionType(
                parameters=(("value", ANY, True),),
                return_type=TSTypeReference("Promise", (ANY,)),
            ),
            "Promise.reject": TSFunctionType(
                parameters=(("reason", ANY, True),),
                return_type=TSTypeReference("Promise", (NEVER,)),
            ),
            # String methods (commonly used)
            "encodeURIComponent": TSFunctionType(
                parameters=(("str", TS_STRING, False),),
                return_type=TS_STRING,
            ),
            "decodeURIComponent": TSFunctionType(
                parameters=(("str", TS_STRING, False),),
                return_type=TS_STRING,
            ),
        }

    def parse_type_annotation(self, annotation: str) -> Type:
        """Parse a TypeScript type annotation string.

        Supports:
        - Primitives: string, number, boolean, bigint, symbol, undefined, null, void, object
        - Special: any, unknown, never
        - Literals: 'hello', 42, true, false
        - Arrays: T[], Array<T>, ReadonlyArray<T>
        - Tuples: [T, U, V], [T, ...U[]], readonly [T, U]
        - Objects: { key: Type }, { key?: Type }, { [key: string]: Type }
        - Functions: (x: T) => R, (x: T, y?: U, ...rest: V[]) => R
        - Unions/Intersections: A | B | C, A & B & C
        - Generics: Type<T>, Type<T, U>
        - Type operators: keyof T, typeof x, T[K]
        - Conditional: T extends U ? X : Y
        - Mapped: { [K in Keys]: Type }
        - Template literals: `${string}`
        """
        annotation = annotation.strip()

        if not annotation:
            raise TypeParseError(annotation, "Empty annotation")

        return self._parse_type(annotation)

    def _parse_type(self, s: str) -> Type:
        """Parse a type string recursively."""
        s = s.strip()

        if not s:
            raise TypeParseError(s, "Empty type")

        # Handle parenthesized types first
        if s.startswith("(") and self._find_matching_bracket(s, 0, "(", ")") == len(s) - 1:
            # Could be parenthesized type or function type
            # Check if it's followed by => for function
            inner = s[1:-1].strip()
            # If it contains => at top level, it's a function type
            if "=>" in inner:
                return self._parse_function_type(s)
            return self._parse_type(inner)

        # Check for function type: (...) => T
        if "=>" in s:
            return self._parse_function_type(s)

        # Check for union type (lowest precedence)
        union_parts = self._split_by_operator(s, "|")
        if len(union_parts) > 1:
            members = frozenset(self._parse_type(p.strip()) for p in union_parts)
            if len(members) == 1:
                return next(iter(members))
            return TSUnionType(members=members)

        # Check for intersection type
        intersection_parts = self._split_by_operator(s, "&")
        if len(intersection_parts) > 1:
            members = frozenset(self._parse_type(p.strip()) for p in intersection_parts)
            if len(members) == 1:
                return next(iter(members))
            return TSIntersectionType(members=members)

        # Check for conditional type: T extends U ? X : Y
        if " extends " in s and " ? " in s and " : " in s:
            return self._parse_conditional_type(s)

        # Check for keyof
        if s.startswith("keyof "):
            return TSKeyofType(target=self._parse_type(s[6:]))

        # Check for typeof
        if s.startswith("typeof "):
            return TSTypeofType(target=s[7:].strip())

        # Check for infer
        if s.startswith("infer "):
            return TSInferType(name=s[6:].strip())

        # Check for readonly prefix (for arrays/tuples)
        if s.startswith("readonly "):
            inner = self._parse_type(s[9:])
            if isinstance(inner, TSArrayType):
                return TSArrayType(element=inner.element, is_readonly=True)
            if isinstance(inner, TSTupleType):
                return TSTupleType(
                    elements=inner.elements,
                    rest_element=inner.rest_element,
                    is_readonly=True,
                )
            return inner

        # Check for array suffix: T[]
        if s.endswith("[]"):
            element = self._parse_type(s[:-2])
            return TSArrayType(element=element)

        # Check for indexed access: T[K]
        if s.endswith("]") and not s.startswith("["):
            bracket_start = self._find_last_bracket_start(s, "[", "]")
            if bracket_start > 0:
                object_type = self._parse_type(s[:bracket_start])
                index_type = self._parse_type(s[bracket_start + 1:-1])
                return TSIndexedAccessType(object_type=object_type, index_type=index_type)

        # Check for tuple: [T, U, V]
        if s.startswith("["):
            return self._parse_tuple(s)

        # Check for object type: { ... }
        if s.startswith("{"):
            return self._parse_object_type(s)

        # Check for template literal type: `...`
        if s.startswith("`") and s.endswith("`"):
            return self._parse_template_literal(s)

        # Check for string literal: 'hello' or "hello"
        if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
            return TSLiteralType(value=s[1:-1], kind="string")

        # Check for numeric literal
        if self._is_numeric_literal(s):
            if "." in s:
                return TSLiteralType(value=float(s), kind="number")
            return TSLiteralType(value=int(s), kind="number")

        # Check for generic type with type arguments: Type<T, U>
        if "<" in s:
            return self._parse_generic_type(s)

        # Check builtins
        if s in self._builtin_types:
            return self._builtin_types[s]

        # Check for type variable (single uppercase letter or starts with T)
        if len(s) == 1 and s.isupper():
            return TypeVar(name=s)

        # Assume it's a type reference
        if self._is_valid_identifier(s):
            return TSTypeReference(name=s)

        raise TypeParseError(s, f"Cannot parse type: {s}")

    def _parse_function_type(self, s: str) -> TSFunctionType:
        """Parse function type: (x: T, y?: U, ...rest: V[]) => R or <T>(...) => R."""
        s = s.strip()

        # Check for type parameters: <T, U>(...) => R
        type_parameters: PyTuple[TSTypeParameter, ...] = ()
        if s.startswith("<"):
            angle_end = self._find_matching_bracket(s, 0, "<", ">")
            type_params_str = s[1:angle_end]
            type_parameters = self._parse_type_parameters(type_params_str)
            s = s[angle_end + 1:].strip()

        # Find arrow position
        arrow_pos = s.rfind("=>")
        if arrow_pos < 0:
            raise TypeParseError(s, "Expected '=>' in function type")

        params_part = s[:arrow_pos].strip()
        return_part = s[arrow_pos + 2:].strip()

        # Parse return type
        return_type = self._parse_type(return_part)

        # Parse parameters
        if not params_part.startswith("(") or not params_part.endswith(")"):
            raise TypeParseError(s, "Expected parentheses around parameters")

        params_str = params_part[1:-1].strip()
        parameters, rest_parameter = self._parse_parameters(params_str)

        return TSFunctionType(
            parameters=parameters,
            return_type=return_type,
            type_parameters=type_parameters,
            rest_parameter=rest_parameter,
        )

    def _parse_parameters(self, params_str: str) -> PyTuple[PyTuple[PyTuple[str, Type, bool], ...], Optional[PyTuple[str, Type]]]:
        """Parse function parameters."""
        if not params_str:
            return (), None

        parameters: List[PyTuple[str, Type, bool]] = []
        rest_parameter: Optional[PyTuple[str, Type]] = None

        parts = self._split_by_comma(params_str)
        for part in parts:
            part = part.strip()
            if not part:
                continue

            # Check for rest parameter: ...name: Type[]
            if part.startswith("..."):
                rest_part = part[3:].strip()
                if ":" in rest_part:
                    name, type_str = rest_part.split(":", 1)
                    name = name.strip()
                    type_str = type_str.strip()
                    # Remove [] suffix if present
                    if type_str.endswith("[]"):
                        type_str = type_str[:-2]
                    rest_parameter = (name, self._parse_type(type_str))
                else:
                    rest_parameter = (rest_part, ANY)
                continue

            # Regular parameter
            optional = "?" in part.split(":")[0]
            if ":" in part:
                name_part, type_str = part.split(":", 1)
                name = name_part.rstrip("?").strip()
                param_type = self._parse_type(type_str.strip())
            else:
                name = part.rstrip("?").strip()
                param_type = ANY

            parameters.append((name, param_type, optional))

        return tuple(parameters), rest_parameter

    def _parse_type_parameters(self, s: str) -> PyTuple[TSTypeParameter, ...]:
        """Parse type parameters: T, U extends string, V = number."""
        if not s.strip():
            return ()

        params: List[TSTypeParameter] = []
        parts = self._split_by_comma(s)

        for part in parts:
            part = part.strip()
            if not part:
                continue

            name = part
            constraint = None
            default = None

            # Check for default: T = Type
            if " = " in part:
                rest, default_str = part.rsplit(" = ", 1)
                default = self._parse_type(default_str.strip())
                part = rest.strip()

            # Check for constraint: T extends Type
            if " extends " in part:
                name, constraint_str = part.split(" extends ", 1)
                constraint = self._parse_type(constraint_str.strip())
                name = name.strip()
            else:
                name = part.strip()

            params.append(TSTypeParameter(name=name, constraint=constraint, default=default))

        return tuple(params)

    def _parse_tuple(self, s: str) -> TSTupleType:
        """Parse tuple type: [T, U, V] or [T, ...U[]]."""
        if not s.startswith("[") or not s.endswith("]"):
            raise TypeParseError(s, "Expected tuple type [...]")

        content = s[1:-1].strip()
        if not content:
            return TSTupleType(elements=())

        parts = self._split_by_comma(content)
        elements: List[Type] = []
        rest_element: Optional[Type] = None

        for part in parts:
            part = part.strip()
            if not part:
                continue

            # Check for rest element: ...T[]
            if part.startswith("..."):
                rest_str = part[3:].strip()
                if rest_str.endswith("[]"):
                    rest_element = self._parse_type(rest_str[:-2])
                else:
                    rest_element = self._parse_type(rest_str)
            else:
                elements.append(self._parse_type(part))

        return TSTupleType(elements=tuple(elements), rest_element=rest_element)

    def _parse_object_type(self, s: str) -> TSObjectType:
        """Parse object type: { key: Type, key?: Type, [index: Type]: Type }."""
        if not s.startswith("{") or not s.endswith("}"):
            raise TypeParseError(s, "Expected object type {...}")

        content = s[1:-1].strip()
        if not content:
            return TSObjectType(properties=())

        properties: List[PyTuple[str, Type]] = []
        optional_properties: set[str] = set()
        index_signature: Optional[PyTuple[Type, Type]] = None
        call_signature: Optional[TSFunctionType] = None

        # Split by ; or ,
        parts = self._split_by_separator(content, ";,")

        for part in parts:
            part = part.strip()
            if not part:
                continue

            # Check for index signature: [key: Type]: Type
            if part.startswith("[") and "]:" in part:
                bracket_end = part.index("]")
                index_content = part[1:bracket_end]
                value_type_str = part[bracket_end + 2:].strip()

                if ":" in index_content:
                    _, key_type_str = index_content.split(":", 1)
                    key_type = self._parse_type(key_type_str.strip())
                else:
                    key_type = TS_STRING

                value_type = self._parse_type(value_type_str)
                index_signature = (key_type, value_type)
                continue

            # Check for call signature: (...): Type
            if part.startswith("("):
                call_signature = self._parse_function_type(part + " => void")
                continue

            # Regular property: name: Type or name?: Type
            if ":" in part:
                name_part, type_str = part.split(":", 1)
                name = name_part.strip()
                optional = name.endswith("?")
                if optional:
                    name = name[:-1].strip()
                    optional_properties.add(name)

                prop_type = self._parse_type(type_str.strip())
                properties.append((name, prop_type))

        return TSObjectType(
            properties=tuple(properties),
            optional_properties=frozenset(optional_properties),
            index_signature=index_signature,
            call_signature=call_signature,
        )

    def _parse_generic_type(self, s: str) -> Type:
        """Parse generic type: Type<T, U>."""
        angle_pos = s.index("<")
        name = s[:angle_pos].strip()
        angle_end = self._find_matching_bracket(s, angle_pos, "<", ">")
        args_str = s[angle_pos + 1:angle_end].strip()

        # Parse type arguments
        type_args = self._split_by_comma(args_str)
        parsed_args: List[Type] = []

        for arg in type_args:
            arg = arg.strip()
            if arg:
                parsed_args.append(self._parse_type(arg))

        # Handle special array types
        if name == "Array" and len(parsed_args) == 1:
            return TSArrayType(element=parsed_args[0])

        if name == "ReadonlyArray" and len(parsed_args) == 1:
            return TSArrayType(element=parsed_args[0], is_readonly=True)

        return TSTypeReference(name=name, type_arguments=tuple(parsed_args))

    def _parse_conditional_type(self, s: str) -> TSConditionalType:
        """Parse conditional type: T extends U ? X : Y."""
        # Find extends keyword
        extends_pos = s.find(" extends ")
        if extends_pos < 0:
            raise TypeParseError(s, "Expected 'extends' in conditional type")

        check_type = self._parse_type(s[:extends_pos].strip())
        rest = s[extends_pos + 9:].strip()

        # Find ? and : at the right nesting level
        question_pos = -1
        colon_pos = -1
        depth = 0

        for i, c in enumerate(rest):
            if c in "<([{":
                depth += 1
            elif c in ">)]}":
                depth -= 1
            elif c == "?" and depth == 0 and question_pos < 0:
                question_pos = i
            elif c == ":" and depth == 0 and question_pos >= 0:
                colon_pos = i
                break

        if question_pos < 0 or colon_pos < 0:
            raise TypeParseError(s, "Invalid conditional type syntax")

        extends_type = self._parse_type(rest[:question_pos].strip())
        true_type = self._parse_type(rest[question_pos + 1:colon_pos].strip())
        false_type = self._parse_type(rest[colon_pos + 1:].strip())

        return TSConditionalType(
            check_type=check_type,
            extends_type=extends_type,
            true_type=true_type,
            false_type=false_type,
        )

    def _parse_template_literal(self, s: str) -> TSTemplateLiteralType:
        """Parse template literal type: `${string}-${number}`."""
        if not s.startswith("`") or not s.endswith("`"):
            raise TypeParseError(s, "Expected template literal")

        content = s[1:-1]
        parts: List[Union[str, Type]] = []
        current_str = ""
        i = 0

        while i < len(content):
            if content[i:i+2] == "${":
                if current_str:
                    parts.append(current_str)
                    current_str = ""
                # Find matching }
                brace_start = i + 2
                brace_end = content.index("}", brace_start)
                type_str = content[brace_start:brace_end]
                parts.append(self._parse_type(type_str))
                i = brace_end + 1
            else:
                current_str += content[i]
                i += 1

        if current_str:
            parts.append(current_str)

        return TSTemplateLiteralType(parts=tuple(parts))

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

    def _find_last_bracket_start(self, s: str, open_b: str, close_b: str) -> int:
        """Find the start of the last bracket pair."""
        depth = 0
        for i in range(len(s) - 1, -1, -1):
            if s[i] == close_b:
                depth += 1
            elif s[i] == open_b:
                depth -= 1
                if depth == 0:
                    return i
        return -1

    def _split_by_comma(self, s: str) -> List[str]:
        """Split by comma, respecting nested brackets."""
        parts = []
        current: List[str] = []
        depth = 0

        for c in s:
            if c in "<([{":
                depth += 1
                current.append(c)
            elif c in ">)]}":
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

    def _split_by_separator(self, s: str, separators: str) -> List[str]:
        """Split by multiple separators, respecting nested brackets."""
        parts = []
        current: List[str] = []
        depth = 0

        for c in s:
            if c in "<([{":
                depth += 1
                current.append(c)
            elif c in ">)]}":
                depth -= 1
                current.append(c)
            elif c in separators and depth == 0:
                parts.append("".join(current))
                current = []
            else:
                current.append(c)

        if current:
            parts.append("".join(current))

        return parts

    def _split_by_operator(self, s: str, op: str) -> List[str]:
        """Split by operator, respecting nested brackets and strings."""
        parts = []
        current: List[str] = []
        depth = 0
        in_string = False
        string_char = None

        i = 0
        while i < len(s):
            c = s[i]

            # Handle strings
            if c in "\"'`" and not in_string:
                in_string = True
                string_char = c
                current.append(c)
            elif c == string_char and in_string:
                in_string = False
                string_char = None
                current.append(c)
            elif in_string:
                current.append(c)
            # Handle brackets
            elif c in "<([{":
                depth += 1
                current.append(c)
            elif c in ">)]}":
                depth -= 1
                current.append(c)
            # Handle operator
            elif depth == 0 and s[i:i+len(op)] == op:
                # Check for && or || vs & or |
                if op == "|" and i + 1 < len(s) and s[i + 1] == "|":
                    current.append(c)
                elif op == "&" and i + 1 < len(s) and s[i + 1] == "&":
                    current.append(c)
                else:
                    parts.append("".join(current))
                    current = []
                    i += len(op) - 1
            else:
                current.append(c)

            i += 1

        if current:
            parts.append("".join(current))

        return parts

    def _is_numeric_literal(self, s: str) -> bool:
        """Check if string is a numeric literal."""
        try:
            float(s)
            return True
        except ValueError:
            return False

    def _is_valid_identifier(self, s: str) -> bool:
        """Check if string is a valid TypeScript identifier."""
        if not s:
            return False
        if s[0].isdigit():
            return False
        return all(c.isalnum() or c == "_" or c == "$" for c in s)

    def infer_literal_type(self, literal: LiteralInfo) -> Type:
        """Infer the type of a TypeScript literal."""
        if literal.kind == LiteralKind.INTEGER:
            return TS_NUMBER

        elif literal.kind == LiteralKind.FLOAT:
            return TS_NUMBER

        elif literal.kind == LiteralKind.STRING:
            return TS_STRING

        elif literal.kind == LiteralKind.BOOLEAN:
            return TS_BOOLEAN

        elif literal.kind == LiteralKind.NONE:
            return TS_UNDEFINED

        elif literal.kind == LiteralKind.LIST:
            return TSArrayType(element=ANY)

        elif literal.kind == LiteralKind.DICT:
            return TSObjectType(properties=())

        else:
            return ANY

    def get_builtin_types(self) -> Dict[str, Type]:
        """Return TypeScript builtin types."""
        return self._builtin_types.copy()

    def get_builtin_functions(self) -> Dict[str, FunctionType]:
        """Return TypeScript builtin function signatures."""
        return self._builtin_functions.copy()

    def check_assignable(self, source: Type, target: Type) -> bool:
        """Check if source type is assignable to target type.

        TypeScript uses structural typing - types are compatible if
        their structures are compatible.
        """
        # Handle any - the escape hatch
        if isinstance(target, AnyType):
            return True
        if isinstance(source, AnyType):
            # any is assignable to everything except never
            return not isinstance(target, NeverType)

        # Handle unknown - accepts everything
        if isinstance(target, PrimitiveType) and target.name == "unknown":
            return True
        # unknown requires narrowing to be assigned to specific types
        if isinstance(source, PrimitiveType) and source.name == "unknown":
            return isinstance(target, AnyType) or (isinstance(target, PrimitiveType) and target.name == "unknown")

        # Handle never - bottom type
        if isinstance(source, NeverType):
            return True  # never is assignable to everything
        if isinstance(target, NeverType):
            return isinstance(source, NeverType)  # Only never is assignable to never

        # Handle holes
        if isinstance(source, HoleType) or isinstance(target, HoleType):
            return True

        # Same type
        if source == target:
            return True

        # TypeVar matching
        if isinstance(source, TypeVar) or isinstance(target, TypeVar):
            return True  # Type variables are flexible during inference

        # Handle void
        if isinstance(target, PrimitiveType) and target.name == "void":
            return isinstance(source, PrimitiveType) and source.name in ("void", "undefined")

        # Literal type widening
        if isinstance(source, TSLiteralType):
            return self._check_literal_assignable(source, target)

        # Union types
        if isinstance(source, TSUnionType):
            # All members of source must be assignable to target
            return all(self.check_assignable(m, target) for m in source.members)

        if isinstance(target, TSUnionType):
            # Source must be assignable to at least one member
            return any(self.check_assignable(source, m) for m in target.members)

        # Intersection types
        if isinstance(target, TSIntersectionType):
            # Source must be assignable to all members
            return all(self.check_assignable(source, m) for m in target.members)

        if isinstance(source, TSIntersectionType):
            # At least one member must be assignable to target
            return any(self.check_assignable(m, target) for m in source.members)

        # Array types
        if isinstance(source, TSArrayType) and isinstance(target, TSArrayType):
            return self.check_assignable(source.element, target.element)

        # Tuple to array
        if isinstance(source, TSTupleType) and isinstance(target, TSArrayType):
            return all(self.check_assignable(e, target.element) for e in source.elements)

        # Tuple types
        if isinstance(source, TSTupleType) and isinstance(target, TSTupleType):
            return self._check_tuple_assignable(source, target)

        # Object types - structural comparison
        if isinstance(source, TSObjectType) and isinstance(target, TSObjectType):
            return self._check_object_assignable(source, target)

        # Function types
        if isinstance(source, TSFunctionType) and isinstance(target, TSFunctionType):
            return self._check_function_assignable(source, target)

        # Type references
        if isinstance(source, TSTypeReference) and isinstance(target, TSTypeReference):
            return self._check_type_reference_assignable(source, target)

        # Promise assignability
        if isinstance(source, TSTypeReference) and source.name == "Promise":
            if isinstance(target, TSTypeReference) and target.name == "Promise":
                if source.type_arguments and target.type_arguments:
                    return self.check_assignable(source.type_arguments[0], target.type_arguments[0])

        # null/undefined to optional (handled by union)
        if isinstance(source, PrimitiveType) and source.name in ("null", "undefined"):
            if isinstance(target, PrimitiveType) and target.name in ("null", "undefined"):
                return source.name == target.name

        return False

    def _check_literal_assignable(self, source: TSLiteralType, target: Type) -> bool:
        """Check if a literal type is assignable to target."""
        # Literal to same literal
        if isinstance(target, TSLiteralType):
            return source.value == target.value and source.kind == target.kind

        # Literal widens to its base type
        base_type_map = {
            "string": TS_STRING,
            "number": TS_NUMBER,
            "boolean": TS_BOOLEAN,
        }
        base_type = base_type_map.get(source.kind)
        if base_type:
            return self.check_assignable(base_type, target)

        return False

    def _check_tuple_assignable(self, source: TSTupleType, target: TSTupleType) -> bool:
        """Check if source tuple is assignable to target tuple."""
        # Check element count
        if len(source.elements) < len(target.elements):
            return False

        # Check element types
        for i, target_elem in enumerate(target.elements):
            if i < len(source.elements):
                if not self.check_assignable(source.elements[i], target_elem):
                    return False

        # Handle rest elements
        if target.rest_element:
            # All remaining source elements must be assignable to target rest
            for i in range(len(target.elements), len(source.elements)):
                if not self.check_assignable(source.elements[i], target.rest_element):
                    return False
        elif len(source.elements) > len(target.elements) and not source.rest_element:
            # Source has extra elements and target doesn't have rest
            return False

        return True

    def _check_object_assignable(self, source: TSObjectType, target: TSObjectType) -> bool:
        """Check if source object is assignable to target object (structural)."""
        source_props = dict(source.properties)
        target_props = dict(target.properties)

        # Target must have all its required properties satisfied by source
        for name, target_type in target_props.items():
            is_optional = name in target.optional_properties

            if name in source_props:
                # Property exists - check type compatibility
                if not self.check_assignable(source_props[name], target_type):
                    return False
            elif not is_optional:
                # Required property missing in source
                # Check index signature
                if target.index_signature:
                    _, value_type = target.index_signature
                    if not self.check_assignable(source_props.get(name, value_type), target_type):
                        return False
                else:
                    return False

        # Check index signature compatibility
        if target.index_signature:
            target_key_type, target_value_type = target.index_signature
            # All source properties must be assignable to target value type
            for prop_type in source_props.values():
                if not self.check_assignable(prop_type, target_value_type):
                    return False

        # Check call signature compatibility
        if target.call_signature:
            if not source.call_signature:
                return False
            if not self._check_function_assignable(source.call_signature, target.call_signature):
                return False

        return True

    def _check_function_assignable(self, source: TSFunctionType, target: TSFunctionType) -> bool:
        """Check if source function is assignable to target function.

        TypeScript uses bivariant parameter checking (historically), but strict mode
        uses contravariance. We implement bivariance for compatibility.
        """
        # Check parameter count - source can have fewer parameters
        source_param_count = len(source.parameters) + (1 if source.rest_parameter else 0)
        target_param_count = len(target.parameters) + (1 if target.rest_parameter else 0)

        # Target parameters must be covered by source
        for i, (_, target_type, target_optional) in enumerate(target.parameters):
            if i < len(source.parameters):
                _, source_type, _ = source.parameters[i]
                # Bivariant check - either direction works
                if not (self.check_assignable(source_type, target_type) or
                        self.check_assignable(target_type, source_type)):
                    return False
            elif not target_optional and not source.rest_parameter:
                return False

        # Check return type (covariant)
        return self.check_assignable(source.return_type, target.return_type)

    def _check_type_reference_assignable(self, source: TSTypeReference, target: TSTypeReference) -> bool:
        """Check if source type reference is assignable to target."""
        # Same name
        if source.name != target.name:
            return False

        # Same number of type arguments
        if len(source.type_arguments) != len(target.type_arguments):
            return False

        # Check type arguments (assume covariance for simplicity)
        for src_arg, tgt_arg in zip(source.type_arguments, target.type_arguments):
            if not self.check_assignable(src_arg, tgt_arg):
                return False

        return True

    def format_type(self, typ: Type) -> str:
        """Format a type as TypeScript syntax."""
        if isinstance(typ, PrimitiveType):
            return typ.name

        if isinstance(typ, AnyType):
            return "any"

        if isinstance(typ, NeverType):
            return "never"

        if isinstance(typ, HoleType):
            return f"/* hole {typ.hole_id} */"

        if isinstance(typ, TypeVar):
            return typ.name

        if isinstance(typ, TSLiteralType):
            if typ.kind == "string":
                return f"'{typ.value}'"
            return str(typ.value).lower() if typ.kind == "boolean" else str(typ.value)

        if isinstance(typ, TSArrayType):
            elem_str = self.format_type(typ.element)
            # Wrap complex types in parens
            if isinstance(typ.element, (TSUnionType, TSIntersectionType, TSFunctionType)):
                elem_str = f"({elem_str})"
            prefix = "readonly " if typ.is_readonly else ""
            return f"{prefix}{elem_str}[]"

        if isinstance(typ, TSTupleType):
            parts = [self.format_type(e) for e in typ.elements]
            if typ.rest_element:
                parts.append(f"...{self.format_type(typ.rest_element)}[]")
            prefix = "readonly " if typ.is_readonly else ""
            return f"{prefix}[{', '.join(parts)}]"

        if isinstance(typ, TSObjectType):
            parts = []
            for name, prop_type in typ.properties:
                opt = "?" if name in typ.optional_properties else ""
                parts.append(f"{name}{opt}: {self.format_type(prop_type)}")
            if typ.index_signature:
                key_type, value_type = typ.index_signature
                parts.append(f"[key: {self.format_type(key_type)}]: {self.format_type(value_type)}")
            return "{ " + "; ".join(parts) + " }"

        if isinstance(typ, TSFunctionType):
            type_params = ""
            if typ.type_parameters:
                type_params = "<" + ", ".join(str(tp) for tp in typ.type_parameters) + ">"

            params = []
            for name, param_type, optional in typ.parameters:
                opt = "?" if optional else ""
                params.append(f"{name}{opt}: {self.format_type(param_type)}")
            if typ.rest_parameter:
                name, rest_type = typ.rest_parameter
                params.append(f"...{name}: {self.format_type(rest_type)}[]")

            return f"{type_params}({', '.join(params)}) => {self.format_type(typ.return_type)}"

        if isinstance(typ, TSUnionType):
            return " | ".join(self.format_type(m) for m in sorted(typ.members, key=lambda t: str(t)))

        if isinstance(typ, TSIntersectionType):
            return " & ".join(self.format_type(m) for m in sorted(typ.members, key=lambda t: str(t)))

        if isinstance(typ, TSConditionalType):
            return (f"{self.format_type(typ.check_type)} extends {self.format_type(typ.extends_type)} "
                    f"? {self.format_type(typ.true_type)} : {self.format_type(typ.false_type)}")

        if isinstance(typ, TSMappedType):
            readonly = f"{typ.readonly_modifier}readonly " if typ.readonly_modifier else ""
            optional = f"{typ.optional_modifier}?" if typ.optional_modifier else ""
            return f"{{ {readonly}[{typ.key_param} in {self.format_type(typ.key_constraint)}]{optional}: {self.format_type(typ.value_type)} }}"

        if isinstance(typ, TSIndexedAccessType):
            return f"{self.format_type(typ.object_type)}[{self.format_type(typ.index_type)}]"

        if isinstance(typ, TSKeyofType):
            return f"keyof {self.format_type(typ.target)}"

        if isinstance(typ, TSTypeofType):
            return f"typeof {typ.target}"

        if isinstance(typ, TSTemplateLiteralType):
            result = "`"
            for part in typ.parts:
                if isinstance(part, str):
                    result += part
                else:
                    result += f"${{{self.format_type(part)}}}"
            result += "`"
            return result

        if isinstance(typ, TSTypeReference):
            if typ.type_arguments:
                args = ", ".join(self.format_type(arg) for arg in typ.type_arguments)
                return f"{typ.name}<{args}>"
            return typ.name

        if isinstance(typ, TSInferType):
            return f"infer {typ.name}"

        if isinstance(typ, TSTypeParameter):
            result = typ.name
            if typ.constraint:
                result += f" extends {self.format_type(typ.constraint)}"
            if typ.default:
                result += f" = {self.format_type(typ.default)}"
            return result

        if isinstance(typ, TupleType):
            # Python TupleType
            return f"[{', '.join(self.format_type(e) for e in typ.elements)}]"

        if isinstance(typ, FunctionType):
            # Python FunctionType
            params = ", ".join(self.format_type(p) for p in typ.params)
            return f"({params}) => {self.format_type(typ.returns)}"

        return str(typ)

    def get_common_imports(self) -> Dict[str, Type]:
        """Return commonly imported types from TypeScript ecosystem."""
        return {
            # Node.js types
            "Buffer": TSTypeReference("Buffer"),
            "NodeJS.Process": TSTypeReference("Process"),
            # DOM types
            "HTMLElement": TSTypeReference("HTMLElement"),
            "Document": TSTypeReference("Document"),
            "Event": TSTypeReference("Event"),
            "EventTarget": TSTypeReference("EventTarget"),
            # React types (common)
            "React.FC": TSTypeReference("FC"),
            "React.Component": TSTypeReference("Component"),
            "React.ReactNode": TSUnionType(frozenset({
                TS_STRING, TS_NUMBER, TS_BOOLEAN, TS_NULL, TS_UNDEFINED,
                TSTypeReference("ReactElement"),
            })),
        }

    def normalize_type(self, typ: Type) -> Type:
        """Normalize a type to canonical form."""
        # Flatten single-element unions
        if isinstance(typ, TSUnionType) and len(typ.members) == 1:
            return self.normalize_type(next(iter(typ.members)))

        # Flatten single-element intersections
        if isinstance(typ, TSIntersectionType) and len(typ.members) == 1:
            return self.normalize_type(next(iter(typ.members)))

        # Normalize nested unions
        if isinstance(typ, TSUnionType):
            normalized_members: set[Type] = set()
            for member in typ.members:
                normalized = self.normalize_type(member)
                if isinstance(normalized, TSUnionType):
                    normalized_members.update(normalized.members)
                else:
                    normalized_members.add(normalized)
            return TSUnionType(members=frozenset(normalized_members))

        return typ

    def lub(self, types: List[Type]) -> Optional[Type]:
        """Compute the least upper bound of a list of types.

        In TypeScript, LUB is typically a union type.
        """
        if not types:
            return NEVER

        if len(types) == 1:
            return types[0]

        # TypeScript uses union types for LUB
        # Deduplicate via frozenset
        members = frozenset(types)
        if len(members) == 1:
            return next(iter(members))
        return TSUnionType(members=members)

    def glb(self, types: List[Type]) -> Optional[Type]:
        """Compute the greatest lower bound of a list of types.

        In TypeScript, GLB is typically an intersection type.
        """
        if not types:
            return ANY

        if len(types) == 1:
            return types[0]

        # TypeScript uses intersection types for GLB
        # Deduplicate via frozenset
        members = frozenset(types)
        if len(members) == 1:
            return next(iter(members))
        return TSIntersectionType(members=members)

    # =========================================================================
    # Conditional Type Evaluation
    # =========================================================================

    def evaluate_conditional(
        self,
        cond: TSConditionalType,
        context: Optional[Dict[str, Type]] = None,
    ) -> Type:
        """Evaluate a conditional type: T extends U ? X : Y.

        If the check type is a concrete type, evaluate the condition.
        If it contains type variables, return the conditional type unchanged
        (or distribute over unions).

        Args:
            cond: The conditional type to evaluate
            context: Optional type variable substitutions

        Returns:
            The evaluated type (true branch, false branch, or conditional)
        """
        context = context or {}

        # Substitute type variables in check and extends types
        check = cond.check_type.substitute(context) if context else cond.check_type
        extends = cond.extends_type.substitute(context) if context else cond.extends_type

        # If check type is still a type variable, return conditional unchanged
        if isinstance(check, TypeVar):
            return cond.substitute(context) if context else cond

        # Distribute over union types (distributive conditional types)
        if isinstance(check, TSUnionType):
            results = []
            for member in check.members:
                member_cond = TSConditionalType(
                    check_type=member,
                    extends_type=cond.extends_type,
                    true_type=cond.true_type,
                    false_type=cond.false_type,
                )
                results.append(self.evaluate_conditional(member_cond, context))

            # Combine results into union
            if len(results) == 1:
                return results[0]
            return TSUnionType(members=frozenset(results))

        # Evaluate the extends check
        if self.check_assignable(check, extends):
            return cond.true_type.substitute(context) if context else cond.true_type
        else:
            return cond.false_type.substitute(context) if context else cond.false_type

    # =========================================================================
    # Utility Type Operations
    # =========================================================================

    def apply_utility_type(self, name: str, args: List[Type]) -> Type:
        """Apply a TypeScript utility type.

        Implements the built-in utility types like Partial<T>, Required<T>, etc.

        Args:
            name: The utility type name (e.g., "Partial", "Required")
            args: Type arguments passed to the utility type

        Returns:
            The resulting type after applying the utility type
        """
        if not args:
            return TSTypeReference(name=name, type_arguments=())

        target = args[0]

        # Object-modifying utility types
        if name == "Partial" and isinstance(target, TSObjectType):
            # Make all properties optional
            return TSObjectType(
                properties=target.properties,
                optional_properties=frozenset(n for n, _ in target.properties),
                index_signature=target.index_signature,
                call_signature=target.call_signature,
                construct_signature=target.construct_signature,
            )

        elif name == "Required" and isinstance(target, TSObjectType):
            # Make all properties required
            return TSObjectType(
                properties=target.properties,
                optional_properties=frozenset(),
                index_signature=target.index_signature,
                call_signature=target.call_signature,
                construct_signature=target.construct_signature,
            )

        elif name == "Readonly" and isinstance(target, TSObjectType):
            # Make all properties readonly (semantic marker)
            return target  # Readonly is structural, same type for assignability

        elif name == "Pick" and len(args) >= 2 and isinstance(target, TSObjectType):
            # Pick<T, K> - pick properties from K
            keys = self._extract_literal_union(args[1])
            new_props = tuple((n, t) for n, t in target.properties if n in keys)
            new_optional = frozenset(k for k in target.optional_properties if k in keys)
            return TSObjectType(
                properties=new_props,
                optional_properties=new_optional,
            )

        elif name == "Omit" and len(args) >= 2 and isinstance(target, TSObjectType):
            # Omit<T, K> - omit properties from K
            keys = self._extract_literal_union(args[1])
            new_props = tuple((n, t) for n, t in target.properties if n not in keys)
            new_optional = frozenset(k for k in target.optional_properties if k not in keys)
            return TSObjectType(
                properties=new_props,
                optional_properties=new_optional,
            )

        elif name == "Record" and len(args) >= 2:
            # Record<K, V> - object with keys K and values V
            key_type = args[0]
            value_type = args[1]
            return TSObjectType(
                properties=(),
                optional_properties=frozenset(),
                index_signature=(key_type, value_type),
            )

        elif name == "Exclude" and len(args) >= 2:
            # Exclude<T, U> - exclude types assignable to U from T
            if isinstance(target, TSUnionType):
                excluded = args[1]
                remaining = frozenset(
                    m for m in target.members
                    if not self.check_assignable(m, excluded)
                )
                if not remaining:
                    return NEVER
                if len(remaining) == 1:
                    return next(iter(remaining))
                return TSUnionType(members=remaining)
            elif self.check_assignable(target, args[1]):
                return NEVER
            return target

        elif name == "Extract" and len(args) >= 2:
            # Extract<T, U> - extract types assignable to U from T
            if isinstance(target, TSUnionType):
                extracted = args[1]
                matching = frozenset(
                    m for m in target.members
                    if self.check_assignable(m, extracted)
                )
                if not matching:
                    return NEVER
                if len(matching) == 1:
                    return next(iter(matching))
                return TSUnionType(members=matching)
            elif self.check_assignable(target, args[1]):
                return target
            return NEVER

        elif name == "NonNullable":
            # NonNullable<T> - exclude null and undefined
            if isinstance(target, TSUnionType):
                non_null = frozenset(
                    m for m in target.members
                    if m != TS_NULL and m != TS_UNDEFINED
                )
                if not non_null:
                    return NEVER
                if len(non_null) == 1:
                    return next(iter(non_null))
                return TSUnionType(members=non_null)
            elif target in (TS_NULL, TS_UNDEFINED):
                return NEVER
            return target

        elif name == "ReturnType" and isinstance(target, TSFunctionType):
            # ReturnType<T> - get function return type
            return target.return_type

        elif name == "Parameters" and isinstance(target, TSFunctionType):
            # Parameters<T> - get function parameters as tuple
            param_types = tuple(typ for _, typ, _ in target.parameters)
            return TSTupleType(elements=param_types)

        elif name == "Awaited":
            # Awaited<T> - unwrap Promise
            if isinstance(target, TSTypeReference) and target.name == "Promise":
                if target.type_arguments:
                    return target.type_arguments[0]
            return target

        # Fall through - return as type reference
        return TSTypeReference(name=name, type_arguments=tuple(args))

    def _extract_literal_union(self, typ: Type) -> FrozenSet[str]:
        """Extract string literal values from a type (for Pick/Omit)."""
        if isinstance(typ, TSLiteralType) and typ.kind == "string":
            return frozenset({str(typ.value)})
        elif isinstance(typ, TSUnionType):
            result: set[str] = set()
            for member in typ.members:
                if isinstance(member, TSLiteralType) and member.kind == "string":
                    result.add(str(member.value))
            return frozenset(result)
        return frozenset()
