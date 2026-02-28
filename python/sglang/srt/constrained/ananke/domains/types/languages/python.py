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
"""Python type system implementation.

This module implements the Python type system following PEP 484 and
compatible with mypy/pyright semantics. Key features:

- Primitive types: int, str, bool, float, bytes, None
- Generic types: List[T], Dict[K,V], Set[T], Tuple[...]
- Callable types: Callable[[Args...], Return]
- Union types: Union[T1, T2], Optional[T]
- Type variables and generic constraints
- Protocol support (structural subtyping)
- Type narrowing and guards

References:
    - PEP 484: Type Hints
    - PEP 544: Protocols
    - PEP 604: Union syntax (|)
    - mypy documentation
    - pyright specification
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Set, Tuple as PyTuple

from domains.types.constraint import (
    Type,
    TypeVar,
    PrimitiveType,
    FunctionType,
    ListType,
    DictType,
    TupleType,
    SetType,
    UnionType,
    ClassType,
    AnyType,
    NeverType,
    HoleType,
    OptionalType,  # This is a function, not a class
    INT,
    STR,
    BOOL,
    FLOAT,
    NONE,
    ANY,
    NEVER,
)


def is_optional_type(typ: Type) -> bool:
    """Check if a type is an Optional type (Union with None)."""
    if not isinstance(typ, UnionType):
        return False
    return NONE in typ.members


def get_optional_inner(typ: Type) -> Type:
    """Get the inner type of an Optional (removes None from union)."""
    if not isinstance(typ, UnionType):
        return typ
    non_none = frozenset(t for t in typ.members if t != NONE)
    if len(non_none) == 0:
        return NONE
    if len(non_none) == 1:
        return next(iter(non_none))
    return UnionType(non_none)


from domains.types.languages.base import (
    LanguageTypeSystem,
    TypeSystemCapabilities,
    TypeParseError,
    LiteralInfo,
    LiteralKind,
)


# Python-specific primitive types
BYTES = PrimitiveType("bytes")
COMPLEX = PrimitiveType("complex")
OBJECT = PrimitiveType("object")


class PythonTypeSystem(LanguageTypeSystem):
    """Python type system following PEP 484 semantics.

    Implements type parsing, inference, and checking compatible with
    mypy and pyright.
    """

    def __init__(self):
        """Initialize the Python type system."""
        self._builtin_types = self._build_builtin_types()
        self._builtin_functions = self._build_builtin_functions()

    @property
    def name(self) -> str:
        return "python"

    @property
    def capabilities(self) -> TypeSystemCapabilities:
        return TypeSystemCapabilities(
            supports_generics=True,
            supports_union_types=True,
            supports_optional_types=True,
            supports_type_inference=True,
            supports_protocols=True,
            supports_variance=True,
            supports_overloading=True,
            supports_ownership=False,
            supports_comptime=False,
        )

    def _build_builtin_types(self) -> Dict[str, Type]:
        """Build the dictionary of Python builtin types."""
        return {
            # Primitives
            "int": INT,
            "str": STR,
            "bool": BOOL,
            "float": FLOAT,
            "bytes": BYTES,
            "complex": COMPLEX,
            "None": NONE,
            "NoneType": NONE,
            "object": OBJECT,
            # Special types
            "Any": ANY,
            "Never": NEVER,
            "NoReturn": NEVER,
            # Generic aliases (without parameters)
            "list": ListType(ANY),
            "dict": DictType(ANY, ANY),
            "set": SetType(ANY),
            "frozenset": SetType(ANY),
            "tuple": TupleType(()),
            # typing module types
            "List": ListType(ANY),
            "Dict": DictType(ANY, ANY),
            "Set": SetType(ANY),
            "FrozenSet": SetType(ANY),
            "Tuple": TupleType(()),
            "Callable": FunctionType((), ANY),
            "Optional": OptionalType(ANY),
            "Union": UnionType(frozenset()),
            "Type": ClassType("type", ()),
            "Sequence": ListType(ANY),  # Simplified
            "Mapping": DictType(ANY, ANY),  # Simplified
            "Iterable": ListType(ANY),  # Simplified
            "Iterator": ListType(ANY),  # Simplified
        }

    def _build_builtin_functions(self) -> Dict[str, FunctionType]:
        """Build the dictionary of Python builtin functions."""
        return {
            # Type conversions
            "int": FunctionType((ANY,), INT),
            "str": FunctionType((ANY,), STR),
            "bool": FunctionType((ANY,), BOOL),
            "float": FunctionType((ANY,), FLOAT),
            "bytes": FunctionType((ANY,), BYTES),
            "list": FunctionType((ANY,), ListType(ANY)),
            "dict": FunctionType((), DictType(ANY, ANY)),
            "set": FunctionType((ANY,), SetType(ANY)),
            "tuple": FunctionType((ANY,), TupleType(())),
            # Numeric functions
            "abs": FunctionType((INT,), INT),  # Simplified
            "round": FunctionType((FLOAT,), INT),
            "min": FunctionType((ANY, ANY), ANY),
            "max": FunctionType((ANY, ANY), ANY),
            "sum": FunctionType((ListType(INT),), INT),  # Simplified
            "pow": FunctionType((INT, INT), INT),
            "divmod": FunctionType((INT, INT), TupleType((INT, INT))),
            # Sequence functions
            "len": FunctionType((ANY,), INT),
            "range": FunctionType((INT,), ListType(INT)),  # Simplified
            "enumerate": FunctionType((ListType(ANY),), ListType(TupleType((INT, ANY)))),
            "zip": FunctionType((ListType(ANY), ListType(ANY)), ListType(TupleType((ANY, ANY)))),
            "map": FunctionType((FunctionType((ANY,), ANY), ListType(ANY)), ListType(ANY)),
            "filter": FunctionType((FunctionType((ANY,), BOOL), ListType(ANY)), ListType(ANY)),
            "sorted": FunctionType((ListType(ANY),), ListType(ANY)),
            "reversed": FunctionType((ListType(ANY),), ListType(ANY)),
            # String functions
            "repr": FunctionType((ANY,), STR),
            "ascii": FunctionType((ANY,), STR),
            "chr": FunctionType((INT,), STR),
            "ord": FunctionType((STR,), INT),
            "format": FunctionType((ANY,), STR),
            # I/O functions
            "print": FunctionType((ANY,), NONE),
            "input": FunctionType((STR,), STR),
            "open": FunctionType((STR,), ANY),  # Simplified
            # Introspection
            "type": FunctionType((ANY,), ClassType("type", ())),
            "isinstance": FunctionType((ANY, ANY), BOOL),
            "issubclass": FunctionType((ANY, ANY), BOOL),
            "hasattr": FunctionType((ANY, STR), BOOL),
            "getattr": FunctionType((ANY, STR), ANY),
            "setattr": FunctionType((ANY, STR, ANY), NONE),
            "delattr": FunctionType((ANY, STR), NONE),
            "callable": FunctionType((ANY,), BOOL),
            # Object operations
            "id": FunctionType((ANY,), INT),
            "hash": FunctionType((ANY,), INT),
            "dir": FunctionType((ANY,), ListType(STR)),
            "vars": FunctionType((ANY,), DictType(STR, ANY)),
            # Boolean operations
            "all": FunctionType((ListType(ANY),), BOOL),
            "any": FunctionType((ListType(ANY),), BOOL),
        }

    def parse_type_annotation(self, annotation: str) -> Type:
        """Parse a Python type annotation string.

        Supports:
        - Simple types: int, str, bool, float, None
        - Generic types: List[int], Dict[str, int]
        - Union types: Union[int, str], int | str
        - Optional types: Optional[int]
        - Callable types: Callable[[int], str]
        - Type variables: T, TypeVar

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

        # Check for union syntax with |
        if "|" in s and not self._is_inside_brackets(s, s.index("|")):
            parts = self._split_union(s)
            types = frozenset(self._parse_type(p.strip()) for p in parts)
            if len(types) == 1:
                return next(iter(types))
            return UnionType(types)

        # Check for generic types with brackets
        if "[" in s:
            return self._parse_generic(s)

        # Check builtins
        if s in self._builtin_types:
            return self._builtin_types[s]

        # Check for None
        if s == "None" or s == "NoneType":
            return NONE

        # Check for type variable syntax ('T, 'U, etc.)
        if s.startswith("'") or (len(s) <= 2 and s.isupper()):
            return TypeVar(s.lstrip("'"))

        # Assume it's a class type
        return ClassType(s, ())

    def _parse_generic(self, s: str) -> Type:
        """Parse a generic type like List[int] or Dict[str, int]."""
        bracket_pos = s.index("[")
        base = s[:bracket_pos].strip()
        args_str = s[bracket_pos + 1 : -1].strip()

        # Handle special cases
        if base in ("List", "list"):
            elem_type = self._parse_type(args_str)
            return ListType(elem_type)

        if base in ("Set", "set", "FrozenSet", "frozenset"):
            elem_type = self._parse_type(args_str)
            return SetType(elem_type)

        if base in ("Dict", "dict"):
            args = self._split_args(args_str)
            if len(args) != 2:
                raise TypeParseError(s, "Dict requires exactly 2 type arguments")
            key_type = self._parse_type(args[0])
            value_type = self._parse_type(args[1])
            return DictType(key_type, value_type)

        if base in ("Tuple", "tuple"):
            if not args_str:
                return TupleType(())
            # Handle Tuple[()] syntax for empty tuples
            if args_str.strip() == "()":
                return TupleType(())
            args = self._split_args(args_str)
            # Handle Tuple[int, ...] syntax
            if len(args) == 2 and args[1].strip() == "...":
                elem_type = self._parse_type(args[0])
                return TupleType((elem_type,), is_variadic=True)
            elem_types = tuple(self._parse_type(a) for a in args)
            return TupleType(elem_types)

        if base == "Optional":
            inner_type = self._parse_type(args_str)
            return OptionalType(inner_type)

        if base == "Union":
            args = self._split_args(args_str)
            types = frozenset(self._parse_type(a) for a in args)
            if len(types) == 1:
                return next(iter(types))
            return UnionType(types)

        if base == "Callable":
            return self._parse_callable(args_str)

        if base == "Type":
            inner_type = self._parse_type(args_str)
            return ClassType("type", (inner_type,))

        # Generic class type
        args = self._split_args(args_str)
        type_args = tuple(self._parse_type(a) for a in args)
        return ClassType(base, type_args)

    def _parse_callable(self, args_str: str) -> FunctionType:
        """Parse a Callable type like Callable[[int, str], bool]."""
        # Find the split between params and return type
        # Format: Callable[[params], return_type]
        if not args_str.startswith("["):
            # Callable[..., ReturnType] or similar
            args = self._split_args(args_str)
            if len(args) == 2 and args[0] == "...":
                return_type = self._parse_type(args[1])
                return FunctionType((), return_type)
            raise TypeParseError(args_str, "Invalid Callable syntax")

        # Find matching bracket for params
        bracket_count = 0
        params_end = 0
        for i, c in enumerate(args_str):
            if c == "[":
                bracket_count += 1
            elif c == "]":
                bracket_count -= 1
                if bracket_count == 0:
                    params_end = i
                    break

        params_str = args_str[1:params_end]
        rest = args_str[params_end + 1 :].strip()
        if rest.startswith(","):
            rest = rest[1:].strip()

        # Parse parameter types
        if params_str:
            param_args = self._split_args(params_str)
            param_types = tuple(self._parse_type(a) for a in param_args)
        else:
            param_types = ()

        # Parse return type
        return_type = self._parse_type(rest) if rest else ANY

        return FunctionType(param_types, return_type)

    def _split_args(self, s: str) -> List[str]:
        """Split type arguments by comma, respecting nested brackets."""
        args = []
        current = []
        bracket_count = 0

        for c in s:
            if c == "[":
                bracket_count += 1
                current.append(c)
            elif c == "]":
                bracket_count -= 1
                current.append(c)
            elif c == "," and bracket_count == 0:
                args.append("".join(current).strip())
                current = []
            else:
                current.append(c)

        if current:
            args.append("".join(current).strip())

        return [a for a in args if a]

    def _split_union(self, s: str) -> List[str]:
        """Split union type by |, respecting nested brackets."""
        parts = []
        current = []
        bracket_count = 0

        for c in s:
            if c == "[":
                bracket_count += 1
                current.append(c)
            elif c == "]":
                bracket_count -= 1
                current.append(c)
            elif c == "|" and bracket_count == 0:
                parts.append("".join(current).strip())
                current = []
            else:
                current.append(c)

        if current:
            parts.append("".join(current).strip())

        return [p for p in parts if p]

    def _is_inside_brackets(self, s: str, pos: int) -> bool:
        """Check if position is inside brackets."""
        count = 0
        for i in range(pos):
            if s[i] == "[":
                count += 1
            elif s[i] == "]":
                count -= 1
        return count > 0

    def infer_literal_type(self, literal: LiteralInfo) -> Type:
        """Infer the type of a Python literal."""
        if literal.kind == LiteralKind.INTEGER:
            return INT
        elif literal.kind == LiteralKind.FLOAT:
            return FLOAT
        elif literal.kind == LiteralKind.STRING:
            return STR
        elif literal.kind == LiteralKind.BOOLEAN:
            return BOOL
        elif literal.kind == LiteralKind.NONE:
            return NONE
        elif literal.kind == LiteralKind.LIST:
            return ListType(ANY)  # Would need element types for more precision
        elif literal.kind == LiteralKind.DICT:
            return DictType(ANY, ANY)
        elif literal.kind == LiteralKind.TUPLE:
            return TupleType(())
        elif literal.kind == LiteralKind.SET:
            return SetType(ANY)
        else:
            return ANY

    def get_builtin_types(self) -> Dict[str, Type]:
        """Return Python builtin types."""
        return self._builtin_types.copy()

    def get_builtin_functions(self) -> Dict[str, FunctionType]:
        """Return Python builtin function signatures."""
        return self._builtin_functions.copy()

    def check_assignable(self, source: Type, target: Type) -> bool:
        """Check if source type is assignable to target type.

        Python assignability rules:
        - Any is assignable to/from anything
        - Never is assignable to anything
        - None is only assignable to None or Optional[T]
        - Subtypes are assignable to supertypes
        - Union member types are assignable to the union
        - Generic types are covariant by default
        """
        # Handle special types
        if isinstance(target, AnyType) or isinstance(source, AnyType):
            return True

        if isinstance(source, NeverType):
            return True

        if isinstance(target, NeverType):
            return False

        # Handle holes (holes accept anything)
        if isinstance(source, HoleType) or isinstance(target, HoleType):
            return True

        # Same type
        if source == target:
            return True

        # None handling
        if isinstance(source, PrimitiveType) and source.name == "none":
            # None is assignable to Optional (union with None) or unions containing None
            if isinstance(target, UnionType):
                return NONE in target.members
            return False

        # Union target: source must be assignable to at least one member
        if isinstance(target, UnionType):
            return any(self.check_assignable(source, t) for t in target.members)

        # Union source: all members must be assignable to target
        if isinstance(source, UnionType):
            return all(self.check_assignable(t, target) for t in source.members)

        # List assignability (covariant)
        if isinstance(source, ListType) and isinstance(target, ListType):
            return self.check_assignable(source.element, target.element)

        # Set assignability (covariant)
        if isinstance(source, SetType) and isinstance(target, SetType):
            return self.check_assignable(source.element, target.element)

        # Dict assignability (covariant in both)
        if isinstance(source, DictType) and isinstance(target, DictType):
            return (
                self.check_assignable(source.key, target.key) and
                self.check_assignable(source.value, target.value)
            )

        # Tuple assignability
        if isinstance(source, TupleType) and isinstance(target, TupleType):
            if len(source.elements) != len(target.elements):
                return False
            return all(
                self.check_assignable(s, t)
                for s, t in zip(source.elements, target.elements)
            )

        # Function assignability (contravariant in params, covariant in return)
        if isinstance(source, FunctionType) and isinstance(target, FunctionType):
            if len(source.params) != len(target.params):
                return False
            # Contravariant in parameters
            if not all(
                self.check_assignable(t, s)
                for s, t in zip(source.params, target.params)
            ):
                return False
            # Covariant in return
            return self.check_assignable(source.returns, target.returns)

        # Class type assignability
        if isinstance(source, ClassType) and isinstance(target, ClassType):
            if source.name != target.name:
                # Check for object (supertype of all)
                if target.name == "object":
                    return True
                return False
            # Same class - check type args
            if len(source.type_args) != len(target.type_args):
                return False
            # Covariant by default
            return all(
                self.check_assignable(s, t)
                for s, t in zip(source.type_args, target.type_args)
            )

        # Numeric hierarchy: int -> float
        if isinstance(source, PrimitiveType) and isinstance(target, PrimitiveType):
            if source.name == "int" and target.name == "float":
                return True
            if source.name == "bool" and target.name in ("int", "float"):
                return True

        return False

    def format_type(self, typ: Type) -> str:
        """Format a type as Python syntax."""
        if isinstance(typ, PrimitiveType):
            name = typ.name
            if name == "none":
                return "None"
            return name

        if isinstance(typ, AnyType):
            return "Any"

        if isinstance(typ, NeverType):
            return "Never"

        if isinstance(typ, HoleType):
            return f"?{typ.hole_id}"

        if isinstance(typ, TypeVar):
            return typ.name

        if isinstance(typ, ListType):
            elem = self.format_type(typ.element)
            return f"list[{elem}]"

        if isinstance(typ, SetType):
            elem = self.format_type(typ.element)
            return f"set[{elem}]"

        if isinstance(typ, DictType):
            key = self.format_type(typ.key)
            val = self.format_type(typ.value)
            return f"dict[{key}, {val}]"

        if isinstance(typ, TupleType):
            if not typ.elements:
                return "tuple[()]"
            elems = ", ".join(self.format_type(t) for t in typ.elements)
            return f"tuple[{elems}]"

        if isinstance(typ, UnionType):
            # Check if it's Optional (union with exactly one non-None type)
            if is_optional_type(typ):
                inner = get_optional_inner(typ)
                if not isinstance(inner, UnionType):
                    return f"{self.format_type(inner)} | None"
            parts = sorted(self.format_type(t) for t in typ.members)
            return " | ".join(parts)

        if isinstance(typ, FunctionType):
            params = ", ".join(self.format_type(p) for p in typ.params)
            ret = self.format_type(typ.returns)
            return f"Callable[[{params}], {ret}]"

        if isinstance(typ, ClassType):
            if typ.type_args:
                args = ", ".join(self.format_type(a) for a in typ.type_args)
                return f"{typ.name}[{args}]"
            return typ.name

        return str(typ)

    def get_common_imports(self) -> Dict[str, Type]:
        """Return types commonly imported in Python."""
        return {
            # typing module
            "typing.List": ListType(ANY),
            "typing.Dict": DictType(ANY, ANY),
            "typing.Set": SetType(ANY),
            "typing.Tuple": TupleType(()),
            "typing.Optional": OptionalType(ANY),
            "typing.Union": UnionType(frozenset()),
            "typing.Callable": FunctionType((), ANY),
            "typing.Any": ANY,
            "typing.TypeVar": TypeVar("T"),
            # collections
            "collections.defaultdict": DictType(ANY, ANY),
            "collections.OrderedDict": DictType(ANY, ANY),
            "collections.Counter": DictType(ANY, INT),
            "collections.deque": ListType(ANY),
            # pathlib
            "pathlib.Path": ClassType("Path", ()),
            # datetime
            "datetime.datetime": ClassType("datetime", ()),
            "datetime.date": ClassType("date", ()),
            "datetime.time": ClassType("time", ()),
            "datetime.timedelta": ClassType("timedelta", ()),
        }

    def normalize_type(self, typ: Type) -> Type:
        """Normalize Python type to canonical form."""
        # Flatten nested unions
        if isinstance(typ, UnionType):
            flattened: Set[Type] = set()
            for t in typ.members:
                normalized = self.normalize_type(t)
                if isinstance(normalized, UnionType):
                    flattened.update(normalized.members)
                else:
                    flattened.add(normalized)
            if len(flattened) == 1:
                return next(iter(flattened))
            return UnionType(frozenset(flattened))

        # Note: OptionalType is a function that creates UnionType,
        # so it's already handled by the UnionType case above

        return typ
