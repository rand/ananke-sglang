# Copyright Rand Arete @ Ananke 2025
# Licensed under the Apache License, Version 2.0
"""Type constraint examples for Python.

This module contains realistic examples of type-level constraints that
demonstrate how Ananke's TypeDomain masks tokens to enforce type correctness
during code generation.
"""

from __future__ import annotations

try:
    from ..base import ConstraintExample
    from .....spec.constraint_spec import (
        ConstraintSpec,
        TypeBinding,
        FunctionSignature,
    )
except ImportError:
    from tests.fixtures.constraints.base import ConstraintExample
    from spec.constraint_spec import (
        ConstraintSpec,
        TypeBinding,
        FunctionSignature,
    )

PYTHON_TYPE_EXAMPLES = [
    ConstraintExample(
        id="py-types-001",
        name="Generic Filter Function",
        description="Filter list while preserving type parameter T",
        scenario=(
            "Developer writing a generic filter function that must return the same "
            "generic type as the input. The function takes List[T] and must return "
            "List[T], not a set, tuple, or dict."
        ),
        prompt="""I need a generic filter function that preserves the input list's type parameter.
It should take a List[T] and a predicate function, returning a filtered List[T] (not set/tuple/dict).

def filter_items(items: List[T], pred: Callable[[T], bool]) -> List[T]:
    """,
        spec=ConstraintSpec(
            language="python",
            expected_type="List[T]",
            # Regex enforces list construction: list literal [...] or list() call
            # Blocks set {}, tuple(), dict {k:v}, and bare iterators
            # Allows optional trailing comments (# type: ignore, etc.)
            regex=r"^return\s+(\[.*\]|list\s*\(.*\))(\s*#.*)?$",
            type_bindings=[
                TypeBinding(name="items", type_expr="List[T]", scope="parameter"),
                TypeBinding(
                    name="pred", type_expr="Callable[[T], bool]", scope="parameter"
                ),
            ],
            function_signatures=[
                FunctionSignature(
                    name="filter_items",
                    params=(
                        TypeBinding(name="items", type_expr="List[T]"),
                        TypeBinding(name="pred", type_expr="Callable[[T], bool]"),
                    ),
                    return_type="List[T]",
                    type_params=("T",),
                )
            ],
        ),
        expected_effect=(
            "Masks tokens that would produce non-List return types. Specifically "
            "blocks set comprehensions {}, dict comprehensions, tuple(), set(), "
            "and any expression that doesn't unify with List[T]."
        ),
        valid_outputs=[
            "return [x for x in items if pred(x)]",
            "return list(filter(pred, items))",
            "return [item for item in items if pred(item)]",
        ],
        invalid_outputs=[
            "return {x for x in items if pred(x)}",  # set, not list
            "return tuple(filter(pred, items))",  # tuple, not list
            "return {x: x for x in items if pred(x)}",  # dict, not list
            "return filter(pred, items)",  # iterator, not list
        ],
        tags=["types", "generics", "inference", "collections"],
        language="python",
        domain="types",
    ),
    ConstraintExample(
        id="py-types-002",
        name="Protocol Implementation",
        description="Implement duck-typed protocol with required methods",
        scenario=(
            "Developer implementing a Drawable protocol that requires both draw() "
            "and get_bounds() methods. The generator must complete a class that "
            "satisfies the protocol's structural requirements."
        ),
        prompt="""I'm implementing a Circle class that satisfies a Drawable protocol.
The protocol requires draw() returning None and get_bounds() returning Tuple[float, float, float, float].

class Circle:
    def __init__(self, radius: float):
        self.radius = radius

    """,
        spec=ConstraintSpec(
            language="python",
            expected_type="Drawable",
            # EBNF enforces method signatures matching the Drawable protocol
            ebnf=r"""
                root ::= method_def
                method_def ::= draw_method | get_bounds_method
                draw_method ::= "def draw(self) -> None:" body
                get_bounds_method ::= "def get_bounds(self) -> Tuple[float, float, float, float]:" body
                body ::= ws statement+
                statement ::= [^\n]+ "\n"?
                ws ::= [ \t\n]*
            """,
            type_bindings=[
                TypeBinding(name="self", type_expr="Circle", scope="local"),
                TypeBinding(name="radius", type_expr="float", scope="local"),
            ],
            type_aliases={
                "Drawable": "Protocol[draw: Callable[[], None], get_bounds: Callable[[], Tuple[float, float, float, float]]]",
            },
        ),
        expected_effect=(
            "Masks tokens that would create methods not matching the protocol. "
            "Ensures draw() has no parameters and returns None, and get_bounds() "
            "returns a 4-tuple of floats representing (x, y, width, height)."
        ),
        valid_outputs=[
            "def draw(self) -> None:\n    print(f'Circle at radius {self.radius}')",
            "def get_bounds(self) -> Tuple[float, float, float, float]:\n    return (0.0, 0.0, self.radius * 2, self.radius * 2)",
        ],
        invalid_outputs=[
            "def draw(self, canvas) -> None: pass",  # Extra parameter
            "def draw(self) -> str: return 'drawn'",  # Wrong return type
            "def get_bounds(self) -> Tuple[int, int]: return (0, 0)",  # Wrong tuple size and type
            "def render(self) -> None: pass",  # Wrong method name
        ],
        tags=["types", "protocols", "duck-typing", "structural"],
        language="python",
        domain="types",
    ),
    ConstraintExample(
        id="py-types-003",
        name="Type Narrowing with isinstance",
        description="Narrow union type based on isinstance check",
        scenario=(
            "Developer handling a Union[int, str, list] type and narrowing it "
            "with isinstance checks. After checking isinstance(value, int), "
            "the type domain should recognize that value is now definitively int."
        ),
        prompt="""I need to handle a value that could be int, str, or list. When it's an int,
I want to perform arithmetic operations on it (not string or list methods).

def process(value: Union[int, str, list]):
    """,
        spec=ConstraintSpec(
            language="python",
            # EBNF enforces isinstance check followed by int-only operations
            # Blocks string methods (.upper, .lower, etc) and list methods (.append, etc)
            ebnf=r"""
                root ::= isinstance_check body
                isinstance_check ::= "if isinstance(value, int):" ws
                body ::= "return " int_expr
                int_expr ::= "value" int_op | int_func "(" "value" ")"
                int_op ::= " * " number | " + " number | " - " number | " & " hex | " | " hex | " // " number | " % " number
                int_func ::= "abs" | "int" | "bin" | "hex" | "oct"
                number ::= [0-9]+
                hex ::= "0x" [0-9a-fA-F]+
                ws ::= [ \t\n]+
            """,
            type_bindings=[
                TypeBinding(
                    name="value", type_expr="Union[int, str, list]", scope="local"
                ),
            ],
            control_flow=None,  # Will be updated based on isinstance check
        ),
        expected_effect=(
            "After isinstance(value, int) check in the true branch, masks tokens "
            "that assume value could be str or list. Allows int-specific operations "
            "like arithmetic, bit operations, but blocks string methods like .upper() "
            "or list methods like .append()."
        ),
        valid_outputs=[
            "if isinstance(value, int):\n    return value * 2",
            "if isinstance(value, int):\n    return value & 0xFF",
            "if isinstance(value, int):\n    return abs(value)",
        ],
        invalid_outputs=[
            "if isinstance(value, int):\n    return value.upper()",  # str method
            "if isinstance(value, int):\n    return value.append(1)",  # list method
            "if isinstance(value, int):\n    return value + 'suffix'",  # Type error
        ],
        tags=["types", "narrowing", "unions", "control-flow"],
        language="python",
        domain="types",
    ),
]
