# Copyright Rand Arete @ Ananke 2025
# Licensed under the Apache License, Version 2.0
"""Semantic constraint examples for Python.

This module contains realistic examples of semantic constraints that
demonstrate how Ananke's SemanticDomain uses SMT solvers to enforce
logical properties like preconditions, postconditions, and invariants.
"""

from __future__ import annotations

try:
    from ..base import ConstraintExample
    from .....spec.constraint_spec import (
        ConstraintSpec,
        SemanticConstraint,
        TypeBinding,
        FunctionSignature,
    )
except ImportError:
    from tests.fixtures.constraints.base import ConstraintExample
    from spec.constraint_spec import (
        ConstraintSpec,
        SemanticConstraint,
        TypeBinding,
        FunctionSignature,
    )

PYTHON_SEMANTICS_EXAMPLES = [
    ConstraintExample(
        id="py-semantics-001",
        name="Non-Negative Result Postcondition",
        description="Ensure function result is always non-negative",
        scenario=(
            "Developer writing an absolute value or distance function that must "
            "guarantee a non-negative result. The semantic domain verifies that "
            "all code paths produce result >= 0."
        ),
        prompt="""I need a distance function that always returns non-negative values.
Use abs(), sqrt(), or max(0, ...) to guarantee result >= 0.

def compute_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """,
        spec=ConstraintSpec(
            language="python",
            # Regex enforces non-negative result functions: abs(), sqrt(), max(0, ...)
            regex=r"^return\s+(?:abs|math\.sqrt|max\s*\(\s*0)",
            semantic_constraints=[
                SemanticConstraint(
                    kind="postcondition",
                    expression="result >= 0",
                    scope="compute_distance",
                    variables=("result",),
                )
            ],
            function_signatures=[
                FunctionSignature(
                    name="compute_distance",
                    params=(
                        TypeBinding(name="x1", type_expr="float"),
                        TypeBinding(name="y1", type_expr="float"),
                        TypeBinding(name="x2", type_expr="float"),
                        TypeBinding(name="y2", type_expr="float"),
                    ),
                    return_type="float",
                )
            ],
            type_bindings=[
                TypeBinding(name="x1", type_expr="float", scope="parameter"),
                TypeBinding(name="y1", type_expr="float", scope="parameter"),
                TypeBinding(name="x2", type_expr="float", scope="parameter"),
                TypeBinding(name="y2", type_expr="float", scope="parameter"),
            ],
        ),
        expected_effect=(
            "Masks expressions that could produce negative results. Blocks direct "
            "subtraction without abs(), allows sqrt() and abs() which guarantee "
            "non-negative outputs. Uses SMT solver to prove result >= 0."
        ),
        valid_outputs=[
            "return abs(x2 - x1) + abs(y2 - y1)",
            "return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)",
            "return max(0, x2 - x1)",
        ],
        invalid_outputs=[
            "return x2 - x1",  # Could be negative
            "return (x2 - x1) + (y2 - y1)",  # Could be negative
            "return -abs(x2 - x1)",  # Explicitly negative
        ],
        tags=["semantics", "postcondition", "smt", "verification"],
        language="python",
        domain="semantics",
    ),
    ConstraintExample(
        id="py-semantics-002",
        name="Array Bounds Check Precondition",
        description="Verify index is within array bounds before access",
        scenario=(
            "Developer writing array indexing code that must verify bounds before "
            "access. The semantic domain checks that 0 <= index < len(array) is "
            "proven before any array[index] access."
        ),
        prompt="""I need safe array access that checks bounds before indexing.
Verify 0 <= index < len(items) before accessing items[index].

def get_item(items: List[int], index: int) -> Optional[int]:
    """,
        spec=ConstraintSpec(
            language="python",
            # EBNF enforces bounds check before array access
            ebnf=r'''
root ::= (if_positive_block | if_negative_block | assert_block) ws fallback
if_positive_block ::= "if " positive_condition ":" ws "    return items[index]"
if_negative_block ::= "if " negative_condition ":" ws "    raise IndexError"
assert_block ::= "assert " positive_condition ", 'Index out of bounds'"
positive_condition ::= "0 <= index < len(items)"
negative_condition ::= "index < 0 or index >= len(items)"
fallback ::= "return " ("items[index]" | "None")
ws ::= [ \t\n]+
''',
            semantic_constraints=[
                SemanticConstraint(
                    kind="precondition",
                    expression="0 <= index < len(items)",
                    scope="get_item",
                    variables=("index", "items"),
                )
            ],
            type_bindings=[
                TypeBinding(name="items", type_expr="List[int]", scope="parameter"),
                TypeBinding(name="index", type_expr="int", scope="parameter"),
            ],
        ),
        expected_effect=(
            "Masks direct array access items[index] unless bounds check is proven. "
            "Requires explicit check like 'if 0 <= index < len(items):' or "
            "'assert 0 <= index < len(items)' before access. Uses SMT to track "
            "bounds in control flow."
        ),
        valid_outputs=[
            "if 0 <= index < len(items):\n    return items[index]\nreturn None",
            "assert 0 <= index < len(items), 'Index out of bounds'\nreturn items[index]",
            "if index < 0 or index >= len(items):\n    raise IndexError\nreturn items[index]",
        ],
        invalid_outputs=[
            "return items[index]",  # No bounds check
            "if index >= 0:\n    return items[index]",  # Incomplete check (missing upper bound)
            "return items[min(index, 100)]",  # Hardcoded bound doesn't prove safety
        ],
        tags=["semantics", "precondition", "bounds-check", "safety"],
        language="python",
        domain="semantics",
    ),
    ConstraintExample(
        id="py-semantics-003",
        name="Class Invariant Balance Non-Negative",
        description="Maintain class invariant that balance >= 0",
        scenario=(
            "Developer writing a BankAccount class where the invariant 'balance >= 0' "
            "must hold after every method. The withdraw() method must check that "
            "sufficient funds exist before deducting."
        ),
        prompt="""I'm implementing a BankAccount withdraw method. The balance must never go negative.
Check that sufficient funds exist before deducting the amount.

class BankAccount:
    def __init__(self, balance: float):
        self.balance = balance

    """,
        spec=ConstraintSpec(
            language="python",
            # EBNF enforces invariant check before balance modification
            ebnf=r'''
root ::= method_sig body1 | method_sig body2
method_sig ::= "def withdraw(self, amount: float) -> bool:" | "def withdraw(self, amount: float) -> None:"
body1 ::= "\n    if amount <= self.balance:\n        self.balance -= amount\n        return True\n    return False"
body2 ::= "\n    if amount > self.balance:\n        raise InsufficientFundsError\n    self.balance -= amount"
''',
            semantic_constraints=[
                SemanticConstraint(
                    kind="invariant",
                    expression="self.balance >= 0",
                    scope="BankAccount",
                    variables=("self.balance",),
                )
            ],
            type_bindings=[
                TypeBinding(name="self.balance", type_expr="float", scope="class:BankAccount"),
                TypeBinding(name="amount", type_expr="float", scope="parameter"),
            ],
        ),
        expected_effect=(
            "Masks operations that could violate the invariant. In withdraw(), "
            "blocks 'self.balance -= amount' unless proven that amount <= self.balance. "
            "Requires explicit check or exception raising when invariant would break."
        ),
        valid_outputs=[
            "def withdraw(self, amount: float) -> bool:\n    if amount <= self.balance:\n        self.balance -= amount\n        return True\n    return False",
            "def withdraw(self, amount: float) -> None:\n    if amount > self.balance:\n        raise InsufficientFundsError\n    self.balance -= amount",
        ],
        invalid_outputs=[
            "def withdraw(self, amount: float) -> None:\n    self.balance -= amount",  # No check
            "def withdraw(self, amount: float) -> None:\n    if amount > 0:\n        self.balance -= amount",  # Insufficient check
        ],
        tags=["semantics", "invariant", "class", "state"],
        language="python",
        domain="semantics",
    ),
]
