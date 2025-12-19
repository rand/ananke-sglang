# Copyright Rand Arete @ Ananke 2025
# Licensed under the Apache License, Version 2.0
"""Go type constraint examples.

Demonstrates type-level constraints specific to Go:
- Interface satisfaction through implicit implementation
- Channel direction constraints (send-only, receive-only)
- Generic type constraints with comparable and any
"""

from __future__ import annotations

from typing import List

try:
    from ..base import ConstraintExample
    from .....spec.constraint_spec import (
        ConstraintSpec,
        TypeBinding,
        FunctionSignature,
        SemanticConstraint,
    )
except ImportError:
    from tests.fixtures.constraints.base import ConstraintExample
    from spec.constraint_spec import (
        ConstraintSpec,
        TypeBinding,
        FunctionSignature,
        SemanticConstraint,
    )


GO_TYPE_EXAMPLES: List[ConstraintExample] = [
    ConstraintExample(
        id="go-types-001",
        name="Interface Satisfaction - Implicit Implementation",
        description="Implement io.Reader interface without explicit declaration",
        scenario=(
            "Developer implementing a custom Reader type. In Go, interfaces are "
            "satisfied implicitly - the type must have Read(p []byte) (n int, err error) "
            "method with exact signature matching io.Reader."
        ),
        prompt="""Implement io.Reader on a custom type. Go interfaces are implicit - just implement
the Read(p []byte) (n int, err error) method with exact signature. No explicit declaration needed.

type CustomReader struct { data []byte }

""",
        spec=ConstraintSpec(
            language="go",
            # Regex enforces io.Reader interface signature with exact Read method
            regex=r"^func\s+\([^)]+\)\s+Read\s*\(\s*p\s+\[\]byte\s*\)\s*\(",
            ebnf=r'''
root ::= named_return | tuple_return
named_return ::= "func (r *CustomReader) Read(p []byte) (n int, err error) {" nl "\tn = copy(p, r.data)" nl "\tif n == 0 {" nl "\t\terr = io.EOF" nl "\t}" nl "\treturn" nl "}"
tuple_return ::= "func (r *CustomReader) Read(p []byte) (int, error) {" nl "\tif len(r.data) == 0 {" nl "\t\treturn 0, io.EOF" nl "\t}" nl "\tn := copy(p, r.data)" nl "\tr.data = r.data[n:]" nl "\treturn n, nil" nl "}"
nl ::= "\n"
''',
            expected_type="io.Reader",
            type_bindings=[
                TypeBinding(name="self", type_expr="*CustomReader", scope="local"),
                TypeBinding(name="data", type_expr="[]byte", scope="local"),
            ],
            type_aliases={
                "io.Reader": "interface { Read(p []byte) (n int, err error) }",
            },
            semantic_constraints=[
                SemanticConstraint(
                    kind="postcondition",
                    expression="n >= 0 && n <= len(p)",
                    scope="Read",
                    variables=("n", "p"),
                ),
                SemanticConstraint(
                    kind="postcondition",
                    expression="(n == 0) == (err != nil)",
                    scope="Read",
                    variables=("n", "err"),
                ),
            ],
        ),
        expected_effect=(
            "Masks tokens that would create a Read method with wrong signature. "
            "Ensures exact match: Read(p []byte) (n int, err error). "
            "Blocks incorrect parameter types, return types, or method names."
        ),
        valid_outputs=[
            "func (r *CustomReader) Read(p []byte) (n int, err error) {\n\tn = copy(p, r.data)\n\tif n == 0 {\n\t\terr = io.EOF\n\t}\n\treturn\n}",
            "func (r *CustomReader) Read(p []byte) (int, error) {\n\tif len(r.data) == 0 {\n\t\treturn 0, io.EOF\n\t}\n\tn := copy(p, r.data)\n\tr.data = r.data[n:]\n\treturn n, nil\n}",
        ],
        invalid_outputs=[
            "func (r *CustomReader) Read(p []byte) int { ... }",  # Missing error return
            "func (r *CustomReader) Read(data []byte) (n int, err error) { ... }",  # Wrong parameter name breaks idiom
            "func (r *CustomReader) ReadData(p []byte) (int, error) { ... }",  # Wrong method name
            "func (r *CustomReader) Read(p string) (int, error) { ... }",  # Wrong parameter type
        ],
        tags=["types", "interfaces", "implicit", "io"],
        language="go",
        domain="types",
    ),
    ConstraintExample(
        id="go-types-002",
        name="Channel Direction - Send vs Receive",
        description="Enforce channel direction constraints for type safety",
        scenario=(
            "Developer writing producer/consumer pattern with directional channels. "
            "Send-only channels (chan<- T) can only be written to, receive-only "
            "channels (<-chan T) can only be read from. This prevents misuse."
        ),
        prompt="""Complete this Go producer function that sends integers to a channel:

func producer(out chan<- int) {
    """,
        spec=ConstraintSpec(
            language="go",
            # Regex enforces sending to out channel (the body after signature)
            regex=r"^(?:for|out\s*<-)",
            ebnf=r'''
root ::= producer_func | consumer_func | consumer_select
producer_func ::= "func producer(out chan<- int) {" nl "\tfor i := 0; i < 10; i++ {" nl "\t\tout <- i" nl "\t}" nl "\tclose(out)" nl "}"
consumer_func ::= "func consumer(in <-chan int) {" nl "\tfor val := range in {" nl "\t\tfmt.Println(val)" nl "\t}" nl "}"
consumer_select ::= "func consumer(in <-chan int) {" nl "\tselect {" nl "\tcase v := <-in:" nl "\t\tprocess(v)" nl "\tdefault:" nl "\t\treturn" nl "\t}" nl "}"
nl ::= "\n"
''',
            type_bindings=[
                TypeBinding(
                    name="out",
                    type_expr="chan<- int",
                    scope="parameter",
                    mutable=False,
                ),
                TypeBinding(
                    name="in",
                    type_expr="<-chan int",
                    scope="parameter",
                    mutable=False,
                ),
            ],
            function_signatures=[
                FunctionSignature(
                    name="producer",
                    params=(TypeBinding("out", "chan<- int"),),
                    return_type="void",
                ),
                FunctionSignature(
                    name="consumer",
                    params=(TypeBinding("in", "<-chan int"),),
                    return_type="void",
                ),
            ],
        ),
        expected_effect=(
            "Masks tokens attempting to receive from send-only channel or send to "
            "receive-only channel. Ensures 'out <- value' for chan<- and "
            "'value := <-in' for <-chan."
        ),
        valid_outputs=[
            "for i := 0; i < 10; i++ {\n\t\tout <- i\n\t}\n\tclose(out)\n}",
            "out <- 1\n\tout <- 2\n\tout <- 3\n\tclose(out)\n}",
            "for _, v := range values {\n\t\tout <- v\n\t}\n}",
        ],
        invalid_outputs=[
            "func producer(out chan<- int) { val := <-out }",  # Can't receive from send-only
            "func consumer(in <-chan int) { in <- 42 }",  # Can't send to receive-only
            "func producer(out chan<- int) { close(<-out) }",  # Type error
        ],
        tags=["types", "channels", "concurrency", "direction"],
        language="go",
        domain="types",
    ),
    ConstraintExample(
        id="go-types-003",
        name="Generic Constraints - comparable and any",
        description="Use Go 1.18+ generics with type parameter constraints",
        scenario=(
            "Developer writing generic functions with type constraints. "
            "'comparable' constraint allows == and != operations. "
            "'any' (interface{}) allows any type but no operations."
        ),
        prompt="""Write a generic Contains function using Go 1.18+ generics.
Use 'comparable' constraint to allow == comparison. Without it, you can't use ==.

""",
        spec=ConstraintSpec(
            language="go",
            # Regex enforces Go 1.18+ generic function with comparable constraint
            regex=r"^func\s+\w+\s*\[\s*T\s+comparable\s*\]\s*\(",
            ebnf=r'''
root ::= range_style | index_style
range_style ::= "func Contains[T comparable](slice []T, target T) bool {" nl "\tfor _, item := range slice {" nl "\t\tif item == target {" nl "\t\t\treturn true" nl "\t\t}" nl "\t}" nl "\treturn false" nl "}"
index_style ::= "func Contains[T comparable](slice []T, target T) bool {" nl "\tfor i := 0; i < len(slice); i++ {" nl "\t\tif slice[i] == target {" nl "\t\t\treturn true" nl "\t\t}" nl "\t}" nl "\treturn false" nl "}"
nl ::= "\n"
''',
            expected_type="bool",
            type_bindings=[
                TypeBinding(name="T", type_expr="comparable", scope="type_parameter"),
                TypeBinding(name="slice", type_expr="[]T", scope="parameter"),
                TypeBinding(name="target", type_expr="T", scope="parameter"),
            ],
            function_signatures=[
                FunctionSignature(
                    name="Contains",
                    params=(
                        TypeBinding("slice", "[]T"),
                        TypeBinding("target", "T"),
                    ),
                    return_type="bool",
                    type_params=("T",),
                ),
            ],
            semantic_constraints=[
                SemanticConstraint(
                    kind="precondition",
                    expression="comparable(T)",
                    scope="Contains",
                    variables=("T",),
                ),
            ],
        ),
        expected_effect=(
            "Masks tokens that don't respect comparable constraint. "
            "Allows == and != on T values. Blocks operations requiring "
            "numeric constraints like < or + on generic T."
        ),
        valid_outputs=[
            "func Contains[T comparable](slice []T, target T) bool {\n\tfor _, item := range slice {\n\t\tif item == target {\n\t\t\treturn true\n\t\t}\n\t}\n\treturn false\n}",
            "func Contains[T comparable](slice []T, target T) bool {\n\tfor i := 0; i < len(slice); i++ {\n\t\tif slice[i] == target {\n\t\t\treturn true\n\t\t}\n\t}\n\treturn false\n}",
        ],
        invalid_outputs=[
            "func Contains[T any](slice []T, target T) bool { if slice[0] == target { ... } }",  # 'any' doesn't support ==
            "func Contains[T comparable](slice []T, target T) bool { return slice[0] > target }",  # comparable doesn't support >
            "func Contains(slice []interface{}, target interface{}) bool { return slice[0] == target }",  # Missing generic constraint
        ],
        tags=["types", "generics", "constraints", "comparable"],
        language="go",
        domain="types",
    ),
]


__all__ = ["GO_TYPE_EXAMPLES"]
