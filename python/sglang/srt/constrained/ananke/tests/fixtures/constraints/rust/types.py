# Copyright Rand Arete @ Ananke 2025
# Licensed under the Apache License, Version 2.0
"""Rust type constraint examples for Ananke.

This module contains realistic examples of type-level constraints in Rust,
demonstrating ownership, lifetimes, trait bounds, and type inference.
"""

from __future__ import annotations

from typing import List

try:
    from ..base import ConstraintExample
    from ....spec.constraint_spec import (
        ConstraintSpec,
        TypeBinding,
        SemanticConstraint,
    )
except ImportError:
    from tests.fixtures.constraints.base import ConstraintExample
    from spec.constraint_spec import (
        ConstraintSpec,
        TypeBinding,
        SemanticConstraint,
    )

# =============================================================================
# Type Constraint Examples
# =============================================================================

RUST_TYPE_001 = ConstraintExample(
    id="rust-types-001",
    name="Ownership Transfer with Box<T>",
    description="Constraint generation for move semantics with heap-allocated Box<T>",
    scenario=(
        "Developer implementing a binary tree where nodes own their children. "
        "The function must return a boxed node, transferring ownership to caller. "
        "The constraint ensures generated code respects move semantics and heap allocation."
    ),
    prompt="""Create a boxed tree node. The node should be heap-allocated using Box::new.
Transfer ownership to the caller - don't return a reference.

fn create_node(value: i32, left: Option<Box<Node>>, right: Option<Box<Node>>) -> Box<Node> {
    """,
    spec=ConstraintSpec(
        language="rust",
        expected_type="Box<Node>",
        type_bindings=[
            TypeBinding(
                name="value",
                type_expr="i32",
                scope="parameter",
                mutable=False,
            ),
            TypeBinding(
                name="left",
                type_expr="Option<Box<Node>>",
                scope="parameter",
                mutable=False,
            ),
            TypeBinding(
                name="right",
                type_expr="Option<Box<Node>>",
                scope="parameter",
                mutable=False,
            ),
        ],
        semantic_constraints=[
            SemanticConstraint(
                kind="postcondition",
                expression="result.is_valid_heap_allocation()",
                scope="create_node",
                variables=("result",),
            ),
        ],
        ebnf=r'''
root ::= box_shorthand | box_none | box_explicit
box_shorthand ::= "Box::new(Node { value, left, right })"
box_none ::= "Box::new(Node { value, left: None, right: None })"
box_explicit ::= "Box::new(Node { value: value, left: left, right: right })"
''',
    ),
    expected_effect=(
        "Masks tokens that would violate ownership rules or produce stack-allocated "
        "structures. Ensures Box::new wraps the value and enforces move semantics."
    ),
    valid_outputs=[
        "Box::new(Node { value, left, right })",
        "Box::new(Node { value, left: None, right: None })",
        "Box::new(Node { value: value, left: left, right: right })",
    ],
    invalid_outputs=[
        "Node { value, left, right }",  # Missing Box, stack allocated
        "&Node { value, left, right }",  # Reference, not owned
        "Box::from_raw(ptr)",  # Unsafe, unverified provenance
    ],
    tags=["types", "ownership", "heap", "box", "move"],
    language="rust",
    domain="types",
)

RUST_TYPE_002 = ConstraintExample(
    id="rust-types-002",
    name="Lifetime-Bounded Reference",
    description="Constraint generation for explicit lifetime parameters in function signatures",
    scenario=(
        "Developer writing a function that returns a reference to data within a struct. "
        "The lifetime 'a must be explicit to tie the output lifetime to input. "
        "The constraint ensures the generated signature prevents dangling references."
    ),
    prompt="""Return a reference to the name field from a Data struct. The lifetime must be tied to the input.
Don't allocate a new String - return a reference with the same lifetime as the input.

fn get_name<'a>(data: &'a Data) -> &'a str {
    """,
    spec=ConstraintSpec(
        language="rust",
        expected_type="&'a str",
        type_bindings=[
            TypeBinding(
                name="data",
                type_expr="&'a Data",
                scope="parameter",
                mutable=False,
            ),
        ],
        semantic_constraints=[
            SemanticConstraint(
                kind="invariant",
                expression="result.lifetime <= data.lifetime",
                scope="get_name",
                variables=("result", "data"),
            ),
        ],
        ebnf=r'''
root ::= ref_direct | ref_slice | as_str
ref_direct ::= "&data.name"
ref_slice ::= "&data.name[..]"
as_str ::= "data.name.as_str()"
''',
    ),
    expected_effect=(
        "Masks tokens producing owned String or 'static lifetimes that would outlive "
        "the borrow. Enforces that returned reference lifetime is tied to input 'a."
    ),
    valid_outputs=[
        "&data.name",
        "&data.name[..]",
        "data.name.as_str()",
    ],
    invalid_outputs=[
        "data.name.clone()",  # Owned String, not reference
        "data.name.to_string()",  # Owned allocation
        "&STATIC_NAME",  # 'static lifetime, not 'a
    ],
    tags=["types", "lifetimes", "borrowing", "references"],
    language="rust",
    domain="types",
)

RUST_TYPE_003 = ConstraintExample(
    id="rust-types-003",
    name="Trait Bounds with Multiple Constraints",
    description="Constraint generation for generic functions with trait bound requirements",
    scenario=(
        "Developer implementing a generic sort-and-debug function. The type T must "
        "implement Ord for comparison, Debug for printing, and Clone for duplication. "
        "The constraint ensures all trait bounds are satisfied in generated code."
    ),
    prompt="""Sort a slice in place and return cloned items as a Vec. T implements Ord + Clone.
Use methods that respect the trait bounds - sort() needs Ord, cloned() needs Clone.

fn sort_and_clone<T: Ord + Clone>(items: &mut [T]) -> Vec<T> {
    """,
    spec=ConstraintSpec(
        language="rust",
        type_bindings=[
            TypeBinding(
                name="items",
                type_expr="&mut [T]",
                scope="parameter",
                mutable=True,
            ),
        ],
        semantic_constraints=[
            SemanticConstraint(
                kind="precondition",
                expression="T: Ord + Debug + Clone",
                scope="sort_and_debug",
                variables=("T",),
            ),
        ],
        ebnf=r'''
root ::= sort_cloned | sort_unstable | sort_by_map
sort_cloned ::= "items.sort();\nitems.iter().cloned().collect()"
sort_unstable ::= "items.sort_unstable();\nitems.iter().cloned().collect::<Vec<_>>()"
sort_by_map ::= "items.sort_by(|a, b| a.cmp(b));\nitems.iter().map(|x| x.clone()).collect()"
''',
    ),
    expected_effect=(
        "Masks tokens that call methods not available on T given the trait bounds. "
        "Ensures sort() is called (Ord), iter() works (slice), and cloned() succeeds (Clone)."
    ),
    valid_outputs=[
        "items.sort();\nitems.iter().cloned().collect()",
        "items.sort_unstable();\nitems.iter().cloned().collect::<Vec<_>>()",
        "items.sort_by(|a, b| a.cmp(b));\nitems.iter().map(|x| x.clone()).collect()",
    ],
    invalid_outputs=[
        "items.sort();\nitems.to_vec()",  # to_vec requires Clone but doesn't use trait bound explicitly
        "items.sort();\nitems.iter().collect()",  # Missing cloned(), produces &T not T
        "items.hash();\nitems.clone()",  # Hash not in bounds
    ],
    tags=["types", "generics", "trait-bounds", "constraints"],
    language="rust",
    domain="types",
)

# =============================================================================
# Exports
# =============================================================================

RUST_TYPE_EXAMPLES: List[ConstraintExample] = [
    RUST_TYPE_001,
    RUST_TYPE_002,
    RUST_TYPE_003,
]

__all__ = ["RUST_TYPE_EXAMPLES"]
