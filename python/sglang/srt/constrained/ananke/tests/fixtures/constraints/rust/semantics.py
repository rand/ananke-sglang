# Copyright Rand Arete @ Ananke 2025
# Licensed under the Apache License, Version 2.0
"""Rust semantic constraint examples for Ananke.

This module contains realistic examples of semantic constraints in Rust,
demonstrating panic-freedom, memory safety invariants, and unsafe block requirements.
"""

from __future__ import annotations

from typing import List

try:
    from ..base import ConstraintExample
    from ....spec.constraint_spec import (
        ConstraintSpec,
        TypeBinding,
        SemanticConstraint,
        ControlFlowContext,
    )
except ImportError:
    from tests.fixtures.constraints.base import ConstraintExample
    from spec.constraint_spec import (
        ConstraintSpec,
        TypeBinding,
        SemanticConstraint,
        ControlFlowContext,
    )

# =============================================================================
# Semantic Constraint Examples
# =============================================================================

RUST_SEMANTIC_001 = ConstraintExample(
    id="rust-semantics-001",
    name="No-Panic Guarantee",
    description="Constraint generation forbidding panic-inducing operations",
    scenario=(
        "Developer implementing code for a safety-critical system where panics are forbidden. "
        "All operations must return Result or Option instead of panicking on failure. "
        "The constraint masks unwrap(), expect(), panic!(), and unchecked indexing."
    ),
    prompt="""Get an item from a Vec by key. Don't use operations that can panic.
No unwrap(), expect(), or [] indexing - use get() which returns Option.

fn get_item<T>(items: &Vec<T>, key: usize) -> Option<&T> {
    """,
    spec=ConstraintSpec(
        language="rust",
        type_bindings=[
            TypeBinding(name="items", type_expr="Vec<T>", scope="parameter"),
            TypeBinding(name="key", type_expr="usize", scope="parameter"),
        ],
        semantic_constraints=[
            SemanticConstraint(
                kind="invariant",
                expression="!contains_panic_sources()",
                scope="get_item",
                variables=(),
            ),
        ],
        ebnf=r'''
root ::= get_key | get_ok_or | get_copied | get_first
get_key ::= "items.get(key)"
get_ok_or ::= "items.get(key).ok_or(Error::NotFound)"
get_copied ::= "items.get(key).copied()"
get_first ::= "items.first()"
''',
    ),
    expected_effect=(
        "Masks tokens producing unwrap(), expect(), panic!(), or indexing with []. "
        "Enforces safe methods like get() that return Option, never panicking."
    ),
    valid_outputs=[
        "items.get(key)",
        "items.get(key).ok_or(Error::NotFound)",
        "items.get(key).copied()",
        "items.first()",
    ],
    invalid_outputs=[
        "items[key]",  # Panics on out-of-bounds
        "items.get(key).unwrap()",  # Panics if None
        "items.get(key).expect(\"must exist\")",  # Panics if None
        "items[key].clone()",  # Unchecked indexing
    ],
    tags=["semantics", "panic", "safety", "no-panic"],
    language="rust",
    domain="semantics",
)

RUST_SEMANTIC_002 = ConstraintExample(
    id="rust-semantics-002",
    name="Memory Safety Invariant",
    description="Constraint generation ensuring memory safety invariants in unsafe code",
    scenario=(
        "Developer implementing a custom allocator that must maintain safety invariants. "
        "When working with raw pointers, alignment and non-null guarantees must hold. "
        "The constraint ensures checks are present before dereferencing raw pointers."
    ),
    prompt="""Write a value through a raw pointer. Check for null first, then use unsafe with SAFETY comment.
Document why the unsafe operation is sound in the SAFETY comment.

fn write_value<T>(ptr: *mut T, value: T) -> Result<(), Error> {
    """,
    spec=ConstraintSpec(
        language="rust",
        type_bindings=[
            TypeBinding(name="ptr", type_expr="*mut T", scope="parameter", mutable=True),
        ],
        control_flow=ControlFlowContext(
            function_name="write_value",
        ),
        semantic_constraints=[
            SemanticConstraint(
                kind="precondition",
                expression="!ptr.is_null() && ptr.is_aligned()",
                scope="write_value",
                variables=("ptr",),
            ),
            SemanticConstraint(
                kind="invariant",
                expression="unsafe_blocks_have_safety_comments()",
                scope="write_value",
                variables=(),
            ),
        ],
        ebnf=r'''
root ::= two_guards | one_guard
two_guards ::= "if ptr.is_null() {\n    return Err(Error::NullPtr);\n}\nif !ptr.is_aligned() {\n    return Err(Error::Misaligned);\n}\nunsafe {\n// SAFETY: ptr is non-null and aligned as checked above\n    ptr.write(value)\n}"
one_guard ::= "if ptr.is_null() {\n    return Err(Error::Null);\n}\nunsafe {\n// SAFETY: ptr is non-null and write is within allocation bounds\n    ptr.write(value)\n}"
''',
    ),
    expected_effect=(
        "Masks tokens that dereference raw pointers without null/alignment checks. "
        "Ensures unsafe blocks have SAFETY comments documenting invariants upheld."
    ),
    valid_outputs=[
        "if ptr.is_null() {\n    return Err(Error::NullPtr);\n}\nif !ptr.is_aligned() {\n    return Err(Error::Misaligned);\n}\nunsafe {\n// SAFETY: ptr is non-null and aligned as checked above\n    ptr.write(value)\n}",
        "if ptr.is_null() {\n    return Err(Error::Null);\n}\nunsafe {\n// SAFETY: ptr is non-null and write is within allocation bounds\n    ptr.write(value)\n}",
    ],
    invalid_outputs=[
        "unsafe { ptr.write(value) }",  # No safety comment
        "unsafe {\n    ptr.write(value)\n}",  # No guards, no comment
        "if ptr.is_null() { return Err(Error::Null); }\nunsafe { ptr.write(value) }",  # Missing SAFETY comment
    ],
    tags=["semantics", "unsafe", "memory-safety", "invariants"],
    language="rust",
    domain="semantics",
)

RUST_SEMANTIC_003 = ConstraintExample(
    id="rust-semantics-003",
    name="Unsafe Block with Safety Comment Requirement",
    description="Constraint generation requiring documentation for all unsafe blocks",
    scenario=(
        "Developer implementing FFI bindings to a C library. Every unsafe block must "
        "document why it's safe, what invariants it relies on, and what could go wrong. "
        "The constraint ensures unsafe code has mandatory SAFETY comments."
    ),
    prompt="""Close a file descriptor using libc. Wrap in unsafe with a // SAFETY: comment.
The comment must explain why this operation is safe.

fn close_fd(fd: RawFd) {
    """,
    spec=ConstraintSpec(
        language="rust",
        type_bindings=[
            TypeBinding(name="fd", type_expr="RawFd", scope="parameter"),
        ],
        semantic_constraints=[
            SemanticConstraint(
                kind="invariant",
                expression="unsafe_block_has_safety_doc()",
                scope="close_fd",
                variables=(),
            ),
        ],
        ebnf=r'''
root ::= "unsafe {" nl comment nl "    " operation nl "}"
comment ::= "// SAFETY: " explanation
explanation ::= [a-zA-Z (),.']+
operation ::= "libc::close(fd)"
nl ::= "\n"
''',
    ),
    expected_effect=(
        "Masks tokens producing unsafe blocks without // SAFETY: comment preceding "
        "the unsafe operation. Enforces documentation of safety reasoning."
    ),
    valid_outputs=[
        "unsafe {\n// SAFETY: fd is a valid file descriptor obtained from open() and has not been closed yet\n    libc::close(fd)\n}",
        "unsafe {\n// SAFETY: caller guarantees fd is valid and owned by this function for closure\n    libc::close(fd)\n}",
    ],
    invalid_outputs=[
        "unsafe { libc::close(fd) }",  # No SAFETY comment
        "unsafe {\n    // Just a regular comment\n    libc::close(fd)\n}",  # Wrong comment format
        "unsafe {\n// TODO: add safety comment\n    libc::close(fd)\n}",  # Placeholder, not real safety doc
    ],
    tags=["semantics", "unsafe", "documentation", "safety"],
    language="rust",
    domain="semantics",
)

# =============================================================================
# Exports
# =============================================================================

RUST_SEMANTIC_EXAMPLES: List[ConstraintExample] = [
    RUST_SEMANTIC_001,
    RUST_SEMANTIC_002,
    RUST_SEMANTIC_003,
]

__all__ = ["RUST_SEMANTIC_EXAMPLES"]
