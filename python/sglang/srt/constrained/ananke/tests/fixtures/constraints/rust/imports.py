# Copyright Rand Arete @ Ananke 2025
# Licensed under the Apache License, Version 2.0
"""Rust import constraint examples for Ananke.

This module contains realistic examples of import/dependency constraints in Rust,
demonstrating no_std environments, unsafe library restrictions, and feature gating.
"""

from __future__ import annotations

from typing import List

try:
    from ..base import ConstraintExample
    from ....spec.constraint_spec import (
        ConstraintSpec,
        ImportBinding,
        SemanticConstraint,
    )
except ImportError:
    from tests.fixtures.constraints.base import ConstraintExample
    from spec.constraint_spec import (
        ConstraintSpec,
        ImportBinding,
        SemanticConstraint,
    )

# =============================================================================
# Import Constraint Examples
# =============================================================================

RUST_IMPORT_001 = ConstraintExample(
    id="rust-imports-001",
    name="no_std Environment Constraint",
    description="Constraint generation for embedded/kernel code forbidding std library",
    scenario=(
        "Developer writing embedded firmware for a microcontroller without OS support. "
        "The code must compile with #![no_std], forbidding any std::* imports. "
        "The constraint ensures generated code only uses core:: and alloc:: when needed."
    ),
    prompt="""Write imports for a #![no_std] embedded environment. Only use core:: and alloc::.
No std:: imports allowed - this code runs without OS support.

#![no_std]

""",
    spec=ConstraintSpec(
        language="rust",
        forbidden_imports={"std", "std::*"},
        imports=[
            ImportBinding(module="core::fmt", name="Write"),
            ImportBinding(module="core::ptr"),
            ImportBinding(module="alloc::vec", name="Vec"),
            ImportBinding(module="alloc::string", name="String"),
        ],
        semantic_constraints=[
            SemanticConstraint(
                kind="invariant",
                expression="!uses_std_features()",
                scope="module",
                variables=(),
            ),
        ],
        ebnf=r'''
root ::= "use " module path ";"
module ::= "core" | "alloc"
path ::= ("::" segment)+
segment ::= [a-zA-Z_][a-zA-Z0-9_]*
''',
    ),
    expected_effect=(
        "Masks tokens that would import std:: modules or use std-only features like "
        "std::io, std::fs, or std::thread. Allows core:: (always available) and alloc:: "
        "(if allocator provided)."
    ),
    valid_outputs=[
        "use core::fmt::Write;",
        "use alloc::vec::Vec;",
        "use core::ptr::write_volatile;",
        "use alloc::string::String;",
    ],
    invalid_outputs=[
        "use std::vec::Vec;",  # std forbidden in no_std
        "use std::io::Write;",  # std::io not available
        "use std::collections::HashMap;",  # std forbidden
    ],
    tags=["imports", "no_std", "embedded", "kernel"],
    language="rust",
    domain="imports",
)

RUST_IMPORT_002 = ConstraintExample(
    id="rust-imports-002",
    name="Unsafe Library Restriction",
    description="Constraint generation forbidding direct unsafe FFI dependencies",
    scenario=(
        "Developer working in a memory-safe codebase where direct libc usage is banned. "
        "All system calls must go through safe wrappers like std::fs or nix crate. "
        "The constraint prevents importing raw libc functions that bypass safety."
    ),
    prompt="""Write imports for filesystem operations using safe wrappers. Don't use libc directly.
Use std::fs, std::io, or nix crate for system calls - no raw FFI.

""",
    spec=ConstraintSpec(
        language="rust",
        forbidden_imports={
            "libc",
            "libc::*",
            "winapi::um::*",  # Windows unsafe APIs
        },
        imports=[
            ImportBinding(module="std::fs", name="File"),
            ImportBinding(module="std::io", name="Read"),
            ImportBinding(module="nix::unistd", name="read"),  # Safe wrapper
        ],
        semantic_constraints=[
            SemanticConstraint(
                kind="invariant",
                expression="!contains_unsafe_blocks() || has_safety_comment()",
                scope="module",
                variables=(),
            ),
        ],
        ebnf=r'''
root ::= use_fs | use_os | use_nix | use_io
use_fs ::= "use std::fs::OpenOptions;"
use_os ::= "use std::os::unix::io::AsRawFd;"
use_nix ::= "use nix::sys::socket::socket;"
use_io ::= "use std::io::{Read, Write};"
''',
    ),
    expected_effect=(
        "Masks tokens importing libc or winapi directly. Forces use of safe wrappers "
        "from std or crates like nix that provide memory-safe interfaces to system calls."
    ),
    valid_outputs=[
        "use std::fs::OpenOptions;",
        "use std::os::unix::io::AsRawFd;",
        "use nix::sys::socket::socket;",
        "use std::io::{Read, Write};",
    ],
    invalid_outputs=[
        "use libc::read;",  # Direct libc forbidden
        "use libc::c_char;",  # libc types forbidden
        "use winapi::um::fileapi::ReadFile;",  # Windows unsafe API
    ],
    tags=["imports", "safety", "unsafe", "ffi"],
    language="rust",
    domain="imports",
)

RUST_IMPORT_003 = ConstraintExample(
    id="rust-imports-003",
    name="Feature-Gated Crate Dependencies",
    description="Constraint generation for imports only allowed under feature flags",
    scenario=(
        "Developer implementing optional functionality that requires heavy dependencies. "
        "The code should only import serde and tokio when their feature flags are enabled. "
        "The constraint ensures generated imports are guarded by cfg attributes."
    ),
    prompt="""Import serde or tokio with proper feature gate. These are optional dependencies.
Add #[cfg(feature = "...")] before the use statement for conditional compilation.

""",
    spec=ConstraintSpec(
        language="rust",
        imports=[
            ImportBinding(module="serde", name="Serialize"),  # Requires feature="serde"
            ImportBinding(module="serde", name="Deserialize"),
            ImportBinding(module="tokio::runtime", name="Runtime"),  # Requires feature="async"
        ],
        semantic_constraints=[
            SemanticConstraint(
                kind="precondition",
                expression='feature_enabled("serde") || feature_enabled("async")',
                scope="module",
                variables=(),
            ),
        ],
        ebnf=r'''
root ::= cfg_attr use_stmt
cfg_attr ::= "#[cfg(feature = " quoted_name ")]" nl
quoted_name ::= "\"serde\"" | "\"async\""
use_stmt ::= "use " module path final_part ";"
module ::= "serde" | "tokio"
path ::= ("::" segment)*
segment ::= [a-z_]+
final_part ::= "::" item | "::{" items "}"
item ::= [a-zA-Z][a-zA-Z0-9_]*
items ::= item (", " item)*
nl ::= "\n"
''',
    ),
    expected_effect=(
        "Masks tokens that import feature-gated dependencies without cfg guards. "
        "Ensures serde imports have #[cfg(feature = \"serde\")] and tokio has "
        "#[cfg(feature = \"async\")]."
    ),
    valid_outputs=[
        '#[cfg(feature = "serde")]\nuse serde::{Serialize, Deserialize};',
        '#[cfg(feature = "async")]\nuse tokio::runtime::Runtime;',
        '#[cfg(feature = "serde")]\nuse serde::de::DeserializeOwned;',
    ],
    invalid_outputs=[
        "use serde::Serialize;",  # Missing cfg guard
        "use tokio::runtime::Runtime;",  # Missing cfg guard
        '#[cfg(test)]\nuse serde::Serialize;',  # Wrong cfg condition
    ],
    tags=["imports", "features", "conditional", "cfg"],
    language="rust",
    domain="imports",
)

# =============================================================================
# Exports
# =============================================================================

RUST_IMPORT_EXAMPLES: List[ConstraintExample] = [
    RUST_IMPORT_001,
    RUST_IMPORT_002,
    RUST_IMPORT_003,
]

__all__ = ["RUST_IMPORT_EXAMPLES"]
