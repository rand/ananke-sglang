# Copyright Rand Arete @ Ananke 2025
# Licensed under the Apache License, Version 2.0
"""Rust syntax constraint examples for Ananke.

This module contains realistic examples of syntax/grammar constraints in Rust,
demonstrating serde schemas, version strings, and macro DSL patterns.
"""

from __future__ import annotations

from typing import List

try:
    from ..base import ConstraintExample
    from ....spec.constraint_spec import (
        ConstraintSpec,
        TypeBinding,
        ImportBinding,
    )
except ImportError:
    from tests.fixtures.constraints.base import ConstraintExample
    from spec.constraint_spec import (
        ConstraintSpec,
        TypeBinding,
        ImportBinding,
    )

# =============================================================================
# Syntax Constraint Examples
# =============================================================================

RUST_SYNTAX_001 = ConstraintExample(
    id="rust-syntax-001",
    name="Serde Struct Schema for Configuration",
    description="Constraint generation for serde-compatible struct definitions",
    scenario=(
        "Developer creating a configuration struct that must serialize/deserialize with serde. "
        "The struct needs derive macros, field attributes for rename/default, and specific types. "
        "The constraint ensures generated struct follows serde conventions."
    ),
    prompt="""Create a Config struct that derives Serialize and Deserialize.
Fields: host (String), port (u16), ssl_enabled (bool with default), timeout_ms (u64 with default).

use serde::{Serialize, Deserialize};

""",
    spec=ConstraintSpec(
        language="rust",
        imports=[
            ImportBinding(module="serde", name="Serialize"),
            ImportBinding(module="serde", name="Deserialize"),
        ],
        json_schema="""{
            "type": "object",
            "properties": {
                "host": {"type": "string"},
                "port": {"type": "integer"},
                "ssl_enabled": {"type": "boolean"},
                "timeout_ms": {"type": "integer"}
            },
            "required": ["host", "port"]
        }""",
    ),
    expected_effect=(
        "Masks tokens producing structs without serde derives or incompatible types. "
        "Ensures #[derive(Serialize, Deserialize)] is present and fields match schema."
    ),
    valid_outputs=[
        """#[derive(Serialize, Deserialize)]
struct Config {
    host: String,
    port: u16,
    #[serde(default)]
    ssl_enabled: bool,
    #[serde(default = "default_timeout")]
    timeout_ms: u64,
}""",
        """#[derive(Serialize, Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
struct Config {
    host: String,
    port: u16,
    ssl_enabled: bool,
    timeout_ms: u64,
}""",
    ],
    invalid_outputs=[
        """struct Config {
    host: String,
    port: u16,
}""",  # Missing serde derives
        """#[derive(Debug)]
struct Config {
    host: String,
    port: u16,
}""",  # Wrong derives
        """#[derive(Serialize, Deserialize)]
struct Config {
    host: &str,
    port: u16,
}""",  # &str not owned, won't deserialize properly
    ],
    tags=["syntax", "serde", "struct", "schema"],
    language="rust",
    domain="syntax",
)

RUST_SYNTAX_002 = ConstraintExample(
    id="rust-syntax-002",
    name="Semantic Version Regex Pattern",
    description="Constraint generation for semver version string validation",
    scenario=(
        "Developer implementing version parsing for Cargo.toml dependencies. "
        "Version strings must match semver 2.0.0 spec: MAJOR.MINOR.PATCH with optional pre/build. "
        "The constraint ensures generated regex correctly validates semver format."
    ),
    prompt="""Generate a valid semver version string like "1.0.0" or "2.1.3-alpha.1".
Format: MAJOR.MINOR.PATCH with optional -prerelease and +build.

version = """,
    spec=ConstraintSpec(
        language="rust",
        # Regex for semver inside quoted strings
        regex=r'"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?"',
        type_bindings=[
            TypeBinding(name="version_str", type_expr="&str", scope="parameter"),
        ],
    ),
    expected_effect=(
        "Masks tokens producing non-semver version patterns. Ensures regex matches "
        "MAJOR.MINOR.PATCH with optional -prerelease and +build metadata."
    ),
    valid_outputs=[
        '"1.0.0"',
        '"2.1.3-alpha.1"',
        '"0.0.1-beta+build.123"',
        '"10.20.30-rc.1+20230101"',
    ],
    invalid_outputs=[
        '"1.0"',  # Missing patch version
        '"v1.0.0"',  # Leading 'v' not in semver
        '"1.0.0.0"',  # Four components not semver
        '"01.0.0"',  # Leading zero on major version
    ],
    tags=["syntax", "regex", "semver", "validation"],
    language="rust",
    domain="syntax",
)

RUST_SYNTAX_003 = ConstraintExample(
    id="rust-syntax-003",
    name="Macro DSL Pattern for Builder",
    description="Constraint generation for declarative macro builder pattern",
    scenario=(
        "Developer creating a builder macro that generates struct builder code. "
        "The macro takes field definitions and produces a fluent builder API. "
        "The constraint ensures generated macro follows Rust macro_rules! syntax."
    ),
    prompt="""Create a macro_rules! macro that generates a Builder struct from field definitions.
Use $name:ident and $ty:ty for capture, wrap fields in Option<$ty>.

""",
    spec=ConstraintSpec(
        language="rust",
        ebnf=r'''
root ::= macro_single | macro_double
macro_single ::= "macro_rules! build_struct {\n    ({ $name:ident : $ty:ty }) => {\n        pub struct Builder { $name: Option<$ty> }\n    };\n}"
macro_double ::= "macro_rules! builder {\n    ({ $field1:ident : $type1:ty , $field2:ident : $type2:ty }) => {\n        pub struct Builder { $field1: Option<$type1> , $field2: Option<$type2> }\n    };\n}"
''',
    ),
    expected_effect=(
        "Masks tokens that don't follow macro_rules! syntax. Ensures proper capture "
        "of :ident and :ty fragments with $ prefix and correct expansion."
    ),
    valid_outputs=[
        """macro_rules! build_struct {
    ({ $name:ident : $ty:ty }) => {
        pub struct Builder { $name: Option<$ty> }
    };
}""",
        """macro_rules! builder {
    ({ $field1:ident : $type1:ty , $field2:ident : $type2:ty }) => {
        pub struct Builder { $field1: Option<$type1> , $field2: Option<$type2> }
    };
}""",
    ],
    invalid_outputs=[
        """macro_rules! build_struct {
    (name: ty) => {
        pub struct Builder { name: Option<ty> }
    };
}""",  # Missing $ and :ident/:ty
        """macro_rules! build_struct {
    ({ $name }) => {
        pub struct Builder { $name }
    };
}""",  # Missing fragment specifier
        """fn build_struct() {
    pub struct Builder { name: Option<String> }
}""",  # Not a macro at all
    ],
    tags=["syntax", "macro", "dsl", "builder"],
    language="rust",
    domain="syntax",
)

# =============================================================================
# Exports
# =============================================================================

RUST_SYNTAX_EXAMPLES: List[ConstraintExample] = [
    RUST_SYNTAX_001,
    RUST_SYNTAX_002,
    RUST_SYNTAX_003,
]

__all__ = ["RUST_SYNTAX_EXAMPLES"]
