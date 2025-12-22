# Copyright Rand Arete @ Ananke 2025
# Licensed under the Apache License, Version 2.0
"""Rust control flow constraint examples for Ananke.

This module contains realistic examples of control flow constraints in Rust,
demonstrating match exhaustiveness, ? operator, and early return patterns.
"""

from __future__ import annotations

from typing import List

try:
    from ..base import ConstraintExample
    from ....spec.constraint_spec import (
        ConstraintSpec,
        ControlFlowContext,
        FunctionSignature,
        TypeBinding,
        SemanticConstraint,
    )
except ImportError:
    from tests.fixtures.constraints.base import ConstraintExample
    from spec.constraint_spec import (
        ConstraintSpec,
        ControlFlowContext,
        FunctionSignature,
        TypeBinding,
        SemanticConstraint,
    )

# =============================================================================
# Control Flow Constraint Examples
# =============================================================================

RUST_CONTROLFLOW_001 = ConstraintExample(
    id="rust-controlflow-001",
    name="Match Exhaustiveness with Enums",
    description="Constraint generation ensuring all enum variants are handled in match",
    scenario=(
        "Developer implementing command processing for an enum with multiple variants. "
        "Rust requires exhaustive matching, so all variants must be covered. "
        "The constraint ensures generated match arms handle every case without wildcard."
    ),
    prompt="""Handle all Command enum variants (Start, Stop, Restart, Status) in a match expression.
Cover all cases explicitly - don't use _ wildcard.

fn execute_command(cmd: Command) -> Result<(), Error> {
    """,
    spec=ConstraintSpec(
        language="rust",
        control_flow=ControlFlowContext(
            function_name="execute_command",
            expected_return_type="Result<(), Error>",
            in_async_context=False,
        ),
        type_bindings=[
            TypeBinding(
                name="cmd",
                type_expr="Command",
                scope="parameter",
                mutable=False,
            ),
        ],
        semantic_constraints=[
            SemanticConstraint(
                kind="invariant",
                expression="all_variants_covered(match_expr)",
                scope="execute_command",
                variables=("match_expr",),
            ),
        ],
        ebnf=r'''
root ::= match_ok | match_service
match_ok ::= "match cmd {\n    Command::Start(id) => Ok(()),\n    Command::Stop(id) => Ok(()),\n    Command::Restart(id) => Ok(()),\n    Command::Status => Ok(()),\n}"
match_service ::= "match cmd {\n    Command::Start(_) => start_service(),\n    Command::Stop(_) => stop_service(),\n    Command::Restart(_) => restart_service(),\n    Command::Status => get_status(),\n}"
''',
    ),
    expected_effect=(
        "Masks tokens that use _ wildcard or omit enum variants. Enforces explicit "
        "handling of Command::Start, Stop, Restart, and Status without catch-all."
    ),
    valid_outputs=[
        "match cmd {\n    Command::Start(id) => Ok(()),\n    Command::Stop(id) => Ok(()),\n    Command::Restart(id) => Ok(()),\n    Command::Status => Ok(()),\n}",
        "match cmd {\n    Command::Start(_) => start_service(),\n    Command::Stop(_) => stop_service(),\n    Command::Restart(_) => restart_service(),\n    Command::Status => get_status(),\n}",
    ],
    invalid_outputs=[
        "match cmd {\n    Command::Start(id) => Ok(()),\n    _ => Err(Error::Unknown),\n}",  # Wildcard, non-exhaustive
        "match cmd {\n    Command::Start(id) => Ok(()),\n    Command::Stop(id) => Ok(()),\n}",  # Missing variants
    ],
    tags=["controlflow", "match", "exhaustiveness", "enums"],
    language="rust",
    domain="controlflow",
)

RUST_CONTROLFLOW_002 = ConstraintExample(
    id="rust-controlflow-002",
    name="Question Mark Operator in Result Context",
    description="Constraint generation for ? operator with proper Result propagation",
    scenario=(
        "Developer writing a function that calls multiple fallible operations. "
        "The ? operator propagates errors automatically but requires Result return type. "
        "The constraint ensures ? is used correctly and errors are compatible."
    ),
    prompt="""Load config from a file path. Use ? operator for error propagation.
Don't use unwrap() or expect() - propagate errors with ? and wrap success in Ok().

fn load_config(path: &Path) -> Result<Config, io::Error> {
    """,
    spec=ConstraintSpec(
        language="rust",
        control_flow=ControlFlowContext(
            function_name="load_config",
            function_signature=FunctionSignature(
                name="load_config",
                params=(TypeBinding("path", "&Path", scope="parameter"),),
                return_type="Result<Config, io::Error>",
            ),
            expected_return_type="Result<Config, io::Error>",
        ),
        type_bindings=[
            TypeBinding(name="path", type_expr="&Path", scope="parameter"),
        ],
        semantic_constraints=[
            SemanticConstraint(
                kind="invariant",
                expression="error_type_compatible(fn_error, op_error)",
                scope="load_config",
                variables=("fn_error", "op_error"),
            ),
        ],
        ebnf=r'''
root ::= read_simple | read_buf
read_simple ::= "let file = File::open(path)?;\nlet contents = read_to_string(file)?;\nOk(parse_config(&contents)?)"
read_buf ::= "let mut file = File::open(path)?;\nlet mut buf = String::new();\nfile.read_to_string(&mut buf)?;\nOk(Config::parse(&buf)?)"
''',
    ),
    expected_effect=(
        "Masks tokens that use unwrap() or expect() instead of ?. Ensures all fallible "
        "operations use ? for error propagation and final value is wrapped in Ok()."
    ),
    valid_outputs=[
        "let file = File::open(path)?;\nlet contents = read_to_string(file)?;\nOk(parse_config(&contents)?)",
        "let mut file = File::open(path)?;\nlet mut buf = String::new();\nfile.read_to_string(&mut buf)?;\nOk(Config::parse(&buf)?)",
    ],
    invalid_outputs=[
        "let file = File::open(path).unwrap();\nOk(parse_config(&file))",  # unwrap instead of ?
        "let file = File::open(path)?;\nparse_config(&file)",  # Missing Ok() wrapper
        "File::open(path)?",  # Incomplete, no final Ok()
    ],
    tags=["controlflow", "error-handling", "result", "question-mark"],
    language="rust",
    domain="controlflow",
)

RUST_CONTROLFLOW_003 = ConstraintExample(
    id="rust-controlflow-003",
    name="Early Return with Guard Clauses",
    description="Constraint generation for guard clause pattern with early returns",
    scenario=(
        "Developer implementing validation logic with multiple preconditions. "
        "Guard clauses check conditions early and return errors, avoiding nested ifs. "
        "The constraint ensures each guard returns appropriate error without continuing."
    ),
    prompt="""Validate user fields with guard clauses. Check name and age, return Err() for invalid cases.
Use early returns for each validation - end with Ok(()) for valid users.

fn validate_user(user: &User) -> Result<(), ValidationError> {
    """,
    spec=ConstraintSpec(
        language="rust",
        control_flow=ControlFlowContext(
            function_name="validate_user",
            expected_return_type="Result<(), ValidationError>",
        ),
        type_bindings=[
            TypeBinding(name="user", type_expr="&User", scope="parameter"),
        ],
        semantic_constraints=[
            SemanticConstraint(
                kind="invariant",
                expression="every_guard_returns()",
                scope="validate_user",
                variables=(),
            ),
        ],
        ebnf=r'''
root ::= guard+ success_return
guard ::= "if " condition " {" nl indent "return Err(" error ");" nl "}" nl
condition ::= var "." accessor " " op " " value
accessor ::= "name" | "email" | "age"
op ::= "==" | "!=" | "<" | ">" | "<=" | ">="
value ::= "\"\"" | [0-9]+
error ::= "ValidationError::" error_name
error_name ::= [A-Z][a-zA-Z]*
success_return ::= "Ok(())"
var ::= "user"
indent ::= "    "
nl ::= "\n"
''',
    ),
    expected_effect=(
        "Masks tokens that continue execution after failed validation. Ensures each guard "
        "clause has an explicit return Err() and success path returns Ok(()) at the end."
    ),
    valid_outputs=[
        'if user.name == "" {\n    return Err(ValidationError::EmptyName);\n}\nif user.age < 18 {\n    return Err(ValidationError::TooYoung);\n}\nOk(())',
        'if user.email == "" {\n    return Err(ValidationError::NoEmail);\n}\nif user.name == "" {\n    return Err(ValidationError::NoName);\n}\nOk(())',
    ],
    invalid_outputs=[
        'if user.name == "" {\n    ValidationError::EmptyName\n}\nOk(())',  # Missing return
        'if user.age < 18 {\n    return Err(ValidationError::TooYoung);\n}',  # Missing final Ok()
        'user.name != "" && user.age >= 18',  # Not using guards, just expression
    ],
    tags=["controlflow", "guards", "early-return", "validation"],
    language="rust",
    domain="controlflow",
)

# =============================================================================
# Exports
# =============================================================================

RUST_CONTROLFLOW_EXAMPLES: List[ConstraintExample] = [
    RUST_CONTROLFLOW_001,
    RUST_CONTROLFLOW_002,
    RUST_CONTROLFLOW_003,
]

__all__ = ["RUST_CONTROLFLOW_EXAMPLES"]
