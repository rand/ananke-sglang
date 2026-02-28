# Copyright Rand Arete @ Ananke 2025
# Licensed under the Apache License, Version 2.0
"""Type constraint examples for Kotlin.

This module contains realistic examples of type-level constraints that
demonstrate how Ananke's TypeDomain masks tokens to enforce Kotlin's
type system during code generation, including nullable types, sealed classes,
and suspend function contexts.
"""

from __future__ import annotations

try:
    from ..base import ConstraintExample
    from .....spec.constraint_spec import (
        ConstraintSpec,
        TypeBinding,
        FunctionSignature,
        ClassDefinition,
    )
except ImportError:
    from tests.fixtures.constraints.base import ConstraintExample
    from spec.constraint_spec import (
        ConstraintSpec,
        TypeBinding,
        FunctionSignature,
        ClassDefinition,
    )

KOTLIN_TYPE_EXAMPLES = [
    ConstraintExample(
        id="kt-types-001",
        name="Nullable Type Safe Navigation",
        description="Use safe call operator ?. with nullable types",
        scenario=(
            "Developer working with a nullable User? type and needing to safely "
            "access properties. Kotlin's type system requires using safe call "
            "operators (?.) or Elvis operator (?:) when working with nullable types. "
            "Direct property access on nullable types is a compile error."
        ),
        prompt="""Access the name property of a nullable User safely in Kotlin.
Use safe call operator (?.) since user might be null - direct access like user.name won't compile.

val user: User? = getUser()
""",
        spec=ConstraintSpec(
            language="kotlin",
            # Regex enforces safe call (?.) or Elvis (?:) operators for nullable access
            regex=r"^user\?\.",
            ebnf=r'''
root ::= safe_name | elvis_email | let_name | chain_upper
safe_name ::= "user?.name"
elvis_email ::= "user?.email ?: \"unknown\""
let_name ::= "user?.let { it.name }"
chain_upper ::= "user?.name?.uppercase()"
''',
            expected_type="String?",
            type_bindings=[
                TypeBinding(name="user", type_expr="User?", scope="local"),
            ],
            class_definitions=[
                ClassDefinition(
                    name="User",
                    instance_vars=(
                        TypeBinding(name="name", type_expr="String"),
                        TypeBinding(name="email", type_expr="String"),
                        TypeBinding(name="age", type_expr="Int"),
                    ),
                )
            ],
        ),
        expected_effect=(
            "Masks tokens that attempt direct property access on nullable types. "
            "Blocks 'user.name' (unsafe) and allows 'user?.name' (safe call), "
            "'user!!.name' (non-null assertion), or 'user?.name ?: default' (Elvis). "
            "Enforces Kotlin's null safety at the token level."
        ),
        valid_outputs=[
            "user?.name",
            "user?.email ?: \"unknown\"",
            "user?.let { it.name }",
            "user?.name?.uppercase()",
        ],
        invalid_outputs=[
            "user.name",  # Unsafe: nullable receiver
            "user.email",  # Unsafe: nullable receiver
            "user.age.toString()",  # Unsafe: nullable receiver
        ],
        tags=["types", "nullable", "null-safety", "kotlin"],
        language="kotlin",
        domain="types",
    ),
    ConstraintExample(
        id="kt-types-002",
        name="Sealed Class Exhaustive When",
        description="Enforce exhaustive when expression with sealed classes",
        scenario=(
            "Developer implementing a when expression on a sealed class hierarchy. "
            "Kotlin's type system requires exhaustive coverage of all sealed class "
            "subtypes in when expressions used as expressions (not statements). "
            "Missing a branch causes a compile error."
        ),
        prompt="""Write a when expression to handle all cases of a sealed Result class.
The sealed class has three subtypes: Success(data), Error(message), and Loading.
Handle each case - missing any will cause a compile error.

val result: Result = getResult()
return """,
        spec=ConstraintSpec(
            language="kotlin",
            # EBNF enforces exhaustive when expression for sealed class
            ebnf=r'''
root ::= when1 | when2
when1 ::= "when (result) {\n                is Success -> result.data\n                is Error -> result.message\n                is Loading -> \"Loading...\"\n            }"
when2 ::= "when (result) {\n                is Success -> \"Got: ${result.data}\"\n                is Error -> \"Failed: ${result.message}\"\n                is Loading -> \"Please wait\"\n            }"
''',
            expected_type="String",
            type_bindings=[
                TypeBinding(name="result", type_expr="Result", scope="local"),
            ],
            class_definitions=[
                ClassDefinition(
                    name="Result",
                    bases=("sealed class",),
                ),
                ClassDefinition(
                    name="Success",
                    bases=("Result",),
                    instance_vars=(TypeBinding(name="data", type_expr="String"),),
                ),
                ClassDefinition(
                    name="Error",
                    bases=("Result",),
                    instance_vars=(TypeBinding(name="message", type_expr="String"),),
                ),
                ClassDefinition(
                    name="Loading",
                    bases=("Result",),
                ),
            ],
        ),
        expected_effect=(
            "Masks tokens that create non-exhaustive when expressions. Requires all "
            "sealed class subtypes (Success, Error, Loading) to be handled. Blocks "
            "when expressions missing branches or using 'else' prematurely."
        ),
        valid_outputs=[
            """when (result) {
                is Success -> result.data
                is Error -> result.message
                is Loading -> "Loading..."
            }""",
            """when (result) {
                is Success -> "Got: ${result.data}"
                is Error -> "Failed: ${result.message}"
                is Loading -> "Please wait"
            }""",
        ],
        invalid_outputs=[
            """when (result) {
                is Success -> result.data
                is Error -> result.message
            }""",  # Missing Loading case
            """when (result) {
                is Success -> result.data
            }""",  # Missing Error and Loading cases
            "result.data",  # Direct access without when/smart cast
        ],
        tags=["types", "sealed-class", "when-expression", "exhaustiveness", "kotlin"],
        language="kotlin",
        domain="types",
    ),
    ConstraintExample(
        id="kt-types-003",
        name="Suspend Function Context",
        description="Enforce suspend context requirements",
        scenario=(
            "Developer calling a suspend function which can only be called from "
            "coroutine contexts (other suspend functions, coroutine builders). "
            "Kotlin enforces that suspend functions cannot be called from regular "
            "synchronous contexts - this is a compile-time error."
        ),
        prompt="""Call the suspend function fetchUser(userId) from within this suspend function.
Since we're already in a suspend context, we can call it directly.

suspend fun processUser(userId: String): User {
    """,
        spec=ConstraintSpec(
            language="kotlin",
            # Regex enforces direct suspend function call pattern (inside suspend context)
            regex=r"^(?:return\s+)?(?:val\s+\w+\s*=\s*)?fetchUser\s*\(",
            ebnf=r'''
root ::= return_direct | val_assign | also_chain
return_direct ::= "return fetchUser(userId)"
val_assign ::= "val user = fetchUser(userId)" nl "return user"
also_chain ::= "return fetchUser(userId).also { println(it) }"
nl ::= "\n"
''',
            expected_type="User",
            type_bindings=[
                TypeBinding(name="userId", type_expr="String", scope="parameter"),
            ],
            function_signatures=[
                FunctionSignature(
                    name="fetchUser",
                    params=(TypeBinding(name="id", type_expr="String"),),
                    return_type="User",
                    is_async=True,  # Maps to suspend
                ),
                FunctionSignature(
                    name="processUser",
                    params=(TypeBinding(name="userId", type_expr="String"),),
                    return_type="User",
                    is_async=True,  # We're in a suspend context
                ),
            ],
            control_flow=None,
        ),
        expected_effect=(
            "Masks tokens that call suspend functions without proper context. "
            "Allows direct suspend function calls when inside a suspend function. "
            "Blocks direct calls from non-suspend contexts; requires coroutine "
            "builders like runBlocking, launch, async."
        ),
        valid_outputs=[
            "return fetchUser(userId)",  # Direct call in suspend context
            "val user = fetchUser(userId)\nreturn user",
            "return fetchUser(userId).also { println(it) }",
        ],
        invalid_outputs=[
            "Thread { fetchUser(userId) }.start()",  # Calling suspend from Thread
            "val handler = Handler()\nhandler.post { fetchUser(userId) }",  # Non-suspend lambda
            "object : Runnable { override fun run() = fetchUser(userId) }",  # Runnable context
        ],
        tags=["types", "suspend", "coroutines", "async", "kotlin"],
        language="kotlin",
        domain="types",
    ),
]
