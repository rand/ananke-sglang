# Copyright Rand Arete @ Ananke 2025
# Licensed under the Apache License, Version 2.0
"""Control flow constraint examples for Kotlin.

This module contains realistic examples of control flow constraints that
demonstrate how Ananke's ControlFlowDomain masks tokens to enforce Kotlin's
control flow semantics, including when exhaustiveness, coroutine scope/context,
and null-check smart casts.
"""

from __future__ import annotations

try:
    from ..base import ConstraintExample
    from .....spec.constraint_spec import (
        ConstraintSpec,
        TypeBinding,
        FunctionSignature,
        ClassDefinition,
        ControlFlowContext,
    )
except ImportError:
    from tests.fixtures.constraints.base import ConstraintExample
    from spec.constraint_spec import (
        ConstraintSpec,
        TypeBinding,
        FunctionSignature,
        ClassDefinition,
        ControlFlowContext,
    )

KOTLIN_CONTROLFLOW_EXAMPLES = [
    ConstraintExample(
        id="kt-cf-001",
        name="When Expression Exhaustiveness with Sealed Classes",
        description="Enforce all sealed class branches in when expression",
        scenario=(
            "Developer using a when expression to handle a sealed class hierarchy. "
            "When used as an expression (returning a value), when must be exhaustive. "
            "The compiler tracks which sealed class subtypes have been handled and "
            "requires all branches or an else clause."
        ),
        prompt="""Complete this Kotlin when expression for a sealed UiState class (Loading, Success, Error):

when (state) {
    """,
        spec=ConstraintSpec(
            language="kotlin",
            # Regex enforces when branches starting with is Loading/Success/Error
            regex=r"^is\s+(?:Loading|Success|Error)\s*->",
            ebnf=r'''
root ::= exhaustive_when | else_when
exhaustive_when ::= "when (state) {" nl "                is Loading -> 0" nl "                is Success -> 200" nl "                is Error -> 500" nl "            }"
else_when ::= "when (state) {" nl "                is Loading -> 0" nl "                is Success -> 200" nl "                else -> 500" nl "            }"
nl ::= "\n"
''',
            expected_type="Int",
            type_bindings=[
                TypeBinding(name="state", type_expr="UiState", scope="local"),
            ],
            class_definitions=[
                ClassDefinition(
                    name="UiState",
                    bases=("sealed class",),
                ),
                ClassDefinition(
                    name="Loading",
                    bases=("UiState",),
                ),
                ClassDefinition(
                    name="Success",
                    bases=("UiState",),
                    instance_vars=(TypeBinding(name="data", type_expr="List<String>"),),
                ),
                ClassDefinition(
                    name="Error",
                    bases=("UiState",),
                    instance_vars=(TypeBinding(name="error", type_expr="Throwable"),),
                ),
            ],
            control_flow=ControlFlowContext(
                function_name="getStatusCode",
                expected_return_type="Int",
            ),
        ),
        expected_effect=(
            "Masks tokens that create non-exhaustive when expressions. Tracks which "
            "sealed class subtypes have been handled in branches. Requires all three "
            "subtypes (Loading, Success, Error) to be covered, or an else branch."
        ),
        valid_outputs=[
            """is Loading -> 0
                is Success -> 200
                is Error -> 500
            }""",
            """is Loading -> 0
                is Success -> 200
                else -> 500
            }""",
        ],
        invalid_outputs=[
            """when (state) {
                is Loading -> 0
                is Success -> 200
            }""",  # Missing Error branch
            """when (state) {
                is Success -> 200
            }""",  # Missing Loading and Error
            "state.data.size",  # No smart cast outside when
        ],
        tags=["controlflow", "when", "sealed-class", "exhaustiveness", "kotlin"],
        language="kotlin",
        domain="controlflow",
    ),
    ConstraintExample(
        id="kt-cf-002",
        name="Coroutine Scope and Context",
        description="Enforce proper coroutine scope for suspend calls",
        scenario=(
            "Developer working inside a coroutine scope (launch, async block) and "
            "calling suspend functions. Coroutine context tracking ensures suspend "
            "functions are only called from valid contexts (other suspend functions "
            "or coroutine builders). Using coroutineContext property requires "
            "being inside a suspend context."
        ),
        prompt="""Call the suspend function fetchData(userId) from a non-suspend context.
Since we're not in a suspend function, we need a coroutine builder like scope.launch{} or runBlocking{}.

val scope: CoroutineScope = ...
""",
        spec=ConstraintSpec(
            language="kotlin",
            # Regex enforces coroutine builder usage for suspend calls
            regex=r"^(?:scope\.launch|scope\.async|runBlocking)\s*\{",
            ebnf=r'''
root ::= launch_builder | async_builder | blocking_builder
launch_builder ::= "scope.launch { fetchData(userId) }"
async_builder ::= "scope.async { fetchData(userId) }"
blocking_builder ::= "runBlocking { fetchData(userId) }"
''',
            expected_type="Unit",
            type_bindings=[
                TypeBinding(name="scope", type_expr="CoroutineScope", scope="local"),
                TypeBinding(name="userId", type_expr="String", scope="local"),
            ],
            function_signatures=[
                FunctionSignature(
                    name="fetchData",
                    params=(TypeBinding(name="id", type_expr="String"),),
                    return_type="Data",
                    is_async=True,  # suspend function
                ),
            ],
            control_flow=ControlFlowContext(
                in_async_context=False,  # Outside suspend context
                function_name="loadUser",
            ),
        ),
        expected_effect=(
            "Masks direct suspend function calls when outside suspend context. "
            "Requires coroutine builders (scope.launch { }, scope.async { }, "
            "runBlocking { }) to call suspend functions from regular contexts. "
            "Allows direct calls inside suspend functions."
        ),
        valid_outputs=[
            "scope.launch { fetchData(userId) }",
            "scope.async { fetchData(userId) }",
            "runBlocking { fetchData(userId) }",
        ],
        invalid_outputs=[
            "fetchData(userId)",  # Direct call outside suspend context
            "val data = fetchData(userId)",  # Direct call, no builder
        ],
        tags=["controlflow", "coroutines", "suspend", "scope", "kotlin"],
        language="kotlin",
        domain="controlflow",
    ),
    ConstraintExample(
        id="kt-cf-003",
        name="Null-Check Smart Cast Branches",
        description="Track smart casts after null checks with let/run/also",
        scenario=(
            "Developer using Kotlin's scope functions (let, run, also) with nullable "
            "types. After a null check (user != null, user?.let), the type is smart "
            "cast to non-null within the safe scope. The control flow domain must "
            "track which variables have been null-checked in which branches."
        ),
        prompt="""Get the user's name from a nullable User?, returning "Unknown" if null.
Use safe call with Elvis operator (?:) or let{} scope function.

val user: User? = findUser(id)
return """,
        spec=ConstraintSpec(
            language="kotlin",
            # Regex enforces safe call pattern (?.let, ?., ?: or null check with if)
            regex=r"(?:user\?\.|if\s*\(\s*user\s*!=\s*null\s*\))",
            ebnf=r'''
root ::= let_elvis | safe_elvis | if_else | run_elvis
let_elvis ::= "user?.let { it.name } ?: \"Unknown\""
safe_elvis ::= "user?.name ?: \"Unknown\""
if_else ::= "if (user != null) user.name else \"Unknown\""
run_elvis ::= "user?.run { name } ?: \"Unknown\""
''',
            expected_type="String",
            type_bindings=[
                TypeBinding(name="user", type_expr="User?", scope="local"),
            ],
            class_definitions=[
                ClassDefinition(
                    name="User",
                    instance_vars=(
                        TypeBinding(name="name", type_expr="String"),
                        TypeBinding(name="email", type_expr="String"),
                    ),
                )
            ],
            control_flow=ControlFlowContext(
                function_name="getUserName",
                expected_return_type="String",
            ),
        ),
        expected_effect=(
            "Masks unsafe nullable access outside null-check scopes. After "
            "'user?.let { }', inside the lambda 'it' is User (non-null). "
            "After 'if (user != null)', inside the true branch 'user' is smart "
            "cast to User. Blocks unsafe access outside these scopes."
        ),
        valid_outputs=[
            'user?.let { it.name } ?: "Unknown"',
            'user?.name ?: "Unknown"',
            'if (user != null) user.name else "Unknown"',
            'user?.run { name } ?: "Unknown"',
        ],
        invalid_outputs=[
            "user.name",  # Unsafe: no null check
            "user.email",  # Unsafe: no null check
            'if (user != null) "name" else user.name',  # user.name in wrong branch
        ],
        tags=["controlflow", "smart-cast", "null-safety", "scope-functions", "kotlin"],
        language="kotlin",
        domain="controlflow",
    ),
]
