# Copyright Rand Arete @ Ananke 2025
# Licensed under the Apache License, Version 2.0
"""Control flow constraint examples for TypeScript.

This module contains realistic examples of TypeScript control flow constraints
demonstrating discriminated union exhaustiveness checking, async/await Promise
context handling, and never type usage in exhaustive checks.
"""

from __future__ import annotations

try:
    from ..base import ConstraintExample
    from .....spec.constraint_spec import (
        ConstraintSpec,
        TypeBinding,
        ControlFlowContext,
        FunctionSignature,
    )
except ImportError:
    from tests.fixtures.constraints.base import ConstraintExample
    from spec.constraint_spec import (
        ConstraintSpec,
        TypeBinding,
        ControlFlowContext,
        FunctionSignature,
    )

TYPESCRIPT_CONTROLFLOW_EXAMPLES = [
    ConstraintExample(
        id="ts-controlflow-001",
        name="Discriminated Union Exhaustiveness",
        description="Ensure all union variants are handled in switch statement",
        scenario=(
            "Developer implementing a reducer pattern with discriminated unions "
            "representing different action types. TypeScript's control flow analysis "
            "should ensure all action types are handled, with the default case "
            "using 'never' to catch unhandled variants at compile time."
        ),
        prompt="""Write a switch statement for a reducer handling Action types: 'increment', 'decrement', 'reset'.
Handle each case and use a default with "const _exhaustive: never = action" to catch
unhandled cases at compile time. Return the new state for each action.

switch (action.type) {
    """,
        spec=ConstraintSpec(
            language="typescript",
            # Regex enforces exhaustive switch with never assertion
            regex=r"^switch\s*\(\s*\w+\.type\s*\)[\s\S]*case\s+'increment'[\s\S]*case\s+'decrement'[\s\S]*case\s+'reset'",
            ebnf=r'''
root ::= "switch (action.type) {" nl cases default_case "}"
cases ::= case_increment case_decrement case_reset
case_increment ::= "  case 'increment':" nl "    return { count: state.count + action.amount };" nl
case_decrement ::= "  case 'decrement':" nl "    return { count: state.count - action.amount };" nl
case_reset ::= "  case 'reset':" nl "    return { count: 0 };" nl
default_case ::= "  default:" nl "    " default_body nl
default_body ::= "const _exhaustive: never = action;" nl "    return state;" | "throw new Error(`Unhandled action type: ${(action as any).type}`);"
nl ::= "\n"
''',
            type_bindings=[
                TypeBinding(
                    name="action",
                    type_expr="Action",
                    scope="parameter",
                ),
                TypeBinding(
                    name="state",
                    type_expr="State",
                    scope="parameter",
                ),
            ],
            type_aliases={
                "Action": "{ type: 'increment'; amount: number } | { type: 'decrement'; amount: number } | { type: 'reset' }",
                "State": "{ count: number }",
            },
            control_flow=ControlFlowContext(
                function_name="reducer",
                function_signature=FunctionSignature(
                    name="reducer",
                    params=(
                        TypeBinding(name="state", type_expr="State"),
                        TypeBinding(name="action", type_expr="Action"),
                    ),
                    return_type="State",
                ),
                expected_return_type="State",
            ),
        ),
        expected_effect=(
            "Masks tokens that would create non-exhaustive switch statements or "
            "miss handling a discriminated union variant. Ensures the default case "
            "assigns action to 'never' type, proving all variants are covered. "
            "Blocks code that doesn't handle all action.type values."
        ),
        valid_outputs=[
            """switch (action.type) {
  case 'increment':
    return { count: state.count + action.amount };
  case 'decrement':
    return { count: state.count - action.amount };
  case 'reset':
    return { count: 0 };
  default:
    const _exhaustive: never = action;
    return state;
}""",
            """switch (action.type) {
  case 'increment':
    return { count: state.count + action.amount };
  case 'decrement':
    return { count: state.count - action.amount };
  case 'reset':
    return { count: 0 };
  default:
    throw new Error(`Unhandled action type: ${(action as any).type}`);
}""",
        ],
        invalid_outputs=[
            """switch (action.type) {
  case 'increment':
    return { count: state.count + action.amount };
  case 'decrement':
    return { count: state.count - action.amount };
}""",  # Missing 'reset' case, non-exhaustive
            """switch (action.type) {
  case 'increment':
    return { count: state.count + action.amount };
  default:
    return state;
}""",  # Lumps all other cases into default, not exhaustive
            """if (action.type === 'increment') {
  return { count: state.count + action.amount };
}
return state;""",  # Only handles one case, not exhaustive
        ],
        tags=["controlflow", "discriminated-unions", "exhaustiveness", "switch"],
        language="typescript",
        domain="controlflow",
    ),
    ConstraintExample(
        id="ts-controlflow-002",
        name="Async/Await Promise Context",
        description="Enforce proper async/await usage and Promise type handling",
        scenario=(
            "Developer writing async function that fetches data from an API and "
            "processes it. Must properly await Promise values before using them, "
            "and ensure return type matches declared Promise<T>. TypeScript's "
            "control flow should track Promise contexts and prevent using Promise "
            "values as if they were resolved."
        ),
        prompt="""Write an async function fetchUser that takes userId: string and returns Promise<User>.
Use await on fetch() and response.json(). Don't forget: missing await causes the function
to return Promise<Promise<User>> instead of Promise<User>.

""",
        spec=ConstraintSpec(
            language="typescript",
            # Regex enforces async function with await on fetch
            regex=r"^async\s+function\s+\w+\s*\([^)]*\)\s*:\s*Promise<[\s\S]*await\s+fetch\s*\(",
            ebnf=r'''
root ::= "async function fetchUser(userId: string): Promise<User> {" nl body "}"
body ::= simple_body | error_check_body | try_catch_body
simple_body ::= "  const response = await fetch(`/api/users/${userId}`);" nl "  const data = await response.json();" nl "  return data;" nl
error_check_body ::= "  const response = await fetch(`/api/users/${userId}`);" nl "  if (!response.ok) throw new Error('Fetch failed');" nl "  return await response.json();" nl
try_catch_body ::= "  try {" nl "    const response = await fetch(`/api/users/${userId}`);" nl "    return await response.json();" nl "  } catch (error) {" nl "    throw new Error('Failed to fetch user');" nl "  }" nl
nl ::= "\n"
''',
            type_bindings=[
                TypeBinding(name="userId", type_expr="string", scope="parameter"),
                TypeBinding(name="response", type_expr="Promise<Response>", scope="local"),
                TypeBinding(name="data", type_expr="User", scope="local"),
            ],
            type_aliases={
                "User": "{ id: string; name: string; email: string }",
            },
            control_flow=ControlFlowContext(
                function_name="fetchUser",
                function_signature=FunctionSignature(
                    name="fetchUser",
                    params=(TypeBinding(name="userId", type_expr="string"),),
                    return_type="Promise<User>",
                    is_async=True,
                ),
                expected_return_type="Promise<User>",
                in_async_context=True,
            ),
        ),
        expected_effect=(
            "Masks tokens that would use Promise values without awaiting them, or "
            "return non-Promise values from async functions. Ensures fetch results "
            "are awaited before accessing properties. Blocks synchronous operations "
            "on Promise types."
        ),
        valid_outputs=[
            """async function fetchUser(userId: string): Promise<User> {
  const response = await fetch(`/api/users/${userId}`);
  const data = await response.json();
  return data;
}""",
            """async function fetchUser(userId: string): Promise<User> {
  const response = await fetch(`/api/users/${userId}`);
  if (!response.ok) throw new Error('Fetch failed');
  return await response.json();
}""",
            """async function fetchUser(userId: string): Promise<User> {
  try {
    const response = await fetch(`/api/users/${userId}`);
    return await response.json();
  } catch (error) {
    throw new Error('Failed to fetch user');
  }
}""",
        ],
        invalid_outputs=[
            """async function fetchUser(userId: string): Promise<User> {
  const response = fetch(`/api/users/${userId}`);
  return response.json();
}""",  # Missing await, returns Promise<Promise<User>>
            """function fetchUser(userId: string): Promise<User> {
  const response = await fetch(`/api/users/${userId}`);
  return response.json();
}""",  # await in non-async function
            """async function fetchUser(userId: string): Promise<User> {
  const response = fetch(`/api/users/${userId}`);
  return response.data;
}""",  # Accessing .data on Promise without await
        ],
        tags=["controlflow", "async", "await", "promises"],
        language="typescript",
        domain="controlflow",
    ),
    ConstraintExample(
        id="ts-controlflow-003",
        name="Never Type in Unreachable Code",
        description="Use 'never' type to mark unreachable code paths",
        scenario=(
            "Developer implementing error handling where certain code paths should "
            "be statically proven unreachable. Uses 'never' type annotations to "
            "make control flow guarantees explicit. TypeScript's control flow "
            "analysis ensures functions that never return are properly typed."
        ),
        prompt="""Write a function with return type "never" that always throws an error or loops forever.
Example: function fail(message: string): never { ... }. The function must not have any
reachable return path - TypeScript enforces this at the type level.

""",
        spec=ConstraintSpec(
            language="typescript",
            # Regex enforces function with never return type (throw, infinite loop, or process.exit)
            regex=r"function\s+\w+\s*\([^)]*\)\s*:\s*never\s*\{",
            ebnf=r'''
root ::= assert_never | fail_func | infinite_loop | process_exit
assert_never ::= "function assertNever(value: never): never {" nl "  throw new Error(`Unexpected value: ${value}`);" nl "}"
fail_func ::= "function fail(message: string): never {" nl "  throw new Error(message);" nl "}"
infinite_loop ::= "function infiniteLoop(): never {" nl "  while (true) {" nl "    console.log('running');" nl "  }" nl "}"
process_exit ::= "function processExit(): never {" nl "  process.exit(1);" nl "}"
nl ::= "\n"
''',
            type_bindings=[
                TypeBinding(name="error", type_expr="Error", scope="parameter"),
                TypeBinding(name="message", type_expr="string", scope="parameter"),
            ],
            control_flow=ControlFlowContext(
                function_name="assertNever",
                function_signature=FunctionSignature(
                    name="assertNever",
                    params=(TypeBinding(name="value", type_expr="never"),),
                    return_type="never",
                ),
                expected_return_type="never",
                reachable=False,
            ),
        ),
        expected_effect=(
            "Masks tokens that would create reachable return paths in functions "
            "marked with 'never' return type. Ensures functions that should never "
            "return (like error throwers) actually throw or loop infinitely. "
            "Blocks normal return statements or code after exhaustive checks."
        ),
        valid_outputs=[
            """function assertNever(value: never): never {
  throw new Error(`Unexpected value: ${value}`);
}""",
            """function fail(message: string): never {
  throw new Error(message);
}""",
            """function infiniteLoop(): never {
  while (true) {
    console.log('running');
  }
}""",
            """function processExit(): never {
  process.exit(1);
}""",
        ],
        invalid_outputs=[
            """function assertNever(value: never): never {
  console.error(`Unexpected value: ${value}`);
  return;
}""",  # Has reachable return path
            """function assertNever(value: never): never {
  if (value) {
    throw new Error('error');
  }
}""",  # Missing throw in else case, implicit return
            """function assertNever(value: never): never {
  console.error(value);
}""",  # No throw, function returns normally
        ],
        tags=["controlflow", "never", "exhaustiveness", "unreachable"],
        language="typescript",
        domain="controlflow",
    ),
]
