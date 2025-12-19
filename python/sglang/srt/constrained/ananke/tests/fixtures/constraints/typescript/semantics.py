# Copyright Rand Arete @ Ananke 2025
# Licensed under the Apache License, Version 2.0
"""Semantic constraint examples for TypeScript.

This module contains realistic examples of TypeScript semantic constraints
demonstrating non-null assertion justification, readonly enforcement, and
exhaustiveness checks with type narrowing.
"""

from __future__ import annotations

try:
    from ..base import ConstraintExample
    from .....spec.constraint_spec import (
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

TYPESCRIPT_SEMANTICS_EXAMPLES = [
    ConstraintExample(
        id="ts-semantics-001",
        name="Non-Null Assertion Justification",
        description="Justify use of non-null assertion operator (!)",
        scenario=(
            "Developer using the non-null assertion operator (!) to tell TypeScript "
            "that a potentially null/undefined value is definitely defined. This "
            "should only be used when there's a runtime guarantee (like after a "
            "null check, or when dealing with DOM elements known to exist). The "
            "constraint ensures the assertion is justified by context."
        ),
        prompt="""Write code that handles potentially null values safely. Instead of using element!.property
directly, first check for null: "if (element !== null) { element.style... }".
Or use a guard: "if (!element) throw new Error(...)".

const element = document.getElementById('app');
""",
        spec=ConstraintSpec(
            language="typescript",
            # Regex enforces null check pattern (if check, guard check, or function with null check)
            regex=r"(?:if\s*\(\s*(?:\w+\s*[!=]==?\s*(?:null|undefined)|!\s*\w+)|function\s+\w+\s*\([^)]*\)[\s\S]*if\s*\(\s*\w+\s*===\s*undefined\s*\))",
            ebnf=r'''
root ::= element_check | guard_throw | function_check
element_check ::= "const element = document.getElementById('app');" nl "if (element !== null) {" nl "  element.style.color = 'red';" nl "}"
guard_throw ::= "const element = document.getElementById('app');" nl "if (!element) throw new Error('Element not found');" nl "element.style.color = 'red';"
function_check ::= "function process(user: User | undefined): string {" nl "  if (user === undefined) {" nl "    return 'No user';" nl "  }" nl "  return user.name;" nl "}"
nl ::= "\n"
''',
            type_bindings=[
                TypeBinding(
                    name="element",
                    type_expr="HTMLElement | null",
                    scope="local",
                ),
                TypeBinding(
                    name="user",
                    type_expr="User | undefined",
                    scope="local",
                ),
            ],
            semantic_constraints=[
                SemanticConstraint(
                    kind="precondition",
                    expression="element !== null",
                    scope="local",
                    variables=("element",),
                ),
            ],
            type_aliases={
                "User": "{ id: string; name: string }",
            },
        ),
        expected_effect=(
            "Masks tokens that would use non-null assertion (!) without proper "
            "justification. Requires prior null check or context proving the value "
            "is non-null. Blocks unsafe assertions that could cause runtime errors. "
            "Allows assertions after explicit checks like 'if (x !== null)'."
        ),
        valid_outputs=[
            """const element = document.getElementById('app');
if (element !== null) {
  element.style.color = 'red';
}""",
            """const element = document.getElementById('app');
if (!element) throw new Error('Element not found');
element.style.color = 'red';""",
            """function process(user: User | undefined): string {
  if (user === undefined) {
    return 'No user';
  }
  return user.name;
}""",
        ],
        invalid_outputs=[
            """const element = document.getElementById('app');
element!.style.color = 'red';""",  # Unsafe assertion without check
            """function process(user: User | undefined): string {
  return user!.name;
}""",  # Assertion without validation
            """const element = document.querySelector('.item');
element!.textContent = 'text';""",  # querySelector can return null
        ],
        tags=["semantics", "non-null", "assertions", "safety"],
        language="typescript",
        domain="semantics",
    ),
    ConstraintExample(
        id="ts-semantics-002",
        name="Readonly Enforcement",
        description="Enforce readonly property constraints",
        scenario=(
            "Developer working with readonly properties and arrays where mutation "
            "should be prevented. TypeScript's readonly modifier and ReadonlyArray "
            "type prevent accidental mutations. The constraint ensures code doesn't "
            "attempt to modify readonly data structures."
        ),
        prompt="""Write a function that takes Readonly<Config> or ReadonlyArray<string> and returns
a modified version without mutating the input. Use spread: "{ ...config, timeout: 5000 }"
or filter: "items.filter(x => x.length > 0)". Never assign to readonly properties.

""",
        spec=ConstraintSpec(
            language="typescript",
            # Regex enforces immutable patterns (spread, filter, freeze) with Readonly/ReadonlyArray types
            regex=r"function\s+\w+\s*\([^)]*(?:Readonly<\w+>|ReadonlyArray<\w+>)[^)]*\)[\s\S]*(?:return\s+(?:\{[\s\S]*\.\.\.|\w+\.filter|Object\.freeze))",
            ebnf=r'''
root ::= spread_config | filter_items | freeze_config
spread_config ::= "function processConfig(config: Readonly<Config>): Config {" nl "  return { ...config, timeout: config.timeout * 2 };" nl "}"
filter_items ::= "function filterItems(items: ReadonlyArray<string>): string[] {" nl "  return items.filter(item => item.length > 0);" nl "}"
freeze_config ::= "function updateConfig(config: Readonly<Config>): Readonly<Config> {" nl "  return Object.freeze({ ...config, retries: 5 });" nl "}"
nl ::= "\n"
''',
            type_bindings=[
                TypeBinding(
                    name="config",
                    type_expr="Readonly<Config>",
                    scope="parameter",
                    mutable=False,
                ),
                TypeBinding(
                    name="items",
                    type_expr="ReadonlyArray<string>",
                    scope="parameter",
                    mutable=False,
                ),
            ],
            semantic_constraints=[
                SemanticConstraint(
                    kind="invariant",
                    expression="not_mutated(config)",
                    scope="function",
                    variables=("config",),
                ),
                SemanticConstraint(
                    kind="invariant",
                    expression="not_mutated(items)",
                    scope="function",
                    variables=("items",),
                ),
            ],
            type_aliases={
                "Config": "{ apiUrl: string; timeout: number; retries: number }",
            },
        ),
        expected_effect=(
            "Masks tokens that would mutate readonly properties or arrays. Blocks "
            "assignments to readonly properties (config.timeout = 5000), mutating "
            "array methods (items.push(), items.sort()), and any operations that "
            "modify readonly data in place. Allows reading and creating new objects."
        ),
        valid_outputs=[
            """function processConfig(config: Readonly<Config>): Config {
  return { ...config, timeout: config.timeout * 2 };
}""",
            """function filterItems(items: ReadonlyArray<string>): string[] {
  return items.filter(item => item.length > 0);
}""",
            """function updateConfig(config: Readonly<Config>): Readonly<Config> {
  return Object.freeze({ ...config, retries: 5 });
}""",
        ],
        invalid_outputs=[
            """function processConfig(config: Readonly<Config>): void {
  config.timeout = 5000;
}""",  # Mutation of readonly property
            """function filterItems(items: ReadonlyArray<string>): void {
  items.push('new');
}""",  # Mutation of readonly array
            """function updateConfig(config: Readonly<Config>): void {
  config.apiUrl = 'new-url';
  config.retries = 3;
}""",  # Multiple readonly mutations
        ],
        tags=["semantics", "readonly", "immutability", "mutation"],
        language="typescript",
        domain="semantics",
    ),
    ConstraintExample(
        id="ts-semantics-003",
        name="Type Narrowing Exhaustiveness",
        description="Ensure exhaustive type narrowing with proper guards",
        scenario=(
            "Developer using type guards to narrow union types and ensure all "
            "variants are handled. After a type guard like 'typeof x === \"string\"', "
            "the type should be narrowed to string in that branch. The constraint "
            "ensures narrowing is complete and no cases are missed."
        ),
        prompt="""Write a function processValue(value: string | number | boolean): string that handles
all three types exhaustively. Use typeof checks for each case. Return string for all paths -
missing cases would cause implicit undefined return, violating the return type.

""",
        spec=ConstraintSpec(
            language="typescript",
            # Regex enforces type narrowing with typeof checks (if or switch)
            regex=r"function\s+\w+\s*\([^)]*string\s*\|[^)]*\)[\s\S]*typeof\s+\w+",
            ebnf=r'''
root ::= if_chain | switch_exhaustive
if_chain ::= "function processValue(value: string | number | boolean): string {" nl "  if (typeof value === 'string') {" nl "    return value.toUpperCase();" nl "  } else if (typeof value === 'number') {" nl "    return value.toString();" nl "  } else {" nl "    return value ? 'true' : 'false';" nl "  }" nl "}"
switch_exhaustive ::= "function processValue(value: string | number | boolean): string {" nl "  switch (typeof value) {" nl "    case 'string':" nl "      return value.toUpperCase();" nl "    case 'number':" nl "      return value.toString();" nl "    case 'boolean':" nl "      return value ? 'true' : 'false';" nl "    default:" nl "      const _exhaustive: never = value;" nl "      throw new Error('Unreachable');" nl "  }" nl "}"
nl ::= "\n"
''',
            type_bindings=[
                TypeBinding(
                    name="value",
                    type_expr="string | number | boolean",
                    scope="parameter",
                ),
            ],
            control_flow=ControlFlowContext(
                function_name="processValue",
                expected_return_type="string",
            ),
            semantic_constraints=[
                SemanticConstraint(
                    kind="postcondition",
                    expression="exhaustive(value)",
                    scope="function",
                    variables=("value",),
                ),
            ],
        ),
        expected_effect=(
            "Masks tokens that would create non-exhaustive type narrowing or miss "
            "union variants. Ensures type guards cover all possibilities (string, "
            "number, boolean). Blocks code that only checks some types and leaves "
            "others unhandled, which would cause the function to implicitly return "
            "undefined."
        ),
        valid_outputs=[
            """function processValue(value: string | number | boolean): string {
  if (typeof value === 'string') {
    return value.toUpperCase();
  } else if (typeof value === 'number') {
    return value.toString();
  } else {
    return value ? 'true' : 'false';
  }
}""",
            """function processValue(value: string | number | boolean): string {
  switch (typeof value) {
    case 'string':
      return value.toUpperCase();
    case 'number':
      return value.toString();
    case 'boolean':
      return value ? 'true' : 'false';
    default:
      const _exhaustive: never = value;
      throw new Error('Unreachable');
  }
}""",
        ],
        invalid_outputs=[
            """function processValue(value: string | number | boolean): string {
  if (typeof value === 'string') {
    return value.toUpperCase();
  }
  return value.toString();
}""",  # Assumes number/boolean both have toString, not explicit
            """function processValue(value: string | number | boolean): string {
  if (typeof value === 'string') {
    return value.toUpperCase();
  } else if (typeof value === 'number') {
    return value.toString();
  }
}""",  # Missing boolean case, implicitly returns undefined
            """function processValue(value: string | number | boolean): string {
  if (typeof value === 'string') {
    return value.toUpperCase();
  }
}""",  # Only handles string case, non-exhaustive
        ],
        tags=["semantics", "type-narrowing", "exhaustiveness", "type-guards"],
        language="typescript",
        domain="semantics",
    ),
]
