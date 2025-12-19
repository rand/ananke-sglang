# Copyright Rand Arete @ Ananke 2025
# Licensed under the Apache License, Version 2.0
"""Type constraint examples for TypeScript.

This module contains realistic examples of TypeScript type-level constraints
demonstrating how Ananke's TypeDomain enforces conditional types, mapped types,
and template literal types during code generation.
"""

from __future__ import annotations

try:
    from ..base import ConstraintExample
    from .....spec.constraint_spec import (
        ConstraintSpec,
        TypeBinding,
        FunctionSignature,
    )
except ImportError:
    from tests.fixtures.constraints.base import ConstraintExample
    from spec.constraint_spec import (
        ConstraintSpec,
        TypeBinding,
        FunctionSignature,
    )

TYPESCRIPT_TYPE_EXAMPLES = [
    ConstraintExample(
        id="ts-types-001",
        name="Conditional Type Extraction",
        description="Extract element type from array using conditional types",
        scenario=(
            "Developer writing a utility type to extract the element type from "
            "an array type. Given Array<string>, should produce string. Given "
            "Promise<number[]>, should extract number. Uses T extends Array<infer U> "
            "pattern for type inference."
        ),
        prompt="""Write a TypeScript utility type ArrayElement<T> that extracts the element type
from an array. Use conditional types with "T extends Array<infer U>" to infer the element.
If T is not an array, return never.

""",
        spec=ConstraintSpec(
            language="typescript",
            # Regex enforces conditional type with infer pattern for array extraction
            # Matches: T extends Array<infer U> OR T extends (infer U)[] OR T extends ReadonlyArray<infer U>
            regex=r"type\s+\w+<T>\s*=\s*T\s+extends\s+(?:Array<\s*infer\s+\w+\s*>|\(\s*infer\s+\w+\s*\)\[\]|ReadonlyArray<\s*infer\s+\w+\s*>)",
            ebnf=r'''
root ::= "type ArrayElement<T> = T extends " array_pattern " ? " infer_var " : never;"
array_pattern ::= "Array<infer " infer_var ">" | "(infer " infer_var ")[]" | "ReadonlyArray<infer " infer_var ">"
infer_var ::= [A-Z]
''',
            expected_type="U",
            type_bindings=[
                TypeBinding(
                    name="T", type_expr="Array<infer U>", scope="type_parameter"
                ),
            ],
            type_aliases={
                "ArrayElement": "T extends Array<infer U> ? U : never",
            },
        ),
        expected_effect=(
            "Masks tokens that would produce types inconsistent with the conditional "
            "type pattern. Ensures the inferred type U is properly extracted from the "
            "array structure. Blocks patterns that don't match Array<T> or don't use "
            "infer correctly."
        ),
        valid_outputs=[
            "type ArrayElement<T> = T extends Array<infer U> ? U : never;",
            "type ArrayElement<T> = T extends (infer U)[] ? U : never;",
            "type ArrayElement<T> = T extends ReadonlyArray<infer U> ? U : never;",
        ],
        invalid_outputs=[
            "type ArrayElement<T> = T[0];",  # Index access, not conditional
            "type ArrayElement<T> = T extends Array<U> ? U : never;",  # Missing infer
            "type ArrayElement<T> = T extends infer U ? U : never;",  # Not extracting from Array
            "type ArrayElement<T> = Array<T>;",  # Not a conditional type
        ],
        tags=["types", "conditional", "inference", "utility-types"],
        language="typescript",
        domain="types",
    ),
    ConstraintExample(
        id="ts-types-002",
        name="Mapped Type Transformation",
        description="Create readonly version of type using mapped types",
        scenario=(
            "Developer implementing a DeepReadonly<T> utility type that recursively "
            "makes all properties and nested properties readonly. This demonstrates "
            "mapped type patterns with recursive conditional types."
        ),
        prompt="""Write a DeepReadonly<T> utility type that recursively makes all properties readonly.
Use mapped types with "{ readonly [P in keyof T]: DeepReadonly<T[P]> }".
Must handle nested objects, not just top-level properties.

""",
        spec=ConstraintSpec(
            language="typescript",
            # Regex enforces mapped type with readonly modifier and recursive application
            # Matches various DeepReadonly patterns including conditional and non-conditional
            regex=r"type\s+DeepReadonly<T>\s*=[\s\S]*readonly\s+\[\s*\w+\s+in\s+keyof\s+T\s*\]",
            ebnf=r'''
root ::= "type DeepReadonly<T> = " body ";"
body ::= simple_mapped | conditional_mapped | nested_conditional
simple_mapped ::= "{ readonly [" param " in keyof T]: DeepReadonly<T[" param "]> }"
conditional_mapped ::= "T extends object ? { readonly [" param " in keyof T]: DeepReadonly<T[" param "]> } : T"
nested_conditional ::= "{ readonly [" param " in keyof T]: T[" param "] extends object ? DeepReadonly<T[" param "]> : T[" param "] }"
param ::= "P" | "K"
''',
            expected_type="DeepReadonly<T>",
            type_bindings=[
                TypeBinding(name="T", type_expr="object", scope="type_parameter"),
            ],
            type_aliases={
                "DeepReadonly": "{readonly [P in keyof T]: DeepReadonly<T[P]>}",
            },
        ),
        expected_effect=(
            "Masks tokens that would create mutable properties or fail to recurse "
            "into nested objects. Ensures all properties are marked readonly and "
            "the type recursively applies to nested object structures."
        ),
        valid_outputs=[
            "type DeepReadonly<T> = { readonly [P in keyof T]: DeepReadonly<T[P]> };",
            "type DeepReadonly<T> = T extends object ? { readonly [P in keyof T]: DeepReadonly<T[P]> } : T;",
            "type DeepReadonly<T> = { readonly [K in keyof T]: T[K] extends object ? DeepReadonly<T[K]> : T[K] };",
        ],
        invalid_outputs=[
            "type DeepReadonly<T> = { [P in keyof T]: T[P] };",  # Missing readonly
            "type DeepReadonly<T> = Readonly<T>;",  # Only shallow, not deep
            "type DeepReadonly<T> = { readonly [P in keyof T]: T[P] };",  # Not recursive
            "type DeepReadonly<T> = T;",  # Identity, doesn't transform
        ],
        tags=["types", "mapped", "readonly", "utility-types", "recursive"],
        language="typescript",
        domain="types",
    ),
    ConstraintExample(
        id="ts-types-003",
        name="Template Literal Type Pattern",
        description="Generate event handler type names from events",
        scenario=(
            "Developer creating a type system for React-like event handlers where "
            "event names like 'click', 'focus', 'blur' are automatically transformed "
            "to handler names like 'onClick', 'onFocus', 'onBlur' using template "
            "literal types with Capitalize utility."
        ),
        prompt="""Write an EventHandler<E> type that transforms event names to handler names.
Given 'click' should produce 'onClick'. Use template literal types with Capitalize:
`on${Capitalize<E>}`. The E parameter should extend string.

""",
        spec=ConstraintSpec(
            language="typescript",
            # Regex enforces template literal type with on prefix and Capitalize
            # Matches both direct template literal types and mapped types with template key remapping
            regex=r"type\s+\w+<\w+[\s\S]*=[\s\S]*`on\$\{Capitalize<",
            ebnf=r'''
root ::= simple_handler | mapped_handlers | handler_suffix
simple_handler ::= "type EventHandler<E extends string> = `on${Capitalize<E>}`;"
mapped_handlers ::= "type EventHandlers<Events> = { [K in keyof Events as `on${Capitalize<K & string>}`]: (event: Events[K]) => void };"
handler_suffix ::= "type Handler<E extends string> = `on${Capitalize<E>}Handler`;"
''',
            expected_type="EventHandler<E>",
            type_bindings=[
                TypeBinding(name="E", type_expr="string", scope="type_parameter"),
            ],
            type_aliases={
                "EventHandler": "`on${Capitalize<E>}`",
                "EventMap": "{ [K in keyof Events as `on${Capitalize<K>}`]: (event: Events[K]) => void }",
            },
        ),
        expected_effect=(
            "Masks tokens that don't follow the template literal type pattern. "
            "Ensures event handler names are correctly prefixed with 'on' and "
            "the first letter is capitalized. Blocks raw concatenation or incorrect "
            "transformation patterns."
        ),
        valid_outputs=[
            "type EventHandler<E extends string> = `on${Capitalize<E>}`;",
            "type EventHandlers<Events> = { [K in keyof Events as `on${Capitalize<K & string>}`]: (event: Events[K]) => void };",
            "type Handler<E extends string> = `on${Capitalize<E>}Handler`;",
        ],
        invalid_outputs=[
            "type EventHandler<E extends string> = `on${E}`;",  # Missing Capitalize
            "type EventHandler<E extends string> = E;",  # Not a template literal
            "type EventHandler<E extends string> = 'on' + E;",  # Runtime concat, not type-level
            "type EventHandler<E extends string> = string;",  # Too generic
        ],
        tags=["types", "template-literals", "string-manipulation", "events"],
        language="typescript",
        domain="types",
    ),
]
