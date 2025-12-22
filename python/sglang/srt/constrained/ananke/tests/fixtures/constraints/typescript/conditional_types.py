# Copyright Rand Arete @ Ananke 2025
# Licensed under the Apache License, Version 2.0
"""Advanced conditional type examples for TypeScript.

This module contains deep-dive examples of advanced TypeScript type system
features including recursive conditional types, template literal inference,
and distributive conditional types. These examples demonstrate the most
sophisticated type-level programming patterns in modern TypeScript.
"""

from __future__ import annotations

try:
    from ..base import ConstraintExample
    from .....spec.constraint_spec import (
        ConstraintSpec,
        TypeBinding,
    )
except ImportError:
    from tests.fixtures.constraints.base import ConstraintExample
    from spec.constraint_spec import (
        ConstraintSpec,
        TypeBinding,
    )

TYPESCRIPT_CONDITIONAL_TYPES_EXAMPLES = [
    ConstraintExample(
        id="ts-conditional-001",
        name="Recursive Conditional Type Flattening",
        description="Recursively flatten nested array types to any depth",
        scenario=(
            "Developer implementing a DeepFlatten<T> utility type that recursively "
            "unwraps nested arrays to extract the innermost element type. For "
            "example: DeepFlatten<number[][][]> should yield number. This requires "
            "recursive conditional types with proper termination conditions."
        ),
        prompt="""Write a DeepFlatten<T> utility type that recursively extracts the innermost type
from nested arrays. DeepFlatten<number[][][]> should yield number.
Use "T extends Array<infer U> ? DeepFlatten<U> : T" for the recursive pattern.

""",
        spec=ConstraintSpec(
            language="typescript",
            expected_type="DeepFlatten<T>",
            # Regex enforces recursive conditional type with Array<infer U> pattern
            regex=r"^type\s+DeepFlatten\s*<\s*T\s*>\s*=\s*T\s+extends\s+(?:Array<infer\s+\w+>|\(\s*infer\s+\w+\s*\)\[\])",
            ebnf=r'''
root ::= array_infer_style | parens_style | nested_check
array_infer_style ::= "type DeepFlatten<T> = T extends Array<infer U> ? DeepFlatten<U> : T;"
parens_style ::= "type DeepFlatten<T> = T extends (infer U)[] ? DeepFlatten<U> : T;"
nested_check ::= "type DeepFlatten<T> = T extends Array<infer U> ? U extends Array<any> ? DeepFlatten<U> : U : T;"
''',
            type_bindings=[
                TypeBinding(name="T", type_expr="any", scope="type_parameter"),
            ],
            type_aliases={
                "DeepFlatten": "T extends Array<infer U> ? DeepFlatten<U> : T",
            },
        ),
        expected_effect=(
            "Masks tokens that would create non-recursive or incorrectly recursive "
            "type definitions. Ensures the type recursively checks if T is an array, "
            "extracts the element type with 'infer U', and recursively applies "
            "DeepFlatten to U. Base case returns T when it's not an array."
        ),
        valid_outputs=[
            """type DeepFlatten<T> = T extends Array<infer U> ? DeepFlatten<U> : T;

// Usage examples:
type Test1 = DeepFlatten<number[][][]>; // number
type Test2 = DeepFlatten<string[][]>; // string
type Test3 = DeepFlatten<boolean>; // boolean""",
            """type DeepFlatten<T> = T extends (infer U)[] ? DeepFlatten<U> : T;

// With readonly arrays:
type DeepFlattenReadonly<T> = T extends ReadonlyArray<infer U> ? DeepFlattenReadonly<U> : T;""",
            """type DeepFlatten<T> = T extends Array<infer U>
  ? U extends Array<any>
    ? DeepFlatten<U>
    : U
  : T;""",
        ],
        invalid_outputs=[
            """type DeepFlatten<T> = T extends Array<infer U> ? U : T;""",  # Not recursive
            """type DeepFlatten<T> = T[number];""",  # Index access, doesn't work for nested
            """type DeepFlatten<T> = T extends any[] ? T[0] : T;""",  # Only unwraps one level
            """type DeepFlatten<T> = T;""",  # Identity, doesn't flatten
        ],
        tags=["types", "conditional", "recursive", "advanced", "utility-types"],
        language="typescript",
        domain="types",
    ),
    ConstraintExample(
        id="ts-conditional-002",
        name="Template Literal Inference Pattern",
        description="Extract parts from template literal types using inference",
        scenario=(
            "Developer implementing a type that parses a route pattern string like "
            "'/users/:userId/posts/:postId' and extracts the parameter names as a "
            "union type 'userId' | 'postId'. This uses template literal types with "
            "recursive inference to extract all :param patterns."
        ),
        prompt="""Write ExtractRouteParams<T> that extracts parameter names from route strings.
For "/users/:userId/posts/:postId" should return 'userId' | 'postId'.
Use template literal inference: `${string}:${infer Param}/${infer Rest}`.

""",
        spec=ConstraintSpec(
            language="typescript",
            expected_type="ExtractRouteParams<T>",
            # Regex enforces template literal inference with :${infer pattern for route params
            regex=r"type\s+(?:ExtractRouteParams|ExtractParam|SplitRoute)\s*<[\s\S]*=[\s\S]*:\$\{infer\s+\w+\}",
            ebnf=r'''
root ::= direct_style | string_style | split_style
direct_style ::= "type ExtractRouteParams<T extends string> = T extends `${infer _Start}:${infer Param}/${infer Rest}` ? Param | ExtractRouteParams<`/${Rest}`> : T extends `${infer _Start}:${infer Param}` ? Param : never;"
string_style ::= "type ExtractRouteParams<T extends string> = T extends `${string}:${infer Param}/${infer Rest}` ? Param | ExtractRouteParams<Rest> : T extends `${string}:${infer Param}` ? Param : never;"
split_style ::= "type ExtractParam<T extends string> = T extends `:${infer Param}` ? Param : never;" nl "type SplitRoute<T extends string> = T extends `${infer First}/${infer Rest}` ? ExtractParam<First> | SplitRoute<Rest> : ExtractParam<T>;" nl "type ExtractRouteParams<T extends string> = SplitRoute<T>;"
nl ::= "\n"
''',
            type_bindings=[
                TypeBinding(name="T", type_expr="string", scope="type_parameter"),
            ],
            type_aliases={
                "ExtractRouteParams": """T extends `${infer _Start}:${infer Param}/${infer Rest}`
  ? Param | ExtractRouteParams<`/${Rest}`>
  : T extends `${infer _Start}:${infer Param}`
  ? Param
  : never""",
            },
        ),
        expected_effect=(
            "Masks tokens that would create non-recursive or incorrect template "
            "literal inference patterns. Ensures the type matches :param patterns, "
            "extracts the parameter name with 'infer Param', and recursively "
            "processes the rest of the string. Handles both intermediate and "
            "terminal parameters correctly."
        ),
        valid_outputs=[
            """type ExtractRouteParams<T extends string> =
  T extends `${infer _Start}:${infer Param}/${infer Rest}`
    ? Param | ExtractRouteParams<`/${Rest}`>
    : T extends `${infer _Start}:${infer Param}`
    ? Param
    : never;

// Usage:
type Params = ExtractRouteParams<'/users/:userId/posts/:postId'>; // 'userId' | 'postId'
type SingleParam = ExtractRouteParams<'/users/:id'>; // 'id'""",
            """type ExtractRouteParams<T extends string> =
  T extends `${string}:${infer Param}/${infer Rest}`
    ? Param | ExtractRouteParams<Rest>
    : T extends `${string}:${infer Param}`
    ? Param
    : never;""",
            """type ExtractParam<T extends string> =
  T extends `:${infer Param}` ? Param : never;

type SplitRoute<T extends string> =
  T extends `${infer First}/${infer Rest}`
    ? ExtractParam<First> | SplitRoute<Rest>
    : ExtractParam<T>;

type ExtractRouteParams<T extends string> = SplitRoute<T>;""",
        ],
        invalid_outputs=[
            """type ExtractRouteParams<T extends string> = T extends `:${infer Param}` ? Param : never;""",  # Doesn't handle paths
            """type ExtractRouteParams<T extends string> = string;""",  # Too generic
            """type ExtractRouteParams<T extends string> = T extends `${infer Param}` ? Param : never;""",  # Doesn't extract :param pattern
            """type ExtractRouteParams<T extends string> = T;""",  # Identity, doesn't extract
        ],
        tags=["types", "template-literals", "inference", "recursive", "parsing"],
        language="typescript",
        domain="types",
    ),
    ConstraintExample(
        id="ts-conditional-003",
        name="Distributive Conditional Types",
        description="Apply conditional type distributively over union types",
        scenario=(
            "Developer implementing a ToArray<T> utility that wraps each member of "
            "a union type in an array. For example: ToArray<string | number> should "
            "yield string[] | number[], not (string | number)[]. This demonstrates "
            "how conditional types distribute over unions when T is a naked type "
            "parameter."
        ),
        prompt="""Write ToArray<T> that wraps union members individually in arrays.
ToArray<string | number> should yield string[] | number[] (not (string|number)[]).
The trick is using "T extends any ? T[] : never" - naked T causes distribution.

""",
        spec=ConstraintSpec(
            language="typescript",
            expected_type="ToArray<T>",
            # Regex enforces distributive conditional type with T extends pattern yielding T[] or Array<T>
            regex=r"type\s+ToArray\s*<\s*T\s*>\s*=\s*T\s+extends\s+\w+[\s\S]*\?\s*(?:T\[\]|Array<T>|\w+\[\])",
            ebnf=r'''
root ::= basic_dist | infer_dist | array_generic
basic_dist ::= "type ToArray<T> = T extends any ? T[] : never;"
infer_dist ::= "type ToArray<T> = T extends infer U ? U[] : never;"
array_generic ::= "type ToArray<T> = T extends any ? Array<T> : never;"
''',
            type_bindings=[
                TypeBinding(name="T", type_expr="any", scope="type_parameter"),
            ],
            type_aliases={
                "ToArray": "T extends any ? T[] : never",
                "NonDistributive": "[T] extends [any] ? T[] : never",
            },
        ),
        expected_effect=(
            "Masks tokens that would create non-distributive conditional types or "
            "incorrectly wrap union types. Ensures the naked type parameter T "
            "causes distribution over unions. Demonstrates the difference between "
            "distributive (T extends any) and non-distributive ([T] extends [any]) "
            "patterns."
        ),
        valid_outputs=[
            """type ToArray<T> = T extends any ? T[] : never;

// Distributive behavior:
type Test1 = ToArray<string | number>; // string[] | number[]
type Test2 = ToArray<'a' | 'b' | 'c'>; // 'a'[] | 'b'[] | 'c'[]

// Non-distributive for comparison:
type ToArrayNonDist<T> = [T] extends [any] ? T[] : never;
type Test3 = ToArrayNonDist<string | number>; // (string | number)[]""",
            """type ToArray<T> = T extends infer U ? U[] : never;

// With additional constraint:
type ToArrayIfNotNull<T> = T extends null ? never : T[];""",
            """type ToArray<T> = T extends any ? Array<T> : never;

// More complex distribution:
type WrapInPromise<T> = T extends any ? Promise<T> : never;
type Test = WrapInPromise<string | number>; // Promise<string> | Promise<number>""",
        ],
        invalid_outputs=[
            """type ToArray<T> = T[];""",  # Not distributive, yields (string | number)[]
            """type ToArray<T> = [T] extends [any] ? T[] : never;""",  # Non-distributive due to tuple wrapper
            """type ToArray<T> = Array<T>;""",  # Not distributive, same as T[]
            """type ToArray<T> = T extends any ? [T] : never;""",  # Wraps in tuple, not array
        ],
        tags=["types", "conditional", "distributive", "unions", "advanced"],
        language="typescript",
        domain="types",
    ),
    ConstraintExample(
        id="ts-conditional-004",
        name="Mapped Types with Key Remapping",
        description="Transform object keys using template literals and conditional types",
        scenario=(
            "Developer creating a utility type that converts object properties to "
            "getter methods. For example: { name: string, age: number } becomes "
            "{ getName: () => string, getAge: () => number }. Uses mapped types "
            "with 'as' clause for key remapping and template literals."
        ),
        prompt="""Write Getters<T> that transforms { name: string } into { getName: () => string }.
Use mapped type key remapping: [K in keyof T as `get${Capitalize<K & string>}`].
The "K & string" constrains K to string keys for Capitalize to work.

""",
        spec=ConstraintSpec(
            language="typescript",
            expected_type="Getters<T>",
            # Regex enforces mapped type with key remapping (as clause) and get prefix
            regex=r"type\s+Getters\s*<\s*T\s*>\s*=\s*\{[\s\S]*\[\s*K\s+in\s+keyof\s+T\s+as[\s\S]*`get\$\{Capitalize<",
            ebnf=r'''
root ::= basic_getters | flipped_and | conditional_getters
basic_getters ::= "type Getters<T> = { [K in keyof T as `get${Capitalize<K & string>}`]: () => T[K]; };"
flipped_and ::= "type Getters<T> = { [K in keyof T as `get${Capitalize<string & K>}`]: () => T[K]; };"
conditional_getters ::= "type Getters<T> = { readonly [K in keyof T as K extends string ? `get${Capitalize<K>}` : never]: () => T[K]; };"
''',
            type_bindings=[
                TypeBinding(name="T", type_expr="object", scope="type_parameter"),
            ],
            type_aliases={
                "Getters": "{ [K in keyof T as `get${Capitalize<K & string>}`]: () => T[K] }",
            },
        ),
        expected_effect=(
            "Masks tokens that would create mapped types without proper key "
            "remapping. Ensures keys are transformed using template literal type "
            "with Capitalize utility. Blocks patterns that don't use 'as' clause "
            "or don't properly constrain K to string (K & string)."
        ),
        valid_outputs=[
            """type Getters<T> = {
  [K in keyof T as `get${Capitalize<K & string>}`]: () => T[K];
};

interface Person {
  name: string;
  age: number;
}

type PersonGetters = Getters<Person>;
// { getName: () => string; getAge: () => number; }""",
            """type Getters<T> = {
  [K in keyof T as `get${Capitalize<string & K>}`]: () => T[K];
};

// With setters too:
type Accessors<T> = {
  [K in keyof T as `get${Capitalize<K & string>}`]: () => T[K];
} & {
  [K in keyof T as `set${Capitalize<K & string>}`]: (value: T[K]) => void;
};""",
            """type Getters<T> = {
  readonly [K in keyof T as K extends string ? `get${Capitalize<K>}` : never]: () => T[K];
};""",
        ],
        invalid_outputs=[
            """type Getters<T> = {
  [K in keyof T]: () => T[K];
};""",  # No key remapping, keeps original keys
            """type Getters<T> = {
  [K in keyof T as `get${K}`]: () => T[K];
};""",  # Missing Capitalize
            """type Getters<T> = {
  [K in keyof T as `get${Capitalize<K>}`]: () => T[K];
};""",  # Missing K & string constraint
            """type Getters<T> = {
  [K in keyof T as string]: () => T[K];
};""",  # Loses connection between key and value
        ],
        tags=["types", "mapped", "key-remapping", "template-literals", "advanced"],
        language="typescript",
        domain="types",
    ),
    ConstraintExample(
        id="ts-conditional-005",
        name="Conditional Type with Constraint Inference",
        description="Extract constraint-respecting types using conditional inference",
        scenario=(
            "Developer implementing a GetRequired<T> utility that extracts only the "
            "required (non-optional) properties from an object type. For example: "
            "{ a: string; b?: number; c: boolean } becomes { a: string; c: boolean }. "
            "This uses conditional types to test if a key is required."
        ),
        prompt="""Write GetRequired<T> that keeps only required properties, filtering out optional ones.
{ a: string; b?: number } becomes { a: string }. The trick is checking if
{} extends Pick<T, K> - if true, K is optional and should be filtered out.

""",
        spec=ConstraintSpec(
            language="typescript",
            expected_type="GetRequired<T>",
            # Regex enforces required key extraction (various patterns including {} extends Pick, Record, or -?)
            regex=r"type\s+(?:RequiredKeys|GetRequired)\s*<\s*T\s*>[\s\S]*\[\s*K\s+in\s+keyof\s+T",
            ebnf=r'''
root ::= two_type_pick | single_type_as | record_style
two_type_pick ::= "type RequiredKeys<T> = { [K in keyof T]-?: {} extends Pick<T, K> ? never : K; }[keyof T];" nl "type GetRequired<T> = Pick<T, RequiredKeys<T>>;"
single_type_as ::= "type GetRequired<T> = { [K in keyof T as {} extends Pick<T, K> ? never : K]: T[K]; };"
record_style ::= "type RequiredKeys<T> = { [K in keyof T]: T extends Record<K, T[K]> ? K : never; }[keyof T];" nl "type GetRequired<T> = { [K in RequiredKeys<T>]: T[K]; };"
nl ::= "\n"
''',
            type_bindings=[
                TypeBinding(name="T", type_expr="object", scope="type_parameter"),
            ],
            type_aliases={
                "RequiredKeys": "{ [K in keyof T]-?: {} extends Pick<T, K> ? never : K }[keyof T]",
                "GetRequired": "Pick<T, RequiredKeys<T>>",
            },
        ),
        expected_effect=(
            "Masks tokens that would create incorrect required key detection logic. "
            "Ensures the type uses the -? modifier to remove optionality temporarily, "
            "tests if {} extends Pick<T, K> to detect optional properties, and uses "
            "indexed access [keyof T] to extract the union of required keys."
        ),
        valid_outputs=[
            """type RequiredKeys<T> = {
  [K in keyof T]-?: {} extends Pick<T, K> ? never : K;
}[keyof T];

type GetRequired<T> = Pick<T, RequiredKeys<T>>;

interface Example {
  a: string;
  b?: number;
  c: boolean;
}

type Required = GetRequired<Example>; // { a: string; c: boolean; }""",
            """type GetRequired<T> = {
  [K in keyof T as {} extends Pick<T, K> ? never : K]: T[K];
};""",
            """type RequiredKeys<T> = {
  [K in keyof T]: T extends Record<K, T[K]> ? K : never;
}[keyof T];

type GetRequired<T> = {
  [K in RequiredKeys<T>]: T[K];
};""",
        ],
        invalid_outputs=[
            """type GetRequired<T> = Required<T>;""",  # Makes all required, doesn't extract
            """type GetRequired<T> = {
  [K in keyof T]: T[K];
};""",  # Identity, doesn't filter optional
            """type GetRequired<T> = Pick<T, keyof T>;""",  # Doesn't filter optional keys
            """type GetRequired<T> = Omit<T, OptionalKeys<T>>;""",  # Assumes OptionalKeys exists
        ],
        tags=["types", "conditional", "utility-types", "inference", "advanced"],
        language="typescript",
        domain="types",
    ),
]
