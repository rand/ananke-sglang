# Copyright Rand Arete @ Ananke 2025
# Licensed under the Apache License, Version 2.0
"""Semantic constraint examples for Kotlin.

This module contains realistic examples of semantic constraints that
demonstrate how Ananke's SemanticDomain masks tokens to enforce Kotlin's
runtime contracts, collection bounds, and coroutine cancellation semantics.
"""

from __future__ import annotations

try:
    from ..base import ConstraintExample
    from .....spec.constraint_spec import (
        ConstraintSpec,
        TypeBinding,
        FunctionSignature,
        SemanticConstraint,
    )
except ImportError:
    from tests.fixtures.constraints.base import ConstraintExample
    from spec.constraint_spec import (
        ConstraintSpec,
        TypeBinding,
        FunctionSignature,
        SemanticConstraint,
    )

KOTLIN_SEMANTICS_EXAMPLES = [
    ConstraintExample(
        id="kt-sem-001",
        name="Non-Null Contract with require/check",
        description="Enforce Kotlin contracts with require/check/checkNotNull",
        scenario=(
            "Developer using Kotlin's contract functions (require, check, checkNotNull) "
            "to enforce preconditions and invariants. These functions have compiler "
            "contracts that enable smart casts and null-safety guarantees. After "
            "'requireNotNull(value)', the compiler knows value is non-null."
        ),
        prompt="""Validate that the input parameter is not null using Kotlin's contract functions.
Use requireNotNull() or require() - after these calls, the compiler knows input is non-null.

fun processInput(input: String?): String {
    """,
        spec=ConstraintSpec(
            language="kotlin",
            # Regex enforces require/check contract patterns (including return statements)
            regex=r"(?:requireNotNull|checkNotNull|require)\s*\(",
            ebnf=r'''
root ::= require_not_null | check_not_null | require_cond | return_require
require_not_null ::= "requireNotNull(input) { \"Input cannot be null\" }"
check_not_null ::= "checkNotNull(input)"
require_cond ::= "require(input != null)"
return_require ::= "return requireNotNull(input).uppercase()"
''',
            expected_type="String",
            type_bindings=[
                TypeBinding(name="input", type_expr="String?", scope="parameter"),
            ],
            semantic_constraints=[
                SemanticConstraint(
                    kind="precondition",
                    expression="input != null",
                    scope="processInput",
                    variables=("input",),
                ),
            ],
            function_signatures=[
                FunctionSignature(
                    name="processInput",
                    params=(TypeBinding(name="input", type_expr="String?"),),
                    return_type="String",
                ),
            ],
        ),
        expected_effect=(
            "Masks tokens that don't enforce the non-null precondition. Allows "
            "requireNotNull(input), require(input != null), checkNotNull(input). "
            "After these calls, masks unsafe nullable operations since the compiler "
            "knows the value is non-null via contracts."
        ),
        valid_outputs=[
            "requireNotNull(input) { \"Input cannot be null\" }",
            "checkNotNull(input)",
            "require(input != null)",
            "return requireNotNull(input).uppercase()",
        ],
        invalid_outputs=[
            "return input.uppercase()",  # Unsafe: input is nullable
            "if (input != null) return input else throw Exception()",  # Verbose
        ],
        tags=["semantics", "contracts", "preconditions", "null-safety", "kotlin"],
        language="kotlin",
        domain="semantics",
    ),
    ConstraintExample(
        id="kt-sem-002",
        name="Collection Bounds with indices/getOrNull",
        description="Enforce safe collection access with bounds checking",
        scenario=(
            "Developer accessing collection elements safely using Kotlin's "
            "safe collection operators. Unlike Java, Kotlin provides getOrNull, "
            "getOrElse, indices property, and elementAtOrNull to avoid "
            "IndexOutOfBoundsException. Code should prefer these safe APIs."
        ),
        prompt="""Access an element from a List<String> at a given index safely.
Use getOrNull() or check with 'in items.indices' to avoid IndexOutOfBoundsException.

val items: List<String> = ...
val index: Int = ...
return """,
        spec=ConstraintSpec(
            language="kotlin",
            # Regex enforces safe collection access patterns
            regex=r"^items\.(?:getOrNull|getOrElse|elementAtOrNull)\s*\(|^if\s*\(\s*index\s+in\s+items\.indices\s*\)",
            ebnf=r'''
root ::= get_or_null | indices_check | element_or_null | get_or_else
get_or_null ::= "items.getOrNull(index)"
indices_check ::= "if (index in items.indices) items[index] else null"
element_or_null ::= "items.elementAtOrNull(index)"
get_or_else ::= "items.getOrElse(index) { null }"
''',
            expected_type="String?",
            type_bindings=[
                TypeBinding(name="items", type_expr="List<String>", scope="local"),
                TypeBinding(name="index", type_expr="Int", scope="local"),
            ],
            semantic_constraints=[
                SemanticConstraint(
                    kind="precondition",
                    expression="index >= 0",
                    scope="getItem",
                    variables=("index",),
                ),
                SemanticConstraint(
                    kind="invariant",
                    expression="index in items.indices || result == null",
                    scope="getItem",
                    variables=("index", "items", "result"),
                ),
            ],
        ),
        expected_effect=(
            "Masks unsafe collection access patterns. Prefers getOrNull(index) "
            "over get(index), 'index in items.indices' checks over manual "
            "comparisons, elementAtOrNull over elementAt. Blocks direct array-style "
            "access items[index] without bounds checking."
        ),
        valid_outputs=[
            "items.getOrNull(index)",
            "if (index in items.indices) items[index] else null",
            "items.elementAtOrNull(index)",
            "items.getOrElse(index) { null }",
        ],
        invalid_outputs=[
            "items[index]",  # Unsafe: no bounds check
            "items.get(index)",  # Unsafe: throws exception
            "if (index >= 0 && index < items.size) items[index] else null",  # Verbose
        ],
        tags=["semantics", "collections", "bounds-checking", "safety", "kotlin"],
        language="kotlin",
        domain="semantics",
    ),
    ConstraintExample(
        id="kt-sem-003",
        name="Coroutine Cancellation Check",
        description="Enforce cooperative cancellation with isActive/ensureActive",
        scenario=(
            "Developer writing a long-running suspend function that should respect "
            "coroutine cancellation. Kotlin coroutines are cooperatively cancelled, "
            "meaning the code must explicitly check for cancellation using "
            "isActive, ensureActive(), or yield(). Tight loops without cancellation "
            "checks can block cancellation."
        ),
        prompt="""Complete this Kotlin suspend function that processes items with cancellation support:

for (item in items) {
    """,
        spec=ConstraintSpec(
            language="kotlin",
            # Regex enforces cancellation check at start of loop body
            regex=r"^if\s*\(\s*!isActive\s*\)",
            ebnf=r'''
root ::= map_ensure | for_isactive | map_yield
map_ensure ::= "items.map { item ->" nl "                ensureActive()" nl "                processItem(item)" nl "            }"
for_isactive ::= "for (item in items) {" nl "                if (!isActive) break" nl "                processItem(item)" nl "            }"
map_yield ::= "items.map { item ->" nl "                yield()" nl "                processItem(item)" nl "            }"
nl ::= "\n"
''',
            expected_type="List<Result>",
            type_bindings=[
                TypeBinding(name="items", type_expr="List<Item>", scope="parameter"),
            ],
            function_signatures=[
                FunctionSignature(
                    name="processItems",
                    params=(TypeBinding(name="items", type_expr="List<Item>"),),
                    return_type="List<Result>",
                    is_async=True,  # suspend function
                ),
            ],
            semantic_constraints=[
                SemanticConstraint(
                    kind="invariant",
                    expression="coroutineContext[Job]?.isActive == true",
                    scope="processItems",
                    variables=("coroutineContext",),
                ),
            ],
            control_flow=None,
        ),
        expected_effect=(
            "Masks tight loops without cancellation checks in suspend functions. "
            "Requires periodic calls to ensureActive(), isActive checks, or yield() "
            "within loops. Blocks simple for/while loops over large collections "
            "without cancellation support."
        ),
        valid_outputs=[
            """if (!isActive) break
                processItem(item)
            }""",
            """if (!isActive) return@processItems emptyList()
                processItem(item)
            }""",
            """if (!isActive) throw CancellationException()
                processItem(item)
            }""",
        ],
        invalid_outputs=[
            "items.map { processItem(it) }",  # No cancellation check
            "for (item in items) { processItem(item) }",  # No cancellation check
        ],
        tags=["semantics", "coroutines", "cancellation", "cooperative", "kotlin"],
        language="kotlin",
        domain="semantics",
    ),
]
