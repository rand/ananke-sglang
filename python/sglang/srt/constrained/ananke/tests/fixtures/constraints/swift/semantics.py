# Copyright Rand Arete @ Ananke 2025
# Licensed under the Apache License, Version 2.0
"""Swift semantic constraint examples for Ananke.

This module contains realistic examples of semantic constraints in Swift,
demonstrating force unwrap justification, actor state consistency, and Sendable conformance.
"""

from __future__ import annotations

from typing import List

try:
    from ..base import ConstraintExample
    from ....spec.constraint_spec import (
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

# =============================================================================
# Semantic Constraint Examples
# =============================================================================

SWIFT_SEMANTIC_001 = ConstraintExample(
    id="swift-semantics-001",
    name="Force Unwrap Justification",
    description="Force unwrap only after precondition checks that guarantee non-nil",
    scenario=(
        "Developer force unwrapping an optional, which is normally discouraged. "
        "Force unwrap (!) is acceptable only when a precondition or prior check "
        "guarantees the value is non-nil. The constraint ensures this invariant."
    ),
    prompt="""Return value! only after checking it's non-nil with precondition() or assert().
Force unwrap without a prior check is unsafe.

func getValue(value: Int?) -> Int {
    """,
    spec=ConstraintSpec(
        language="swift",
        type_bindings=[
            TypeBinding(
                name="value",
                type_expr="Int?",
                scope="local",
            ),
        ],
        semantic_constraints=[
            SemanticConstraint(
                kind="precondition",
                expression="value != nil",
                scope="force_unwrap_context",
                variables=("value",),
            ),
        ],
        ebnf=r'''
root ::= check nl force_unwrap
check ::= "precondition(" identifier " != nil)" | "assert(" identifier " != nil, " qstring ")" | "guard " identifier " != nil else { fatalError() }"
force_unwrap ::= "return " identifier "!"
identifier ::= [a-z][a-zA-Z0-9]*
qstring ::= "'" [^']* "'"
nl ::= "\n"
''',
    ),
    expected_effect=(
        "Masks force unwrap tokens unless preceded by precondition/assert. "
        "Ensures force unwraps are justified with explicit nil checks."
    ),
    valid_outputs=[
        "precondition(value != nil)\nreturn value!",
        "assert(value != nil, 'Value must be set')\nreturn value!",
        "guard value != nil else { fatalError() }\nreturn value!",
    ],
    invalid_outputs=[
        "return value!",  # No precondition
        "if value != nil { return value! }",  # if is not precondition (use guard let instead)
        "assert(value != 0)\nreturn value!",  # Wrong assertion
    ],
    tags=["semantics", "safety", "force-unwrap", "preconditions"],
    language="swift",
    domain="semantics",
)

SWIFT_SEMANTIC_002 = ConstraintExample(
    id="swift-semantics-002",
    name="Actor State Consistency",
    description="Maintain actor state consistency across async suspension points",
    scenario=(
        "Developer modifying actor state across await suspension points. "
        "Must ensure state is consistent before and after suspension, "
        "avoiding data races or invalid intermediate states."
    ),
    prompt="""Update actor state across an async suspension point safely.
Set isLoading=true before await, and isLoading=false after - use defer for safety.

actor DataStore {
    var data: [String: Any] = [:]
    var isLoading = false

    func updateData() async {
        """,
    spec=ConstraintSpec(
        language="swift",
        type_bindings=[
            TypeBinding(
                name="self",
                type_expr="actor DataStore",
                scope="local",
            ),
            TypeBinding(
                name="data",
                type_expr="[String: Any]",
                scope="local",
                mutable=True,
            ),
            TypeBinding(
                name="isLoading",
                type_expr="Bool",
                scope="local",
                mutable=True,
            ),
        ],
        control_flow=ControlFlowContext(
            in_async_context=True,
            function_name="updateData",
        ),
        semantic_constraints=[
            SemanticConstraint(
                kind="invariant",
                expression="!(isLoading && data.isEmpty == false)",
                scope="updateData",
                variables=("isLoading", "data"),
            ),
            SemanticConstraint(
                kind="postcondition",
                expression="isLoading == false",
                scope="updateData",
                variables=("isLoading",),
            ),
        ],
        ebnf=r'''
root ::= basic_pattern | defer_pattern | do_catch_pattern
basic_pattern ::= "isLoading = true\nlet result = await fetch()\ndata = result\nisLoading = false"
defer_pattern ::= "isLoading = true\ndefer { isLoading = false }\nlet result = await fetch()\ndata = result"
do_catch_pattern ::= "isLoading = true\ndo { data = await fetch(); isLoading = false } catch { isLoading = false; throw }"
''',
    ),
    expected_effect=(
        "Masks tokens that would leave actor state inconsistent across suspension. "
        "Ensures isLoading flag is properly managed and data updates are atomic."
    ),
    valid_outputs=[
        "isLoading = true\nlet result = await fetch()\ndata = result\nisLoading = false",
        "isLoading = true\ndefer { isLoading = false }\nlet result = await fetch()\ndata = result",
        "isLoading = true\ndo { data = await fetch(); isLoading = false } catch { isLoading = false; throw }",
    ],
    invalid_outputs=[
        "let result = await fetch()\ndata = result",  # No isLoading management
        "isLoading = true\ndata = await fetch()",  # Never set back to false
        "data = await fetch()\nisLoading = false",  # Never set to true first
    ],
    tags=["semantics", "actors", "consistency", "async", "state"],
    language="swift",
    domain="semantics",
)

SWIFT_SEMANTIC_003 = ConstraintExample(
    id="swift-semantics-003",
    name="Sendable Conformance Requirements",
    description="Ensure types crossing actor boundaries conform to Sendable",
    scenario=(
        "Developer passing data between actors or to global async functions. "
        "Swift 6 strict concurrency requires types to be Sendable when crossing "
        "isolation boundaries. The constraint ensures only Sendable-conforming "
        "types are used in cross-actor communication."
    ),
    prompt="""Send a Message struct to an actor. The Message must be Sendable.
Only pass value types or Sendable-conforming types across actor boundaries.

actor MessageQueue {
    func send(_ message: Message) async { ... }
}
""",
    spec=ConstraintSpec(
        language="swift",
        type_bindings=[
            TypeBinding(
                name="message",
                type_expr="Message",
                scope="parameter",
            ),
        ],
        type_aliases={
            "Message": "struct Message: Sendable { let id: String; let content: String }",
            "UnsafeMessage": "class UnsafeMessage { var content: String }",
        },
        semantic_constraints=[
            SemanticConstraint(
                kind="precondition",
                expression="Message: Sendable",
                scope="sendMessage",
                variables=("Message",),
            ),
        ],
        control_flow=ControlFlowContext(
            in_async_context=True,
        ),
        ebnf=r'''
root ::= send_message | send_list | process_string
send_message ::= "await actor.send(Message(id: '1', content: 'Hello'))"
send_list ::= "await actor.send([Message(id: '1', content: 'A'), Message(id: '2', content: 'B')])"
process_string ::= "await actor.process('string literal')"
''',
    ),
    expected_effect=(
        "Masks tokens that would pass non-Sendable types across actor boundaries. "
        "Ensures only value types, immutable references, or Sendable-conforming "
        "types are used in async actor calls."
    ),
    valid_outputs=[
        "await actor.send(Message(id: '1', content: 'Hello'))",
        "await actor.send([Message(id: '1', content: 'A'), Message(id: '2', content: 'B')])",
        "await actor.process('string literal')",
    ],
    invalid_outputs=[
        "await actor.send(UnsafeMessage())",  # Class not Sendable
        "await actor.send(mutableArray)",  # Non-Sendable collection
        "var msg = Message(...); msg.content = 'new'; await actor.send(msg)",  # Mutated before send
    ],
    tags=["semantics", "sendable", "concurrency", "actors", "safety"],
    language="swift",
    domain="semantics",
)

# =============================================================================
# Exports
# =============================================================================

SWIFT_SEMANTIC_EXAMPLES: List[ConstraintExample] = [
    SWIFT_SEMANTIC_001,
    SWIFT_SEMANTIC_002,
    SWIFT_SEMANTIC_003,
]

__all__ = ["SWIFT_SEMANTIC_EXAMPLES"]
