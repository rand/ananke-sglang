# Copyright Rand Arete @ Ananke 2025
# Licensed under the Apache License, Version 2.0
"""Swift type constraint examples for Ananke.

This module contains realistic examples of type-level constraints in Swift,
demonstrating optional chaining, protocols with associated types, and actor isolation.
"""

from __future__ import annotations

from typing import List

try:
    from ..base import ConstraintExample
    from ....spec.constraint_spec import (
        ConstraintSpec,
        TypeBinding,
        ControlFlowContext,
        SemanticConstraint,
    )
except ImportError:
    from tests.fixtures.constraints.base import ConstraintExample
    from spec.constraint_spec import (
        ConstraintSpec,
        TypeBinding,
        ControlFlowContext,
        SemanticConstraint,
    )

# =============================================================================
# Type Constraint Examples
# =============================================================================

SWIFT_TYPE_001 = ConstraintExample(
    id="swift-types-001",
    name="Optional Chaining",
    description="Safe unwrapping with optional chaining for nested optionals",
    scenario=(
        "Developer accessing nested properties through optional values. "
        "Must use optional chaining (?.) instead of force unwrapping (!) "
        "to safely handle nil values at any level of the chain."
    ),
    prompt="""Access the bio property through nested optionals safely in Swift.
Use optional chaining (?.) since user and profile might be nil.

let user: User? = getUser()
""",
    spec=ConstraintSpec(
        language="swift",
        expected_type="String?",
        type_bindings=[
            TypeBinding(
                name="user",
                type_expr="User?",
                scope="local",
                mutable=False,
            ),
        ],
        type_aliases={
            "User": "struct { profile: Profile?, name: String }",
            "Profile": "struct { bio: String?, avatar: URL? }",
        },
        ebnf=r'''
root ::= "user?." property_chain
property_chain ::= "profile?.bio" | "profile?.avatar?.absoluteString" | "name"
''',
    ),
    expected_effect=(
        "Masks tokens that would force unwrap (!) or access properties without "
        "optional chaining. Ensures safe navigation through optional hierarchy."
    ),
    valid_outputs=[
        "user?.profile?.bio",
        "user?.profile?.avatar?.absoluteString",
        "user?.name",
    ],
    invalid_outputs=[
        "user!.profile!.bio",  # Force unwrap - unsafe
        "user.profile.bio",  # Missing optional handling
        "(user?.profile)!.bio",  # Partially force unwrapped
    ],
    tags=["types", "optionals", "chaining", "safety"],
    language="swift",
    domain="types",
)

SWIFT_TYPE_002 = ConstraintExample(
    id="swift-types-002",
    name="Protocol with Associated Types",
    description="Generic protocol conformance using associatedtype",
    scenario=(
        "Developer implementing a Container protocol that requires an associated type Item. "
        "The implementation must specify the concrete type for Item and provide methods "
        "that work with that type parameter."
    ),
    prompt="""Implement the Container protocol for a generic Stack<Element> struct.
Define the associated type Item to be Element using typealias.

struct Stack<Element>: Container {
    var items: [Element] = []

    """,
    spec=ConstraintSpec(
        language="swift",
        type_bindings=[
            TypeBinding(
                name="self",
                type_expr="Stack<Element>",
                scope="local",
            ),
            TypeBinding(
                name="items",
                type_expr="[Element]",
                scope="local",
                mutable=True,
            ),
        ],
        type_aliases={
            "Container": "protocol { associatedtype Item; func count() -> Int; subscript(i: Int) -> Item }",
        },
        semantic_constraints=[
            SemanticConstraint(
                kind="invariant",
                expression="Item == Element",
                scope="Stack",
                variables=("Item", "Element"),
            ),
        ],
        ebnf=r'''
root ::= "typealias Item = Element" nl method_impl
method_impl ::= "var count: Int { items.count }" | "subscript(i: Int) -> Element { items[i] }" | func_append
func_append ::= "func append(_ item: Element) { items.append(item) }"
nl ::= "\n"
''',
    ),
    expected_effect=(
        "Masks tokens that would use incompatible types for the associated type. "
        "Ensures Item matches Element and all protocol requirements are satisfied."
    ),
    valid_outputs=[
        "typealias Item = Element\nvar count: Int { items.count }",
        "typealias Item = Element\nsubscript(i: Int) -> Element { items[i] }",
        "typealias Item = Element\nfunc append(_ item: Element) { items.append(item) }",
    ],
    invalid_outputs=[
        "typealias Item = String",  # Incompatible with generic Element
        "typealias Item = Any",  # Too broad, violates type safety
        "var count: String { '\\(items.count)' }",  # Wrong return type
    ],
    tags=["types", "protocols", "generics", "associated-types"],
    language="swift",
    domain="types",
)

SWIFT_TYPE_003 = ConstraintExample(
    id="swift-types-003",
    name="Actor Isolation and MainActor",
    description="Constraint generation for actor-isolated state and MainActor",
    scenario=(
        "Developer working with actor-isolated state that must be accessed on the MainActor. "
        "Properties marked @MainActor can only be accessed from the main thread, "
        "requiring await in async contexts or direct access in @MainActor functions."
    ),
    prompt="""Write a function to update a @MainActor ViewModel's title property.
The function must be marked @MainActor to access UI-bound state safely.

@MainActor class ViewModel { var title: String = "" }
""",
    spec=ConstraintSpec(
        language="swift",
        type_bindings=[
            TypeBinding(
                name="self",
                type_expr="@MainActor ViewModel",
                scope="local",
            ),
            TypeBinding(
                name="title",
                type_expr="String",
                scope="local",
                mutable=True,
            ),
        ],
        semantic_constraints=[
            SemanticConstraint(
                kind="precondition",
                expression="@MainActor.run or await MainActor",
                scope="updateTitle",
                variables=("title",),
            ),
        ],
        control_flow=ControlFlowContext(
            in_async_context=True,
        ),
        ebnf=r'''
root ::= main_self | main_direct | nonisolated
main_self ::= "@MainActor func updateTitle(_ newTitle: String) { self.title = newTitle }"
main_direct ::= "@MainActor func updateTitle(_ newTitle: String) { title = newTitle }"
nonisolated ::= "nonisolated func scheduleUpdate() { Task { await updateTitle('New') } }"
''',
    ),
    expected_effect=(
        "Masks tokens that would access MainActor-isolated state without proper isolation. "
        "Enforces @MainActor annotation on functions that update UI-bound state."
    ),
    valid_outputs=[
        "@MainActor func updateTitle(_ newTitle: String) { self.title = newTitle }",
        "@MainActor func updateTitle(_ newTitle: String) { title = newTitle }",
        "nonisolated func scheduleUpdate() { Task { await updateTitle('New') } }",
    ],
    invalid_outputs=[
        "func updateTitle(_ newTitle: String) { self.title = newTitle }",  # Missing @MainActor
        "nonisolated func updateTitle(_ newTitle: String) { title = newTitle }",  # Cannot access from nonisolated
        "func updateTitle(_ newTitle: String) async { title = newTitle }",  # Needs explicit await
    ],
    tags=["types", "actors", "concurrency", "main-actor", "isolation"],
    language="swift",
    domain="types",
)

# =============================================================================
# Exports
# =============================================================================

SWIFT_TYPE_EXAMPLES: List[ConstraintExample] = [
    SWIFT_TYPE_001,
    SWIFT_TYPE_002,
    SWIFT_TYPE_003,
]

__all__ = ["SWIFT_TYPE_EXAMPLES"]
