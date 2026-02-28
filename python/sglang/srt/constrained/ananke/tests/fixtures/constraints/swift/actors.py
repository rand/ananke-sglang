# Copyright Rand Arete @ Ananke 2025
# Licensed under the Apache License, Version 2.0
"""Swift actor deep dive examples for Ananke.

This module contains in-depth examples of Swift's actor concurrency model,
demonstrating actor isolation, Sendable protocol conformance, and async/await
patterns with Task groups. This is a comprehensive exploration of Swift 6
strict concurrency.
"""

from __future__ import annotations

from typing import List

try:
    from ..base import ConstraintExample
    from ....spec.constraint_spec import (
        ConstraintSpec,
        TypeBinding,
        FunctionSignature,
        ClassDefinition,
        ControlFlowContext,
        SemanticConstraint,
    )
except ImportError:
    from tests.fixtures.constraints.base import ConstraintExample
    from spec.constraint_spec import (
        ConstraintSpec,
        TypeBinding,
        FunctionSignature,
        ClassDefinition,
        ControlFlowContext,
        SemanticConstraint,
    )

# =============================================================================
# Actor Deep Dive Examples
# =============================================================================

SWIFT_ACTOR_001 = ConstraintExample(
    id="swift-actors-001",
    name="Actor Isolation with @MainActor",
    description="Deep dive: MainActor isolation for UI state management",
    scenario=(
        "Developer building a view model that manages UI state. All properties "
        "and methods must be @MainActor isolated to ensure UI updates happen "
        "on the main thread. Async methods can be called from any context but "
        "will automatically hop to MainActor."
    ),
    prompt="""Create a @MainActor ViewModel class with @Published properties for SwiftUI.
All UI state mutations must happen on MainActor for thread safety.

""",
    spec=ConstraintSpec(
        language="swift",
        class_definitions=[
            ClassDefinition(
                name="ViewModel",
                methods=(
                    FunctionSignature(
                        name="loadData",
                        params=(),
                        return_type="Void",
                        is_async=True,
                        decorators=("@MainActor",),
                    ),
                    FunctionSignature(
                        name="updateUI",
                        params=(TypeBinding(name="data", type_expr="Data"),),
                        return_type="Void",
                        decorators=("@MainActor",),
                    ),
                ),
                instance_vars=(
                    TypeBinding(name="items", type_expr="[String]", mutable=True),
                    TypeBinding(name="isLoading", type_expr="Bool", mutable=True),
                    TypeBinding(name="error", type_expr="Error?", mutable=True),
                ),
            ),
        ],
        type_bindings=[
            TypeBinding(
                name="self",
                type_expr="@MainActor ViewModel",
                scope="local",
            ),
        ],
        control_flow=ControlFlowContext(
            in_async_context=True,
        ),
        semantic_constraints=[
            SemanticConstraint(
                kind="invariant",
                expression="on_main_actor(ui_mutations)",
                scope="ViewModel",
                variables=("items", "isLoading", "error", "ui_mutations"),
            ),
        ],
        ebnf=r'''
root ::= viewmodel1 | viewmodel2
viewmodel1 ::= "@MainActor\nclass ViewModel: ObservableObject {\n    @Published var items: [String] = []\n    @Published var isLoading = false\n\n    func loadData() async {\n        isLoading = true\n        let data = await fetchData()\n        items = data\n        isLoading = false\n    }\n}"
viewmodel2 ::= "@MainActor\nclass ViewModel: ObservableObject {\n    @Published var items: [String] = []\n\n    nonisolated func startLoad() {\n        Task { @MainActor in\n            await loadData()\n        }\n    }\n\n    func loadData() async {\n        items = await fetchData()\n    }\n}"
''',
    ),
    expected_effect=(
        "Masks tokens that would access MainActor-isolated state without proper "
        "isolation. Ensures all UI state is protected by @MainActor and async "
        "methods properly await when crossing isolation boundaries."
    ),
    valid_outputs=[
        """@MainActor
class ViewModel: ObservableObject {
    @Published var items: [String] = []
    @Published var isLoading = false

    func loadData() async {
        isLoading = true
        let data = await fetchData()
        items = data
        isLoading = false
    }
}""",
        """@MainActor
class ViewModel: ObservableObject {
    @Published var items: [String] = []

    nonisolated func startLoad() {
        Task { @MainActor in
            await loadData()
        }
    }

    func loadData() async {
        items = await fetchData()
    }
}""",
    ],
    invalid_outputs=[
        """class ViewModel: ObservableObject {
    @Published var items: [String] = []
    func loadData() async { items = await fetchData() }
}""",  # Missing @MainActor
        """@MainActor
class ViewModel: ObservableObject {
    @Published var items: [String] = []
    nonisolated func loadData() async { items = await fetchData() }
}""",  # nonisolated accessing MainActor property
    ],
    tags=["actors", "main-actor", "isolation", "ui", "swiftui", "deep-dive"],
    language="swift",
    domain="actors",
)

SWIFT_ACTOR_002 = ConstraintExample(
    id="swift-actors-002",
    name="Sendable Protocol Conformance",
    description="Deep dive: Sendable for safe cross-actor data transfer",
    scenario=(
        "Developer creating data models that will be passed between actors. "
        "Types must conform to Sendable to guarantee thread-safety. Value types "
        "are implicitly Sendable if all properties are Sendable. Classes need "
        "@unchecked Sendable or final + immutable properties."
    ),
    prompt="""Define a Message struct that conforms to Sendable for actor communication.
Use let for all properties - var would break Sendable safety.

""",
    spec=ConstraintSpec(
        language="swift",
        class_definitions=[
            ClassDefinition(
                name="Message",
                instance_vars=(
                    TypeBinding(name="id", type_expr="UUID", mutable=False),
                    TypeBinding(name="content", type_expr="String", mutable=False),
                    TypeBinding(name="timestamp", type_expr="Date", mutable=False),
                ),
            ),
        ],
        semantic_constraints=[
            SemanticConstraint(
                kind="precondition",
                expression="all_sendable(properties) && all_immutable(properties)",
                scope="Message",
                variables=("id", "content", "timestamp", "properties"),
            ),
        ],
        ebnf=r'''
root ::= struct_message | final_class_message | struct_container
struct_message ::= "struct Message: Sendable {\n    let id: UUID\n    let content: String\n    let timestamp: Date\n}"
final_class_message ::= "final class Message: @unchecked Sendable {\n    let id: UUID\n    let content: String\n    init(id: UUID, content: String) {\n        self.id = id\n        self.content = content\n    }\n}"
struct_container ::= "struct Container: Sendable {\n    let items: [String]\n    let metadata: [String: Int]\n}"
''',
    ),
    expected_effect=(
        "Masks tokens that would create non-Sendable types (mutable properties, "
        "non-final classes, closures capturing mutable state). Ensures safe "
        "cross-actor communication."
    ),
    valid_outputs=[
        """struct Message: Sendable {
    let id: UUID
    let content: String
    let timestamp: Date
}""",
        """final class Message: @unchecked Sendable {
    let id: UUID
    let content: String
    init(id: UUID, content: String) {
        self.id = id
        self.content = content
    }
}""",
        """struct Container: Sendable {
    let items: [String]
    let metadata: [String: Int]
}""",
    ],
    invalid_outputs=[
        """struct Message: Sendable {
    var content: String
}""",  # Mutable property
        """class Message: Sendable {
    let content: String
}""",  # Non-final class
        """struct Message: Sendable {
    let handler: () -> Void
}""",  # Closure may not be Sendable
    ],
    tags=["actors", "sendable", "concurrency", "thread-safety", "deep-dive"],
    language="swift",
    domain="actors",
)

SWIFT_ACTOR_003 = ConstraintExample(
    id="swift-actors-003",
    name="Task Groups with Async/Await",
    description="Deep dive: Structured concurrency with TaskGroup for parallel operations",
    scenario=(
        "Developer implementing parallel async operations using TaskGroup. "
        "Must properly add child tasks, await results, and handle errors. "
        "All child tasks must return Sendable types to safely collect results."
    ),
    prompt="""Fetch data from multiple URLs in parallel using withTaskGroup.
Add tasks with group.addTask{} and collect results with for await.

func fetchAll(urls: [URL]) async -> [Data] {
    """,
    spec=ConstraintSpec(
        language="swift",
        type_bindings=[
            TypeBinding(
                name="urls",
                type_expr="[URL]",
                scope="parameter",
            ),
        ],
        control_flow=ControlFlowContext(
            function_name="fetchAll",
            function_signature=FunctionSignature(
                name="fetchAll",
                params=(TypeBinding(name="urls", type_expr="[URL]"),),
                return_type="[Data]",
                is_async=True,
            ),
            in_async_context=True,
        ),
        semantic_constraints=[
            SemanticConstraint(
                kind="precondition",
                expression="is_sendable(result_type)",
                scope="withTaskGroup",
                variables=("result_type",),
            ),
            SemanticConstraint(
                kind="postcondition",
                expression="all_completed_or_cancelled(tasks)",
                scope="fetchAll",
                variables=("tasks",),
            ),
        ],
        ebnf=r'''
root ::= taskgroup_basic | taskgroup_throwing | taskgroup_reduce
taskgroup_basic ::= "await withTaskGroup(of: Data.self) { group in\n    for url in urls {\n        group.addTask { await fetch(url) }\n    }\n    var results: [Data] = []\n    for await result in group {\n        results.append(result)\n    }\n    return results\n}"
taskgroup_throwing ::= "await withThrowingTaskGroup(of: Data.self) { group in\n    for url in urls {\n        group.addTask { try await fetch(url) }\n    }\n    var results: [Data] = []\n    for try await result in group {\n        results.append(result)\n    }\n    return results\n}"
taskgroup_reduce ::= "await withTaskGroup(of: Data?.self) { group in\n    urls.forEach { url in\n        group.addTask { await fetch(url) }\n    }\n    return await group.reduce(into: [Data]()) { results, data in\n        if let data = data { results.append(data) }\n    }\n}"
''',
    ),
    expected_effect=(
        "Masks tokens that would create TaskGroup with non-Sendable types, "
        "forget to await results, or improperly handle task lifecycle. "
        "Ensures structured concurrency with proper cleanup."
    ),
    valid_outputs=[
        """await withTaskGroup(of: Data.self) { group in
    for url in urls {
        group.addTask { await fetch(url) }
    }
    var results: [Data] = []
    for await result in group {
        results.append(result)
    }
    return results
}""",
        """await withThrowingTaskGroup(of: Data.self) { group in
    for url in urls {
        group.addTask { try await fetch(url) }
    }
    var results: [Data] = []
    for try await result in group {
        results.append(result)
    }
    return results
}""",
        """await withTaskGroup(of: Data?.self) { group in
    urls.forEach { url in
        group.addTask { await fetch(url) }
    }
    return await group.reduce(into: [Data]()) { results, data in
        if let data = data { results.append(data) }
    }
}""",
    ],
    invalid_outputs=[
        """withTaskGroup(of: Data.self) { group in
    // Missing await
    return []
}""",
        """await withTaskGroup(of: NSMutableData.self) { group in
    // NSMutableData is not Sendable
    return []
}""",
        """await withTaskGroup(of: Data.self) { group in
    for url in urls {
        Task { await fetch(url) }  // Wrong: detached task, not grouped
    }
}""",
    ],
    tags=["actors", "task-group", "async", "await", "parallel", "deep-dive"],
    language="swift",
    domain="actors",
)

SWIFT_ACTOR_004 = ConstraintExample(
    id="swift-actors-004",
    name="Actor Reentrancy and State Races",
    description="Deep dive: Understanding actor reentrancy across suspension points",
    scenario=(
        "Developer implementing actor methods with suspension points. Must understand "
        "that actor state can change between suspension points due to reentrancy. "
        "Need to capture state before await and validate after."
    ),
    prompt="""Implement an actor method that safely handles reentrancy.
Capture state before await and validate it after - state may change during suspension.

""",
    spec=ConstraintSpec(
        language="swift",
        type_bindings=[
            TypeBinding(
                name="self",
                type_expr="actor Counter",
                scope="local",
            ),
            TypeBinding(
                name="value",
                type_expr="Int",
                scope="local",
                mutable=True,
            ),
        ],
        control_flow=ControlFlowContext(
            function_name="incrementSlowly",
            in_async_context=True,
        ),
        semantic_constraints=[
            SemanticConstraint(
                kind="invariant",
                expression="value >= 0",
                scope="Counter",
                variables=("value",),
            ),
            SemanticConstraint(
                kind="assertion",
                expression="captured_before_suspension(state) && validated_after_suspension(state)",
                scope="incrementSlowly",
                variables=("value", "state"),
            ),
        ],
        ebnf=r'''
root ::= counter_actor | bank_actor
counter_actor ::= "actor Counter {\n    private var value: Int = 0\n\n    func incrementSlowly() async {\n        let before = value\n        await Task.sleep(nanoseconds: 100_000)\n        let after = value\n        value = after + 1\n        precondition(after >= before, \"Value decreased during suspension\")\n    }\n}"
bank_actor ::= "actor BankAccount {\n    private var balance: Int = 0\n\n    func withdraw(_ amount: Int) async -> Bool {\n        let currentBalance = balance\n        await validateWithServer(amount)\n        guard balance >= amount else { return false }\n        precondition(balance <= currentBalance, \"Balance increased unexpectedly\")\n        balance -= amount\n        return true\n    }\n}"
''',
    ),
    expected_effect=(
        "Masks tokens that would ignore reentrancy concerns. Ensures developers "
        "capture state before suspension, validate after, and handle potential races."
    ),
    valid_outputs=[
        """actor Counter {
    private var value: Int = 0

    func incrementSlowly() async {
        let before = value
        await Task.sleep(nanoseconds: 100_000)
        let after = value
        value = after + 1
        precondition(after >= before, "Value decreased during suspension")
    }
}""",
        """actor BankAccount {
    private var balance: Int = 0

    func withdraw(_ amount: Int) async -> Bool {
        let currentBalance = balance
        await validateWithServer(amount)
        guard balance >= amount else { return false }
        precondition(balance <= currentBalance, "Balance increased unexpectedly")
        balance -= amount
        return true
    }
}""",
    ],
    invalid_outputs=[
        """actor Counter {
    private var value: Int = 0
    func incrementSlowly() async {
        await Task.sleep(nanoseconds: 100_000)
        value += 1  // Race: value could have changed
    }
}""",
        """actor Counter {
    private var value: Int = 0
    func incrementSlowly() async {
        value += 1
        await Task.sleep(nanoseconds: 100_000)
        // No validation after suspension
    }
}""",
    ],
    tags=["actors", "reentrancy", "suspension", "race-conditions", "deep-dive"],
    language="swift",
    domain="actors",
)

# =============================================================================
# Exports
# =============================================================================

SWIFT_ACTOR_EXAMPLES: List[ConstraintExample] = [
    SWIFT_ACTOR_001,
    SWIFT_ACTOR_002,
    SWIFT_ACTOR_003,
    SWIFT_ACTOR_004,
]

__all__ = ["SWIFT_ACTOR_EXAMPLES"]
