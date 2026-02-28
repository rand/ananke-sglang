# Copyright Rand Arete @ Ananke 2025
# Licensed under the Apache License, Version 2.0
"""Swift control flow constraint examples for Ananke.

This module contains realistic examples of control flow constraints in Swift,
demonstrating guard-let early exit, actor async contexts, and result builders.
"""

from __future__ import annotations

from typing import List

try:
    from ..base import ConstraintExample
    from ....spec.constraint_spec import (
        ConstraintSpec,
        TypeBinding,
        ControlFlowContext,
        FunctionSignature,
        SemanticConstraint,
    )
except ImportError:
    from tests.fixtures.constraints.base import ConstraintExample
    from spec.constraint_spec import (
        ConstraintSpec,
        TypeBinding,
        ControlFlowContext,
        FunctionSignature,
        SemanticConstraint,
    )

# =============================================================================
# Control Flow Constraint Examples
# =============================================================================

SWIFT_CONTROLFLOW_001 = ConstraintExample(
    id="swift-controlflow-001",
    name="Guard Let Early Exit",
    description="Use guard let to unwrap optionals with early return",
    scenario=(
        "Developer unwrapping multiple optional values in a function. "
        "Must use guard let statements that exit early if any value is nil, "
        "making the unwrapped values available in the happy path scope."
    ),
    prompt="""Unwrap userId and token optionals using guard let with early return.
After guard, the unwrapped values should be available in the function scope.

func authenticate(userId: String?, token: String?) -> User? {
    """,
    spec=ConstraintSpec(
        language="swift",
        type_bindings=[
            TypeBinding(
                name="userId",
                type_expr="String?",
                scope="parameter",
            ),
            TypeBinding(
                name="token",
                type_expr="String?",
                scope="parameter",
            ),
        ],
        control_flow=ControlFlowContext(
            function_name="authenticate",
            function_signature=FunctionSignature(
                name="authenticate",
                params=(
                    TypeBinding(name="userId", type_expr="String?"),
                    TypeBinding(name="token", type_expr="String?"),
                ),
                return_type="User?",
            ),
            expected_return_type="User?",
        ),
        semantic_constraints=[
            SemanticConstraint(
                kind="postcondition",
                expression="userId != nil && token != nil",
                scope="authenticate_body",
                variables=("userId", "token"),
            ),
        ],
        ebnf=r'''
root ::= guard_combined | guard_separate | guard_auth
guard_combined ::= "guard let userId = userId, let token = token else { return nil }\nreturn User(id: userId, token: token)"
guard_separate ::= "guard let id = userId else { return nil }\nguard let tok = token else { return nil }\nreturn User(id: id, token: tok)"
guard_auth ::= "guard let userId = userId, let token = token else { return nil }\nreturn authenticateUser(userId, token)"
''',
    ),
    expected_effect=(
        "Masks tokens that would access optionals without unwrapping or fail to "
        "provide early exit. Ensures guard let pattern with proper scope handling."
    ),
    valid_outputs=[
        "guard let userId = userId, let token = token else { return nil }\nreturn User(id: userId, token: token)",
        "guard let id = userId else { return nil }\nguard let tok = token else { return nil }\nreturn User(id: id, token: tok)",
        "guard let userId = userId, let token = token else { return nil }\nreturn authenticateUser(userId, token)",
    ],
    invalid_outputs=[
        "if let userId = userId { return User(id: userId, token: token!) }",  # Force unwrap
        "return User(id: userId!, token: token!)",  # No guard, unsafe
        "guard userId != nil else { return nil }\nreturn User(id: userId, token: token)",  # Still optional
    ],
    tags=["controlflow", "guard", "optionals", "early-exit"],
    language="swift",
    domain="controlflow",
)

SWIFT_CONTROLFLOW_002 = ConstraintExample(
    id="swift-controlflow-002",
    name="Actor Async Context and MainActor",
    description="Constrain async/await calls within actor isolation boundaries",
    scenario=(
        "Developer writing async code that interacts with MainActor-isolated state. "
        "Must use await when calling MainActor methods from async context, and "
        "can elide await when already on MainActor."
    ),
    prompt="""Fetch data asynchronously then update a @MainActor ViewModel.
Use await for both the fetch and the MainActor viewModel.update() call.

func loadData() async {
    """,
    spec=ConstraintSpec(
        language="swift",
        type_bindings=[
            TypeBinding(
                name="self",
                type_expr="DataManager",
                scope="local",
            ),
            TypeBinding(
                name="viewModel",
                type_expr="@MainActor ViewModel",
                scope="local",
            ),
        ],
        control_flow=ControlFlowContext(
            function_name="loadData",
            function_signature=FunctionSignature(
                name="loadData",
                params=(),
                return_type="Void",
                is_async=True,
            ),
            in_async_context=True,
        ),
        semantic_constraints=[
            SemanticConstraint(
                kind="precondition",
                expression="await for @MainActor access from non-MainActor context",
                scope="loadData",
                variables=("viewModel",),
            ),
        ],
        ebnf=r'''
root ::= fetch_update | fetch_mainactor | async_block
fetch_update ::= "let data = await fetchData()\nawait viewModel.update(data)"
fetch_mainactor ::= "let data = await fetchData()\nawait MainActor.run { viewModel.update(data) }"
async_block ::= "async { let data = await fetchData(); await viewModel.update(data) }"
''',
    ),
    expected_effect=(
        "Masks tokens that would access MainActor-isolated state without await. "
        "Enforces proper async/await usage across actor boundaries."
    ),
    valid_outputs=[
        "let data = await fetchData()\nawait viewModel.update(data)",
        "let data = await fetchData()\nawait MainActor.run { viewModel.update(data) }",
        "async { let data = await fetchData(); await viewModel.update(data) }",
    ],
    invalid_outputs=[
        "let data = await fetchData()\nviewModel.update(data)",  # Missing await for MainActor
        "let data = fetchData()\nawait viewModel.update(data)",  # Missing await for async
        "viewModel.update(await fetchData())",  # await in wrong position
    ],
    tags=["controlflow", "async", "await", "actors", "main-actor"],
    language="swift",
    domain="controlflow",
)

SWIFT_CONTROLFLOW_003 = ConstraintExample(
    id="swift-controlflow-003",
    name="Result Builder Blocks",
    description="Constrain code within @resultBuilder blocks like @ViewBuilder",
    scenario=(
        "Developer building SwiftUI views using @ViewBuilder result builder. "
        "Within the builder block, only view-returning expressions are allowed, "
        "and control flow must use special builder constructs."
    ),
    prompt="""Write a SwiftUI view body using @ViewBuilder syntax.
Use if/else for conditionals - guard and for loops aren't allowed in result builders.

@ViewBuilder var body: some View {
    """,
    spec=ConstraintSpec(
        language="swift",
        type_bindings=[
            TypeBinding(
                name="self",
                type_expr="ContentView",
                scope="local",
            ),
            TypeBinding(
                name="isLoading",
                type_expr="Bool",
                scope="local",
            ),
        ],
        control_flow=ControlFlowContext(
            function_name="body",
            function_signature=FunctionSignature(
                name="body",
                params=(),
                return_type="some View",
                decorators=("@ViewBuilder",),
            ),
            expected_return_type="some View",
        ),
        type_aliases={
            "View": "protocol { var body: some View }",
        },
        ebnf=r'''
root ::= if_loading | text_chain | vstack_views
if_loading ::= "if isLoading { ProgressView() } else { Text('Loaded') }"
text_chain ::= "Text('Title').font(.headline).padding()"
vstack_views ::= "VStack { Text('Hello'); Image(systemName: 'star') }"
''',
    ),
    expected_effect=(
        "Masks tokens that would use standard Swift control flow (guard, for) "
        "instead of result builder equivalents. Ensures all expressions return View types."
    ),
    valid_outputs=[
        "if isLoading { ProgressView() } else { Text('Loaded') }",
        "Text('Title').font(.headline).padding()",
        "VStack { Text('Hello'); Image(systemName: 'star') }",
    ],
    invalid_outputs=[
        "guard isLoading else { return Text('Error') }",  # guard not allowed in result builder
        "for item in items { Text(item) }",  # Use ForEach instead
        "let title = 'Hello'; Text(title)",  # Statements not allowed
        "print('debug'); Text('Hello')",  # Non-view expression
    ],
    tags=["controlflow", "result-builder", "swiftui", "dsl"],
    language="swift",
    domain="controlflow",
)

# =============================================================================
# Exports
# =============================================================================

SWIFT_CONTROLFLOW_EXAMPLES: List[ConstraintExample] = [
    SWIFT_CONTROLFLOW_001,
    SWIFT_CONTROLFLOW_002,
    SWIFT_CONTROLFLOW_003,
]

__all__ = ["SWIFT_CONTROLFLOW_EXAMPLES"]
