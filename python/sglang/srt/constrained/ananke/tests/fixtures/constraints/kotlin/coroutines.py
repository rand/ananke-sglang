# Copyright Rand Arete @ Ananke 2025
# Licensed under the Apache License, Version 2.0
"""Deep dive: Kotlin Coroutines constraint examples.

This module provides in-depth constraint examples for Kotlin coroutines,
focusing on Flow operators, suspend function composition, and CoroutineScope
management. These examples demonstrate how Ananke enforces correct coroutine
usage patterns at a detailed level.
"""

from __future__ import annotations

try:
    from ..base import ConstraintExample
    from .....spec.constraint_spec import (
        ConstraintSpec,
        TypeBinding,
        FunctionSignature,
        ImportBinding,
        ControlFlowContext,
        SemanticConstraint,
    )
except ImportError:
    from tests.fixtures.constraints.base import ConstraintExample
    from spec.constraint_spec import (
        ConstraintSpec,
        TypeBinding,
        FunctionSignature,
        ImportBinding,
        ControlFlowContext,
        SemanticConstraint,
    )

KOTLIN_COROUTINES_EXAMPLES = [
    ConstraintExample(
        id="kt-coro-001",
        name="Flow Operators and Cold Streams",
        description="Enforce Flow operator chains with cold stream semantics",
        scenario=(
            "Developer building a Flow pipeline using operators like map, filter, "
            "collect. Flow is a cold stream - it doesn't emit values until collected. "
            "Common mistakes include not calling collect(), mixing Flow with sequences, "
            "or trying to get values synchronously. The constraint enforces proper "
            "Flow usage including terminal operators and suspend context."
        ),
        prompt="""Process a Flow<Int> by mapping, filtering, then collecting the results.
A Flow needs a terminal operator like collect(), toList(), or first() - without it, nothing happens.

val numbers: Flow<Int> = getNumbers()
""",
        spec=ConstraintSpec(
            language="kotlin",
            # Simplified: Just require any Flow terminal operator
            regex=r"\.(?:collect|toList|toSet|first|single|fold|reduce)\s*[\(\{]",
            ebnf=r'''
root ::= map_filter_collect | map_tolist | filter_first | fold_sum
map_filter_collect ::= "numbers" nl "    .map { it * 2 }" nl "    .filter { it > 10 }" nl "    .collect { println(it) }"
map_tolist ::= "numbers" nl "    .map { it.toString() }" nl "    .toList()"
filter_first ::= "numbers" nl "    .filter { it % 2 == 0 }" nl "    .first()"
fold_sum ::= "val sum = numbers" nl "    .map { it * 2 }" nl "    .fold(0) { acc, value -> acc + value }"
nl ::= "\n"
''',
            expected_type="Unit",
            type_bindings=[
                TypeBinding(name="numbers", type_expr="Flow<Int>", scope="local"),
            ],
            imports=[
                ImportBinding(module="kotlinx.coroutines.flow", name="Flow"),
                ImportBinding(module="kotlinx.coroutines.flow", name="map"),
                ImportBinding(module="kotlinx.coroutines.flow", name="filter"),
                ImportBinding(module="kotlinx.coroutines.flow", name="collect"),
            ],
            function_signatures=[
                FunctionSignature(
                    name="processNumbers",
                    params=(TypeBinding(name="numbers", type_expr="Flow<Int>"),),
                    return_type="Unit",
                    is_async=True,  # suspend function
                ),
            ],
            control_flow=ControlFlowContext(
                in_async_context=True,
                function_name="processNumbers",
            ),
            semantic_constraints=[
                SemanticConstraint(
                    kind="invariant",
                    expression="has_terminal_operator(numbers)",
                    scope="processNumbers",
                    variables=("numbers",),
                ),
            ],
        ),
        expected_effect=(
            "Masks tokens that create incomplete Flow chains without terminal operators. "
            "Blocks 'numbers.map { it * 2 }' without collect(). Requires terminal "
            "operators: collect(), toList(), toSet(), first(), single(). Enforces "
            "suspend context for Flow collection."
        ),
        valid_outputs=[
            """numbers
    .map { it * 2 }
    .filter { it > 10 }
    .collect { println(it) }""",
            """numbers
    .map { it.toString() }
    .toList()""",
            """numbers
    .filter { it % 2 == 0 }
    .first()""",
            """val sum = numbers
    .map { it * 2 }
    .fold(0) { acc, value -> acc + value }""",
        ],
        invalid_outputs=[
            "numbers.map { it * 2 }",  # No terminal operator
            "val flow = numbers.filter { it > 0 }",  # No collection
            "numbers.collect { println(it) }; numbers.collect { }",  # Double collect
        ],
        tags=["coroutines", "flow", "cold-stream", "operators", "kotlin"],
        language="kotlin",
        domain="coroutines",
    ),
    ConstraintExample(
        id="kt-coro-002",
        name="Suspend Function Composition",
        description="Enforce proper suspend function composition with coroutineScope",
        scenario=(
            "Developer composing multiple suspend function calls, possibly in parallel. "
            "Kotlin provides coroutineScope { } for structured concurrency and parallel "
            "execution with async/await. Common patterns: sequential awaits, parallel "
            "async then awaitAll, or mixed. Constraints ensure proper scoping and "
            "exception handling."
        ),
        prompt="""Fetch user data and posts in parallel using coroutineScope with async/await.
Use coroutineScope { } for structured concurrency - don't forget to await() each Deferred.

suspend fun loadUserData(userId: String): Result {
    """,
        spec=ConstraintSpec(
            language="kotlin",
            # Regex enforces coroutineScope { } with async/await for parallel execution
            regex=r"^(?:coroutineScope\s*\{[\s\S]*async\s*\{|val\s+\w+\s*=\s*fetch)",
            ebnf=r'''
root ::= sequential | parallel_await | parallel_awaitall
sequential ::= "val user = fetchUser(userId)" nl "val posts = fetchPosts(userId)" nl "val comments = fetchComments(userId)" nl "Result(user, posts, comments)"
parallel_await ::= "coroutineScope {" nl "    val userDeferred = async { fetchUser(userId) }" nl "    val postsDeferred = async { fetchPosts(userId) }" nl "    val user = userDeferred.await()" nl "    val posts = postsDeferred.await()" nl "    Result(user, posts, emptyList())" nl "}"
parallel_awaitall ::= "coroutineScope {" nl "    val (user, posts) = listOf(" nl "        async { fetchUser(userId) }," nl "        async { fetchPosts(userId) }" nl "    ).awaitAll()" nl "    Result(user, posts, emptyList())" nl "}"
nl ::= "\n"
''',
            expected_type="Result",
            type_bindings=[
                TypeBinding(name="userId", type_expr="String", scope="parameter"),
            ],
            imports=[
                ImportBinding(module="kotlinx.coroutines", name="coroutineScope"),
                ImportBinding(module="kotlinx.coroutines", name="async"),
                ImportBinding(module="kotlinx.coroutines", name="awaitAll"),
            ],
            function_signatures=[
                FunctionSignature(
                    name="fetchUser",
                    params=(TypeBinding(name="id", type_expr="String"),),
                    return_type="User",
                    is_async=True,
                ),
                FunctionSignature(
                    name="fetchPosts",
                    params=(TypeBinding(name="userId", type_expr="String"),),
                    return_type="List<Post>",
                    is_async=True,
                ),
                FunctionSignature(
                    name="fetchComments",
                    params=(TypeBinding(name="userId", type_expr="String"),),
                    return_type="List<Comment>",
                    is_async=True,
                ),
                FunctionSignature(
                    name="loadUserData",
                    params=(TypeBinding(name="userId", type_expr="String"),),
                    return_type="Result",
                    is_async=True,
                ),
            ],
            control_flow=ControlFlowContext(
                in_async_context=True,
                function_name="loadUserData",
            ),
            semantic_constraints=[
                SemanticConstraint(
                    kind="invariant",
                    expression="all_awaited(async_calls)",
                    scope="loadUserData",
                    variables=("async_calls",),
                ),
            ],
        ),
        expected_effect=(
            "Masks tokens that launch async operations without proper scoping. "
            "Requires coroutineScope { } for parallel operations. Blocks raw async { } "
            "without await, enforces structured concurrency. Prevents forgotten await() "
            "calls on Deferred values."
        ),
        valid_outputs=[
            # Sequential
            """val user = fetchUser(userId)
val posts = fetchPosts(userId)
val comments = fetchComments(userId)
Result(user, posts, comments)""",
            # Parallel with coroutineScope
            """coroutineScope {
    val userDeferred = async { fetchUser(userId) }
    val postsDeferred = async { fetchPosts(userId) }
    val user = userDeferred.await()
    val posts = postsDeferred.await()
    Result(user, posts, emptyList())
}""",
            # Parallel with awaitAll
            """coroutineScope {
    val (user, posts) = listOf(
        async { fetchUser(userId) },
        async { fetchPosts(userId) }
    ).awaitAll()
    Result(user, posts, emptyList())
}""",
        ],
        invalid_outputs=[
            """val deferred = async { fetchUser(userId) }
fetchPosts(userId)""",  # async without await
            """async { fetchUser(userId) }
async { fetchPosts(userId) }""",  # Multiple async without scoping
            """GlobalScope.async { fetchUser(userId) }""",  # GlobalScope is discouraged
        ],
        tags=["coroutines", "suspend", "composition", "async-await", "kotlin"],
        language="kotlin",
        domain="coroutines",
    ),
    ConstraintExample(
        id="kt-coro-003",
        name="CoroutineScope Management and Lifecycle",
        description="Enforce proper CoroutineScope lifecycle and cancellation",
        scenario=(
            "Developer managing a CoroutineScope tied to a lifecycle (Android ViewModel, "
            "Service class). Proper scope management means: creating scope with "
            "SupervisorJob for independent child failures, using lifecycleScope or "
            "viewModelScope when available, always canceling scopes in cleanup, "
            "avoiding GlobalScope. Constraints enforce these best practices."
        ),
        prompt="""Create a CoroutineScope with proper Job management for a custom class.
Use SupervisorJob() or Job() - don't use GlobalScope. Include a way to cancel.

""",
        spec=ConstraintSpec(
            language="kotlin",
            # Regex enforces proper CoroutineScope creation with Job (including private val)
            regex=r"(?:CoroutineScope\s*\(\s*(?:SupervisorJob|Job)|viewModelScope\.|lifecycleScope\.)",
            ebnf=r'''
root ::= supervisor_scope | job_scope | viewmodel_scope | lifecycle_scope | scope_with_cancel
supervisor_scope ::= "CoroutineScope(SupervisorJob() + Dispatchers.Main)"
job_scope ::= "CoroutineScope(Job() + Dispatchers.IO)"
viewmodel_scope ::= "viewModelScope.launch { }"
lifecycle_scope ::= "lifecycleScope.launch { }"
scope_with_cancel ::= "private val scope = CoroutineScope(SupervisorJob())" nl nl "fun onCleanup() {" nl "    scope.cancel()" nl "}"
nl ::= "\n"
''',
            expected_type="CoroutineScope",
            imports=[
                ImportBinding(module="kotlinx.coroutines", name="CoroutineScope"),
                ImportBinding(module="kotlinx.coroutines", name="SupervisorJob"),
                ImportBinding(module="kotlinx.coroutines", name="Dispatchers"),
                ImportBinding(module="kotlinx.coroutines", name="cancel"),
            ],
            class_definitions=[],
            semantic_constraints=[
                SemanticConstraint(
                    kind="invariant",
                    expression="has_job(scope)",
                    scope="createScope",
                    variables=("scope",),
                ),
                SemanticConstraint(
                    kind="postcondition",
                    expression="is_cancellable(scope)",
                    scope="createScope",
                    variables=("scope",),
                ),
            ],
        ),
        expected_effect=(
            "Masks tokens that create scopes without proper Job management. Requires "
            "SupervisorJob() or Job() in scope creation. Blocks GlobalScope usage. "
            "In cleanup methods, requires scope.cancel() calls. Enforces Dispatcher "
            "selection (Main, IO, Default)."
        ),
        valid_outputs=[
            # Proper scope creation
            "CoroutineScope(SupervisorJob() + Dispatchers.Main)",
            "CoroutineScope(Job() + Dispatchers.IO)",
            # Using lifecycle scopes (Android)
            "viewModelScope.launch { }",
            "lifecycleScope.launch { }",
            # Cancellation
            """private val scope = CoroutineScope(SupervisorJob())

fun onCleanup() {
    scope.cancel()
}""",
        ],
        invalid_outputs=[
            "GlobalScope.launch { }",  # GlobalScope discouraged
            "CoroutineScope(Dispatchers.Main)",  # No Job
            """val scope = CoroutineScope(SupervisorJob())
// ... no cancel() call in cleanup""",
        ],
        tags=["coroutines", "scope", "lifecycle", "cancellation", "kotlin"],
        language="kotlin",
        domain="coroutines",
    ),
    ConstraintExample(
        id="kt-coro-004",
        name="Flow Exception Handling with catch",
        description="Enforce proper exception handling in Flow pipelines",
        scenario=(
            "Developer building a Flow pipeline that can throw exceptions during "
            "collection or transformation. Flow provides catch { } operator for "
            "exception handling. Unlike try-catch, catch { } only handles upstream "
            "exceptions. Constraints enforce proper exception handling patterns and "
            "prevent swallowed exceptions."
        ),
        prompt="""Add error handling to a Flow pipeline using the catch{} operator.
The catch block should either emit a fallback value or rethrow - don't silently swallow errors.

val source: Flow<Data> = ...
source
    .map { processData(it) }
    """,
        spec=ConstraintSpec(
            language="kotlin",
            # Simplified: Just require catch operator with handler action
            regex=r"\.catch\s*\{\s*\w+\s*->[\s\S]*(?:emit|throw)",
            ebnf=r'''
root ::= emit_fallback | rethrow_log | when_handler
emit_fallback ::= "source" nl "    .map { processData(it) }" nl "    .catch { e ->" nl "        emit(Result.Error(e.message))" nl "    }"
rethrow_log ::= "source" nl "    .map { processData(it) }" nl "    .catch { e ->" nl "        logger.error(\"Processing failed\", e)" nl "        throw e" nl "    }"
when_handler ::= "source" nl "    .map { processData(it) }" nl "    .catch { e ->" nl "        when (e) {" nl "            is IOException -> emit(Result.NetworkError)" nl "            else -> throw e" nl "        }" nl "    }"
nl ::= "\n"
''',
            expected_type="Flow<Result>",
            type_bindings=[
                TypeBinding(name="source", type_expr="Flow<Data>", scope="parameter"),
            ],
            imports=[
                ImportBinding(module="kotlinx.coroutines.flow", name="Flow"),
                ImportBinding(module="kotlinx.coroutines.flow", name="catch"),
                ImportBinding(module="kotlinx.coroutines.flow", name="map"),
            ],
            function_signatures=[
                FunctionSignature(
                    name="processData",
                    params=(TypeBinding(name="data", type_expr="Data"),),
                    return_type="Result",
                ),
            ],
        ),
        expected_effect=(
            "Masks unsafe Flow exception handling. Requires catch { } operator for "
            "flows that can throw. Blocks try-catch around collect() without rethrowing. "
            "Enforces that catch { } emits fallback values or rethrows to prevent "
            "silent failures."
        ),
        valid_outputs=[
            """source
    .map { processData(it) }
    .catch { e ->
        emit(Result.Error(e.message))
    }""",
            """source
    .map { processData(it) }
    .catch { e ->
        logger.error("Processing failed", e)
        throw e
    }""",
            """source
    .map { processData(it) }
    .catch { e ->
        when (e) {
            is IOException -> emit(Result.NetworkError)
            else -> throw e
        }
    }""",
        ],
        invalid_outputs=[
            """source
    .map { processData(it) }
    .catch { }""",  # Silent catch, no emit or rethrow
            """try {
    source.collect { processData(it) }
} catch (e: Exception) { }""",  # try-catch instead of catch operator
        ],
        tags=["coroutines", "flow", "exceptions", "error-handling", "kotlin"],
        language="kotlin",
        domain="coroutines",
    ),
    ConstraintExample(
        id="kt-coro-005",
        name="StateFlow and SharedFlow Hot Streams",
        description="Enforce proper hot stream creation and subscription",
        scenario=(
            "Developer using StateFlow or SharedFlow for hot streams (e.g., UI state, "
            "events). Unlike Flow, these are hot - they emit even without collectors. "
            "StateFlow requires initial value, SharedFlow needs replay/buffer config. "
            "Common mistakes: not setting replay, forgetting initial value, not using "
            "stateIn/shareIn for cold-to-hot conversion."
        ),
        prompt="""Create a MutableStateFlow to hold UI state:

MutableStateFlow""",
        spec=ConstraintSpec(
            language="kotlin",
            # Regex enforces StateFlow with type parameter and initial value
            regex=r"^[<\(]\s*(?:\w+\s*[>\)]|UiState)",
            ebnf=r'''
root ::= typed_stateflow | inferred_stateflow | state_in | shared_flow
typed_stateflow ::= "MutableStateFlow<UiState>(UiState.Loading)"
inferred_stateflow ::= "MutableStateFlow(UiState.Idle)"
state_in ::= "coldFlow" nl "    .stateIn(" nl "        scope = scope," nl "        started = SharingStarted.WhileSubscribed(5000)," nl "        initialValue = UiState.Loading" nl "    )"
shared_flow ::= "MutableSharedFlow<Event>(" nl "    replay = 1," nl "    extraBufferCapacity = 64" nl ")"
nl ::= "\n"
''',
            expected_type="StateFlow<UiState>",
            imports=[
                ImportBinding(module="kotlinx.coroutines.flow", name="MutableStateFlow"),
                ImportBinding(module="kotlinx.coroutines.flow", name="StateFlow"),
                ImportBinding(module="kotlinx.coroutines.flow", name="stateIn"),
            ],
            type_bindings=[
                TypeBinding(name="scope", type_expr="CoroutineScope", scope="local"),
            ],
        ),
        expected_effect=(
            "Masks tokens that create StateFlow without initial value or SharedFlow "
            "without configuration. Requires MutableStateFlow(initialValue). For "
            "cold-to-hot conversion, requires stateIn with SharingStarted policy. "
            "Blocks missing scope or improper sharing configuration."
        ),
        valid_outputs=[
            "<UiState>(UiState.Loading)",
            "(UiState.Idle)",
            "<UiState>(UiState.Success(data))",
            "(initialState)",
        ],
        invalid_outputs=[
            "MutableStateFlow<UiState>()",  # No initial value
            "coldFlow.stateIn(scope)",  # Missing started and initialValue
            "MutableSharedFlow<Event>()",  # No replay configuration
        ],
        tags=["coroutines", "stateflow", "sharedflow", "hot-streams", "kotlin"],
        language="kotlin",
        domain="coroutines",
    ),
]

# Export combined list
ALL_KOTLIN_COROUTINES_EXAMPLES = KOTLIN_COROUTINES_EXAMPLES
