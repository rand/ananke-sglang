# Copyright Rand Arete @ Ananke 2025
# Licensed under the Apache License, Version 2.0
"""Go control flow constraint examples.

Demonstrates control flow constraints specific to Go:
- Defer ordering and cleanup patterns
- Goroutine context with select statements
- Error return patterns and nil checks
"""

from __future__ import annotations

from typing import List

try:
    from ..base import ConstraintExample
    from .....spec.constraint_spec import (
        ConstraintSpec,
        ControlFlowContext,
        TypeBinding,
        FunctionSignature,
        SemanticConstraint,
    )
except ImportError:
    from tests.fixtures.constraints.base import ConstraintExample
    from spec.constraint_spec import (
        ConstraintSpec,
        ControlFlowContext,
        TypeBinding,
        FunctionSignature,
        SemanticConstraint,
    )


GO_CONTROLFLOW_EXAMPLES: List[ConstraintExample] = [
    ConstraintExample(
        id="go-controlflow-001",
        name="Defer Ordering and Cleanup",
        description="Proper defer statement ordering for resource cleanup",
        scenario=(
            "Developer opening a file and must ensure it's closed via defer. "
            "Defer statements execute in LIFO order. The defer must come "
            "immediately after successful resource acquisition and before "
            "error checks that might cause early return."
        ),
        prompt="""Write a processFile function that opens a file and ensures cleanup via defer.
Place defer f.Close() after successful os.Open, before any error returns.

""",
        spec=ConstraintSpec(
            language="go",
            # Regex enforces defer cleanup pattern
            regex=r"\bdefer\s+\w+\.Close\(\)",
            ebnf=r'''
root ::= simple_pattern | wrapped_error
simple_pattern ::= "func processFile(filename string) error {" nl "\tf, err := os.Open(filename)" nl "\tif err != nil {" nl "\t\treturn err" nl "\t}" nl "\tdefer f.Close()" nl "\t// process file" nl "\treturn nil" nl "}"
wrapped_error ::= "func processFile(filename string) error {" nl "\tf, err := os.Open(filename)" nl "\tif err != nil {" nl "\t\treturn fmt.Errorf(\"open: %w\", err)" nl "\t}" nl "\tdefer f.Close()" nl "\tdata, err := io.ReadAll(f)" nl "\tif err != nil {" nl "\t\treturn err" nl "\t}" nl "\treturn processData(data)" nl "}"
nl ::= "\n"
''',
            control_flow=ControlFlowContext(
                function_name="processFile",
                expected_return_type="error",
                in_try_block=False,
            ),
            type_bindings=[
                TypeBinding(name="filename", type_expr="string", scope="parameter"),
                TypeBinding(name="f", type_expr="*os.File", scope="local"),
                TypeBinding(name="err", type_expr="error", scope="local"),
            ],
            function_signatures=[
                FunctionSignature(
                    name="processFile",
                    params=(TypeBinding("filename", "string"),),
                    return_type="error",
                ),
            ],
            semantic_constraints=[
                SemanticConstraint(
                    kind="invariant",
                    expression="precedes(defer_stmt, error_return_paths)",
                    scope="processFile",
                    variables=("defer_stmt", "error_return_paths"),
                ),
            ],
        ),
        expected_effect=(
            "Masks control flow that doesn't properly defer cleanup. "
            "Ensures defer f.Close() appears after successful os.Open. "
            "Blocks patterns that skip cleanup on error paths."
        ),
        valid_outputs=[
            'func processFile(filename string) error {\n\tf, err := os.Open(filename)\n\tif err != nil {\n\t\treturn err\n\t}\n\tdefer f.Close()\n\t// process file\n\treturn nil\n}',
            'func processFile(filename string) error {\n\tf, err := os.Open(filename)\n\tif err != nil {\n\t\treturn fmt.Errorf("open: %w", err)\n\t}\n\tdefer f.Close()\n\tdata, err := io.ReadAll(f)\n\tif err != nil {\n\t\treturn err\n\t}\n\treturn processData(data)\n}',
        ],
        invalid_outputs=[
            'func processFile(filename string) error {\n\tf, _ := os.Open(filename)\n\t// missing defer\n\treturn nil\n}',  # No defer
            'func processFile(filename string) error {\n\tf, err := os.Open(filename)\n\tdefer f.Close()\n\tif err != nil { return err }\n\t...\n}',  # defer before error check
            'func processFile(filename string) error {\n\tf, err := os.Open(filename)\n\tif err == nil {\n\t\tf.Close()\n\t}\n\treturn err\n}',  # Manual close instead of defer
        ],
        tags=["controlflow", "defer", "cleanup", "resources"],
        language="go",
        domain="controlflow",
    ),
    ConstraintExample(
        id="go-controlflow-002",
        name="Goroutine Context with Select",
        description="Context-aware goroutine with select statement for cancellation",
        scenario=(
            "Developer writing a goroutine that must respect context cancellation. "
            "The select statement multiplexes on context.Done() and work channel. "
            "This is idiomatic Go for cancellable concurrent operations."
        ),
        prompt="""Write a worker function with context cancellation support using select.
Must have case <-ctx.Done() alongside the work channel case for graceful shutdown.

func worker(ctx context.Context, jobs <-chan Job, results chan<- Result) {
    """,
        spec=ConstraintSpec(
            language="go",
            # Regex enforces select statement with ctx.Done() case
            regex=r"^func\s+\w+\s*\([^)]*context\.Context[^)]*\)[\s\S]*select\s*\{[\s\S]*case\s+<-ctx\.Done\(\)",
            ebnf=r'''
root ::= done_first | done_second
done_first ::= "func worker(ctx context.Context, jobs <-chan Job, results chan<- Result) {" nl "\tfor {" nl "\t\tselect {" nl "\t\tcase <-ctx.Done():" nl "\t\t\treturn" nl "\t\tcase job := <-jobs:" nl "\t\t\tresult := process(job)" nl "\t\t\tresults <- result" nl "\t\t}" nl "\t}" nl "}"
done_second ::= "func worker(ctx context.Context, jobs <-chan Job, results chan<- Result) {" nl "\tfor {" nl "\t\tselect {" nl "\t\tcase job, ok := <-jobs:" nl "\t\t\tif !ok {" nl "\t\t\t\treturn" nl "\t\t\t}" nl "\t\t\tresults <- process(job)" nl "\t\tcase <-ctx.Done():" nl "\t\t\treturn" nl "\t\t}" nl "\t}" nl "}"
nl ::= "\n"
''',
            control_flow=ControlFlowContext(
                function_name="worker",
                in_async_context=True,  # Goroutine
            ),
            type_bindings=[
                TypeBinding(name="ctx", type_expr="context.Context", scope="parameter"),
                TypeBinding(name="jobs", type_expr="<-chan Job", scope="parameter"),
                TypeBinding(name="results", type_expr="chan<- Result", scope="parameter"),
            ],
            function_signatures=[
                FunctionSignature(
                    name="worker",
                    params=(
                        TypeBinding("ctx", "context.Context"),
                        TypeBinding("jobs", "<-chan Job"),
                        TypeBinding("results", "chan<- Result"),
                    ),
                    return_type="void",
                ),
            ],
            semantic_constraints=[
                SemanticConstraint(
                    kind="invariant",
                    expression="contains_case(select_stmt, '<-ctx.Done()')",
                    scope="worker",
                    variables=("select_stmt",),
                ),
            ],
        ),
        expected_effect=(
            "Masks control flow missing context cancellation handling. "
            "Ensures select has case <-ctx.Done() alongside work channel. "
            "Enforces graceful shutdown pattern."
        ),
        valid_outputs=[
            'func worker(ctx context.Context, jobs <-chan Job, results chan<- Result) {\n\tfor {\n\t\tselect {\n\t\tcase <-ctx.Done():\n\t\t\treturn\n\t\tcase job := <-jobs:\n\t\t\tresult := process(job)\n\t\t\tresults <- result\n\t\t}\n\t}\n}',
            'func worker(ctx context.Context, jobs <-chan Job, results chan<- Result) {\n\tfor {\n\t\tselect {\n\t\tcase job, ok := <-jobs:\n\t\t\tif !ok {\n\t\t\t\treturn\n\t\t\t}\n\t\t\tresults <- process(job)\n\t\tcase <-ctx.Done():\n\t\t\treturn\n\t\t}\n\t}\n}',
        ],
        invalid_outputs=[
            'func worker(ctx context.Context, jobs <-chan Job, results chan<- Result) {\n\tfor job := range jobs {\n\t\tresults <- process(job)\n\t}\n}',  # No context cancellation
            'func worker(ctx context.Context, jobs <-chan Job, results chan<- Result) {\n\tfor {\n\t\tjob := <-jobs\n\t\tresults <- process(job)\n\t}\n}',  # No select, no cancellation
            'func worker(ctx context.Context, jobs <-chan Job, results chan<- Result) {\n\tselect {\n\tcase job := <-jobs:\n\t\tresults <- process(job)\n\t}\n}',  # select without ctx.Done()
        ],
        tags=["controlflow", "context", "goroutines", "select"],
        language="go",
        domain="controlflow",
    ),
    ConstraintExample(
        id="go-controlflow-003",
        name="Error Return Pattern - nil Check",
        description="Idiomatic error handling with if err != nil pattern",
        scenario=(
            "Developer calling a function that returns (T, error) and must "
            "check error before using result. Go idiom: check err != nil "
            "immediately, handle error, then use result. Never ignore errors."
        ),
        prompt="""Write a fetchUser function returning (*User, error). Use idiomatic Go error handling:
check if err != nil immediately after db.Query and return early. Never ignore errors.

""",
        spec=ConstraintSpec(
            language="go",
            # Regex enforces error check pattern: variable, err := ... if err != nil { return nil, ...}
            regex=r"func\s+\w+\([^)]*\)\s*\([^)]*error[^)]*\)[\s\S]*if\s+err\s*!=\s*nil\s*\{",
            ebnf=r'''
root ::= simple_check | wrapped_check
simple_check ::= "func fetchUser(id int) (*User, error) {" nl "\tuser, err := db.Query(id)" nl "\tif err != nil {" nl "\t\treturn nil, err" nl "\t}" nl "\treturn user, nil" nl "}"
wrapped_check ::= "func fetchUser(id int) (*User, error) {" nl "\tuser, err := db.Query(id)" nl "\tif err != nil {" nl "\t\treturn nil, fmt.Errorf(\"fetch user %d: %w\", id, err)" nl "\t}" nl "\tif user == nil {" nl "\t\treturn nil, ErrNotFound" nl "\t}" nl "\treturn user, nil" nl "}"
nl ::= "\n"
''',
            control_flow=ControlFlowContext(
                function_name="fetchUser",
                expected_return_type="(*User, error)",
            ),
            type_bindings=[
                TypeBinding(name="id", type_expr="int", scope="parameter"),
                TypeBinding(name="user", type_expr="*User", scope="local"),
                TypeBinding(name="err", type_expr="error", scope="local"),
            ],
            function_signatures=[
                FunctionSignature(
                    name="fetchUser",
                    params=(TypeBinding("id", "int"),),
                    return_type="(*User, error)",
                ),
            ],
            semantic_constraints=[
                SemanticConstraint(
                    kind="invariant",
                    expression="if err != nil { return nil, err }",
                    scope="fetchUser",
                ),
                SemanticConstraint(
                    kind="postcondition",
                    expression="(user == nil) != (err == nil)",
                    scope="fetchUser",
                    variables=("user", "err"),
                ),
            ],
        ),
        expected_effect=(
            "Masks control flow that doesn't check errors. "
            "Ensures if err != nil pattern appears before result usage. "
            "Blocks patterns that ignore error returns."
        ),
        valid_outputs=[
            'func fetchUser(id int) (*User, error) {\n\tuser, err := db.Query(id)\n\tif err != nil {\n\t\treturn nil, err\n\t}\n\treturn user, nil\n}',
            'func fetchUser(id int) (*User, error) {\n\tuser, err := db.Query(id)\n\tif err != nil {\n\t\treturn nil, fmt.Errorf("fetch user %d: %w", id, err)\n\t}\n\tif user == nil {\n\t\treturn nil, ErrNotFound\n\t}\n\treturn user, nil\n}',
        ],
        invalid_outputs=[
            'func fetchUser(id int) (*User, error) {\n\tuser, _ := db.Query(id)\n\treturn user, nil\n}',  # Ignored error
            'func fetchUser(id int) (*User, error) {\n\tuser, err := db.Query(id)\n\treturn user, err\n}',  # No error check
            'func fetchUser(id int) (*User, error) {\n\tuser, err := db.Query(id)\n\tif user != nil {\n\t\treturn user, nil\n\t}\n\treturn nil, err\n}',  # Check user before err (wrong order)
        ],
        tags=["controlflow", "errors", "nil-check", "idioms"],
        language="go",
        domain="controlflow",
    ),
]


__all__ = ["GO_CONTROLFLOW_EXAMPLES"]
