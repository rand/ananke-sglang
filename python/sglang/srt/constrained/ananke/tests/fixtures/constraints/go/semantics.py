# Copyright Rand Arete @ Ananke 2025
# Licensed under the Apache License, Version 2.0
"""Go semantic constraint examples.

Demonstrates semantic constraints specific to Go:
- Nil check requirements before pointer/interface use
- Context propagation (context.Context first parameter convention)
- Resource cleanup with defer
"""

from __future__ import annotations

from typing import List

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


GO_SEMANTIC_EXAMPLES: List[ConstraintExample] = [
    ConstraintExample(
        id="go-semantics-001",
        name="Nil Check Before Pointer Dereference",
        description="Require nil check before dereferencing pointers or calling methods",
        scenario=(
            "Developer working with pointer types in Go. Before dereferencing "
            "or calling methods on a pointer, must verify it's not nil to avoid "
            "panic. This is a critical safety invariant in Go."
        ),
        prompt="""Write a processUser function that takes *User. Always check if user == nil
before accessing fields or calling methods. Nil dereference causes panic.

func processUser(user *User) error {
    """,
        spec=ConstraintSpec(
            language="go",
            # Regex enforces nil check before pointer dereference (either == nil or != nil patterns)
            regex=r"func\s+\w+\s*\(\s*\w+\s+\*\w+\s*\)[\s\S]*if\s+\w+\s*[!=]=\s*nil\s*\{",
            ebnf=r'''
root ::= eq_nil_check | eq_nil_sentinel | neq_nil_check
eq_nil_check ::= "func processUser(user *User) error {" nl "\tif user == nil {" nl "\t\treturn errors.New(\"user is nil\")" nl "\t}" nl "\treturn user.Validate()" nl "}"
eq_nil_sentinel ::= "func processUser(user *User) error {" nl "\tif user == nil {" nl "\t\treturn ErrNilUser" nl "\t}" nl "\tfmt.Println(user.Name)" nl "\treturn nil" nl "}"
neq_nil_check ::= "func processUser(user *User) error {" nl "\tif user != nil {" nl "\t\treturn user.Save()" nl "\t}" nl "\treturn nil" nl "}"
nl ::= "\n"
''',
            type_bindings=[
                TypeBinding(name="user", type_expr="*User", scope="parameter"),
            ],
            function_signatures=[
                FunctionSignature(
                    name="processUser",
                    params=(TypeBinding("user", "*User"),),
                    return_type="error",
                ),
            ],
            semantic_constraints=[
                SemanticConstraint(
                    kind="precondition",
                    expression="precedes(nil_check(user), access(user))",
                    scope="processUser",
                    variables=("user",),
                ),
            ],
        ),
        expected_effect=(
            "Masks code that dereferences pointers without nil checks. "
            "Requires if user == nil check or guard before access. "
            "Prevents nil pointer dereference panics."
        ),
        valid_outputs=[
            'func processUser(user *User) error {\n\tif user == nil {\n\t\treturn errors.New("user is nil")\n\t}\n\treturn user.Validate()\n}',
            'func processUser(user *User) error {\n\tif user == nil {\n\t\treturn ErrNilUser\n\t}\n\tfmt.Println(user.Name)\n\treturn nil\n}',
            'func processUser(user *User) error {\n\tif user != nil {\n\t\treturn user.Save()\n\t}\n\treturn nil\n}',
        ],
        invalid_outputs=[
            'func processUser(user *User) error {\n\treturn user.Validate()\n}',  # No nil check
            'func processUser(user *User) error {\n\tfmt.Println(user.Name)\n\treturn nil\n}',  # Direct field access without check
            'func processUser(user *User) error {\n\tif user.ID > 0 {\n\t\treturn nil\n\t}\n\treturn errors.New("invalid")\n}',  # Access before nil check
        ],
        tags=["semantics", "nil-check", "pointers", "safety"],
        language="go",
        domain="semantics",
    ),
    ConstraintExample(
        id="go-semantics-002",
        name="Context Propagation - First Parameter",
        description="Enforce context.Context as first parameter convention",
        scenario=(
            "Developer writing functions that perform I/O, call services, or "
            "can be cancelled. Go convention: context.Context must be the first "
            "parameter (except for receiver). This enables cancellation and "
            "deadline propagation throughout the call chain."
        ),
        prompt="""Write a function that does I/O (fetching user data). Go convention requires
context.Context as the first parameter to support cancellation and timeouts.

""",
        spec=ConstraintSpec(
            language="go",
            # Regex enforces context.Context as first parameter
            regex=r"^func\s+(?:\([^)]+\)\s+)?\w+\s*\(\s*ctx\s+context\.Context\s*,",
            ebnf=r'''
root ::= func_with_select | method_with_ctx
func_with_select ::= "func fetchUserData(ctx context.Context, userID string) (*UserData, error) {" nl "\tselect {" nl "\tcase <-ctx.Done():" nl "\t\treturn nil, ctx.Err()" nl "\tdefault:" nl "\t}" nl "\treturn db.Get(ctx, userID)" nl "}"
method_with_ctx ::= "func (s *Service) fetchUserData(ctx context.Context, userID string) (*UserData, error) {" nl "\treturn s.client.Fetch(ctx, userID)" nl "}"
nl ::= "\n"
''',
            type_bindings=[
                TypeBinding(
                    name="ctx",
                    type_expr="context.Context",
                    scope="parameter",
                    mutable=False,
                ),
                TypeBinding(name="userID", type_expr="string", scope="parameter"),
            ],
            function_signatures=[
                FunctionSignature(
                    name="fetchUserData",
                    params=(
                        TypeBinding("ctx", "context.Context"),  # Must be first
                        TypeBinding("userID", "string"),
                    ),
                    return_type="(*UserData, error)",
                ),
            ],
            semantic_constraints=[
                SemanticConstraint(
                    kind="invariant",
                    expression="first_param_type(io_functions) == 'context.Context'",
                    scope="fetchUserData",
                    variables=("io_functions",),
                ),
                SemanticConstraint(
                    kind="precondition",
                    expression="ctx != nil",
                    scope="fetchUserData",
                    variables=("ctx",),
                ),
            ],
        ),
        expected_effect=(
            "Masks function signatures with context.Context not as first parameter. "
            "Ensures context is available for cancellation checks. "
            "Enforces Go API design guidelines."
        ),
        valid_outputs=[
            'func fetchUserData(ctx context.Context, userID string) (*UserData, error) {\n\tselect {\n\tcase <-ctx.Done():\n\t\treturn nil, ctx.Err()\n\tdefault:\n\t}\n\treturn db.Get(ctx, userID)\n}',
            'func (s *Service) fetchUserData(ctx context.Context, userID string) (*UserData, error) {\n\treturn s.client.Fetch(ctx, userID)\n}',
        ],
        invalid_outputs=[
            'func fetchUserData(userID string, ctx context.Context) (*UserData, error) { ... }',  # ctx not first
            'func fetchUserData(userID string) (*UserData, error) { ... }',  # Missing ctx
            'func fetchUserData(ctx context.Context) (*UserData, error) { ... }',  # Has ctx but signature incomplete
        ],
        tags=["semantics", "context", "conventions", "api-design"],
        language="go",
        domain="semantics",
    ),
    ConstraintExample(
        id="go-semantics-003",
        name="Resource Cleanup with Defer",
        description="Ensure defer is used for resource cleanup",
        scenario=(
            "Developer acquiring resources (files, locks, connections) must "
            "use defer to ensure cleanup happens even if function panics or "
            "returns early. Defer guarantees cleanup runs in all exit paths."
        ),
        prompt="""Write a short Go function that acquires a resource and immediately defers cleanup.
Pattern: mu.Lock() then defer mu.Unlock(), or os.Open() then defer file.Close().

func critical_section() error {
	mu.Lock()
	defer mu.Unlock()
""",
        spec=ConstraintSpec(
            language="go",
            # Regex enforces defer cleanup pattern (Lock/Unlock, Open/Close, etc.)
            regex=r"\bdefer\s+\w+\.(?:Unlock|Close)\s*\(\)",
            ebnf=r'''
root ::= mutex_pattern | sql_pattern | file_pattern
mutex_pattern ::= "func critical_section() error {" nl "\tmu.Lock()" nl "\tdefer mu.Unlock()" nl "\t// critical work" nl "\treturn nil" nl "}"
sql_pattern ::= "func queryDB() (*Result, error) {" nl "\tconn, err := sql.Open(\"postgres\", dsn)" nl "\tif err != nil {" nl "\t\treturn nil, err" nl "\t}" nl "\tdefer conn.Close()" nl "\treturn conn.Query(query)" nl "}"
file_pattern ::= "func readFile(path string) ([]byte, error) {" nl "\tfile, err := os.Open(path)" nl "\tif err != nil {" nl "\t\treturn nil, err" nl "\t}" nl "\tdefer file.Close()" nl "\treturn io.ReadAll(file)" nl "}"
nl ::= "\n"
''',
            type_bindings=[
                TypeBinding(name="mu", type_expr="*sync.Mutex", scope="local"),
                TypeBinding(name="conn", type_expr="*sql.DB", scope="local"),
                TypeBinding(name="file", type_expr="*os.File", scope="local"),
            ],
            semantic_constraints=[
                SemanticConstraint(
                    kind="invariant",
                    expression="followed_by_defer(acquire_call, release_call)",
                    scope="critical_section",
                    variables=("acquire_call", "release_call"),
                ),
                SemanticConstraint(
                    kind="postcondition",
                    expression="cleanup_in_all_exits(resource)",
                    scope="critical_section",
                    variables=("resource",),
                ),
            ],
        ),
        expected_effect=(
            "Masks resource usage patterns without defer cleanup. "
            "Requires defer after successful resource acquisition. "
            "Prevents resource leaks."
        ),
        valid_outputs=[
            'func critical_section() error {\n\tmu.Lock()\n\tdefer mu.Unlock()\n\t// critical work\n\treturn nil\n}',
            'func queryDB() (*Result, error) {\n\tconn, err := sql.Open("postgres", dsn)\n\tif err != nil {\n\t\treturn nil, err\n\t}\n\tdefer conn.Close()\n\treturn conn.Query(query)\n}',
            'func readFile(path string) ([]byte, error) {\n\tfile, err := os.Open(path)\n\tif err != nil {\n\t\treturn nil, err\n\t}\n\tdefer file.Close()\n\treturn io.ReadAll(file)\n}',
        ],
        invalid_outputs=[
            'func critical_section() error {\n\tmu.Lock()\n\t// work\n\tmu.Unlock()\n\treturn nil\n}',  # Manual unlock, not defer
            'func queryDB() (*Result, error) {\n\tconn, _ := sql.Open("postgres", dsn)\n\tres := conn.Query(query)\n\tconn.Close()\n\treturn res, nil\n}',  # Close at end, not defer
            'func readFile(path string) ([]byte, error) {\n\tfile, _ := os.Open(path)\n\treturn io.ReadAll(file)\n}',  # Missing close entirely
        ],
        tags=["semantics", "defer", "cleanup", "resources"],
        language="go",
        domain="semantics",
    ),
]


__all__ = ["GO_SEMANTIC_EXAMPLES"]
