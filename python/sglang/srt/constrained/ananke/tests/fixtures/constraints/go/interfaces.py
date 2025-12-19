# Copyright Rand Arete @ Ananke 2025
# Licensed under the Apache License, Version 2.0
"""Go interface constraint examples - Deep Dive.

Comprehensive examples demonstrating Go's interface system:
- Interface embedding and composition
- Empty interface (interface{} / any) and type assertions
- Standard library Reader/Writer interface patterns
- Method set requirements and pointer vs value receivers
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


GO_INTERFACE_EXAMPLES: List[ConstraintExample] = [
    ConstraintExample(
        id="go-interfaces-001",
        name="Interface Embedding and Composition",
        description="Compose interfaces through embedding",
        scenario=(
            "Developer creating ReadWriteCloser interface by embedding io.Reader, "
            "io.Writer, and io.Closer. Go allows interface composition through "
            "embedding, creating a new interface with combined method sets."
        ),
        prompt="""Implement io.ReadWriteCloser: a type with Read, Write, and Close methods.
All three methods must have exact signatures to satisfy the embedded interfaces.

""",
        spec=ConstraintSpec(
            language="go",
            # Regex enforces all three methods: Read, Write, Close
            regex=r"^type\s+\w+\s+struct[\s\S]*func\s+\([^)]+\)\s+Read\s*\([\s\S]*func\s+\([^)]+\)\s+Write\s*\([\s\S]*func\s+\([^)]+\)\s+Close\s*\(",
            ebnf=r'''
root ::= file_impl | conn_impl
file_impl ::= "type File struct {" nl "\tf *os.File" nl "}" nl nl "func (f *File) Read(p []byte) (int, error) {" nl "\treturn f.f.Read(p)" nl "}" nl nl "func (f *File) Write(p []byte) (int, error) {" nl "\treturn f.f.Write(p)" nl "}" nl nl "func (f *File) Close() error {" nl "\treturn f.f.Close()" nl "}"
conn_impl ::= "type BufferedConn struct {" nl "\tconn net.Conn" nl "\tbuf []byte" nl "}" nl nl "func (b *BufferedConn) Read(p []byte) (int, error) { /* ... */ }" nl "func (b *BufferedConn) Write(p []byte) (int, error) { /* ... */ }" nl "func (b *BufferedConn) Close() error { return b.conn.Close() }"
nl ::= "\n"
''',
            expected_type="io.ReadWriteCloser",
            type_aliases={
                "io.Reader": "interface { Read(p []byte) (n int, err error) }",
                "io.Writer": "interface { Write(p []byte) (n int, err error) }",
                "io.Closer": "interface { Close() error }",
                "io.ReadWriteCloser": "interface { io.Reader; io.Writer; io.Closer }",
            },
            semantic_constraints=[
                SemanticConstraint(
                    kind="invariant",
                    expression="implements_all(type, embedded_methods)",
                    scope="ReadWriteCloser",
                    variables=("type", "embedded_methods"),
                ),
            ],
        ),
        expected_effect=(
            "Masks implementations missing any of the three required methods. "
            "Ensures Read, Write, and Close methods all have correct signatures. "
            "Validates interface composition through embedding."
        ),
        valid_outputs=[
            'type File struct {\n\tf *os.File\n}\n\nfunc (f *File) Read(p []byte) (int, error) {\n\treturn f.f.Read(p)\n}\n\nfunc (f *File) Write(p []byte) (int, error) {\n\treturn f.f.Write(p)\n}\n\nfunc (f *File) Close() error {\n\treturn f.f.Close()\n}',
            'type BufferedConn struct {\n\tconn net.Conn\n\tbuf []byte\n}\n\nfunc (b *BufferedConn) Read(p []byte) (int, error) { /* ... */ }\nfunc (b *BufferedConn) Write(p []byte) (int, error) { /* ... */ }\nfunc (b *BufferedConn) Close() error { return b.conn.Close() }',
        ],
        invalid_outputs=[
            'type File struct { f *os.File }\nfunc (f *File) Read(p []byte) (int, error) { ... }\nfunc (f *File) Write(p []byte) (int, error) { ... }',  # Missing Close
            'type File struct { f *os.File }\nfunc (f File) Read(p []byte) (int, error) { ... }',  # Value receiver instead of pointer
            'type File struct { f *os.File }\nfunc (f *File) ReadData(p []byte) (int, error) { ... }',  # Wrong method name
        ],
        tags=["interfaces", "embedding", "composition", "io"],
        language="go",
        domain="types",
    ),
    ConstraintExample(
        id="go-interfaces-002",
        name="Empty Interface and Type Assertions",
        description="Type assertions and type switches with interface{}/any",
        scenario=(
            "Developer working with interface{} (or 'any' in Go 1.18+) and "
            "using type assertions to extract concrete types. Type assertion "
            "syntax: value.(Type) for single type, value.(type) in switch "
            "for multiple types."
        ),
        prompt="""Write a function that processes interface{} values. Use type switch or comma-ok
idiom (v, ok := val.(Type)) for safe type assertions. Never use unchecked assertions.

func processValue(val interface{}) string {
    """,
        spec=ConstraintSpec(
            language="go",
            # Regex enforces type switch or comma-ok type assertion
            regex=r"^func\s+\w+\s*\(\s*val\s+interface\{\}\s*\)[\s\S]*(?:switch\s+\w+\s*:=\s*val\.\(type\)|,\s*ok\s*:=\s*val\.\()",
            ebnf=r'''
root ::= type_switch | comma_ok_chain | comma_ok_single
type_switch ::= "func processValue(val interface{}) string {" nl "\tswitch v := val.(type) {" nl "\tcase string:" nl "\t\treturn v" nl "\tcase int:" nl "\t\treturn strconv.Itoa(v)" nl "\tdefault:" nl "\t\treturn fmt.Sprintf(\"%v\", v)" nl "\t}" nl "}"
comma_ok_chain ::= "func processValue(val interface{}) string {" nl "\tif s, ok := val.(string); ok {" nl "\t\treturn s" nl "\t}" nl "\tif i, ok := val.(int); ok {" nl "\t\treturn strconv.Itoa(i)" nl "\t}" nl "\treturn \"unknown\"" nl "}"
comma_ok_single ::= "func processValue(val interface{}) string {" nl "\tv, ok := val.(string)" nl "\tif !ok {" nl "\t\treturn \"not a string\"" nl "\t}" nl "\treturn v" nl "}"
nl ::= "\n"
''',
            type_bindings=[
                TypeBinding(name="val", type_expr="interface{}", scope="parameter"),
            ],
            function_signatures=[
                FunctionSignature(
                    name="processValue",
                    params=(TypeBinding("val", "interface{}"),),
                    return_type="string",
                ),
            ],
            semantic_constraints=[
                SemanticConstraint(
                    kind="precondition",
                    expression="checked_assertion(type_assert) || in_switch(type_assert)",
                    scope="processValue",
                    variables=("val", "type_assert"),
                ),
            ],
        ),
        expected_effect=(
            "Masks unsafe type assertions without checks. "
            "Requires either comma-ok idiom (v, ok := val.(Type)) or type switch. "
            "Prevents runtime panics from failed assertions."
        ),
        valid_outputs=[
            'func processValue(val interface{}) string {\n\tswitch v := val.(type) {\n\tcase string:\n\t\treturn v\n\tcase int:\n\t\treturn strconv.Itoa(v)\n\tdefault:\n\t\treturn fmt.Sprintf("%v", v)\n\t}\n}',
            'func processValue(val interface{}) string {\n\tif s, ok := val.(string); ok {\n\t\treturn s\n\t}\n\tif i, ok := val.(int); ok {\n\t\treturn strconv.Itoa(i)\n\t}\n\treturn "unknown"\n}',
            'func processValue(val interface{}) string {\n\tv, ok := val.(string)\n\tif !ok {\n\t\treturn "not a string"\n\t}\n\treturn v\n}',
        ],
        invalid_outputs=[
            'func processValue(val interface{}) string {\n\treturn val.(string)\n}',  # Unchecked assertion
            'func processValue(val interface{}) string {\n\ts := val.(string)\n\treturn s\n}',  # No comma-ok check
            'func processValue(val interface{}) string {\n\tswitch val.(type) {\n\tcase string:\n\t\treturn val.(string)\n\t}\n}',  # No type switch variable binding
        ],
        tags=["interfaces", "type-assertions", "type-switches", "any"],
        language="go",
        domain="types",
    ),
    ConstraintExample(
        id="go-interfaces-003",
        name="Reader/Writer Interface Patterns",
        description="Implement standard library io.Reader and io.Writer patterns",
        scenario=(
            "Developer implementing custom io.Reader and io.Writer for streaming "
            "data processing. These are foundational interfaces in Go. "
            "Reader: Read(p []byte) (n int, err error) must return bytes read "
            "and error. Writer: Write(p []byte) (n int, err error) must return "
            "bytes written. Both have specific contracts about n and err values."
        ),
        prompt="""Implement io.Reader on a custom type. Must follow the contract:
- n is in range [0, len(p)]
- Return io.EOF when no more data
- Partial reads are allowed

""",
        spec=ConstraintSpec(
            language="go",
            # Regex enforces io.Reader/Writer signature pattern
            regex=r"^func\s+\([^)]+\)\s+(?:Read|Write)\s*\(\s*\w+\s+\[\]byte\s*\)\s*\(\s*(?:n\s+)?int\s*,\s*(?:err\s+)?error\s*\)",
            ebnf=r'''
root ::= reader_impl | writer_impl
reader_impl ::= "type ByteReader struct {" nl "\tdata []byte" nl "\tpos int" nl "}" nl nl "func (r *ByteReader) Read(p []byte) (n int, err error) {" nl "\tif r.pos >= len(r.data) {" nl "\t\treturn 0, io.EOF" nl "\t}" nl "\tn = copy(p, r.data[r.pos:])" nl "\tr.pos += n" nl "\tif r.pos >= len(r.data) {" nl "\t\terr = io.EOF" nl "\t}" nl "\treturn" nl "}"
writer_impl ::= "type ByteWriter struct {" nl "\tbuf []byte" nl "}" nl nl "func (w *ByteWriter) Write(p []byte) (n int, err error) {" nl "\tw.buf = append(w.buf, p...)" nl "\treturn len(p), nil" nl "}"
nl ::= "\n"
''',
            type_bindings=[
                TypeBinding(name="buf", type_expr="[]byte", scope="local"),
            ],
            type_aliases={
                "io.Reader": "interface { Read(p []byte) (n int, err error) }",
                "io.Writer": "interface { Write(p []byte) (n int, err error) }",
            },
            semantic_constraints=[
                SemanticConstraint(
                    kind="postcondition",
                    expression="0 <= n <= len(p)",
                    scope="Read",
                    variables=("n", "p"),
                ),
                SemanticConstraint(
                    kind="postcondition",
                    expression="n < len(p) implies err != nil",
                    scope="Read",
                    variables=("n", "p", "err"),
                ),
                SemanticConstraint(
                    kind="postcondition",
                    expression="err == io.EOF or err == nil or n > 0",
                    scope="Read",
                    variables=("err", "n"),
                ),
            ],
        ),
        expected_effect=(
            "Masks Reader/Writer implementations violating io contracts. "
            "Ensures n is in valid range [0, len(p)]. "
            "Enforces error semantics: n < len(p) means err must be set."
        ),
        valid_outputs=[
            'type ByteReader struct {\n\tdata []byte\n\tpos int\n}\n\nfunc (r *ByteReader) Read(p []byte) (n int, err error) {\n\tif r.pos >= len(r.data) {\n\t\treturn 0, io.EOF\n\t}\n\tn = copy(p, r.data[r.pos:])\n\tr.pos += n\n\tif r.pos >= len(r.data) {\n\t\terr = io.EOF\n\t}\n\treturn\n}',
            'type ByteWriter struct {\n\tbuf []byte\n}\n\nfunc (w *ByteWriter) Write(p []byte) (n int, err error) {\n\tw.buf = append(w.buf, p...)\n\treturn len(p), nil\n}',
        ],
        invalid_outputs=[
            'func (r *ByteReader) Read(p []byte) (int, error) {\n\treturn len(r.data), nil\n}',  # Can return > len(p)
            'func (r *ByteReader) Read(p []byte) (int, error) {\n\tif r.pos >= len(r.data) {\n\t\treturn 0, nil\n\t}\n\treturn copy(p, r.data[r.pos:]), nil\n}',  # Should return EOF when exhausted
            'func (r *ByteReader) Read(p []byte) (int, error) {\n\treturn -1, io.EOF\n}',  # Negative n invalid
        ],
        tags=["interfaces", "io", "reader", "writer", "contracts"],
        language="go",
        domain="types",
    ),
    ConstraintExample(
        id="go-interfaces-004",
        name="Method Set - Pointer vs Value Receivers",
        description="Understand method sets for pointer vs value types",
        scenario=(
            "Developer implementing an interface where method set differs between "
            "*T and T. A type T's method set includes only value receiver methods. "
            "A type *T's method set includes both value and pointer receiver methods. "
            "This affects interface satisfaction."
        ),
        prompt="""Implement an Increment method on Counter. Since it modifies state, use pointer receiver.
Value receivers can't modify the original - mutations are lost.

type Counter struct { value int }

""",
        spec=ConstraintSpec(
            language="go",
            # Regex enforces pointer receiver for mutating methods
            regex=r"^type\s+\w+\s+struct[\s\S]*func\s+\(\s*\w+\s+\*\w+\s*\)\s+Increment\s*\(\s*\)",
            ebnf=r'''
root ::= increment_only | increment_with_getter
increment_only ::= "type Counter struct {" nl "\tvalue int" nl "}" nl nl "func (c *Counter) Increment() {" nl "\tc.value++" nl "}"
increment_with_getter ::= "type Counter struct {" nl "\tcount int" nl "}" nl nl "func (c *Counter) Increment() {" nl "\tc.count += 1" nl "}" nl nl "func (c Counter) Value() int {" nl "\treturn c.count" nl "}"
nl ::= "\n"
''',
            type_bindings=[
                TypeBinding(name="self", type_expr="*Counter", scope="local"),
            ],
            type_aliases={
                "Incrementer": "interface { Increment() }",
            },
            semantic_constraints=[
                SemanticConstraint(
                    kind="invariant",
                    expression="pointer_receiver(Increment)",
                    scope="Counter",
                    variables=("Increment",),
                ),
            ],
        ),
        expected_effect=(
            "Masks value receiver implementation for methods that modify state. "
            "Requires pointer receiver for mutating methods. "
            "Ensures correct method set for interface satisfaction."
        ),
        valid_outputs=[
            'type Counter struct {\n\tvalue int\n}\n\nfunc (c *Counter) Increment() {\n\tc.value++\n}',
            'type Counter struct {\n\tcount int\n}\n\nfunc (c *Counter) Increment() {\n\tc.count += 1\n}\n\nfunc (c Counter) Value() int {\n\treturn c.count\n}',  # Read-only can use value receiver
        ],
        invalid_outputs=[
            'func (c Counter) Increment() {\n\tc.value++\n}',  # Value receiver won't modify original
            'type Counter int\nfunc (c Counter) Increment() { c++ }',  # Value receiver on basic type
        ],
        tags=["interfaces", "method-sets", "receivers", "pointers"],
        language="go",
        domain="types",
    ),
    ConstraintExample(
        id="go-interfaces-005",
        name="Interface Nil Check Subtlety",
        description="Handle nil interface vs nil concrete value distinction",
        scenario=(
            "Developer checking if interface is nil. In Go, an interface value "
            "is nil only if both type and value are nil. An interface holding a "
            "nil pointer of a concrete type is NOT nil. This is a common gotcha."
        ),
        prompt="""Write a function that checks or handles nil correctly. Remember: interface with
nil concrete value is NOT nil. To return a nil error, return nil directly, not a typed nil.

""",
        spec=ConstraintSpec(
            language="go",
            # Regex enforces nil checking pattern (err != nil or val == nil)
            regex=r"func\s+\w+\s*\([^)]*\)[\s\S]*(?:\w+\s*[!=]=\s*nil|return\s+nil)",
            ebnf=r'''
root ::= check_error | is_nil | return_error
check_error ::= "func checkError(err error) bool {" nl "\tif err != nil {" nl "\t\treturn true" nl "\t}" nl "\treturn false" nl "}"
is_nil ::= "func isNil(val interface{}) bool {" nl "\treturn val == nil" nl "}"
return_error ::= "func returnError() error {" nl "\tvar err *MyError = nil" nl "\tif err != nil {" nl "\t\treturn err" nl "\t}" nl "\treturn nil" nl "}"
nl ::= "\n"
''',
            type_bindings=[
                TypeBinding(name="err", type_expr="error", scope="local"),
                TypeBinding(name="val", type_expr="interface{}", scope="local"),
            ],
            semantic_constraints=[
                SemanticConstraint(
                    kind="invariant",
                    expression="(interface == nil) == (type_part == nil && value_part == nil)",
                    scope="checkNil",
                    variables=("interface", "type_part", "value_part"),
                ),
            ],
        ),
        expected_effect=(
            "Masks incorrect nil checks on interfaces. "
            "Requires understanding that interface != nil can be true even "
            "if concrete value is nil. Shows correct checking patterns."
        ),
        valid_outputs=[
            'func checkError(err error) bool {\n\tif err != nil {\n\t\treturn true\n\t}\n\treturn false\n}',
            'func isNil(val interface{}) bool {\n\treturn val == nil\n}',
            'func returnError() error {\n\tvar err *MyError = nil\n\tif err != nil {\n\t\treturn err\n\t}\n\treturn nil\n}',  # Correct: don't wrap nil pointer
        ],
        invalid_outputs=[
            'func returnError() error {\n\tvar err *MyError = nil\n\treturn err\n}',  # Returns non-nil interface with nil value
            'func checkNil(val interface{}) bool {\n\tif v, ok := val.(*Type); ok {\n\t\treturn v == nil\n\t}\n\treturn false\n}',  # Confused logic
        ],
        tags=["interfaces", "nil", "gotchas", "error-handling"],
        language="go",
        domain="semantics",
    ),
]


__all__ = ["GO_INTERFACE_EXAMPLES"]
