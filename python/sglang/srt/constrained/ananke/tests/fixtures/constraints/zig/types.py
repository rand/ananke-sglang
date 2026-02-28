# Copyright Rand Arete @ Ananke 2025
# Licensed under the Apache License, Version 2.0
"""Zig type constraint examples.

Demonstrates type-level constraints specific to Zig:
- Comptime type selection with @TypeOf and comptime parameters
- Error union unwrapping (!T patterns)
- Sentinel-terminated arrays ([*:0]u8 null-terminated strings)
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


ZIG_TYPE_EXAMPLES: List[ConstraintExample] = [
    ConstraintExample(
        id="zig-types-001",
        name="Comptime Type Selection",
        description="Generic function using @TypeOf and comptime type parameters",
        scenario="Developer writing a generic swap function that preserves exact types",
        prompt="""Write a generic swap function in Zig using comptime type parameter.
Use pointers to swap values in place - must work for any type T.

""",
        spec=ConstraintSpec(
            language="zig",
            # Regex enforces comptime generic swap function signature
            regex=r"^(?:pub\s+)?fn\s+swap\s*\(\s*comptime\s+T:\s*type\s*,\s*a:\s*\*T\s*,\s*b:\s*\*T\s*\)\s*void",
            ebnf=r'''
root ::= private_swap | pub_swap
private_swap ::= "fn swap(comptime T: type, a: *T, b: *T) void {" nl "    const tmp: T = a.*;" nl "    a.* = b.*;" nl "    b.* = tmp;" nl "}"
pub_swap ::= "pub fn swap(comptime T: type, a: *T, b: *T) void {" nl "    const temp = a.*;" nl "    a.* = b.*;" nl "    b.* = temp;" nl "}"
nl ::= "\n"
''',
            expected_type="fn(comptime T: type, a: *T, b: *T) void",
            type_bindings=[
                TypeBinding(
                    name="T",
                    type_expr="comptime type",
                    scope="parameter",
                    mutable=False,
                ),
                TypeBinding(name="a", type_expr="*T", scope="parameter", mutable=True),
                TypeBinding(name="b", type_expr="*T", scope="parameter", mutable=True),
            ],
            semantic_constraints=[
                SemanticConstraint(
                    kind="postcondition",
                    expression="@TypeOf(a.*) == @TypeOf(b.*)",
                    scope="swap",
                    variables=("a", "b"),
                ),
            ],
            cache_scope="full_context",
        ),
        expected_effect="Masks tokens that don't maintain comptime type parameter T; ensures pointer dereference matches generic type",
        valid_outputs=[
            "fn swap(comptime T: type, a: *T, b: *T) void {\n    const tmp: T = a.*;\n    a.* = b.*;\n    b.* = tmp;\n}",
            "pub fn swap(comptime T: type, a: *T, b: *T) void {\n    const temp = a.*;\n    a.* = b.*;\n    b.* = temp;\n}",
        ],
        invalid_outputs=[
            "fn swap(a: *anyopaque, b: *anyopaque) void { ... }",  # Lost type parameter
            "fn swap(comptime T: type, a: T, b: T) void { ... }",  # Missing pointer indirection
            "fn swap(comptime T: type, a: *T, b: *T) void { a = b; }",  # Type error: assigns pointer not value
        ],
        tags=["types", "comptime", "generics", "pointers"],
        language="zig",
        domain="types",
    ),
    ConstraintExample(
        id="zig-types-002",
        name="Error Union Unwrap",
        description="Function returning error union with explicit error handling",
        scenario="Developer parsing an integer from a string with proper error handling",
        prompt="""Write a parseInt function that takes a string slice and returns !i32 (error union).
Use std.fmt.parseInt for the actual parsing. The error union return type allows callers
to handle parse failures with catch or try.

""",
        spec=ConstraintSpec(
            language="zig",
            # Regex enforces error union return type (!i32) and function signature
            regex=r"^fn\s+parseInt\s*\(\s*input:\s*\[\]const\s+u8\s*\)\s*!i32",
            ebnf=r'''
root ::= simple_parse | catch_parse | guard_parse
simple_parse ::= "fn parseInt(input: []const u8) !i32 {" nl "    return std.fmt.parseInt(i32, input, 10);" nl "}"
catch_parse ::= "fn parseInt(input: []const u8) !i32 {" nl "    const result = std.fmt.parseInt(i32, input, 10) catch return error.InvalidFormat;" nl "    return result;" nl "}"
guard_parse ::= "fn parseInt(input: []const u8) !i32 {" nl "    if (input.len == 0) return error.EmptyInput;" nl "    return try std.fmt.parseInt(i32, input, 10);" nl "}"
nl ::= "\n"
''',
            expected_type="!i32",
            function_signatures=[
                FunctionSignature(
                    name="parseInt",
                    params=(TypeBinding("input", "[]const u8", scope="parameter"),),
                    return_type="!i32",
                ),
            ],
            semantic_constraints=[
                SemanticConstraint(
                    kind="postcondition",
                    expression="result >= std.math.minInt(i32) and result <= std.math.maxInt(i32)",
                    scope="parseInt",
                    variables=("result",),
                ),
            ],
            cache_scope="full_context",
        ),
        expected_effect="Masks tokens producing non-error-union types; requires explicit error handling syntax",
        valid_outputs=[
            "fn parseInt(input: []const u8) !i32 {\n    return std.fmt.parseInt(i32, input, 10);\n}",
            "fn parseInt(input: []const u8) !i32 {\n    const result = std.fmt.parseInt(i32, input, 10) catch return error.InvalidFormat;\n    return result;\n}",
            "fn parseInt(input: []const u8) !i32 {\n    if (input.len == 0) return error.EmptyInput;\n    return try std.fmt.parseInt(i32, input, 10);\n}",
        ],
        invalid_outputs=[
            "fn parseInt(input: []const u8) i32 { ... }",  # Missing error union
            "fn parseInt(input: []const u8) !i32 { return 42; }",  # Missing error context
            "fn parseInt(input: []const u8) error{InvalidFormat}!i32 { ... }",  # Narrower than spec
        ],
        tags=["types", "error-union", "error-handling"],
        language="zig",
        domain="types",
    ),
    ConstraintExample(
        id="zig-types-003",
        name="Sentinel-Terminated Array",
        description="Function working with null-terminated C strings using [*:0]u8",
        scenario="Developer writing FFI wrapper for C library expecting null-terminated strings",
        prompt="""Write a strlen function for C interop. Takes [*:0]const u8 (sentinel-terminated pointer
to null-terminated string) and returns usize. Loop until you hit the null terminator.
Note: sentinel pointers don't have .len - you must scan for the sentinel.

""",
        spec=ConstraintSpec(
            language="zig",
            # Regex enforces sentinel-terminated pointer type [*:0]const u8
            regex=r"^(?:pub\s+)?fn\s+strlen\s*\(\s*str:\s*\[\*:0\]const\s+u8\s*\)\s*usize",
            ebnf=r'''
root ::= while_expr_style | while_block_style | stdlib_style | pub_while_style
while_expr_style ::= "fn strlen(str: [*:0]const u8) usize {" nl "    var len: usize = 0;" nl "    while (str[len] != 0) : (len += 1) {}" nl "    return len;" nl "}"
while_block_style ::= "fn strlen(str: [*:0]const u8) usize {" nl "    var i: usize = 0;" nl "    while (str[i] != 0) { i += 1; }" nl "    return i;" nl "}"
stdlib_style ::= "fn strlen(str: [*:0]const u8) usize {" nl "    return std.mem.len(str);" nl "}"
pub_while_style ::= "pub fn strlen(str: [*:0]const u8) usize {" nl "    var i: usize = 0;" nl "    while (str[i] != 0) { i += 1; }" nl "    return i;" nl "}"
nl ::= "\n"
''',
            expected_type="usize",
            type_bindings=[
                TypeBinding(
                    name="str",
                    type_expr="[*:0]const u8",
                    scope="parameter",
                    mutable=False,
                ),
            ],
            function_signatures=[
                FunctionSignature(
                    name="strlen",
                    params=(
                        TypeBinding(
                            "str", "[*:0]const u8", scope="parameter", mutable=False
                        ),
                    ),
                    return_type="usize",
                ),
            ],
            semantic_constraints=[
                SemanticConstraint(
                    kind="precondition",
                    expression="str[result] == 0",
                    scope="strlen",
                    variables=("str", "result"),
                ),
            ],
            cache_scope="full_context",
        ),
        expected_effect="Masks tokens not handling sentinel-terminated pointers; ensures null terminator awareness",
        valid_outputs=[
            "fn strlen(str: [*:0]const u8) usize {\n    var len: usize = 0;\n    while (str[len] != 0) : (len += 1) {}\n    return len;\n}",
            "pub fn strlen(str: [*:0]const u8) usize {\n    var i: usize = 0;\n    while (str[i] != 0) { i += 1; }\n    return i;\n}",
            "fn strlen(str: [*:0]const u8) usize {\n    return std.mem.len(str);\n}",
        ],
        invalid_outputs=[
            "fn strlen(str: []const u8) usize { ... }",  # Wrong type: slice not sentinel pointer
            "fn strlen(str: [*]const u8) usize { ... }",  # Missing sentinel annotation
            "fn strlen(str: [*:0]const u8) usize { return str.len; }",  # No .len on sentinel pointers
        ],
        tags=["types", "sentinel", "c-interop", "pointers"],
        language="zig",
        domain="types",
    ),
]


__all__ = ["ZIG_TYPE_EXAMPLES"]
