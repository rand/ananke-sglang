# Copyright Rand Arete @ Ananke 2025
# Licensed under the Apache License, Version 2.0
"""Zig control flow constraint examples.

Demonstrates control flow constraints specific to Zig:
- Error defer cleanup (errdefer for error path cleanup)
- Comptime branches (@compileLog, comptime if)
- Unreachable assertions (safety in exhaustive switches)
"""

from __future__ import annotations

from typing import List

try:
    from ..base import ConstraintExample
    from .....spec.constraint_spec import (
        ConstraintSpec,
        TypeBinding,
        FunctionSignature,
        ControlFlowContext,
        SemanticConstraint,
    )
except ImportError:
    from tests.fixtures.constraints.base import ConstraintExample
    from spec.constraint_spec import (
        ConstraintSpec,
        TypeBinding,
        FunctionSignature,
        ControlFlowContext,
        SemanticConstraint,
    )


ZIG_CONTROLFLOW_EXAMPLES: List[ConstraintExample] = [
    ConstraintExample(
        id="zig-controlflow-001",
        name="Error Defer Cleanup",
        description="Resource cleanup using errdefer for error paths",
        scenario="Developer acquiring resources that must be cleaned up on error",
        prompt="""Complete this Zig function that allocates a resource with proper cleanup on error using errdefer:

pub fn createResource(allocator: std.mem.Allocator) !*Resource {
    """,
        spec=ConstraintSpec(
            language="zig",
            # Regex enforces try allocation followed by errdefer cleanup
            regex=r"^(?:const|var)\s+\w+\s*=\s*try[\s\S]*errdefer",
            ebnf=r'''
root ::= single_errdefer | chained_errdefer | private_errdefer
single_errdefer ::= "pub fn createResource(allocator: std.mem.Allocator) !*Resource {" nl "    const resource = try allocator.create(Resource);" nl "    errdefer allocator.destroy(resource);" nl "    try resource.init();" nl "    return resource;" nl "}"
chained_errdefer ::= "pub fn createResource(allocator: std.mem.Allocator) !*Resource {" nl "    const resource = try allocator.create(Resource);" nl "    errdefer allocator.destroy(resource);" nl "    resource.* = Resource{};" nl "    try resource.connect();" nl "    errdefer resource.disconnect();" nl "    return resource;" nl "}"
private_errdefer ::= "fn createResource(allocator: std.mem.Allocator) !*Resource {" nl "    var resource = try allocator.create(Resource);" nl "    errdefer allocator.destroy(resource);" nl "    resource.state = try initState(allocator);" nl "    errdefer resource.state.deinit();" nl "    return resource;" nl "}"
nl ::= "\n"
''',
            function_signatures=[
                FunctionSignature(
                    name="createResource",
                    params=(
                        TypeBinding(
                            "allocator", "std.mem.Allocator", scope="parameter"
                        ),
                    ),
                    return_type="!*Resource",
                ),
            ],
            control_flow=ControlFlowContext(
                function_name="createResource",
                in_try_block=True,
                exception_types=("OutOfMemory", "InvalidResource"),
            ),
            semantic_constraints=[
                SemanticConstraint(
                    kind="invariant",
                    expression="on_error_path(fn) ? resource_freed(resource) : true",
                    scope="createResource",
                    variables=("fn", "resource"),
                ),
            ],
            domain_configs={
                "controlflow": {
                    "require_errdefer": True,
                    "track_resource_lifetime": True,
                }
            },
            cache_scope="full_context",
        ),
        expected_effect="Masks control flow tokens that don't include errdefer cleanup; enforces error-path resource management",
        valid_outputs=[
            "const resource = try allocator.create(Resource);\n    errdefer allocator.destroy(resource);\n    try resource.init();\n    return resource;\n}",
            "const resource = try allocator.create(Resource);\n    errdefer allocator.destroy(resource);\n    resource.* = Resource{};\n    try resource.connect();\n    errdefer resource.disconnect();\n    return resource;\n}",
            "var resource = try allocator.create(Resource);\n    errdefer allocator.destroy(resource);\n    resource.state = try initState(allocator);\n    errdefer resource.state.deinit();\n    return resource;\n}",
        ],
        invalid_outputs=[
            "pub fn createResource(allocator: std.mem.Allocator) !*Resource {\n    const resource = try allocator.create(Resource);\n    try resource.init();\n    return resource;\n}",  # Missing errdefer
            "pub fn createResource(allocator: std.mem.Allocator) !*Resource {\n    const resource = try allocator.create(Resource);\n    defer allocator.destroy(resource);\n    try resource.init();\n    return resource;\n}",  # Wrong: defer runs on success too
            "pub fn createResource(allocator: std.mem.Allocator) !*Resource {\n    const resource = allocator.create(Resource) catch |err| { return err; };\n    try resource.init();\n    return resource;\n}",  # No cleanup on early return
        ],
        tags=["controlflow", "errdefer", "resource-management", "error-handling"],
        language="zig",
        domain="controlflow",
    ),
    ConstraintExample(
        id="zig-controlflow-002",
        name="Comptime Branches",
        description="Compile-time conditional logic using comptime if",
        scenario="Developer writing code that branches at compile time based on type properties",
        prompt="""Write a generic printValue function that handles different types at comptime.
Use @typeInfo to inspect the comptime type parameter, and use "if (comptime ...)"
to branch based on type. Handle at least Int and Float differently.

""",
        spec=ConstraintSpec(
            language="zig",
            # Regex enforces comptime type parameter with type introspection
            regex=r"(?:pub\s+)?fn\s+\w+\s*\(\s*comptime\s+\w+:\s*type[\s\S]*(?:if\s*\(\s*comptime|comptime\s*\{|@typeInfo)",
            ebnf=r'''
root ::= comptime_if_chain | comptime_log | comptime_block
comptime_if_chain ::= "pub fn printValue(comptime T: type, value: T) void {" nl "    if (comptime @typeInfo(T) == .Int) {" nl "        std.debug.print(\"Integer: {}\\n\", .{value});" nl "    } else if (comptime @typeInfo(T) == .Float) {" nl "        std.debug.print(\"Float: {d}\\n\", .{value});" nl "    } else {" nl "        @compileError(\"Unsupported type\");" nl "    }" nl "}"
comptime_log ::= "pub fn printValue(comptime T: type, value: T) void {" nl "    const info = @typeInfo(T);" nl "    if (comptime info == .Struct) {" nl "        @compileLog(\"Printing struct\", T);" nl "    }" nl "    std.debug.print(\"{any}\\n\", .{value});" nl "}"
comptime_block ::= "fn printValue(comptime T: type, value: T) void {" nl "    comptime {" nl "        const ti = @typeInfo(T);" nl "        if (ti != .Int and ti != .Float) {" nl "            @compileError(\"Only numeric types supported\");" nl "        }" nl "    }" nl "    std.debug.print(\"{}\\n\", .{value});" nl "}"
nl ::= "\n"
''',
            type_bindings=[
                TypeBinding(name="T", type_expr="comptime type", scope="parameter"),
            ],
            function_signatures=[
                FunctionSignature(
                    name="printValue",
                    params=(
                        TypeBinding("comptime_T", "type", scope="parameter"),
                        TypeBinding("value", "T", scope="parameter"),
                    ),
                    return_type="void",
                ),
            ],
            semantic_constraints=[
                SemanticConstraint(
                    kind="assertion",
                    expression="@typeInfo(T) != .Undefined",
                    scope="printValue",
                    variables=("T",),
                ),
            ],
            domain_configs={
                "controlflow": {
                    "require_comptime_evaluation": True,
                    "check_type_properties": True,
                }
            },
            cache_scope="full_context",
        ),
        expected_effect="Masks runtime branches where comptime branches required; ensures compile-time type introspection",
        valid_outputs=[
            'pub fn printValue(comptime T: type, value: T) void {\n    if (comptime @typeInfo(T) == .Int) {\n        std.debug.print("Integer: {}\\n", .{value});\n    } else if (comptime @typeInfo(T) == .Float) {\n        std.debug.print("Float: {d}\\n", .{value});\n    } else {\n        @compileError("Unsupported type");\n    }\n}',
            "pub fn printValue(comptime T: type, value: T) void {\n    const info = @typeInfo(T);\n    if (comptime info == .Struct) {\n        @compileLog(\"Printing struct\", T);\n    }\n    std.debug.print(\"{any}\\n\", .{value});\n}",
            'fn printValue(comptime T: type, value: T) void {\n    comptime {\n        const ti = @typeInfo(T);\n        if (ti != .Int and ti != .Float) {\n            @compileError("Only numeric types supported");\n        }\n    }\n    std.debug.print("{}\\n", .{value});\n}',
        ],
        invalid_outputs=[
            "pub fn printValue(comptime T: type, value: T) void {\n    if (@typeInfo(T) == .Int) { ... }\n}",  # Missing comptime keyword in if
            "pub fn printValue(comptime T: type, value: T) void {\n    const is_int = @typeInfo(T) == .Int;\n    if (is_int) { ... }\n}",  # Runtime check of comptime value
            "pub fn printValue(T: type, value: T) void { ... }",  # Missing comptime on parameter
        ],
        tags=["controlflow", "comptime", "type-introspection", "metaprogramming"],
        language="zig",
        domain="controlflow",
    ),
    ConstraintExample(
        id="zig-controlflow-003",
        name="Unreachable Assertions",
        description="Exhaustive switch with unreachable for safety guarantees",
        scenario="Developer handling all enum cases with unreachable for impossible states",
        prompt="""Write an exhaustive switch on an enum. Zig requires all cases to be handled -
either list each case explicitly (.idle, .running, .stopped) or use combined arms
(.idle, .stopped => {...}). Don't use else/_ wildcard - be exhaustive.

""",
        spec=ConstraintSpec(
            language="zig",
            # Simplified: Just require switch on state with enum cases
            regex=r"switch\s*\([^)]+\)\s*\{[\s\S]*\.\w+\s*=>",
            ebnf=r'''
root ::= enum_switch_print | switch_return | combined_arms
enum_switch_print ::= "const State = enum { idle, running, stopped };" nl nl "pub fn handleState(state: State) void {" nl "    switch (state) {" nl "        .idle => std.debug.print(\"Idle\\n\", .{})," nl "        .running => std.debug.print(\"Running\\n\", .{})," nl "        .stopped => std.debug.print(\"Stopped\\n\", .{})," nl "    }" nl "}"
switch_return ::= "fn handleState(state: State) u32 {" nl "    return switch (state) {" nl "        .idle => 0," nl "        .running => 1," nl "        .stopped => 2," nl "    };" nl "}"
combined_arms ::= "pub fn handleState(state: State) void {" nl "    switch (state) {" nl "        .idle, .stopped => { /* handle both */ }," nl "        .running => { /* handle running */ }," nl "    }" nl "}"
nl ::= "\n"
''',
            type_bindings=[
                TypeBinding(
                    name="state",
                    type_expr="enum { idle, running, stopped }",
                    scope="parameter",
                ),
            ],
            control_flow=ControlFlowContext(
                function_name="handleState",
                reachable=True,
            ),
            semantic_constraints=[
                SemanticConstraint(
                    kind="assertion",
                    expression="switch_exhaustive(state)",
                    scope="handleState",
                    variables=("state",),
                ),
            ],
            domain_configs={
                "controlflow": {
                    "require_exhaustive_switch": True,
                    "enforce_unreachable": True,
                }
            },
            cache_scope="full_context",
        ),
        expected_effect="Masks non-exhaustive switches; requires unreachable for provably impossible paths",
        valid_outputs=[
            "const State = enum { idle, running, stopped };\n\npub fn handleState(state: State) void {\n    switch (state) {\n        .idle => std.debug.print(\"Idle\\n\", .{}),\n        .running => std.debug.print(\"Running\\n\", .{}),\n        .stopped => std.debug.print(\"Stopped\\n\", .{}),\n    }\n}",
            "fn handleState(state: State) u32 {\n    return switch (state) {\n        .idle => 0,\n        .running => 1,\n        .stopped => 2,\n    };\n}",
            "pub fn handleState(state: State) void {\n    switch (state) {\n        .idle, .stopped => { /* handle both */ },\n        .running => { /* handle running */ },\n    }\n}",
        ],
        invalid_outputs=[
            "pub fn handleState(state: State) void {\n    switch (state) {\n        .idle => {},\n        .running => {},\n        else => unreachable,\n    }\n}",  # Should list .stopped explicitly
            "pub fn handleState(state: State) void {\n    if (state == .idle) { ... }\n    else if (state == .running) { ... }\n}",  # Non-exhaustive if chain
            "pub fn handleState(state: State) void {\n    switch (state) {\n        .idle => {},\n        _ => {},\n    }\n}",  # Wildcard instead of exhaustive listing
        ],
        tags=["controlflow", "unreachable", "exhaustiveness", "switch", "safety"],
        language="zig",
        domain="controlflow",
    ),
]


__all__ = ["ZIG_CONTROLFLOW_EXAMPLES"]
