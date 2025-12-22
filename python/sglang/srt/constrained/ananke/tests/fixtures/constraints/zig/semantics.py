# Copyright Rand Arete @ Ananke 2025
# Licensed under the Apache License, Version 2.0
"""Zig semantic constraint examples.

Demonstrates semantic-level constraints specific to Zig:
- Allocation safety (must free what you allocate)
- Memory leak prevention (paired alloc/free)
- Comptime assertions (@compileError for impossible states)
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
        ControlFlowContext,
    )
except ImportError:
    from tests.fixtures.constraints.base import ConstraintExample
    from spec.constraint_spec import (
        ConstraintSpec,
        TypeBinding,
        FunctionSignature,
        SemanticConstraint,
        ControlFlowContext,
    )


ZIG_SEMANTICS_EXAMPLES: List[ConstraintExample] = [
    ConstraintExample(
        id="zig-semantics-001",
        name="Allocation Safety",
        description="Every allocation must have corresponding free or transfer ownership",
        scenario="Developer writing function that allocates memory and must ensure cleanup",
        prompt="""Write a processData function that allocates a buffer using std.mem.Allocator.
Always use defer or errdefer to ensure the buffer is freed - never leak memory.
Zig doesn't have automatic memory management, so cleanup must be explicit.

pub fn processData(allocator: std.mem.Allocator, input: []const u8) void {
    """,
        spec=ConstraintSpec(
            language="zig",
            # Regex enforces defer/errdefer for allocator.free after alloc
            regex=r"allocator\.alloc[\s\S]*(?:defer|errdefer)\s+allocator\.free",
            ebnf=r'''
root ::= basic_defer | size_copy | errdefer_explicit
basic_defer ::= "pub fn processData(allocator: std.mem.Allocator, input: []const u8) void {" nl "    const buffer = allocator.alloc(u8, input.len) catch return;" nl "    defer allocator.free(buffer);" nl "    @memcpy(buffer, input);" nl "    // process buffer" nl "}"
size_copy ::= "pub fn processData(allocator: std.mem.Allocator, input: []const u8) void {" nl "    const buffer = allocator.alloc(u8, 1024) catch return;" nl "    defer allocator.free(buffer);" nl "    const copied = @min(buffer.len, input.len);" nl "    @memcpy(buffer[0..copied], input[0..copied]);" nl "}"
errdefer_explicit ::= "pub fn processData(allocator: std.mem.Allocator, input: []const u8) void {" nl "    var buffer = allocator.alloc(u8, input.len) catch return;" nl "    errdefer allocator.free(buffer);" nl "    @memcpy(buffer, input);" nl "    process(buffer);" nl "    allocator.free(buffer);" nl "}"
nl ::= "\n"
''',
            function_signatures=[
                FunctionSignature(
                    name="processData",
                    params=(
                        TypeBinding(
                            "allocator", "std.mem.Allocator", scope="parameter"
                        ),
                        TypeBinding("input", "[]const u8", scope="parameter"),
                    ),
                    return_type="void",
                ),
            ],
            semantic_constraints=[
                SemanticConstraint(
                    kind="postcondition",
                    expression="forall(buffer, allocated_in_scope(buffer) => freed_in_scope(buffer) || returned(buffer))",
                    scope="processData",
                    variables=("allocator", "buffer"),
                ),
                SemanticConstraint(
                    kind="invariant",
                    expression="alloc_count(allocator) == free_count(allocator) || ownership_transferred(allocator)",
                    scope="processData",
                    variables=("allocator",),
                ),
            ],
            domain_configs={
                "semantic": {
                    "track_allocations": True,
                    "require_paired_alloc_free": True,
                    "allow_ownership_transfer": True,
                }
            },
            cache_scope="full_context",
        ),
        expected_effect="Masks code paths with unpaired allocations; requires defer or errdefer for cleanup",
        valid_outputs=[
            "pub fn processData(allocator: std.mem.Allocator, input: []const u8) void {\n    const buffer = allocator.alloc(u8, input.len) catch return;\n    defer allocator.free(buffer);\n    @memcpy(buffer, input);\n    // process buffer\n}",
            "pub fn processData(allocator: std.mem.Allocator, input: []const u8) void {\n    const buffer = allocator.alloc(u8, 1024) catch return;\n    defer allocator.free(buffer);\n    const copied = @min(buffer.len, input.len);\n    @memcpy(buffer[0..copied], input[0..copied]);\n}",
            "pub fn processData(allocator: std.mem.Allocator, input: []const u8) void {\n    var buffer = allocator.alloc(u8, input.len) catch return;\n    errdefer allocator.free(buffer);\n    @memcpy(buffer, input);\n    process(buffer);\n    allocator.free(buffer);\n}",
        ],
        invalid_outputs=[
            "pub fn processData(allocator: std.mem.Allocator, input: []const u8) void {\n    const buffer = allocator.alloc(u8, input.len) catch return;\n    @memcpy(buffer, input);\n}",  # Memory leak: no free
            "pub fn processData(allocator: std.mem.Allocator, input: []const u8) void {\n    const buffer = allocator.alloc(u8, input.len) catch return;\n    if (input.len > 0) {\n        allocator.free(buffer);\n    }\n}",  # Conditional free: potential leak
            "pub fn processData(allocator: std.mem.Allocator, input: []const u8) void {\n    var buffer = allocator.alloc(u8, 1024) catch return;\n    if (shouldProcess(input)) {\n        defer allocator.free(buffer);\n    }\n}",  # Conditional cleanup: leak on else path
        ],
        tags=["semantics", "memory-safety", "allocation", "resource-management"],
        language="zig",
        domain="semantics",
    ),
    ConstraintExample(
        id="zig-semantics-002",
        name="Memory Leak Prevention",
        description="Detect and prevent memory leaks through ownership tracking",
        scenario="Developer creating data structure that owns allocated memory",
        prompt="""Define a Container struct that owns allocated memory. Store the allocator as a field
so deinit can free the data. Every struct owning memory must have a deinit method.
Use errdefer when allocating multiple fields to clean up partial state on failure.

""",
        spec=ConstraintSpec(
            language="zig",
            # Regex enforces Container with both init and deinit methods
            regex=r"^const\s+Container\s*=\s*struct[\s\S]*pub\s+fn\s+init[\s\S]*pub\s+fn\s+deinit",
            ebnf=r'''
root ::= single_field | multi_field
single_field ::= "const Container = struct {" nl "    allocator: std.mem.Allocator," nl "    data: []u8," nl nl "    pub fn init(allocator: std.mem.Allocator) !Container {" nl "        const data = try allocator.alloc(u8, 256);" nl "        return Container{ .allocator = allocator, .data = data };" nl "    }" nl nl "    pub fn deinit(self: *Container) void {" nl "        self.allocator.free(self.data);" nl "    }" nl "};"
multi_field ::= "const Container = struct {" nl "    allocator: std.mem.Allocator," nl "    data: []u8," nl "    extra: []u32," nl nl "    pub fn init(allocator: std.mem.Allocator) !Container {" nl "        const data = try allocator.alloc(u8, 256);" nl "        errdefer allocator.free(data);" nl "        const extra = try allocator.alloc(u32, 64);" nl "        return Container{ .allocator = allocator, .data = data, .extra = extra };" nl "    }" nl nl "    pub fn deinit(self: *Container) void {" nl "        self.allocator.free(self.data);" nl "        self.allocator.free(self.extra);" nl "    }" nl "};"
nl ::= "\n"
''',
            function_signatures=[
                FunctionSignature(
                    name="init",
                    params=(
                        TypeBinding(
                            "allocator", "std.mem.Allocator", scope="parameter"
                        ),
                    ),
                    return_type="!Container",
                ),
                FunctionSignature(
                    name="deinit",
                    params=(TypeBinding("self", "*Container", scope="parameter"),),
                    return_type="void",
                ),
            ],
            type_bindings=[
                TypeBinding(
                    name="Container",
                    type_expr="struct { allocator: std.mem.Allocator, data: []u8 }",
                    scope="module",
                ),
            ],
            semantic_constraints=[
                SemanticConstraint(
                    kind="invariant",
                    expression="has_deinit(structs_with_owned_allocs)",
                    scope="Container",
                    variables=("Container", "structs_with_owned_allocs"),
                ),
                SemanticConstraint(
                    kind="postcondition",
                    expression="post_deinit(self) => owned_memory(self).len == 0",
                    scope="Container.deinit",
                    variables=("self",),
                ),
            ],
            domain_configs={
                "semantic": {
                    "require_deinit_for_owned_memory": True,
                    "track_struct_ownership": True,
                }
            },
            cache_scope="full_context",
        ),
        expected_effect="Masks struct definitions lacking deinit when owning memory; enforces cleanup method pattern",
        valid_outputs=[
            "const Container = struct {\n    allocator: std.mem.Allocator,\n    data: []u8,\n\n    pub fn init(allocator: std.mem.Allocator) !Container {\n        const data = try allocator.alloc(u8, 256);\n        return Container{ .allocator = allocator, .data = data };\n    }\n\n    pub fn deinit(self: *Container) void {\n        self.allocator.free(self.data);\n    }\n};",
            "const Container = struct {\n    allocator: std.mem.Allocator,\n    data: []u8,\n    extra: []u32,\n\n    pub fn init(allocator: std.mem.Allocator) !Container {\n        const data = try allocator.alloc(u8, 256);\n        errdefer allocator.free(data);\n        const extra = try allocator.alloc(u32, 64);\n        return Container{ .allocator = allocator, .data = data, .extra = extra };\n    }\n\n    pub fn deinit(self: *Container) void {\n        self.allocator.free(self.data);\n        self.allocator.free(self.extra);\n    }\n};",
        ],
        invalid_outputs=[
            "const Container = struct {\n    allocator: std.mem.Allocator,\n    data: []u8,\n\n    pub fn init(allocator: std.mem.Allocator) !Container {\n        const data = try allocator.alloc(u8, 256);\n        return Container{ .allocator = allocator, .data = data };\n    }\n};",  # Missing deinit method
            "const Container = struct {\n    allocator: std.mem.Allocator,\n    data: []u8,\n\n    pub fn deinit(self: *Container) void {\n        // Empty deinit doesn't free data\n    }\n};",  # Incomplete deinit
            "const Container = struct {\n    data: []u8,\n\n    pub fn deinit(self: *Container) void {\n        // Can't free without allocator reference\n    }\n};",  # Missing allocator field
        ],
        tags=["semantics", "memory-leak", "ownership", "deinit-pattern"],
        language="zig",
        domain="semantics",
        max_tokens=2048,  # Container with init/deinit is verbose
    ),
    ConstraintExample(
        id="zig-semantics-003",
        name="Comptime Assertions",
        description="Use @compileError to prevent impossible or unsafe states at compile time",
        scenario="Developer enforcing type constraints that must be verified at compile time",
        prompt="""Write a generic sum function that only accepts numeric types (Int or Float).
Use @typeInfo to check the type at comptime, and @compileError to reject invalid types.
This catches type errors at compile time rather than runtime.

""",
        spec=ConstraintSpec(
            language="zig",
            # Regex enforces @compileError for type validation
            regex=r"^(?:pub\s+)?fn\s+\w+\s*\(\s*comptime\s+\w+:\s*type[\s\S]*@compileError",
            ebnf=r'''
root ::= if_check | switch_check | private_check
if_check ::= "pub fn sum(comptime T: type, slice: []const T) T {" nl "    const info = @typeInfo(T);" nl "    if (info != .Int and info != .Float) {" nl "        @compileError(\"sum() requires numeric type\");" nl "    }" nl "    var total: T = 0;" nl "    for (slice) |item| { total += item; }" nl "    return total;" nl "}"
switch_check ::= "pub fn sum(comptime T: type, slice: []const T) T {" nl "    comptime {" nl "        switch (@typeInfo(T)) {" nl "            .Int, .Float => {}," nl "            else => @compileError(\"T must be Int or Float\")," nl "        }" nl "    }" nl "    var result: T = 0;" nl "    for (slice) |value| { result += value; }" nl "    return result;" nl "}"
private_check ::= "fn sum(comptime T: type, slice: []const T) T {" nl "    if (!@typeInfo(T).Int and !@typeInfo(T).Float) {" nl "        @compileError(\"Unsupported type for sum\");" nl "    }" nl "    var s: T = 0;" nl "    for (slice) |x| s += x;" nl "    return s;" nl "}"
nl ::= "\n"
''',
            type_bindings=[
                TypeBinding(name="T", type_expr="comptime type", scope="parameter"),
            ],
            function_signatures=[
                FunctionSignature(
                    name="sum",
                    params=(
                        TypeBinding("comptime_T", "type", scope="parameter"),
                        TypeBinding("slice", "[]const T", scope="parameter"),
                    ),
                    return_type="T",
                ),
            ],
            semantic_constraints=[
                SemanticConstraint(
                    kind="precondition",
                    expression="@typeInfo(T) == .Int || @typeInfo(T) == .Float",
                    scope="sum",
                    variables=("T",),
                ),
                SemanticConstraint(
                    kind="assertion",
                    expression="is_numeric_type(T)",
                    scope="sum",
                    variables=("T",),
                ),
            ],
            domain_configs={
                "semantic": {
                    "require_comptime_validation": True,
                    "enforce_type_constraints": True,
                }
            },
            cache_scope="full_context",
        ),
        expected_effect="Masks code without comptime type validation; requires @compileError for invalid types",
        valid_outputs=[
            'pub fn sum(comptime T: type, slice: []const T) T {\n    const info = @typeInfo(T);\n    if (info != .Int and info != .Float) {\n        @compileError("sum() requires numeric type");\n    }\n    var total: T = 0;\n    for (slice) |item| { total += item; }\n    return total;\n}',
            'pub fn sum(comptime T: type, slice: []const T) T {\n    comptime {\n        switch (@typeInfo(T)) {\n            .Int, .Float => {},\n            else => @compileError("T must be Int or Float"),\n        }\n    }\n    var result: T = 0;\n    for (slice) |value| { result += value; }\n    return result;\n}',
            'fn sum(comptime T: type, slice: []const T) T {\n    if (!@typeInfo(T).Int and !@typeInfo(T).Float) {\n        @compileError("Unsupported type for sum");\n    }\n    var s: T = 0;\n    for (slice) |x| s += x;\n    return s;\n}',
        ],
        invalid_outputs=[
            "pub fn sum(comptime T: type, slice: []const T) T {\n    var total: T = 0;\n    for (slice) |item| { total += item; }\n    return total;\n}",  # No compile-time validation
            "pub fn sum(comptime T: type, slice: []const T) T {\n    if (@typeInfo(T) != .Int) {\n        return error.InvalidType;\n    }\n    ...\n}",  # Runtime error not compile-time
            'pub fn sum(comptime T: type, slice: []const T) T {\n    std.debug.assert(@typeInfo(T) == .Int);\n    ...\n}',  # Runtime assert not @compileError
        ],
        tags=["semantics", "comptime", "compile-error", "type-safety", "assertions"],
        language="zig",
        domain="semantics",
    ),
]


__all__ = ["ZIG_SEMANTICS_EXAMPLES"]
