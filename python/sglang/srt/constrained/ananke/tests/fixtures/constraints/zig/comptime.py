# Copyright Rand Arete @ Ananke 2025
# Licensed under the Apache License, Version 2.0
"""Zig comptime deep dive examples.

Advanced examples demonstrating Zig's unique compile-time features:
- Generic data structures with comptime metaprogramming
- Custom allocator patterns and interfaces
- Error set inference and composition
"""

from __future__ import annotations

from typing import List

try:
    from ..base import ConstraintExample
    from .....spec.constraint_spec import (
        ConstraintSpec,
        TypeBinding,
        FunctionSignature,
        ClassDefinition,
        SemanticConstraint,
        ControlFlowContext,
    )
except ImportError:
    from tests.fixtures.constraints.base import ConstraintExample
    from spec.constraint_spec import (
        ConstraintSpec,
        TypeBinding,
        FunctionSignature,
        ClassDefinition,
        SemanticConstraint,
        ControlFlowContext,
    )


ZIG_COMPTIME_EXAMPLES: List[ConstraintExample] = [
    ConstraintExample(
        id="zig-comptime-001",
        name="Generic Data Structure with Comptime",
        description="Implement generic ArrayList-like container using comptime type parameters",
        scenario="Developer creating a reusable generic container with compile-time specialization",
        prompt="""Create a generic List type using Zig's comptime. Define fn List(comptime T: type) type
that returns a struct. The struct needs init, deinit, and append methods.
Store allocator and items as fields. Use @This() for Self reference.

""",
        spec=ConstraintSpec(
            language="zig",
            # Simplified: Just require comptime type parameter in generic function
            regex=r"(?:pub\s+)?fn\s+\w+\s*\(\s*comptime\s+\w+:\s*type\s*\)",
            ebnf=r'''
root ::= pub_list | private_list
pub_list ::= "pub fn List(comptime T: type) type {" nl "    return struct {" nl "        const Self = @This();" nl nl "        allocator: std.mem.Allocator," nl "        items: []T," nl "        capacity: usize," nl nl "        pub fn init(allocator: std.mem.Allocator) Self {" nl "            return Self{" nl "                .allocator = allocator," nl "                .items = &[_]T{}," nl "                .capacity = 0," nl "            };" nl "        }" nl nl "        pub fn deinit(self: *Self) void {" nl "            if (self.capacity > 0) {" nl "                self.allocator.free(self.items.ptr[0..self.capacity]);" nl "            }" nl "        }" nl nl "        pub fn append(self: *Self, item: T) !void {" nl "            if (self.items.len >= self.capacity) {" nl "                try self.grow();" nl "            }" nl "            self.items.len += 1;" nl "            self.items[self.items.len - 1] = item;" nl "        }" nl nl "        fn grow(self: *Self) !void {" nl "            const new_capacity = if (self.capacity == 0) 8 else self.capacity * 2;" nl "            const new_memory = try self.allocator.alloc(T, new_capacity);" nl "            if (self.capacity > 0) {" nl "                @memcpy(new_memory[0..self.items.len], self.items);" nl "                self.allocator.free(self.items.ptr[0..self.capacity]);" nl "            }" nl "            self.items.ptr = new_memory.ptr;" nl "            self.capacity = new_capacity;" nl "        }" nl "    };" nl "}"
private_list ::= "fn List(comptime T: type) type {" nl "    return struct {" nl "        allocator: std.mem.Allocator," nl "        items: []T," nl "        capacity: usize," nl nl "        const Self = @This();" nl nl "        pub fn init(allocator: std.mem.Allocator) Self {" nl "            return .{ .allocator = allocator, .items = &.{}, .capacity = 0 };" nl "        }" nl nl "        pub fn deinit(self: *Self) void {" nl "            if (self.items.ptr) |ptr| {" nl "                self.allocator.free(ptr[0..self.capacity]);" nl "            }" nl "        }" nl nl "        pub fn append(self: *Self, item: T) !void {" nl "            if (self.items.len == self.capacity) {" nl "                const new_cap = @max(8, self.capacity * 2);" nl "                const new_mem = try self.allocator.realloc(self.items.ptr[0..self.capacity], new_cap);" nl "                self.items.ptr = new_mem.ptr;" nl "                self.capacity = new_cap;" nl "            }" nl "            self.items.len += 1;" nl "            self.items[self.items.len - 1] = item;" nl "        }" nl "    };" nl "}"
nl ::= "\n"
''',
            class_definitions=[
                ClassDefinition(
                    name="List",
                    type_params=("T",),
                    instance_vars=(
                        TypeBinding("allocator", "std.mem.Allocator"),
                        TypeBinding("items", "[]T"),
                        TypeBinding("capacity", "usize"),
                    ),
                    methods=(
                        FunctionSignature(
                            name="init",
                            params=(
                                TypeBinding(
                                    "allocator", "std.mem.Allocator", scope="parameter"
                                ),
                            ),
                            return_type="List(T)",
                        ),
                        FunctionSignature(
                            name="deinit",
                            params=(
                                TypeBinding("self", "*List(T)", scope="parameter"),
                            ),
                            return_type="void",
                        ),
                        FunctionSignature(
                            name="append",
                            params=(
                                TypeBinding("self", "*List(T)", scope="parameter"),
                                TypeBinding("item", "T", scope="parameter"),
                            ),
                            return_type="!void",
                        ),
                    ),
                ),
            ],
            semantic_constraints=[
                SemanticConstraint(
                    kind="invariant",
                    expression="items.len <= capacity",
                    scope="List",
                    variables=("items", "capacity"),
                ),
                SemanticConstraint(
                    kind="postcondition",
                    expression="after deinit: items.ptr == null",
                    scope="List.deinit",
                    variables=("self",),
                ),
            ],
            domain_configs={
                "comptime": {
                    "require_generic_fn": True,
                    "enforce_type_parameter": True,
                    "check_comptime_evaluation": True,
                }
            },
            cache_scope="full_context",
        ),
        expected_effect="Masks non-generic implementations; requires proper comptime type parameterization",
        valid_outputs=[
            """pub fn List(comptime T: type) type {
    return struct {
        const Self = @This();

        allocator: std.mem.Allocator,
        items: []T,
        capacity: usize,

        pub fn init(allocator: std.mem.Allocator) Self {
            return Self{
                .allocator = allocator,
                .items = &[_]T{},
                .capacity = 0,
            };
        }

        pub fn deinit(self: *Self) void {
            if (self.capacity > 0) {
                self.allocator.free(self.items.ptr[0..self.capacity]);
            }
        }

        pub fn append(self: *Self, item: T) !void {
            if (self.items.len >= self.capacity) {
                try self.grow();
            }
            self.items.len += 1;
            self.items[self.items.len - 1] = item;
        }

        fn grow(self: *Self) !void {
            const new_capacity = if (self.capacity == 0) 8 else self.capacity * 2;
            const new_memory = try self.allocator.alloc(T, new_capacity);
            if (self.capacity > 0) {
                @memcpy(new_memory[0..self.items.len], self.items);
                self.allocator.free(self.items.ptr[0..self.capacity]);
            }
            self.items.ptr = new_memory.ptr;
            self.capacity = new_capacity;
        }
    };
}""",
            """fn List(comptime T: type) type {
    return struct {
        allocator: std.mem.Allocator,
        items: []T,
        capacity: usize,

        const Self = @This();

        pub fn init(allocator: std.mem.Allocator) Self {
            return .{ .allocator = allocator, .items = &.{}, .capacity = 0 };
        }

        pub fn deinit(self: *Self) void {
            if (self.items.ptr) |ptr| {
                self.allocator.free(ptr[0..self.capacity]);
            }
        }

        pub fn append(self: *Self, item: T) !void {
            if (self.items.len == self.capacity) {
                const new_cap = @max(8, self.capacity * 2);
                const new_mem = try self.allocator.realloc(self.items.ptr[0..self.capacity], new_cap);
                self.items.ptr = new_mem.ptr;
                self.capacity = new_cap;
            }
            self.items.len += 1;
            self.items[self.items.len - 1] = item;
        }
    };
}""",
        ],
        invalid_outputs=[
            "pub const List = struct { items: []anyopaque, ... };",  # Not generic
            "pub fn List(T: type) type { ... }",  # Missing comptime keyword
            "pub fn List(comptime T: type) struct { ... }",  # Doesn't return type
        ],
        tags=["comptime", "generics", "data-structures", "metaprogramming"],
        language="zig",
        domain="comptime",
        max_tokens=2048,  # Complex generic struct needs more tokens
    ),
    ConstraintExample(
        id="zig-comptime-002",
        name="Custom Allocator Pattern",
        description="Implement custom allocator conforming to std.mem.Allocator interface",
        scenario="Developer creating arena allocator with proper Allocator interface",
        prompt="""Implement a custom arena allocator that conforms to std.mem.Allocator interface.
Define ArenaAllocator struct with an allocator() method that returns std.mem.Allocator.
The vtable needs alloc, resize, and free function pointers with the exact signatures.

""",
        spec=ConstraintSpec(
            language="zig",
            # Simplified: Just require struct with allocator method
            regex=r"const\s+\w+\s*=\s*struct[\s\S]*(?:pub\s+)?fn\s+allocator",
            ebnf=r'''
root ::= arena_allocator
arena_allocator ::= "const ArenaAllocator = struct {" nl "    child_allocator: std.mem.Allocator," nl "    buffer: []u8," nl "    offset: usize," nl nl "    const Self = @This();" nl nl "    pub fn init(child: std.mem.Allocator) Self {" nl "        return Self{" nl "            .child_allocator = child," nl "            .buffer = &[_]u8{}," nl "            .offset = 0," nl "        };" nl "    }" nl nl "    pub fn allocator(self: *Self) std.mem.Allocator {" nl "        return .{" nl "            .ptr = self," nl "            .vtable = &.{" nl "                .alloc = alloc," nl "                .resize = resize," nl "                .free = free," nl "            }," nl "        };" nl "    }" nl nl "    fn alloc(ctx: *anyopaque, len: usize, ptr_align: u8, ret_addr: usize) ?[*]u8 {" nl "        const self: *Self = @ptrCast(@alignCast(ctx));" nl "        _ = ret_addr;" nl "        const align_offset = std.mem.alignForward(usize, self.offset, ptr_align);" nl "        const new_offset = align_offset + len;" nl "        if (new_offset > self.buffer.len) {" nl "            const new_buffer = self.child_allocator.alloc(u8, new_offset * 2) catch return null;" nl "            @memcpy(new_buffer[0..self.offset], self.buffer[0..self.offset]);" nl "            if (self.buffer.len > 0) {" nl "                self.child_allocator.free(self.buffer);" nl "            }" nl "            self.buffer = new_buffer;" nl "        }" nl "        const result = self.buffer[align_offset..new_offset];" nl "        self.offset = new_offset;" nl "        return result.ptr;" nl "    }" nl nl "    fn resize(ctx: *anyopaque, buf: []u8, buf_align: u8, new_len: usize, ret_addr: usize) bool {" nl "        _ = ctx;" nl "        _ = buf;" nl "        _ = buf_align;" nl "        _ = new_len;" nl "        _ = ret_addr;" nl "        return false;" nl "    }" nl nl "    fn free(ctx: *anyopaque, buf: []u8, buf_align: u8, ret_addr: usize) void {" nl "        _ = ctx;" nl "        _ = buf;" nl "        _ = buf_align;" nl "        _ = ret_addr;" nl "    }" nl "};"
nl ::= "\n"
''',
            class_definitions=[
                ClassDefinition(
                    name="ArenaAllocator",
                    instance_vars=(
                        TypeBinding("child_allocator", "std.mem.Allocator"),
                        TypeBinding("buffer", "[]u8"),
                        TypeBinding("offset", "usize"),
                    ),
                    methods=(
                        FunctionSignature(
                            name="init",
                            params=(
                                TypeBinding(
                                    "child", "std.mem.Allocator", scope="parameter"
                                ),
                            ),
                            return_type="ArenaAllocator",
                        ),
                        FunctionSignature(
                            name="allocator",
                            params=(
                                TypeBinding(
                                    "self", "*ArenaAllocator", scope="parameter"
                                ),
                            ),
                            return_type="std.mem.Allocator",
                        ),
                        FunctionSignature(
                            name="alloc",
                            params=(
                                TypeBinding("ctx", "*anyopaque", scope="parameter"),
                                TypeBinding("len", "usize", scope="parameter"),
                                TypeBinding("ptr_align", "u8", scope="parameter"),
                                TypeBinding("ret_addr", "usize", scope="parameter"),
                            ),
                            return_type="?[*]u8",
                        ),
                    ),
                ),
            ],
            semantic_constraints=[
                SemanticConstraint(
                    kind="assertion",
                    expression="allocator returns std.mem.Allocator vtable",
                    scope="ArenaAllocator",
                    variables=(),
                ),
                SemanticConstraint(
                    kind="invariant",
                    expression="offset <= buffer.len",
                    scope="ArenaAllocator",
                    variables=("offset", "buffer"),
                ),
            ],
            domain_configs={
                "comptime": {
                    "require_allocator_interface": True,
                    "validate_vtable": True,
                }
            },
            cache_scope="full_context",
        ),
        expected_effect="Masks allocator implementations not conforming to std.mem.Allocator interface",
        valid_outputs=[
            """const ArenaAllocator = struct {
    child_allocator: std.mem.Allocator,
    buffer: []u8,
    offset: usize,

    const Self = @This();

    pub fn init(child: std.mem.Allocator) Self {
        return Self{
            .child_allocator = child,
            .buffer = &[_]u8{},
            .offset = 0,
        };
    }

    pub fn allocator(self: *Self) std.mem.Allocator {
        return .{
            .ptr = self,
            .vtable = &.{
                .alloc = alloc,
                .resize = resize,
                .free = free,
            },
        };
    }

    fn alloc(ctx: *anyopaque, len: usize, ptr_align: u8, ret_addr: usize) ?[*]u8 {
        const self: *Self = @ptrCast(@alignCast(ctx));
        _ = ret_addr;
        const align_offset = std.mem.alignForward(usize, self.offset, ptr_align);
        const new_offset = align_offset + len;
        if (new_offset > self.buffer.len) {
            const new_buffer = self.child_allocator.alloc(u8, new_offset * 2) catch return null;
            @memcpy(new_buffer[0..self.offset], self.buffer[0..self.offset]);
            if (self.buffer.len > 0) {
                self.child_allocator.free(self.buffer);
            }
            self.buffer = new_buffer;
        }
        const result = self.buffer[align_offset..new_offset];
        self.offset = new_offset;
        return result.ptr;
    }

    fn resize(ctx: *anyopaque, buf: []u8, buf_align: u8, new_len: usize, ret_addr: usize) bool {
        _ = ctx;
        _ = buf;
        _ = buf_align;
        _ = new_len;
        _ = ret_addr;
        return false;
    }

    fn free(ctx: *anyopaque, buf: []u8, buf_align: u8, ret_addr: usize) void {
        _ = ctx;
        _ = buf;
        _ = buf_align;
        _ = ret_addr;
    }
};""",
        ],
        invalid_outputs=[
            "const ArenaAllocator = struct {\n    pub fn alloc(self: *Self, size: usize) ![]u8 { ... }\n};",  # Wrong signature
            "const ArenaAllocator = struct {\n    pub fn allocator(self: *Self) Allocator { ... }\n};",  # Wrong return type
            "const ArenaAllocator = struct {\n    // Missing vtable implementation\n};",
        ],
        tags=["comptime", "allocator", "interface", "vtable", "memory-management"],
        language="zig",
        domain="comptime",
        max_tokens=8192,  # Allocator with vtable is very verbose, needs extra room
    ),
    ConstraintExample(
        id="zig-comptime-003",
        name="Error Set Inference and Composition",
        description="Compose and infer error sets across function boundaries",
        scenario="Developer building error handling hierarchy with automatic error set inference",
        prompt="""Write functions with error sets - either explicit or inferred. Explicit: define
const MyError = error{...} then return MyError!T. Inferred: use !T return type and let
Zig compose the error set from all try/catch operations in the function body.

""",
        spec=ConstraintSpec(
            language="zig",
            # Regex enforces error set definition or error union inference
            regex=r"^(?:const\s+\w+Error\s*=\s*error\s*\{|fn\s+\w+\s*\([^)]*\)\s*!)",
            ebnf=r'''
root ::= explicit_error_set | union_error_set | inferred_errors
explicit_error_set ::= "const ConfigError = error{" nl "    FileNotFound," nl "    ParseError," nl "    InvalidFormat," nl "};" nl nl "const Config = struct {" nl "    name: []const u8," nl "    port: u16," nl "};" nl nl "fn parseConfig(data: []const u8) ConfigError!Config {" nl "    if (data.len == 0) return error.InvalidFormat;" nl "    // Parse logic" nl "    return Config{ .name = \"default\", .port = 8080 };" nl "}" nl nl "fn readConfig(path: []const u8) !Config {" nl "    const file = std.fs.cwd().openFile(path, .{}) catch return error.FileNotFound;" nl "    defer file.close();" nl nl "    const data = file.readToEndAlloc(std.heap.page_allocator, 1024 * 1024) catch return error.ParseError;" nl "    defer std.heap.page_allocator.free(data);" nl nl "    return try parseConfig(data);" nl "}"
union_error_set ::= "const ParseError = error{ InvalidFormat, MissingField };" nl "const IOError = error{ FileNotFound, ReadError };" nl nl "fn parseConfig(data: []const u8) ParseError!Config {" nl "    if (data.len == 0) return error.InvalidFormat;" nl "    return Config{ .name = \"config\", .port = 3000 };" nl "}" nl nl "fn readConfig(path: []const u8) (IOError || ParseError)!Config {" nl "    const file = std.fs.cwd().openFile(path, .{}) catch return error.FileNotFound;" nl "    defer file.close();" nl "    const contents = file.readToEndAlloc(allocator, 1024) catch return error.ReadError;" nl "    defer allocator.free(contents);" nl "    return try parseConfig(contents);" nl "}"
inferred_errors ::= "fn parseConfig(data: []const u8) !Config {" nl "    // Inferred error set from std.fmt.parseInt and other operations" nl "    const port = try std.fmt.parseInt(u16, data, 10);" nl "    return Config{ .name = \"app\", .port = port };" nl "}" nl nl "fn readConfig(path: []const u8) !Config {" nl "    // Inferred error set includes parseConfig errors + file errors" nl "    const file = try std.fs.cwd().openFile(path, .{});" nl "    defer file.close();" nl "    const data = try file.readToEndAlloc(allocator, 4096);" nl "    defer allocator.free(data);" nl "    return try parseConfig(data);" nl "}"
nl ::= "\n"
''',
            function_signatures=[
                FunctionSignature(
                    name="readConfig",
                    params=(TypeBinding("path", "[]const u8", scope="parameter"),),
                    return_type="!Config",
                ),
                FunctionSignature(
                    name="parseConfig",
                    params=(TypeBinding("data", "[]const u8", scope="parameter"),),
                    return_type="!Config",
                ),
            ],
            type_bindings=[
                TypeBinding(
                    name="ConfigError",
                    type_expr="error{ FileNotFound, ParseError, InvalidFormat }",
                    scope="module",
                ),
            ],
            semantic_constraints=[
                SemanticConstraint(
                    kind="invariant",
                    expression="readConfig error set ⊇ parseConfig error set",
                    scope="module",
                    variables=(),
                ),
                SemanticConstraint(
                    kind="assertion",
                    expression="error set inferred from called functions",
                    scope="readConfig",
                    variables=(),
                ),
            ],
            domain_configs={
                "comptime": {
                    "infer_error_sets": True,
                    "compose_error_unions": True,
                    "validate_error_coverage": True,
                }
            },
            cache_scope="full_context",
        ),
        expected_effect="Masks explicit error sets where inference should apply; validates error set composition",
        valid_outputs=[
            """const ConfigError = error{
    FileNotFound,
    ParseError,
    InvalidFormat,
};

const Config = struct {
    name: []const u8,
    port: u16,
};

fn parseConfig(data: []const u8) ConfigError!Config {
    if (data.len == 0) return error.InvalidFormat;
    // Parse logic
    return Config{ .name = "default", .port = 8080 };
}

fn readConfig(path: []const u8) !Config {
    const file = std.fs.cwd().openFile(path, .{}) catch return error.FileNotFound;
    defer file.close();

    const data = file.readToEndAlloc(std.heap.page_allocator, 1024 * 1024) catch return error.ParseError;
    defer std.heap.page_allocator.free(data);

    return try parseConfig(data);
}""",
            """const ParseError = error{ InvalidFormat, MissingField };
const IOError = error{ FileNotFound, ReadError };

fn parseConfig(data: []const u8) ParseError!Config {
    if (data.len == 0) return error.InvalidFormat;
    return Config{ .name = "config", .port = 3000 };
}

fn readConfig(path: []const u8) (IOError || ParseError)!Config {
    const file = std.fs.cwd().openFile(path, .{}) catch return error.FileNotFound;
    defer file.close();
    const contents = file.readToEndAlloc(allocator, 1024) catch return error.ReadError;
    defer allocator.free(contents);
    return try parseConfig(contents);
}""",
            """fn parseConfig(data: []const u8) !Config {
    // Inferred error set from std.fmt.parseInt and other operations
    const port = try std.fmt.parseInt(u16, data, 10);
    return Config{ .name = "app", .port = port };
}

fn readConfig(path: []const u8) !Config {
    // Inferred error set includes parseConfig errors + file errors
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();
    const data = try file.readToEndAlloc(allocator, 4096);
    defer allocator.free(data);
    return try parseConfig(data);
}""",
        ],
        invalid_outputs=[
            "fn readConfig(path: []const u8) ConfigError!Config {\n    return try parseConfig(data);\n}",  # Too narrow: missing file I/O errors
            "fn parseConfig(data: []const u8) anyerror!Config { ... }",  # Too broad: anyerror
            "fn readConfig(path: []const u8) error{ParseError}!Config {\n    const file = try std.fs.cwd().openFile(path, .{});\n}",  # Missing FileNotFound
        ],
        tags=["comptime", "error-sets", "error-handling", "inference", "composition"],
        language="zig",
        domain="comptime",
    ),
]


__all__ = ["ZIG_COMPTIME_EXAMPLES"]
