# Copyright Rand Arete @ Ananke 2025
# Licensed under the Apache License, Version 2.0
"""Zig import constraint examples.

Demonstrates import/module constraints specific to Zig:
- std module restrictions (e.g., forbid std.os for WASM target)
- Build-time feature flags via @import("build_options")
- Allocator requirements (must use provided allocator, not global)
"""

from __future__ import annotations

from typing import List

try:
    from ..base import ConstraintExample
    from .....spec.constraint_spec import (
        ConstraintSpec,
        ImportBinding,
        TypeBinding,
        SemanticConstraint,
    )
except ImportError:
    from tests.fixtures.constraints.base import ConstraintExample
    from spec.constraint_spec import (
        ConstraintSpec,
        ImportBinding,
        TypeBinding,
        SemanticConstraint,
    )


ZIG_IMPORT_EXAMPLES: List[ConstraintExample] = [
    ConstraintExample(
        id="zig-imports-001",
        name="WASM Target std.os Restriction",
        description="Prevent use of std.os when targeting WebAssembly",
        scenario="Developer writing cross-platform code that must work in WASM environment",
        prompt="""Complete this Zig code that uses only WASM-compatible std modules (no std.os, std.fs, std.net):

const std = @import("std");
const mem = std.mem;

pub fn process(data: []const u8) void {
    """,
        spec=ConstraintSpec(
            language="zig",
            # Regex enforces using mem functions in function body (no std.os/fs/net)
            regex=r"^(?:const|var|_\s*=|mem\.|std\.debug)",
            ebnf=r'''
root ::= mem_dupe_pattern | arraylist_pattern
mem_dupe_pattern ::= "const std = @import(\"std\");" nl "const mem = std.mem;" nl nl "pub fn process(data: []const u8) void {" nl "    const copy = mem.dupe(std.heap.page_allocator, u8, data) catch unreachable;" nl "    defer std.heap.page_allocator.free(copy);" nl "}"
arraylist_pattern ::= "const std = @import(\"std\");" nl "const ArrayList = std.ArrayList;" nl nl "pub fn buildList(allocator: std.mem.Allocator) !ArrayList(u32) {" nl "    var list = ArrayList(u32).init(allocator);" nl "    try list.append(42);" nl "    return list;" nl "}"
nl ::= "\n"
''',
            forbidden_imports={"std.os", "std.fs", "std.net"},
            imports=[
                ImportBinding(module="std"),
                ImportBinding(module="std.mem"),
                ImportBinding(module="std.debug"),
            ],
            available_modules={
                "std",
                "std.mem",
                "std.debug",
                "std.math",
                "std.ArrayList",
                "std.HashMap",
            },
            semantic_constraints=[
                SemanticConstraint(
                    kind="invariant",
                    expression="!@hasDecl(@import(\"std\"), \"os\")",
                    scope="module",
                    variables=(),
                ),
            ],
            domain_configs={
                "import": {
                    "target": "wasm32-freestanding",
                    "strict_platform_check": True,
                }
            },
            cache_scope="full_context",
        ),
        expected_effect="Masks tokens importing or using std.os, std.fs, std.net; allows only WASM-compatible std modules",
        valid_outputs=[
            "const std = @import(\"std\");\nconst mem = std.mem;\n\npub fn process(data: []const u8) void {\n    const copy = mem.dupe(std.heap.page_allocator, u8, data) catch unreachable;\n    defer std.heap.page_allocator.free(copy);\n}",
            "const std = @import(\"std\");\nconst ArrayList = std.ArrayList;\n\npub fn buildList(allocator: std.mem.Allocator) !ArrayList(u32) {\n    var list = ArrayList(u32).init(allocator);\n    try list.append(42);\n    return list;\n}",
        ],
        invalid_outputs=[
            "const std = @import(\"std\");\nconst os = std.os;\n...",  # Forbidden: std.os
            "const fs = @import(\"std\").fs;\npub fn readFile(path: []const u8) ![]u8 { ... }",  # Forbidden: std.fs
            "const net = @import(\"std\").net;",  # Forbidden: std.net
        ],
        tags=["imports", "platform", "wasm", "constraints"],
        language="zig",
        domain="imports",
    ),
    ConstraintExample(
        id="zig-imports-002",
        name="Build-time Feature Flags",
        description="Use @import(\"build_options\") for compile-time configuration",
        scenario="Developer using build-time feature flags to enable/disable functionality",
        prompt="""Write Zig code that uses build-time feature flags. Import build_options with
@import("build_options") and check flags like enable_logging or enable_debug.
These are set in build.zig and evaluated at comptime, not runtime.

""",
        spec=ConstraintSpec(
            language="zig",
            # Regex enforces @import("build_options") pattern
            regex=r'^const\s+(?:build_options|options)\s*=\s*@import\s*\(\s*"build_options"\s*\)',
            ebnf=r'''
root ::= logging_pattern | debug_pattern | feature_pattern
logging_pattern ::= "const build_options = @import(\"build_options\");" nl "const std = @import(\"std\");" nl nl "pub fn log(msg: []const u8) void {" nl "    if (build_options.enable_logging) {" nl "        std.debug.print(\"{s}\\n\", .{msg});" nl "    }" nl "}"
debug_pattern ::= "const options = @import(\"build_options\");" nl nl "pub fn process() void {" nl "    if (comptime options.enable_debug) {" nl "        @compileLog(\"Debug mode enabled\");" nl "    }" nl "}"
feature_pattern ::= "const build_options = @import(\"build_options\");" nl "const Feature = enum { basic, advanced };" nl nl "pub fn getFeatureLevel() Feature {" nl "    return if (build_options.advanced_features) .advanced else .basic;" nl "}"
nl ::= "\n"
''',
            imports=[
                ImportBinding(module="std"),
                ImportBinding(module="build_options"),
            ],
            type_bindings=[
                TypeBinding(
                    name="build_options",
                    type_expr="type",
                    scope="module",
                    mutable=False,
                ),
                TypeBinding(
                    name="enable_logging",
                    type_expr="bool",
                    scope="module",
                    mutable=False,
                ),
            ],
            semantic_constraints=[
                SemanticConstraint(
                    kind="assertion",
                    expression="@typeInfo(@TypeOf(build_options.enable_logging)) == .Bool",
                    scope="module",
                    variables=("build_options",),
                ),
            ],
            cache_scope="full_context",
        ),
        expected_effect="Masks tokens not using build_options correctly; requires comptime feature flag checks",
        valid_outputs=[
            'const build_options = @import("build_options");\nconst std = @import("std");\n\npub fn log(msg: []const u8) void {\n    if (build_options.enable_logging) {\n        std.debug.print("{s}\\n", .{msg});\n    }\n}',
            'const options = @import("build_options");\n\npub fn process() void {\n    if (comptime options.enable_debug) {\n        @compileLog("Debug mode enabled");\n    }\n}',
            'const build_options = @import("build_options");\nconst Feature = enum { basic, advanced };\n\npub fn getFeatureLevel() Feature {\n    return if (build_options.advanced_features) .advanced else .basic;\n}',
        ],
        invalid_outputs=[
            'const build_options = @import("build_options");\nvar enable_logging = true;\n...',  # Runtime mutation of build constant
            'pub fn log(msg: []const u8) void {\n    if (std.os.getenv("ENABLE_LOGGING")) { ... }\n}',  # Runtime not build-time
            'const opts = @import("build_options");\npub fn process() void { opts.enable_logging = false; }',  # Can't mutate
        ],
        tags=["imports", "comptime", "build-options", "feature-flags"],
        language="zig",
        domain="imports",
    ),
    ConstraintExample(
        id="zig-imports-003",
        name="Explicit Allocator Requirement",
        description="Functions must accept allocator parameter, not use global allocators",
        scenario="Developer writing library code that requires explicit allocator passing",
        prompt="""Write a Zig function that takes an allocator parameter explicitly. Library code
should never use std.heap.page_allocator or c_allocator directly - always accept
allocator: std.mem.Allocator as a parameter to let callers control allocation.

""",
        spec=ConstraintSpec(
            language="zig",
            # Regex enforces explicit allocator parameter in function signatures
            regex=r"^(?:pub\s+)?fn\s+\w+\s*\(\s*allocator\s*:\s*(?:std\.mem\.)?Allocator",
            ebnf=r'''
root ::= create_buffer | duplicate | process_data
create_buffer ::= "const std = @import(\"std\");" nl "const Allocator = std.mem.Allocator;" nl nl "pub fn createBuffer(allocator: Allocator, size: usize) ![]u8 {" nl "    return try allocator.alloc(u8, size);" nl "}"
duplicate ::= "pub fn duplicate(allocator: std.mem.Allocator, data: []const u8) ![]u8 {" nl "    const copy = try allocator.alloc(u8, data.len);" nl "    @memcpy(copy, data);" nl "    return copy;" nl "}"
process_data ::= "const std = @import(\"std\");" nl nl "pub fn processData(allocator: std.mem.Allocator, items: []const u32) !std.ArrayList(u32) {" nl "    var list = std.ArrayList(u32).init(allocator);" nl "    for (items) |item| { try list.append(item * 2); }" nl "    return list;" nl "}"
nl ::= "\n"
''',
            imports=[
                ImportBinding(module="std"),
                ImportBinding(module="std.mem", name="Allocator"),
            ],
            type_bindings=[
                TypeBinding(
                    name="allocator",
                    type_expr="std.mem.Allocator",
                    scope="parameter",
                    mutable=False,
                ),
            ],
            forbidden_imports={"std.heap.page_allocator", "std.heap.c_allocator"},
            semantic_constraints=[
                SemanticConstraint(
                    kind="precondition",
                    expression="@TypeOf(allocator) == std.mem.Allocator",
                    scope="function",
                    variables=("allocator",),
                ),
            ],
            domain_configs={
                "semantic": {
                    "forbid_globals": ["std.heap.page_allocator", "std.heap.c_allocator"],
                    "require_explicit_allocator": True,
                }
            },
            cache_scope="full_context",
        ),
        expected_effect="Masks tokens using global allocators; requires allocator parameter in function signatures",
        valid_outputs=[
            "const std = @import(\"std\");\nconst Allocator = std.mem.Allocator;\n\npub fn createBuffer(allocator: Allocator, size: usize) ![]u8 {\n    return try allocator.alloc(u8, size);\n}",
            "pub fn duplicate(allocator: std.mem.Allocator, data: []const u8) ![]u8 {\n    const copy = try allocator.alloc(u8, data.len);\n    @memcpy(copy, data);\n    return copy;\n}",
            "const std = @import(\"std\");\n\npub fn processData(allocator: std.mem.Allocator, items: []const u32) !std.ArrayList(u32) {\n    var list = std.ArrayList(u32).init(allocator);\n    for (items) |item| { try list.append(item * 2); }\n    return list;\n}",
        ],
        invalid_outputs=[
            "pub fn createBuffer(size: usize) ![]u8 {\n    return try std.heap.page_allocator.alloc(u8, size);\n}",  # Forbidden global allocator
            "pub fn duplicate(data: []const u8) ![]u8 {\n    const allocator = std.heap.c_allocator;\n    return try allocator.alloc(u8, data.len);\n}",  # Forbidden c_allocator
            "var gpa = std.heap.GeneralPurposeAllocator(.{}){};\npub fn createBuffer(size: usize) ![]u8 {\n    return try gpa.allocator().alloc(u8, size);\n}",  # Global state
        ],
        tags=["imports", "allocator", "memory-management", "best-practices"],
        language="zig",
        domain="imports",
    ),
]


__all__ = ["ZIG_IMPORT_EXAMPLES"]
