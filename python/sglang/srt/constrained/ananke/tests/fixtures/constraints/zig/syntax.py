# Copyright Rand Arete @ Ananke 2025
# Licensed under the Apache License, Version 2.0
"""Zig syntax constraint examples.

Demonstrates syntax-level constraints specific to Zig:
- Build.zig configuration schema
- Identifier patterns (snake_case convention)
- Comptime format string validation
"""

from __future__ import annotations

from typing import List

try:
    from ..base import ConstraintExample
    from .....spec.constraint_spec import (
        ConstraintSpec,
        TypeBinding,
        SemanticConstraint,
    )
except ImportError:
    from tests.fixtures.constraints.base import ConstraintExample
    from spec.constraint_spec import (
        ConstraintSpec,
        TypeBinding,
        SemanticConstraint,
    )


ZIG_SYNTAX_EXAMPLES: List[ConstraintExample] = [
    ConstraintExample(
        id="zig-syntax-001",
        name="Build.zig Configuration Schema",
        description="Enforce valid Build.zig structure with proper std.Build API usage",
        scenario="Developer creating build configuration following Zig build system conventions",
        prompt="""Write a build.zig file for a Zig project. Must start with @import("std"),
then define pub fn build(b: *std.Build) void. Use b.standardTargetOptions(.{})
and b.standardOptimizeOption(.{}) for cross-platform builds.

""",
        spec=ConstraintSpec(
            language="zig",
            structural_tag="build_zig_file",
            # Regex enforces pub fn build(b: *std.Build) void signature
            regex=r'^const\s+std\s*=\s*@import\s*\(\s*"std"\s*\)[\s\S]*pub\s+fn\s+build\s*\(\s*b\s*:\s*\*std\.Build\s*\)\s*void\s*\{',
            ebnf=r'''
root ::= exe_build | lib_with_tests
exe_build ::= "const std = @import(\"std\");" nl nl "pub fn build(b: *std.Build) void {" nl "    const target = b.standardTargetOptions(.{});" nl "    const optimize = b.standardOptimizeOption(.{});" nl "    const exe = b.addExecutable(.{" nl "        .name = \"myapp\"," nl "        .root_source_file = .{ .path = \"src/main.zig\" }," nl "        .target = target," nl "        .optimize = optimize," nl "    });" nl "    b.installArtifact(exe);" nl "}"
lib_with_tests ::= "const std = @import(\"std\");" nl nl "pub fn build(b: *std.Build) void {" nl "    const lib = b.addStaticLibrary(.{" nl "        .name = \"mylib\"," nl "        .root_source_file = .{ .path = \"src/lib.zig\" }," nl "        .target = b.standardTargetOptions(.{})," nl "        .optimize = b.standardOptimizeOption(.{})," nl "    });" nl "    b.installArtifact(lib);" nl "    const tests = b.addTest(.{" nl "        .root_source_file = .{ .path = \"src/lib.zig\" }," nl "    });" nl "    const test_step = b.step(\"test\", \"Run tests\");" nl "    test_step.dependOn(&b.addRunArtifact(tests).step);" nl "}"
nl ::= "\n"
''',
            type_bindings=[
                TypeBinding(name="b", type_expr="*std.Build", scope="parameter"),
            ],
            semantic_constraints=[
                SemanticConstraint(
                    kind="assertion",
                    expression="function_name == 'build'",
                    scope="module",
                    variables=(),
                ),
                SemanticConstraint(
                    kind="assertion",
                    expression="parameter_type == '*std.Build'",
                    scope="build",
                    variables=("b",),
                ),
            ],
            domain_configs={
                "syntax": {
                    "enforce_build_convention": True,
                    "require_pub_fn": True,
                }
            },
            cache_scope="syntax_and_lang",
        ),
        expected_effect="Masks non-conformant Build.zig syntax; requires pub fn build(b: *std.Build) void signature",
        valid_outputs=[
            'const std = @import("std");\n\npub fn build(b: *std.Build) void {\n    const target = b.standardTargetOptions(.{});\n    const optimize = b.standardOptimizeOption(.{});\n    const exe = b.addExecutable(.{\n        .name = "myapp",\n        .root_source_file = .{ .path = "src/main.zig" },\n        .target = target,\n        .optimize = optimize,\n    });\n    b.installArtifact(exe);\n}',
            'const std = @import("std");\n\npub fn build(b: *std.Build) void {\n    const lib = b.addStaticLibrary(.{\n        .name = "mylib",\n        .root_source_file = .{ .path = "src/lib.zig" },\n        .target = b.standardTargetOptions(.{}),\n        .optimize = b.standardOptimizeOption(.{}),\n    });\n    b.installArtifact(lib);\n    const tests = b.addTest(.{\n        .root_source_file = .{ .path = "src/lib.zig" },\n    });\n    const test_step = b.step("test", "Run tests");\n    test_step.dependOn(&b.addRunArtifact(tests).step);\n}',
        ],
        invalid_outputs=[
            'pub fn main(b: *std.Build) void { ... }',  # Wrong function name
            'fn build(b: *std.Build) void { ... }',  # Missing pub keyword
            'pub fn build(builder: *Builder) void { ... }',  # Wrong type name
            'pub fn build(b: std.Build) void { ... }',  # Missing pointer
        ],
        tags=["syntax", "build-system", "conventions", "structure"],
        language="zig",
        domain="syntax",
    ),
    ConstraintExample(
        id="zig-syntax-002",
        name="Identifier Pattern snake_case",
        description="Enforce snake_case naming convention for Zig identifiers",
        scenario="Developer writing Zig code following community style guidelines",
        prompt="""Write Zig declarations using proper naming conventions. Variables and functions
use snake_case (like max_buffer_size, process_user_input). Types use PascalCase
(like ServerConfig). Zig style guide forbids camelCase for functions/variables.

""",
        spec=ConstraintSpec(
            language="zig",
            # Regex matches snake_case identifiers or PascalCase type names in declarations
            regex=r"(?:const|var|fn|pub\s+fn)\s+[a-zA-Z][a-zA-Z0-9_]*",
            ebnf=r'''
root ::= const_decl | fn_decl | var_decl | pub_fn_decl | struct_decl
const_decl ::= "const max_buffer_size = 1024;"
fn_decl ::= "fn process_user_input(data: []const u8) void { }"
var_decl ::= "var connection_count: u32 = 0;"
pub_fn_decl ::= "pub fn init_database() !void { }"
struct_decl ::= "const ServerConfig = struct { port: u16 };"
''',
            semantic_constraints=[
                SemanticConstraint(
                    kind="invariant",
                    expression="matches(identifier, '^[a-z][a-z0-9_]*$')",
                    scope="module",
                    variables=("identifier",),
                ),
            ],
            domain_configs={
                "syntax": {
                    "naming_convention": "snake_case",
                    "allow_leading_underscore": False,
                    "check_function_names": True,
                    "check_variable_names": True,
                }
            },
            cache_scope="syntax_and_lang",
        ),
        expected_effect="Masks identifiers not in snake_case; enforces lowercase with underscores",
        valid_outputs=[
            "const max_buffer_size = 1024;",
            "fn process_user_input(data: []const u8) void { }",
            "var connection_count: u32 = 0;",
            "pub fn init_database() !void { }",
            "const ServerConfig = struct { port: u16 };",  # Types use PascalCase
        ],
        invalid_outputs=[
            "const maxBufferSize = 1024;",  # camelCase not allowed
            "fn ProcessUserInput(data: []const u8) void { }",  # PascalCase for function
            "var ConnectionCount: u32 = 0;",  # PascalCase for variable
            "const MAX_BUFFER_SIZE = 1024;",  # SCREAMING_SNAKE_CASE for const
        ],
        tags=["syntax", "naming", "conventions", "style"],
        language="zig",
        domain="syntax",
    ),
    ConstraintExample(
        id="zig-syntax-003",
        name="Comptime Format String Validation",
        description="Validate format strings at compile time for std.fmt functions",
        scenario="Developer using std.debug.print with type-safe format strings",
        prompt="""Write std.debug.print statements with proper format strings. Use {} for most types,
{s} for strings, {d:.2} for floats with precision. Args go in .{ arg1, arg2 } tuple.
Number of {} specifiers must match number of args - Zig checks this at comptime.

""",
        spec=ConstraintSpec(
            language="zig",
            # Regex enforces std.debug.print with format string and args (allows const prefix)
            regex=r'std\.debug\.print\s*\(\s*(?:"[^"]*\{[^}]*\}[^"]*"|\w+)\s*,\s*\.\{',
            ebnf=r'''
root ::= simple_print | multi_args | string_format | const_format | tagname_format
simple_print ::= "std.debug.print(\"Value: {}\\n\", .{42});"
multi_args ::= "std.debug.print(\"x={}, y={}\\n\", .{ x, y });"
string_format ::= "std.debug.print(\"Name: {s}, Age: {}\\n\", .{ name, age });"
const_format ::= "const msg = \"Result: {d:.2}\\n\";" nl "std.debug.print(msg, .{3.14159});"
tagname_format ::= "std.debug.print(\"Status: {s}\\n\", .{@tagName(status)});"
nl ::= "\n"
''',
            type_bindings=[
                TypeBinding(
                    name="format",
                    type_expr="[]const u8",
                    scope="parameter",
                    mutable=False,
                ),
                TypeBinding(name="args", type_expr="anytype", scope="parameter"),
            ],
            semantic_constraints=[
                SemanticConstraint(
                    kind="precondition",
                    expression="is_comptime_literal(format)",
                    scope="print",
                    variables=("format",),
                ),
                SemanticConstraint(
                    kind="assertion",
                    expression="count_specifiers(format) == len(args)",
                    scope="print",
                    variables=("format", "args"),
                ),
            ],
            domain_configs={
                "syntax": {
                    "validate_format_strings": True,
                    "require_comptime_format": True,
                    "check_format_args_match": True,
                }
            },
            cache_scope="full_context",
        ),
        expected_effect="Masks format strings with mismatched placeholders; validates at compile time",
        valid_outputs=[
            'std.debug.print("Value: {}\\n", .{42});',
            'std.debug.print("x={}, y={}\\n", .{ x, y });',
            'std.debug.print("Name: {s}, Age: {}\\n", .{ name, age });',
            'const msg = "Result: {d:.2}\\n";\nstd.debug.print(msg, .{3.14159});',
            'std.debug.print("Status: {s}\\n", .{@tagName(status)});',
        ],
        invalid_outputs=[
            'std.debug.print("Value: {}, {}\\n", .{42});',  # Too few args
            'std.debug.print("Value: {}\\n", .{ 42, 43 });',  # Too many args
            'std.debug.print("Value: {d}\\n", .{"string"});',  # Type mismatch: {d} expects number
            'var fmt = "Value: {}\\n";\nstd.debug.print(fmt, .{42});',  # Non-comptime format string
        ],
        tags=["syntax", "format-strings", "type-safety", "comptime-validation"],
        language="zig",
        domain="syntax",
    ),
]


__all__ = ["ZIG_SYNTAX_EXAMPLES"]
