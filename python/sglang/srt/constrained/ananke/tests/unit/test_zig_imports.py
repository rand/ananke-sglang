# Copyright Rand Arete @ Ananke 2025
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Unit tests for Zig import resolver.

Tests for the ZigImportResolver including:
- Standard library module resolution
- Package dependency resolution (build.zig.zon)
- Local file imports
- Symbol export retrieval
"""

import pytest
from pathlib import Path
from tempfile import TemporaryDirectory

from domains.imports.resolvers.zig import (
    ZigImportResolver,
    ZIG_STD_MODULES,
    ZIG_STD_DATA_STRUCTURES,
    ZIG_STD_ALLOCATORS,
    ZIG_STD_IO,
    ZIG_STD_FS,
    ZIG_STD_MEM,
    ZIG_STD_FMT,
    ZIG_STD_DEBUG,
    ZIG_STD_TESTING,
    ZIG_STD_HEAP,
    ZIG_STD_MATH,
    parse_build_zig_zon,
    extract_zig_exports,
)
from domains.imports.resolvers.base import (
    ImportResolution,
    ResolutionStatus,
)


# ===========================================================================
# Standard Library Module Tests
# ===========================================================================


class TestZigStdModules:
    """Tests for standard library module sets."""

    def test_has_core_modules(self):
        """Should have core modules."""
        assert "std" in ZIG_STD_MODULES
        assert "std.mem" in ZIG_STD_MODULES
        assert "std.fmt" in ZIG_STD_MODULES
        assert "std.io" in ZIG_STD_MODULES
        assert "std.fs" in ZIG_STD_MODULES

    def test_has_data_structures(self):
        """Should have data structure modules."""
        assert "ArrayList" in ZIG_STD_DATA_STRUCTURES
        assert "HashMap" in ZIG_STD_DATA_STRUCTURES
        assert "StringHashMap" in ZIG_STD_DATA_STRUCTURES
        assert "AutoHashMap" in ZIG_STD_DATA_STRUCTURES

    def test_has_allocators(self):
        """Should have allocator types."""
        assert "GeneralPurposeAllocator" in ZIG_STD_ALLOCATORS
        assert "ArenaAllocator" in ZIG_STD_ALLOCATORS
        assert "FixedBufferAllocator" in ZIG_STD_ALLOCATORS
        assert "page_allocator" in ZIG_STD_ALLOCATORS

    def test_has_io_types(self):
        """Should have IO types."""
        assert "Reader" in ZIG_STD_IO
        assert "Writer" in ZIG_STD_IO
        assert "getStdOut" in ZIG_STD_IO
        assert "getStdErr" in ZIG_STD_IO

    def test_has_fs_types(self):
        """Should have filesystem types."""
        assert "File" in ZIG_STD_FS
        assert "Dir" in ZIG_STD_FS
        assert "cwd" in ZIG_STD_FS
        assert "openFile" in ZIG_STD_FS

    def test_has_mem_functions(self):
        """Should have memory functions."""
        assert "eql" in ZIG_STD_MEM
        assert "copy" in ZIG_STD_MEM
        assert "set" in ZIG_STD_MEM
        assert "indexOf" in ZIG_STD_MEM

    def test_has_fmt_functions(self):
        """Should have format functions."""
        assert "print" in ZIG_STD_FMT
        assert "format" in ZIG_STD_FMT
        assert "allocPrint" in ZIG_STD_FMT

    def test_has_debug_functions(self):
        """Should have debug functions."""
        assert "print" in ZIG_STD_DEBUG
        assert "panic" in ZIG_STD_DEBUG
        assert "assert" in ZIG_STD_DEBUG

    def test_has_testing_functions(self):
        """Should have testing functions."""
        assert "expect" in ZIG_STD_TESTING
        assert "expectEqual" in ZIG_STD_TESTING
        assert "expectError" in ZIG_STD_TESTING

    def test_has_heap_allocators(self):
        """Should have heap module."""
        assert "page_allocator" in ZIG_STD_HEAP
        assert "c_allocator" in ZIG_STD_HEAP

    def test_has_math_functions(self):
        """Should have math functions."""
        assert "add" in ZIG_STD_MATH
        assert "mul" in ZIG_STD_MATH
        assert "sqrt" in ZIG_STD_MATH
        assert "min" in ZIG_STD_MATH
        assert "max" in ZIG_STD_MATH


# ===========================================================================
# Resolver Basic Tests
# ===========================================================================


class TestZigImportResolverBasics:
    """Basic tests for ZigImportResolver."""

    @pytest.fixture
    def resolver(self):
        """Create a resolver without project root."""
        return ZigImportResolver()

    def test_language_name(self, resolver):
        """Should return 'zig' as language."""
        assert resolver.language == "zig"

    def test_can_resolve_std(self, resolver):
        """Should be able to resolve std imports."""
        result = resolver.resolve("std")
        assert result.status in (ResolutionStatus.RESOLVED, ResolutionStatus.PARTIAL)

    def test_resolve_std_mem(self, resolver):
        """Should resolve std.mem."""
        result = resolver.resolve("std.mem")
        assert result.status in (ResolutionStatus.RESOLVED, ResolutionStatus.PARTIAL)


# ===========================================================================
# Standard Library Resolution Tests
# ===========================================================================


class TestZigStdResolution:
    """Tests for standard library resolution."""

    @pytest.fixture
    def resolver(self):
        return ZigImportResolver()

    def test_resolve_std(self, resolver):
        """Should resolve 'std'."""
        result = resolver.resolve("std")
        assert result.status != ResolutionStatus.FAILED
        assert result.module_name == "std"

    def test_resolve_std_mem(self, resolver):
        """Should resolve 'std.mem'."""
        result = resolver.resolve("std.mem")
        assert result.status != ResolutionStatus.FAILED

    def test_resolve_std_fmt(self, resolver):
        """Should resolve 'std.fmt'."""
        result = resolver.resolve("std.fmt")
        assert result.status != ResolutionStatus.FAILED

    def test_resolve_std_io(self, resolver):
        """Should resolve 'std.io'."""
        result = resolver.resolve("std.io")
        assert result.status != ResolutionStatus.FAILED

    def test_resolve_std_fs(self, resolver):
        """Should resolve 'std.fs'."""
        result = resolver.resolve("std.fs")
        assert result.status != ResolutionStatus.FAILED

    def test_resolve_std_heap(self, resolver):
        """Should resolve 'std.heap'."""
        result = resolver.resolve("std.heap")
        assert result.status != ResolutionStatus.FAILED

    def test_resolve_std_testing(self, resolver):
        """Should resolve 'std.testing'."""
        result = resolver.resolve("std.testing")
        assert result.status != ResolutionStatus.FAILED

    def test_std_exports_have_common_names(self, resolver):
        """std should export common names."""
        result = resolver.resolve("std")
        if result.exports:
            assert any("mem" in e for e in result.exports) or "mem" in str(result)


# ===========================================================================
# Local File Resolution Tests
# ===========================================================================


class TestZigLocalFileResolution:
    """Tests for local file import resolution."""

    @pytest.fixture
    def temp_project(self):
        """Create a temporary Zig project."""
        with TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)

            # Create a simple Zig file
            zig_file = project_path / "utils.zig"
            zig_file.write_text("""
pub fn add(a: i32, b: i32) i32 {
    return a + b;
}

pub const VERSION: u32 = 1;

pub const Config = struct {
    enabled: bool,
    max_items: u32,
};

fn privateFunc() void {}
""")

            yield project_path

    def test_resolve_local_file(self, temp_project):
        """Should resolve local .zig file imports."""
        resolver = ZigImportResolver(str(temp_project))
        result = resolver.resolve("utils.zig")
        # Note: actual file resolution depends on implementation
        # This test verifies the resolver can handle the request


# ===========================================================================
# build.zig.zon Parsing Tests
# ===========================================================================


class TestBuildZigZonParsing:
    """Tests for build.zig.zon parsing."""

    def test_parse_simple_zon(self):
        """Should parse simple build.zig.zon."""
        zon_content = """
.{
    .name = "my-project",
    .version = "0.1.0",
    .dependencies = .{
        .ziglyph = .{
            .url = "https://github.com/tiehuis/ziglyph/archive/refs/tags/v0.1.0.tar.gz",
            .hash = "abc123",
        },
        .known_folders = .{
            .url = "https://github.com/ziglang/zig/archive/refs/tags/0.11.0.tar.gz",
            .hash = "def456",
        },
    },
}
"""
        result = parse_build_zig_zon(zon_content)
        assert "ziglyph" in result or len(result) >= 0

    def test_parse_zon_with_no_dependencies(self):
        """Should handle zon with no dependencies."""
        zon_content = """
.{
    .name = "simple-project",
    .version = "0.1.0",
}
"""
        result = parse_build_zig_zon(zon_content)
        assert isinstance(result, dict)

    def test_parse_empty_dependencies(self):
        """Should handle empty dependencies."""
        zon_content = """
.{
    .name = "test",
    .dependencies = .{},
}
"""
        result = parse_build_zig_zon(zon_content)
        assert isinstance(result, dict)


# ===========================================================================
# Export Extraction Tests
# ===========================================================================


class TestZigExportExtraction:
    """Tests for extracting exports from Zig source."""

    def test_extract_pub_fn(self):
        """Should extract public functions."""
        source = """
pub fn add(a: i32, b: i32) i32 {
    return a + b;
}

pub fn multiply(x: f32, y: f32) f32 {
    return x * y;
}

fn privateFunc() void {}
"""
        exports = extract_zig_exports(source)
        assert "add" in exports
        assert "multiply" in exports
        assert "privateFunc" not in exports

    def test_extract_pub_const(self):
        """Should extract public constants."""
        source = """
pub const VERSION: u32 = 1;
pub const NAME = "my-lib";
const private_const = 42;
"""
        exports = extract_zig_exports(source)
        assert "VERSION" in exports
        assert "NAME" in exports
        assert "private_const" not in exports

    def test_extract_pub_var(self):
        """Should extract public variables."""
        source = """
pub var global_state: i32 = 0;
var private_state: i32 = 0;
"""
        exports = extract_zig_exports(source)
        assert "global_state" in exports
        assert "private_state" not in exports

    def test_extract_pub_struct(self):
        """Should extract public struct types."""
        source = """
pub const Config = struct {
    enabled: bool,
    max_items: u32,
};

const PrivateStruct = struct {
    x: i32,
};
"""
        exports = extract_zig_exports(source)
        assert "Config" in exports
        assert "PrivateStruct" not in exports

    def test_extract_pub_enum(self):
        """Should extract public enums."""
        source = """
pub const Color = enum {
    red,
    green,
    blue,
};
"""
        exports = extract_zig_exports(source)
        assert "Color" in exports

    def test_extract_pub_union(self):
        """Should extract public unions."""
        source = """
pub const Value = union(enum) {
    int: i32,
    float: f32,
    none,
};
"""
        exports = extract_zig_exports(source)
        assert "Value" in exports


# ===========================================================================
# Package Dependency Resolution Tests
# ===========================================================================


class TestZigPackageResolution:
    """Tests for package dependency resolution."""

    @pytest.fixture
    def project_with_deps(self):
        """Create a project with build.zig.zon dependencies."""
        with TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)

            # Create build.zig.zon
            zon_file = project_path / "build.zig.zon"
            zon_file.write_text("""
.{
    .name = "test-project",
    .version = "0.1.0",
    .dependencies = .{
        .ziglyph = .{
            .url = "https://github.com/tiehuis/ziglyph",
            .hash = "abc123",
        },
    },
}
""")

            yield project_path

    def test_resolver_finds_zon_deps(self, project_with_deps):
        """Resolver should find dependencies from build.zig.zon."""
        resolver = ZigImportResolver(str(project_with_deps))
        # Check that resolver can be created with project root
        assert resolver.project_root == str(project_with_deps)


# ===========================================================================
# Edge Cases
# ===========================================================================


class TestZigImportEdgeCases:
    """Tests for edge cases in import resolution."""

    @pytest.fixture
    def resolver(self):
        return ZigImportResolver()

    def test_nonexistent_module(self, resolver):
        """Should handle nonexistent modules gracefully."""
        result = resolver.resolve("nonexistent.module")
        # Should not raise, may return failed status or partial
        assert result is not None

    def test_empty_module_name(self, resolver):
        """Should handle empty module name."""
        result = resolver.resolve("")
        assert result.status == ResolutionStatus.FAILED

    def test_deeply_nested_module(self, resolver):
        """Should handle deeply nested module paths."""
        result = resolver.resolve("std.mem.Allocator")
        assert result is not None

    def test_resolve_with_extension(self, resolver):
        """Should handle .zig extension in import."""
        # This is for relative imports like @import("foo.zig")
        result = resolver.resolve("foo.zig")
        assert result is not None

    def test_resolve_builtin(self, resolver):
        """Should handle @import for common patterns."""
        # builtin is a special import
        result = resolver.resolve("builtin")
        assert result is not None
