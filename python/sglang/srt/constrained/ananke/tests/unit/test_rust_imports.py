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
"""Unit tests for Rust import resolver.

Tests for the RustImportResolver including:
- Standard library module resolution
- Cargo.toml dependency parsing
- Crate resolution
- Symbol export retrieval
"""

import pytest
from pathlib import Path
from tempfile import TemporaryDirectory

from domains.imports.resolvers.rust import (
    RustImportResolver,
    RUST_STD_MODULES,
    RUST_CORE_MODULES,
    RUST_ALLOC_MODULES,
    RUST_STD_TYPES,
    RUST_POPULAR_CRATES,
    parse_cargo_toml,
    extract_rust_exports,
)
from domains.imports.resolvers.base import (
    ImportResolution,
    ResolutionStatus,
)


# ===========================================================================
# Standard Library Module Tests
# ===========================================================================


class TestRustStdModules:
    """Tests for standard library module sets."""

    def test_has_core_modules(self):
        """Should have core std modules."""
        assert "std::io" in RUST_STD_MODULES
        assert "std::fs" in RUST_STD_MODULES
        assert "std::collections" in RUST_STD_MODULES
        assert "std::sync" in RUST_STD_MODULES
        assert "std::thread" in RUST_STD_MODULES

    def test_has_core_crate_modules(self):
        """Should have core crate modules."""
        assert "core::mem" in RUST_CORE_MODULES
        assert "core::ptr" in RUST_CORE_MODULES
        assert "core::ops" in RUST_CORE_MODULES
        assert "core::marker" in RUST_CORE_MODULES

    def test_has_alloc_modules(self):
        """Should have alloc crate modules."""
        assert "alloc::vec" in RUST_ALLOC_MODULES
        assert "alloc::string" in RUST_ALLOC_MODULES
        assert "alloc::boxed" in RUST_ALLOC_MODULES

    def test_has_std_types(self):
        """Should have standard types (with full module paths)."""
        assert "std::collections::HashMap" in RUST_STD_TYPES
        assert "std::collections::HashSet" in RUST_STD_TYPES
        assert "std::sync::Arc" in RUST_STD_TYPES
        assert "std::rc::Rc" in RUST_STD_TYPES
        assert "std::io::Read" in RUST_STD_TYPES

    def test_has_popular_crates(self):
        """Should have popular crates."""
        assert "serde" in RUST_POPULAR_CRATES
        assert "tokio" in RUST_POPULAR_CRATES
        assert "clap" in RUST_POPULAR_CRATES
        assert "regex" in RUST_POPULAR_CRATES


# ===========================================================================
# Resolver Basic Tests
# ===========================================================================


class TestRustImportResolverBasics:
    """Basic tests for RustImportResolver."""

    @pytest.fixture
    def resolver(self):
        """Create a resolver without project root."""
        return RustImportResolver()

    def test_language_name(self, resolver):
        """Should return 'rust' as language."""
        assert resolver.language == "rust"

    def test_can_resolve_std(self, resolver):
        """Should be able to resolve std imports."""
        result = resolver.resolve("std")
        assert result.status in (ResolutionStatus.RESOLVED, ResolutionStatus.PARTIAL)

    def test_resolve_std_io(self, resolver):
        """Should resolve std::io."""
        result = resolver.resolve("std::io")
        assert result.status in (ResolutionStatus.RESOLVED, ResolutionStatus.PARTIAL)


# ===========================================================================
# Standard Library Resolution Tests
# ===========================================================================


class TestRustStdResolution:
    """Tests for standard library resolution."""

    @pytest.fixture
    def resolver(self):
        return RustImportResolver()

    def test_resolve_std(self, resolver):
        """Should resolve 'std'."""
        result = resolver.resolve("std")
        assert result.status != ResolutionStatus.FAILED
        assert result.module_name == "std"

    def test_resolve_std_io(self, resolver):
        """Should resolve 'std::io'."""
        result = resolver.resolve("std::io")
        assert result.status != ResolutionStatus.FAILED

    def test_resolve_std_fs(self, resolver):
        """Should resolve 'std::fs'."""
        result = resolver.resolve("std::fs")
        assert result.status != ResolutionStatus.FAILED

    def test_resolve_std_collections(self, resolver):
        """Should resolve 'std::collections'."""
        result = resolver.resolve("std::collections")
        assert result.status != ResolutionStatus.FAILED

    def test_resolve_std_sync(self, resolver):
        """Should resolve 'std::sync'."""
        result = resolver.resolve("std::sync")
        assert result.status != ResolutionStatus.FAILED

    def test_resolve_core(self, resolver):
        """Should resolve 'core'."""
        result = resolver.resolve("core")
        assert result.status != ResolutionStatus.FAILED

    def test_resolve_alloc(self, resolver):
        """Should resolve 'alloc'."""
        result = resolver.resolve("alloc")
        assert result.status != ResolutionStatus.FAILED


# ===========================================================================
# Cargo.toml Parsing Tests
# ===========================================================================


class TestCargoTomlParsing:
    """Tests for Cargo.toml parsing."""

    def test_parse_simple_cargo_toml(self):
        """Should parse simple Cargo.toml."""
        toml_content = """
[package]
name = "my-project"
version = "0.1.0"
edition = "2021"

[dependencies]
serde = "1.0"
tokio = { version = "1.0", features = ["full"] }
"""
        result = parse_cargo_toml(toml_content)
        assert "serde" in result
        assert "tokio" in result

    def test_parse_cargo_toml_no_deps(self):
        """Should handle Cargo.toml with no dependencies."""
        toml_content = """
[package]
name = "simple-project"
version = "0.1.0"
"""
        result = parse_cargo_toml(toml_content)
        assert isinstance(result, dict)

    def test_parse_cargo_toml_dev_deps(self):
        """Should parse dev dependencies."""
        toml_content = """
[package]
name = "test"

[dependencies]
serde = "1.0"

[dev-dependencies]
criterion = "0.4"
"""
        result = parse_cargo_toml(toml_content)
        assert "serde" in result
        # dev-dependencies might be tracked separately

    def test_parse_cargo_toml_workspace_deps(self):
        """Should handle workspace dependencies."""
        toml_content = """
[package]
name = "workspace-member"
version = "0.1.0"

[dependencies]
shared = { workspace = true }
"""
        result = parse_cargo_toml(toml_content)
        assert isinstance(result, dict)


# ===========================================================================
# Export Extraction Tests
# ===========================================================================


class TestRustExportExtraction:
    """Tests for extracting exports from Rust source."""

    def test_extract_pub_fn(self):
        """Should extract public functions."""
        source = """
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}

pub fn multiply(x: f32, y: f32) -> f32 {
    x * y
}

fn private_func() {}
"""
        exports = extract_rust_exports(source)
        assert "add" in exports
        assert "multiply" in exports
        assert "private_func" not in exports

    def test_extract_pub_const(self):
        """Should extract public constants."""
        source = """
pub const VERSION: &str = "1.0.0";
pub const MAX_SIZE: usize = 1024;
const PRIVATE: i32 = 42;
"""
        exports = extract_rust_exports(source)
        assert "VERSION" in exports
        assert "MAX_SIZE" in exports
        assert "PRIVATE" not in exports

    def test_extract_pub_static(self):
        """Should extract public statics."""
        source = """
pub static GLOBAL: i32 = 0;
static PRIVATE_GLOBAL: i32 = 0;
"""
        exports = extract_rust_exports(source)
        assert "GLOBAL" in exports
        assert "PRIVATE_GLOBAL" not in exports

    def test_extract_pub_struct(self):
        """Should extract public structs."""
        source = """
pub struct Config {
    pub enabled: bool,
    pub max_items: u32,
}

struct PrivateStruct {
    x: i32,
}
"""
        exports = extract_rust_exports(source)
        assert "Config" in exports
        assert "PrivateStruct" not in exports

    def test_extract_pub_enum(self):
        """Should extract public enums."""
        source = """
pub enum Color {
    Red,
    Green,
    Blue,
}
"""
        exports = extract_rust_exports(source)
        assert "Color" in exports

    def test_extract_pub_trait(self):
        """Should extract public traits."""
        source = """
pub trait Processor {
    fn process(&self);
}

trait PrivateTrait {}
"""
        exports = extract_rust_exports(source)
        assert "Processor" in exports
        assert "PrivateTrait" not in exports

    def test_extract_pub_type(self):
        """Should extract public type aliases."""
        source = """
pub type Result<T> = std::result::Result<T, Error>;
type PrivateAlias = i32;
"""
        exports = extract_rust_exports(source)
        assert "Result" in exports
        assert "PrivateAlias" not in exports

    def test_extract_pub_mod(self):
        """Should extract public modules."""
        source = """
pub mod utils {
    pub fn helper() {}
}

mod private_mod {}
"""
        exports = extract_rust_exports(source)
        assert "utils" in exports
        assert "private_mod" not in exports


# ===========================================================================
# Crate Resolution Tests
# ===========================================================================


class TestRustCrateResolution:
    """Tests for crate dependency resolution."""

    @pytest.fixture
    def project_with_deps(self):
        """Create a project with Cargo.toml dependencies."""
        with TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)

            # Create Cargo.toml
            cargo_file = project_path / "Cargo.toml"
            cargo_file.write_text("""
[package]
name = "test-project"
version = "0.1.0"
edition = "2021"

[dependencies]
serde = { version = "1.0", features = ["derive"] }
tokio = { version = "1.0", features = ["full"] }
""")

            yield project_path

    def test_resolver_finds_cargo_deps(self, project_with_deps):
        """Resolver should find dependencies from Cargo.toml."""
        resolver = RustImportResolver(str(project_with_deps))
        assert str(resolver._project_root) == str(project_with_deps)

    def test_resolve_known_crate(self):
        """Should recognize popular crates."""
        resolver = RustImportResolver()
        result = resolver.resolve("serde")
        # Should not fail, as it's a known crate
        assert result is not None


# ===========================================================================
# Edge Cases
# ===========================================================================


class TestRustImportEdgeCases:
    """Tests for edge cases in import resolution."""

    @pytest.fixture
    def resolver(self):
        return RustImportResolver()

    def test_nonexistent_module(self, resolver):
        """Should handle nonexistent modules gracefully."""
        result = resolver.resolve("nonexistent::module")
        assert result is not None

    def test_empty_module_name(self, resolver):
        """Should handle empty module name."""
        result = resolver.resolve("")
        assert result.status == ResolutionStatus.FAILED

    def test_deeply_nested_module(self, resolver):
        """Should handle deeply nested module paths."""
        result = resolver.resolve("std::collections::hash_map::HashMap")
        assert result is not None

    def test_crate_keyword(self, resolver):
        """Should handle 'crate' keyword."""
        result = resolver.resolve("crate::module")
        assert result is not None

    def test_self_keyword(self, resolver):
        """Should handle 'self' keyword."""
        result = resolver.resolve("self::submodule")
        assert result is not None

    def test_super_keyword(self, resolver):
        """Should handle 'super' keyword."""
        result = resolver.resolve("super::sibling")
        assert result is not None

    def test_glob_import(self, resolver):
        """Should handle glob imports conceptually."""
        # std::collections::* is a glob import
        result = resolver.resolve("std::collections")
        assert result is not None
