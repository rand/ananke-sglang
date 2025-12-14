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
"""Rust import resolver.

This module provides import resolution for Rust crates and modules:

- std::*, core::*, alloc::* - Standard library
- Cargo.toml dependencies
- Local modules (mod, use)
- External crates (crates.io availability check)

References:
    - Cargo Reference: https://doc.rust-lang.org/cargo/reference/
    - Rust Modules: https://doc.rust-lang.org/book/ch07-00-managing-growing-projects-with-packages-crates-and-modules.html
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from domains.imports.resolvers.base import (
    ImportResolver,
    ImportResolution,
    ResolvedModule,
    ResolutionStatus,
)


# =============================================================================
# Rust Standard Library Modules
# =============================================================================

# Core std modules
RUST_STD_MODULES: Set[str] = {
    "std",
    "std::alloc",
    "std::any",
    "std::arch",
    "std::array",
    "std::ascii",
    "std::backtrace",
    "std::borrow",
    "std::boxed",
    "std::cell",
    "std::char",
    "std::clone",
    "std::cmp",
    "std::collections",
    "std::convert",
    "std::default",
    "std::env",
    "std::error",
    "std::f32",
    "std::f64",
    "std::ffi",
    "std::fmt",
    "std::fs",
    "std::future",
    "std::hash",
    "std::hint",
    "std::io",
    "std::iter",
    "std::marker",
    "std::mem",
    "std::net",
    "std::num",
    "std::ops",
    "std::option",
    "std::os",
    "std::panic",
    "std::path",
    "std::pin",
    "std::prelude",
    "std::primitive",
    "std::process",
    "std::ptr",
    "std::rc",
    "std::result",
    "std::slice",
    "std::str",
    "std::string",
    "std::sync",
    "std::task",
    "std::thread",
    "std::time",
    "std::vec",
}

# Core library (no_std)
RUST_CORE_MODULES: Set[str] = {
    "core",
    "core::alloc",
    "core::any",
    "core::arch",
    "core::array",
    "core::ascii",
    "core::borrow",
    "core::cell",
    "core::char",
    "core::clone",
    "core::cmp",
    "core::convert",
    "core::default",
    "core::f32",
    "core::f64",
    "core::ffi",
    "core::fmt",
    "core::future",
    "core::hash",
    "core::hint",
    "core::iter",
    "core::marker",
    "core::mem",
    "core::num",
    "core::ops",
    "core::option",
    "core::panic",
    "core::pin",
    "core::prelude",
    "core::primitive",
    "core::ptr",
    "core::result",
    "core::slice",
    "core::str",
    "core::task",
    "core::time",
}

# Alloc library
RUST_ALLOC_MODULES: Set[str] = {
    "alloc",
    "alloc::alloc",
    "alloc::borrow",
    "alloc::boxed",
    "alloc::collections",
    "alloc::fmt",
    "alloc::rc",
    "alloc::slice",
    "alloc::str",
    "alloc::string",
    "alloc::sync",
    "alloc::vec",
}

# Common std types and traits
RUST_STD_TYPES: Set[str] = {
    # Collections
    "std::collections::HashMap",
    "std::collections::HashSet",
    "std::collections::BTreeMap",
    "std::collections::BTreeSet",
    "std::collections::VecDeque",
    "std::collections::LinkedList",
    "std::collections::BinaryHeap",
    # Smart pointers
    "std::rc::Rc",
    "std::rc::Weak",
    "std::sync::Arc",
    "std::sync::Weak",
    "std::boxed::Box",
    "std::borrow::Cow",
    # Cell types
    "std::cell::Cell",
    "std::cell::RefCell",
    "std::cell::UnsafeCell",
    # Sync primitives
    "std::sync::Mutex",
    "std::sync::RwLock",
    "std::sync::Condvar",
    "std::sync::Barrier",
    "std::sync::Once",
    "std::sync::mpsc",
    # I/O
    "std::io::Read",
    "std::io::Write",
    "std::io::Seek",
    "std::io::BufRead",
    "std::io::BufReader",
    "std::io::BufWriter",
    "std::io::Cursor",
    "std::io::Error",
    "std::io::ErrorKind",
    # Filesystem
    "std::fs::File",
    "std::fs::OpenOptions",
    "std::fs::DirEntry",
    "std::fs::ReadDir",
    "std::fs::Metadata",
    # Path
    "std::path::Path",
    "std::path::PathBuf",
    # Network
    "std::net::TcpListener",
    "std::net::TcpStream",
    "std::net::UdpSocket",
    "std::net::IpAddr",
    "std::net::SocketAddr",
    # Process
    "std::process::Command",
    "std::process::Child",
    "std::process::Output",
    "std::process::Stdio",
    # Time
    "std::time::Duration",
    "std::time::Instant",
    "std::time::SystemTime",
    # Thread
    "std::thread::Thread",
    "std::thread::JoinHandle",
    "std::thread::Builder",
    # Common traits
    "std::clone::Clone",
    "std::cmp::Eq",
    "std::cmp::Ord",
    "std::cmp::PartialEq",
    "std::cmp::PartialOrd",
    "std::default::Default",
    "std::fmt::Debug",
    "std::fmt::Display",
    "std::hash::Hash",
    "std::iter::Iterator",
    "std::iter::IntoIterator",
    "std::convert::From",
    "std::convert::Into",
    "std::convert::TryFrom",
    "std::convert::TryInto",
    "std::convert::AsRef",
    "std::convert::AsMut",
    "std::ops::Deref",
    "std::ops::DerefMut",
    "std::ops::Drop",
}

# All std symbols
RUST_ALL_STD: Set[str] = (
    RUST_STD_MODULES |
    RUST_CORE_MODULES |
    RUST_ALLOC_MODULES |
    RUST_STD_TYPES
)


# =============================================================================
# Popular Crates
# =============================================================================

RUST_POPULAR_CRATES: Set[str] = {
    # Async
    "tokio",
    "async-std",
    "futures",
    "async-trait",
    # Web
    "reqwest",
    "hyper",
    "axum",
    "actix-web",
    "warp",
    "rocket",
    # Serialization
    "serde",
    "serde_json",
    "serde_yaml",
    "toml",
    "ron",
    "bincode",
    # Error handling
    "anyhow",
    "thiserror",
    "eyre",
    # CLI
    "clap",
    "structopt",
    "argh",
    # Logging
    "log",
    "env_logger",
    "tracing",
    "tracing-subscriber",
    # Testing
    "criterion",
    "proptest",
    "quickcheck",
    # Database
    "diesel",
    "sqlx",
    "rusqlite",
    # Regex
    "regex",
    # Random
    "rand",
    # Crypto
    "ring",
    "openssl",
    "rustls",
    # UUID
    "uuid",
    # Time
    "chrono",
    "time",
    # Collections
    "indexmap",
    "hashbrown",
    "smallvec",
    "tinyvec",
    # Lazy
    "lazy_static",
    "once_cell",
    # Derive
    "derive_more",
    "derive_builder",
    # Itertools
    "itertools",
    # Parsing
    "nom",
    "pest",
    "lalrpop",
    "syn",
    "quote",
    "proc-macro2",
    # Bytes
    "bytes",
    # Misc
    "rayon",
    "crossbeam",
    "parking_lot",
    "dashmap",
    "num",
    "num-traits",
    "bitflags",
    "libc",
    "winapi",
    "nix",
}


# =============================================================================
# Cargo.toml Parser (Simplified)
# =============================================================================

@dataclass
class CargoDependency:
    """A dependency from Cargo.toml."""
    name: str
    version: Optional[str] = None
    path: Optional[str] = None
    git: Optional[str] = None
    features: List[str] = None
    optional: bool = False

    def __post_init__(self):
        if self.features is None:
            self.features = []


def parse_cargo_toml(cargo_content: str) -> Dict[str, CargoDependency]:
    """Parse Cargo.toml content to extract dependencies.

    This is a simplified parser - TOML parsing would ideally use
    a proper TOML library.

    Args:
        cargo_content: The content of Cargo.toml

    Returns:
        Dictionary mapping dependency names to CargoDependency objects
    """
    deps: Dict[str, CargoDependency] = {}

    # Find [dependencies] section
    in_deps = False
    in_dev_deps = False
    in_build_deps = False

    for line in cargo_content.split("\n"):
        line = line.strip()

        # Skip comments
        if line.startswith("#"):
            continue

        # Check section headers
        if line == "[dependencies]":
            in_deps = True
            in_dev_deps = False
            in_build_deps = False
            continue
        elif line == "[dev-dependencies]":
            in_deps = False
            in_dev_deps = True
            in_build_deps = False
            continue
        elif line == "[build-dependencies]":
            in_deps = False
            in_dev_deps = False
            in_build_deps = True
            continue
        elif line.startswith("["):
            in_deps = False
            in_dev_deps = False
            in_build_deps = False
            continue

        # Parse dependency line
        if not (in_deps or in_dev_deps or in_build_deps):
            continue

        # Simple format: name = "version"
        simple_match = re.match(r'(\w[\w-]*)\s*=\s*"([^"]*)"', line)
        if simple_match:
            name = simple_match.group(1)
            version = simple_match.group(2)
            deps[name] = CargoDependency(name=name, version=version)
            continue

        # Complex format: name = { ... }
        complex_match = re.match(r'(\w[\w-]*)\s*=\s*\{(.+)\}', line)
        if complex_match:
            name = complex_match.group(1)
            content = complex_match.group(2)

            version = None
            path = None
            git = None

            # Extract fields
            version_match = re.search(r'version\s*=\s*"([^"]*)"', content)
            if version_match:
                version = version_match.group(1)

            path_match = re.search(r'path\s*=\s*"([^"]*)"', content)
            if path_match:
                path = path_match.group(1)

            git_match = re.search(r'git\s*=\s*"([^"]*)"', content)
            if git_match:
                git = git_match.group(1)

            deps[name] = CargoDependency(
                name=name,
                version=version,
                path=path,
                git=git,
            )

    return deps


# =============================================================================
# Rust Import Resolver
# =============================================================================

class RustImportResolver(ImportResolver):
    """Import resolver for Rust crates and modules.

    Resolves imports in the following order:
    1. Standard library (std::*, core::*, alloc::*)
    2. Cargo.toml dependencies
    3. Local modules
    4. Popular crates (for suggestion)

    Example:
        >>> resolver = RustImportResolver(project_root="/path/to/project")
        >>> result = resolver.resolve("std::collections::HashMap")
        >>> result.success
        True

        >>> result = resolver.resolve("serde")
        >>> # Checks Cargo.toml for serde dependency
    """

    def __init__(
        self,
        project_root: Optional[str] = None,
        cargo_toml_path: Optional[str] = None,
        check_crates_io: bool = False,
    ) -> None:
        """Initialize the Rust import resolver.

        Args:
            project_root: Root directory of the Rust project
            cargo_toml_path: Path to Cargo.toml
            check_crates_io: Whether to check crates.io for unknown crates
        """
        self._project_root = Path(project_root) if project_root else None
        self._check_crates_io = check_crates_io

        # Determine Cargo.toml path
        if cargo_toml_path:
            self._cargo_toml_path = Path(cargo_toml_path)
        elif self._project_root:
            self._cargo_toml_path = self._project_root / "Cargo.toml"
        else:
            self._cargo_toml_path = None

        # Parse dependencies from Cargo.toml
        self._cargo_deps: Dict[str, CargoDependency] = {}
        if self._cargo_toml_path and self._cargo_toml_path.exists():
            try:
                cargo_content = self._cargo_toml_path.read_text()
                self._cargo_deps = parse_cargo_toml(cargo_content)
            except Exception:
                pass

        # Cache for resolved modules
        self._cache: Dict[str, ImportResolution] = {}

    @property
    def language(self) -> str:
        return "rust"

    def resolve(self, module_name: str) -> ImportResolution:
        """Resolve a Rust import.

        Args:
            module_name: The import path (e.g., "std::collections::HashMap")

        Returns:
            ImportResolution with success/failure and module info
        """
        # Check cache
        if module_name in self._cache:
            return self._cache[module_name]

        result: ImportResolution

        # Standard library
        if module_name.startswith("std::") or module_name == "std":
            result = self._resolve_std(module_name)
        elif module_name.startswith("core::") or module_name == "core":
            result = self._resolve_core(module_name)
        elif module_name.startswith("alloc::") or module_name == "alloc":
            result = self._resolve_alloc(module_name)

        # Cargo.toml dependency
        elif self._get_crate_name(module_name) in self._cargo_deps:
            result = self._resolve_cargo_dep(module_name)

        # Local module (crate::, super::, self::)
        elif module_name.startswith(("crate::", "super::", "self::")):
            result = self._resolve_local_module(module_name)

        # Check if it's a known popular crate
        elif self._get_crate_name(module_name) in RUST_POPULAR_CRATES:
            result = self._resolve_popular_crate(module_name)

        # Unknown
        else:
            result = self._resolve_unknown(module_name)

        # Cache and return
        self._cache[module_name] = result
        return result

    def _get_crate_name(self, module_name: str) -> str:
        """Extract the crate name from a module path."""
        # Convert snake_case to kebab-case for crate names
        parts = module_name.split("::")
        if parts:
            return parts[0].replace("_", "-")
        return module_name.replace("_", "-")

    def _resolve_std(self, module_name: str) -> ImportResolution:
        """Resolve a standard library import."""
        if module_name in RUST_ALL_STD or module_name == "std":
            return ImportResolution(
                status=ResolutionStatus.RESOLVED,
                success=True,
                module=ResolvedModule(
                    name=module_name,
                    is_builtin=True,
                    is_available=True,
                ),
                module_name=module_name,
            )

        # Allow any std::* path
        if module_name.startswith("std::"):
            return ImportResolution(
                status=ResolutionStatus.PARTIAL,
                success=True,
                module=ResolvedModule(
                    name=module_name,
                    is_builtin=True,
                    is_available=True,
                ),
                module_name=module_name,
            )

        return ImportResolution(
            status=ResolutionStatus.FAILED,
            success=False,
            module_name=module_name,
            error=f"Unknown std module: {module_name}",
            alternatives=self._suggest_std_alternatives(module_name),
        )

    def _resolve_core(self, module_name: str) -> ImportResolution:
        """Resolve a core library import."""
        if module_name in RUST_CORE_MODULES or module_name.startswith("core::"):
            return ImportResolution(
                status=ResolutionStatus.RESOLVED,
                success=True,
                module=ResolvedModule(
                    name=module_name,
                    is_builtin=True,
                    is_available=True,
                ),
                module_name=module_name,
            )

        return ImportResolution(
            status=ResolutionStatus.FAILED,
            success=False,
            module_name=module_name,
            error=f"Unknown core module: {module_name}",
        )

    def _resolve_alloc(self, module_name: str) -> ImportResolution:
        """Resolve an alloc library import."""
        if module_name in RUST_ALLOC_MODULES or module_name.startswith("alloc::"):
            return ImportResolution(
                status=ResolutionStatus.RESOLVED,
                success=True,
                module=ResolvedModule(
                    name=module_name,
                    is_builtin=True,
                    is_available=True,
                ),
                module_name=module_name,
            )

        return ImportResolution(
            status=ResolutionStatus.FAILED,
            success=False,
            module_name=module_name,
            error=f"Unknown alloc module: {module_name}",
        )

    def _resolve_cargo_dep(self, module_name: str) -> ImportResolution:
        """Resolve a Cargo.toml dependency."""
        crate_name = self._get_crate_name(module_name)
        dep = self._cargo_deps.get(crate_name)

        if not dep:
            # Try with underscores
            crate_name_underscore = crate_name.replace("-", "_")
            dep = self._cargo_deps.get(crate_name_underscore)

        if dep:
            return ImportResolution(
                status=ResolutionStatus.RESOLVED,
                success=True,
                module=ResolvedModule(
                    name=module_name,
                    version=dep.version,
                    path=dep.path,
                    is_available=True,
                ),
                module_name=module_name,
            )

        return ImportResolution(
            status=ResolutionStatus.FAILED,
            success=False,
            module_name=module_name,
            error=f"Crate not in Cargo.toml: {crate_name}",
        )

    def _resolve_local_module(self, module_name: str) -> ImportResolution:
        """Resolve a local module (crate::, super::, self::)."""
        # These are always considered valid in context
        return ImportResolution(
            status=ResolutionStatus.PARTIAL,
            success=True,
            module=ResolvedModule(
                name=module_name,
                is_available=True,
            ),
            module_name=module_name,
        )

    def _resolve_popular_crate(self, module_name: str) -> ImportResolution:
        """Resolve a known popular crate that's not in Cargo.toml."""
        crate_name = self._get_crate_name(module_name)

        return ImportResolution(
            status=ResolutionStatus.FAILED,
            success=False,
            module_name=module_name,
            error=f"Crate '{crate_name}' is not in Cargo.toml. Add it with: cargo add {crate_name}",
            alternatives=[f"Add to Cargo.toml: {crate_name}"],
        )

    def _resolve_unknown(self, module_name: str) -> ImportResolution:
        """Resolve an unknown import."""
        crate_name = self._get_crate_name(module_name)

        suggestions = self._suggest_alternatives(crate_name)

        return ImportResolution(
            status=ResolutionStatus.FAILED,
            success=False,
            module_name=module_name,
            error=f"Unknown crate or module: {module_name}",
            alternatives=suggestions,
        )

    def is_available(self, module_name: str) -> bool:
        """Check if a module is available for import."""
        result = self.resolve(module_name)
        return result.success and (result.module is None or result.module.is_available)

    def get_version(self, module_name: str) -> Optional[str]:
        """Get the version of a crate."""
        crate_name = self._get_crate_name(module_name)

        if crate_name in self._cargo_deps:
            return self._cargo_deps[crate_name].version

        return None

    def get_exports(self, module_name: str) -> Set[str]:
        """Get names exported by a module.

        For Rust, this would require parsing the source files.
        Returns empty set for now.
        """
        return set()

    def suggest_alternatives(self, module_name: str) -> List[str]:
        """Suggest alternative modules."""
        return self._suggest_alternatives(module_name)

    def _suggest_std_alternatives(self, module_name: str) -> List[str]:
        """Suggest standard library alternatives."""
        suggestions = []
        module_lower = module_name.lower()

        for std_module in RUST_ALL_STD:
            if module_lower in std_module.lower():
                suggestions.append(std_module)

        return sorted(suggestions)[:5]

    def _suggest_alternatives(self, crate_name: str) -> List[str]:
        """Suggest crate alternatives."""
        suggestions = []
        crate_lower = crate_name.lower()

        # Check popular crates
        for crate in RUST_POPULAR_CRATES:
            if crate_lower in crate.lower() or crate.lower() in crate_lower:
                suggestions.append(crate)

        # Check Cargo.toml deps
        for dep in self._cargo_deps.keys():
            if crate_lower in dep.lower():
                suggestions.append(dep)

        return sorted(set(suggestions))[:5]

    def get_cargo_dependencies(self) -> Dict[str, CargoDependency]:
        """Get all dependencies from Cargo.toml."""
        return self._cargo_deps.copy()

    def refresh(self) -> None:
        """Refresh the resolver by re-reading Cargo.toml."""
        self._cache.clear()

        if self._cargo_toml_path and self._cargo_toml_path.exists():
            try:
                cargo_content = self._cargo_toml_path.read_text()
                self._cargo_deps = parse_cargo_toml(cargo_content)
            except Exception:
                self._cargo_deps = {}


def extract_rust_exports(source: str) -> Set[str]:
    """Extract public exports from Rust source code.

    This function uses regex patterns to find public items:
    - pub fn name
    - pub const NAME
    - pub static NAME
    - pub struct Name
    - pub enum Name
    - pub trait Name
    - pub type Name
    - pub mod name

    Args:
        source: Rust source code

    Returns:
        Set of exported symbol names
    """
    import re

    exports: Set[str] = set()

    # Pattern for pub fn
    for match in re.finditer(r'pub\s+fn\s+(\w+)', source):
        exports.add(match.group(1))

    # Pattern for pub const
    for match in re.finditer(r'pub\s+const\s+(\w+)', source):
        exports.add(match.group(1))

    # Pattern for pub static
    for match in re.finditer(r'pub\s+static\s+(\w+)', source):
        exports.add(match.group(1))

    # Pattern for pub struct
    for match in re.finditer(r'pub\s+struct\s+(\w+)', source):
        exports.add(match.group(1))

    # Pattern for pub enum
    for match in re.finditer(r'pub\s+enum\s+(\w+)', source):
        exports.add(match.group(1))

    # Pattern for pub trait
    for match in re.finditer(r'pub\s+trait\s+(\w+)', source):
        exports.add(match.group(1))

    # Pattern for pub type
    for match in re.finditer(r'pub\s+type\s+(\w+)', source):
        exports.add(match.group(1))

    # Pattern for pub mod
    for match in re.finditer(r'pub\s+mod\s+(\w+)', source):
        exports.add(match.group(1))

    return exports


def create_rust_resolver(
    project_root: Optional[str] = None,
) -> RustImportResolver:
    """Create a Rust import resolver with auto-detection.

    Args:
        project_root: Optional project root (auto-detected if not provided)

    Returns:
        Configured RustImportResolver
    """
    # Try to auto-detect project root
    if project_root is None:
        cwd = Path.cwd()
        # Look for Cargo.toml
        for parent in [cwd] + list(cwd.parents):
            if (parent / "Cargo.toml").exists():
                project_root = str(parent)
                break

    return RustImportResolver(project_root=project_root)
