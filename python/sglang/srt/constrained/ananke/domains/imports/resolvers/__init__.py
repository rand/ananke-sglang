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
"""Language-specific import resolvers for Python, Zig, Rust, and more.

This module provides import resolution for various languages:
- PythonImportResolver: Python stdlib and pip packages
- ZigImportResolver: Zig std library and build.zig.zon dependencies
- RustImportResolver: Rust std/core/alloc libraries and Cargo.toml dependencies
- PassthroughResolver: Testing resolver that always succeeds
- DenyListResolver: Security-focused resolver that blocks specific modules
"""

from __future__ import annotations

from .base import (
    ImportResolver,
    ImportResolution,
    ResolvedModule,
    ResolutionStatus,
    PassthroughResolver,
    DenyListResolver,
)
from .python import (
    PythonImportResolver,
    PYTHON_STDLIB,
    create_python_resolver,
)
from .zig import (
    ZigImportResolver,
    ZIG_STD_MODULES,
    parse_build_zig_zon,
    create_zig_resolver,
)
from .rust import (
    RustImportResolver,
    RUST_STD_MODULES,
    RUST_CORE_MODULES,
    RUST_ALLOC_MODULES,
    RUST_STD_TYPES,
    RUST_POPULAR_CRATES,
    parse_cargo_toml,
    create_rust_resolver,
)
from .typescript import (
    TypeScriptImportResolver,
    NODE_BUILTIN_MODULES,
    POPULAR_NPM_PACKAGES,
    parse_package_json,
    parse_tsconfig,
    extract_typescript_exports,
    create_typescript_resolver,
)
from .go import (
    GoImportResolver,
    GO_STANDARD_LIBRARY,
    GO_POPULAR_PACKAGES,
    create_go_resolver,
)
from .kotlin import (
    KotlinImportResolver,
    KOTLIN_STANDARD_LIBRARY,
    KOTLINX_COROUTINES,
    KOTLINX_SERIALIZATION,
    KOTLIN_POPULAR_PACKAGES,
    create_kotlin_resolver,
)
from .swift import (
    SwiftImportResolver,
    SWIFT_STANDARD_LIBRARY,
    FOUNDATION_FRAMEWORK,
    UIKIT_FRAMEWORK,
    SWIFTUI_FRAMEWORK,
    COMBINE_FRAMEWORK,
    SWIFT_POPULAR_PACKAGES,
    create_swift_resolver,
)

__all__ = [
    # Base types
    "ImportResolver",
    "ImportResolution",
    "ResolvedModule",
    "ResolutionStatus",
    # Utility resolvers
    "PassthroughResolver",
    "DenyListResolver",
    # Python resolver
    "PythonImportResolver",
    "PYTHON_STDLIB",
    "create_python_resolver",
    # Zig resolver
    "ZigImportResolver",
    "ZIG_STD_MODULES",
    "parse_build_zig_zon",
    "create_zig_resolver",
    # Rust resolver
    "RustImportResolver",
    "RUST_STD_MODULES",
    "RUST_CORE_MODULES",
    "RUST_ALLOC_MODULES",
    "RUST_STD_TYPES",
    "RUST_POPULAR_CRATES",
    "parse_cargo_toml",
    "create_rust_resolver",
    # TypeScript resolver
    "TypeScriptImportResolver",
    "NODE_BUILTIN_MODULES",
    "POPULAR_NPM_PACKAGES",
    "parse_package_json",
    "parse_tsconfig",
    "extract_typescript_exports",
    "create_typescript_resolver",
    # Go resolver
    "GoImportResolver",
    "GO_STANDARD_LIBRARY",
    "GO_POPULAR_PACKAGES",
    "create_go_resolver",
    # Kotlin resolver
    "KotlinImportResolver",
    "KOTLIN_STANDARD_LIBRARY",
    "KOTLINX_COROUTINES",
    "KOTLINX_SERIALIZATION",
    "KOTLIN_POPULAR_PACKAGES",
    "create_kotlin_resolver",
    # Swift resolver
    "SwiftImportResolver",
    "SWIFT_STANDARD_LIBRARY",
    "FOUNDATION_FRAMEWORK",
    "UIKIT_FRAMEWORK",
    "SWIFTUI_FRAMEWORK",
    "COMBINE_FRAMEWORK",
    "SWIFT_POPULAR_PACKAGES",
    "create_swift_resolver",
]
