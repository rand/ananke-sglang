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
"""Unit tests for TypeScript import resolver.

Tests for the TypeScriptImportResolver implementation including:
- Node.js builtin module resolution
- package.json parsing
- tsconfig.json parsing
- Path alias resolution
- npm package dependency resolution
- Relative import resolution
- Export extraction
"""

import json
import pytest
from pathlib import Path

from domains.imports.resolvers import (
    TypeScriptImportResolver,
    NODE_BUILTIN_MODULES,
    POPULAR_NPM_PACKAGES,
    parse_package_json,
    parse_tsconfig,
    extract_typescript_exports,
    create_typescript_resolver,
    ResolutionStatus,
)
from domains.imports.resolvers.typescript import (
    PackageJson,
    TSConfig,
    ALL_NODE_BUILTINS,
    TYPESCRIPT_LIB_REFERENCES,
)


# ===========================================================================
# Node.js Builtin Module Tests
# ===========================================================================


class TestNodeBuiltinModules:
    """Tests for Node.js builtin module constants."""

    def test_node_builtins_contains_fs(self):
        """NODE_BUILTIN_MODULES should contain 'fs'."""
        assert "fs" in NODE_BUILTIN_MODULES

    def test_node_builtins_contains_path(self):
        """NODE_BUILTIN_MODULES should contain 'path'."""
        assert "path" in NODE_BUILTIN_MODULES

    def test_node_builtins_contains_http(self):
        """NODE_BUILTIN_MODULES should contain 'http'."""
        assert "http" in NODE_BUILTIN_MODULES

    def test_node_builtins_contains_crypto(self):
        """NODE_BUILTIN_MODULES should contain 'crypto'."""
        assert "crypto" in NODE_BUILTIN_MODULES

    def test_all_node_builtins_includes_prefixed(self):
        """ALL_NODE_BUILTINS should include node: prefix versions."""
        assert "node:fs" in ALL_NODE_BUILTINS
        assert "node:path" in ALL_NODE_BUILTINS

    def test_all_node_builtins_includes_submodules(self):
        """ALL_NODE_BUILTINS should include submodule paths."""
        assert "fs/promises" in ALL_NODE_BUILTINS
        assert "stream/web" in ALL_NODE_BUILTINS


# ===========================================================================
# Popular npm Packages Tests
# ===========================================================================


class TestPopularNpmPackages:
    """Tests for popular npm package constants."""

    def test_contains_react(self):
        """Should contain React ecosystem packages."""
        assert "react" in POPULAR_NPM_PACKAGES
        assert "react-dom" in POPULAR_NPM_PACKAGES

    def test_contains_typescript(self):
        """Should contain TypeScript."""
        assert "typescript" in POPULAR_NPM_PACKAGES

    def test_contains_express(self):
        """Should contain Express."""
        assert "express" in POPULAR_NPM_PACKAGES

    def test_contains_lodash(self):
        """Should contain lodash."""
        assert "lodash" in POPULAR_NPM_PACKAGES

    def test_contains_scoped_packages(self):
        """Should contain scoped packages."""
        assert "@types/node" in POPULAR_NPM_PACKAGES


# ===========================================================================
# TypeScript Lib References Tests
# ===========================================================================


class TestTypeScriptLibReferences:
    """Tests for TypeScript lib reference constants."""

    def test_contains_dom(self):
        """Should contain DOM."""
        assert "DOM" in TYPESCRIPT_LIB_REFERENCES

    def test_contains_es_versions(self):
        """Should contain ES versions."""
        assert "ES2015" in TYPESCRIPT_LIB_REFERENCES
        assert "ES2020" in TYPESCRIPT_LIB_REFERENCES
        assert "ESNext" in TYPESCRIPT_LIB_REFERENCES


# ===========================================================================
# Package.json Parsing Tests
# ===========================================================================


class TestPackageJsonParsing:
    """Tests for package.json parsing."""

    def test_parse_minimal_package_json(self, tmp_path):
        """Should parse minimal package.json."""
        pkg_json = tmp_path / "package.json"
        pkg_json.write_text('{"name": "test", "version": "1.0.0"}')

        result = parse_package_json(pkg_json)
        assert result is not None
        assert result.name == "test"
        assert result.version == "1.0.0"

    def test_parse_with_dependencies(self, tmp_path):
        """Should parse package.json with dependencies."""
        pkg_json = tmp_path / "package.json"
        pkg_json.write_text(json.dumps({
            "name": "test",
            "version": "1.0.0",
            "dependencies": {
                "react": "^18.0.0",
                "lodash": "^4.17.0"
            }
        }))

        result = parse_package_json(pkg_json)
        assert result.dependencies["react"] == "^18.0.0"
        assert result.dependencies["lodash"] == "^4.17.0"

    def test_parse_with_dev_dependencies(self, tmp_path):
        """Should parse devDependencies."""
        pkg_json = tmp_path / "package.json"
        pkg_json.write_text(json.dumps({
            "name": "test",
            "version": "1.0.0",
            "devDependencies": {
                "typescript": "^5.0.0",
                "jest": "^29.0.0"
            }
        }))

        result = parse_package_json(pkg_json)
        assert result.dev_dependencies["typescript"] == "^5.0.0"
        assert result.dev_dependencies["jest"] == "^29.0.0"

    def test_parse_with_types_field(self, tmp_path):
        """Should parse types field."""
        pkg_json = tmp_path / "package.json"
        pkg_json.write_text(json.dumps({
            "name": "test",
            "version": "1.0.0",
            "types": "dist/index.d.ts"
        }))

        result = parse_package_json(pkg_json)
        assert result.types == "dist/index.d.ts"

    def test_parse_with_exports(self, tmp_path):
        """Should parse exports field."""
        pkg_json = tmp_path / "package.json"
        pkg_json.write_text(json.dumps({
            "name": "test",
            "version": "1.0.0",
            "exports": {
                ".": "./dist/index.js",
                "./utils": "./dist/utils.js"
            }
        }))

        result = parse_package_json(pkg_json)
        assert result.exports["."] == "./dist/index.js"
        assert result.exports["./utils"] == "./dist/utils.js"

    def test_parse_nonexistent_returns_none(self, tmp_path):
        """Should return None for nonexistent file."""
        result = parse_package_json(tmp_path / "nonexistent.json")
        assert result is None

    def test_all_dependencies_combined(self, tmp_path):
        """all_dependencies should combine all dependency types."""
        pkg_json = tmp_path / "package.json"
        pkg_json.write_text(json.dumps({
            "name": "test",
            "version": "1.0.0",
            "dependencies": {"react": "^18.0.0"},
            "devDependencies": {"jest": "^29.0.0"},
            "peerDependencies": {"react-dom": "^18.0.0"}
        }))

        result = parse_package_json(pkg_json)
        all_deps = result.all_dependencies
        assert "react" in all_deps
        assert "jest" in all_deps
        assert "react-dom" in all_deps


# ===========================================================================
# tsconfig.json Parsing Tests
# ===========================================================================


class TestTSConfigParsing:
    """Tests for tsconfig.json parsing."""

    def test_parse_minimal_tsconfig(self, tmp_path):
        """Should parse minimal tsconfig.json."""
        tsconfig = tmp_path / "tsconfig.json"
        tsconfig.write_text('{}')

        result = parse_tsconfig(tsconfig)
        assert result is not None

    def test_parse_with_base_url(self, tmp_path):
        """Should parse baseUrl."""
        tsconfig = tmp_path / "tsconfig.json"
        tsconfig.write_text(json.dumps({
            "compilerOptions": {
                "baseUrl": "./src"
            }
        }))

        result = parse_tsconfig(tsconfig)
        assert result.base_url == "./src"

    def test_parse_with_paths(self, tmp_path):
        """Should parse paths."""
        tsconfig = tmp_path / "tsconfig.json"
        tsconfig.write_text(json.dumps({
            "compilerOptions": {
                "baseUrl": ".",
                "paths": {
                    "@utils/*": ["src/utils/*"],
                    "@components/*": ["src/components/*"]
                }
            }
        }))

        result = parse_tsconfig(tsconfig)
        assert "@utils/*" in result.paths
        assert result.paths["@utils/*"] == ["src/utils/*"]

    def test_parse_with_strict(self, tmp_path):
        """Should parse strict mode."""
        tsconfig = tmp_path / "tsconfig.json"
        tsconfig.write_text(json.dumps({
            "compilerOptions": {
                "strict": True
            }
        }))

        result = parse_tsconfig(tsconfig)
        assert result.strict is True

    def test_parse_with_comments(self, tmp_path):
        """Should parse tsconfig with comments."""
        tsconfig = tmp_path / "tsconfig.json"
        tsconfig.write_text('''
        {
            // This is a comment
            "compilerOptions": {
                /* Multi-line
                   comment */
                "strict": true
            }
        }
        ''')

        result = parse_tsconfig(tsconfig)
        assert result is not None
        assert result.strict is True

    def test_parse_nonexistent_returns_none(self, tmp_path):
        """Should return None for nonexistent file."""
        result = parse_tsconfig(tmp_path / "nonexistent.json")
        assert result is None


# ===========================================================================
# TypeScript Import Resolver Tests
# ===========================================================================


class TestTypeScriptImportResolver:
    """Tests for TypeScriptImportResolver."""

    @pytest.fixture
    def resolver(self):
        """Create a resolver without project context."""
        return TypeScriptImportResolver()

    def test_language_property(self, resolver):
        """Should return 'typescript' as language."""
        assert resolver.language == "typescript"

    # Node.js builtin resolution
    def test_resolve_fs(self, resolver):
        """Should resolve 'fs' as builtin."""
        result = resolver.resolve("fs")
        assert result.success
        assert result.status == ResolutionStatus.RESOLVED
        assert result.module.is_builtin

    def test_resolve_node_fs(self, resolver):
        """Should resolve 'node:fs' as builtin."""
        result = resolver.resolve("node:fs")
        assert result.success
        assert result.module.is_builtin

    def test_resolve_fs_promises(self, resolver):
        """Should resolve 'fs/promises' as builtin."""
        result = resolver.resolve("fs/promises")
        assert result.success
        assert result.module.is_builtin

    def test_resolve_path(self, resolver):
        """Should resolve 'path' as builtin."""
        result = resolver.resolve("path")
        assert result.success
        assert result.module.is_builtin

    def test_resolve_http(self, resolver):
        """Should resolve 'http' as builtin."""
        result = resolver.resolve("http")
        assert result.success
        assert result.module.is_builtin

    # Relative imports
    def test_resolve_relative_import(self, resolver):
        """Should handle relative import."""
        result = resolver.resolve("./utils")
        assert result.success
        assert result.status == ResolutionStatus.PARTIAL  # No file context

    def test_resolve_parent_relative_import(self, resolver):
        """Should handle parent relative import."""
        result = resolver.resolve("../utils")
        assert result.success
        assert result.status == ResolutionStatus.PARTIAL

    # Unknown modules
    def test_resolve_unknown_module(self, resolver):
        """Should fail for unknown module."""
        result = resolver.resolve("unknown-module-xyz")
        assert not result.success
        assert result.status == ResolutionStatus.FAILED


class TestTypeScriptImportResolverWithProject:
    """Tests for resolver with project context."""

    @pytest.fixture
    def project_dir(self, tmp_path):
        """Create a mock project directory."""
        # Create package.json
        pkg_json = tmp_path / "package.json"
        pkg_json.write_text(json.dumps({
            "name": "test-project",
            "version": "1.0.0",
            "dependencies": {
                "react": "^18.0.0",
                "lodash": "^4.17.0"
            },
            "devDependencies": {
                "typescript": "^5.0.0"
            }
        }))

        # Create tsconfig.json
        tsconfig = tmp_path / "tsconfig.json"
        tsconfig.write_text(json.dumps({
            "compilerOptions": {
                "baseUrl": ".",
                "paths": {
                    "@utils/*": ["src/utils/*"],
                    "@components/*": ["src/components/*"]
                }
            }
        }))

        return tmp_path

    @pytest.fixture
    def resolver(self, project_dir):
        """Create resolver with project context."""
        return TypeScriptImportResolver(project_root=str(project_dir))

    def test_resolve_package_dependency(self, resolver):
        """Should resolve package from package.json."""
        result = resolver.resolve("react")
        assert result.success
        assert result.module.version == "^18.0.0"

    def test_resolve_dev_dependency(self, resolver):
        """Should resolve devDependency."""
        result = resolver.resolve("typescript")
        assert result.success
        assert result.module.version == "^5.0.0"

    def test_resolve_path_alias(self, resolver):
        """Should resolve path alias from tsconfig."""
        result = resolver.resolve("@utils/helper")
        assert result.success

    def test_get_package_dependencies(self, resolver):
        """Should return all dependencies."""
        deps = resolver.get_package_dependencies()
        assert "react" in deps
        assert "lodash" in deps
        assert "typescript" in deps

    def test_get_path_aliases(self, resolver):
        """Should return path aliases."""
        aliases = resolver.get_path_aliases()
        assert "@utils/*" in aliases
        assert "@components/*" in aliases


class TestTypeScriptImportResolverSuggestions:
    """Tests for import resolver suggestions."""

    @pytest.fixture
    def resolver(self):
        return TypeScriptImportResolver()

    def test_suggest_popular_package(self, resolver):
        """Should suggest popular package not in package.json."""
        result = resolver.resolve("react")
        assert not result.success
        assert len(result.alternatives) > 0

    def test_suggest_alternatives_for_typo(self, resolver):
        """Should suggest alternatives for typo."""
        result = resolver.resolve("reac")  # Typo
        assert not result.success
        suggestions = resolver.suggest_alternatives("reac")
        # Should suggest react
        assert any("react" in s for s in suggestions)


# ===========================================================================
# Export Extraction Tests
# ===========================================================================


class TestExportExtraction:
    """Tests for TypeScript export extraction."""

    def test_extract_const_export(self):
        """Should extract const export."""
        source = "export const foo = 42;"
        exports = extract_typescript_exports(source)
        names = [e.name for e in exports]
        assert "foo" in names

    def test_extract_function_export(self):
        """Should extract function export."""
        source = "export function bar() {}"
        exports = extract_typescript_exports(source)
        names = [e.name for e in exports]
        assert "bar" in names

    def test_extract_async_function_export(self):
        """Should extract async function export."""
        source = "export async function fetchData() {}"
        exports = extract_typescript_exports(source)
        names = [e.name for e in exports]
        assert "fetchData" in names

    def test_extract_class_export(self):
        """Should extract class export."""
        source = "export class MyClass {}"
        exports = extract_typescript_exports(source)
        names = [e.name for e in exports]
        assert "MyClass" in names

    def test_extract_interface_export(self):
        """Should extract interface export."""
        source = "export interface MyInterface {}"
        exports = extract_typescript_exports(source)
        assert any(e.name == "MyInterface" and e.is_type_only for e in exports)

    def test_extract_type_export(self):
        """Should extract type export."""
        source = "export type MyType = string;"
        exports = extract_typescript_exports(source)
        assert any(e.name == "MyType" and e.is_type_only for e in exports)

    def test_extract_enum_export(self):
        """Should extract enum export."""
        source = "export enum Status { Active, Inactive }"
        exports = extract_typescript_exports(source)
        names = [e.name for e in exports]
        assert "Status" in names

    def test_extract_default_export(self):
        """Should extract default export."""
        source = "export default function main() {}"
        exports = extract_typescript_exports(source)
        names = [e.name for e in exports]
        assert "default" in names

    def test_extract_named_exports(self):
        """Should extract named exports."""
        source = "export { foo, bar, baz };"
        exports = extract_typescript_exports(source)
        names = [e.name for e in exports]
        assert "foo" in names
        assert "bar" in names
        assert "baz" in names

    def test_extract_renamed_export(self):
        """Should extract renamed export."""
        source = "export { foo as bar };"
        exports = extract_typescript_exports(source)
        names = [e.name for e in exports]
        assert "bar" in names

    def test_extract_type_only_export(self):
        """Should extract type-only export."""
        source = "export type { MyType };"
        exports = extract_typescript_exports(source)
        assert any(e.name == "MyType" and e.is_type_only for e in exports)


# ===========================================================================
# Factory Function Tests
# ===========================================================================


class TestCreateTypeScriptResolver:
    """Tests for create_typescript_resolver factory."""

    def test_create_without_args(self):
        """Should create resolver without arguments."""
        resolver = create_typescript_resolver()
        assert isinstance(resolver, TypeScriptImportResolver)

    def test_create_with_project_root(self, tmp_path):
        """Should create resolver with project root."""
        pkg_json = tmp_path / "package.json"
        pkg_json.write_text('{"name": "test", "version": "1.0.0"}')

        resolver = create_typescript_resolver(project_root=str(tmp_path))
        assert isinstance(resolver, TypeScriptImportResolver)


# ===========================================================================
# Refresh Tests
# ===========================================================================


class TestTypeScriptImportResolverRefresh:
    """Tests for resolver refresh functionality."""

    def test_refresh_clears_cache(self, tmp_path):
        """Refresh should clear resolution cache."""
        pkg_json = tmp_path / "package.json"
        pkg_json.write_text(json.dumps({
            "name": "test",
            "version": "1.0.0",
            "dependencies": {"react": "^18.0.0"}
        }))

        resolver = TypeScriptImportResolver(project_root=str(tmp_path))

        # First resolution
        result1 = resolver.resolve("react")
        assert result1.success

        # Update package.json
        pkg_json.write_text(json.dumps({
            "name": "test",
            "version": "1.0.0",
            "dependencies": {"react": "^17.0.0"}
        }))

        # Refresh and resolve again
        resolver.refresh()
        result2 = resolver.resolve("react")
        assert result2.module.version == "^17.0.0"
