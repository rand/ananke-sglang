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
"""TypeScript import resolver.

This module provides import resolution for TypeScript/JavaScript modules:

- Node.js built-in modules (fs, path, http, etc.)
- npm package dependencies
- TypeScript path aliases (from tsconfig.json)
- Relative imports
- Scoped packages (@types/*, @org/*)

References:
    - Node.js Module Resolution: https://nodejs.org/api/modules.html
    - TypeScript Module Resolution: https://www.typescriptlang.org/docs/handbook/module-resolution.html
    - package.json exports: https://nodejs.org/api/packages.html#packages_exports
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from domains.imports.resolvers.base import (
    ImportResolver,
    ImportResolution,
    ResolvedModule,
    ResolutionStatus,
)


# =============================================================================
# Node.js Built-in Modules
# =============================================================================

NODE_BUILTIN_MODULES: Set[str] = {
    # Core
    "assert",
    "async_hooks",
    "buffer",
    "child_process",
    "cluster",
    "console",
    "constants",
    "crypto",
    "dgram",
    "diagnostics_channel",
    "dns",
    "domain",
    "events",
    "fs",
    "http",
    "http2",
    "https",
    "inspector",
    "module",
    "net",
    "os",
    "path",
    "perf_hooks",
    "process",
    "punycode",
    "querystring",
    "readline",
    "repl",
    "stream",
    "string_decoder",
    "sys",
    "timers",
    "tls",
    "trace_events",
    "tty",
    "url",
    "util",
    "v8",
    "vm",
    "wasi",
    "worker_threads",
    "zlib",
}

# node: prefixed versions
NODE_PREFIXED_MODULES: Set[str] = {f"node:{mod}" for mod in NODE_BUILTIN_MODULES}

# Common submodules
NODE_SUBMODULES: Set[str] = {
    "fs/promises",
    "stream/web",
    "stream/promises",
    "stream/consumers",
    "util/types",
    "dns/promises",
    "timers/promises",
    "readline/promises",
    "node:fs/promises",
    "node:stream/web",
    "node:stream/promises",
    "node:stream/consumers",
    "node:util/types",
    "node:dns/promises",
    "node:timers/promises",
    "node:readline/promises",
}

ALL_NODE_BUILTINS: Set[str] = (
    NODE_BUILTIN_MODULES |
    NODE_PREFIXED_MODULES |
    NODE_SUBMODULES
)


# =============================================================================
# TypeScript Built-in Lib References
# =============================================================================

TYPESCRIPT_LIB_REFERENCES: Set[str] = {
    # ES versions
    "ES5",
    "ES6",
    "ES2015",
    "ES2016",
    "ES2017",
    "ES2018",
    "ES2019",
    "ES2020",
    "ES2021",
    "ES2022",
    "ES2023",
    "ESNext",
    # ES features
    "ES2015.Core",
    "ES2015.Collection",
    "ES2015.Generator",
    "ES2015.Iterable",
    "ES2015.Promise",
    "ES2015.Proxy",
    "ES2015.Reflect",
    "ES2015.Symbol",
    "ES2015.Symbol.WellKnown",
    "ES2016.Array.Include",
    "ES2017.Object",
    "ES2017.SharedMemory",
    "ES2017.String",
    "ES2017.Intl",
    "ES2017.TypedArrays",
    "ES2018.AsyncGenerator",
    "ES2018.AsyncIterable",
    "ES2018.Intl",
    "ES2018.Promise",
    "ES2018.Regexp",
    "ES2019.Array",
    "ES2019.Object",
    "ES2019.String",
    "ES2019.Symbol",
    "ES2020.BigInt",
    "ES2020.Promise",
    "ES2020.SharedMemory",
    "ES2020.String",
    "ES2020.Symbol.WellKnown",
    "ES2020.Intl",
    "ES2021.Promise",
    "ES2021.String",
    "ES2021.WeakRef",
    "ES2021.Intl",
    "ES2022.Array",
    "ES2022.Error",
    "ES2022.Intl",
    "ES2022.Object",
    "ES2022.SharedMemory",
    "ES2022.String",
    "ES2023.Array",
    # DOM
    "DOM",
    "DOM.Iterable",
    "WebWorker",
    "WebWorker.ImportScripts",
    "ScriptHost",
    # Decorators
    "Decorators",
    "Decorators.Legacy",
}


# =============================================================================
# Popular npm Packages
# =============================================================================

POPULAR_NPM_PACKAGES: Set[str] = {
    # Build tools
    "typescript",
    "webpack",
    "vite",
    "esbuild",
    "rollup",
    "parcel",
    "swc",
    "babel",
    "@babel/core",
    "@babel/preset-env",
    "@babel/preset-typescript",
    # Frameworks
    "react",
    "react-dom",
    "vue",
    "angular",
    "@angular/core",
    "@angular/common",
    "svelte",
    "next",
    "nuxt",
    "remix",
    "astro",
    # State management
    "redux",
    "@reduxjs/toolkit",
    "zustand",
    "mobx",
    "recoil",
    "jotai",
    "valtio",
    # Server
    "express",
    "fastify",
    "koa",
    "hapi",
    "nest",
    "@nestjs/core",
    "@nestjs/common",
    "hono",
    # Testing
    "jest",
    "@jest/core",
    "vitest",
    "mocha",
    "chai",
    "playwright",
    "@playwright/test",
    "cypress",
    "@testing-library/react",
    "@testing-library/dom",
    # Database
    "prisma",
    "@prisma/client",
    "mongoose",
    "typeorm",
    "sequelize",
    "drizzle-orm",
    "knex",
    # Utilities
    "lodash",
    "underscore",
    "ramda",
    "date-fns",
    "dayjs",
    "moment",
    "uuid",
    "nanoid",
    # HTTP
    "axios",
    "node-fetch",
    "got",
    "ky",
    "superagent",
    # Validation
    "zod",
    "yup",
    "joi",
    "io-ts",
    "ajv",
    "class-validator",
    # CLI
    "commander",
    "yargs",
    "inquirer",
    "chalk",
    "ora",
    "execa",
    # Types
    "@types/node",
    "@types/react",
    "@types/react-dom",
    "@types/express",
    "@types/lodash",
    "@types/jest",
    # Monorepo
    "lerna",
    "turbo",
    "nx",
    "@nx/workspace",
    "pnpm",
    # Misc
    "dotenv",
    "cors",
    "helmet",
    "morgan",
    "winston",
    "pino",
    "debug",
    "rxjs",
    "immer",
    "graphql",
    "@apollo/client",
    "@apollo/server",
    "trpc",
    "@trpc/server",
    "@trpc/client",
}


# =============================================================================
# Package Manifest Parsing
# =============================================================================

@dataclass
class PackageJson:
    """Parsed package.json metadata.

    Attributes:
        name: Package name
        version: Package version
        dependencies: Runtime dependencies
        dev_dependencies: Development dependencies
        peer_dependencies: Peer dependencies
        types: Path to .d.ts file (types or typings field)
        main: Main entry point
        module: ES module entry point
        exports: Package exports map
        imports: Package imports map (subpath imports)
    """
    name: str = ""
    version: str = "0.0.0"
    dependencies: Dict[str, str] = field(default_factory=dict)
    dev_dependencies: Dict[str, str] = field(default_factory=dict)
    peer_dependencies: Dict[str, str] = field(default_factory=dict)
    types: Optional[str] = None
    main: Optional[str] = None
    module: Optional[str] = None
    exports: Dict[str, Any] = field(default_factory=dict)
    imports: Dict[str, Any] = field(default_factory=dict)

    @property
    def all_dependencies(self) -> Dict[str, str]:
        """Get all dependencies combined."""
        return {
            **self.dependencies,
            **self.dev_dependencies,
            **self.peer_dependencies,
        }


@dataclass
class TSConfig:
    """Parsed tsconfig.json relevant fields.

    Attributes:
        base_url: Base URL for non-relative module names
        paths: Path alias mappings
        types: Type roots to include
        type_roots: Directories to search for type definitions
        strict: Whether strict mode is enabled
        module: Module system (commonjs, esnext, etc.)
        target: ECMAScript target version
        root_dir: Root directory of source files
        out_dir: Output directory
        extends: Extended config file
    """
    base_url: Optional[str] = None
    paths: Dict[str, List[str]] = field(default_factory=dict)
    types: List[str] = field(default_factory=list)
    type_roots: List[str] = field(default_factory=list)
    strict: bool = False
    module: Optional[str] = None
    target: Optional[str] = None
    root_dir: Optional[str] = None
    out_dir: Optional[str] = None
    extends: Optional[str] = None


def _strip_json_comments(text: str) -> str:
    """Strip comments from JSON (TypeScript allows comments in tsconfig.json).

    Args:
        text: JSON text with possible comments

    Returns:
        JSON text with comments removed
    """
    result = []
    i = 0
    in_string = False
    escape_next = False

    while i < len(text):
        char = text[i]

        if escape_next:
            result.append(char)
            escape_next = False
            i += 1
            continue

        if char == '\\' and in_string:
            result.append(char)
            escape_next = True
            i += 1
            continue

        if char == '"' and not escape_next:
            in_string = not in_string
            result.append(char)
            i += 1
            continue

        if not in_string:
            # Check for // comment
            if char == '/' and i + 1 < len(text) and text[i + 1] == '/':
                # Skip to end of line
                while i < len(text) and text[i] != '\n':
                    i += 1
                continue

            # Check for /* */ comment
            if char == '/' and i + 1 < len(text) and text[i + 1] == '*':
                # Skip to */
                i += 2
                while i + 1 < len(text) and not (text[i] == '*' and text[i + 1] == '/'):
                    i += 1
                i += 2
                continue

        result.append(char)
        i += 1

    return ''.join(result)


def parse_package_json(path: Path) -> Optional[PackageJson]:
    """Parse package.json file.

    Args:
        path: Path to package.json

    Returns:
        PackageJson if successful, None if file doesn't exist or is invalid
    """
    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text())
        return PackageJson(
            name=data.get("name", ""),
            version=data.get("version", "0.0.0"),
            dependencies=data.get("dependencies", {}),
            dev_dependencies=data.get("devDependencies", {}),
            peer_dependencies=data.get("peerDependencies", {}),
            types=data.get("types") or data.get("typings"),
            main=data.get("main"),
            module=data.get("module"),
            exports=data.get("exports", {}),
            imports=data.get("imports", {}),
        )
    except (json.JSONDecodeError, OSError):
        return None


def parse_tsconfig(path: Path) -> Optional[TSConfig]:
    """Parse tsconfig.json file.

    Args:
        path: Path to tsconfig.json

    Returns:
        TSConfig if successful, None if file doesn't exist or is invalid
    """
    if not path.exists():
        return None

    try:
        # Handle JSON with comments (TypeScript allows them)
        text = _strip_json_comments(path.read_text())
        data = json.loads(text)
        compiler_options = data.get("compilerOptions", {})

        return TSConfig(
            base_url=compiler_options.get("baseUrl"),
            paths=compiler_options.get("paths", {}),
            types=compiler_options.get("types", []),
            type_roots=compiler_options.get("typeRoots", []),
            strict=compiler_options.get("strict", False),
            module=compiler_options.get("module"),
            target=compiler_options.get("target"),
            root_dir=compiler_options.get("rootDir"),
            out_dir=compiler_options.get("outDir"),
            extends=data.get("extends"),
        )
    except (json.JSONDecodeError, OSError):
        return None


# =============================================================================
# Export Extraction
# =============================================================================

@dataclass
class ExportInfo:
    """Information about an exported symbol.

    Attributes:
        name: Export name (or "default" for default export)
        kind: Kind of export (value, function, class, interface, type, enum)
        is_type_only: Whether this is a type-only export
    """
    name: str
    kind: str
    is_type_only: bool = False


def extract_typescript_exports(source: str) -> List[ExportInfo]:
    """Extract exports from TypeScript source.

    Args:
        source: TypeScript source code

    Returns:
        List of ExportInfo for each export found
    """
    exports: List[ExportInfo] = []

    # export const/let/var name
    for match in re.finditer(r'export\s+(?:const|let|var)\s+(\w+)', source):
        exports.append(ExportInfo(name=match.group(1), kind="value"))

    # export function name
    for match in re.finditer(r'export\s+(?:async\s+)?function\s+(\w+)', source):
        exports.append(ExportInfo(name=match.group(1), kind="function"))

    # export class name
    for match in re.finditer(r'export\s+(?:abstract\s+)?class\s+(\w+)', source):
        exports.append(ExportInfo(name=match.group(1), kind="class"))

    # export interface name
    for match in re.finditer(r'export\s+interface\s+(\w+)', source):
        exports.append(ExportInfo(name=match.group(1), kind="interface", is_type_only=True))

    # export type name
    for match in re.finditer(r'export\s+type\s+(\w+)', source):
        exports.append(ExportInfo(name=match.group(1), kind="type", is_type_only=True))

    # export enum name
    for match in re.finditer(r'export\s+(?:const\s+)?enum\s+(\w+)', source):
        exports.append(ExportInfo(name=match.group(1), kind="enum"))

    # export default
    if re.search(r'export\s+default\s+', source):
        exports.append(ExportInfo(name="default", kind="default"))

    # export { name, name as alias }
    for match in re.finditer(r'export\s*\{([^}]+)\}', source):
        for item in match.group(1).split(','):
            item = item.strip()
            if not item:
                continue
            if ' as ' in item:
                _, alias = item.split(' as ', 1)
                exports.append(ExportInfo(name=alias.strip(), kind="reexport"))
            else:
                exports.append(ExportInfo(name=item, kind="value"))

    # export type { name } (type-only)
    for match in re.finditer(r'export\s+type\s*\{([^}]+)\}', source):
        for item in match.group(1).split(','):
            item = item.strip()
            if not item:
                continue
            if ' as ' in item:
                _, alias = item.split(' as ', 1)
                exports.append(ExportInfo(name=alias.strip(), kind="type", is_type_only=True))
            else:
                exports.append(ExportInfo(name=item, kind="type", is_type_only=True))

    return exports


# =============================================================================
# TypeScript Import Resolver
# =============================================================================

class TypeScriptImportResolver(ImportResolver):
    """Import resolver for TypeScript/JavaScript modules.

    Resolves imports in the following order:
    1. Node.js built-in modules (fs, path, etc.)
    2. Path aliases from tsconfig.json
    3. Relative imports (./foo, ../bar)
    4. Package dependencies from package.json
    5. Popular npm packages (for suggestions)

    Example:
        >>> resolver = TypeScriptImportResolver(project_root="/path/to/project")
        >>> result = resolver.resolve("fs")
        >>> result.success
        True

        >>> result = resolver.resolve("@types/node")
        >>> # Checks package.json for @types/node dependency
    """

    def __init__(
        self,
        project_root: Optional[str] = None,
        package_json_path: Optional[str] = None,
        tsconfig_path: Optional[str] = None,
        check_npm_registry: bool = False,
    ) -> None:
        """Initialize the TypeScript import resolver.

        Args:
            project_root: Root directory of the TypeScript project
            package_json_path: Path to package.json
            tsconfig_path: Path to tsconfig.json
            check_npm_registry: Whether to check npm registry for unknown packages
        """
        self._project_root = Path(project_root) if project_root else None
        self._check_npm_registry = check_npm_registry

        # Determine package.json path
        if package_json_path:
            self._package_json_path = Path(package_json_path)
        elif self._project_root:
            self._package_json_path = self._project_root / "package.json"
        else:
            self._package_json_path = None

        # Determine tsconfig.json path
        if tsconfig_path:
            self._tsconfig_path = Path(tsconfig_path)
        elif self._project_root:
            self._tsconfig_path = self._project_root / "tsconfig.json"
        else:
            self._tsconfig_path = None

        # Parse configuration files
        self._package_json: Optional[PackageJson] = None
        if self._package_json_path:
            self._package_json = parse_package_json(self._package_json_path)

        self._tsconfig: Optional[TSConfig] = None
        if self._tsconfig_path:
            self._tsconfig = parse_tsconfig(self._tsconfig_path)

        # Cache for resolved modules
        self._cache: Dict[str, ImportResolution] = {}

    @property
    def language(self) -> str:
        return "typescript"

    def resolve(self, module_name: str) -> ImportResolution:
        """Resolve a TypeScript/JavaScript import.

        Args:
            module_name: The import specifier

        Returns:
            ImportResolution with success/failure and module info
        """
        # Check cache
        if module_name in self._cache:
            return self._cache[module_name]

        result: ImportResolution

        # Node.js built-in module
        if module_name in ALL_NODE_BUILTINS:
            result = self._resolve_builtin(module_name)

        # TypeScript lib reference
        elif module_name in TYPESCRIPT_LIB_REFERENCES:
            result = self._resolve_lib_reference(module_name)

        # Relative import
        elif module_name.startswith(".") or module_name.startswith("/"):
            result = self._resolve_relative(module_name)

        # Path alias from tsconfig.json
        elif self._is_path_alias(module_name):
            result = self._resolve_path_alias(module_name)

        # Package from package.json
        elif self._is_package_dependency(module_name):
            result = self._resolve_package_dependency(module_name)

        # Check if it's a popular package
        elif self._get_package_name(module_name) in POPULAR_NPM_PACKAGES:
            result = self._resolve_popular_package(module_name)

        # Unknown
        else:
            result = self._resolve_unknown(module_name)

        # Cache and return
        self._cache[module_name] = result
        return result

    def _get_package_name(self, module_name: str) -> str:
        """Extract the package name from a module specifier.

        Handles scoped packages (@org/pkg) and subpath imports (pkg/subpath).
        """
        # Scoped package
        if module_name.startswith("@"):
            parts = module_name.split("/")
            if len(parts) >= 2:
                return f"{parts[0]}/{parts[1]}"
            return module_name

        # Regular package
        parts = module_name.split("/")
        return parts[0]

    def _is_path_alias(self, module_name: str) -> bool:
        """Check if module_name matches a path alias."""
        if not self._tsconfig or not self._tsconfig.paths:
            return False

        for pattern in self._tsconfig.paths.keys():
            # Handle wildcard patterns like @utils/*
            if pattern.endswith("/*"):
                prefix = pattern[:-2]
                if module_name.startswith(prefix):
                    return True
            elif module_name == pattern:
                return True

        return False

    def _is_package_dependency(self, module_name: str) -> bool:
        """Check if module_name is a dependency in package.json."""
        if not self._package_json:
            return False

        pkg_name = self._get_package_name(module_name)
        return pkg_name in self._package_json.all_dependencies

    def _resolve_builtin(self, module_name: str) -> ImportResolution:
        """Resolve a Node.js built-in module."""
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

    def _resolve_lib_reference(self, module_name: str) -> ImportResolution:
        """Resolve a TypeScript lib reference."""
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

    def _resolve_relative(self, module_name: str) -> ImportResolution:
        """Resolve a relative import.

        Relative imports are always considered valid in context,
        actual file resolution would require the importing file's location.
        """
        return ImportResolution(
            status=ResolutionStatus.PARTIAL,
            success=True,
            module=ResolvedModule(
                name=module_name,
                is_available=True,
            ),
            module_name=module_name,
        )

    def _resolve_path_alias(self, module_name: str) -> ImportResolution:
        """Resolve a path alias from tsconfig.json."""
        if not self._tsconfig or not self._tsconfig.paths:
            return self._resolve_unknown(module_name)

        for pattern, targets in self._tsconfig.paths.items():
            if pattern.endswith("/*"):
                prefix = pattern[:-2]
                if module_name.startswith(prefix):
                    # Found matching alias
                    return ImportResolution(
                        status=ResolutionStatus.RESOLVED,
                        success=True,
                        module=ResolvedModule(
                            name=module_name,
                            is_available=True,
                        ),
                        module_name=module_name,
                    )
            elif module_name == pattern:
                return ImportResolution(
                    status=ResolutionStatus.RESOLVED,
                    success=True,
                    module=ResolvedModule(
                        name=module_name,
                        is_available=True,
                    ),
                    module_name=module_name,
                )

        return self._resolve_unknown(module_name)

    def _resolve_package_dependency(self, module_name: str) -> ImportResolution:
        """Resolve a package from package.json."""
        pkg_name = self._get_package_name(module_name)

        if not self._package_json:
            return self._resolve_unknown(module_name)

        version = self._package_json.all_dependencies.get(pkg_name)

        return ImportResolution(
            status=ResolutionStatus.RESOLVED,
            success=True,
            module=ResolvedModule(
                name=module_name,
                version=version,
                is_available=True,
            ),
            module_name=module_name,
        )

    def _resolve_popular_package(self, module_name: str) -> ImportResolution:
        """Resolve a known popular package not in package.json."""
        pkg_name = self._get_package_name(module_name)

        return ImportResolution(
            status=ResolutionStatus.FAILED,
            success=False,
            module_name=module_name,
            error=f"Package '{pkg_name}' is not in package.json. Install with: npm install {pkg_name}",
            alternatives=[f"npm install {pkg_name}", f"pnpm add {pkg_name}", f"yarn add {pkg_name}"],
        )

    def _resolve_unknown(self, module_name: str) -> ImportResolution:
        """Resolve an unknown import."""
        pkg_name = self._get_package_name(module_name)

        suggestions = self._suggest_alternatives(pkg_name)

        return ImportResolution(
            status=ResolutionStatus.FAILED,
            success=False,
            module_name=module_name,
            error=f"Unknown module: {module_name}",
            alternatives=suggestions,
        )

    def is_available(self, module_name: str) -> bool:
        """Check if a module is available for import."""
        result = self.resolve(module_name)
        return result.success and (result.module is None or result.module.is_available)

    def get_version(self, module_name: str) -> Optional[str]:
        """Get the version of a package."""
        pkg_name = self._get_package_name(module_name)

        if self._package_json:
            return self._package_json.all_dependencies.get(pkg_name)

        return None

    def get_exports(self, module_name: str) -> Set[str]:
        """Get names exported by a module.

        For TypeScript, this would require parsing the source files.
        Returns empty set for now.
        """
        return set()

    def suggest_alternatives(self, module_name: str) -> List[str]:
        """Suggest alternative modules."""
        return self._suggest_alternatives(module_name)

    def _suggest_alternatives(self, pkg_name: str) -> List[str]:
        """Suggest package alternatives."""
        suggestions = []
        pkg_lower = pkg_name.lower()

        # Check popular packages
        for pkg in POPULAR_NPM_PACKAGES:
            if pkg_lower in pkg.lower() or pkg.lower() in pkg_lower:
                suggestions.append(pkg)

        # Check package.json deps
        if self._package_json:
            for dep in self._package_json.all_dependencies.keys():
                if pkg_lower in dep.lower():
                    suggestions.append(dep)

        return sorted(set(suggestions))[:5]

    def resolve_from_file(
        self,
        module_name: str,
        from_file: Path,
    ) -> ImportResolution:
        """Resolve an import relative to a specific file.

        Args:
            module_name: The import specifier
            from_file: The file containing the import

        Returns:
            ImportResolution with file path if found
        """
        # For non-relative imports, use standard resolution
        if not module_name.startswith("."):
            return self.resolve(module_name)

        # Resolve relative to the importing file
        base_dir = from_file.parent
        target = (base_dir / module_name).resolve()

        # Try various extensions
        extensions = [".ts", ".tsx", ".d.ts", ".js", ".jsx", ".mjs", ".cjs"]

        for ext in extensions:
            candidate = target.with_suffix(ext)
            if candidate.exists():
                return ImportResolution(
                    status=ResolutionStatus.RESOLVED,
                    success=True,
                    module=ResolvedModule(
                        name=module_name,
                        path=str(candidate),
                        is_available=True,
                    ),
                    module_name=module_name,
                )

        # Try index file
        for ext in extensions:
            candidate = target / f"index{ext}"
            if candidate.exists():
                return ImportResolution(
                    status=ResolutionStatus.RESOLVED,
                    success=True,
                    module=ResolvedModule(
                        name=module_name,
                        path=str(candidate),
                        is_available=True,
                    ),
                    module_name=module_name,
                )

        # Not found
        return ImportResolution(
            status=ResolutionStatus.FAILED,
            success=False,
            module_name=module_name,
            error=f"Could not resolve relative import: {module_name}",
        )

    def resolve_node_module(
        self,
        module_name: str,
        search_dir: Optional[Path] = None,
    ) -> ImportResolution:
        """Resolve a module from node_modules.

        Args:
            module_name: The module specifier
            search_dir: Directory to start searching from

        Returns:
            ImportResolution with path to module
        """
        pkg_name = self._get_package_name(module_name)
        subpath = module_name[len(pkg_name):].lstrip("/") if len(module_name) > len(pkg_name) else ""

        # Start search from project root or search_dir
        start_dir = search_dir or self._project_root or Path.cwd()

        # Walk up looking for node_modules
        for parent in [start_dir] + list(start_dir.parents):
            node_modules = parent / "node_modules" / pkg_name

            if node_modules.exists():
                pkg_json_path = node_modules / "package.json"
                pkg_json = parse_package_json(pkg_json_path)

                if pkg_json:
                    # Resolve entry point
                    if subpath:
                        # Check exports map
                        if pkg_json.exports and f"./{subpath}" in pkg_json.exports:
                            entry = pkg_json.exports[f"./{subpath}"]
                            if isinstance(entry, dict):
                                entry = (
                                    entry.get("types") or
                                    entry.get("import") or
                                    entry.get("require") or
                                    entry.get("default")
                                )
                            if entry:
                                entry_path = node_modules / entry
                                return ImportResolution(
                                    status=ResolutionStatus.RESOLVED,
                                    success=True,
                                    module=ResolvedModule(
                                        name=module_name,
                                        version=pkg_json.version,
                                        path=str(entry_path),
                                        is_available=True,
                                    ),
                                    module_name=module_name,
                                )
                    else:
                        # Main entry
                        entry = pkg_json.types or pkg_json.main or "index.js"
                        entry_path = node_modules / entry
                        return ImportResolution(
                            status=ResolutionStatus.RESOLVED,
                            success=True,
                            module=ResolvedModule(
                                name=module_name,
                                version=pkg_json.version,
                                path=str(entry_path),
                                is_available=True,
                            ),
                            module_name=module_name,
                        )

        # Not found in node_modules
        return ImportResolution(
            status=ResolutionStatus.FAILED,
            success=False,
            module_name=module_name,
            error=f"Could not find package in node_modules: {pkg_name}",
        )

    def get_package_dependencies(self) -> Dict[str, str]:
        """Get all dependencies from package.json."""
        if self._package_json:
            return self._package_json.all_dependencies.copy()
        return {}

    def get_path_aliases(self) -> Dict[str, List[str]]:
        """Get path aliases from tsconfig.json."""
        if self._tsconfig:
            return self._tsconfig.paths.copy()
        return {}

    def refresh(self) -> None:
        """Refresh the resolver by re-reading config files."""
        self._cache.clear()

        if self._package_json_path:
            self._package_json = parse_package_json(self._package_json_path)

        if self._tsconfig_path:
            self._tsconfig = parse_tsconfig(self._tsconfig_path)


def create_typescript_resolver(
    project_root: Optional[str] = None,
) -> TypeScriptImportResolver:
    """Create a TypeScript import resolver with auto-detection.

    Args:
        project_root: Optional project root (auto-detected if not provided)

    Returns:
        Configured TypeScriptImportResolver
    """
    # Try to auto-detect project root
    if project_root is None:
        cwd = Path.cwd()
        # Look for package.json or tsconfig.json
        for parent in [cwd] + list(cwd.parents):
            if (parent / "package.json").exists() or (parent / "tsconfig.json").exists():
                project_root = str(parent)
                break

    return TypeScriptImportResolver(project_root=project_root)
