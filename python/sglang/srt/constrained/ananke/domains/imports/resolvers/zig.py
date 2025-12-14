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
"""Zig import resolver.

This module provides import resolution for Zig's @import builtin:

- @import("std") - Standard library
- @import("package") - Dependencies from build.zig.zon
- @import("file.zig") - Local file imports
- @embedFile("path") - File embedding

The resolver checks:
1. Standard library module existence
2. build.zig.zon dependencies
3. Local file paths relative to project root

References:
    - Zig Build System: https://ziglang.org/documentation/master/#Build-System
    - build.zig.zon format: https://ziglang.org/documentation/master/#Build-Mode
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
# Zig Standard Library Modules
# =============================================================================

# Top-level std modules
ZIG_STD_MODULES: Set[str] = {
    "std",
    "std.mem",
    "std.fmt",
    "std.debug",
    "std.io",
    "std.fs",
    "std.os",
    "std.heap",
    "std.math",
    "std.json",
    "std.sort",
    "std.rand",
    "std.time",
    "std.meta",
    "std.testing",
    "std.log",
    "std.http",
    "std.net",
    "std.crypto",
    "std.compress",
    "std.unicode",
    "std.ascii",
    "std.simd",
    "std.atomic",
    "std.Thread",
    "std.process",
    "std.Progress",
    "std.enums",
    "std.fifo",
    "std.builtin",
    "std.c",
    "std.zig",
    "std.start",
    "std.target",
}

# Common std data structures
ZIG_STD_DATA_STRUCTURES: Set[str] = {
    "ArrayList",
    "ArrayListUnmanaged",
    "AutoHashMap",
    "AutoHashMapUnmanaged",
    "AutoArrayHashMap",
    "AutoArrayHashMapUnmanaged",
    "HashMap",
    "HashMapUnmanaged",
    "ArrayHashMap",
    "ArrayHashMapUnmanaged",
    "StringHashMap",
    "StringHashMapUnmanaged",
    "StringArrayHashMap",
    "BufSet",
    "BufMap",
    "ComptimeStringMap",
    "EnumSet",
    "EnumMap",
    "EnumArray",
    "BoundedArray",
    "SegmentedList",
    "DoublyLinkedList",
    "SinglyLinkedList",
    "TailQueue",
    "PriorityQueue",
    "PriorityDequeue",
    "Treap",
    "RingBuffer",
    "BitSet",
    "DynamicBitSet",
    "StaticBitSet",
}

# Memory/allocator types
ZIG_STD_ALLOCATORS: Set[str] = {
    "Allocator",
    "GeneralPurposeAllocator",
    "ArenaAllocator",
    "FixedBufferAllocator",
    "StackFallbackAllocator",
    "LoggingAllocator",
    "page_allocator",
    "c_allocator",
    "raw_c_allocator",
    "allocator",
    "failing_allocator",
}

# IO types
ZIG_STD_IO: Set[str] = {
    "Reader",
    "Writer",
    "SeekableStream",
    "AnyReader",
    "AnyWriter",
    "BufferedReader",
    "BufferedWriter",
    "FixedBufferStream",
    "getStdIn",
    "getStdOut",
    "getStdErr",
}

# Filesystem types
ZIG_STD_FS: Set[str] = {
    "File",
    "Dir",
    "IterableDir",
    "path",
    "cwd",
    "openFile",
    "openFileAbsolute",
    "openDirAbsolute",
    "accessAbsolute",
    "selfExePath",
}

# Memory functions
ZIG_STD_MEM: Set[str] = {
    "eql",
    "copy",
    "set",
    "indexOf",
    "lastIndexOf",
    "replace",
    "split",
    "tokenize",
    "trim",
    "trimLeft",
    "trimRight",
    "span",
    "sliceTo",
    "alignForward",
    "alignBackward",
    "zeroes",
}

# Format functions
ZIG_STD_FMT: Set[str] = {
    "print",
    "format",
    "formatIntValue",
    "fmtSliceEscapeUpper",
    "fmtSliceHexLower",
    "bufPrint",
    "allocPrint",
    "comptimePrint",
    "parseInt",
    "parseUnsigned",
}

# Debug functions
ZIG_STD_DEBUG: Set[str] = {
    "print",
    "panic",
    "assert",
    "dumpCurrentStackTrace",
    "dumpStackTrace",
    "getStackTrace",
    "getSelfDebugInfo",
}

# Testing functions
ZIG_STD_TESTING: Set[str] = {
    "expect",
    "expectEqual",
    "expectEqualSlices",
    "expectEqualStrings",
    "expectError",
    "expectApproxEqAbs",
    "expectApproxEqRel",
    "allocator",
    "failing_allocator",
}

# Heap types
ZIG_STD_HEAP: Set[str] = {
    "GeneralPurposeAllocator",
    "ArenaAllocator",
    "FixedBufferAllocator",
    "StackFallbackAllocator",
    "LoggingAllocator",
    "page_allocator",
    "c_allocator",
    "raw_c_allocator",
}

# Math functions
ZIG_STD_MATH: Set[str] = {
    "add",
    "sub",
    "mul",
    "div",
    "min",
    "max",
    "clamp",
    "sqrt",
    "pow",
    "log",
    "sin",
    "cos",
    "tan",
    "exp",
    "floor",
    "ceil",
    "round",
}

# All known std symbols
ZIG_ALL_STD: Set[str] = (
    ZIG_STD_MODULES |
    ZIG_STD_DATA_STRUCTURES |
    ZIG_STD_ALLOCATORS |
    ZIG_STD_IO |
    ZIG_STD_FS
)


# =============================================================================
# build.zig.zon Parser (Simplified)
# =============================================================================

@dataclass
class ZonDependency:
    """A dependency from build.zig.zon."""
    name: str
    url: Optional[str] = None
    hash: Optional[str] = None
    path: Optional[str] = None


def parse_build_zig_zon(zon_content: str) -> Dict[str, ZonDependency]:
    """Parse build.zig.zon content to extract dependencies.

    This is a simplified parser - Zon is actually Zig syntax
    and would ideally be parsed with a proper Zig parser.

    Args:
        zon_content: The content of build.zig.zon

    Returns:
        Dictionary mapping dependency names to ZonDependency objects
    """
    deps: Dict[str, ZonDependency] = {}

    # Look for .dependencies = .{ ... }
    deps_match = re.search(r"\.dependencies\s*=\s*\.{([^}]*)}", zon_content, re.DOTALL)
    if not deps_match:
        return deps

    deps_content = deps_match.group(1)

    # Parse each dependency entry
    # Pattern: .name = .{ .url = "...", .hash = "..." } or .path = "..."
    dep_pattern = re.compile(
        r'\.(\w+)\s*=\s*\.{([^}]*)}',
        re.DOTALL
    )

    for match in dep_pattern.finditer(deps_content):
        name = match.group(1)
        content = match.group(2)

        url_match = re.search(r'\.url\s*=\s*"([^"]*)"', content)
        hash_match = re.search(r'\.hash\s*=\s*"([^"]*)"', content)
        path_match = re.search(r'\.path\s*=\s*"([^"]*)"', content)

        deps[name] = ZonDependency(
            name=name,
            url=url_match.group(1) if url_match else None,
            hash=hash_match.group(1) if hash_match else None,
            path=path_match.group(1) if path_match else None,
        )

    return deps


# =============================================================================
# Zig Import Resolver
# =============================================================================

class ZigImportResolver(ImportResolver):
    """Import resolver for Zig's @import builtin.

    Resolves imports in the following order:
    1. Standard library (@import("std"))
    2. build.zig.zon dependencies (@import("package"))
    3. Local files (@import("file.zig"))

    Example:
        >>> resolver = ZigImportResolver(project_root="/path/to/project")
        >>> result = resolver.resolve("std")
        >>> result.success
        True

        >>> result = resolver.resolve("my_local.zig")
        >>> # Checks if /path/to/project/my_local.zig exists
    """

    def __init__(
        self,
        project_root: Optional[str] = None,
        zon_path: Optional[str] = None,
        zig_lib_dir: Optional[str] = None,
    ) -> None:
        """Initialize the Zig import resolver.

        Args:
            project_root: Root directory of the Zig project
            zon_path: Path to build.zig.zon (defaults to project_root/build.zig.zon)
            zig_lib_dir: Path to Zig standard library (for verification)
        """
        self._project_root = Path(project_root) if project_root else None
        self._zig_lib_dir = Path(zig_lib_dir) if zig_lib_dir else None

        # Determine zon path
        if zon_path:
            self._zon_path = Path(zon_path)
        elif self._project_root:
            self._zon_path = self._project_root / "build.zig.zon"
        else:
            self._zon_path = None

        # Parse dependencies from build.zig.zon
        self._zon_deps: Dict[str, ZonDependency] = {}
        if self._zon_path and self._zon_path.exists():
            try:
                zon_content = self._zon_path.read_text()
                self._zon_deps = parse_build_zig_zon(zon_content)
            except Exception:
                pass  # Ignore parse errors

        # Cache for resolved modules
        self._cache: Dict[str, ImportResolution] = {}

    @property
    def language(self) -> str:
        return "zig"

    @property
    def project_root(self) -> Optional[str]:
        """Return the project root path as a string."""
        return str(self._project_root) if self._project_root else None

    def resolve(self, module_name: str) -> ImportResolution:
        """Resolve a Zig @import.

        Args:
            module_name: The import string (without quotes)

        Returns:
            ImportResolution with success/failure and module info
        """
        # Check cache
        if module_name in self._cache:
            return self._cache[module_name]

        # Strip quotes if present (from @import("name"))
        module_name = module_name.strip('"\'')

        result: ImportResolution

        # Standard library
        if module_name == "std" or module_name.startswith("std."):
            result = self._resolve_std(module_name)

        # build.zig.zon dependency
        elif module_name in self._zon_deps:
            result = self._resolve_zon_dependency(module_name)

        # Local .zig file
        elif module_name.endswith(".zig"):
            result = self._resolve_local_file(module_name)

        # Unknown - might be a package or file
        else:
            result = self._resolve_unknown(module_name)

        # Cache and return
        self._cache[module_name] = result
        return result

    def _resolve_std(self, module_name: str) -> ImportResolution:
        """Resolve a standard library import."""
        # Check if it's a known std module
        if module_name in ZIG_ALL_STD or module_name == "std":
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

        # Check if it could be a valid std submodule
        if module_name.startswith("std."):
            # Allow it - we can't enumerate all possible std paths
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
            error=f"Unknown standard library module: {module_name}",
            alternatives=self._suggest_std_alternatives(module_name),
        )

    def _resolve_zon_dependency(self, module_name: str) -> ImportResolution:
        """Resolve a build.zig.zon dependency."""
        dep = self._zon_deps[module_name]

        # Check if it's a path dependency (local)
        if dep.path and self._project_root:
            dep_path = self._project_root / dep.path
            is_available = dep_path.exists()
            return ImportResolution(
                status=ResolutionStatus.RESOLVED if is_available else ResolutionStatus.PARTIAL,
                success=True,
                module=ResolvedModule(
                    name=module_name,
                    path=str(dep_path) if is_available else None,
                    is_available=is_available,
                ),
                module_name=module_name,
            )

        # URL dependency - assume available if hash is present
        return ImportResolution(
            status=ResolutionStatus.PARTIAL,
            success=True,
            module=ResolvedModule(
                name=module_name,
                version=dep.hash[:12] if dep.hash else None,
                is_available=dep.hash is not None,
            ),
            module_name=module_name,
        )

    def _resolve_local_file(self, module_name: str) -> ImportResolution:
        """Resolve a local .zig file import."""
        if not self._project_root:
            # Can't verify without project root, assume valid
            return ImportResolution(
                status=ResolutionStatus.PARTIAL,
                success=True,
                module=ResolvedModule(
                    name=module_name,
                    is_available=True,  # Optimistic
                ),
                module_name=module_name,
            )

        # Check if file exists
        file_path = self._project_root / module_name
        if file_path.exists():
            return ImportResolution(
                status=ResolutionStatus.RESOLVED,
                success=True,
                module=ResolvedModule(
                    name=module_name,
                    path=str(file_path),
                    is_available=True,
                ),
                module_name=module_name,
            )

        # Try src/ subdirectory
        src_path = self._project_root / "src" / module_name
        if src_path.exists():
            return ImportResolution(
                status=ResolutionStatus.RESOLVED,
                success=True,
                module=ResolvedModule(
                    name=module_name,
                    path=str(src_path),
                    is_available=True,
                ),
                module_name=module_name,
            )

        return ImportResolution(
            status=ResolutionStatus.FAILED,
            success=False,
            module_name=module_name,
            error=f"Local file not found: {module_name}",
            alternatives=self._suggest_local_alternatives(module_name),
        )

    def _resolve_unknown(self, module_name: str) -> ImportResolution:
        """Resolve an unknown import (might be package or file)."""
        # Check build.zig for any reference
        if self._project_root:
            build_zig = self._project_root / "build.zig"
            if build_zig.exists():
                try:
                    content = build_zig.read_text()
                    if f'"{module_name}"' in content:
                        # Referenced in build.zig, assume valid
                        return ImportResolution(
                            status=ResolutionStatus.PARTIAL,
                            success=True,
                            module=ResolvedModule(
                                name=module_name,
                                is_available=True,
                            ),
                            module_name=module_name,
                        )
                except Exception:
                    pass

        # Unknown - provide helpful error
        suggestions = []
        if self._zon_deps:
            suggestions = [d for d in self._zon_deps.keys()
                         if d.startswith(module_name[:3])]

        return ImportResolution(
            status=ResolutionStatus.FAILED,
            success=False,
            module_name=module_name,
            error=f"Unknown import: {module_name}. "
                  f"Not found in std, build.zig.zon, or as local file.",
            alternatives=suggestions,
        )

    def is_available(self, module_name: str) -> bool:
        """Check if a module is available for import."""
        result = self.resolve(module_name)
        return result.success and (result.module is None or result.module.is_available)

    def get_version(self, module_name: str) -> Optional[str]:
        """Get the version of a module.

        For Zig, this returns the hash for zon dependencies.
        """
        if module_name in self._zon_deps:
            dep = self._zon_deps[module_name]
            return dep.hash[:12] if dep.hash else None

        # Std library - return Zig version if known
        if module_name.startswith("std"):
            # Could detect from `zig version` if needed
            return None

        return None

    def get_exports(self, module_name: str) -> Set[str]:
        """Get names exported by a module.

        For Zig, this would require parsing the module file.
        Returns empty set for now.
        """
        # Would need to parse the .zig file to get exports
        return set()

    def suggest_alternatives(self, module_name: str) -> List[str]:
        """Suggest alternative modules."""
        suggestions = []

        # Check std alternatives
        suggestions.extend(self._suggest_std_alternatives(module_name))

        # Check zon dependency alternatives
        for dep_name in self._zon_deps:
            if module_name.lower() in dep_name.lower():
                suggestions.append(dep_name)

        return suggestions[:5]  # Limit to 5 suggestions

    def _suggest_std_alternatives(self, module_name: str) -> List[str]:
        """Suggest standard library alternatives."""
        suggestions = []
        module_lower = module_name.lower()

        for std_module in ZIG_ALL_STD:
            # Check for prefix match
            if std_module.lower().startswith(module_lower):
                suggestions.append(std_module)
            # Check for substring match
            elif module_lower in std_module.lower():
                suggestions.append(std_module)

        return sorted(suggestions)[:5]

    def _suggest_local_alternatives(self, module_name: str) -> List[str]:
        """Suggest local file alternatives."""
        if not self._project_root:
            return []

        suggestions = []
        base_name = module_name.replace(".zig", "")

        # Search for similar .zig files
        for zig_file in self._project_root.glob("**/*.zig"):
            file_name = zig_file.stem
            if base_name.lower() in file_name.lower():
                rel_path = zig_file.relative_to(self._project_root)
                suggestions.append(str(rel_path))

        return suggestions[:5]

    def get_zon_dependencies(self) -> Dict[str, ZonDependency]:
        """Get all dependencies from build.zig.zon."""
        return self._zon_deps.copy()

    def refresh(self) -> None:
        """Refresh the resolver by re-reading build.zig.zon."""
        self._cache.clear()

        if self._zon_path and self._zon_path.exists():
            try:
                zon_content = self._zon_path.read_text()
                self._zon_deps = parse_build_zig_zon(zon_content)
            except Exception:
                self._zon_deps = {}


def create_zig_resolver(
    project_root: Optional[str] = None,
) -> ZigImportResolver:
    """Create a Zig import resolver with auto-detection.

    Args:
        project_root: Optional project root (auto-detected if not provided)

    Returns:
        Configured ZigImportResolver
    """
    # Try to auto-detect project root
    if project_root is None:
        cwd = Path.cwd()
        # Look for build.zig or build.zig.zon
        for parent in [cwd] + list(cwd.parents):
            if (parent / "build.zig").exists() or (parent / "build.zig.zon").exists():
                project_root = str(parent)
                break

    return ZigImportResolver(project_root=project_root)


def extract_zig_exports(zig_source: str) -> Set[str]:
    """Extract public exports from Zig source code.

    This is a simplified parser that looks for pub declarations.

    Args:
        zig_source: Zig source code string

    Returns:
        Set of exported names (functions, types, variables)
    """
    exports: Set[str] = set()

    # Simple regex patterns for pub declarations
    patterns = [
        # pub fn name(
        r'\bpub\s+fn\s+(\w+)\s*\(',
        # pub const name : type = or pub const name = value
        r'\bpub\s+const\s+(\w+)\s*[=:]',
        # pub var name =
        r'\bpub\s+var\s+(\w+)\s*[=:]',
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, zig_source):
            exports.add(match.group(1))

    return exports
