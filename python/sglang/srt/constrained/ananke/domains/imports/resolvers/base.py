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
"""Base class for import resolvers.

Import resolvers are responsible for:
1. Checking if a module exists and is importable
2. Resolving module versions
3. Providing type information for imported modules
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Set


class ResolutionStatus(Enum):
    """Status of import resolution.

    Attributes:
        RESOLVED: Module fully resolved and available
        PARTIAL: Some information resolved but not complete (e.g., version unknown)
        FAILED: Module could not be resolved
        UNKNOWN: Resolution status not determined
    """

    RESOLVED = auto()
    PARTIAL = auto()
    FAILED = auto()
    UNKNOWN = auto()


@dataclass
class ResolvedModule:
    """Information about a resolved module.

    Attributes:
        name: Full module name
        version: Module version (if known)
        path: File path (if known)
        exports: Names exported by the module
        is_builtin: Whether this is a built-in module
        is_available: Whether the module is importable
    """

    name: str
    version: Optional[str] = None
    path: Optional[str] = None
    exports: Set[str] = None
    is_builtin: bool = False
    is_available: bool = True

    def __post_init__(self):
        if self.exports is None:
            self.exports = set()


@dataclass
class ImportResolution:
    """Result of resolving an import.

    Attributes:
        status: Resolution status (RESOLVED, PARTIAL, FAILED, UNKNOWN)
        success: Whether resolution succeeded (for backward compatibility)
        module: The resolved module (if successful)
        module_name: Name of the module being resolved
        error: Error message (if failed)
        alternatives: Alternative modules that might work
        exports: Set of exported names (if known)
    """

    status: ResolutionStatus = ResolutionStatus.UNKNOWN
    success: bool = False
    module: Optional[ResolvedModule] = None
    module_name: Optional[str] = None
    error: Optional[str] = None
    alternatives: List[str] = None
    exports: Set[str] = None

    def __post_init__(self):
        if self.alternatives is None:
            self.alternatives = []
        if self.exports is None:
            self.exports = set()
        # Synchronize success with status
        if self.status == ResolutionStatus.RESOLVED or self.status == ResolutionStatus.PARTIAL:
            self.success = True
        elif self.status == ResolutionStatus.FAILED:
            self.success = False


class ImportResolver(ABC):
    """Abstract base class for import resolvers.

    Import resolvers determine whether a module can be imported
    in a particular language/environment.

    Subclasses implement language-specific resolution:
    - PythonImportResolver: pip packages, stdlib
    - TypeScriptImportResolver: npm packages
    - RustImportResolver: cargo crates
    - etc.
    """

    @property
    @abstractmethod
    def language(self) -> str:
        """Return the language this resolver handles."""
        pass

    @abstractmethod
    def resolve(self, module_name: str) -> ImportResolution:
        """Resolve a module by name.

        Args:
            module_name: The module to resolve

        Returns:
            ImportResolution with success/failure and module info
        """
        pass

    @abstractmethod
    def is_available(self, module_name: str) -> bool:
        """Check if a module is available for import.

        Args:
            module_name: Module name to check

        Returns:
            True if the module can be imported
        """
        pass

    @abstractmethod
    def get_version(self, module_name: str) -> Optional[str]:
        """Get the installed version of a module.

        Args:
            module_name: Module name

        Returns:
            Version string, or None if unknown
        """
        pass

    def get_exports(self, module_name: str) -> Set[str]:
        """Get names exported by a module.

        Default implementation returns empty set.
        Subclasses may override for better type support.

        Args:
            module_name: Module name

        Returns:
            Set of exported names
        """
        return set()

    def suggest_alternatives(self, module_name: str) -> List[str]:
        """Suggest alternative modules if one isn't found.

        Default implementation returns empty list.

        Args:
            module_name: Module that wasn't found

        Returns:
            List of alternative module names
        """
        return []


class PassthroughResolver(ImportResolver):
    """Resolver that always succeeds (for testing/development).

    All modules are considered available.
    """

    def __init__(self, language: str = "python"):
        self._language = language

    @property
    def language(self) -> str:
        return self._language

    def resolve(self, module_name: str) -> ImportResolution:
        """Always successfully resolves."""
        return ImportResolution(
            status=ResolutionStatus.RESOLVED,
            success=True,
            module=ResolvedModule(name=module_name, is_available=True),
            module_name=module_name,
        )

    def is_available(self, module_name: str) -> bool:
        """Always returns True."""
        return True

    def get_version(self, module_name: str) -> Optional[str]:
        """Returns None (unknown version)."""
        return None


class DenyListResolver(ImportResolver):
    """Resolver that denies specific modules.

    Used to enforce security policies.
    """

    def __init__(
        self,
        language: str = "python",
        denied: Optional[Set[str]] = None,
        fallback: Optional[ImportResolver] = None,
    ):
        """Initialize with denied modules.

        Args:
            language: Target language
            denied: Set of denied module names
            fallback: Resolver to use for non-denied modules
        """
        self._language = language
        self._denied = denied or set()
        self._fallback = fallback or PassthroughResolver(language)

    @property
    def language(self) -> str:
        return self._language

    def resolve(self, module_name: str) -> ImportResolution:
        """Resolve, denying specific modules."""
        if module_name in self._denied:
            return ImportResolution(
                status=ResolutionStatus.FAILED,
                success=False,
                module_name=module_name,
                error=f"Module '{module_name}' is not allowed",
            )
        return self._fallback.resolve(module_name)

    def is_available(self, module_name: str) -> bool:
        """Check availability, respecting deny list."""
        if module_name in self._denied:
            return False
        return self._fallback.is_available(module_name)

    def get_version(self, module_name: str) -> Optional[str]:
        """Get version via fallback."""
        if module_name in self._denied:
            return None
        return self._fallback.get_version(module_name)
