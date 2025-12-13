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
"""Environment capture for typed holes.

Captures the typing environment at a hole site, enabling:
- Fill-and-resume with correct type context
- Scope-aware code generation
- Variable suggestions

Following Hazel's approach, the captured environment includes
all bindings visible at the hole position.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set

from .hole import TypeEnvironment, SourceLocation


@dataclass(frozen=True)
class CapturedBinding:
    """A captured variable binding.

    Attributes:
        name: Variable name
        typ: Variable type
        source: Where the binding was defined
        is_parameter: True if this is a function parameter
        is_import: True if this is an imported name
        is_global: True if this is a global binding
    """

    name: str
    typ: Any
    source: Optional[SourceLocation] = None
    is_parameter: bool = False
    is_import: bool = False
    is_global: bool = False

    def __str__(self) -> str:
        return f"{self.name}: {self.typ}"


@dataclass(frozen=True)
class ScopeInfo:
    """Information about a lexical scope.

    Attributes:
        name: Scope name (e.g., function name, class name)
        kind: Scope kind (e.g., "function", "class", "module")
        bindings: Bindings introduced in this scope
        parent: Parent scope (if any)
    """

    name: str
    kind: str
    bindings: FrozenSet[CapturedBinding] = field(default_factory=frozenset)
    parent: Optional[ScopeInfo] = None

    @property
    def depth(self) -> int:
        """Get the nesting depth of this scope."""
        if self.parent is None:
            return 0
        return 1 + self.parent.depth

    def all_bindings(self) -> FrozenSet[CapturedBinding]:
        """Get all bindings including from parent scopes."""
        result = set(self.bindings)
        if self.parent:
            result.update(self.parent.all_bindings())
        return frozenset(result)


@dataclass
class EnvironmentCapture:
    """Captured typing environment at a code location.

    Provides a rich view of the typing environment including:
    - Local variables in scope
    - Function parameters
    - Imported names
    - Global bindings
    - Scope structure

    Attributes:
        bindings: All captured bindings
        scope: Current scope info
        position: Position where capture was made
    """

    bindings: Dict[str, CapturedBinding] = field(default_factory=dict)
    scope: Optional[ScopeInfo] = None
    position: Optional[SourceLocation] = None

    def add_binding(self, binding: CapturedBinding) -> None:
        """Add a binding to the capture.

        Args:
            binding: The binding to add
        """
        self.bindings[binding.name] = binding

    def lookup(self, name: str) -> Optional[CapturedBinding]:
        """Look up a binding by name.

        Args:
            name: The name to look up

        Returns:
            The binding if found, None otherwise
        """
        return self.bindings.get(name)

    def get_type(self, name: str) -> Optional[Any]:
        """Get the type of a name.

        Args:
            name: The name to look up

        Returns:
            The type if found, None otherwise
        """
        binding = self.bindings.get(name)
        return binding.typ if binding else None

    def names(self) -> Set[str]:
        """Get all captured names."""
        return set(self.bindings.keys())

    def local_names(self) -> Set[str]:
        """Get names from local scope only."""
        return {
            name for name, binding in self.bindings.items()
            if not binding.is_global and not binding.is_import
        }

    def parameter_names(self) -> Set[str]:
        """Get names that are function parameters."""
        return {
            name for name, binding in self.bindings.items()
            if binding.is_parameter
        }

    def imported_names(self) -> Set[str]:
        """Get names that are imported."""
        return {
            name for name, binding in self.bindings.items()
            if binding.is_import
        }

    def global_names(self) -> Set[str]:
        """Get names from global scope."""
        return {
            name for name, binding in self.bindings.items()
            if binding.is_global
        }

    def to_type_environment(self) -> TypeEnvironment:
        """Convert to a TypeEnvironment.

        Returns:
            TypeEnvironment with all bindings
        """
        binding_pairs = frozenset(
            (name, binding.typ)
            for name, binding in self.bindings.items()
        )
        return TypeEnvironment(bindings=binding_pairs)

    def filter_by_type(self, type_predicate: Any) -> List[CapturedBinding]:
        """Get bindings matching a type predicate.

        Args:
            type_predicate: A type or callable to match

        Returns:
            List of matching bindings
        """
        if callable(type_predicate):
            return [
                b for b in self.bindings.values()
                if type_predicate(b.typ)
            ]
        else:
            return [
                b for b in self.bindings.values()
                if b.typ == type_predicate
            ]

    @classmethod
    def empty(cls) -> EnvironmentCapture:
        """Create an empty capture."""
        return cls()

    @classmethod
    def from_dict(
        cls,
        bindings: Dict[str, Any],
        scope: Optional[ScopeInfo] = None,
    ) -> EnvironmentCapture:
        """Create a capture from a dictionary of name->type.

        Args:
            bindings: Dictionary of name to type
            scope: Optional scope info

        Returns:
            New EnvironmentCapture
        """
        capture = cls(scope=scope)
        for name, typ in bindings.items():
            capture.add_binding(CapturedBinding(name=name, typ=typ))
        return capture


class EnvironmentCapturer:
    """Captures typing environments from code.

    Extracts type bindings from:
    - Variable assignments
    - Function parameters
    - Import statements
    - Class definitions

    This is a framework class - language-specific implementations
    would subclass this.
    """

    def __init__(self) -> None:
        """Initialize the capturer."""
        self._global_bindings: Dict[str, CapturedBinding] = {}
        self._current_scope: Optional[ScopeInfo] = None

    def capture_at(
        self,
        source: str,
        position: SourceLocation,
    ) -> EnvironmentCapture:
        """Capture environment at a position in source code.

        Args:
            source: Source code text
            position: Position to capture at

        Returns:
            Captured environment
        """
        # Basic implementation - subclasses would parse the source
        capture = EnvironmentCapture(
            scope=self._current_scope,
            position=position,
        )

        # Add global bindings
        for binding in self._global_bindings.values():
            capture.add_binding(binding)

        return capture

    def add_global(self, name: str, typ: Any) -> None:
        """Add a global binding.

        Args:
            name: Name to bind
            typ: Type of the binding
        """
        self._global_bindings[name] = CapturedBinding(
            name=name,
            typ=typ,
            is_global=True,
        )

    def enter_scope(self, name: str, kind: str) -> None:
        """Enter a new scope.

        Args:
            name: Name of the scope
            kind: Kind of scope
        """
        self._current_scope = ScopeInfo(
            name=name,
            kind=kind,
            parent=self._current_scope,
        )

    def exit_scope(self) -> None:
        """Exit the current scope."""
        if self._current_scope and self._current_scope.parent:
            self._current_scope = self._current_scope.parent
        else:
            self._current_scope = None

    def reset(self) -> None:
        """Reset the capturer state."""
        self._global_bindings.clear()
        self._current_scope = None


def capture_environment(
    bindings: Dict[str, Any],
    *,
    position: Optional[SourceLocation] = None,
    scope_name: Optional[str] = None,
    scope_kind: str = "local",
) -> EnvironmentCapture:
    """Convenience function to capture an environment.

    Args:
        bindings: Dictionary of name to type
        position: Optional source position
        scope_name: Optional scope name
        scope_kind: Kind of scope

    Returns:
        Captured environment
    """
    scope = None
    if scope_name:
        scope = ScopeInfo(name=scope_name, kind=scope_kind)

    capture = EnvironmentCapture(scope=scope, position=position)
    for name, typ in bindings.items():
        capture.add_binding(CapturedBinding(name=name, typ=typ))

    return capture


def merge_captures(
    *captures: EnvironmentCapture,
) -> EnvironmentCapture:
    """Merge multiple environment captures.

    Later captures take precedence for duplicate names.

    Args:
        captures: Captures to merge

    Returns:
        Merged capture
    """
    result = EnvironmentCapture()

    for capture in captures:
        for binding in capture.bindings.values():
            result.add_binding(binding)

    # Use the scope from the last capture with a scope
    for capture in reversed(captures):
        if capture.scope:
            result.scope = capture.scope
            break

    return result
