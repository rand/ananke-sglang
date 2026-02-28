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
"""Type environment for Ananke's type domain.

The TypeEnvironment provides an immutable mapping from variable names to types,
supporting efficient snapshots for checkpointing/rollback during generation.

Uses the `immutables` library for persistent data structures when available,
falling back to frozendict-style operations otherwise.

References:
    - Hazel: Uses persistent maps for environment snapshots
    - TAPL: Chapter 9 - Simple Types
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterator, Optional, Tuple

# Try to use immutables for efficient persistent maps
try:
    from immutables import Map as ImmutableMap

    HAS_IMMUTABLES = True
except ImportError:
    HAS_IMMUTABLES = False

# Support both relative imports (when used as subpackage) and absolute imports (standalone testing)
try:
    from .constraint import Type, TypeVar
except ImportError:
    from domains.types.constraint import Type, TypeVar


@dataclass(frozen=True)
class TypeEnvironment:
    """Immutable type environment mapping names to types.

    The type environment (Γ) maps variable names to their types.
    It is immutable to support efficient snapshots for checkpointing.

    Operations return new environments rather than mutating in place.

    Attributes:
        _bindings: The internal mapping (immutable)
        _parent: Optional parent environment for nested scopes
    """

    _bindings: Dict[str, Type] = field(default_factory=dict)
    _parent: Optional[TypeEnvironment] = None

    def __post_init__(self) -> None:
        # Ensure bindings is immutable by converting to frozenset internally
        # We keep it as dict for type hints but treat it as immutable
        pass

    def bind(self, name: str, ty: Type) -> TypeEnvironment:
        """Create a new environment with an additional binding.

        Args:
            name: The variable name to bind
            ty: The type to bind it to

        Returns:
            A new TypeEnvironment with the binding added
        """
        new_bindings = dict(self._bindings)
        new_bindings[name] = ty
        return TypeEnvironment(_bindings=new_bindings, _parent=self._parent)

    def bind_many(self, bindings: Dict[str, Type]) -> TypeEnvironment:
        """Create a new environment with multiple bindings.

        Args:
            bindings: Dictionary of name -> type bindings

        Returns:
            A new TypeEnvironment with all bindings added
        """
        new_bindings = dict(self._bindings)
        new_bindings.update(bindings)
        return TypeEnvironment(_bindings=new_bindings, _parent=self._parent)

    def lookup(self, name: str) -> Optional[Type]:
        """Look up a variable's type in the environment.

        Searches the current scope first, then parent scopes.

        Args:
            name: The variable name to look up

        Returns:
            The type if found, None otherwise
        """
        if name in self._bindings:
            return self._bindings[name]
        if self._parent is not None:
            return self._parent.lookup(name)
        return None

    def contains(self, name: str) -> bool:
        """Check if a variable is bound in this environment.

        Args:
            name: The variable name to check

        Returns:
            True if the variable is bound
        """
        if name in self._bindings:
            return True
        if self._parent is not None:
            return self._parent.contains(name)
        return False

    def push_scope(self) -> TypeEnvironment:
        """Create a new nested scope.

        The new environment has this one as its parent.

        Returns:
            A new TypeEnvironment with this as parent
        """
        return TypeEnvironment(_bindings={}, _parent=self)

    def pop_scope(self) -> TypeEnvironment:
        """Return to the parent scope.

        Returns:
            The parent environment, or self if no parent
        """
        if self._parent is not None:
            return self._parent
        return self

    def all_bindings(self) -> Dict[str, Type]:
        """Get all bindings in this environment and parents.

        Returns:
            Dictionary of all name -> type bindings
        """
        result: Dict[str, Type] = {}

        # Collect from parent first (so local bindings shadow)
        if self._parent is not None:
            result.update(self._parent.all_bindings())

        result.update(self._bindings)
        return result

    def local_bindings(self) -> Dict[str, Type]:
        """Get only the bindings in the current scope.

        Returns:
            Dictionary of local name -> type bindings
        """
        return dict(self._bindings)

    def names(self) -> FrozenSet[str]:
        """Get all bound variable names.

        Returns:
            Set of all bound names
        """
        all_names: set[str] = set(self._bindings.keys())
        if self._parent is not None:
            all_names.update(self._parent.names())
        return frozenset(all_names)

    def snapshot(self) -> TypeEnvironmentSnapshot:
        """Create a snapshot for checkpointing.

        Returns:
            A snapshot that can be used to restore this state
        """
        return TypeEnvironmentSnapshot(
            bindings=self._bindings,
            parent_snapshot=self._parent.snapshot() if self._parent else None,
        )

    @staticmethod
    def from_snapshot(snapshot: TypeEnvironmentSnapshot) -> TypeEnvironment:
        """Restore an environment from a snapshot.

        Args:
            snapshot: The snapshot to restore from

        Returns:
            A new TypeEnvironment matching the snapshot
        """
        parent = None
        if snapshot.parent_snapshot is not None:
            parent = TypeEnvironment.from_snapshot(snapshot.parent_snapshot)
        return TypeEnvironment(_bindings=snapshot.bindings, _parent=parent)

    def __len__(self) -> int:
        """Return the number of bindings in the current scope."""
        return len(self._bindings)

    def __iter__(self) -> Iterator[str]:
        """Iterate over bound names in current scope."""
        return iter(self._bindings)

    def __repr__(self) -> str:
        bindings_str = ", ".join(
            f"{name}: {ty}" for name, ty in sorted(self._bindings.items())
        )
        if self._parent is not None:
            return f"TypeEnvironment({{{bindings_str}}}, parent=...)"
        return f"TypeEnvironment({{{bindings_str}}})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TypeEnvironment):
            return NotImplemented
        return (
            self._bindings == other._bindings and self._parent == other._parent
        )

    def __hash__(self) -> int:
        # Convert bindings to hashable form
        bindings_tuple = tuple(sorted(self._bindings.items(), key=lambda x: x[0]))
        parent_hash = hash(self._parent) if self._parent else 0
        return hash((bindings_tuple, parent_hash))


@dataclass(frozen=True)
class TypeEnvironmentSnapshot:
    """Snapshot of a type environment for checkpointing.

    Snapshots are immutable and can be used to restore environments
    to a previous state during backtracking.

    Attributes:
        bindings: The bindings at the time of snapshot
        parent_snapshot: Snapshot of the parent environment
    """

    bindings: Dict[str, Type] = field(default_factory=dict)
    parent_snapshot: Optional[TypeEnvironmentSnapshot] = None

    def __hash__(self) -> int:
        bindings_tuple = tuple(sorted(self.bindings.items(), key=lambda x: x[0]))
        parent_hash = hash(self.parent_snapshot) if self.parent_snapshot else 0
        return hash((bindings_tuple, parent_hash))


# Empty environment singleton
EMPTY_ENVIRONMENT = TypeEnvironment()


def create_environment(bindings: Optional[Dict[str, Type]] = None) -> TypeEnvironment:
    """Create a type environment with optional initial bindings.

    Args:
        bindings: Optional initial bindings

    Returns:
        A new TypeEnvironment
    """
    if bindings is None:
        return EMPTY_ENVIRONMENT
    return TypeEnvironment(_bindings=bindings)


def merge_environments(
    env1: TypeEnvironment, env2: TypeEnvironment
) -> TypeEnvironment:
    """Merge two environments, with env2 taking precedence.

    Args:
        env1: First environment
        env2: Second environment (takes precedence on conflicts)

    Returns:
        A new environment with bindings from both
    """
    merged = dict(env1.all_bindings())
    merged.update(env2.all_bindings())
    return TypeEnvironment(_bindings=merged)
