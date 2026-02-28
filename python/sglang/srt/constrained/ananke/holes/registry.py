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
"""Registry for managing typed holes.

The HoleRegistry tracks all holes during generation, maintaining:
- Parent-child relationships for nested holes
- Fill/unfill history for backtracking
- Hole selection ordering

Following Hazel's fill-and-resume semantics, holes can be:
- Created dynamically during parsing
- Filled with code
- Unfilled for backtracking
- Nested hierarchically
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generic, Iterator, List, Optional, Set, TypeVar

from .hole import Hole, HoleId, HoleState, TypeEnvironment, HoleGranularity

# Type variable for constraint types
C = TypeVar("C")


@dataclass
class RegistryCheckpoint(Generic[C]):
    """Checkpoint for registry state.

    Enables efficient backtracking by storing:
    - All holes at checkpoint time
    - Fill history up to checkpoint
    - Counter state

    Attributes:
        holes: Copy of all holes
        fill_history: History of fills
        next_index: Next hole index counter
    """

    holes: Dict[HoleId, Hole[C]]
    fill_history: List[tuple[HoleId, Optional[str]]]
    next_index: int


class HoleRegistry(Generic[C]):
    """Registry for managing typed holes.

    The registry maintains a collection of holes with:
    - O(1) lookup by ID
    - Parent-child relationship tracking
    - Fill/unfill history for backtracking
    - Automatic index assignment

    Example:
        >>> registry = HoleRegistry()
        >>> hole = registry.create("body", expected_type=int)
        >>> registry.fill(hole.id, "return 42")
        >>> registry.unfill(hole.id)  # Backtrack
    """

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._holes: Dict[HoleId, Hole[C]] = {}
        self._children: Dict[HoleId, Set[HoleId]] = {}
        self._fill_history: List[tuple[HoleId, Optional[str]]] = []
        self._next_index: int = 0

    @property
    def count(self) -> int:
        """Get the number of registered holes."""
        return len(self._holes)

    @property
    def empty_count(self) -> int:
        """Get the number of empty (unfilled) holes."""
        return sum(1 for h in self._holes.values() if h.is_empty)

    @property
    def filled_count(self) -> int:
        """Get the number of filled holes."""
        return sum(1 for h in self._holes.values() if h.is_filled)

    def create(
        self,
        name: str = "hole",
        *,
        namespace: str = "default",
        expected_type: Optional[Any] = None,
        environment: Optional[TypeEnvironment] = None,
        constraint: Optional[C] = None,
        granularity: HoleGranularity = HoleGranularity.EXPRESSION,
        parent: Optional[HoleId] = None,
    ) -> Hole[C]:
        """Create and register a new hole.

        Args:
            name: Descriptive name for the hole
            namespace: Organizational namespace
            expected_type: Expected type for filled code
            environment: Captured typing environment
            constraint: Domain-specific constraint
            granularity: Code granularity level
            parent: Parent hole ID (for nested holes)

        Returns:
            New registered Hole
        """
        # Determine depth from parent
        depth = 0
        if parent and parent in self._holes:
            depth = self._holes[parent].depth + 1

        # Create hole ID with auto-incremented index
        hole_id = HoleId(
            namespace=namespace,
            name=name,
            index=self._next_index,
            depth=depth,
        )
        self._next_index += 1

        # Create hole
        hole: Hole[C] = Hole(
            id=hole_id,
            expected_type=expected_type,
            environment=environment or TypeEnvironment.empty(),
            constraint=constraint,
            granularity=granularity,
            parent=parent,
        )

        # Register hole
        self._holes[hole_id] = hole

        # Track parent-child relationship
        if parent:
            if parent not in self._children:
                self._children[parent] = set()
            self._children[parent].add(hole_id)

        return hole

    def register(self, hole: Hole[C]) -> Hole[C]:
        """Register an existing hole.

        Args:
            hole: The hole to register

        Returns:
            The registered hole
        """
        self._holes[hole.id] = hole

        # Track parent-child relationship
        if hole.parent:
            if hole.parent not in self._children:
                self._children[hole.parent] = set()
            self._children[hole.parent].add(hole.id)

        return hole

    def lookup(self, hole_id: HoleId) -> Optional[Hole[C]]:
        """Look up a hole by ID.

        Args:
            hole_id: The hole identifier

        Returns:
            The hole if found, None otherwise
        """
        return self._holes.get(hole_id)

    def get(self, hole_id: HoleId) -> Hole[C]:
        """Get a hole by ID, raising if not found.

        Args:
            hole_id: The hole identifier

        Returns:
            The hole

        Raises:
            KeyError: If hole not found
        """
        if hole_id not in self._holes:
            raise KeyError(f"Hole not found: {hole_id}")
        return self._holes[hole_id]

    def contains(self, hole_id: HoleId) -> bool:
        """Check if a hole is registered.

        Args:
            hole_id: The hole identifier

        Returns:
            True if the hole is registered
        """
        return hole_id in self._holes

    def fill(self, hole_id: HoleId, content: str) -> Hole[C]:
        """Fill a hole with content.

        Records the fill in history for backtracking.

        Args:
            hole_id: The hole to fill
            content: The content to fill with

        Returns:
            The filled hole

        Raises:
            KeyError: If hole not found
        """
        hole = self.get(hole_id)

        # Record previous content in history
        self._fill_history.append((hole_id, hole.content))

        # Fill and update
        filled = hole.fill(content)
        self._holes[hole_id] = filled

        return filled

    def unfill(self, hole_id: HoleId) -> Hole[C]:
        """Unfill a hole, making it empty.

        Records the unfill in history for potential redo.

        Args:
            hole_id: The hole to unfill

        Returns:
            The unfilled hole

        Raises:
            KeyError: If hole not found
        """
        hole = self.get(hole_id)

        # Record current content in history
        self._fill_history.append((hole_id, hole.content))

        # Unfill and update
        unfilled = hole.unfill()
        self._holes[hole_id] = unfilled

        return unfilled

    def validate(self, hole_id: HoleId) -> Hole[C]:
        """Mark a hole as validated.

        Args:
            hole_id: The hole to validate

        Returns:
            The validated hole

        Raises:
            KeyError: If hole not found
        """
        hole = self.get(hole_id)
        validated = hole.validate()
        self._holes[hole_id] = validated
        return validated

    def invalidate(self, hole_id: HoleId) -> Hole[C]:
        """Mark a hole as invalid.

        Args:
            hole_id: The hole to invalidate

        Returns:
            The invalidated hole

        Raises:
            KeyError: If hole not found
        """
        hole = self.get(hole_id)
        invalidated = hole.invalidate()
        self._holes[hole_id] = invalidated
        return invalidated

    def update_constraint(self, hole_id: HoleId, constraint: C) -> Hole[C]:
        """Update a hole's constraint.

        Args:
            hole_id: The hole to update
            constraint: New constraint

        Returns:
            The updated hole

        Raises:
            KeyError: If hole not found
        """
        hole = self.get(hole_id)
        updated = hole.with_constraint(constraint)
        self._holes[hole_id] = updated
        return updated

    def update_type(self, hole_id: HoleId, expected_type: Any) -> Hole[C]:
        """Update a hole's expected type.

        Args:
            hole_id: The hole to update
            expected_type: New expected type

        Returns:
            The updated hole

        Raises:
            KeyError: If hole not found
        """
        hole = self.get(hole_id)
        updated = hole.with_type(expected_type)
        self._holes[hole_id] = updated
        return updated

    def children(self, hole_id: HoleId) -> Set[HoleId]:
        """Get the child holes of a hole.

        Args:
            hole_id: The parent hole ID

        Returns:
            Set of child hole IDs
        """
        return self._children.get(hole_id, set()).copy()

    def descendants(self, hole_id: HoleId) -> Set[HoleId]:
        """Get all descendant holes of a hole.

        Args:
            hole_id: The ancestor hole ID

        Returns:
            Set of all descendant hole IDs
        """
        result: Set[HoleId] = set()
        stack = list(self.children(hole_id))

        while stack:
            child_id = stack.pop()
            if child_id not in result:
                result.add(child_id)
                stack.extend(self.children(child_id))

        return result

    def ancestors(self, hole_id: HoleId) -> List[HoleId]:
        """Get all ancestor holes of a hole.

        Args:
            hole_id: The descendant hole ID

        Returns:
            List of ancestor hole IDs (immediate parent first)
        """
        result: List[HoleId] = []
        hole = self.lookup(hole_id)

        while hole and hole.parent:
            result.append(hole.parent)
            hole = self.lookup(hole.parent)

        return result

    def root_holes(self) -> Set[HoleId]:
        """Get all root (top-level) holes.

        Returns:
            Set of hole IDs with no parent
        """
        return {
            hole_id for hole_id, hole in self._holes.items()
            if hole.parent is None
        }

    def empty_holes(self) -> Iterator[Hole[C]]:
        """Iterate over all empty holes.

        Yields:
            Empty holes
        """
        for hole in self._holes.values():
            if hole.is_empty:
                yield hole

    def filled_holes(self) -> Iterator[Hole[C]]:
        """Iterate over all filled holes.

        Yields:
            Filled holes
        """
        for hole in self._holes.values():
            if hole.is_filled:
                yield hole

    def all_holes(self) -> Iterator[Hole[C]]:
        """Iterate over all holes.

        Yields:
            All registered holes
        """
        yield from self._holes.values()

    def next_hole(
        self,
        predicate: Optional[Callable[[Hole[C]], bool]] = None,
    ) -> Optional[Hole[C]]:
        """Get the next hole to fill.

        Default behavior: returns first empty hole by index order.
        Use HoleSelectionStrategy for more sophisticated ordering.

        Args:
            predicate: Optional filter predicate

        Returns:
            Next hole to fill, or None if all filled
        """
        candidates = [
            hole for hole in self._holes.values()
            if hole.is_empty and (predicate is None or predicate(hole))
        ]

        if not candidates:
            return None

        # Sort by index (creation order)
        candidates.sort(key=lambda h: h.id.index)
        return candidates[0]

    def checkpoint(self) -> RegistryCheckpoint[C]:
        """Create a checkpoint of current registry state.

        Returns:
            Checkpoint for later restoration
        """
        return RegistryCheckpoint(
            holes={k: v for k, v in self._holes.items()},
            fill_history=list(self._fill_history),
            next_index=self._next_index,
        )

    def restore(self, checkpoint: RegistryCheckpoint[C]) -> None:
        """Restore registry state from a checkpoint.

        Args:
            checkpoint: Previously created checkpoint
        """
        self._holes = {k: v for k, v in checkpoint.holes.items()}
        self._fill_history = list(checkpoint.fill_history)
        self._next_index = checkpoint.next_index

        # Rebuild parent-child relationships
        self._children.clear()
        for hole in self._holes.values():
            if hole.parent:
                if hole.parent not in self._children:
                    self._children[hole.parent] = set()
                self._children[hole.parent].add(hole.id)

    def clear(self) -> None:
        """Clear all holes from the registry."""
        self._holes.clear()
        self._children.clear()
        self._fill_history.clear()
        self._next_index = 0

    def remove(self, hole_id: HoleId) -> Optional[Hole[C]]:
        """Remove a hole from the registry.

        Also removes from parent's children set.

        Args:
            hole_id: The hole to remove

        Returns:
            The removed hole, or None if not found
        """
        hole = self._holes.pop(hole_id, None)
        if hole is None:
            return None

        # Remove from parent's children
        if hole.parent and hole.parent in self._children:
            self._children[hole.parent].discard(hole_id)

        # Remove from children tracking
        self._children.pop(hole_id, None)

        return hole

    def __len__(self) -> int:
        """Return the number of registered holes."""
        return len(self._holes)

    def __contains__(self, hole_id: HoleId) -> bool:
        """Check if a hole ID is registered."""
        return hole_id in self._holes

    def __iter__(self) -> Iterator[Hole[C]]:
        """Iterate over all holes."""
        return iter(self._holes.values())


def create_registry() -> HoleRegistry[Any]:
    """Factory function to create a new HoleRegistry.

    Returns:
        New empty HoleRegistry
    """
    return HoleRegistry()
