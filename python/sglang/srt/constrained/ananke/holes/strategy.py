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
"""Hole selection strategies.

Determines which hole to fill next during generation:
- DepthFirst: Complete inner holes before outer
- BreadthFirst: Complete outer holes before inner
- TypeGuided: Fill most-constrained holes first
- Priority: User-specified order
- SourceOrder: Fill in source code order
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generic, Iterator, List, Optional, TypeVar

from .hole import Hole, HoleId, HoleGranularity

# Type variable for constraint types
C = TypeVar("C")


class HoleSelectionStrategy(ABC, Generic[C]):
    """Abstract base class for hole selection strategies.

    Determines the order in which holes are filled during
    code generation.
    """

    @abstractmethod
    def select(self, holes: List[Hole[C]]) -> Optional[Hole[C]]:
        """Select the next hole to fill.

        Args:
            holes: List of unfilled holes

        Returns:
            The hole to fill next, or None if no valid hole
        """
        pass

    @abstractmethod
    def rank(self, holes: List[Hole[C]]) -> List[Hole[C]]:
        """Rank holes by priority for filling.

        Args:
            holes: List of unfilled holes

        Returns:
            Holes sorted by priority (highest first)
        """
        pass

    def filter(
        self,
        holes: List[Hole[C]],
        predicate: Callable[[Hole[C]], bool],
    ) -> List[Hole[C]]:
        """Filter holes by a predicate.

        Args:
            holes: Holes to filter
            predicate: Filter function

        Returns:
            Filtered holes
        """
        return [h for h in holes if predicate(h)]


class DepthFirstStrategy(HoleSelectionStrategy[C]):
    """Fill innermost (deepest) holes first.

    Good for completing sub-expressions before their
    containing expressions.
    """

    def select(self, holes: List[Hole[C]]) -> Optional[Hole[C]]:
        """Select the deepest hole."""
        if not holes:
            return None
        ranked = self.rank(holes)
        return ranked[0] if ranked else None

    def rank(self, holes: List[Hole[C]]) -> List[Hole[C]]:
        """Rank by depth (deepest first)."""
        return sorted(holes, key=lambda h: (-h.depth, h.id.index))


class BreadthFirstStrategy(HoleSelectionStrategy[C]):
    """Fill outermost (shallowest) holes first.

    Good for establishing high-level structure before
    filling in details.
    """

    def select(self, holes: List[Hole[C]]) -> Optional[Hole[C]]:
        """Select the shallowest hole."""
        if not holes:
            return None
        ranked = self.rank(holes)
        return ranked[0] if ranked else None

    def rank(self, holes: List[Hole[C]]) -> List[Hole[C]]:
        """Rank by depth (shallowest first)."""
        return sorted(holes, key=lambda h: (h.depth, h.id.index))


class SourceOrderStrategy(HoleSelectionStrategy[C]):
    """Fill holes in source code order.

    Based on source location (line, column).
    """

    def select(self, holes: List[Hole[C]]) -> Optional[Hole[C]]:
        """Select hole that appears first in source."""
        if not holes:
            return None
        ranked = self.rank(holes)
        return ranked[0] if ranked else None

    def rank(self, holes: List[Hole[C]]) -> List[Hole[C]]:
        """Rank by source location."""
        def sort_key(h: Hole[C]) -> tuple:
            if h.location:
                return (h.location.line, h.location.column, h.id.index)
            return (float("inf"), float("inf"), h.id.index)

        return sorted(holes, key=sort_key)


class TypeGuidedStrategy(HoleSelectionStrategy[C]):
    """Fill most-constrained holes first.

    Prioritizes holes with:
    1. Specific expected types (not Any)
    2. Smaller type (fewer options)
    3. More constrained by context
    """

    def __init__(
        self,
        type_score: Optional[Callable[[Any], float]] = None,
    ) -> None:
        """Initialize the strategy.

        Args:
            type_score: Function to score types (higher = more constrained)
        """
        self._type_score = type_score or self._default_type_score

    def _default_type_score(self, typ: Any) -> float:
        """Default type scoring function.

        Scores types by how constrained they are:
        - None/Any: 0 (least constrained)
        - Primitive: 3
        - Collection: 2
        - Function: 4
        - Union: 1
        """
        if typ is None:
            return 0.0

        type_name = str(typ)

        # Check for primitive types
        primitives = ["int", "str", "bool", "float", "None"]
        if any(p in type_name for p in primitives):
            return 3.0

        # Check for function types
        if "Callable" in type_name or "Function" in type_name or "->" in type_name:
            return 4.0

        # Check for collection types
        collections = ["List", "Dict", "Set", "Tuple"]
        if any(c in type_name for c in collections):
            return 2.0

        # Check for union types (less constrained)
        if "Union" in type_name or "|" in type_name:
            return 1.0

        # Check for Any (least constrained)
        if "Any" in type_name:
            return 0.0

        # Default score for other types
        return 2.5

    def select(self, holes: List[Hole[C]]) -> Optional[Hole[C]]:
        """Select the most constrained hole."""
        if not holes:
            return None
        ranked = self.rank(holes)
        return ranked[0] if ranked else None

    def rank(self, holes: List[Hole[C]]) -> List[Hole[C]]:
        """Rank by type constraint (most constrained first)."""
        def sort_key(h: Hole[C]) -> tuple:
            score = self._type_score(h.expected_type)
            # Higher score = more constrained = fill first
            return (-score, h.id.index)

        return sorted(holes, key=sort_key)


class PriorityStrategy(HoleSelectionStrategy[C]):
    """Fill holes in user-specified priority order.

    Allows explicit control over fill order.
    """

    def __init__(self) -> None:
        """Initialize the strategy."""
        self._priorities: Dict[HoleId, int] = {}

    def set_priority(self, hole_id: HoleId, priority: int) -> None:
        """Set priority for a hole.

        Higher priority = fill first.

        Args:
            hole_id: Hole identifier
            priority: Priority value
        """
        self._priorities[hole_id] = priority

    def get_priority(self, hole_id: HoleId) -> int:
        """Get priority for a hole.

        Args:
            hole_id: Hole identifier

        Returns:
            Priority value (default 0)
        """
        return self._priorities.get(hole_id, 0)

    def select(self, holes: List[Hole[C]]) -> Optional[Hole[C]]:
        """Select highest priority hole."""
        if not holes:
            return None
        ranked = self.rank(holes)
        return ranked[0] if ranked else None

    def rank(self, holes: List[Hole[C]]) -> List[Hole[C]]:
        """Rank by priority (highest first)."""
        def sort_key(h: Hole[C]) -> tuple:
            priority = self._priorities.get(h.id, 0)
            return (-priority, h.id.index)

        return sorted(holes, key=sort_key)

    def clear_priorities(self) -> None:
        """Clear all priority assignments."""
        self._priorities.clear()


class GranularityStrategy(HoleSelectionStrategy[C]):
    """Fill holes by granularity level.

    Can prioritize larger or smaller holes.
    """

    # Granularity ordering (smaller to larger)
    _GRANULARITY_ORDER = {
        HoleGranularity.TOKEN: 0,
        HoleGranularity.EXPRESSION: 1,
        HoleGranularity.STATEMENT: 2,
        HoleGranularity.BLOCK: 3,
        HoleGranularity.FUNCTION: 4,
        HoleGranularity.MODULE: 5,
        HoleGranularity.SYSTEM: 6,
    }

    def __init__(self, larger_first: bool = False) -> None:
        """Initialize the strategy.

        Args:
            larger_first: If True, fill larger holes first
        """
        self._larger_first = larger_first

    def select(self, holes: List[Hole[C]]) -> Optional[Hole[C]]:
        """Select hole by granularity."""
        if not holes:
            return None
        ranked = self.rank(holes)
        return ranked[0] if ranked else None

    def rank(self, holes: List[Hole[C]]) -> List[Hole[C]]:
        """Rank by granularity."""
        def sort_key(h: Hole[C]) -> tuple:
            order = self._GRANULARITY_ORDER.get(h.granularity, 3)
            if self._larger_first:
                order = -order
            return (order, h.id.index)

        return sorted(holes, key=sort_key)


class CompositeStrategy(HoleSelectionStrategy[C]):
    """Combine multiple strategies with fallback.

    Tries strategies in order until one returns a hole.
    """

    def __init__(
        self,
        strategies: Optional[List[HoleSelectionStrategy[C]]] = None,
    ) -> None:
        """Initialize the strategy.

        Args:
            strategies: List of strategies to try
        """
        self._strategies = strategies or []

    def add_strategy(self, strategy: HoleSelectionStrategy[C]) -> None:
        """Add a strategy to the chain.

        Args:
            strategy: Strategy to add
        """
        self._strategies.append(strategy)

    def select(self, holes: List[Hole[C]]) -> Optional[Hole[C]]:
        """Select using first strategy that returns a hole."""
        for strategy in self._strategies:
            result = strategy.select(holes)
            if result is not None:
                return result
        return None

    def rank(self, holes: List[Hole[C]]) -> List[Hole[C]]:
        """Rank using first strategy."""
        if self._strategies:
            return self._strategies[0].rank(holes)
        return holes


# Strategy presets
def depth_first() -> DepthFirstStrategy[Any]:
    """Create a depth-first strategy."""
    return DepthFirstStrategy()


def breadth_first() -> BreadthFirstStrategy[Any]:
    """Create a breadth-first strategy."""
    return BreadthFirstStrategy()


def source_order() -> SourceOrderStrategy[Any]:
    """Create a source-order strategy."""
    return SourceOrderStrategy()


def type_guided() -> TypeGuidedStrategy[Any]:
    """Create a type-guided strategy."""
    return TypeGuidedStrategy()


def priority_based() -> PriorityStrategy[Any]:
    """Create a priority-based strategy."""
    return PriorityStrategy()


def granularity_based(larger_first: bool = False) -> GranularityStrategy[Any]:
    """Create a granularity-based strategy.

    Args:
        larger_first: If True, fill larger holes first

    Returns:
        GranularityStrategy instance
    """
    return GranularityStrategy(larger_first=larger_first)


def default_strategy() -> CompositeStrategy[Any]:
    """Create the default strategy.

    Default order:
    1. Type-guided (most constrained first)
    2. Source order (as tiebreaker)

    Returns:
        CompositeStrategy with default configuration
    """
    return CompositeStrategy([
        TypeGuidedStrategy(),
        SourceOrderStrategy(),
    ])
