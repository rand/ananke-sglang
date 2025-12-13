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
"""Hole closures for fill-and-resume semantics.

Following Hazel's approach, a hole closure captures the evaluation
context around a hole, enabling:
- Partial evaluation that proceeds around holes
- Fill-and-resume: fill a hole and continue evaluation
- Backtracking: unfill and try different content

References:
- Live Functional Programming with Typed Holes (ICFP 2019)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar

from .hole import Hole, HoleId, TypeEnvironment

# Type variable for constraint types
C = TypeVar("C")


class ContinuationKind(Enum):
    """Kind of continuation in a hole closure.

    Different continuations represent different evaluation contexts:
    - IDENTITY: No continuation (hole result is final result)
    - APPLICATION: Hole appears in function application context
    - BINDING: Hole appears in let-binding context
    - CONDITIONAL: Hole appears in conditional context
    - SEQUENCE: Hole appears in sequence context
    """

    IDENTITY = auto()
    APPLICATION = auto()
    BINDING = auto()
    CONDITIONAL = auto()
    SEQUENCE = auto()


@dataclass(frozen=True)
class Continuation:
    """A continuation representing evaluation context.

    Captures what happens after a hole is filled and evaluated.

    Attributes:
        kind: Type of continuation
        context: Additional context data
        next: Chained continuation (if any)
    """

    kind: ContinuationKind
    context: Dict[str, Any] = field(default_factory=dict)
    next: Optional[Continuation] = None

    def apply(self, value: Any) -> Any:
        """Apply this continuation to a value.

        For now, this is a placeholder that returns the value.
        Full implementation would evaluate the continuation.

        Args:
            value: The result of filling the hole

        Returns:
            Result of applying continuation
        """
        # For identity, just return the value
        if self.kind == ContinuationKind.IDENTITY:
            result = value
        else:
            # Placeholder for other continuation kinds
            result = value

        # Chain to next continuation if present
        if self.next:
            return self.next.apply(result)

        return result

    def compose(self, other: Continuation) -> Continuation:
        """Compose this continuation with another.

        Creates a new continuation that applies this one,
        then the other.

        Args:
            other: Continuation to compose with

        Returns:
            Composed continuation
        """
        if self.next is None:
            return Continuation(
                kind=self.kind,
                context=self.context,
                next=other,
            )
        else:
            return Continuation(
                kind=self.kind,
                context=self.context,
                next=self.next.compose(other),
            )

    @classmethod
    def identity(cls) -> Continuation:
        """Create an identity continuation."""
        return cls(kind=ContinuationKind.IDENTITY)


@dataclass
class HoleClosure(Generic[C]):
    """A hole closure capturing evaluation state around a hole.

    Following Hazel, holes serve as "membranes" around missing code.
    A closure captures:
    - The hole itself
    - The typing environment at the hole site
    - The continuation (what happens after the hole is filled)

    This enables:
    - Evaluate program, pausing at holes
    - Fill a hole, resume evaluation
    - Unfill and try different content

    Attributes:
        hole: The hole being closed over
        environment: Captured typing environment
        continuation: What to do after hole is filled
        partial_result: Any partial evaluation result
    """

    hole: Hole[C]
    environment: TypeEnvironment
    continuation: Continuation
    partial_result: Optional[Any] = None

    @property
    def hole_id(self) -> HoleId:
        """Get the hole's ID."""
        return self.hole.id

    @property
    def expected_type(self) -> Optional[Any]:
        """Get the expected type for the hole."""
        return self.hole.expected_type

    def fill(self, content: str) -> FilledClosure[C]:
        """Fill the hole and return a filled closure.

        Does not evaluate - use evaluate() on the result.

        Args:
            content: Content to fill the hole with

        Returns:
            FilledClosure with filled hole
        """
        filled_hole = self.hole.fill(content)
        return FilledClosure(
            hole=filled_hole,
            environment=self.environment,
            continuation=self.continuation,
            content=content,
        )

    def with_continuation(self, continuation: Continuation) -> HoleClosure[C]:
        """Create a new closure with a different continuation.

        Args:
            continuation: New continuation

        Returns:
            New closure with the continuation
        """
        return HoleClosure(
            hole=self.hole,
            environment=self.environment,
            continuation=continuation,
            partial_result=self.partial_result,
        )

    def extend_continuation(self, continuation: Continuation) -> HoleClosure[C]:
        """Extend the current continuation with another.

        Args:
            continuation: Continuation to add

        Returns:
            New closure with extended continuation
        """
        return HoleClosure(
            hole=self.hole,
            environment=self.environment,
            continuation=self.continuation.compose(continuation),
            partial_result=self.partial_result,
        )

    @classmethod
    def create(
        cls,
        hole: Hole[C],
        environment: Optional[TypeEnvironment] = None,
        continuation: Optional[Continuation] = None,
    ) -> HoleClosure[C]:
        """Factory method to create a hole closure.

        Args:
            hole: The hole to close over
            environment: Typing environment (uses hole's if None)
            continuation: Continuation (identity if None)

        Returns:
            New HoleClosure
        """
        return cls(
            hole=hole,
            environment=environment or hole.environment,
            continuation=continuation or Continuation.identity(),
        )


@dataclass
class FilledClosure(Generic[C]):
    """A closure for a filled hole.

    After filling a hole, this closure can be evaluated
    to produce a result.

    Attributes:
        hole: The filled hole
        environment: Typing environment
        continuation: What to do with the result
        content: The filled content
        evaluated_value: Result of evaluation (if done)
    """

    hole: Hole[C]
    environment: TypeEnvironment
    continuation: Continuation
    content: str
    evaluated_value: Optional[Any] = None

    @property
    def hole_id(self) -> HoleId:
        """Get the hole's ID."""
        return self.hole.id

    def evaluate(
        self,
        evaluator: Optional[Callable[[str, TypeEnvironment], Any]] = None,
    ) -> Any:
        """Evaluate the filled content and apply continuation.

        Args:
            evaluator: Function to evaluate content
                      (defaults to returning content as-is)

        Returns:
            Result of evaluation with continuation applied
        """
        # Evaluate content (or use as-is if no evaluator)
        if evaluator:
            value = evaluator(self.content, self.environment)
        else:
            value = self.content

        # Store evaluated value
        object.__setattr__(self, "evaluated_value", value)

        # Apply continuation
        return self.continuation.apply(value)

    def unfill(self) -> HoleClosure[C]:
        """Unfill and return to a hole closure.

        Used for backtracking.

        Returns:
            HoleClosure with empty hole
        """
        unfilled_hole = self.hole.unfill()
        return HoleClosure(
            hole=unfilled_hole,
            environment=self.environment,
            continuation=self.continuation,
        )


@dataclass
class ClosureManager(Generic[C]):
    """Manager for multiple hole closures.

    Tracks closures during evaluation, enabling:
    - Evaluation that stops at holes
    - Fill-and-resume across multiple holes
    - Backtracking to any previous state

    Attributes:
        closures: Map from hole ID to closure
        filled: Map from hole ID to filled closure
        evaluation_order: Order in which holes were encountered
    """

    closures: Dict[HoleId, HoleClosure[C]] = field(default_factory=dict)
    filled: Dict[HoleId, FilledClosure[C]] = field(default_factory=dict)
    evaluation_order: List[HoleId] = field(default_factory=list)

    def add_closure(self, closure: HoleClosure[C]) -> None:
        """Add a hole closure to the manager.

        Args:
            closure: Closure to add
        """
        hole_id = closure.hole_id
        self.closures[hole_id] = closure
        if hole_id not in self.evaluation_order:
            self.evaluation_order.append(hole_id)

    def get_closure(self, hole_id: HoleId) -> Optional[HoleClosure[C]]:
        """Get a hole closure by ID.

        Args:
            hole_id: The hole identifier

        Returns:
            The closure if found
        """
        return self.closures.get(hole_id)

    def fill(self, hole_id: HoleId, content: str) -> Optional[FilledClosure[C]]:
        """Fill a hole and track the filled closure.

        Args:
            hole_id: The hole to fill
            content: Content to fill with

        Returns:
            The filled closure, or None if hole not found
        """
        closure = self.closures.get(hole_id)
        if closure is None:
            return None

        filled = closure.fill(content)
        self.filled[hole_id] = filled

        # Remove from unfilled closures
        del self.closures[hole_id]

        return filled

    def unfill(self, hole_id: HoleId) -> Optional[HoleClosure[C]]:
        """Unfill a hole and return to closure state.

        Args:
            hole_id: The hole to unfill

        Returns:
            The hole closure, or None if not found
        """
        filled = self.filled.get(hole_id)
        if filled is None:
            return None

        unfilled = filled.unfill()
        self.closures[hole_id] = unfilled

        # Remove from filled
        del self.filled[hole_id]

        return unfilled

    def next_unfilled(self) -> Optional[HoleClosure[C]]:
        """Get the next unfilled hole closure.

        Returns closures in evaluation order.

        Returns:
            Next unfilled closure, or None if all filled
        """
        for hole_id in self.evaluation_order:
            if hole_id in self.closures:
                return self.closures[hole_id]
        return None

    @property
    def unfilled_count(self) -> int:
        """Get the number of unfilled holes."""
        return len(self.closures)

    @property
    def filled_count(self) -> int:
        """Get the number of filled holes."""
        return len(self.filled)

    @property
    def all_filled(self) -> bool:
        """Check if all holes are filled."""
        return len(self.closures) == 0

    def clear(self) -> None:
        """Clear all closures."""
        self.closures.clear()
        self.filled.clear()
        self.evaluation_order.clear()


def create_closure(
    hole: Hole[C],
    environment: Optional[TypeEnvironment] = None,
) -> HoleClosure[C]:
    """Factory function to create a hole closure.

    Args:
        hole: The hole to close over
        environment: Optional typing environment

    Returns:
        New HoleClosure
    """
    return HoleClosure.create(hole, environment)
