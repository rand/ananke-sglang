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
"""Fill-and-resume evaluation engine.

Implements Hazel's fill-and-resume semantics:
- Evaluation proceeds around holes, producing partial results
- Filling a hole enables continuing evaluation
- Unfilling allows backtracking to try different content

This enables:
- Live programming with incomplete code
- Interactive exploration of alternatives
- Efficient backtracking during generation

References:
- Live Functional Programming with Typed Holes (ICFP 2019)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar

from .hole import Hole, HoleId, HoleState, TypeEnvironment
from .registry import HoleRegistry, RegistryCheckpoint
from .closure import HoleClosure, FilledClosure, Continuation, ClosureManager

# Type variable for constraint types
C = TypeVar("C")


class EvaluationState(Enum):
    """State of evaluation.

    States:
    - READY: Ready to start evaluation
    - RUNNING: Evaluation in progress
    - PAUSED: Paused at a hole
    - COMPLETE: Evaluation finished
    - ERROR: Evaluation encountered an error
    """

    READY = auto()
    RUNNING = auto()
    PAUSED = auto()
    COMPLETE = auto()
    ERROR = auto()


@dataclass
class EvaluationResult:
    """Result of evaluation.

    Attributes:
        state: Final evaluation state
        value: Result value (if complete)
        paused_at: Hole ID where paused (if paused)
        error: Error message (if error)
        closures: Hole closures created during evaluation
    """

    state: EvaluationState
    value: Optional[Any] = None
    paused_at: Optional[HoleId] = None
    error: Optional[str] = None
    closures: Dict[HoleId, HoleClosure[Any]] = field(default_factory=dict)

    @property
    def is_complete(self) -> bool:
        """Check if evaluation completed."""
        return self.state == EvaluationState.COMPLETE

    @property
    def is_paused(self) -> bool:
        """Check if evaluation is paused at a hole."""
        return self.state == EvaluationState.PAUSED

    @property
    def is_error(self) -> bool:
        """Check if evaluation encountered an error."""
        return self.state == EvaluationState.ERROR


@dataclass
class FillResult:
    """Result of filling a hole.

    Attributes:
        hole: The filled hole
        success: Whether the fill was successful
        type_valid: Whether the content type-checks
        error: Error message if unsuccessful
    """

    hole: Hole[Any]
    success: bool
    type_valid: bool = True
    error: Optional[str] = None


class FillAndResumeEngine(Generic[C]):
    """Engine for fill-and-resume evaluation.

    Manages the lifecycle of evaluation with holes:
    1. Start evaluation, which pauses at holes
    2. Fill holes with content
    3. Resume evaluation after filling
    4. Backtrack by unfilling and trying alternatives

    Example:
        >>> engine = FillAndResumeEngine(registry)
        >>> result = engine.start_evaluation(code)
        >>> if result.is_paused:
        ...     engine.fill(result.paused_at, "return 42")
        ...     result = engine.resume()
    """

    def __init__(
        self,
        registry: HoleRegistry[C],
        evaluator: Optional[Callable[[str, TypeEnvironment], Any]] = None,
        type_checker: Optional[Callable[[str, Any, TypeEnvironment], bool]] = None,
    ) -> None:
        """Initialize the engine.

        Args:
            registry: Hole registry to use
            evaluator: Function to evaluate code strings
            type_checker: Function to check types
        """
        self._registry = registry
        self._evaluator = evaluator
        self._type_checker = type_checker
        self._closures = ClosureManager[C]()
        self._checkpoints: List[RegistryCheckpoint[C]] = []
        self._state = EvaluationState.READY
        self._current_result: Optional[EvaluationResult] = None

    @property
    def state(self) -> EvaluationState:
        """Get current evaluation state."""
        return self._state

    @property
    def registry(self) -> HoleRegistry[C]:
        """Get the hole registry."""
        return self._registry

    def start_evaluation(
        self,
        code: str,
        environment: Optional[TypeEnvironment] = None,
    ) -> EvaluationResult:
        """Start evaluation of code.

        Proceeds until hitting a hole or completing.

        Args:
            code: Code to evaluate
            environment: Typing environment

        Returns:
            Evaluation result
        """
        self._state = EvaluationState.RUNNING
        env = environment or TypeEnvironment.empty()

        # Check for holes in the registry
        next_hole = self._registry.next_hole()

        if next_hole is None:
            # No holes - evaluate directly
            if self._evaluator:
                try:
                    value = self._evaluator(code, env)
                    self._state = EvaluationState.COMPLETE
                    return EvaluationResult(
                        state=EvaluationState.COMPLETE,
                        value=value,
                    )
                except Exception as e:
                    self._state = EvaluationState.ERROR
                    return EvaluationResult(
                        state=EvaluationState.ERROR,
                        error=str(e),
                    )
            else:
                # No evaluator - just return the code
                self._state = EvaluationState.COMPLETE
                return EvaluationResult(
                    state=EvaluationState.COMPLETE,
                    value=code,
                )

        # Create closure for the hole and pause
        closure = HoleClosure.create(next_hole, env)
        self._closures.add_closure(closure)

        self._state = EvaluationState.PAUSED
        result = EvaluationResult(
            state=EvaluationState.PAUSED,
            paused_at=next_hole.id,
            closures={next_hole.id: closure},
        )
        self._current_result = result
        return result

    def fill(
        self,
        hole_id: HoleId,
        content: str,
        *,
        validate: bool = True,
    ) -> FillResult:
        """Fill a hole with content.

        Args:
            hole_id: The hole to fill
            content: Content to fill with
            validate: Whether to type-check the content

        Returns:
            FillResult indicating success/failure
        """
        # Get the hole
        hole = self._registry.lookup(hole_id)
        if hole is None:
            return FillResult(
                hole=Hole(id=hole_id),
                success=False,
                error=f"Hole not found: {hole_id}",
            )

        # Type check if requested and checker available
        type_valid = True
        if validate and self._type_checker and hole.expected_type:
            try:
                type_valid = self._type_checker(
                    content,
                    hole.expected_type,
                    hole.environment,
                )
            except Exception as e:
                return FillResult(
                    hole=hole,
                    success=False,
                    type_valid=False,
                    error=f"Type check failed: {e}",
                )

            if not type_valid:
                return FillResult(
                    hole=hole,
                    success=False,
                    type_valid=False,
                    error="Content does not match expected type",
                )

        # Fill the hole
        filled_hole = self._registry.fill(hole_id, content)

        # Update closure if exists
        if hole_id in self._closures.closures:
            self._closures.fill(hole_id, content)

        return FillResult(
            hole=filled_hole,
            success=True,
            type_valid=type_valid,
        )

    def unfill(self, hole_id: HoleId) -> Optional[Hole[C]]:
        """Unfill a hole for backtracking.

        Args:
            hole_id: The hole to unfill

        Returns:
            The unfilled hole, or None if not found
        """
        if not self._registry.contains(hole_id):
            return None

        # Unfill in registry
        unfilled = self._registry.unfill(hole_id)

        # Unfill in closures if exists
        if hole_id in self._closures.filled:
            self._closures.unfill(hole_id)

        return unfilled

    def resume(self) -> EvaluationResult:
        """Resume evaluation after filling holes.

        Returns:
            Evaluation result
        """
        self._state = EvaluationState.RUNNING

        # Find next unfilled hole
        next_hole = self._registry.next_hole()

        if next_hole is None:
            # All holes filled - can complete
            self._state = EvaluationState.COMPLETE
            return EvaluationResult(state=EvaluationState.COMPLETE)

        # Still have holes - pause at next one
        closure = self._closures.get_closure(next_hole.id)
        if closure is None:
            closure = HoleClosure.create(next_hole)
            self._closures.add_closure(closure)

        self._state = EvaluationState.PAUSED
        return EvaluationResult(
            state=EvaluationState.PAUSED,
            paused_at=next_hole.id,
            closures={next_hole.id: closure},
        )

    def checkpoint(self) -> int:
        """Create a checkpoint of current state.

        Returns:
            Checkpoint index for later restoration
        """
        checkpoint = self._registry.checkpoint()
        self._checkpoints.append(checkpoint)
        return len(self._checkpoints) - 1

    def restore(self, checkpoint_index: int) -> bool:
        """Restore state from a checkpoint.

        Args:
            checkpoint_index: Index returned from checkpoint()

        Returns:
            True if restoration successful
        """
        if checkpoint_index < 0 or checkpoint_index >= len(self._checkpoints):
            return False

        checkpoint = self._checkpoints[checkpoint_index]
        self._registry.restore(checkpoint)

        # Discard later checkpoints
        self._checkpoints = self._checkpoints[:checkpoint_index + 1]

        return True

    def rollback(self) -> bool:
        """Roll back to the previous checkpoint.

        Returns:
            True if rollback successful
        """
        if not self._checkpoints:
            return False

        # Restore to last checkpoint
        checkpoint = self._checkpoints.pop()
        self._registry.restore(checkpoint)
        return True

    def get_unfilled_holes(self) -> List[Hole[C]]:
        """Get all unfilled holes.

        Returns:
            List of unfilled holes
        """
        return list(self._registry.empty_holes())

    def get_filled_holes(self) -> List[Hole[C]]:
        """Get all filled holes.

        Returns:
            List of filled holes
        """
        return list(self._registry.filled_holes())

    def get_closures(self) -> Dict[HoleId, HoleClosure[C]]:
        """Get all hole closures.

        Returns:
            Dictionary of hole ID to closure
        """
        return dict(self._closures.closures)

    def reset(self) -> None:
        """Reset the engine to initial state."""
        self._registry.clear()
        self._closures.clear()
        self._checkpoints.clear()
        self._state = EvaluationState.READY
        self._current_result = None


def create_fill_resume_engine(
    registry: Optional[HoleRegistry[Any]] = None,
) -> FillAndResumeEngine[Any]:
    """Factory function to create a fill-and-resume engine.

    Args:
        registry: Optional hole registry (creates new if None)

    Returns:
        New FillAndResumeEngine
    """
    reg = registry or HoleRegistry()
    return FillAndResumeEngine(reg)
