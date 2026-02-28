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
"""Sudoku-style hole filling with constraint propagation and MCV heuristic.

This module implements a constraint satisfaction approach to code generation,
treating typed holes as variables in a CSP. Like Sudoku solving, we:

1. Most Constrained Variable (MCV): Fill holes with fewest valid options first
2. Constraint Propagation: After each fill, propagate to dependent holes
3. Backtracking: If stuck, try alternative fills

Key Insight:
    The MCV heuristic (also called "fail-first") prunes the search tree early
    by prioritizing holes that are most constrained. If a hole has only one
    valid fill, filling it first propagates maximum information to other holes.

Soundness:
    This is a SEARCH algorithm, not a masking algorithm. Soundness is preserved
    because we only score and rank candidates - never block valid code paths.
    Backtracking ensures we explore alternatives when constraints are violated.

References:
- GenCP: LLM Meets Constraint Propagation (arXiv 2024)
- ROCODE: Backtracking for Code Generation (EMNLP 2024)
- Hazel Typed Holes (OOPSLA 2024)
"""

from __future__ import annotations

import logging
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Generic,
    Iterator,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    TypeVar,
)

# Handle imports for both package and standalone usage
try:
    from ..holes.hole import Hole, HoleId, HoleState, TypeEnvironment
except ImportError:
    # Standalone mode - add package root to path
    _SEARCH_DIR = Path(__file__).parent
    _ANANKE_ROOT = _SEARCH_DIR.parent
    if str(_ANANKE_ROOT) not in sys.path:
        sys.path.insert(0, str(_ANANKE_ROOT))
    from holes.hole import Hole, HoleId, HoleState, TypeEnvironment

logger = logging.getLogger(__name__)

# Type variable for constraint types
C = TypeVar("C")


class FillStrategy(Enum):
    """Strategy for selecting the next hole to fill.

    MCV (Most Constrained Variable):
        Fill the hole with fewest valid options first.
        Also known as "fail-first" - catches inconsistencies early.

    MRV (Minimum Remaining Values):
        Alias for MCV, used in CSP literature.

    LARGEST_DEGREE:
        Fill the hole connected to most unfilled holes first.
        Maximizes constraint propagation impact.

    SEQUENTIAL:
        Fill holes in order of appearance.
        Simple but may miss propagation opportunities.

    RANDOM:
        Fill holes in random order.
        Useful for diversification in ensemble methods.
    """

    MCV = auto()  # Most Constrained Variable (fail-first)
    MRV = auto()  # Minimum Remaining Values (alias for MCV)
    LARGEST_DEGREE = auto()  # Most connections first
    SEQUENTIAL = auto()  # Order of appearance
    RANDOM = auto()  # Random selection


@dataclass
class FillCandidate:
    """A candidate fill for a hole.

    Attributes:
        value: The fill value (code string)
        score: Constraint satisfaction score (0.0 to 1.0)
        constraint_violations: List of violated constraints
        metadata: Additional information about the fill
    """

    value: str
    score: float = 1.0
    constraint_violations: Tuple[str, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other: FillCandidate) -> bool:
        """Order by score descending (higher scores first)."""
        return self.score > other.score


@dataclass
class FillResult:
    """Result of hole filling attempt.

    Attributes:
        success: Whether all holes were successfully filled
        filled_code: The final code with holes filled (if success)
        fill_history: List of (hole_id, fill_value) in fill order
        unfilled_holes: Holes that could not be filled
        backtrack_count: Number of backtracks during search
        propagation_count: Number of constraint propagations
        latency_ns: Total time taken (nanoseconds)
        metadata: Additional result information
    """

    success: bool
    filled_code: Optional[str] = None
    fill_history: List[Tuple[HoleId, str]] = field(default_factory=list)
    unfilled_holes: List[HoleId] = field(default_factory=list)
    backtrack_count: int = 0
    propagation_count: int = 0
    latency_ns: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "success": self.success,
            "filled_code": self.filled_code,
            "fill_history": [(str(h), v) for h, v in self.fill_history],
            "unfilled_holes": [str(h) for h in self.unfilled_holes],
            "backtrack_count": self.backtrack_count,
            "propagation_count": self.propagation_count,
            "latency_ns": self.latency_ns,
            "metadata": self.metadata,
        }


class FillGenerator(Protocol):
    """Protocol for generating fill candidates for a hole."""

    def generate_candidates(
        self,
        hole: Hole[Any],
        context: str,
        max_candidates: int = 10,
    ) -> List[FillCandidate]:
        """Generate candidate fills for a hole.

        Args:
            hole: The hole to fill
            context: Surrounding code context
            max_candidates: Maximum number of candidates to generate

        Returns:
            List of FillCandidate ordered by score (best first)
        """
        ...


class ConstraintChecker(Protocol):
    """Protocol for checking constraint satisfaction."""

    def check_fill(
        self,
        hole: Hole[Any],
        fill: str,
        context: HoledCode,
    ) -> Tuple[bool, float, Tuple[str, ...]]:
        """Check if a fill satisfies constraints.

        Args:
            hole: The hole being filled
            fill: The fill value
            context: Current state of holed code

        Returns:
            (valid, score, violations) tuple
        """
        ...


@dataclass
class HoledCode:
    """Code with typed holes.

    Represents a partially complete code string with typed holes
    that need to be filled. Provides operations for filling holes
    and tracking dependencies.

    Attributes:
        template: Code template with hole markers
        holes: Map from HoleId to Hole
        hole_markers: Map from HoleId to marker string (e.g., "?hole:0")
        dependencies: Map from HoleId to dependent HoleIds
        language: Target programming language
    """

    template: str
    holes: Dict[HoleId, Hole[Any]] = field(default_factory=dict)
    hole_markers: Dict[HoleId, str] = field(default_factory=dict)
    dependencies: Dict[HoleId, FrozenSet[HoleId]] = field(default_factory=dict)
    language: str = "python"

    def has_holes(self) -> bool:
        """Check if there are unfilled holes."""
        return any(
            hole.state in (HoleState.EMPTY, HoleState.PARTIAL)
            for hole in self.holes.values()
        )

    def unfilled_holes(self) -> List[Hole[Any]]:
        """Get list of unfilled holes."""
        return [
            hole
            for hole in self.holes.values()
            if hole.state in (HoleState.EMPTY, HoleState.PARTIAL)
        ]

    def get_hole(self, hole_id: HoleId) -> Optional[Hole[Any]]:
        """Get a hole by ID."""
        return self.holes.get(hole_id)

    def fill_hole(self, hole_id: HoleId, value: str) -> HoledCode:
        """Create a new HoledCode with the specified hole filled.

        Does not modify this instance - returns a new one.

        Args:
            hole_id: ID of the hole to fill
            value: Fill value

        Returns:
            New HoledCode with hole filled
        """
        if hole_id not in self.holes:
            raise ValueError(f"Unknown hole: {hole_id}")

        hole = self.holes[hole_id]
        filled_hole = hole.fill(value)

        # Update template
        marker = self.hole_markers.get(hole_id, str(hole_id))
        new_template = self.template.replace(marker, value, 1)

        # Update holes map
        new_holes = self.holes.copy()
        new_holes[hole_id] = filled_hole

        return HoledCode(
            template=new_template,
            holes=new_holes,
            hole_markers=self.hole_markers.copy(),
            dependencies=self.dependencies.copy(),
            language=self.language,
        )

    def unfill_hole(self, hole_id: HoleId) -> HoledCode:
        """Create a new HoledCode with the specified hole unfilled.

        Used for backtracking.

        Args:
            hole_id: ID of the hole to unfill

        Returns:
            New HoledCode with hole unfilled
        """
        if hole_id not in self.holes:
            raise ValueError(f"Unknown hole: {hole_id}")

        hole = self.holes[hole_id]

        # Can only unfill filled holes
        if hole.state not in (HoleState.FILLED, HoleState.VALIDATED, HoleState.PARTIAL):
            return self

        unfilled_hole = hole.unfill()
        marker = self.hole_markers.get(hole_id, str(hole_id))

        # Restore marker in template (approximate - may not be exact position)
        # For precise restoration, we'd need to track positions
        new_template = self.template
        if hole.content:
            new_template = new_template.replace(hole.content, marker, 1)

        new_holes = self.holes.copy()
        new_holes[hole_id] = unfilled_hole

        return HoledCode(
            template=new_template,
            holes=new_holes,
            hole_markers=self.hole_markers.copy(),
            dependencies=self.dependencies.copy(),
            language=self.language,
        )

    def update_hole_constraint(
        self,
        hole_id: HoleId,
        constraint: Any,
    ) -> HoledCode:
        """Create a new HoledCode with updated hole constraint.

        Args:
            hole_id: ID of the hole to update
            constraint: New constraint

        Returns:
            New HoledCode with updated constraint
        """
        if hole_id not in self.holes:
            raise ValueError(f"Unknown hole: {hole_id}")

        hole = self.holes[hole_id]
        updated_hole = hole.with_constraint(constraint)

        new_holes = self.holes.copy()
        new_holes[hole_id] = updated_hole

        return HoledCode(
            template=self.template,
            holes=new_holes,
            hole_markers=self.hole_markers.copy(),
            dependencies=self.dependencies.copy(),
            language=self.language,
        )

    def dependent_holes(self, hole_id: HoleId) -> List[Hole[Any]]:
        """Get holes that depend on the given hole."""
        dep_ids = self.dependencies.get(hole_id, frozenset())
        return [self.holes[hid] for hid in dep_ids if hid in self.holes]

    def to_string(self) -> str:
        """Convert to final code string."""
        return self.template

    def is_consistent(self) -> bool:
        """Check if current state is consistent (no invalid holes)."""
        return not any(
            hole.state == HoleState.INVALID for hole in self.holes.values()
        )

    @classmethod
    def from_code_with_markers(
        cls,
        code: str,
        hole_marker_pattern: str = r"\?(\w+):(\w+)\[(\d+)\]",
        language: str = "python",
    ) -> HoledCode:
        """Create HoledCode from code with hole markers.

        Parses code for markers matching the pattern and creates
        corresponding Hole instances.

        Args:
            code: Code string with hole markers
            hole_marker_pattern: Regex pattern for hole markers
            language: Target programming language

        Returns:
            HoledCode instance
        """
        import re

        holes: Dict[HoleId, Hole[Any]] = {}
        hole_markers: Dict[HoleId, str] = {}

        pattern = re.compile(hole_marker_pattern)
        for match in pattern.finditer(code):
            marker = match.group(0)
            namespace = match.group(1) if len(match.groups()) >= 1 else "default"
            name = match.group(2) if len(match.groups()) >= 2 else "hole"
            index = int(match.group(3)) if len(match.groups()) >= 3 else 0

            hole_id = HoleId(namespace=namespace, name=name, index=index)
            hole = Hole(id=hole_id)

            holes[hole_id] = hole
            hole_markers[hole_id] = marker

        return cls(
            template=code,
            holes=holes,
            hole_markers=hole_markers,
            language=language,
        )


class SudokuStyleHoleFiller:
    """Fill typed holes using constraint propagation and MCV heuristic.

    Treats code generation as a Constraint Satisfaction Problem (CSP)
    and applies Sudoku-solving techniques:

    1. Most Constrained Variable (MCV): Fill holes with fewest valid
       options first. This "fail-first" approach prunes the search
       tree early by catching inconsistencies.

    2. Constraint Propagation: After filling a hole, propagate the
       effects to dependent holes. This narrows valid options and
       may reveal forced fills or inconsistencies.

    3. Backtracking: If a hole has no valid fills, backtrack to the
       most recent decision point and try an alternative.

    Attributes:
        fill_generator: Generator for fill candidates
        constraint_checker: Checker for constraint satisfaction
        strategy: Hole selection strategy
        max_backtracks: Maximum backtracking attempts
        propagation_enabled: Whether to propagate constraints
        beam_width: Number of candidates to consider per hole

    Example:
        >>> filler = SudokuStyleHoleFiller(generator, checker)
        >>> result = filler.fill(holed_code)
        >>> if result.success:
        ...     print(result.filled_code)
    """

    def __init__(
        self,
        fill_generator: Optional[FillGenerator] = None,
        constraint_checker: Optional[ConstraintChecker] = None,
        strategy: FillStrategy = FillStrategy.MCV,
        max_backtracks: int = 100,
        propagation_enabled: bool = True,
        beam_width: int = 5,
    ):
        """Initialize SudokuStyleHoleFiller.

        Args:
            fill_generator: Generator for fill candidates (uses default if None)
            constraint_checker: Checker for constraints (uses default if None)
            strategy: Hole selection strategy
            max_backtracks: Maximum number of backtracks
            propagation_enabled: Enable constraint propagation
            beam_width: Number of candidates to consider per hole
        """
        self.fill_generator = fill_generator
        self.constraint_checker = constraint_checker
        self.strategy = strategy
        self.max_backtracks = max_backtracks
        self.propagation_enabled = propagation_enabled
        self.beam_width = beam_width

        # Statistics
        self._stats = {
            "fills": 0,
            "backtracks": 0,
            "propagations": 0,
            "total_candidates": 0,
        }

    def fill(
        self,
        holed_code: HoledCode,
        max_iterations: int = 1000,
    ) -> FillResult:
        """Fill all holes in the code using CSP search.

        Uses the configured strategy to select holes and the MCV
        heuristic for efficient search. Backtracks when constraints
        are violated.

        Args:
            holed_code: Code with holes to fill
            max_iterations: Maximum search iterations

        Returns:
            FillResult with filled code or failure information
        """
        start_time = time.perf_counter_ns()

        # Initialize search state
        stack: List[Tuple[HoledCode, List[Tuple[HoleId, str]]]] = [
            (holed_code, [])
        ]
        backtrack_count = 0
        propagation_count = 0
        iteration = 0

        best_result: Optional[Tuple[HoledCode, List[Tuple[HoleId, str]]]] = None
        best_unfilled_count = float("inf")

        while stack and iteration < max_iterations:
            iteration += 1
            current_code, fill_history = stack.pop()

            # Check if done
            if not current_code.has_holes():
                end_time = time.perf_counter_ns()
                return FillResult(
                    success=True,
                    filled_code=current_code.to_string(),
                    fill_history=fill_history,
                    unfilled_holes=[],
                    backtrack_count=backtrack_count,
                    propagation_count=propagation_count,
                    latency_ns=end_time - start_time,
                )

            # Track best partial result
            unfilled_count = len(current_code.unfilled_holes())
            if unfilled_count < best_unfilled_count:
                best_unfilled_count = unfilled_count
                best_result = (current_code, fill_history)

            # Check consistency
            if not current_code.is_consistent():
                backtrack_count += 1
                if backtrack_count > self.max_backtracks:
                    break
                continue

            # Select next hole using strategy
            unfilled = current_code.unfilled_holes()
            if not unfilled:
                continue

            hole = self._select_hole(unfilled, current_code)

            # Generate candidates
            candidates = self._generate_candidates(hole, current_code)
            self._stats["total_candidates"] += len(candidates)

            if not candidates:
                # No valid candidates - backtrack
                backtrack_count += 1
                if backtrack_count > self.max_backtracks:
                    break
                continue

            # Try each candidate (push to stack in reverse order for DFS)
            for candidate in reversed(candidates[: self.beam_width]):
                # Fill the hole
                try:
                    new_code = current_code.fill_hole(hole.id, candidate.value)
                except Exception as e:
                    logger.debug(f"Fill failed: {e}")
                    continue

                new_history = fill_history + [(hole.id, candidate.value)]

                # Propagate constraints if enabled
                if self.propagation_enabled:
                    new_code, prop_count = self._propagate_constraints(
                        new_code, hole.id, candidate.value
                    )
                    propagation_count += prop_count

                stack.append((new_code, new_history))

        end_time = time.perf_counter_ns()

        # Return best partial result if no complete solution
        if best_result:
            code, history = best_result
            return FillResult(
                success=False,
                filled_code=code.to_string(),
                fill_history=history,
                unfilled_holes=[h.id for h in code.unfilled_holes()],
                backtrack_count=backtrack_count,
                propagation_count=propagation_count,
                latency_ns=end_time - start_time,
                metadata={"partial": True, "iterations": iteration},
            )

        return FillResult(
            success=False,
            filled_code=None,
            fill_history=[],
            unfilled_holes=[h.id for h in holed_code.unfilled_holes()],
            backtrack_count=backtrack_count,
            propagation_count=propagation_count,
            latency_ns=end_time - start_time,
            metadata={"iterations": iteration},
        )

    def _select_hole(
        self,
        unfilled: List[Hole[Any]],
        context: HoledCode,
    ) -> Hole[Any]:
        """Select the next hole to fill using the configured strategy.

        MCV Heuristic:
            Select the hole with the fewest valid candidates.
            This "fail-first" approach prunes the search tree early.

        Args:
            unfilled: List of unfilled holes
            context: Current code context

        Returns:
            Selected hole
        """
        if self.strategy == FillStrategy.SEQUENTIAL:
            return unfilled[0]

        if self.strategy == FillStrategy.RANDOM:
            import random
            return random.choice(unfilled)

        if self.strategy in (FillStrategy.MCV, FillStrategy.MRV):
            # Most Constrained Variable: fewest valid candidates
            def candidate_count(hole: Hole[Any]) -> int:
                candidates = self._generate_candidates(hole, context)
                return len(candidates) if candidates else float("inf")

            return min(unfilled, key=candidate_count)

        if self.strategy == FillStrategy.LARGEST_DEGREE:
            # Most connections to unfilled holes
            def dependency_count(hole: Hole[Any]) -> int:
                deps = context.dependencies.get(hole.id, frozenset())
                unfilled_ids = {h.id for h in unfilled}
                return len(deps & unfilled_ids)

            return max(unfilled, key=dependency_count)

        # Default: first unfilled
        return unfilled[0]

    def _generate_candidates(
        self,
        hole: Hole[Any],
        context: HoledCode,
    ) -> List[FillCandidate]:
        """Generate fill candidates for a hole.

        If a fill_generator is provided, uses it. Otherwise returns
        a placeholder list (actual generation requires LLM integration).

        Args:
            hole: The hole to fill
            context: Current code context

        Returns:
            List of FillCandidate ordered by score (best first)
        """
        if self.fill_generator:
            candidates = self.fill_generator.generate_candidates(
                hole, context.template, max_candidates=self.beam_width * 2
            )
        else:
            # Default: check if constraint checker can provide candidates
            # This is a placeholder - real implementation would use LLM
            candidates = self._generate_placeholder_candidates(hole, context)

        # Filter by constraint checker if available
        if self.constraint_checker:
            valid_candidates = []
            for candidate in candidates:
                valid, score, violations = self.constraint_checker.check_fill(
                    hole, candidate.value, context
                )
                if valid or score > 0.5:  # Allow partial matches
                    valid_candidates.append(
                        FillCandidate(
                            value=candidate.value,
                            score=score,
                            constraint_violations=violations,
                            metadata=candidate.metadata,
                        )
                    )
            candidates = valid_candidates

        # Sort by score (best first)
        return sorted(candidates)

    def _generate_placeholder_candidates(
        self,
        hole: Hole[Any],
        context: HoledCode,
    ) -> List[FillCandidate]:
        """Generate placeholder candidates when no generator is provided.

        This is a fallback for testing and demonstration. Real usage
        requires an LLM-based fill_generator.

        Args:
            hole: The hole to fill
            context: Current code context

        Returns:
            List of placeholder candidates
        """
        # Use expected type to generate simple placeholders
        if hole.expected_type:
            type_str = str(hole.expected_type).lower()

            if "int" in type_str:
                return [
                    FillCandidate("0", score=0.9),
                    FillCandidate("1", score=0.85),
                    FillCandidate("-1", score=0.8),
                ]
            elif "str" in type_str:
                return [
                    FillCandidate('""', score=0.9),
                    FillCandidate('"placeholder"', score=0.8),
                ]
            elif "bool" in type_str:
                return [
                    FillCandidate("True", score=0.9),
                    FillCandidate("False", score=0.9),
                ]
            elif "list" in type_str:
                return [
                    FillCandidate("[]", score=0.9),
                ]
            elif "dict" in type_str:
                return [
                    FillCandidate("{}", score=0.9),
                ]
            elif "none" in type_str:
                return [
                    FillCandidate("None", score=1.0),
                ]

        # Generic placeholder
        return [
            FillCandidate("pass", score=0.5),
            FillCandidate("...", score=0.4),
        ]

    def _propagate_constraints(
        self,
        code: HoledCode,
        filled_hole_id: HoleId,
        fill_value: str,
    ) -> Tuple[HoledCode, int]:
        """Propagate constraints after filling a hole.

        When a hole is filled, this may:
        1. Narrow valid options for dependent holes
        2. Reveal forced fills (only one valid option)
        3. Detect inconsistencies early

        Args:
            code: Current code state
            filled_hole_id: ID of the just-filled hole
            fill_value: The fill value

        Returns:
            (updated_code, propagation_count) tuple
        """
        propagation_count = 0
        current_code = code

        # Get dependent holes
        dependent = current_code.dependent_holes(filled_hole_id)
        if not dependent:
            return current_code, 0

        # Propagate to each dependent hole
        for dep_hole in dependent:
            if dep_hole.state != HoleState.EMPTY:
                continue

            # Infer constraint refinement from fill
            inferred_constraint = self._infer_constraint_from_fill(
                current_code.get_hole(filled_hole_id),
                fill_value,
                dep_hole,
            )

            if inferred_constraint is not None:
                # Update dependent hole's constraint
                if dep_hole.constraint is not None:
                    # Meet with existing constraint
                    if hasattr(dep_hole.constraint, "meet"):
                        new_constraint = dep_hole.constraint.meet(inferred_constraint)
                    else:
                        new_constraint = inferred_constraint
                else:
                    new_constraint = inferred_constraint

                current_code = current_code.update_hole_constraint(
                    dep_hole.id, new_constraint
                )
                propagation_count += 1

        self._stats["propagations"] += propagation_count
        return current_code, propagation_count

    def _infer_constraint_from_fill(
        self,
        filled_hole: Optional[Hole[Any]],
        fill_value: str,
        dependent_hole: Hole[Any],
    ) -> Optional[Any]:
        """Infer constraint for dependent hole from a fill.

        Uses TypeConstraintInferencer to analyze the relationship between
        filled and dependent holes and determine appropriate constraints.

        The inference is based on:
        1. The type of the fill value (literals, expressions)
        2. The relationship between holes (return type, argument, etc.)
        3. The expected types from hole constraints

        Args:
            filled_hole: The hole that was filled
            fill_value: The fill value
            dependent_hole: The dependent hole to constrain

        Returns:
            Inferred constraint (TypeConstraint) or None if no constraint inferred
        """
        if filled_hole is None:
            return None

        # Use TypeConstraintInferencer for real constraint inference
        try:
            from .generators import TypeConstraintInferencer
            inferencer = TypeConstraintInferencer()
            return inferencer.infer_constraint_from_fill(
                filled_hole, fill_value, dependent_hole
            )
        except ImportError:
            logger.debug("TypeConstraintInferencer not available, skipping inference")
            return None
        except Exception as e:
            logger.debug(f"Constraint inference failed: {e}")
            return None

    def valid_fills(
        self,
        hole: Hole[Any],
        context: HoledCode,
    ) -> List[str]:
        """Get valid fill values for a hole.

        Convenience method that returns just the fill values.

        Args:
            hole: The hole to fill
            context: Current code context

        Returns:
            List of valid fill values
        """
        candidates = self._generate_candidates(hole, context)
        return [c.value for c in candidates]

    def get_stats(self) -> Dict[str, Any]:
        """Get filler statistics."""
        return self._stats.copy()

    def reset_stats(self) -> None:
        """Reset filler statistics."""
        self._stats = {
            "fills": 0,
            "backtracks": 0,
            "propagations": 0,
            "total_candidates": 0,
        }


def fill_with_mcv_heuristic(
    code: HoledCode,
    fill_generator: Optional[FillGenerator] = None,
    constraint_checker: Optional[ConstraintChecker] = None,
    max_backtracks: int = 100,
) -> FillResult:
    """Convenience function to fill code using MCV heuristic.

    Creates a SudokuStyleHoleFiller with MCV strategy and fills the code.

    Args:
        code: HoledCode to fill
        fill_generator: Optional fill generator
        constraint_checker: Optional constraint checker
        max_backtracks: Maximum backtrack attempts

    Returns:
        FillResult with filled code or failure information

    Example:
        >>> result = fill_with_mcv_heuristic(holed_code)
        >>> if result.success:
        ...     print(result.filled_code)
    """
    filler = SudokuStyleHoleFiller(
        fill_generator=fill_generator,
        constraint_checker=constraint_checker,
        strategy=FillStrategy.MCV,
        max_backtracks=max_backtracks,
    )
    return filler.fill(code)
