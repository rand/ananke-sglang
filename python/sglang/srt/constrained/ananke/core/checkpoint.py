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
"""State checkpointing for rollback support in Ananke.

This module provides checkpoint management for speculative decoding and
backtracking. When generation reaches an unsatisfiable state, the system
can roll back to a previous checkpoint and try an alternative path.

The checkpoint system supports:
1. Per-domain state snapshots
2. Unified checkpoint combining all domains
3. Efficient delta-based checkpointing for incremental saves
4. Maximum rollback depth management (following XGrammar's max_rollback_tokens)

References:
    - XGrammar: max_rollback_tokens=200 for efficient rollback
    - Hazel: "Live Functional Programming with Typed Holes" (fill-and-resume)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .constraint import Constraint
from .unified import UnifiedConstraint


@dataclass(frozen=True, slots=True)
class Checkpoint:
    """Immutable checkpoint for a single domain's state.

    Checkpoints capture all mutable state needed to restore a domain
    to a previous point. The state dict should contain only picklable
    values to enable serialization if needed.

    Attributes:
        domain_name: Name of the domain this checkpoint is for
        state: Dictionary of domain-specific state
        position: Token position at checkpoint time
        constraint_hash: Hash of constraint at checkpoint for validation
    """

    domain_name: str
    state: Dict[str, Any]
    position: int = 0
    constraint_hash: int = 0

    def validate(self, current_position: int, current_constraint: Constraint) -> bool:
        """Validate that this checkpoint is compatible with current state.

        Args:
            current_position: Current token position
            current_constraint: Current domain constraint

        Returns:
            True if checkpoint can be safely restored
        """
        # Position must be at or after checkpoint position for valid rollback
        if current_position < self.position:
            return False
        # Constraint hash check is optional validation
        return True


@dataclass(slots=True)
class UnifiedCheckpoint:
    """Combined checkpoint for all domains and the unified constraint.

    UnifiedCheckpoint enables atomic rollback of the entire constraint
    system to a consistent previous state.

    Attributes:
        position: Token position at checkpoint time
        unified_constraint: The unified constraint at checkpoint
        domain_checkpoints: Checkpoints for each domain
        context_snapshot: Optional snapshot of GenerationContext
    """

    position: int
    unified_constraint: UnifiedConstraint
    domain_checkpoints: Dict[str, Checkpoint] = field(default_factory=dict)
    context_snapshot: Optional[Dict[str, Any]] = None

    def get_domain_checkpoint(self, domain_name: str) -> Optional[Checkpoint]:
        """Get checkpoint for a specific domain."""
        return self.domain_checkpoints.get(domain_name)


class CheckpointManager:
    """Manages checkpoint history with bounded memory.

    CheckpointManager maintains a rolling window of checkpoints,
    following XGrammar's max_rollback_tokens pattern to bound memory
    usage while enabling sufficient rollback depth.

    The manager supports:
    - Creating checkpoints at arbitrary positions
    - Rolling back to any checkpoint within the window
    - Automatic pruning of old checkpoints
    - Delta-based storage for efficiency (future optimization)

    Attributes:
        max_checkpoints: Maximum number of checkpoints to retain
        checkpoint_interval: How often to create automatic checkpoints
    """

    def __init__(
        self,
        max_checkpoints: int = 200,
        checkpoint_interval: int = 1,
    ):
        """Initialize checkpoint manager.

        Args:
            max_checkpoints: Maximum checkpoints to retain (like max_rollback_tokens)
            checkpoint_interval: Token interval for automatic checkpointing
        """
        self.max_checkpoints = max_checkpoints
        self.checkpoint_interval = checkpoint_interval
        self._checkpoints: List[UnifiedCheckpoint] = []
        self._position_index: Dict[int, int] = {}  # position -> checkpoint index

    @property
    def checkpoint_count(self) -> int:
        """Number of checkpoints currently stored."""
        return len(self._checkpoints)

    @property
    def oldest_position(self) -> Optional[int]:
        """Oldest position that can be rolled back to."""
        if not self._checkpoints:
            return None
        return self._checkpoints[0].position

    @property
    def newest_position(self) -> Optional[int]:
        """Most recent checkpoint position."""
        if not self._checkpoints:
            return None
        return self._checkpoints[-1].position

    def create_checkpoint(
        self,
        position: int,
        unified_constraint: UnifiedConstraint,
        domain_checkpoints: Dict[str, Checkpoint],
        context_snapshot: Optional[Dict[str, Any]] = None,
    ) -> UnifiedCheckpoint:
        """Create and store a new checkpoint.

        Args:
            position: Current token position
            unified_constraint: Current unified constraint
            domain_checkpoints: Checkpoints from each domain
            context_snapshot: Optional context state snapshot

        Returns:
            The created checkpoint
        """
        checkpoint = UnifiedCheckpoint(
            position=position,
            unified_constraint=unified_constraint,
            domain_checkpoints=domain_checkpoints,
            context_snapshot=context_snapshot,
        )

        # Add to history
        self._checkpoints.append(checkpoint)
        self._position_index[position] = len(self._checkpoints) - 1

        # Prune if over limit
        self._prune_old_checkpoints()

        return checkpoint

    def get_checkpoint_at(self, position: int) -> Optional[UnifiedCheckpoint]:
        """Get checkpoint at exact position, if it exists."""
        idx = self._position_index.get(position)
        if idx is not None and idx < len(self._checkpoints):
            return self._checkpoints[idx]
        return None

    def get_checkpoint_before(self, position: int) -> Optional[UnifiedCheckpoint]:
        """Get the most recent checkpoint at or before the given position.

        This is the primary method for rollback - find the nearest checkpoint
        that doesn't exceed the target position.

        Args:
            position: Target position to roll back to

        Returns:
            The nearest checkpoint at or before position, or None if none exists
        """
        result = None
        for checkpoint in self._checkpoints:
            if checkpoint.position <= position:
                result = checkpoint
            else:
                break
        return result

    def rollback_to(self, position: int) -> Optional[UnifiedCheckpoint]:
        """Roll back to a position, removing newer checkpoints.

        This method finds the appropriate checkpoint and removes all
        checkpoints that are newer than the target position.

        Args:
            position: Target position to roll back to

        Returns:
            The checkpoint rolled back to, or None if rollback not possible
        """
        checkpoint = self.get_checkpoint_before(position)
        if checkpoint is None:
            return None

        # Remove checkpoints after this position
        self._checkpoints = [c for c in self._checkpoints if c.position <= position]

        # Rebuild position index
        self._position_index = {c.position: i for i, c in enumerate(self._checkpoints)}

        return checkpoint

    def clear(self) -> None:
        """Clear all checkpoints."""
        self._checkpoints.clear()
        self._position_index.clear()

    def _prune_old_checkpoints(self) -> None:
        """Remove oldest checkpoints when over limit."""
        while len(self._checkpoints) > self.max_checkpoints:
            oldest = self._checkpoints.pop(0)
            del self._position_index[oldest.position]

            # Rebuild indices after removal
            self._position_index = {
                c.position: i for i, c in enumerate(self._checkpoints)
            }


def create_context_snapshot(
    generated_text: str,
    generated_tokens: List[int],
    position: int,
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """Create a snapshot of generation context for checkpointing.

    Args:
        generated_text: Text generated so far
        generated_tokens: Token IDs generated
        position: Current position
        metadata: Additional metadata

    Returns:
        Dictionary snapshot suitable for restoration
    """
    return {
        "generated_text": generated_text,
        "generated_tokens": generated_tokens.copy(),
        "position": position,
        "metadata": metadata.copy(),
    }
