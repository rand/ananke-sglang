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
"""Trajectory tracking for efficient backtracking in code generation search.

This module provides trie-based trajectory tracking for code generation,
enabling efficient rollback during CSP search. Key features:

1. Shared Prefixes: Common generation prefixes are stored once
2. O(1) Checkpoint: Create checkpoints without copying state
3. Efficient Rollback: Jump to any previous state quickly

Based on ROCODE (EMNLP 2024) which uses trie trees for generation
trajectory management.

References:
- ROCODE: Backtracking for Code Generation (EMNLP 2024)
- AdapTrack: Dynamic Backtracking (arXiv 2024)
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generic, Iterator, List, Optional, Tuple, TypeVar

# Handle imports for both package and standalone usage
try:
    from ..holes.hole import HoleId
except ImportError:
    # Standalone mode - add package root to path
    _SEARCH_DIR = Path(__file__).parent
    _ANANKE_ROOT = _SEARCH_DIR.parent
    if str(_ANANKE_ROOT) not in sys.path:
        sys.path.insert(0, str(_ANANKE_ROOT))
    from holes.hole import HoleId

# Type variable for state
S = TypeVar("S")


@dataclass
class TrajectoryNode(Generic[S]):
    """A node in the trajectory trie.

    Each node represents a decision point (hole fill) in the generation
    trajectory. Nodes share common prefixes via parent pointers.

    Attributes:
        hole_id: The hole that was filled at this node
        fill_value: The value used to fill the hole
        state: Associated state at this node (e.g., HoledCode)
        parent: Parent node (None for root)
        children: Child nodes keyed by fill value
        depth: Depth in the trie (number of fills)
        score: Constraint satisfaction score at this node
        metadata: Additional node information
    """

    hole_id: Optional[HoleId] = None
    fill_value: Optional[str] = None
    state: Optional[S] = None
    parent: Optional[TrajectoryNode[S]] = None
    children: Dict[str, TrajectoryNode[S]] = field(default_factory=dict)
    depth: int = 0
    score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Set depth from parent."""
        if self.parent is not None:
            self.depth = self.parent.depth + 1

    def add_child(
        self,
        hole_id: HoleId,
        fill_value: str,
        state: S,
        score: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TrajectoryNode[S]:
        """Add a child node representing a fill decision.

        Args:
            hole_id: The hole being filled
            fill_value: The fill value
            state: State after filling
            score: Constraint satisfaction score
            metadata: Additional metadata

        Returns:
            The new child node
        """
        # Use compound key for uniqueness
        key = f"{hole_id}:{fill_value}"

        if key in self.children:
            # Update existing child
            child = self.children[key]
            child.state = state
            child.score = score
            if metadata:
                child.metadata.update(metadata)
            return child

        # Create new child
        child = TrajectoryNode(
            hole_id=hole_id,
            fill_value=fill_value,
            state=state,
            parent=self,
            score=score,
            metadata=metadata or {},
        )
        self.children[key] = child
        return child

    def get_child(
        self,
        hole_id: HoleId,
        fill_value: str,
    ) -> Optional[TrajectoryNode[S]]:
        """Get a child node by hole ID and fill value.

        Args:
            hole_id: The hole ID
            fill_value: The fill value

        Returns:
            Child node or None if not found
        """
        key = f"{hole_id}:{fill_value}"
        return self.children.get(key)

    def path_to_root(self) -> List[TrajectoryNode[S]]:
        """Get the path from this node to the root.

        Returns:
            List of nodes from this node to root (inclusive)
        """
        path = []
        current: Optional[TrajectoryNode[S]] = self
        while current is not None:
            path.append(current)
            current = current.parent
        return path

    def fill_history(self) -> List[Tuple[HoleId, str]]:
        """Get the fill history leading to this node.

        Returns:
            List of (hole_id, fill_value) tuples from root to here
        """
        history = []
        for node in reversed(self.path_to_root()):
            if node.hole_id is not None and node.fill_value is not None:
                history.append((node.hole_id, node.fill_value))
        return history

    def is_root(self) -> bool:
        """Check if this is the root node."""
        return self.parent is None

    def is_leaf(self) -> bool:
        """Check if this is a leaf node (no children)."""
        return len(self.children) == 0

    def subtree_size(self) -> int:
        """Get the size of the subtree rooted at this node."""
        return 1 + sum(child.subtree_size() for child in self.children.values())

    def subtree_leaves(self) -> List[TrajectoryNode[S]]:
        """Get all leaf nodes in the subtree."""
        if self.is_leaf():
            return [self]
        leaves = []
        for child in self.children.values():
            leaves.extend(child.subtree_leaves())
        return leaves


@dataclass
class Trajectory(Generic[S]):
    """A single path through the trajectory trie.

    Represents a specific sequence of fills from root to a node.
    Provides efficient operations for extending and backtracking.

    Attributes:
        current: Current node in the trajectory
        trie: The underlying TrajectoryTrie
    """

    current: TrajectoryNode[S]
    trie: TrajectoryTrie[S]

    def extend(
        self,
        hole_id: HoleId,
        fill_value: str,
        state: S,
        score: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Trajectory[S]:
        """Extend the trajectory with a new fill.

        Creates a new Trajectory pointing to the new node.
        Does not modify this Trajectory.

        Args:
            hole_id: The hole being filled
            fill_value: The fill value
            state: State after filling
            score: Constraint satisfaction score
            metadata: Additional metadata

        Returns:
            New Trajectory at the extended position
        """
        child = self.current.add_child(
            hole_id=hole_id,
            fill_value=fill_value,
            state=state,
            score=score,
            metadata=metadata,
        )
        return Trajectory(current=child, trie=self.trie)

    def backtrack(self, steps: int = 1) -> Trajectory[S]:
        """Backtrack by the specified number of steps.

        Args:
            steps: Number of steps to backtrack

        Returns:
            New Trajectory at the backtracked position

        Raises:
            ValueError: If trying to backtrack past root
        """
        current = self.current
        for _ in range(steps):
            if current.parent is None:
                raise ValueError("Cannot backtrack past root")
            current = current.parent
        return Trajectory(current=current, trie=self.trie)

    def backtrack_to_depth(self, depth: int) -> Trajectory[S]:
        """Backtrack to a specific depth.

        Args:
            depth: Target depth (0 = root)

        Returns:
            New Trajectory at the specified depth

        Raises:
            ValueError: If depth is greater than current depth
        """
        if depth > self.current.depth:
            raise ValueError(f"Cannot backtrack forward: {depth} > {self.current.depth}")
        return self.backtrack(self.current.depth - depth)

    def backtrack_to_root(self) -> Trajectory[S]:
        """Backtrack to the root node.

        Returns:
            New Trajectory at root
        """
        return Trajectory(current=self.trie.root, trie=self.trie)

    @property
    def depth(self) -> int:
        """Get current depth (number of fills)."""
        return self.current.depth

    @property
    def state(self) -> Optional[S]:
        """Get current state."""
        return self.current.state

    @property
    def score(self) -> float:
        """Get current score."""
        return self.current.score

    def fill_history(self) -> List[Tuple[HoleId, str]]:
        """Get fill history leading to current position."""
        return self.current.fill_history()

    def cumulative_score(self) -> float:
        """Get product of scores along path (for beam search)."""
        score = 1.0
        for node in self.current.path_to_root():
            score *= node.score
        return score


@dataclass
class TrajectoryTrie(Generic[S]):
    """Trie for tracking generation trajectories.

    Provides efficient storage and navigation of code generation
    trajectories. Common prefixes are shared to save memory and
    enable fast rollback.

    Attributes:
        root: Root node of the trie
        _node_count: Total number of nodes

    Example:
        >>> trie = TrajectoryTrie()
        >>> traj = trie.new_trajectory(initial_state)
        >>> traj = traj.extend(hole_id, "value", new_state)
        >>> traj = traj.backtrack()  # Efficient rollback
    """

    root: TrajectoryNode[S] = field(default_factory=lambda: TrajectoryNode())
    _node_count: int = 1

    def new_trajectory(self, initial_state: S) -> Trajectory[S]:
        """Create a new trajectory starting from root.

        Args:
            initial_state: Initial state at root

        Returns:
            New Trajectory at root
        """
        self.root.state = initial_state
        return Trajectory(current=self.root, trie=self)

    def checkpoint(self, trajectory: Trajectory[S]) -> TrajectoryNode[S]:
        """Create a checkpoint at the current trajectory position.

        Checkpoints are O(1) since we just return the node reference.
        The state is already stored in the trie.

        Args:
            trajectory: Current trajectory

        Returns:
            Node representing the checkpoint
        """
        return trajectory.current

    def restore(
        self,
        checkpoint: TrajectoryNode[S],
    ) -> Trajectory[S]:
        """Restore to a checkpoint.

        O(1) operation since we just create a new Trajectory
        pointing to the checkpoint node.

        Args:
            checkpoint: Node to restore to

        Returns:
            Trajectory at checkpoint position
        """
        return Trajectory(current=checkpoint, trie=self)

    def best_leaf(self) -> Optional[Trajectory[S]]:
        """Get trajectory to the best leaf node by score.

        Finds the leaf with highest cumulative score along its path.

        Returns:
            Trajectory to best leaf, or None if only root
        """
        leaves = self.root.subtree_leaves()
        if not leaves or (len(leaves) == 1 and leaves[0].is_root()):
            return None

        def path_score(node: TrajectoryNode[S]) -> float:
            score = 1.0
            for n in node.path_to_root():
                score *= n.score
            return score

        best = max(leaves, key=path_score)
        return Trajectory(current=best, trie=self)

    def all_leaves(self) -> List[Trajectory[S]]:
        """Get trajectories to all leaf nodes.

        Returns:
            List of Trajectories, one per leaf
        """
        return [
            Trajectory(current=leaf, trie=self)
            for leaf in self.root.subtree_leaves()
        ]

    def prune_below_score(self, threshold: float) -> int:
        """Prune nodes with score below threshold.

        Removes subtrees where the cumulative score falls below
        the threshold. Useful for beam search pruning.

        Args:
            threshold: Minimum cumulative score to keep

        Returns:
            Number of nodes pruned
        """
        pruned = 0

        def prune_recursive(node: TrajectoryNode[S], cum_score: float) -> bool:
            """Returns True if node should be pruned."""
            nonlocal pruned
            node_score = cum_score * node.score

            # Prune children first
            to_remove = []
            for key, child in node.children.items():
                if prune_recursive(child, node_score):
                    to_remove.append(key)
                    pruned += child.subtree_size()

            for key in to_remove:
                del node.children[key]

            # Prune this node if below threshold and is leaf
            return node_score < threshold and node.is_leaf() and not node.is_root()

        prune_recursive(self.root, 1.0)
        return pruned

    @property
    def node_count(self) -> int:
        """Get total number of nodes in the trie."""
        return self.root.subtree_size()

    def depth_stats(self) -> Dict[str, Any]:
        """Get statistics about trie depth.

        Returns:
            Dictionary with min, max, avg depth of leaves
        """
        leaves = self.root.subtree_leaves()
        if not leaves:
            return {"min": 0, "max": 0, "avg": 0.0, "count": 0}

        depths = [leaf.depth for leaf in leaves]
        return {
            "min": min(depths),
            "max": max(depths),
            "avg": sum(depths) / len(depths),
            "count": len(leaves),
        }


def create_trajectory_trie(initial_state: Any = None) -> Tuple[TrajectoryTrie, Trajectory]:
    """Convenience function to create a trajectory trie with initial trajectory.

    Args:
        initial_state: Initial state at root

    Returns:
        (trie, trajectory) tuple

    Example:
        >>> trie, traj = create_trajectory_trie(holed_code)
        >>> traj = traj.extend(hole_id, "value", new_state)
    """
    trie: TrajectoryTrie = TrajectoryTrie()
    trajectory = trie.new_trajectory(initial_state)
    return trie, trajectory
