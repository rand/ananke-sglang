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
"""Reachability analysis for control flow graphs.

Provides algorithms for:
- Forward reachability: What blocks are reachable from entry?
- Backward reachability: What blocks can reach exit?
- Dead code detection: What blocks are unreachable?
- Return path analysis: Do all paths lead to exit?
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set

from .cfg import BasicBlock, CFGSketch, EdgeKind


@dataclass
class ReachabilityResult:
    """Result of reachability analysis.

    Attributes:
        reachable_from_entry: Blocks reachable from entry
        can_reach_exit: Blocks that can reach an exit
        dead_blocks: Blocks that are unreachable
        all_paths_return: Whether all paths from entry reach exit
        has_infinite_loop: Whether the CFG contains potential infinite loops
    """

    reachable_from_entry: FrozenSet[str] = field(default_factory=frozenset)
    can_reach_exit: FrozenSet[str] = field(default_factory=frozenset)
    dead_blocks: FrozenSet[str] = field(default_factory=frozenset)
    all_paths_return: bool = True
    has_infinite_loop: bool = False


class ReachabilityAnalyzer:
    """Analyzer for reachability properties of a CFG.

    Provides methods for computing various reachability properties:
    - Forward reachability from entry
    - Backward reachability to exit
    - Dead code detection
    - Return path analysis
    """

    def __init__(self, cfg: CFGSketch):
        """Initialize the analyzer.

        Args:
            cfg: The control flow graph to analyze
        """
        self._cfg = cfg
        self._forward_cache: Optional[FrozenSet[str]] = None
        self._backward_cache: Optional[FrozenSet[str]] = None

    def analyze(self) -> ReachabilityResult:
        """Perform full reachability analysis.

        Returns:
            ReachabilityResult with all computed properties
        """
        reachable = self.forward_reachable()
        can_exit = self.backward_reachable()

        all_blocks = frozenset(self._cfg.blocks.keys())
        dead = all_blocks - reachable

        # Check if all reachable blocks can reach exit
        all_paths_return = reachable <= can_exit

        # Check for infinite loops
        has_infinite_loop = self._detect_infinite_loops(reachable, can_exit)

        return ReachabilityResult(
            reachable_from_entry=reachable,
            can_reach_exit=can_exit,
            dead_blocks=dead,
            all_paths_return=all_paths_return,
            has_infinite_loop=has_infinite_loop,
        )

    def forward_reachable(self) -> FrozenSet[str]:
        """Compute blocks reachable from entry via forward traversal.

        Uses BFS from entry block, following successor edges.

        Returns:
            FrozenSet of reachable block IDs
        """
        if self._forward_cache is not None:
            return self._forward_cache

        if not self._cfg.entry_id:
            return frozenset()

        reachable: Set[str] = set()
        worklist = deque([self._cfg.entry_id])

        while worklist:
            block_id = worklist.popleft()
            if block_id in reachable:
                continue
            reachable.add(block_id)

            block = self._cfg.get_block(block_id)
            if block:
                for succ in block.successors:
                    if succ not in reachable:
                        worklist.append(succ)

        self._forward_cache = frozenset(reachable)
        return self._forward_cache

    def backward_reachable(self) -> FrozenSet[str]:
        """Compute blocks that can reach exit via backward traversal.

        Uses BFS from exit blocks, following predecessor edges.

        Returns:
            FrozenSet of block IDs that can reach exit
        """
        if self._backward_cache is not None:
            return self._backward_cache

        if not self._cfg.exit_ids:
            return frozenset()

        reachable: Set[str] = set()
        worklist = deque(self._cfg.exit_ids)

        while worklist:
            block_id = worklist.popleft()
            if block_id in reachable:
                continue
            reachable.add(block_id)

            block = self._cfg.get_block(block_id)
            if block:
                for pred in block.predecessors:
                    if pred not in reachable:
                        worklist.append(pred)

        self._backward_cache = frozenset(reachable)
        return self._backward_cache

    def is_reachable(self, block_id: str) -> bool:
        """Check if a block is reachable from entry.

        Args:
            block_id: The block ID to check

        Returns:
            True if reachable from entry
        """
        return block_id in self.forward_reachable()

    def can_reach_exit(self, block_id: str) -> bool:
        """Check if a block can reach exit.

        Args:
            block_id: The block ID to check

        Returns:
            True if can reach exit
        """
        return block_id in self.backward_reachable()

    def is_dead(self, block_id: str) -> bool:
        """Check if a block is dead (unreachable).

        Args:
            block_id: The block ID to check

        Returns:
            True if dead
        """
        return not self.is_reachable(block_id)

    def is_path_blocked(self, block_id: str) -> bool:
        """Check if a block cannot reach exit.

        A blocked path indicates code that runs but never returns.

        Args:
            block_id: The block ID to check

        Returns:
            True if path is blocked
        """
        return self.is_reachable(block_id) and not self.can_reach_exit(block_id)

    def _detect_infinite_loops(
        self,
        reachable: FrozenSet[str],
        can_exit: FrozenSet[str],
    ) -> bool:
        """Detect potential infinite loops.

        An infinite loop exists if there's a cycle among blocks that
        cannot reach exit.

        Args:
            reachable: Blocks reachable from entry
            can_exit: Blocks that can reach exit

        Returns:
            True if potential infinite loop detected
        """
        # Blocks that are reachable but cannot exit
        trapped = reachable - can_exit

        if not trapped:
            return False

        # Check if any trapped blocks form a cycle
        for block_id in trapped:
            block = self._cfg.get_block(block_id)
            if not block:
                continue

            # If a trapped block has successors in trapped, it's a loop
            for succ in block.successors:
                if succ in trapped:
                    return True

        return False

    def get_dead_blocks(self) -> List[BasicBlock]:
        """Get all dead blocks.

        Returns:
            List of dead BasicBlocks
        """
        reachable = self.forward_reachable()
        return [
            block for block_id, block in self._cfg.blocks.items()
            if block_id not in reachable
        ]

    def get_blocked_paths(self) -> List[BasicBlock]:
        """Get blocks on blocked paths.

        These are blocks that are reachable but cannot reach exit.

        Returns:
            List of BasicBlocks on blocked paths
        """
        reachable = self.forward_reachable()
        can_exit = self.backward_reachable()
        blocked = reachable - can_exit

        return [
            block for block_id, block in self._cfg.blocks.items()
            if block_id in blocked
        ]


def compute_dominators(cfg: CFGSketch) -> Dict[str, FrozenSet[str]]:
    """Compute dominators for each block.

    A block A dominates block B if every path from entry to B
    passes through A.

    Args:
        cfg: The control flow graph

    Returns:
        Dict mapping block ID to its set of dominators
    """
    if not cfg.entry_id:
        return {}

    all_blocks = frozenset(cfg.blocks.keys())
    dominators: Dict[str, Set[str]] = {}

    # Entry dominates only itself
    dominators[cfg.entry_id] = {cfg.entry_id}

    # Initialize all others to all blocks
    for block_id in cfg.blocks:
        if block_id != cfg.entry_id:
            dominators[block_id] = set(all_blocks)

    # Iterate until fixpoint
    changed = True
    while changed:
        changed = False
        for block_id in cfg.blocks:
            if block_id == cfg.entry_id:
                continue

            block = cfg.get_block(block_id)
            if not block or not block.predecessors:
                continue

            # Dom(n) = {n} ∪ (∩ Dom(p) for p in predecessors)
            pred_doms = [dominators.get(p, set()) for p in block.predecessors]
            if pred_doms:
                new_dom = set.intersection(*pred_doms) | {block_id}
            else:
                new_dom = {block_id}

            if new_dom != dominators[block_id]:
                dominators[block_id] = new_dom
                changed = True

    return {k: frozenset(v) for k, v in dominators.items()}


def compute_post_dominators(cfg: CFGSketch) -> Dict[str, FrozenSet[str]]:
    """Compute post-dominators for each block.

    A block A post-dominates block B if every path from B to exit
    passes through A.

    Args:
        cfg: The control flow graph

    Returns:
        Dict mapping block ID to its set of post-dominators
    """
    if not cfg.exit_ids:
        return {}

    all_blocks = frozenset(cfg.blocks.keys())
    post_dominators: Dict[str, Set[str]] = {}

    # Exits post-dominate only themselves
    for exit_id in cfg.exit_ids:
        post_dominators[exit_id] = {exit_id}

    # Initialize all others to all blocks
    for block_id in cfg.blocks:
        if block_id not in cfg.exit_ids:
            post_dominators[block_id] = set(all_blocks)

    # Iterate until fixpoint
    changed = True
    while changed:
        changed = False
        for block_id in cfg.blocks:
            if block_id in cfg.exit_ids:
                continue

            block = cfg.get_block(block_id)
            if not block or not block.successors:
                continue

            # PostDom(n) = {n} ∪ (∩ PostDom(s) for s in successors)
            succ_pdoms = [post_dominators.get(s, set()) for s in block.successors]
            if succ_pdoms:
                new_pdom = set.intersection(*succ_pdoms) | {block_id}
            else:
                new_pdom = {block_id}

            if new_pdom != post_dominators[block_id]:
                post_dominators[block_id] = new_pdom
                changed = True

    return {k: frozenset(v) for k, v in post_dominators.items()}


def find_loop_bodies(cfg: CFGSketch) -> Dict[str, FrozenSet[str]]:
    """Find the body of each loop.

    For each back edge (to a loop header), find all blocks
    that are part of that loop.

    Args:
        cfg: The control flow graph

    Returns:
        Dict mapping loop header ID to its body block IDs
    """
    loops: Dict[str, Set[str]] = {}

    for edge in cfg.edges:
        if edge.kind != EdgeKind.LOOP_BACK:
            continue

        header = edge.target

        # Find all blocks in this loop via backward traversal
        # from the back edge source to the header
        body: Set[str] = {header}
        worklist = [edge.source]

        while worklist:
            block_id = worklist.pop()
            if block_id in body:
                continue
            body.add(block_id)

            block = cfg.get_block(block_id)
            if block:
                for pred in block.predecessors:
                    if pred not in body:
                        worklist.append(pred)

        loops[header] = body

    return {k: frozenset(v) for k, v in loops.items()}
