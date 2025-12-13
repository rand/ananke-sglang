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
"""Control Flow Graph (CFG) representation.

Provides a lightweight CFG sketch for tracking control flow during generation:
- BasicBlock: A sequence of statements with single entry/exit
- CFGEdge: A directed edge between blocks (with optional condition)
- CFGSketch: The complete graph with entry/exit points
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, FrozenSet, Iterator, List, Optional, Set, Tuple

from .constraint import CodePoint


class EdgeKind(Enum):
    """Kind of control flow edge."""

    SEQUENTIAL = auto()  # Normal sequential flow
    CONDITIONAL_TRUE = auto()  # True branch of condition
    CONDITIONAL_FALSE = auto()  # False branch of condition
    LOOP_BACK = auto()  # Back edge to loop header
    LOOP_EXIT = auto()  # Exit edge from loop
    EXCEPTION = auto()  # Exception handling edge
    RETURN = auto()  # Return from function
    BREAK = auto()  # Break from loop
    CONTINUE = auto()  # Continue in loop


@dataclass(frozen=True, slots=True)
class CFGEdge:
    """A directed edge in the CFG.

    Attributes:
        source: Source block ID
        target: Target block ID
        kind: Type of control flow
        condition: Optional condition expression (for conditional edges)
    """

    source: str
    target: str
    kind: EdgeKind = EdgeKind.SEQUENTIAL
    condition: Optional[str] = None

    def __repr__(self) -> str:
        cond = f" [{self.condition}]" if self.condition else ""
        return f"CFGEdge({self.source} -> {self.target}, {self.kind.name}{cond})"


@dataclass
class BasicBlock:
    """A basic block in the CFG.

    A basic block is a sequence of statements with:
    - A single entry point (no jumps into the middle)
    - A single exit point (no jumps out from the middle)

    Attributes:
        id: Unique identifier for this block
        kind: Description of what this block represents
        statements: List of statement descriptions
        predecessors: IDs of blocks that can flow to this one
        successors: IDs of blocks this can flow to
        is_entry: True if this is an entry block
        is_exit: True if this is an exit block
        is_loop_header: True if this is a loop header
        source_lines: Optional source line range
    """

    id: str
    kind: str = "block"
    statements: List[str] = field(default_factory=list)
    predecessors: Set[str] = field(default_factory=set)
    successors: Set[str] = field(default_factory=set)
    is_entry: bool = False
    is_exit: bool = False
    is_loop_header: bool = False
    source_lines: Optional[Tuple[int, int]] = None

    def add_statement(self, stmt: str) -> None:
        """Add a statement to this block."""
        self.statements.append(stmt)

    def to_code_point(self) -> CodePoint:
        """Convert to a CodePoint for constraint tracking."""
        line = self.source_lines[0] if self.source_lines else None
        return CodePoint(label=self.id, kind=self.kind, line=line)

    def __repr__(self) -> str:
        flags = []
        if self.is_entry:
            flags.append("entry")
        if self.is_exit:
            flags.append("exit")
        if self.is_loop_header:
            flags.append("loop_header")
        flag_str = f" ({', '.join(flags)})" if flags else ""
        return f"BasicBlock({self.id}{flag_str})"


@dataclass
class CFGSketch:
    """A sketch of the control flow graph.

    The CFG sketch provides a simplified view of control flow for
    constraint checking during generation. It tracks:
    - Basic blocks with their properties
    - Edges between blocks
    - Entry and exit points

    Attributes:
        blocks: Map from block ID to BasicBlock
        edges: Set of CFGEdge instances
        entry_id: ID of the entry block
        exit_ids: IDs of exit blocks
    """

    blocks: Dict[str, BasicBlock] = field(default_factory=dict)
    edges: Set[CFGEdge] = field(default_factory=set)
    entry_id: Optional[str] = None
    exit_ids: Set[str] = field(default_factory=set)

    def add_block(self, block: BasicBlock) -> None:
        """Add a basic block to the CFG.

        Args:
            block: The block to add
        """
        self.blocks[block.id] = block

        if block.is_entry:
            self.entry_id = block.id
        if block.is_exit:
            self.exit_ids.add(block.id)

    def add_edge(self, edge: CFGEdge) -> None:
        """Add an edge to the CFG.

        Args:
            edge: The edge to add
        """
        self.edges.add(edge)

        # Update block predecessor/successor sets
        if edge.source in self.blocks:
            self.blocks[edge.source].successors.add(edge.target)
        if edge.target in self.blocks:
            self.blocks[edge.target].predecessors.add(edge.source)

    def get_block(self, block_id: str) -> Optional[BasicBlock]:
        """Get a block by ID.

        Args:
            block_id: The block ID

        Returns:
            The BasicBlock or None if not found
        """
        return self.blocks.get(block_id)

    def get_entry(self) -> Optional[BasicBlock]:
        """Get the entry block.

        Returns:
            The entry BasicBlock or None
        """
        if self.entry_id:
            return self.blocks.get(self.entry_id)
        return None

    def get_exits(self) -> List[BasicBlock]:
        """Get all exit blocks.

        Returns:
            List of exit BasicBlocks
        """
        return [self.blocks[eid] for eid in self.exit_ids if eid in self.blocks]

    def get_successors(self, block_id: str) -> List[BasicBlock]:
        """Get successor blocks.

        Args:
            block_id: The source block ID

        Returns:
            List of successor BasicBlocks
        """
        block = self.blocks.get(block_id)
        if not block:
            return []
        return [self.blocks[sid] for sid in block.successors if sid in self.blocks]

    def get_predecessors(self, block_id: str) -> List[BasicBlock]:
        """Get predecessor blocks.

        Args:
            block_id: The target block ID

        Returns:
            List of predecessor BasicBlocks
        """
        block = self.blocks.get(block_id)
        if not block:
            return []
        return [self.blocks[pid] for pid in block.predecessors if pid in self.blocks]

    def get_outgoing_edges(self, block_id: str) -> List[CFGEdge]:
        """Get outgoing edges from a block.

        Args:
            block_id: The source block ID

        Returns:
            List of outgoing CFGEdges
        """
        return [e for e in self.edges if e.source == block_id]

    def get_incoming_edges(self, block_id: str) -> List[CFGEdge]:
        """Get incoming edges to a block.

        Args:
            block_id: The target block ID

        Returns:
            List of incoming CFGEdges
        """
        return [e for e in self.edges if e.target == block_id]

    def get_loop_headers(self) -> List[BasicBlock]:
        """Get all loop header blocks.

        Returns:
            List of loop header BasicBlocks
        """
        return [b for b in self.blocks.values() if b.is_loop_header]

    def get_back_edges(self) -> List[CFGEdge]:
        """Get all back edges (edges to loop headers).

        Returns:
            List of back CFGEdges
        """
        return [e for e in self.edges if e.kind == EdgeKind.LOOP_BACK]

    def block_count(self) -> int:
        """Get the number of blocks."""
        return len(self.blocks)

    def edge_count(self) -> int:
        """Get the number of edges."""
        return len(self.edges)

    def iter_blocks(self) -> Iterator[BasicBlock]:
        """Iterate over all blocks."""
        return iter(self.blocks.values())

    def iter_edges(self) -> Iterator[CFGEdge]:
        """Iterate over all edges."""
        return iter(self.edges)

    def to_code_points(self) -> FrozenSet[CodePoint]:
        """Convert all blocks to CodePoints.

        Returns:
            FrozenSet of CodePoints
        """
        return frozenset(b.to_code_point() for b in self.blocks.values())

    def __repr__(self) -> str:
        return f"CFGSketch(blocks={self.block_count()}, edges={self.edge_count()})"


class CFGBuilder:
    """Builder for constructing CFG sketches.

    Provides a fluent interface for building CFGs:

        cfg = (CFGBuilder()
            .entry("start")
            .block("body")
            .block("end", is_exit=True)
            .edge("start", "body")
            .edge("body", "end")
            .build())
    """

    def __init__(self) -> None:
        """Initialize the builder."""
        self._blocks: Dict[str, BasicBlock] = {}
        self._edges: Set[CFGEdge] = set()
        self._entry_id: Optional[str] = None
        self._exit_ids: Set[str] = set()

    def entry(
        self,
        block_id: str,
        kind: str = "entry",
        statements: Optional[List[str]] = None,
    ) -> CFGBuilder:
        """Add an entry block.

        Args:
            block_id: Unique block ID
            kind: Block kind description
            statements: Optional list of statements

        Returns:
            self for chaining
        """
        block = BasicBlock(
            id=block_id,
            kind=kind,
            statements=statements or [],
            is_entry=True,
        )
        self._blocks[block_id] = block
        self._entry_id = block_id
        return self

    def exit(
        self,
        block_id: str,
        kind: str = "exit",
        statements: Optional[List[str]] = None,
    ) -> CFGBuilder:
        """Add an exit block.

        Args:
            block_id: Unique block ID
            kind: Block kind description
            statements: Optional list of statements

        Returns:
            self for chaining
        """
        block = BasicBlock(
            id=block_id,
            kind=kind,
            statements=statements or [],
            is_exit=True,
        )
        self._blocks[block_id] = block
        self._exit_ids.add(block_id)
        return self

    def block(
        self,
        block_id: str,
        kind: str = "block",
        statements: Optional[List[str]] = None,
        is_loop_header: bool = False,
        is_entry: bool = False,
        is_exit: bool = False,
    ) -> CFGBuilder:
        """Add a basic block.

        Args:
            block_id: Unique block ID
            kind: Block kind description
            statements: Optional list of statements
            is_loop_header: Whether this is a loop header
            is_entry: Whether this is an entry block
            is_exit: Whether this is an exit block

        Returns:
            self for chaining
        """
        block = BasicBlock(
            id=block_id,
            kind=kind,
            statements=statements or [],
            is_loop_header=is_loop_header,
            is_entry=is_entry,
            is_exit=is_exit,
        )
        self._blocks[block_id] = block

        if is_entry:
            self._entry_id = block_id
        if is_exit:
            self._exit_ids.add(block_id)

        return self

    def edge(
        self,
        source: str,
        target: str,
        kind: EdgeKind = EdgeKind.SEQUENTIAL,
        condition: Optional[str] = None,
    ) -> CFGBuilder:
        """Add an edge.

        Args:
            source: Source block ID
            target: Target block ID
            kind: Edge kind
            condition: Optional condition expression

        Returns:
            self for chaining
        """
        edge = CFGEdge(source=source, target=target, kind=kind, condition=condition)
        self._edges.add(edge)
        return self

    def conditional(
        self,
        source: str,
        true_target: str,
        false_target: str,
        condition: str,
    ) -> CFGBuilder:
        """Add conditional edges.

        Args:
            source: Source block ID
            true_target: True branch target
            false_target: False branch target
            condition: Condition expression

        Returns:
            self for chaining
        """
        self.edge(source, true_target, EdgeKind.CONDITIONAL_TRUE, condition)
        self.edge(source, false_target, EdgeKind.CONDITIONAL_FALSE, f"not ({condition})")
        return self

    def loop(
        self,
        header: str,
        body: str,
        exit: str,
        condition: str,
    ) -> CFGBuilder:
        """Add loop structure.

        Args:
            header: Loop header block ID
            body: Loop body block ID
            exit: Loop exit block ID
            condition: Loop condition expression

        Returns:
            self for chaining
        """
        # Header -> body (if condition true)
        self.edge(header, body, EdgeKind.CONDITIONAL_TRUE, condition)
        # Header -> exit (if condition false)
        self.edge(header, exit, EdgeKind.LOOP_EXIT, f"not ({condition})")
        # Body -> header (back edge)
        self.edge(body, header, EdgeKind.LOOP_BACK)
        return self

    def build(self) -> CFGSketch:
        """Build the CFG sketch.

        Returns:
            The constructed CFGSketch
        """
        cfg = CFGSketch(
            blocks=self._blocks.copy(),
            edges=self._edges.copy(),
            entry_id=self._entry_id,
            exit_ids=self._exit_ids.copy(),
        )

        # Update predecessor/successor sets
        for edge in cfg.edges:
            if edge.source in cfg.blocks:
                cfg.blocks[edge.source].successors.add(edge.target)
            if edge.target in cfg.blocks:
                cfg.blocks[edge.target].predecessors.add(edge.source)

        return cfg
