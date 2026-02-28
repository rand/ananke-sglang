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

# Try to use immutables for efficient persistent maps
try:
    from immutables import Map as ImmutableMap

    HAS_IMMUTABLES = True
except ImportError:
    HAS_IMMUTABLES = False

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


@dataclass(frozen=True, slots=True)
class ImmutableBasicBlock:
    """Immutable basic block for structural sharing.

    Unlike BasicBlock, this class is completely immutable and hashable,
    enabling O(1) checkpointing via reference sharing.

    Attributes:
        id: Unique identifier for this block
        kind: Description of what this block represents
        statements: Tuple of statement descriptions (immutable)
        predecessors: FrozenSet of predecessor block IDs
        successors: FrozenSet of successor block IDs
        is_entry: True if this is an entry block
        is_exit: True if this is an exit block
        is_loop_header: True if this is a loop header
        source_lines: Optional source line range
    """

    id: str
    kind: str = "block"
    statements: Tuple[str, ...] = ()
    predecessors: FrozenSet[str] = frozenset()
    successors: FrozenSet[str] = frozenset()
    is_entry: bool = False
    is_exit: bool = False
    is_loop_header: bool = False
    source_lines: Optional[Tuple[int, int]] = None

    def with_successor(self, successor_id: str) -> ImmutableBasicBlock:
        """Return a new block with an added successor."""
        return ImmutableBasicBlock(
            id=self.id,
            kind=self.kind,
            statements=self.statements,
            predecessors=self.predecessors,
            successors=self.successors | {successor_id},
            is_entry=self.is_entry,
            is_exit=self.is_exit,
            is_loop_header=self.is_loop_header,
            source_lines=self.source_lines,
        )

    def with_predecessor(self, predecessor_id: str) -> ImmutableBasicBlock:
        """Return a new block with an added predecessor."""
        return ImmutableBasicBlock(
            id=self.id,
            kind=self.kind,
            statements=self.statements,
            predecessors=self.predecessors | {predecessor_id},
            successors=self.successors,
            is_entry=self.is_entry,
            is_exit=self.is_exit,
            is_loop_header=self.is_loop_header,
            source_lines=self.source_lines,
        )

    def with_statement(self, stmt: str) -> ImmutableBasicBlock:
        """Return a new block with an added statement."""
        return ImmutableBasicBlock(
            id=self.id,
            kind=self.kind,
            statements=self.statements + (stmt,),
            predecessors=self.predecessors,
            successors=self.successors,
            is_entry=self.is_entry,
            is_exit=self.is_exit,
            is_loop_header=self.is_loop_header,
            source_lines=self.source_lines,
        )

    def to_code_point(self) -> CodePoint:
        """Convert to a CodePoint for constraint tracking."""
        line = self.source_lines[0] if self.source_lines else None
        return CodePoint(label=self.id, kind=self.kind, line=line)

    def to_mutable(self) -> BasicBlock:
        """Convert to a mutable BasicBlock."""
        return BasicBlock(
            id=self.id,
            kind=self.kind,
            statements=list(self.statements),
            predecessors=set(self.predecessors),
            successors=set(self.successors),
            is_entry=self.is_entry,
            is_exit=self.is_exit,
            is_loop_header=self.is_loop_header,
            source_lines=self.source_lines,
        )

    @staticmethod
    def from_mutable(block: BasicBlock) -> ImmutableBasicBlock:
        """Create from a mutable BasicBlock."""
        return ImmutableBasicBlock(
            id=block.id,
            kind=block.kind,
            statements=tuple(block.statements),
            predecessors=frozenset(block.predecessors),
            successors=frozenset(block.successors),
            is_entry=block.is_entry,
            is_exit=block.is_exit,
            is_loop_header=block.is_loop_header,
            source_lines=block.source_lines,
        )


class ImmutableCFGSketch:
    """Immutable CFG sketch with structural sharing.

    This class provides O(1) checkpointing by using:
    - immutables.Map for blocks (structural sharing on updates)
    - frozenset for edges
    - Reference equality for snapshots

    All modification methods return new instances, sharing
    unmodified portions with the original.

    Example:
        >>> cfg = ImmutableCFGSketch()
        >>> cfg2 = cfg.add_block(block)  # O(log n) copy, shares most data
        >>> checkpoint = cfg2  # O(1) - just reference copy

    Attributes:
        _blocks: Immutable map from block ID to ImmutableBasicBlock
        _edges: Frozen set of CFGEdge instances
        _entry_id: ID of the entry block (or None)
        _exit_ids: Frozen set of exit block IDs
    """

    __slots__ = ("_blocks", "_edges", "_entry_id", "_exit_ids")

    def __init__(
        self,
        blocks: Optional[Dict[str, ImmutableBasicBlock]] = None,
        edges: Optional[FrozenSet[CFGEdge]] = None,
        entry_id: Optional[str] = None,
        exit_ids: Optional[FrozenSet[str]] = None,
    ):
        """Initialize an immutable CFG sketch.

        Args:
            blocks: Initial blocks (will be converted to immutable map)
            edges: Initial edges (will be converted to frozenset)
            entry_id: ID of entry block
            exit_ids: IDs of exit blocks
        """
        if HAS_IMMUTABLES:
            self._blocks: ImmutableMap[str, ImmutableBasicBlock] = (
                ImmutableMap(blocks) if blocks else ImmutableMap()
            )
        else:
            # Fallback: use a regular dict (less efficient but functional)
            self._blocks = dict(blocks) if blocks else {}

        self._edges: FrozenSet[CFGEdge] = edges if edges is not None else frozenset()
        self._entry_id: Optional[str] = entry_id
        self._exit_ids: FrozenSet[str] = exit_ids if exit_ids is not None else frozenset()

    def add_block(self, block: ImmutableBasicBlock) -> ImmutableCFGSketch:
        """Return a new CFG with the block added.

        This is O(log n) due to structural sharing.

        Args:
            block: The block to add

        Returns:
            New ImmutableCFGSketch with the block added
        """
        if HAS_IMMUTABLES:
            new_blocks = self._blocks.set(block.id, block)
        else:
            new_blocks = dict(self._blocks)
            new_blocks[block.id] = block

        new_entry = block.id if block.is_entry else self._entry_id
        new_exits = self._exit_ids | {block.id} if block.is_exit else self._exit_ids

        result = ImmutableCFGSketch.__new__(ImmutableCFGSketch)
        result._blocks = new_blocks
        result._edges = self._edges
        result._entry_id = new_entry
        result._exit_ids = new_exits
        return result

    def add_edge(self, edge: CFGEdge) -> ImmutableCFGSketch:
        """Return a new CFG with the edge added.

        Also updates predecessor/successor sets in affected blocks.

        Args:
            edge: The edge to add

        Returns:
            New ImmutableCFGSketch with the edge added
        """
        new_edges = self._edges | {edge}

        # Update source block's successors
        new_blocks = self._blocks
        if edge.source in new_blocks:
            source_block = new_blocks[edge.source]
            new_source = source_block.with_successor(edge.target)
            if HAS_IMMUTABLES:
                new_blocks = new_blocks.set(edge.source, new_source)
            else:
                new_blocks = dict(new_blocks)
                new_blocks[edge.source] = new_source

        # Update target block's predecessors
        if edge.target in new_blocks:
            target_block = new_blocks[edge.target]
            new_target = target_block.with_predecessor(edge.source)
            if HAS_IMMUTABLES:
                new_blocks = new_blocks.set(edge.target, new_target)
            else:
                if not isinstance(new_blocks, dict):
                    new_blocks = dict(new_blocks)
                new_blocks[edge.target] = new_target

        result = ImmutableCFGSketch.__new__(ImmutableCFGSketch)
        result._blocks = new_blocks
        result._edges = new_edges
        result._entry_id = self._entry_id
        result._exit_ids = self._exit_ids
        return result

    @property
    def blocks(self) -> Dict[str, ImmutableBasicBlock]:
        """Get blocks as a dictionary (for compatibility)."""
        return dict(self._blocks)

    @property
    def edges(self) -> FrozenSet[CFGEdge]:
        """Get edges."""
        return self._edges

    @property
    def entry_id(self) -> Optional[str]:
        """Get entry block ID."""
        return self._entry_id

    @property
    def exit_ids(self) -> FrozenSet[str]:
        """Get exit block IDs."""
        return self._exit_ids

    def get_block(self, block_id: str) -> Optional[ImmutableBasicBlock]:
        """Get a block by ID."""
        return self._blocks.get(block_id)

    def get_entry(self) -> Optional[ImmutableBasicBlock]:
        """Get the entry block."""
        if self._entry_id:
            return self._blocks.get(self._entry_id)
        return None

    def get_exits(self) -> List[ImmutableBasicBlock]:
        """Get all exit blocks."""
        return [self._blocks[eid] for eid in self._exit_ids if eid in self._blocks]

    def get_successors(self, block_id: str) -> List[ImmutableBasicBlock]:
        """Get successor blocks."""
        block = self._blocks.get(block_id)
        if not block:
            return []
        return [self._blocks[sid] for sid in block.successors if sid in self._blocks]

    def get_predecessors(self, block_id: str) -> List[ImmutableBasicBlock]:
        """Get predecessor blocks."""
        block = self._blocks.get(block_id)
        if not block:
            return []
        return [self._blocks[pid] for pid in block.predecessors if pid in self._blocks]

    def get_outgoing_edges(self, block_id: str) -> List[CFGEdge]:
        """Get outgoing edges from a block."""
        return [e for e in self._edges if e.source == block_id]

    def get_incoming_edges(self, block_id: str) -> List[CFGEdge]:
        """Get incoming edges to a block."""
        return [e for e in self._edges if e.target == block_id]

    def get_loop_headers(self) -> List[ImmutableBasicBlock]:
        """Get all loop header blocks."""
        return [b for b in self._blocks.values() if b.is_loop_header]

    def get_back_edges(self) -> List[CFGEdge]:
        """Get all back edges."""
        return [e for e in self._edges if e.kind == EdgeKind.LOOP_BACK]

    def block_count(self) -> int:
        """Get the number of blocks."""
        return len(self._blocks)

    def edge_count(self) -> int:
        """Get the number of edges."""
        return len(self._edges)

    def iter_blocks(self) -> Iterator[ImmutableBasicBlock]:
        """Iterate over all blocks."""
        return iter(self._blocks.values())

    def iter_edges(self) -> Iterator[CFGEdge]:
        """Iterate over all edges."""
        return iter(self._edges)

    def to_code_points(self) -> FrozenSet[CodePoint]:
        """Convert all blocks to CodePoints."""
        return frozenset(b.to_code_point() for b in self._blocks.values())

    def to_mutable(self) -> CFGSketch:
        """Convert to a mutable CFGSketch."""
        return CFGSketch(
            blocks={k: v.to_mutable() for k, v in self._blocks.items()},
            edges=set(self._edges),
            entry_id=self._entry_id,
            exit_ids=set(self._exit_ids),
        )

    @staticmethod
    def from_mutable(cfg: CFGSketch) -> ImmutableCFGSketch:
        """Create from a mutable CFGSketch."""
        return ImmutableCFGSketch(
            blocks={k: ImmutableBasicBlock.from_mutable(v) for k, v in cfg.blocks.items()},
            edges=frozenset(cfg.edges),
            entry_id=cfg.entry_id,
            exit_ids=frozenset(cfg.exit_ids),
        )

    def __repr__(self) -> str:
        return f"ImmutableCFGSketch(blocks={self.block_count()}, edges={self.edge_count()})"


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
