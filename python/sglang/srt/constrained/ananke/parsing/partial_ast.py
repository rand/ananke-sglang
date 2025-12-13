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
"""Partial AST representation for incremental parsing.

A PartialAST represents a program that may be incomplete, containing
holes where code is missing. This abstraction enables type checking
of partial programs during generation.

Key concepts:
- Holes: Positions where code is expected but not yet generated
- Completeness: Whether all holes are filled
- Consistency: Whether the partial AST is well-formed

References:
    - Hazel: "Live Functional Programming with Typed Holes" (ICFP 2019)
    - POPL 2024: "Total Type Error Localization and Recovery with Holes"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple, Iterator, Callable

from domains.types.marking.provenance import SourceSpan, UNKNOWN_SPAN
from domains.types.marking.marked_ast import (
    MarkedASTNode,
    ASTNodeKind,
    MarkedAST,
    create_hole_node,
    create_literal_node,
    create_variable_node,
)
from domains.types.constraint import Type, ANY


class HoleKind(Enum):
    """Classification of holes in partial programs.

    Different kinds of holes have different implications for
    type checking and generation.
    """

    EXPRESSION = auto()    # Missing expression (e.g., "x + _")
    STATEMENT = auto()     # Missing statement (e.g., function body)
    TYPE = auto()          # Missing type annotation
    IDENTIFIER = auto()    # Missing identifier
    ARGUMENT = auto()      # Missing function argument
    BODY = auto()          # Missing block/body
    PATTERN = auto()       # Missing pattern (in match/case)


@dataclass
class HoleInfo:
    """Information about a hole in a partial AST.

    Attributes:
        hole_id: Unique identifier for this hole
        kind: Classification of the hole
        span: Source location
        expected_type: Type expected at this hole (if known)
        context: Description of the syntactic context
        parent_id: ID of the parent hole (for nested holes)
    """

    hole_id: str
    kind: HoleKind
    span: SourceSpan
    expected_type: Optional[Type] = None
    context: Optional[str] = None
    parent_id: Optional[str] = None

    def with_expected_type(self, typ: Type) -> 'HoleInfo':
        """Return a copy with updated expected type."""
        return HoleInfo(
            hole_id=self.hole_id,
            kind=self.kind,
            span=self.span,
            expected_type=typ,
            context=self.context,
            parent_id=self.parent_id,
        )


@dataclass
class PartialAST:
    """A partial AST that may contain holes.

    The PartialAST is the primary data structure for representing
    programs during generation. It tracks:
    - The root of the AST
    - All holes and their types
    - Source text for error messages
    - Completion status

    Example:
        >>> partial = PartialAST.from_source("def foo(x: int) -> ")
        >>> partial.is_complete
        False
        >>> partial.holes
        [HoleInfo(hole_id='h0', kind=HoleKind.TYPE, ...)]
    """

    root: MarkedASTNode
    holes: Dict[str, HoleInfo] = field(default_factory=dict)
    source: Optional[str] = None
    _filled_holes: Set[str] = field(default_factory=set)

    @property
    def is_complete(self) -> bool:
        """Whether all holes are filled."""
        return len(self.unfilled_holes) == 0

    @property
    def unfilled_holes(self) -> List[HoleInfo]:
        """Get list of unfilled holes."""
        return [
            h for h in self.holes.values()
            if h.hole_id not in self._filled_holes
        ]

    @property
    def hole_count(self) -> int:
        """Number of unfilled holes."""
        return len(self.unfilled_holes)

    def get_hole(self, hole_id: str) -> Optional[HoleInfo]:
        """Get information about a specific hole.

        Args:
            hole_id: The hole identifier

        Returns:
            HoleInfo if found, None otherwise
        """
        return self.holes.get(hole_id)

    def first_unfilled_hole(self) -> Optional[HoleInfo]:
        """Get the first unfilled hole (by source position).

        Returns:
            HoleInfo for the first unfilled hole, or None
        """
        unfilled = self.unfilled_holes
        if not unfilled:
            return None
        return min(unfilled, key=lambda h: h.span.start)

    def add_hole(self, info: HoleInfo) -> None:
        """Add a hole to the partial AST.

        Args:
            info: Information about the hole
        """
        self.holes[info.hole_id] = info

    def fill_hole(self, hole_id: str, node: MarkedASTNode) -> 'PartialAST':
        """Fill a hole with a node, returning a new PartialAST.

        This is a functional operation - the original is not modified.

        Args:
            hole_id: ID of the hole to fill
            node: The node to fill the hole with

        Returns:
            New PartialAST with the hole filled

        Raises:
            ValueError: If hole_id is not found
        """
        if hole_id not in self.holes:
            raise ValueError(f"Unknown hole: {hole_id}")

        # Create new AST with hole replaced
        new_root = self._replace_hole(self.root, hole_id, node)

        # Create new PartialAST
        new_ast = PartialAST(
            root=new_root,
            holes=self.holes.copy(),
            source=self.source,
            _filled_holes=self._filled_holes | {hole_id},
        )

        # Add any new holes from the filled node
        for h in self._find_holes_in_node(node):
            new_ast.holes[h.hole_id] = h

        return new_ast

    def _replace_hole(
        self,
        node: MarkedASTNode,
        hole_id: str,
        replacement: MarkedASTNode
    ) -> MarkedASTNode:
        """Replace a hole with a new node."""
        if node.node_id == hole_id and node.kind == ASTNodeKind.HOLE:
            return replacement

        # Recursively replace in children
        new_children = [
            self._replace_hole(child, hole_id, replacement)
            for child in node.children
        ]

        # Return updated node if children changed
        if new_children != node.children:
            return MarkedASTNode(
                kind=node.kind,
                span=node.span,
                synthesized_type=node.synthesized_type,
                mark=node.mark,
                children=new_children,
                data=node.data,
                node_id=node.node_id,
            )

        return node

    def _find_holes_in_node(self, node: MarkedASTNode) -> List[HoleInfo]:
        """Find all holes in a node tree."""
        holes: List[HoleInfo] = []

        if node.kind == ASTNodeKind.HOLE:
            holes.append(HoleInfo(
                hole_id=node.node_id or f"auto_{id(node)}",
                kind=HoleKind.EXPRESSION,
                span=node.span,
            ))

        for child in node.children:
            holes.extend(self._find_holes_in_node(child))

        return holes

    def to_marked_ast(self) -> MarkedAST:
        """Convert to a MarkedAST for type checking.

        Returns:
            MarkedAST wrapping the root node
        """
        return MarkedAST(root=self.root, source=self.source)

    def copy(self) -> 'PartialAST':
        """Create a deep copy of this partial AST."""
        return PartialAST(
            root=self._copy_node(self.root),
            holes=self.holes.copy(),
            source=self.source,
            _filled_holes=self._filled_holes.copy(),
        )

    def _copy_node(self, node: MarkedASTNode) -> MarkedASTNode:
        """Deep copy a node and its children."""
        return MarkedASTNode(
            kind=node.kind,
            span=node.span,
            synthesized_type=node.synthesized_type,
            mark=node.mark,
            children=[self._copy_node(c) for c in node.children],
            data=node.data.copy() if node.data else {},
            node_id=node.node_id,
        )

    @staticmethod
    def empty() -> 'PartialAST':
        """Create an empty partial AST with a single hole."""
        hole = create_hole_node(
            hole_id="root",
            span=UNKNOWN_SPAN,
            expected_type=ANY,
        )
        ast = PartialAST(root=hole)
        ast.add_hole(HoleInfo(
            hole_id="root",
            kind=HoleKind.EXPRESSION,
            span=UNKNOWN_SPAN,
        ))
        return ast

    @staticmethod
    def from_node(node: MarkedASTNode, source: Optional[str] = None) -> 'PartialAST':
        """Create a PartialAST from a MarkedASTNode.

        Args:
            node: The root node
            source: Optional source text

        Returns:
            PartialAST with holes extracted
        """
        ast = PartialAST(root=node, source=source)

        # Extract holes from the tree
        holes = ast._find_holes_in_node(node)
        for h in holes:
            ast.holes[h.hole_id] = h

        return ast


@dataclass
class ASTDiff:
    """Represents a difference between two ASTs.

    Used for incremental type checking - only the changed
    portions need to be rechecked.
    """

    changed_spans: List[SourceSpan]
    added_holes: List[str]
    removed_holes: List[str]
    modified_nodes: List[str]  # Node IDs

    @property
    def has_changes(self) -> bool:
        """Whether there are any changes."""
        return bool(
            self.changed_spans or
            self.added_holes or
            self.removed_holes or
            self.modified_nodes
        )

    @staticmethod
    def compute(old: PartialAST, new: PartialAST) -> 'ASTDiff':
        """Compute the diff between two partial ASTs.

        Args:
            old: The old AST
            new: The new AST

        Returns:
            ASTDiff describing the changes
        """
        old_holes = set(old.holes.keys())
        new_holes = set(new.holes.keys())

        return ASTDiff(
            changed_spans=[],  # Would need tree diff for full implementation
            added_holes=list(new_holes - old_holes),
            removed_holes=list(old_holes - new_holes),
            modified_nodes=[],
        )


class PartialASTBuilder:
    """Builder for constructing partial ASTs.

    Provides a fluent interface for building ASTs programmatically,
    useful for testing and manual AST construction.
    """

    def __init__(self) -> None:
        """Initialize the builder."""
        self._stack: List[MarkedASTNode] = []
        self._holes: Dict[str, HoleInfo] = {}
        self._hole_counter: int = 0

    def literal(self, value: Any, typ: Type, span: SourceSpan = UNKNOWN_SPAN) -> 'PartialASTBuilder':
        """Add a literal node.

        Args:
            value: The literal value
            typ: The type of the literal
            span: Source span

        Returns:
            Self for chaining
        """
        node = create_literal_node(value=value, ty=typ, span=span)
        self._stack.append(node)
        return self

    def variable(self, name: str, typ: Type, span: SourceSpan = UNKNOWN_SPAN) -> 'PartialASTBuilder':
        """Add a variable node.

        Args:
            name: Variable name
            typ: Variable type
            span: Source span

        Returns:
            Self for chaining
        """
        node = create_variable_node(name=name, ty=typ, span=span)
        self._stack.append(node)
        return self

    def hole(
        self,
        kind: HoleKind = HoleKind.EXPRESSION,
        expected_type: Optional[Type] = None,
        span: SourceSpan = UNKNOWN_SPAN,
        hole_id: Optional[str] = None,
    ) -> 'PartialASTBuilder':
        """Add a hole node.

        Args:
            kind: Kind of hole
            expected_type: Expected type at this hole
            span: Source span
            hole_id: Optional ID (auto-generated if None)

        Returns:
            Self for chaining
        """
        if hole_id is None:
            hole_id = f"hole_{self._hole_counter}"
            self._hole_counter += 1

        node = create_hole_node(
            hole_id=hole_id,
            span=span,
            expected_type=expected_type,
        )
        self._stack.append(node)

        self._holes[hole_id] = HoleInfo(
            hole_id=hole_id,
            kind=kind,
            span=span,
            expected_type=expected_type,
        )

        return self

    def list_node(self, num_children: int, span: SourceSpan = UNKNOWN_SPAN) -> 'PartialASTBuilder':
        """Create a list node from the top N stack items.

        Args:
            num_children: Number of children to pop
            span: Source span

        Returns:
            Self for chaining
        """
        if len(self._stack) < num_children:
            raise ValueError(f"Not enough items on stack: {len(self._stack)} < {num_children}")

        children = [self._stack.pop() for _ in range(num_children)]
        children.reverse()

        node = MarkedASTNode(
            kind=ASTNodeKind.LIST,
            span=span,
            children=children,
        )
        self._stack.append(node)
        return self

    def tuple_node(self, num_children: int, span: SourceSpan = UNKNOWN_SPAN) -> 'PartialASTBuilder':
        """Create a tuple node from the top N stack items.

        Args:
            num_children: Number of children to pop
            span: Source span

        Returns:
            Self for chaining
        """
        if len(self._stack) < num_children:
            raise ValueError(f"Not enough items on stack: {len(self._stack)} < {num_children}")

        children = [self._stack.pop() for _ in range(num_children)]
        children.reverse()

        node = MarkedASTNode(
            kind=ASTNodeKind.TUPLE,
            span=span,
            children=children,
        )
        self._stack.append(node)
        return self

    def build(self) -> PartialAST:
        """Build the final PartialAST.

        Returns:
            The constructed PartialAST

        Raises:
            ValueError: If stack doesn't have exactly one item
        """
        if len(self._stack) != 1:
            raise ValueError(f"Stack should have exactly 1 item, has {len(self._stack)}")

        return PartialAST(
            root=self._stack[0],
            holes=self._holes.copy(),
        )

    def reset(self) -> 'PartialASTBuilder':
        """Reset the builder to initial state.

        Returns:
            Self for chaining
        """
        self._stack = []
        self._holes = {}
        self._hole_counter = 0
        return self
