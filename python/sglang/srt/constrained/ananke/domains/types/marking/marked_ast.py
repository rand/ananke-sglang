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
"""Marked AST representation (POPL 2024).

A MarkedAST is an AST where each node carries:
- An optional Mark indicating holes or type errors
- A synthesized type (the type inferred for this node)
- References to child nodes

This allows type information to be attached to every node,
enabling total type error localization where every partial
program has a well-defined type.

References:
    - POPL 2024: "Total Type Error Localization and Recovery with Holes"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Iterator, List, Optional, Tuple

# Support both relative imports (when used as subpackage) and absolute imports (standalone testing)
try:
    from ..constraint import Type, ANY
    from .marks import Mark, HoleMark, InconsistentMark, NonEmptyHoleMark
    from .provenance import SourceSpan, Provenance
except ImportError:
    from domains.types.constraint import Type, ANY
    from domains.types.marking.marks import Mark, HoleMark, InconsistentMark, NonEmptyHoleMark
    from domains.types.marking.provenance import SourceSpan, Provenance


class ASTNodeKind(Enum):
    """Kinds of AST nodes."""

    # Expressions
    LITERAL = auto()
    VARIABLE = auto()
    LAMBDA = auto()
    APPLICATION = auto()
    CALL = auto()  # Function call
    LET = auto()
    IF = auto()
    BINARY_OP = auto()
    UNARY_OP = auto()
    LIST = auto()
    DICT = auto()
    TUPLE = auto()
    ATTRIBUTE = auto()
    SUBSCRIPT = auto()
    EXPRESSION = auto()  # Generic expression

    # Statements
    ASSIGNMENT = auto()
    RETURN = auto()
    FUNCTION_DEF = auto()
    CLASS_DEF = auto()
    IMPORT = auto()
    EXPR_STMT = auto()

    # Special
    HOLE = auto()
    MODULE = auto()
    BLOCK = auto()
    ERROR = auto()  # Error recovery node


@dataclass
class MarkedASTNode:
    """A single node in a marked AST.

    Each node carries type information and optional marks.

    Attributes:
        kind: What kind of AST node this is
        span: Source location of this node
        synthesized_type: The type inferred for this node
        mark: Optional mark (hole or error)
        children: Child nodes
        data: Additional node-specific data (e.g., literal value, variable name)
        node_id: Unique identifier for this node
    """

    kind: ASTNodeKind
    span: SourceSpan
    synthesized_type: Optional[Type] = None
    mark: Optional[Mark] = None
    children: List[MarkedASTNode] = field(default_factory=list)
    data: Dict[str, object] = field(default_factory=dict)
    node_id: Optional[str] = None

    def is_hole(self) -> bool:
        """Check if this node is a hole."""
        return self.kind == ASTNodeKind.HOLE or (
            self.mark is not None and self.mark.is_hole()
        )

    def is_error(self) -> bool:
        """Check if this node has a type error."""
        return self.mark is not None and self.mark.is_error()

    def get_type(self) -> Type:
        """Get the type of this node.

        Returns:
            The synthesized type, or Any if none
        """
        if self.synthesized_type is not None:
            return self.synthesized_type
        if self.mark is not None:
            mark_type = self.mark.synthesized_type()
            if mark_type is not None:
                return mark_type
        return ANY

    def with_type(self, ty: Type) -> MarkedASTNode:
        """Create a copy with updated type."""
        return MarkedASTNode(
            kind=self.kind,
            span=self.span,
            synthesized_type=ty,
            mark=self.mark,
            children=self.children,
            data=self.data,
            node_id=self.node_id,
        )

    def with_mark(self, mark: Optional[Mark]) -> MarkedASTNode:
        """Create a copy with updated mark."""
        return MarkedASTNode(
            kind=self.kind,
            span=self.span,
            synthesized_type=self.synthesized_type,
            mark=mark,
            children=self.children,
            data=self.data,
            node_id=self.node_id,
        )

    def with_children(self, children: List[MarkedASTNode]) -> MarkedASTNode:
        """Create a copy with updated children."""
        return MarkedASTNode(
            kind=self.kind,
            span=self.span,
            synthesized_type=self.synthesized_type,
            mark=self.mark,
            children=children,
            data=self.data,
            node_id=self.node_id,
        )

    def __repr__(self) -> str:
        parts = [self.kind.name]
        if self.synthesized_type:
            parts.append(f"type={self.synthesized_type}")
        if self.mark:
            parts.append(f"mark={type(self.mark).__name__}")
        if self.children:
            parts.append(f"children={len(self.children)}")
        return f"MarkedASTNode({', '.join(parts)})"


@dataclass
class MarkedAST:
    """A complete marked AST with type annotations.

    The MarkedAST represents a (possibly partial) program where
    every node has been assigned a type, even in the presence of
    holes and errors.

    Attributes:
        root: The root node of the AST
        source: The original source code (for error messages)
        holes: Mapping from hole IDs to their nodes
        errors: List of all error nodes
    """

    root: MarkedASTNode
    source: Optional[str] = None
    _holes: Dict[str, MarkedASTNode] = field(default_factory=dict)
    _errors: List[MarkedASTNode] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Index holes and errors after construction."""
        self._index_nodes(self.root)

    def _index_nodes(self, node: MarkedASTNode) -> None:
        """Recursively index holes and errors."""
        if node.is_hole() and isinstance(node.mark, (HoleMark, NonEmptyHoleMark)):
            self._holes[node.mark.hole_id] = node
        if node.is_error():
            self._errors.append(node)
        for child in node.children:
            self._index_nodes(child)

    def find_hole(self, hole_id: str) -> Optional[MarkedASTNode]:
        """Find a hole by its ID.

        Args:
            hole_id: The hole identifier

        Returns:
            The hole node, or None if not found
        """
        return self._holes.get(hole_id)

    def find_first_unfilled_hole(self) -> Optional[MarkedASTNode]:
        """Find the first unfilled hole (depth-first).

        Returns:
            The first hole node, or None if no holes
        """
        return self._find_first_hole(self.root)

    def _find_first_hole(self, node: MarkedASTNode) -> Optional[MarkedASTNode]:
        """Recursively find first hole."""
        if node.is_hole() and isinstance(node.mark, HoleMark):
            return node
        for child in node.children:
            result = self._find_first_hole(child)
            if result is not None:
                return result
        return None

    def all_holes(self) -> List[MarkedASTNode]:
        """Get all hole nodes.

        Returns:
            List of all holes in document order
        """
        return list(self._holes.values())

    def all_errors(self) -> List[MarkedASTNode]:
        """Get all error nodes.

        Returns:
            List of all nodes with errors
        """
        return list(self._errors)

    def collect_errors(self) -> List[Tuple[InconsistentMark, MarkedASTNode]]:
        """Collect all error marks with their nodes.

        Returns:
            List of (mark, node) pairs for all errors
        """
        result: List[Tuple[InconsistentMark, MarkedASTNode]] = []
        for node in self._errors:
            if isinstance(node.mark, InconsistentMark):
                result.append((node.mark, node))
        return result

    def has_errors(self) -> bool:
        """Check if the AST has any type errors.

        Returns:
            True if there are any error marks
        """
        return len(self._errors) > 0

    def has_holes(self) -> bool:
        """Check if the AST has any unfilled holes.

        Returns:
            True if there are any hole marks
        """
        return len(self._holes) > 0

    def is_complete(self) -> bool:
        """Check if the AST is fully typed without holes or errors.

        Returns:
            True if no holes and no errors
        """
        return not self.has_holes() and not self.has_errors()

    def hole_count(self) -> int:
        """Get the number of holes."""
        return len(self._holes)

    def error_count(self) -> int:
        """Get the number of errors."""
        return len(self._errors)

    def traverse(self) -> Iterator[MarkedASTNode]:
        """Traverse all nodes in the AST.

        Yields:
            Each node in depth-first order
        """
        yield from self._traverse(self.root)

    def _traverse(self, node: MarkedASTNode) -> Iterator[MarkedASTNode]:
        """Recursive traversal helper."""
        yield node
        for child in node.children:
            yield from self._traverse(child)

    def get_type(self) -> Type:
        """Get the type of the entire program.

        Returns:
            The type of the root node
        """
        return self.root.get_type()

    def __repr__(self) -> str:
        return (
            f"MarkedAST(holes={self.hole_count()}, "
            f"errors={self.error_count()}, "
            f"type={self.get_type()})"
        )


def create_hole_node(
    hole_id: str,
    span: SourceSpan,
    expected_type: Optional[Type] = None,
    provenance: Optional[Provenance] = None,
) -> MarkedASTNode:
    """Create a hole AST node.

    Args:
        hole_id: Unique identifier for the hole
        span: Source location
        expected_type: Expected type from context
        provenance: Where the hole was introduced

    Returns:
        A new MarkedASTNode representing a hole
    """
    return MarkedASTNode(
        kind=ASTNodeKind.HOLE,
        span=span,
        synthesized_type=expected_type,
        mark=HoleMark(hole_id, expected_type, provenance),
        node_id=hole_id,
    )


def create_literal_node(
    value: object,
    ty: Type,
    span: SourceSpan,
) -> MarkedASTNode:
    """Create a literal AST node.

    Args:
        value: The literal value
        ty: The type of the literal
        span: Source location

    Returns:
        A new MarkedASTNode for the literal
    """
    return MarkedASTNode(
        kind=ASTNodeKind.LITERAL,
        span=span,
        synthesized_type=ty,
        data={"value": value},
    )


def create_variable_node(
    name: str,
    ty: Type,
    span: SourceSpan,
) -> MarkedASTNode:
    """Create a variable reference AST node.

    Args:
        name: The variable name
        ty: The type of the variable
        span: Source location

    Returns:
        A new MarkedASTNode for the variable
    """
    return MarkedASTNode(
        kind=ASTNodeKind.VARIABLE,
        span=span,
        synthesized_type=ty,
        data={"name": name},
    )
