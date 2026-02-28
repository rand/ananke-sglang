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
"""Totalization: assigning types to ALL partial programs (POPL 2024).

The key insight from the marked lambda calculus is that EVERY partial
program should have a well-defined type. This is achieved through
totalization - a process that:

1. Assigns types to complete expressions via standard type checking
2. Marks holes with their expected types from context
3. Marks type mismatches as InconsistentMarks without failing
4. Propagates types through the AST

After totalization, the program may have marks (holes, errors) but
always has a defined type at every node.

References:
    - POPL 2024: "Total Type Error Localization and Recovery with Holes"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# Support both relative imports (when used as subpackage) and absolute imports (standalone testing)
try:
    from ..constraint import Type, ANY, NEVER
    from ..environment import TypeEnvironment
    from .marks import Mark, HoleMark, InconsistentMark, create_inconsistent_mark
    from .marked_ast import (
        MarkedAST,
        MarkedASTNode,
        ASTNodeKind,
        create_hole_node,
    )
    from .provenance import Provenance, SourceSpan, UNKNOWN_SPAN, CONTEXT_EXPRESSION
except ImportError:
    from domains.types.constraint import Type, ANY, NEVER
    from domains.types.environment import TypeEnvironment
    from domains.types.marking.marks import (
        Mark,
        HoleMark,
        InconsistentMark,
        create_inconsistent_mark,
    )
    from domains.types.marking.marked_ast import (
        MarkedAST,
        MarkedASTNode,
        ASTNodeKind,
        create_hole_node,
    )
    from domains.types.marking.provenance import (
        Provenance,
        SourceSpan,
        UNKNOWN_SPAN,
        CONTEXT_EXPRESSION,
    )


@dataclass
class TotalizationResult:
    """Result of totalization.

    Contains the marked AST along with statistics about
    holes and errors encountered.

    Attributes:
        ast: The totalized marked AST
        hole_count: Number of holes in the result
        error_count: Number of type errors
    """

    ast: MarkedAST
    hole_count: int = 0
    error_count: int = 0

    @property
    def has_errors(self) -> bool:
        """True if there are any type errors."""
        return self.error_count > 0

    @property
    def has_holes(self) -> bool:
        """True if there are any unfilled holes."""
        return self.hole_count > 0

    @property
    def is_complete(self) -> bool:
        """True if no holes and no errors."""
        return not self.has_holes and not self.has_errors


def totalize(
    node: MarkedASTNode,
    expected: Optional[Type],
    env: TypeEnvironment,
    context: str = CONTEXT_EXPRESSION,
) -> TotalizationResult:
    """Totalize an AST node, ensuring it has a well-defined type.

    This is the core totalization function that:
    1. If the node is a hole, marks it with the expected type
    2. If the node has a type, checks it against expected
    3. Recursively totalizes children
    4. Never fails - always produces a typed result

    Args:
        node: The AST node to totalize
        expected: The expected type from context (may be None)
        env: The current type environment
        context: Description of the current context

    Returns:
        TotalizationResult with the totalized AST
    """
    # Handle hole nodes specially
    if node.kind == ASTNodeKind.HOLE or isinstance(node.mark, HoleMark):
        return _totalize_hole(node, expected, env, context)

    # Get the synthesized type
    synthesized = node.synthesized_type

    # If no synthesized type, use Any (shouldn't happen in well-formed ASTs)
    if synthesized is None:
        synthesized = ANY

    # Check against expected type if provided
    if expected is not None and not _types_compatible(synthesized, expected):
        # Type mismatch - create an inconsistent mark
        provenance = Provenance(
            location=node.span,
            context=context,
        )
        mark = create_inconsistent_mark(synthesized, expected, provenance)
        marked_node = node.with_mark(mark)
        ast = MarkedAST(root=marked_node)
        return TotalizationResult(ast=ast, error_count=1)

    # Totalize children recursively
    total_holes = 0
    total_errors = 0
    new_children = []

    for i, child in enumerate(node.children):
        # Determine expected type for child based on node kind
        child_expected = _infer_child_expected_type(node, i, env)
        child_context = _infer_child_context(node, i)

        child_result = totalize(child, child_expected, env, child_context)
        new_children.append(child_result.ast.root)
        total_holes += child_result.hole_count
        total_errors += child_result.error_count

    # Create new node with totalized children
    new_node = node.with_children(new_children)
    ast = MarkedAST(root=new_node)

    return TotalizationResult(
        ast=ast,
        hole_count=total_holes,
        error_count=total_errors,
    )


def _totalize_hole(
    node: MarkedASTNode,
    expected: Optional[Type],
    env: TypeEnvironment,
    context: str,
) -> TotalizationResult:
    """Totalize a hole node.

    Holes are marked with their expected type from context.
    """
    # Get existing hole mark if present
    existing_mark = node.mark
    if isinstance(existing_mark, HoleMark):
        hole_id = existing_mark.hole_id
        # Use provided expected type, or fall back to mark's expected
        final_expected = expected if expected is not None else existing_mark.expected_type
    else:
        hole_id = node.node_id or f"hole_{id(node)}"
        final_expected = expected

    # Create hole mark with expected type
    provenance = Provenance(location=node.span, context=context)
    mark = HoleMark(hole_id, final_expected, provenance)

    # Update node with mark and synthesized type
    new_node = node.with_mark(mark).with_type(final_expected if final_expected else ANY)
    ast = MarkedAST(root=new_node)

    return TotalizationResult(ast=ast, hole_count=1, error_count=0)


def _types_compatible(synthesized: Type, expected: Type) -> bool:
    """Check if synthesized type is compatible with expected.

    This is a simplified compatibility check. A full implementation
    would use the subtyping relation from the type system.

    Args:
        synthesized: The inferred type
        expected: The expected type

    Returns:
        True if compatible
    """
    # Any is compatible with everything
    if isinstance(synthesized, type(ANY)) or isinstance(expected, type(ANY)):
        return True

    # Never is a subtype of everything
    if isinstance(synthesized, type(NEVER)):
        return True

    # Simple equality check for now
    # A real implementation would have proper subtyping
    return synthesized == expected


def _infer_child_expected_type(
    parent: MarkedASTNode,
    child_index: int,
    env: TypeEnvironment,
) -> Optional[Type]:
    """Infer the expected type for a child node.

    Based on the parent node kind and child position, determine
    what type the child should have.

    Args:
        parent: The parent node
        child_index: Index of the child
        env: Type environment

    Returns:
        Expected type for the child, or None if unknown
    """
    # This would be fleshed out based on node kinds
    # For now, return None (no constraint)
    return None


def _infer_child_context(parent: MarkedASTNode, child_index: int) -> str:
    """Infer the context string for a child node.

    Args:
        parent: The parent node
        child_index: Index of the child

    Returns:
        Context description string
    """
    if parent.kind == ASTNodeKind.APPLICATION:
        if child_index == 0:
            return "function"
        return f"argument {child_index}"

    if parent.kind == ASTNodeKind.IF:
        if child_index == 0:
            return "condition"
        elif child_index == 1:
            return "then branch"
        return "else branch"

    if parent.kind == ASTNodeKind.LET:
        if child_index == 0:
            return "binding value"
        return "body"

    if parent.kind == ASTNodeKind.LAMBDA:
        return "lambda body"

    if parent.kind == ASTNodeKind.LIST:
        return f"list element {child_index}"

    if parent.kind == ASTNodeKind.TUPLE:
        return f"tuple element {child_index}"

    return CONTEXT_EXPRESSION


def totalize_program(
    program: MarkedAST,
    env: Optional[TypeEnvironment] = None,
) -> TotalizationResult:
    """Totalize an entire program.

    Convenience function that totalizes from the root with no
    expected type (the program's type is synthesized).

    Args:
        program: The program to totalize
        env: Optional type environment

    Returns:
        TotalizationResult
    """
    if env is None:
        env = TypeEnvironment()

    return totalize(program.root, None, env, "program")
