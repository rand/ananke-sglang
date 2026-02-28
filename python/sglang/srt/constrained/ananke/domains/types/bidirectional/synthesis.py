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
"""Type synthesis (↑) for bidirectional typing.

Synthesis infers types from expressions without context. The type
flows UP from leaves to the root.

Synthesizable forms:
- Literals (int, str, bool, etc.) - type is obvious
- Variables - look up in environment
- Function application - infer from function type
- Annotated expressions - use the annotation

Non-synthesizable forms (require analysis):
- Lambda expressions - need parameter types from context
- Holes - need expected type from context

References:
    - Pierce & Turner (2000). "Local Type Inference"
    - Dunfield & Krishnaswami (2019). "Bidirectional Typing"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# Support both relative imports (when used as subpackage) and absolute imports (standalone testing)
try:
    from ..constraint import (
        ANY,
        BOOL,
        FLOAT,
        INT,
        NONE,
        STR,
        AnyType,
        DictType,
        FunctionType,
        HoleType,
        ListType,
        NeverType,
        TupleType,
        Type,
    )
    from ..environment import TypeEnvironment
    from ..marking.marks import HoleMark, InconsistentMark, create_inconsistent_mark
    from ..marking.marked_ast import (
        ASTNodeKind,
        MarkedAST,
        MarkedASTNode,
    )
    from ..marking.provenance import Provenance, CONTEXT_EXPRESSION
except ImportError:
    from domains.types.constraint import (
        ANY,
        BOOL,
        FLOAT,
        INT,
        NONE,
        STR,
        AnyType,
        DictType,
        FunctionType,
        HoleType,
        ListType,
        NeverType,
        TupleType,
        Type,
    )
    from domains.types.environment import TypeEnvironment
    from domains.types.marking.marks import (
        HoleMark,
        InconsistentMark,
        create_inconsistent_mark,
    )
    from domains.types.marking.marked_ast import (
        ASTNodeKind,
        MarkedAST,
        MarkedASTNode,
    )
    from domains.types.marking.provenance import Provenance, CONTEXT_EXPRESSION


@dataclass
class SynthesisResult:
    """Result of type synthesis.

    Attributes:
        synthesized_type: The inferred type (never None after synthesis)
        node: The marked AST node with type annotation
        errors: List of any type errors encountered
    """

    synthesized_type: Type
    node: MarkedASTNode
    errors: list = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []

    @property
    def has_errors(self) -> bool:
        """True if synthesis encountered errors."""
        return len(self.errors) > 0


def synthesize(
    node: MarkedASTNode,
    env: TypeEnvironment,
) -> SynthesisResult:
    """Synthesize a type for an AST node.

    Type flows UP from the node. The node must be in synthesizable form.

    Args:
        node: The AST node to synthesize
        env: The type environment

    Returns:
        SynthesisResult with the synthesized type
    """
    # Dispatch based on node kind
    if node.kind == ASTNodeKind.LITERAL:
        return _synthesize_literal(node, env)

    if node.kind == ASTNodeKind.VARIABLE:
        return _synthesize_variable(node, env)

    if node.kind == ASTNodeKind.APPLICATION:
        return _synthesize_application(node, env)

    if node.kind == ASTNodeKind.LIST:
        return _synthesize_list(node, env)

    if node.kind == ASTNodeKind.TUPLE:
        return _synthesize_tuple(node, env)

    if node.kind == ASTNodeKind.DICT:
        return _synthesize_dict(node, env)

    if node.kind == ASTNodeKind.IF:
        return _synthesize_if(node, env)

    if node.kind == ASTNodeKind.ATTRIBUTE:
        return _synthesize_attribute(node, env)

    if node.kind == ASTNodeKind.BINARY_OP:
        return _synthesize_binary_op(node, env)

    if node.kind == ASTNodeKind.UNARY_OP:
        return _synthesize_unary_op(node, env)

    if node.kind == ASTNodeKind.HOLE:
        return _synthesize_hole(node, env)

    # If node already has a type, use it
    if node.synthesized_type is not None:
        return SynthesisResult(
            synthesized_type=node.synthesized_type,
            node=node,
        )

    # Cannot synthesize - return Any with the node
    return SynthesisResult(
        synthesized_type=ANY,
        node=node.with_type(ANY),
    )


def _synthesize_literal(
    node: MarkedASTNode,
    env: TypeEnvironment,
) -> SynthesisResult:
    """Synthesize type for a literal."""
    value = node.data.get("value")

    if isinstance(value, bool):
        ty = BOOL
    elif isinstance(value, int):
        ty = INT
    elif isinstance(value, float):
        ty = FLOAT
    elif isinstance(value, str):
        ty = STR
    elif value is None:
        ty = NONE
    else:
        ty = ANY

    return SynthesisResult(
        synthesized_type=ty,
        node=node.with_type(ty),
    )


def _synthesize_variable(
    node: MarkedASTNode,
    env: TypeEnvironment,
) -> SynthesisResult:
    """Synthesize type for a variable reference."""
    name = node.data.get("name")

    if name is None:
        return SynthesisResult(
            synthesized_type=ANY,
            node=node.with_type(ANY),
        )

    ty = env.lookup(name)
    if ty is None:
        # Unbound variable - could be an error or use Any
        return SynthesisResult(
            synthesized_type=ANY,
            node=node.with_type(ANY),
            errors=[f"Unbound variable: {name}"],
        )

    return SynthesisResult(
        synthesized_type=ty,
        node=node.with_type(ty),
    )


def _synthesize_application(
    node: MarkedASTNode,
    env: TypeEnvironment,
) -> SynthesisResult:
    """Synthesize type for function application."""
    if len(node.children) < 1:
        return SynthesisResult(
            synthesized_type=ANY,
            node=node.with_type(ANY),
        )

    # Synthesize the function type
    func_node = node.children[0]
    func_result = synthesize(func_node, env)

    func_type = func_result.synthesized_type
    errors = list(func_result.errors)

    # If not a function type, return Any
    if not isinstance(func_type, FunctionType):
        return SynthesisResult(
            synthesized_type=ANY,
            node=node.with_type(ANY),
            errors=errors + [f"Cannot call non-function type: {func_type}"],
        )

    # Synthesize argument types
    arg_nodes = node.children[1:]
    new_children = [func_result.node]

    for i, arg_node in enumerate(arg_nodes):
        arg_result = synthesize(arg_node, env)
        new_children.append(arg_result.node)
        errors.extend(arg_result.errors)

    # The result type is the function's return type
    result_type = func_type.returns

    return SynthesisResult(
        synthesized_type=result_type,
        node=node.with_type(result_type).with_children(new_children),
        errors=errors,
    )


def _synthesize_list(
    node: MarkedASTNode,
    env: TypeEnvironment,
) -> SynthesisResult:
    """Synthesize type for a list expression."""
    if not node.children:
        # Empty list - type is List[Any]
        ty = ListType(ANY)
        return SynthesisResult(
            synthesized_type=ty,
            node=node.with_type(ty),
        )

    # Synthesize all element types
    new_children = []
    errors = []
    element_types = []

    for child in node.children:
        child_result = synthesize(child, env)
        new_children.append(child_result.node)
        errors.extend(child_result.errors)
        element_types.append(child_result.synthesized_type)

    # Find common type (simplified - just use first or Any)
    if element_types:
        # Check if all same
        first_type = element_types[0]
        if all(t == first_type for t in element_types):
            element_type = first_type
        else:
            element_type = ANY
    else:
        element_type = ANY

    ty = ListType(element_type)
    return SynthesisResult(
        synthesized_type=ty,
        node=node.with_type(ty).with_children(new_children),
        errors=errors,
    )


def _synthesize_tuple(
    node: MarkedASTNode,
    env: TypeEnvironment,
) -> SynthesisResult:
    """Synthesize type for a tuple expression."""
    new_children = []
    errors = []
    element_types = []

    for child in node.children:
        child_result = synthesize(child, env)
        new_children.append(child_result.node)
        errors.extend(child_result.errors)
        element_types.append(child_result.synthesized_type)

    ty = TupleType(tuple(element_types))
    return SynthesisResult(
        synthesized_type=ty,
        node=node.with_type(ty).with_children(new_children),
        errors=errors,
    )


def _synthesize_dict(
    node: MarkedASTNode,
    env: TypeEnvironment,
) -> SynthesisResult:
    """Synthesize type for a dictionary expression."""
    if not node.children:
        ty = DictType(ANY, ANY)
        return SynthesisResult(
            synthesized_type=ty,
            node=node.with_type(ty),
        )

    # Assuming children alternate key, value
    new_children = []
    errors = []
    key_types = []
    value_types = []

    for i, child in enumerate(node.children):
        child_result = synthesize(child, env)
        new_children.append(child_result.node)
        errors.extend(child_result.errors)

        if i % 2 == 0:
            key_types.append(child_result.synthesized_type)
        else:
            value_types.append(child_result.synthesized_type)

    # Determine common key and value types
    key_type = key_types[0] if key_types and all(t == key_types[0] for t in key_types) else ANY
    value_type = value_types[0] if value_types and all(t == value_types[0] for t in value_types) else ANY

    ty = DictType(key_type, value_type)
    return SynthesisResult(
        synthesized_type=ty,
        node=node.with_type(ty).with_children(new_children),
        errors=errors,
    )


def _synthesize_if(
    node: MarkedASTNode,
    env: TypeEnvironment,
) -> SynthesisResult:
    """Synthesize type for if expression."""
    if len(node.children) < 3:
        return SynthesisResult(
            synthesized_type=ANY,
            node=node.with_type(ANY),
        )

    cond_node, then_node, else_node = node.children[:3]

    # Synthesize all branches
    cond_result = synthesize(cond_node, env)
    then_result = synthesize(then_node, env)
    else_result = synthesize(else_node, env)

    errors = list(cond_result.errors) + list(then_result.errors) + list(else_result.errors)

    # The result type is the join of then and else types
    # Simplified: if same, use that; otherwise Any
    if then_result.synthesized_type == else_result.synthesized_type:
        result_type = then_result.synthesized_type
    else:
        result_type = ANY

    new_children = [cond_result.node, then_result.node, else_result.node]
    new_children.extend(node.children[3:])

    return SynthesisResult(
        synthesized_type=result_type,
        node=node.with_type(result_type).with_children(new_children),
        errors=errors,
    )


def _synthesize_attribute(
    node: MarkedASTNode,
    env: TypeEnvironment,
) -> SynthesisResult:
    """Synthesize type for attribute access."""
    # For now, return Any - full implementation would check class types
    return SynthesisResult(
        synthesized_type=ANY,
        node=node.with_type(ANY),
    )


def _synthesize_binary_op(
    node: MarkedASTNode,
    env: TypeEnvironment,
) -> SynthesisResult:
    """Synthesize type for binary operator."""
    if len(node.children) < 2:
        return SynthesisResult(
            synthesized_type=ANY,
            node=node.with_type(ANY),
        )

    left_result = synthesize(node.children[0], env)
    right_result = synthesize(node.children[1], env)

    errors = list(left_result.errors) + list(right_result.errors)

    # Get operator from data
    op = node.data.get("operator", "")

    # Infer result type based on operator
    if op in ("+", "-", "*", "/", "//", "%", "**"):
        # Arithmetic - result is numeric
        if left_result.synthesized_type == INT and right_result.synthesized_type == INT:
            if op == "/":
                result_type = FLOAT
            else:
                result_type = INT
        else:
            result_type = FLOAT
    elif op in ("==", "!=", "<", ">", "<=", ">=", "in", "not in", "is", "is not"):
        # Comparison - result is bool
        result_type = BOOL
    elif op in ("and", "or"):
        # Logical - result is bool
        result_type = BOOL
    else:
        result_type = ANY

    new_children = [left_result.node, right_result.node]
    return SynthesisResult(
        synthesized_type=result_type,
        node=node.with_type(result_type).with_children(new_children),
        errors=errors,
    )


def _synthesize_unary_op(
    node: MarkedASTNode,
    env: TypeEnvironment,
) -> SynthesisResult:
    """Synthesize type for unary operator."""
    if not node.children:
        return SynthesisResult(
            synthesized_type=ANY,
            node=node.with_type(ANY),
        )

    operand_result = synthesize(node.children[0], env)
    op = node.data.get("operator", "")

    if op == "not":
        result_type = BOOL
    elif op == "-":
        result_type = operand_result.synthesized_type
    else:
        result_type = ANY

    return SynthesisResult(
        synthesized_type=result_type,
        node=node.with_type(result_type).with_children([operand_result.node]),
        errors=operand_result.errors,
    )


def _synthesize_hole(
    node: MarkedASTNode,
    env: TypeEnvironment,
) -> SynthesisResult:
    """Synthesize type for a hole.

    Holes cannot truly be synthesized - they need context.
    We return the expected type from the mark if available,
    otherwise Any.
    """
    if isinstance(node.mark, HoleMark) and node.mark.expected_type:
        ty = node.mark.expected_type
    else:
        ty = ANY

    return SynthesisResult(
        synthesized_type=ty,
        node=node.with_type(ty),
    )
