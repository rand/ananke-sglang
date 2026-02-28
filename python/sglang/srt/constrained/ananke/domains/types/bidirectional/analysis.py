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
"""Type analysis (↓) for bidirectional typing.

Analysis checks expressions against an expected type from context.
The type flows DOWN from the context to the expression.

Analyzable forms:
- Lambda expressions - get parameter types from expected function type
- Holes - get expected type from context
- Any synthesizable form - synthesize then check against expected

The key insight is that analysis enables:
- Lambda expressions to be checked (they can't synthesize alone)
- Better error messages (we know what was expected)
- Type-directed completion (holes know their expected type)

References:
    - Pierce & Turner (2000). "Local Type Inference"
    - Dunfield & Krishnaswami (2019). "Bidirectional Typing"
    - POPL 2024: "Total Type Error Localization and Recovery with Holes"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

# Support both relative imports (when used as subpackage) and absolute imports (standalone testing)
try:
    from ..constraint import (
        ANY,
        AnyType,
        FunctionType,
        HoleType,
        Type,
    )
    from ..environment import TypeEnvironment
    from ..marking.marks import HoleMark, InconsistentMark, create_inconsistent_mark
    from ..marking.marked_ast import (
        ASTNodeKind,
        MarkedASTNode,
    )
    from ..marking.provenance import Provenance, CONTEXT_EXPRESSION
    from .synthesis import synthesize, SynthesisResult
    from .subsumption import check_subsumption, subsumes
except ImportError:
    from domains.types.constraint import (
        ANY,
        AnyType,
        FunctionType,
        HoleType,
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
        MarkedASTNode,
    )
    from domains.types.marking.provenance import Provenance, CONTEXT_EXPRESSION
    from domains.types.bidirectional.synthesis import synthesize, SynthesisResult
    from domains.types.bidirectional.subsumption import check_subsumption, subsumes


@dataclass
class AnalysisResult:
    """Result of type analysis.

    Attributes:
        success: True if the expression matches the expected type
        node: The marked AST node (may have error marks)
        errors: List of type errors encountered
    """

    success: bool
    node: MarkedASTNode
    errors: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []

    @property
    def has_errors(self) -> bool:
        """True if analysis encountered errors."""
        return not self.success or len(self.errors) > 0


def analyze(
    node: MarkedASTNode,
    expected: Type,
    env: TypeEnvironment,
) -> AnalysisResult:
    """Analyze an AST node against an expected type.

    Type flows DOWN from context. The node is checked to see if it
    can produce the expected type.

    Args:
        node: The AST node to analyze
        expected: The expected type from context
        env: The type environment

    Returns:
        AnalysisResult with success status
    """
    # If expected is Any, anything is fine
    if isinstance(expected, AnyType):
        synth_result = synthesize(node, env)
        return AnalysisResult(
            success=True,
            node=synth_result.node,
            errors=synth_result.errors,
        )

    # Dispatch based on node kind
    if node.kind == ASTNodeKind.LAMBDA:
        return _analyze_lambda(node, expected, env)

    if node.kind == ASTNodeKind.HOLE:
        return _analyze_hole(node, expected, env)

    if node.kind == ASTNodeKind.LET:
        return _analyze_let(node, expected, env)

    if node.kind == ASTNodeKind.IF:
        return _analyze_if(node, expected, env)

    # For other forms: synthesize then check subsumption
    return _analyze_via_synthesis(node, expected, env)


def _analyze_via_synthesis(
    node: MarkedASTNode,
    expected: Type,
    env: TypeEnvironment,
) -> AnalysisResult:
    """Analyze by synthesizing then checking subsumption.

    This is the fallback for nodes that can be synthesized.
    """
    synth_result = synthesize(node, env)
    synthesized = synth_result.synthesized_type

    # Check if synthesized type subsumes expected
    subsumption = check_subsumption(synthesized, expected)

    if subsumption.success:
        return AnalysisResult(
            success=True,
            node=synth_result.node,
            errors=synth_result.errors,
        )

    # Type mismatch - create error mark
    provenance = Provenance(
        location=node.span,
        context=CONTEXT_EXPRESSION,
    )
    mark = create_inconsistent_mark(synthesized, expected, provenance)

    return AnalysisResult(
        success=False,
        node=synth_result.node.with_mark(mark),
        errors=synth_result.errors + [
            f"Type mismatch: expected {expected}, got {synthesized}"
        ],
    )


def _analyze_lambda(
    node: MarkedASTNode,
    expected: Type,
    env: TypeEnvironment,
) -> AnalysisResult:
    """Analyze a lambda expression against an expected function type.

    This is where bidirectional typing shines - lambdas get their
    parameter types from the expected type.
    """
    # Expected must be a function type
    if not isinstance(expected, FunctionType):
        provenance = Provenance(
            location=node.span,
            context="lambda expression",
        )
        mark = create_inconsistent_mark(
            FunctionType((), ANY),  # Approximate lambda type
            expected,
            provenance,
        )
        return AnalysisResult(
            success=False,
            node=node.with_mark(mark),
            errors=[f"Lambda expected function type, got {expected}"],
        )

    # Get parameter names from node data
    param_names = node.data.get("params", [])

    # Check arity
    if len(param_names) != len(expected.params):
        provenance = Provenance(
            location=node.span,
            context="lambda parameters",
        )
        return AnalysisResult(
            success=False,
            node=node,
            errors=[
                f"Lambda has {len(param_names)} parameters, "
                f"expected {len(expected.params)}"
            ],
        )

    # Extend environment with parameter types from expected
    new_env = env
    for name, param_type in zip(param_names, expected.params):
        new_env = new_env.bind(name, param_type)

    # Analyze body against expected return type
    if node.children:
        body_node = node.children[0]
        body_result = analyze(body_node, expected.returns, new_env)

        new_children = [body_result.node] + node.children[1:]
        result_node = node.with_children(new_children).with_type(expected)

        return AnalysisResult(
            success=body_result.success,
            node=result_node,
            errors=body_result.errors,
        )

    # No body - success with expected type
    return AnalysisResult(
        success=True,
        node=node.with_type(expected),
    )


def _analyze_hole(
    node: MarkedASTNode,
    expected: Type,
    env: TypeEnvironment,
) -> AnalysisResult:
    """Analyze a hole against an expected type.

    Holes are marked with the expected type from context.
    """
    # Get or create hole ID
    if isinstance(node.mark, HoleMark):
        hole_id = node.mark.hole_id
        provenance = node.mark.provenance
    else:
        hole_id = node.node_id or f"hole_{id(node)}"
        provenance = Provenance(
            location=node.span,
            context="hole",
        )

    # Create hole mark with expected type
    mark = HoleMark(hole_id, expected, provenance)

    return AnalysisResult(
        success=True,  # Holes always "succeed" - they're just incomplete
        node=node.with_mark(mark).with_type(expected),
    )


def _analyze_let(
    node: MarkedASTNode,
    expected: Type,
    env: TypeEnvironment,
) -> AnalysisResult:
    """Analyze a let expression.

    let x = e1 in e2

    1. Synthesize type of e1
    2. Extend env with x : T1
    3. Analyze e2 against expected
    """
    if len(node.children) < 2:
        return AnalysisResult(
            success=False,
            node=node,
            errors=["Malformed let expression"],
        )

    binding_node = node.children[0]
    body_node = node.children[1]

    # Get binding name
    var_name = node.data.get("binding_name", "_")

    # Synthesize binding type
    binding_result = synthesize(binding_node, env)

    # Extend environment
    new_env = env.bind(var_name, binding_result.synthesized_type)

    # Analyze body against expected
    body_result = analyze(body_node, expected, new_env)

    new_children = [binding_result.node, body_result.node] + node.children[2:]

    return AnalysisResult(
        success=body_result.success,
        node=node.with_children(new_children).with_type(
            body_result.node.get_type()
        ),
        errors=binding_result.errors + body_result.errors,
    )


def _analyze_if(
    node: MarkedASTNode,
    expected: Type,
    env: TypeEnvironment,
) -> AnalysisResult:
    """Analyze an if expression.

    if cond then e1 else e2

    Analyze both branches against the expected type.
    """
    if len(node.children) < 3:
        return AnalysisResult(
            success=False,
            node=node,
            errors=["Malformed if expression"],
        )

    cond_node, then_node, else_node = node.children[:3]

    # Synthesize condition (should be bool)
    cond_result = synthesize(cond_node, env)

    # Analyze both branches against expected
    then_result = analyze(then_node, expected, env)
    else_result = analyze(else_node, expected, env)

    new_children = [
        cond_result.node,
        then_result.node,
        else_result.node,
    ] + node.children[3:]

    success = then_result.success and else_result.success
    errors = cond_result.errors + then_result.errors + else_result.errors

    return AnalysisResult(
        success=success,
        node=node.with_children(new_children).with_type(expected),
        errors=errors,
    )


def analyze_against_expected(
    node: MarkedASTNode,
    expected: Optional[Type],
    env: TypeEnvironment,
) -> AnalysisResult:
    """Analyze or synthesize based on whether expected type is provided.

    This is a convenience function that:
    - If expected is provided, calls analyze
    - If expected is None, calls synthesize and wraps result

    Args:
        node: The AST node
        expected: Optional expected type
        env: Type environment

    Returns:
        AnalysisResult
    """
    if expected is not None:
        return analyze(node, expected, env)

    # No expected type - synthesize
    synth_result = synthesize(node, env)
    return AnalysisResult(
        success=not synth_result.has_errors,
        node=synth_result.node,
        errors=synth_result.errors,
    )
