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
"""ChatLSP-style context extraction for typed holes.

This module implements the 5 ChatLSP methods from "Statically Contextualizing
LLMs with Typed Holes" (OOPSLA 2024). These methods extract rich type context
from holes in partial programs to guide LLM generation.

The 5 methods are:
1. expected_type(hole) → structured type info
2. relevant_types(hole, env) → ranked relevant bindings
3. relevant_headers(hole, env) → function signatures
4. error_report(marked_ast) → actionable type errors
5. ai_tutorial(hole, env) → natural language guidance

References:
    - OOPSLA 2024: "Statically Contextualizing Large Language Models with Typed Holes"
    - https://arxiv.org/abs/2409.00921
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable
from enum import Enum, auto

from domains.types.constraint import (
    Type,
    TypeVar,
    PrimitiveType,
    FunctionType,
    ListType,
    DictType,
    TupleType,
    SetType,
    UnionType,
    ClassType,
    AnyType,
    NeverType,
    HoleType,
    INT,
    STR,
    BOOL,
    FLOAT,
    NONE,
    ANY,
)
from domains.types.environment import TypeEnvironment, EMPTY_ENVIRONMENT
from domains.types.unification import is_subtype as full_is_subtype
from domains.types.marking.marks import Mark, HoleMark, InconsistentMark
from domains.types.marking.provenance import SourceSpan, Provenance
from domains.types.marking.marked_ast import MarkedAST, MarkedASTNode, ASTNodeKind


@dataclass(frozen=True, slots=True)
class ExpectedTypeInfo:
    """Structured information about the expected type at a hole.

    Attributes:
        type: The expected Type object
        type_string: Human-readable type string
        constraints: List of constraint descriptions
        examples: Example values of this type
        is_callable: Whether a callable is expected
        is_iterable: Whether an iterable is expected
    """

    type: Type
    type_string: str
    constraints: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    is_callable: bool = False
    is_iterable: bool = False

    @staticmethod
    def unknown() -> 'ExpectedTypeInfo':
        """Create an unknown type info."""
        return ExpectedTypeInfo(
            type=ANY,
            type_string="Any",
            constraints=[],
            examples=[],
        )


@dataclass(frozen=True, slots=True)
class RelevantBinding:
    """A binding relevant to filling a hole.

    Attributes:
        name: The binding name
        type: The type of the binding
        relevance: Relevance score (0.0 to 1.0)
        source: Where the binding comes from
        doc: Optional documentation
    """

    name: str
    type: Type
    relevance: float
    source: str = "local"  # "local", "parameter", "global", "import"
    doc: Optional[str] = None


@dataclass(frozen=True, slots=True)
class FunctionSignature:
    """A function signature relevant to a hole.

    Attributes:
        name: Function name
        params: Parameter names and types
        return_type: Return type
        doc: Optional documentation
        relevance: Relevance score
    """

    name: str
    params: Tuple[Tuple[str, Type], ...]
    return_type: Type
    doc: Optional[str] = None
    relevance: float = 0.0

    def format(self) -> str:
        """Format as a readable signature."""
        param_strs = [f"{n}: {_format_type(t)}" for n, t in self.params]
        return f"{self.name}({', '.join(param_strs)}) -> {_format_type(self.return_type)}"


@dataclass(frozen=True, slots=True)
class TypeErrorReport:
    """Actionable report of a type error.

    Attributes:
        location: Source span of the error
        message: Human-readable error message
        expected: Expected type
        got: Actual type
        suggestions: List of suggestions to fix
        severity: Error severity
    """

    location: SourceSpan
    message: str
    expected: Type
    got: Type
    suggestions: List[str] = field(default_factory=list)
    severity: str = "error"  # "error", "warning", "hint"


def _format_type(t: Type) -> str:
    """Format a type as a string."""
    if isinstance(t, PrimitiveType):
        return t.name
    elif isinstance(t, FunctionType):
        params = ", ".join(_format_type(p) for p in t.params)
        ret = _format_type(t.returns)
        return f"Callable[[{params}], {ret}]"
    elif isinstance(t, ListType):
        return f"list[{_format_type(t.element)}]"
    elif isinstance(t, DictType):
        return f"dict[{_format_type(t.key)}, {_format_type(t.value)}]"
    elif isinstance(t, TupleType):
        elems = ", ".join(_format_type(e) for e in t.elements)
        return f"tuple[{elems}]"
    elif isinstance(t, SetType):
        return f"set[{_format_type(t.element)}]"
    elif isinstance(t, UnionType):
        members = " | ".join(_format_type(m) for m in t.members)
        return members
    elif isinstance(t, ClassType):
        if t.type_args:
            args = ", ".join(_format_type(a) for a in t.type_args)
            return f"{t.name}[{args}]"
        return t.name
    elif isinstance(t, TypeVar):
        return t.name
    elif isinstance(t, AnyType):
        return "Any"
    elif isinstance(t, NeverType):
        return "Never"
    elif isinstance(t, HoleType):
        return f"?{t.hole_id}"
    else:
        return str(t)


class ContextExtractor:
    """Extracts type context from holes for LLM guidance.

    Implements the 5 ChatLSP methods for extracting rich context
    from typed holes in partial programs.

    Example:
        >>> extractor = ContextExtractor()
        >>> hole = HoleMark("h0", INT, None)
        >>> info = extractor.expected_type(hole, ast)
        >>> print(info.type_string)
        "int"
    """

    def __init__(self, language: str = "python") -> None:
        """Initialize the context extractor.

        Args:
            language: Target language for formatting
        """
        self.language = language

    def expected_type(
        self,
        hole: HoleMark,
        marked_ast: MarkedAST,
    ) -> ExpectedTypeInfo:
        """ChatLSP Method 1: Get expected type at a hole.

        Returns structured type information that can be used
        to constrain generation.

        Args:
            hole: The hole mark
            marked_ast: The containing AST

        Returns:
            ExpectedTypeInfo with type details
        """
        if hole.expected_type is None:
            return ExpectedTypeInfo.unknown()

        typ = hole.expected_type
        type_string = _format_type(typ)

        # Determine constraints
        constraints: List[str] = []
        if isinstance(typ, FunctionType):
            constraints.append(f"Must be callable with {len(typ.params)} arguments")
            constraints.append(f"Must return {_format_type(typ.returns)}")
        elif isinstance(typ, ListType):
            constraints.append(f"Must be a list of {_format_type(typ.element)}")
        elif isinstance(typ, UnionType):
            constraints.append(f"Must be one of: {type_string}")

        # Generate examples
        examples = self._generate_examples(typ)

        return ExpectedTypeInfo(
            type=typ,
            type_string=type_string,
            constraints=constraints,
            examples=examples,
            is_callable=isinstance(typ, FunctionType),
            is_iterable=isinstance(typ, (ListType, SetType, TupleType)),
        )

    def relevant_types(
        self,
        hole: HoleMark,
        env: TypeEnvironment,
        limit: int = 20,
    ) -> List[RelevantBinding]:
        """ChatLSP Method 2: Get types relevant to filling a hole.

        Ranks available bindings by relevance to the expected type.

        Args:
            hole: The hole mark
            env: The type environment
            limit: Maximum number of results

        Returns:
            List of relevant bindings, sorted by relevance
        """
        expected = hole.expected_type or ANY
        bindings: List[RelevantBinding] = []

        for name, typ in env.all_bindings().items():
            relevance = self._compute_relevance(typ, expected)
            if relevance > 0.0:
                bindings.append(RelevantBinding(
                    name=name,
                    type=typ,
                    relevance=relevance,
                ))

        # Sort by relevance descending
        bindings.sort(key=lambda b: b.relevance, reverse=True)
        return bindings[:limit]

    def relevant_headers(
        self,
        hole: HoleMark,
        env: TypeEnvironment,
        limit: int = 10,
    ) -> List[FunctionSignature]:
        """ChatLSP Method 3: Get function signatures relevant to a hole.

        Useful when the hole expects a function type or function call.

        Args:
            hole: The hole mark
            env: The type environment
            limit: Maximum number of results

        Returns:
            List of relevant function signatures
        """
        expected = hole.expected_type or ANY
        signatures: List[FunctionSignature] = []

        for name, typ in env.all_bindings().items():
            if isinstance(typ, FunctionType):
                # Compute relevance based on return type match
                relevance = self._compute_relevance(typ.returns, expected)

                # Also consider if we need a function itself
                if isinstance(expected, FunctionType):
                    fn_relevance = self._compute_function_relevance(typ, expected)
                    relevance = max(relevance, fn_relevance)

                if relevance > 0.0:
                    params = tuple(
                        (f"arg{i}", p) for i, p in enumerate(typ.params)
                    )
                    signatures.append(FunctionSignature(
                        name=name,
                        params=params,
                        return_type=typ.returns,
                        relevance=relevance,
                    ))

        # Sort by relevance descending
        signatures.sort(key=lambda s: s.relevance, reverse=True)
        return signatures[:limit]

    def error_report(
        self,
        marked_ast: MarkedAST,
    ) -> List[TypeErrorReport]:
        """ChatLSP Method 4: Get actionable error reports.

        Collects all type errors in the AST with suggestions.

        Args:
            marked_ast: The marked AST

        Returns:
            List of error reports
        """
        reports: List[TypeErrorReport] = []

        for node in self._iterate_nodes(marked_ast.root):
            if isinstance(node.mark, InconsistentMark):
                mark = node.mark
                suggestions = self._generate_suggestions(mark)

                reports.append(TypeErrorReport(
                    location=mark.provenance.location if mark.provenance else node.span,
                    message=f"Type mismatch: expected {_format_type(mark.expected)}, got {_format_type(mark.synthesized)}",
                    expected=mark.expected,
                    got=mark.synthesized,
                    suggestions=suggestions,
                ))

        return reports

    def ai_tutorial(
        self,
        hole: HoleMark,
        env: TypeEnvironment,
    ) -> str:
        """ChatLSP Method 5: Generate natural language guidance.

        Explains what the hole needs in human-readable form.

        Args:
            hole: The hole mark
            env: The type environment

        Returns:
            Tutorial string
        """
        expected = hole.expected_type
        if expected is None:
            return "Fill this hole with any expression."

        parts: List[str] = []
        type_str = _format_type(expected)

        parts.append(f"Fill this hole with an expression of type `{type_str}`.")

        # Add specific guidance based on type
        if isinstance(expected, PrimitiveType):
            parts.append(self._primitive_tutorial(expected))
        elif isinstance(expected, FunctionType):
            parts.append(self._function_tutorial(expected))
        elif isinstance(expected, ListType):
            parts.append(self._list_tutorial(expected))
        elif isinstance(expected, DictType):
            parts.append(self._dict_tutorial(expected))
        elif isinstance(expected, UnionType):
            parts.append(self._union_tutorial(expected))

        # Add relevant bindings
        relevant = self.relevant_types(hole, env, limit=5)
        if relevant:
            parts.append("\nAvailable bindings that might help:")
            for b in relevant:
                parts.append(f"  - `{b.name}`: {_format_type(b.type)}")

        return "\n".join(parts)

    def _generate_examples(self, typ: Type) -> List[str]:
        """Generate example values for a type."""
        examples: List[str] = []

        if isinstance(typ, PrimitiveType):
            if typ.name == "int":
                examples = ["0", "42", "-1"]
            elif typ.name == "str":
                examples = ['""', '"hello"', '"example"']
            elif typ.name == "bool":
                examples = ["True", "False"]
            elif typ.name == "float":
                examples = ["0.0", "3.14", "-1.5"]
            elif typ.name == "None":
                examples = ["None"]
        elif isinstance(typ, ListType):
            elem = _format_type(typ.element)
            examples = [f"[]", f"[{self._example_value(typ.element)}]"]
        elif isinstance(typ, DictType):
            k = self._example_value(typ.key)
            v = self._example_value(typ.value)
            examples = ["{}", f"{{{k}: {v}}}"]
        elif isinstance(typ, TupleType):
            elems = ", ".join(self._example_value(e) for e in typ.elements)
            examples = [f"({elems})"]
        elif isinstance(typ, FunctionType):
            params = ", ".join(f"x{i}" for i in range(len(typ.params)))
            examples = [f"lambda {params}: ..."]

        return examples

    def _example_value(self, typ: Type) -> str:
        """Generate a single example value for a type."""
        if isinstance(typ, PrimitiveType):
            if typ.name == "int":
                return "0"
            elif typ.name == "str":
                return '""'
            elif typ.name == "bool":
                return "True"
            elif typ.name == "float":
                return "0.0"
            elif typ.name == "None":
                return "None"
        return "..."

    def _compute_relevance(self, typ: Type, expected: Type) -> float:
        """Compute how relevant a type is to an expected type."""
        # Exact match
        if typ == expected:
            return 1.0

        # Any matches everything
        if isinstance(expected, AnyType):
            return 0.5
        if isinstance(typ, AnyType):
            return 0.3

        # Subtype relationship
        if self._is_subtype(typ, expected):
            return 0.8

        # Partial match for generics
        if isinstance(typ, ListType) and isinstance(expected, ListType):
            return 0.7 * self._compute_relevance(typ.element, expected.element)

        if isinstance(typ, DictType) and isinstance(expected, DictType):
            k_rel = self._compute_relevance(typ.key, expected.key)
            v_rel = self._compute_relevance(typ.value, expected.value)
            return 0.7 * (k_rel + v_rel) / 2

        # Union membership
        if isinstance(expected, UnionType):
            if any(self._is_subtype(typ, m) for m in expected.members):
                return 0.6

        return 0.0

    def _compute_function_relevance(
        self,
        func: FunctionType,
        expected: FunctionType
    ) -> float:
        """Compute relevance between function types."""
        # Check parameter count
        if len(func.params) != len(expected.params):
            return 0.0

        # Check return type
        ret_rel = self._compute_relevance(func.returns, expected.returns)

        # Check parameter types (contravariant)
        param_rel = 0.0
        if func.params:
            param_rels = [
                self._compute_relevance(e, f)  # Contravariant
                for f, e in zip(func.params, expected.params)
            ]
            param_rel = sum(param_rels) / len(param_rels)

        return (ret_rel + param_rel) / 2

    def _is_subtype(self, sub: Type, sup: Type) -> bool:
        """Check if sub is a subtype of sup.

        Uses the full is_subtype implementation from unification module which handles:
        - Any/Never as top/bottom types
        - HoleType as wildcard
        - Union subtyping
        - List/Set/Dict/Tuple covariance
        - Function contravariant params, covariant return
        - Protocol structural subtyping
        - Numeric promotion (int <: float)
        """
        return full_is_subtype(sub, sup)

    def _generate_suggestions(self, mark: InconsistentMark) -> List[str]:
        """Generate suggestions for fixing a type error."""
        suggestions: List[str] = []

        expected = mark.expected
        got = mark.synthesized

        # Type conversion suggestions
        if isinstance(expected, PrimitiveType) and isinstance(got, PrimitiveType):
            if expected.name == "str" and got.name == "int":
                suggestions.append("Convert to string with str(...)")
            elif expected.name == "int" and got.name == "str":
                suggestions.append("Convert to int with int(...)")
            elif expected.name == "float" and got.name == "str":
                suggestions.append("Convert to float with float(...)")
            elif expected.name == "int" and got.name == "float":
                suggestions.append("Convert to int with int(...) (may lose precision)")

        # List/collection suggestions
        if isinstance(expected, ListType) and not isinstance(got, ListType):
            suggestions.append(f"Wrap in a list: [{_format_type(got)}]")

        return suggestions

    def _iterate_nodes(self, node: MarkedASTNode) -> List[MarkedASTNode]:
        """Iterate all nodes in an AST."""
        nodes = [node]
        for child in node.children:
            nodes.extend(self._iterate_nodes(child))
        return nodes

    def _primitive_tutorial(self, typ: PrimitiveType) -> str:
        """Generate tutorial for primitive types."""
        tutorials = {
            "int": "An integer like 0, 42, or -1.",
            "str": 'A string like "", "hello", or \'world\'.',
            "bool": "A boolean: True or False.",
            "float": "A floating-point number like 3.14 or -0.5.",
            "None": "The None value.",
        }
        return tutorials.get(typ.name, f"A {typ.name} value.")

    def _function_tutorial(self, typ: FunctionType) -> str:
        """Generate tutorial for function types."""
        n_params = len(typ.params)
        ret = _format_type(typ.returns)
        if n_params == 0:
            return f"A function that takes no arguments and returns {ret}."
        elif n_params == 1:
            param = _format_type(typ.params[0])
            return f"A function that takes a {param} and returns {ret}."
        else:
            params = ", ".join(_format_type(p) for p in typ.params)
            return f"A function that takes ({params}) and returns {ret}."

    def _list_tutorial(self, typ: ListType) -> str:
        """Generate tutorial for list types."""
        elem = _format_type(typ.element)
        return f"A list containing elements of type {elem}. Examples: [], [{elem} value]."

    def _dict_tutorial(self, typ: DictType) -> str:
        """Generate tutorial for dict types."""
        k = _format_type(typ.key)
        v = _format_type(typ.value)
        return f"A dictionary with {k} keys and {v} values. Examples: {{}}, {{{k}: {v}}}."

    def _union_tutorial(self, typ: UnionType) -> str:
        """Generate tutorial for union types."""
        members = " or ".join(_format_type(m) for m in typ.members)
        return f"One of: {members}."


# Convenience function for quick context extraction
def extract_hole_context(
    hole_id: str,
    marked_ast: MarkedAST,
    env: TypeEnvironment,
) -> Dict[str, Any]:
    """Extract context for a specific hole.

    Args:
        hole_id: The hole ID
        marked_ast: The marked AST
        env: The type environment

    Returns:
        Dictionary with all context information
    """
    extractor = ContextExtractor()

    # Find the hole mark
    hole_mark = _find_hole_mark(marked_ast.root, hole_id)
    if hole_mark is None:
        return {"error": f"Hole {hole_id} not found"}

    return {
        "expected_type": extractor.expected_type(hole_mark, marked_ast),
        "relevant_types": extractor.relevant_types(hole_mark, env),
        "relevant_headers": extractor.relevant_headers(hole_mark, env),
        "errors": extractor.error_report(marked_ast),
        "tutorial": extractor.ai_tutorial(hole_mark, env),
    }


def _find_hole_mark(node: MarkedASTNode, hole_id: str) -> Optional[HoleMark]:
    """Find a hole mark by ID in an AST."""
    if isinstance(node.mark, HoleMark) and node.mark.hole_id == hole_id:
        return node.mark

    for child in node.children:
        result = _find_hole_mark(child, hole_id)
        if result:
            return result

    return None
