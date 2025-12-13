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
"""Python incremental parser using tree-sitter.

This module provides a Python-specific incremental parser that converts
token streams into MarkedASTNodes suitable for type checking.

Features:
- Incremental parsing via tree-sitter-python
- Hole detection for incomplete expressions
- Source span tracking for error messages
- Recovery from common Python syntax errors

References:
    - tree-sitter-python: https://github.com/tree-sitter/tree-sitter-python
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set

from parsing.base import (
    IncrementalParser,
    ParseState,
    ParseResult,
    ParseError,
    TokenInfo,
    HoleDetector,
    SourceTracker,
)
from parsing.partial_ast import PartialAST, HoleInfo, HoleKind
from domains.types.marking.provenance import SourceSpan, UNKNOWN_SPAN
from domains.types.marking.marked_ast import (
    MarkedASTNode,
    ASTNodeKind,
    create_hole_node,
    create_literal_node,
    create_variable_node,
)
from domains.types.constraint import (
    Type,
    INT,
    STR,
    BOOL,
    FLOAT,
    NONE,
    ANY,
    ListType,
    DictType,
    TupleType,
)


# Token categories for hole detection
EXPRESSION_STARTERS = {
    "identifier", "number", "string", "true", "false", "none",
    "(", "[", "{", "lambda", "not", "-", "+", "~",
}

EXPRESSION_ENDERS = {
    "identifier", "number", "string", "true", "false", "none",
    ")", "]", "}",
}

INCOMPLETE_CONTEXTS = {
    "binary_operator", "unary_operator", "call", "subscript",
    "assignment", "augmented_assignment", "return_statement",
    "if_statement", "while_statement", "for_statement",
}


@dataclass
class PythonParserCheckpoint:
    """Checkpoint for Python parser state."""

    source: str
    tokens: List[TokenInfo]
    holes: Dict[str, HoleInfo]
    hole_counter: int
    tracker_checkpoint: Any


class PythonIncrementalParser(IncrementalParser):
    """Python-specific incremental parser.

    This parser uses a lightweight approach suitable for token-by-token
    generation. It maintains a partial AST and detects holes where
    more code is expected.

    For full tree-sitter integration, consider using the tree-sitter
    Python bindings directly. This implementation provides a simpler
    fallback that works without the C library.

    Example:
        >>> parser = PythonIncrementalParser()
        >>> result = parser.parse_initial("def foo(x: int)")
        >>> result = parser.extend_with_text(" -> ")
        >>> holes = parser.find_holes()
        >>> # Expect a hole for the return type
    """

    def __init__(self) -> None:
        """Initialize the parser."""
        self._source: str = ""
        self._tokens: List[TokenInfo] = []
        self._holes: Dict[str, HoleInfo] = {}
        self._hole_counter: int = 0
        self._tracker = SourceTracker()
        self._ast: Optional[MarkedASTNode] = None
        self._state: ParseState = ParseState.VALID
        self._errors: List[ParseError] = []

    @property
    def language(self) -> str:
        return "python"

    @property
    def current_source(self) -> str:
        return self._source

    @property
    def current_position(self) -> int:
        return len(self._source)

    def parse_initial(self, source: str) -> ParseResult:
        """Parse an initial source string."""
        self._source = source
        self._tracker.reset()
        self._tokens = []
        self._holes = {}
        self._hole_counter = 0
        self._errors = []

        if source:
            self._tracker.append(source)

        # Parse the source
        self._parse_source()

        return self._create_result()

    def extend_with_token(self, token: TokenInfo) -> ParseResult:
        """Extend with a new token."""
        span = self._tracker.append_token(token)
        self._source += token.text
        self._tokens.append(token)

        # Re-parse with new content
        self._parse_source()

        return self._create_result()

    def extend_with_text(self, text: str) -> ParseResult:
        """Extend with raw text."""
        self._tracker.append(text)
        self._source += text

        # Re-parse with new content
        self._parse_source()

        return self._create_result()

    def find_holes(self) -> List[Tuple[str, SourceSpan]]:
        """Find all holes in the current parse."""
        return [(h.hole_id, h.span) for h in self._holes.values()]

    def get_expected_tokens(self) -> List[str]:
        """Get expected tokens at current position."""
        # Analyze the end of the source to determine what's expected
        source = self._source.rstrip()

        if not source:
            return list(EXPRESSION_STARTERS)

        # Check last significant character/token
        last_char = source[-1]

        if last_char in "([{":
            return list(EXPRESSION_STARTERS)
        elif last_char in "+-*/=<>!&|^%":
            return list(EXPRESSION_STARTERS)
        elif last_char == ",":
            return list(EXPRESSION_STARTERS)
        elif last_char == ":":
            # Could be type annotation or block start
            return ["type", "newline"] + list(EXPRESSION_STARTERS)
        elif last_char == ".":
            return ["identifier"]
        elif source.endswith("def "):
            return ["identifier"]
        elif source.endswith("class "):
            return ["identifier"]
        elif source.endswith("import "):
            return ["identifier", "*"]
        elif source.endswith("from "):
            return ["identifier"]
        elif source.endswith("return"):
            return list(EXPRESSION_STARTERS) + ["newline"]

        return list(EXPRESSION_STARTERS) + list(EXPRESSION_ENDERS)

    def checkpoint(self) -> PythonParserCheckpoint:
        """Create a checkpoint."""
        return PythonParserCheckpoint(
            source=self._source,
            tokens=self._tokens.copy(),
            holes=self._holes.copy(),
            hole_counter=self._hole_counter,
            tracker_checkpoint=self._tracker.checkpoint(),
        )

    def restore(self, checkpoint: Any) -> None:
        """Restore from a checkpoint."""
        if not isinstance(checkpoint, PythonParserCheckpoint):
            raise TypeError("Invalid checkpoint type")

        self._source = checkpoint.source
        self._tokens = checkpoint.tokens.copy()
        self._holes = checkpoint.holes.copy()
        self._hole_counter = checkpoint.hole_counter
        self._tracker.restore(checkpoint.tracker_checkpoint)

        # Re-parse to restore AST
        self._parse_source()

    def get_ast(self) -> Optional[MarkedASTNode]:
        """Get the current AST."""
        return self._ast

    def copy(self) -> 'PythonIncrementalParser':
        """Create an independent copy."""
        new_parser = PythonIncrementalParser()
        checkpoint = self.checkpoint()
        new_parser.restore(checkpoint)
        return new_parser

    def _parse_source(self) -> None:
        """Parse the current source into an AST."""
        self._errors = []
        self._holes = {}

        if not self._source.strip():
            # Empty source - create a hole
            self._ast = self._create_module_hole()
            self._state = ParseState.PARTIAL
            return

        try:
            # Attempt to parse using a simple Python AST approach
            self._ast = self._build_ast_from_source()
            self._detect_holes()

            if self._holes:
                self._state = ParseState.PARTIAL
            else:
                self._state = ParseState.VALID

        except SyntaxError as e:
            # Syntax error - mark as partial with hole
            self._ast = self._create_error_recovery_ast(str(e))
            self._state = ParseState.PARTIAL
            self._errors.append(ParseError(
                message=str(e),
                span=UNKNOWN_SPAN,
            ))

    def _build_ast_from_source(self) -> MarkedASTNode:
        """Build a MarkedASTNode from Python source.

        This is a simplified implementation. For production use,
        integrate with tree-sitter-python for proper incremental parsing.
        """
        source = self._source.strip()

        # Try to classify the top-level construct
        if source.startswith("def "):
            return self._parse_function_def(source)
        elif source.startswith("class "):
            return self._parse_class_def(source)
        elif source.startswith("import ") or source.startswith("from "):
            return self._parse_import(source)
        elif "=" in source and not any(op in source for op in ["==", "!=", "<=", ">="]):
            return self._parse_assignment(source)
        else:
            return self._parse_expression(source)

    def _parse_function_def(self, source: str) -> MarkedASTNode:
        """Parse a function definition."""
        # Simple extraction of function components
        span = self._tracker.span_at(0, len(source))

        # Check if function is complete
        if ":" not in source or not source.rstrip().endswith(":"):
            # Incomplete function - add hole
            hole_id = self._new_hole_id()
            self._holes[hole_id] = HoleInfo(
                hole_id=hole_id,
                kind=HoleKind.BODY,
                span=span,
                context="function body",
            )

        return MarkedASTNode(
            kind=ASTNodeKind.FUNCTION_DEF,
            span=span,
            data={"source": source},
        )

    def _parse_class_def(self, source: str) -> MarkedASTNode:
        """Parse a class definition."""
        span = self._tracker.span_at(0, len(source))

        return MarkedASTNode(
            kind=ASTNodeKind.CLASS_DEF,
            span=span,
            data={"source": source},
        )

    def _parse_import(self, source: str) -> MarkedASTNode:
        """Parse an import statement."""
        span = self._tracker.span_at(0, len(source))

        return MarkedASTNode(
            kind=ASTNodeKind.IMPORT,
            span=span,
            data={"source": source},
        )

    def _parse_assignment(self, source: str) -> MarkedASTNode:
        """Parse an assignment statement."""
        span = self._tracker.span_at(0, len(source))

        # Check for incomplete RHS
        parts = source.split("=", 1)
        if len(parts) == 2 and not parts[1].strip():
            # Incomplete assignment
            hole_id = self._new_hole_id()
            self._holes[hole_id] = HoleInfo(
                hole_id=hole_id,
                kind=HoleKind.EXPRESSION,
                span=span,
                context="assignment value",
            )

        return MarkedASTNode(
            kind=ASTNodeKind.ASSIGNMENT,
            span=span,
            data={"source": source},
        )

    def _parse_expression(self, source: str) -> MarkedASTNode:
        """Parse an expression."""
        span = self._tracker.span_at(0, len(source))

        # Check for literals
        try:
            import ast as python_ast
            tree = python_ast.parse(source, mode='eval')
            return self._convert_python_ast(tree.body, span)
        except SyntaxError:
            # Incomplete expression - create hole
            hole_id = self._new_hole_id()
            self._holes[hole_id] = HoleInfo(
                hole_id=hole_id,
                kind=HoleKind.EXPRESSION,
                span=span,
                context="expression",
            )
            return create_hole_node(
                hole_id=hole_id,
                span=span,
                expected_type=ANY,
            )

    def _convert_python_ast(self, node: Any, span: SourceSpan) -> MarkedASTNode:
        """Convert a Python AST node to MarkedASTNode."""
        import ast as python_ast

        if isinstance(node, python_ast.Constant):
            # Literal value
            value = node.value
            if isinstance(value, int) and not isinstance(value, bool):
                return create_literal_node(value=value, ty=INT, span=span)
            elif isinstance(value, float):
                return create_literal_node(value=value, ty=FLOAT, span=span)
            elif isinstance(value, str):
                return create_literal_node(value=value, ty=STR, span=span)
            elif isinstance(value, bool):
                return create_literal_node(value=value, ty=BOOL, span=span)
            elif value is None:
                return create_literal_node(value=None, ty=NONE, span=span)
            else:
                return create_literal_node(value=value, ty=ANY, span=span)

        elif isinstance(node, python_ast.Name):
            return create_variable_node(name=node.id, ty=ANY, span=span)

        elif isinstance(node, python_ast.List):
            children = [self._convert_python_ast(e, span) for e in node.elts]
            return MarkedASTNode(
                kind=ASTNodeKind.LIST,
                span=span,
                children=children,
            )

        elif isinstance(node, python_ast.Tuple):
            children = [self._convert_python_ast(e, span) for e in node.elts]
            return MarkedASTNode(
                kind=ASTNodeKind.TUPLE,
                span=span,
                children=children,
            )

        elif isinstance(node, python_ast.Dict):
            children = []
            for k, v in zip(node.keys, node.values):
                if k is not None:
                    children.append(self._convert_python_ast(k, span))
                children.append(self._convert_python_ast(v, span))
            return MarkedASTNode(
                kind=ASTNodeKind.DICT,
                span=span,
                children=children,
            )

        elif isinstance(node, python_ast.BinOp):
            left = self._convert_python_ast(node.left, span)
            right = self._convert_python_ast(node.right, span)
            return MarkedASTNode(
                kind=ASTNodeKind.BINARY_OP,
                span=span,
                children=[left, right],
                data={"op": type(node.op).__name__},
            )

        elif isinstance(node, python_ast.Call):
            func = self._convert_python_ast(node.func, span)
            args = [self._convert_python_ast(a, span) for a in node.args]
            return MarkedASTNode(
                kind=ASTNodeKind.CALL,
                span=span,
                children=[func] + args,
            )

        else:
            # Unknown node type - return generic expression
            return MarkedASTNode(
                kind=ASTNodeKind.EXPRESSION,
                span=span,
                data={"ast_type": type(node).__name__},
            )

    def _detect_holes(self) -> None:
        """Detect incomplete constructs that represent holes."""
        source = self._source.rstrip()

        if not source:
            return

        last_char = source[-1]

        # Check for trailing operators (incomplete binary expression)
        if last_char in "+-*/%&|^<>=":
            hole_id = self._new_hole_id()
            self._holes[hole_id] = HoleInfo(
                hole_id=hole_id,
                kind=HoleKind.EXPRESSION,
                span=self._tracker.span_at(len(source), len(source)),
                context="operand",
            )

        # Check for unclosed brackets
        open_brackets = 0
        for char in source:
            if char in "([{":
                open_brackets += 1
            elif char in ")]}":
                open_brackets -= 1

        if open_brackets > 0:
            hole_id = self._new_hole_id()
            self._holes[hole_id] = HoleInfo(
                hole_id=hole_id,
                kind=HoleKind.EXPRESSION,
                span=self._tracker.span_at(len(source), len(source)),
                context="closing bracket",
            )

        # Check for trailing comma (incomplete sequence)
        if last_char == ",":
            hole_id = self._new_hole_id()
            self._holes[hole_id] = HoleInfo(
                hole_id=hole_id,
                kind=HoleKind.EXPRESSION,
                span=self._tracker.span_at(len(source), len(source)),
                context="sequence element",
            )

        # Check for trailing colon (incomplete block)
        if last_char == ":":
            hole_id = self._new_hole_id()
            self._holes[hole_id] = HoleInfo(
                hole_id=hole_id,
                kind=HoleKind.BODY,
                span=self._tracker.span_at(len(source), len(source)),
                context="block body",
            )

    def _create_module_hole(self) -> MarkedASTNode:
        """Create a module-level hole."""
        hole_id = self._new_hole_id()
        span = UNKNOWN_SPAN

        self._holes[hole_id] = HoleInfo(
            hole_id=hole_id,
            kind=HoleKind.STATEMENT,
            span=span,
            context="module",
        )

        return create_hole_node(
            hole_id=hole_id,
            span=span,
            expected_type=ANY,
        )

    def _create_error_recovery_ast(self, error: str) -> MarkedASTNode:
        """Create an AST node for error recovery."""
        hole_id = self._new_hole_id()
        span = self._tracker.span_at(0, len(self._source))

        self._holes[hole_id] = HoleInfo(
            hole_id=hole_id,
            kind=HoleKind.EXPRESSION,
            span=span,
            context=f"error recovery: {error}",
        )

        return MarkedASTNode(
            kind=ASTNodeKind.ERROR,
            span=span,
            children=[
                create_hole_node(
                    hole_id=hole_id,
                    span=span,
                    expected_type=ANY,
                )
            ],
            data={"error": error, "source": self._source},
        )

    def _new_hole_id(self) -> str:
        """Generate a new hole ID."""
        hole_id = f"hole_{self._hole_counter}"
        self._hole_counter += 1
        return hole_id

    def _create_result(self) -> ParseResult:
        """Create a ParseResult from current state."""
        return ParseResult(
            state=self._state,
            ast=self._ast,
            errors=self._errors.copy(),
            holes=list(self._holes.keys()),
            position=len(self._source),
        )


# Factory function
def create_python_parser() -> PythonIncrementalParser:
    """Create a new Python incremental parser.

    Returns:
        A fresh PythonIncrementalParser instance
    """
    return PythonIncrementalParser()
