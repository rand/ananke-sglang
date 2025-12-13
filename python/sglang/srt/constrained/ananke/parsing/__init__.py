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
"""Incremental parsing per language using tree-sitter.

This package provides incremental parsing capabilities for converting
token streams into partial ASTs suitable for type checking.

Key Components:
    - IncrementalParser: ABC for language-specific parsers
    - PartialAST: Representation of incomplete programs with holes
    - HoleDetector: Utility for finding holes in ASTs
    - SourceTracker: Position tracking during parsing

Supported Languages:
    - Python (via tree-sitter-python)
    - TypeScript (planned)
    - Rust (planned)
    - Zig (planned)
    - Go (planned)

Usage:
    >>> from parsing import PythonIncrementalParser
    >>> parser = PythonIncrementalParser()
    >>> result = parser.parse_initial("def foo(x: int)")
    >>> result = parser.extend_with_text(" -> str:")
"""

from parsing.base import (
    IncrementalParser,
    ParseState,
    ParseResult,
    ParseError,
    TokenInfo,
    HoleDetector,
    SourceTracker,
)

from parsing.partial_ast import (
    PartialAST,
    HoleInfo,
    HoleKind,
    ASTDiff,
    PartialASTBuilder,
)

from parsing.languages.python import (
    PythonIncrementalParser,
    create_python_parser,
)


def get_parser(language: str) -> IncrementalParser:
    """Get an incremental parser for a language.

    Args:
        language: The language name (e.g., 'python')

    Returns:
        An IncrementalParser instance

    Raises:
        ValueError: If the language is not supported
    """
    parsers = {
        "python": PythonIncrementalParser,
        "py": PythonIncrementalParser,
    }

    language_lower = language.lower()
    if language_lower not in parsers:
        supported = ", ".join(sorted(set(parsers.keys())))
        raise ValueError(
            f"Unsupported language '{language}'. Supported: {supported}"
        )

    return parsers[language_lower]()


__all__ = [
    # Base
    "IncrementalParser",
    "ParseState",
    "ParseResult",
    "ParseError",
    "TokenInfo",
    "HoleDetector",
    "SourceTracker",
    # Partial AST
    "PartialAST",
    "HoleInfo",
    "HoleKind",
    "ASTDiff",
    "PartialASTBuilder",
    # Python
    "PythonIncrementalParser",
    "create_python_parser",
    # Factory
    "get_parser",
]
