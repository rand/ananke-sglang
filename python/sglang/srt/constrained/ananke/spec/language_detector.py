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
"""Language detection for Ananke constrained generation.

This module provides language detection capabilities for auto-detecting the
programming language during generation, supporting polyglot scenarios.

Detection strategies:
1. Tree-sitter based parsing (most accurate)
2. Heuristic-based detection (faster, fallback)
3. Stack-based context for polyglot generation

Example:
    >>> detector = LanguageDetector()
    >>> detector.detect("def foo(): pass")
    'python'
    >>> detector.detect("function foo() { return 1; }")
    'javascript'
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from .constraint_spec import LanguageFrame

logger = logging.getLogger(__name__)


# Language-specific patterns for heuristic detection
LANGUAGE_PATTERNS: Dict[str, List[Tuple[str, float]]] = {
    "python": [
        (r"^\s*def\s+\w+\s*\(", 0.8),
        (r"^\s*class\s+\w+", 0.7),
        (r"^\s*import\s+\w+", 0.6),
        (r"^\s*from\s+\w+\s+import", 0.7),
        (r":\s*$", 0.3),
        (r"^\s*if\s+.*:", 0.5),
        (r"^\s*for\s+\w+\s+in\s+", 0.6),
        (r"^\s*@\w+", 0.5),  # decorators
        (r"self\.", 0.4),
        (r"__\w+__", 0.5),  # dunder methods
    ],
    "typescript": [
        (r"^\s*function\s+\w+", 0.6),
        (r"^\s*const\s+\w+\s*:", 0.8),
        (r"^\s*let\s+\w+\s*:", 0.7),
        (r"^\s*interface\s+\w+", 0.9),
        (r"^\s*type\s+\w+\s*=", 0.9),
        (r":\s*(string|number|boolean|void)", 0.8),
        (r"<\w+>", 0.5),  # generics
        (r"=>\s*\{", 0.6),  # arrow functions
        (r"async\s+function", 0.6),
        (r"await\s+", 0.5),
    ],
    "javascript": [
        (r"^\s*function\s+\w+", 0.6),
        (r"^\s*const\s+\w+\s*=", 0.5),
        (r"^\s*let\s+\w+\s*=", 0.5),
        (r"^\s*var\s+\w+", 0.4),
        (r"=>\s*\{", 0.6),
        (r"async\s+function", 0.6),
        (r"await\s+", 0.5),
        (r"console\.(log|error)", 0.5),
    ],
    "go": [
        (r"^\s*package\s+\w+", 0.9),
        (r"^\s*func\s+\w+", 0.8),
        (r"^\s*import\s*\(", 0.7),
        (r"^\s*type\s+\w+\s+struct", 0.9),
        (r"^\s*type\s+\w+\s+interface", 0.9),
        (r":=", 0.7),
        (r"func\s*\(\w+\s+\*?\w+\)", 0.8),  # method receivers
        (r"go\s+func", 0.7),  # goroutines
        (r"chan\s+\w+", 0.7),
    ],
    "rust": [
        (r"^\s*fn\s+\w+", 0.8),
        (r"^\s*struct\s+\w+", 0.8),
        (r"^\s*impl\s+", 0.9),
        (r"^\s*use\s+", 0.6),
        (r"^\s*mod\s+\w+", 0.7),
        (r"^\s*pub\s+", 0.5),
        (r"let\s+mut\s+", 0.9),
        (r"->\s*\w+", 0.5),
        (r"&\w+", 0.4),
        (r"\w+!", 0.4),  # macros
    ],
    "kotlin": [
        (r"^\s*fun\s+\w+", 0.8),
        (r"^\s*class\s+\w+", 0.6),
        (r"^\s*data\s+class", 0.9),
        (r"^\s*val\s+\w+", 0.7),
        (r"^\s*var\s+\w+", 0.6),
        (r"^\s*suspend\s+fun", 0.9),
        (r"^\s*object\s+\w+", 0.8),
        (r":\s+\w+\?", 0.7),  # nullable types
    ],
    "swift": [
        (r"^\s*func\s+\w+", 0.7),
        (r"^\s*class\s+\w+", 0.6),
        (r"^\s*struct\s+\w+", 0.7),
        (r"^\s*let\s+\w+", 0.6),
        (r"^\s*var\s+\w+", 0.5),
        (r"^\s*protocol\s+\w+", 0.9),
        (r"^\s*extension\s+\w+", 0.9),
        (r"guard\s+let", 0.9),
        (r"->\s*\w+", 0.5),
    ],
    "zig": [
        (r"^\s*fn\s+\w+", 0.7),
        (r"^\s*const\s+\w+\s*=", 0.6),
        (r"^\s*pub\s+fn", 0.9),
        (r"@import", 0.9),
        (r"@This\(\)", 0.9),
        (r"\?\w+", 0.5),  # optionals
        (r"\.\*", 0.5),  # pointer dereference
        (r"comptime", 0.9),
    ],
}

# Keywords unique to specific languages
UNIQUE_KEYWORDS: Dict[str, Set[str]] = {
    "python": {"elif", "nonlocal", "except", "finally", "lambda", "yield"},
    "typescript": {"interface", "declare", "namespace", "readonly", "enum"},
    "javascript": {"prototype", "undefined"},
    "go": {"chan", "defer", "goroutine", "fallthrough", "select"},
    "rust": {"impl", "trait", "crate", "mod", "pub", "mut", "unsafe", "match"},
    "kotlin": {"suspend", "data", "sealed", "companion", "lateinit", "when"},
    "swift": {"guard", "protocol", "extension", "willSet", "didSet", "inout"},
    "zig": {"comptime", "anytype", "unreachable", "noreturn"},
}


@dataclass
class DetectionResult:
    """Result of language detection.

    Attributes:
        language: Detected language identifier
        confidence: Confidence score (0.0 to 1.0)
        scores: Per-language scores
        method: Detection method used
    """

    language: str
    confidence: float
    scores: Dict[str, float] = field(default_factory=dict)
    method: str = "heuristic"


class LanguageDetector:
    """Detects programming language from text content.

    Supports multiple detection strategies:
    1. Tree-sitter based detection (most accurate, requires tree-sitter)
    2. Heuristic-based detection (fast, pattern matching)
    3. Stack-based detection for polyglot contexts

    Attributes:
        supported_languages: Set of languages this detector can identify
        default_language: Language to return when detection fails

    Example:
        >>> detector = LanguageDetector()
        >>> result = detector.detect_with_confidence("def foo(): pass")
        >>> print(f"{result.language}: {result.confidence:.2f}")
        python: 0.85
    """

    SUPPORTED_LANGUAGES: Set[str] = {
        "python",
        "typescript",
        "javascript",
        "go",
        "rust",
        "kotlin",
        "swift",
        "zig",
        "java",
        "c",
        "cpp",
    }

    def __init__(
        self,
        default_language: str = "python",
        use_tree_sitter: bool = True,
    ) -> None:
        """Initialize language detector.

        Args:
            default_language: Language to return when detection fails
            use_tree_sitter: Whether to try tree-sitter based detection
        """
        self._default_language = default_language
        self._use_tree_sitter = use_tree_sitter
        self._parsers: Dict[str, Any] = {}

        if use_tree_sitter:
            self._load_parsers()

    def _load_parsers(self) -> None:
        """Load tree-sitter parsers for supported languages."""
        try:
            import tree_sitter  # type: ignore
        except ImportError:
            logger.debug("tree-sitter not available, using heuristic detection only")
            self._use_tree_sitter = False
            return

        # Try to load language parsers
        # This requires tree-sitter language packages to be installed
        for lang in self.SUPPORTED_LANGUAGES:
            try:
                # Try to import language module
                module_name = f"tree_sitter_{lang}"
                if lang == "typescript":
                    module_name = "tree_sitter_typescript"
                elif lang == "javascript":
                    module_name = "tree_sitter_javascript"
                elif lang == "cpp":
                    module_name = "tree_sitter_cpp"

                import importlib

                lang_module = importlib.import_module(module_name)
                if hasattr(lang_module, "language"):
                    parser = tree_sitter.Parser()
                    parser.language = lang_module.language()
                    self._parsers[lang] = parser
                    logger.debug(f"Loaded tree-sitter parser for {lang}")
            except (ImportError, AttributeError, Exception) as e:
                logger.debug(f"Could not load tree-sitter parser for {lang}: {e}")

    def detect(
        self,
        text: str,
        candidates: Optional[Set[str]] = None,
    ) -> str:
        """Detect language from text content.

        Args:
            text: Text to analyze
            candidates: Optional set of candidate languages to consider

        Returns:
            Detected language identifier
        """
        result = self.detect_with_confidence(text, candidates)
        return result.language

    def detect_with_confidence(
        self,
        text: str,
        candidates: Optional[Set[str]] = None,
    ) -> DetectionResult:
        """Detect language with confidence score.

        Args:
            text: Text to analyze
            candidates: Optional set of candidate languages to consider

        Returns:
            DetectionResult with language, confidence, and scores
        """
        if not text or not text.strip():
            return DetectionResult(
                language=self._default_language,
                confidence=0.0,
                method="default",
            )

        candidates = candidates or self.SUPPORTED_LANGUAGES

        # Try tree-sitter detection first if available
        if self._use_tree_sitter and self._parsers:
            result = self._detect_tree_sitter(text, candidates)
            if result.confidence > 0.7:
                return result

        # Fall back to heuristic detection
        return self._detect_heuristic(text, candidates)

    def _detect_tree_sitter(
        self,
        text: str,
        candidates: Set[str],
    ) -> DetectionResult:
        """Detect language using tree-sitter parsing.

        Args:
            text: Text to analyze
            candidates: Candidate languages

        Returns:
            DetectionResult from tree-sitter analysis
        """
        scores: Dict[str, float] = {}

        for lang in candidates:
            if lang not in self._parsers:
                continue

            parser = self._parsers[lang]
            try:
                tree = parser.parse(text.encode("utf-8"))
                root = tree.root_node

                # Count syntax errors
                error_count = self._count_errors(root)
                node_count = self._count_nodes(root)

                if node_count > 0:
                    # Score based on error rate
                    error_rate = error_count / node_count
                    scores[lang] = max(0.0, 1.0 - error_rate * 2)
            except Exception:
                pass

        if not scores:
            return DetectionResult(
                language=self._default_language,
                confidence=0.0,
                scores=scores,
                method="tree_sitter",
            )

        best_lang = max(scores, key=scores.get)
        return DetectionResult(
            language=best_lang,
            confidence=scores[best_lang],
            scores=scores,
            method="tree_sitter",
        )

    def _count_errors(self, node: Any) -> int:
        """Count syntax error nodes in parse tree."""
        count = 0
        if node.is_error or node.is_missing:
            count += 1
        for child in node.children:
            count += self._count_errors(child)
        return count

    def _count_nodes(self, node: Any) -> int:
        """Count total nodes in parse tree."""
        count = 1
        for child in node.children:
            count += self._count_nodes(child)
        return count

    def _detect_heuristic(
        self,
        text: str,
        candidates: Set[str],
    ) -> DetectionResult:
        """Detect language using heuristic patterns.

        Args:
            text: Text to analyze
            candidates: Candidate languages

        Returns:
            DetectionResult from heuristic analysis
        """
        scores: Dict[str, float] = {}
        lines = text.split("\n")

        for lang in candidates:
            if lang not in LANGUAGE_PATTERNS:
                continue

            score = 0.0
            patterns = LANGUAGE_PATTERNS[lang]

            # Check patterns against each line
            for line in lines[:50]:  # Limit to first 50 lines
                for pattern, weight in patterns:
                    if re.search(pattern, line, re.MULTILINE):
                        score += weight

            # Check for unique keywords
            if lang in UNIQUE_KEYWORDS:
                text_lower = text.lower()
                for keyword in UNIQUE_KEYWORDS[lang]:
                    if re.search(rf"\b{keyword}\b", text_lower):
                        score += 0.5

            scores[lang] = score

        if not scores or max(scores.values()) == 0:
            return DetectionResult(
                language=self._default_language,
                confidence=0.0,
                scores=scores,
                method="heuristic",
            )

        # Normalize scores
        max_score = max(scores.values())
        if max_score > 0:
            scores = {k: v / max_score for k, v in scores.items()}

        best_lang = max(scores, key=scores.get)
        return DetectionResult(
            language=best_lang,
            confidence=min(scores[best_lang], 1.0),
            scores=scores,
            method="heuristic",
        )

    def detect_at_position(
        self,
        text: str,
        position: int,
        language_stack: List[LanguageFrame],
    ) -> str:
        """Detect language at specific position considering stack.

        For polyglot generation, uses the language stack to determine
        the active language at a given position.

        Args:
            text: Full generated text
            position: Character position
            language_stack: Stack of language frames

        Returns:
            Language at the given position
        """
        # Check language stack for position (most recent first)
        for frame in reversed(language_stack):
            if frame.contains(position):
                return frame.language

        # Fall back to detection on text up to position
        if position > 0 and position <= len(text):
            return self.detect(text[:position])

        return self._default_language


class LanguageStackManager:
    """Manages language stack for polyglot generation.

    Tracks language context changes during generation, supporting
    nested language contexts (e.g., Python calling SQL in strings).

    Example:
        >>> manager = LanguageStackManager("python")
        >>> manager.push("sql", 100, delimiter="'''")
        >>> manager.current_language
        'sql'
        >>> manager.pop()
        >>> manager.current_language
        'python'
    """

    def __init__(self, base_language: str = "python") -> None:
        """Initialize stack manager.

        Args:
            base_language: Base language for the stack
        """
        self._stack: List[LanguageFrame] = [
            LanguageFrame(language=base_language, start_position=0)
        ]

    @property
    def stack(self) -> List[LanguageFrame]:
        """Get current language stack."""
        return list(self._stack)

    @property
    def current_language(self) -> str:
        """Get current language."""
        return self._stack[-1].language if self._stack else "python"

    @property
    def current_frame(self) -> Optional[LanguageFrame]:
        """Get current language frame."""
        return self._stack[-1] if self._stack else None

    def push(
        self,
        language: str,
        position: int,
        delimiter: Optional[str] = None,
        end_delimiter: Optional[str] = None,
    ) -> None:
        """Push a new language frame onto the stack.

        Args:
            language: Language identifier
            position: Character position where this context starts
            delimiter: Token that triggered this switch
            end_delimiter: Token that ends this context
        """
        frame = LanguageFrame(
            language=language,
            start_position=position,
            delimiter=delimiter,
            end_delimiter=end_delimiter or delimiter,
        )
        self._stack.append(frame)

    def pop(self) -> Optional[LanguageFrame]:
        """Pop the current language frame.

        Returns:
            Popped frame or None if only base frame remains
        """
        if len(self._stack) > 1:
            return self._stack.pop()
        return None

    def language_at_position(self, position: int) -> str:
        """Get language at a specific position.

        Args:
            position: Character position

        Returns:
            Language at the position
        """
        for frame in reversed(self._stack):
            if frame.contains(position):
                return frame.language
        return self._stack[0].language if self._stack else "python"

    def check_delimiter(
        self,
        text: str,
        position: int,
    ) -> Optional[Tuple[str, str]]:
        """Check if position contains a language-switching delimiter.

        Args:
            text: Full text
            position: Current position

        Returns:
            Tuple of (action, language) if delimiter found, None otherwise
            action is "push" or "pop"
        """
        current_frame = self.current_frame
        if current_frame is None:
            return None

        # Check for end delimiter (pop)
        if current_frame.end_delimiter and len(self._stack) > 1:
            if text[position:].startswith(current_frame.end_delimiter):
                return ("pop", current_frame.language)

        # Check for known embedded language patterns
        remaining = text[position:]

        # Python f-string/template patterns
        if self.current_language == "python":
            if remaining.startswith("'''") or remaining.startswith('"""'):
                # Could be SQL or other embedded language
                return None  # Let caller decide based on context

        return None
