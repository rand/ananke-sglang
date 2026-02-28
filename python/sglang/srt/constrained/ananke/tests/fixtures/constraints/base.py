# Copyright Rand Arete @ Ananke 2025
# Licensed under the Apache License, Version 2.0
"""Base types for constraint examples.

This module provides the ConstraintExample dataclass used to define
realistic constraint examples for evaluation, documentation, and validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ConstraintComplexity(str, Enum):
    """Complexity classification for constraint examples.

    - SYNTACTIC: Pure token/pattern matching (imports, naming conventions)
    - STRUCTURAL: Multi-line patterns requiring structure (controlflow, syntax)
    - SEMANTIC: Requires understanding beyond syntax (comptime, coroutines, types with inference)
    """
    SYNTACTIC = "syntactic"
    STRUCTURAL = "structural"
    SEMANTIC = "semantic"


# Default complexity by domain (can be overridden per-example)
DOMAIN_COMPLEXITY_MAP = {
    "imports": ConstraintComplexity.SYNTACTIC,
    "syntax": ConstraintComplexity.STRUCTURAL,
    "types": ConstraintComplexity.STRUCTURAL,
    "controlflow": ConstraintComplexity.STRUCTURAL,
    "semantics": ConstraintComplexity.SEMANTIC,
    "comptime": ConstraintComplexity.SEMANTIC,
    "coroutines": ConstraintComplexity.SEMANTIC,
    "actors": ConstraintComplexity.SEMANTIC,
    "ownership": ConstraintComplexity.SEMANTIC,
}

# Default max_tokens by language
# Languages with verbose syntax get higher limits to avoid truncation
LANGUAGE_DEFAULT_TOKENS = {
    "python": 512,
    "rust": 1024,
    "go": 768,
    "typescript": 1024,
    "kotlin": 768,
    "swift": 768,
    "zig": 1536,  # Most verbose (comptime, explicit error handling)
}
DEFAULT_MAX_TOKENS = 1024  # Fallback for unknown languages

# Support both relative imports (when used as subpackage) and absolute imports
try:
    from ....spec.constraint_spec import ConstraintSpec
except ImportError:
    from spec.constraint_spec import ConstraintSpec


@dataclass
class ConstraintExample:
    """A single constraint example with metadata for testing and documentation.

    Each example represents a realistic developer workflow scenario with
    a complete ConstraintSpec and expected masking behavior.

    Attributes:
        id: Unique identifier (e.g., "py-types-001")
        name: Human-readable name (e.g., "Generic Container Inference")
        description: Full description of what this example demonstrates
        scenario: What the developer is doing (real-world context)
        prompt: The actual prompt to send to the model for code generation.
            Should be specific and guide the model toward the expected pattern.
            Example: "Complete the function: def filter_items(items: List[T]) -> List[T]:"
        spec: The actual ConstraintSpec to be used
        expected_effect: Description of what tokens should be masked
        valid_outputs: List of outputs that should pass through the constraint
        invalid_outputs: List of outputs that should be blocked
        tags: Tags for filtering (e.g., ["types", "generics", "inference"])
        language: Target language (for indexing)
        domain: Target domain (for indexing)

    Example:
        >>> example = ConstraintExample(
        ...     id="py-types-001",
        ...     name="Generic Filter",
        ...     description="Filter list while preserving type parameter",
        ...     scenario="Developer writing a generic filter function",
        ...     prompt="Complete this Python function that filters a list:\\ndef filter_items(items: List[T], pred: Callable[[T], bool]) -> List[T]:\\n    ",
        ...     spec=ConstraintSpec(
        ...         language="python",
        ...         expected_type="List[T]",
        ...     ),
        ...     expected_effect="Masks tokens producing non-List types",
        ...     valid_outputs=["return [x for x in items if pred(x)]"],
        ...     invalid_outputs=["return {x: x for x in items}"],
        ...     tags=["types", "generics"],
        ...     language="python",
        ...     domain="types",
        ... )
    """

    id: str
    name: str
    description: str
    scenario: str
    spec: ConstraintSpec
    expected_effect: str
    valid_outputs: List[str] = field(default_factory=list)
    invalid_outputs: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    language: str = ""
    domain: str = ""
    complexity: Optional[ConstraintComplexity] = None  # Auto-derived from domain if not set
    prompt: Optional[str] = None  # If None, falls back to scenario (for backwards compat)
    max_tokens: Optional[int] = None  # Per-example token limit; uses language default if None

    def __post_init__(self):
        """Set default complexity based on domain if not specified."""
        if self.complexity is None and self.domain:
            self.complexity = DOMAIN_COMPLEXITY_MAP.get(
                self.domain, ConstraintComplexity.STRUCTURAL
            )

    @property
    def complexity_str(self) -> str:
        """Get complexity as string for reporting."""
        if self.complexity:
            return self.complexity.value
        return DOMAIN_COMPLEXITY_MAP.get(self.domain, ConstraintComplexity.STRUCTURAL).value

    def get_prompt(self) -> str:
        """Get the prompt for this example.

        Returns prompt if set, otherwise falls back to scenario for backwards compat.
        """
        return self.prompt if self.prompt else self.scenario

    def get_effective_max_tokens(self) -> int:
        """Get the effective max_tokens for this example.

        Returns:
            max_tokens if explicitly set, else language-specific default, else global default.
        """
        if self.max_tokens is not None:
            return self.max_tokens
        return LANGUAGE_DEFAULT_TOKENS.get(self.language, DEFAULT_MAX_TOKENS)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "scenario": self.scenario,
            "spec": self.spec.to_dict(),
            "expected_effect": self.expected_effect,
            "valid_outputs": self.valid_outputs,
            "invalid_outputs": self.invalid_outputs,
            "tags": self.tags,
            "language": self.language,
            "domain": self.domain,
            "complexity": self.complexity_str,
        }
        if self.prompt:
            result["prompt"] = self.prompt
        if self.max_tokens is not None:
            result["max_tokens"] = self.max_tokens
        return result

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ConstraintExample":
        """Create from dictionary."""
        complexity = None
        if "complexity" in d:
            complexity = ConstraintComplexity(d["complexity"])
        return cls(
            id=d["id"],
            name=d["name"],
            description=d["description"],
            scenario=d["scenario"],
            spec=ConstraintSpec.from_dict(d["spec"]),
            expected_effect=d["expected_effect"],
            valid_outputs=d.get("valid_outputs", []),
            invalid_outputs=d.get("invalid_outputs", []),
            tags=d.get("tags", []),
            language=d.get("language", ""),
            domain=d.get("domain", ""),
            complexity=complexity,
            prompt=d.get("prompt"),
            max_tokens=d.get("max_tokens"),
        )


@dataclass
class ExampleCatalog:
    """Collection of constraint examples with metadata.

    Provides indexing and filtering capabilities for the example collection.
    """

    version: str = "1.0"
    examples: List[ConstraintExample] = field(default_factory=list)

    def by_language(self, language: str) -> List[ConstraintExample]:
        """Get examples for a specific language."""
        return [e for e in self.examples if e.language == language]

    def by_domain(self, domain: str) -> List[ConstraintExample]:
        """Get examples for a specific domain."""
        return [e for e in self.examples if e.domain == domain]

    def by_tag(self, tag: str) -> List[ConstraintExample]:
        """Get examples with a specific tag."""
        return [e for e in self.examples if tag in e.tags]

    def by_complexity(self, complexity: ConstraintComplexity) -> List[ConstraintExample]:
        """Get examples with a specific complexity level."""
        return [e for e in self.examples if e.complexity == complexity]

    def by_id(self, example_id: str) -> Optional[ConstraintExample]:
        """Get example by ID."""
        for e in self.examples:
            if e.id == example_id:
                return e
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "version": self.version,
            "total_examples": len(self.examples),
            "by_language": self._count_by_language(),
            "by_domain": self._count_by_domain(),
            "examples": [e.to_dict() for e in self.examples],
        }

    def _count_by_language(self) -> Dict[str, int]:
        """Count examples by language."""
        counts: Dict[str, int] = {}
        for e in self.examples:
            counts[e.language] = counts.get(e.language, 0) + 1
        return counts

    def _count_by_domain(self) -> Dict[str, int]:
        """Count examples by domain."""
        counts: Dict[str, int] = {}
        for e in self.examples:
            counts[e.domain] = counts.get(e.domain, 0) + 1
        return counts


# Language constants
LANGUAGES = ["python", "rust", "zig", "typescript", "go", "kotlin", "swift"]

# Domain constants
DOMAINS = ["types", "imports", "controlflow", "semantics", "syntax"]
