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
"""Constraint intensity levels and task complexity assessment.

This module defines the adaptive constraint intensity system that optimizes
the trade-off between constraint checking overhead and code quality gains.

Problem:
    Fixed ~2.3ms/token constraint overhead hurts simple tasks that would be
    ~90% valid without constraints. Complex tasks (functions, classes, etc.)
    benefit significantly from constraint checking.

Solution:
    Adaptive intensity based on heuristic task complexity assessment.
    Simpler tasks use lighter constraint checking, while complex tasks
    get full multi-domain constraint enforcement.

Performance Targets:
    NONE:       ~0μs (bypass all constraints)
    SYNTAX_ONLY: ~50μs (CFG only via llguidance)
    STANDARD:   ~550μs (syntax + types)
    FULL:       ~2.3ms (all domains)
    EXHAUSTIVE: ~3ms+ (all domains + SMT semantics)

Soundness:
    Lower intensity = more permissive masks = SOUND
    We never block valid tokens by lowering intensity.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, FrozenSet, Optional, Set, Tuple


class ConstraintIntensity(IntEnum):
    """Constraint checking intensity levels.

    Ordered from least to most intensive. Higher values include all
    constraints from lower values plus additional domain checks.

    The intensity directly controls which constraint domains are evaluated
    per token, affecting both latency and code quality assurance.

    Attributes:
        NONE: No constraint checking. Use for trivial completions where
            unconstrained generation is sufficient (e.g., variable names,
            short strings). ~0μs overhead.

        SYNTAX_ONLY: Only CFG-based syntax constraints via llguidance.
            Ensures syntactically valid output without semantic checking.
            Good for simple expressions, literals. ~50μs overhead.

        STANDARD: Syntax + type constraints. The default level for most
            code generation tasks. Catches type errors while maintaining
            reasonable latency. ~550μs overhead.

        FULL: All domains (syntax, types, imports, controlflow).
            Use for complex code structures like functions, classes,
            modules. ~2.3ms overhead.

        EXHAUSTIVE: Full + SMT-based semantic constraints.
            Use for verified code generation where formal guarantees
            are needed. ~3ms+ overhead (Z3 dependent).

    Example:
        >>> intensity = assess_complexity(prompt="def process(x: int):", expected_tokens=50)
        >>> intensity
        ConstraintIntensity.FULL
        >>> domains = domains_for_intensity(intensity)
        >>> domains
        frozenset({'syntax', 'types', 'imports', 'controlflow'})
    """

    NONE = 0
    SYNTAX_ONLY = 1
    STANDARD = 2
    FULL = 3
    EXHAUSTIVE = 4

    def __str__(self) -> str:
        return self.name.lower()

    @classmethod
    def from_string(cls, s: str) -> "ConstraintIntensity":
        """Parse intensity from string (case-insensitive)."""
        normalized = s.strip().upper()
        try:
            return cls[normalized]
        except KeyError:
            # Try numeric
            try:
                return cls(int(s))
            except (ValueError, KeyError):
                return cls.STANDARD  # Default fallback


# Domain sets for each intensity level
_INTENSITY_DOMAINS: Dict[ConstraintIntensity, FrozenSet[str]] = {
    ConstraintIntensity.NONE: frozenset(),
    ConstraintIntensity.SYNTAX_ONLY: frozenset({"syntax"}),
    ConstraintIntensity.STANDARD: frozenset({"syntax", "types"}),
    ConstraintIntensity.FULL: frozenset({"syntax", "types", "imports", "controlflow"}),
    ConstraintIntensity.EXHAUSTIVE: frozenset(
        {"syntax", "types", "imports", "controlflow", "semantics"}
    ),
}


def domains_for_intensity(intensity: ConstraintIntensity) -> FrozenSet[str]:
    """Get the set of domains to enable for a given intensity level.

    Args:
        intensity: The constraint intensity level

    Returns:
        Frozen set of domain names to enable

    Example:
        >>> domains_for_intensity(ConstraintIntensity.STANDARD)
        frozenset({'syntax', 'types'})
    """
    return _INTENSITY_DOMAINS.get(intensity, _INTENSITY_DOMAINS[ConstraintIntensity.STANDARD])


@dataclass(frozen=True)
class IntensityConfig:
    """Configuration for intensity assessment heuristics.

    These thresholds control when to escalate constraint intensity
    based on prompt content and expected generation length.

    Attributes:
        min_tokens_for_types: Minimum expected tokens to enable type checking.
            Below this threshold, syntax-only constraints are used.

        min_tokens_for_full: Minimum expected tokens to enable full domains.
            Longer generations benefit more from comprehensive checking.

        function_keywords: Keywords that indicate function definition.
            Presence triggers FULL intensity.

        class_keywords: Keywords that indicate class definition.
            Presence triggers FULL intensity.

        complex_keywords: Keywords that indicate complex control flow.
            Presence escalates intensity.

        semantic_keywords: Keywords that benefit from SMT checking.
            Presence triggers EXHAUSTIVE intensity.

        high_temp_threshold: Temperature above which to escalate intensity.
            High temperature increases likelihood of invalid generation.

    Example:
        >>> config = IntensityConfig(min_tokens_for_types=30)
        >>> assessor = TaskComplexityAssessor(config)
    """

    # Token thresholds
    min_tokens_for_types: int = 20
    min_tokens_for_full: int = 100

    # Prompt pattern indicators (frozen sets for hashability)
    function_keywords: FrozenSet[str] = frozenset({
        "def ", "async def ", "function ", "fn ", "func ",
        "pub fn ", "private func ", "override fun ",
    })
    class_keywords: FrozenSet[str] = frozenset({
        "class ", "struct ", "interface ", "enum ", "trait ",
        "data class ", "@dataclass", "protocol ",
    })
    complex_keywords: FrozenSet[str] = frozenset({
        "try:", "except", "with ", "match ", "async for",
        "yield ", "await ", "raise ", "finally:",
    })
    semantic_keywords: FrozenSet[str] = frozenset({
        "assert ", "require(", "ensure(", "invariant(",
        "@invariant", "@precondition", "@postcondition",
    })

    # Temperature threshold for escalation
    high_temp_threshold: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "min_tokens_for_types": self.min_tokens_for_types,
            "min_tokens_for_full": self.min_tokens_for_full,
            "function_keywords": list(self.function_keywords),
            "class_keywords": list(self.class_keywords),
            "complex_keywords": list(self.complex_keywords),
            "semantic_keywords": list(self.semantic_keywords),
            "high_temp_threshold": self.high_temp_threshold,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "IntensityConfig":
        """Create from dictionary."""
        return cls(
            min_tokens_for_types=d.get("min_tokens_for_types", 20),
            min_tokens_for_full=d.get("min_tokens_for_full", 100),
            function_keywords=frozenset(d.get("function_keywords", cls.function_keywords)),
            class_keywords=frozenset(d.get("class_keywords", cls.class_keywords)),
            complex_keywords=frozenset(d.get("complex_keywords", cls.complex_keywords)),
            semantic_keywords=frozenset(d.get("semantic_keywords", cls.semantic_keywords)),
            high_temp_threshold=d.get("high_temp_threshold", 1.0),
        )


# Default configuration singleton
DEFAULT_CONFIG = IntensityConfig()


@dataclass
class TaskComplexityAssessor:
    """Assesses task complexity to determine appropriate constraint intensity.

    This class implements heuristic-based task complexity assessment to
    determine the optimal constraint intensity level. The goal is to
    minimize per-token overhead for simple tasks while maintaining
    comprehensive checking for complex code generation.

    The assessment considers:
    1. Prompt content patterns (function/class definitions, control flow)
    2. Expected generation length (longer = more opportunity for errors)
    3. Sampling temperature (higher = more randomness = more errors)
    4. Explicit user/client configuration

    Attributes:
        config: Configuration for assessment heuristics
        stats: Runtime statistics for monitoring

    Example:
        >>> assessor = TaskComplexityAssessor()
        >>> intensity = assessor.assess(
        ...     prompt="def calculate_fibonacci(n: int) -> int:",
        ...     expected_tokens=100,
        ... )
        >>> intensity
        ConstraintIntensity.FULL
    """

    config: IntensityConfig = field(default_factory=lambda: DEFAULT_CONFIG)
    stats: Dict[str, int] = field(default_factory=lambda: {
        "none": 0, "syntax_only": 0, "standard": 0, "full": 0, "exhaustive": 0
    })

    def assess(
        self,
        prompt: str,
        expected_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        explicit_intensity: Optional[ConstraintIntensity] = None,
        language: Optional[str] = None,
    ) -> ConstraintIntensity:
        """Assess task complexity and return appropriate intensity.

        The assessment follows this priority order:
        1. Explicit intensity (if provided) - respect user's choice
        2. Semantic keyword detection - triggers EXHAUSTIVE
        3. Function/class keyword detection - triggers FULL
        4. Complex keyword detection - escalates by 1 level
        5. High temperature - escalates by 1 level
        6. Token count thresholds - base level selection

        Args:
            prompt: The generation prompt text
            expected_tokens: Expected number of tokens to generate.
                If None, uses a default heuristic based on prompt length.
            temperature: Sampling temperature. Higher temperatures
                increase error probability, warranting higher intensity.
            explicit_intensity: If set, overrides heuristic assessment.
                Used when client explicitly requests a specific level.
            language: Target programming language. Some languages may
                warrant different intensity defaults.

        Returns:
            The recommended ConstraintIntensity level

        Example:
            >>> assessor = TaskComplexityAssessor()
            >>> # Simple expression
            >>> assessor.assess("x = ", expected_tokens=5)
            ConstraintIntensity.SYNTAX_ONLY
            >>> # Function definition
            >>> assessor.assess("def process(data):", expected_tokens=50)
            ConstraintIntensity.FULL
        """
        # Honor explicit intensity
        if explicit_intensity is not None:
            self._record_stat(explicit_intensity)
            return explicit_intensity

        # Estimate expected tokens if not provided
        if expected_tokens is None:
            expected_tokens = self._estimate_expected_tokens(prompt)

        # Start with base intensity from token count
        intensity = self._base_intensity_from_tokens(expected_tokens)

        # Check for semantic keywords (triggers EXHAUSTIVE)
        if self._has_semantic_indicators(prompt):
            intensity = max(intensity, ConstraintIntensity.EXHAUSTIVE)

        # Check for function/class definitions (triggers FULL)
        elif self._has_definition_indicators(prompt):
            intensity = max(intensity, ConstraintIntensity.FULL)

        # Check for complex control flow (escalate by 1)
        elif self._has_complexity_indicators(prompt):
            intensity = min(
                ConstraintIntensity(intensity + 1),
                ConstraintIntensity.FULL,
            )

        # High temperature escalation
        if temperature is not None and temperature > self.config.high_temp_threshold:
            intensity = min(
                ConstraintIntensity(intensity + 1),
                ConstraintIntensity.FULL,
            )

        self._record_stat(intensity)
        return intensity

    def _base_intensity_from_tokens(self, expected_tokens: int) -> ConstraintIntensity:
        """Determine base intensity from expected token count."""
        if expected_tokens < self.config.min_tokens_for_types:
            return ConstraintIntensity.SYNTAX_ONLY
        elif expected_tokens < self.config.min_tokens_for_full:
            return ConstraintIntensity.STANDARD
        else:
            return ConstraintIntensity.FULL

    def _has_semantic_indicators(self, prompt: str) -> bool:
        """Check if prompt contains semantic verification keywords."""
        prompt_lower = prompt.lower()
        return any(kw.lower() in prompt_lower for kw in self.config.semantic_keywords)

    def _has_definition_indicators(self, prompt: str) -> bool:
        """Check if prompt contains function/class definition keywords."""
        return (
            any(kw in prompt for kw in self.config.function_keywords)
            or any(kw in prompt for kw in self.config.class_keywords)
        )

    def _has_complexity_indicators(self, prompt: str) -> bool:
        """Check if prompt contains complex control flow keywords."""
        return any(kw in prompt for kw in self.config.complex_keywords)

    def _estimate_expected_tokens(self, prompt: str) -> int:
        """Estimate expected tokens based on prompt characteristics.

        Heuristic: longer prompts with code structure tend to require
        longer completions. This is a rough estimate when max_tokens
        is not specified.
        """
        # Count structural indicators
        has_function = self._has_definition_indicators(prompt)
        has_complexity = self._has_complexity_indicators(prompt)

        # Base estimate on prompt length
        base = min(len(prompt) // 10, 50)

        # Escalate for structural code
        if has_function:
            return max(base, 100)
        elif has_complexity:
            return max(base, 50)
        return max(base, 20)

    def _record_stat(self, intensity: ConstraintIntensity) -> None:
        """Record intensity selection for monitoring."""
        key = str(intensity)
        self.stats[key] = self.stats.get(key, 0) + 1

    def get_stats(self) -> Dict[str, int]:
        """Get assessment statistics."""
        return self.stats.copy()

    def reset_stats(self) -> None:
        """Reset assessment statistics."""
        self.stats = {k: 0 for k in self.stats}


# Module-level convenience function
_DEFAULT_ASSESSOR: Optional[TaskComplexityAssessor] = None


def assess_complexity(
    prompt: str,
    expected_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    explicit_intensity: Optional[ConstraintIntensity] = None,
    language: Optional[str] = None,
) -> ConstraintIntensity:
    """Convenience function to assess task complexity.

    Uses a module-level singleton assessor for stateless calls.

    Args:
        prompt: The generation prompt text
        expected_tokens: Expected number of tokens to generate
        temperature: Sampling temperature
        explicit_intensity: Explicit intensity override
        language: Target programming language

    Returns:
        The recommended ConstraintIntensity level

    Example:
        >>> from sglang.srt.constrained.ananke.adaptive import assess_complexity
        >>> intensity = assess_complexity("def foo():", expected_tokens=50)
        >>> intensity
        ConstraintIntensity.FULL
    """
    global _DEFAULT_ASSESSOR
    if _DEFAULT_ASSESSOR is None:
        _DEFAULT_ASSESSOR = TaskComplexityAssessor()
    return _DEFAULT_ASSESSOR.assess(
        prompt=prompt,
        expected_tokens=expected_tokens,
        temperature=temperature,
        explicit_intensity=explicit_intensity,
        language=language,
    )
