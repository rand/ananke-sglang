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
"""Soft verification using learned models (PRMs) for code quality scoring.

This module provides a protocol and implementations for soft verification using
Process Reward Models (PRMs) and other learned verifiers. Soft verifiers provide
SCORES for ranking, not hard decisions for blocking.

Design Principles (from Plan Phase 2.2):
1. NEVER block tokens - soft verifiers only provide scores
2. Combine with rule-based verification via weighted scoring
3. Fall back gracefully if model unavailable
4. Support multiple PRM backends (local, API-based)

Soundness:
    Soft verifiers preserve soundness by ONLY scoring candidates, never blocking.
    Even a score of 0.0 doesn't prevent code from being selected if it's the
    best available option.

References:
    - Process Reward Models (OpenAI, 2024)
    - Test-Time Compute Scaling (DeepMind, 2024)
    - Collaborative Verification (Oct 2024): https://arxiv.org/abs/2410.05318
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SoftScore:
    """Score from a soft (learned) verifier.

    Unlike hard verification which produces valid/invalid, soft verification
    produces a continuous score that can be combined with rule-based scores.

    Attributes:
        score: Confidence score from 0.0 to 1.0 (higher = better)
        confidence: How confident the model is in this score (0.0 to 1.0)
        details: Optional breakdown of score components
        model_name: Name of the model that produced this score
        latency_ms: Time taken to compute score (milliseconds)
    """

    score: float
    confidence: float = 1.0
    details: Dict[str, float] = field(default_factory=dict)
    model_name: str = "unknown"
    latency_ms: float = 0.0

    def __post_init__(self):
        # Clamp scores to valid range
        object.__setattr__(self, "score", max(0.0, min(1.0, self.score)))
        object.__setattr__(self, "confidence", max(0.0, min(1.0, self.confidence)))


@dataclass
class GenerationContext:
    """Context for soft verification scoring.

    Provides context about what the code is supposed to do, which helps
    learned verifiers produce more accurate scores.

    Attributes:
        prompt: The original prompt/task description
        expected_signature: Expected function signature if known
        expected_types: Expected type annotations
        test_cases: Example test cases for validation
        language: Programming language
        metadata: Additional context
    """

    prompt: str = ""
    expected_signature: Optional[str] = None
    expected_types: Dict[str, str] = field(default_factory=dict)
    test_cases: List[str] = field(default_factory=list)
    language: str = "python"
    metadata: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class SoftVerifier(Protocol):
    """Protocol for soft (learned) verifiers.

    Soft verifiers provide continuous scores for code quality, not hard
    pass/fail decisions. They can be combined with rule-based verification
    for improved ranking in Best-of-N selection.

    Implementations should:
    1. Return scores in [0.0, 1.0] range
    2. Handle errors gracefully (return neutral score, not crash)
    3. Be reasonably fast (<100ms for typical code)

    Example:
        class MyPRM:
            def score(self, code: str, context: GenerationContext) -> SoftScore:
                # Call PRM API or local model
                return SoftScore(score=0.85, model_name="my-prm")
    """

    def score(self, code: str, context: GenerationContext) -> SoftScore:
        """Score code quality using the soft verifier.

        Args:
            code: The code string to score
            context: Generation context for better scoring

        Returns:
            SoftScore with quality assessment
        """
        ...


class NullSoftVerifier:
    """Null soft verifier that returns neutral scores.

    Used as a default when no PRM is configured. Returns score of 0.5
    with low confidence, ensuring rule-based verification dominates.
    """

    def score(self, code: str, context: GenerationContext) -> SoftScore:
        """Return neutral score."""
        return SoftScore(
            score=0.5,
            confidence=0.1,
            model_name="null",
            details={"reason": "No PRM configured"},
        )


class EnsembleSoftVerifier:
    """Combines multiple soft verifiers via ensemble.

    Aggregates scores from multiple verifiers using weighted averaging.
    This can improve robustness by combining different model strengths.

    Attributes:
        verifiers: List of (verifier, weight) tuples
        aggregation: How to aggregate scores ("mean", "max", "weighted")
    """

    def __init__(
        self,
        verifiers: Optional[List[Tuple[SoftVerifier, float]]] = None,
        aggregation: str = "weighted",
    ):
        """Initialize ensemble verifier.

        Args:
            verifiers: List of (verifier, weight) tuples
            aggregation: Aggregation method
        """
        self.verifiers = verifiers or []
        self.aggregation = aggregation

    def add_verifier(self, verifier: SoftVerifier, weight: float = 1.0) -> None:
        """Add a verifier to the ensemble."""
        self.verifiers.append((verifier, weight))

    def score(self, code: str, context: GenerationContext) -> SoftScore:
        """Score using ensemble of verifiers."""
        import time

        if not self.verifiers:
            return NullSoftVerifier().score(code, context)

        start_time = time.perf_counter()
        scores: List[Tuple[SoftScore, float]] = []
        details: Dict[str, float] = {}

        for verifier, weight in self.verifiers:
            try:
                s = verifier.score(code, context)
                scores.append((s, weight))
                details[s.model_name] = s.score
            except Exception as e:
                logger.warning(f"Soft verifier failed: {e}")
                # Skip failed verifiers, don't crash

        if not scores:
            return NullSoftVerifier().score(code, context)

        # Aggregate scores
        if self.aggregation == "max":
            best = max(scores, key=lambda x: x[0].score)
            final_score = best[0].score
            final_confidence = best[0].confidence
        elif self.aggregation == "mean":
            final_score = sum(s.score for s, _ in scores) / len(scores)
            final_confidence = sum(s.confidence for s, _ in scores) / len(scores)
        else:  # weighted
            total_weight = sum(w * s.confidence for s, w in scores)
            if total_weight > 0:
                final_score = sum(s.score * w * s.confidence for s, w in scores) / total_weight
                final_confidence = total_weight / sum(w for _, w in scores)
            else:
                final_score = 0.5
                final_confidence = 0.1

        latency_ms = (time.perf_counter() - start_time) * 1000

        return SoftScore(
            score=final_score,
            confidence=final_confidence,
            details=details,
            model_name=f"ensemble({len(self.verifiers)})",
            latency_ms=latency_ms,
        )


class HeuristicSoftVerifier:
    """Soft verifier using heuristics (no ML model).

    Provides quick heuristic-based scoring without requiring a neural model.
    Useful as a lightweight fallback or baseline.

    Heuristics include:
    - Code length relative to prompt
    - Presence of docstrings
    - Proper type annotations
    - Error handling patterns
    """

    def __init__(self, language: str = "python"):
        """Initialize heuristic verifier.

        Args:
            language: Target programming language
        """
        self.language = language

    def score(self, code: str, context: GenerationContext) -> SoftScore:
        """Score using heuristics."""
        import time

        start_time = time.perf_counter()
        details: Dict[str, float] = {}

        # Heuristic scores
        scores: List[float] = []

        # 1. Code not empty
        if code.strip():
            scores.append(1.0)
            details["non_empty"] = 1.0
        else:
            scores.append(0.0)
            details["non_empty"] = 0.0

        # 2. Reasonable length (not too short, not too long)
        lines = code.strip().split("\n")
        if 1 <= len(lines) <= 200:
            length_score = 1.0 - abs(len(lines) - 20) / 200
            length_score = max(0.0, length_score)
        else:
            length_score = 0.3
        scores.append(length_score)
        details["length"] = length_score

        # 3. Has docstring (Python)
        if self.language == "python":
            has_docstring = '"""' in code or "'''" in code
            docstring_score = 0.8 if has_docstring else 0.5
            scores.append(docstring_score)
            details["docstring"] = docstring_score

        # 4. Has type annotations (Python)
        if self.language == "python":
            has_types = ": " in code and "->" in code
            type_score = 0.9 if has_types else 0.5
            scores.append(type_score)
            details["types"] = type_score

        # 5. Matches expected signature
        if context.expected_signature and context.expected_signature in code:
            scores.append(1.0)
            details["signature_match"] = 1.0
        elif context.expected_signature:
            scores.append(0.3)
            details["signature_match"] = 0.3

        # Aggregate
        final_score = sum(scores) / len(scores) if scores else 0.5
        latency_ms = (time.perf_counter() - start_time) * 1000

        return SoftScore(
            score=final_score,
            confidence=0.6,  # Heuristics have medium confidence
            details=details,
            model_name="heuristic",
            latency_ms=latency_ms,
        )


@dataclass
class CombinedVerificationResult:
    """Combined result from rule-based and soft verification.

    Attributes:
        rule_based_score: Score from ConstraintVerifier (0.0 to 1.0)
        soft_score: Score from SoftVerifier
        combined_score: Weighted combination of both scores
        rule_weight: Weight given to rule-based score
        soft_weight: Weight given to soft score
    """

    rule_based_score: float
    soft_score: SoftScore
    combined_score: float
    rule_weight: float = 0.7
    soft_weight: float = 0.3

    @staticmethod
    def combine(
        rule_score: float,
        soft_score: SoftScore,
        rule_weight: float = 0.7,
        soft_weight: float = 0.3,
    ) -> "CombinedVerificationResult":
        """Combine rule-based and soft scores.

        The combined score is a weighted average, with soft score weight
        scaled by its confidence.

        Args:
            rule_score: Score from rule-based verification
            soft_score: Score from soft verification
            rule_weight: Weight for rule-based score
            soft_weight: Weight for soft score (scaled by confidence)

        Returns:
            CombinedVerificationResult with combined score
        """
        # Scale soft weight by confidence
        effective_soft_weight = soft_weight * soft_score.confidence

        # Normalize weights
        total_weight = rule_weight + effective_soft_weight
        if total_weight > 0:
            combined = (
                rule_score * rule_weight + soft_score.score * effective_soft_weight
            ) / total_weight
        else:
            combined = rule_score

        return CombinedVerificationResult(
            rule_based_score=rule_score,
            soft_score=soft_score,
            combined_score=combined,
            rule_weight=rule_weight,
            soft_weight=soft_weight,
        )


def create_soft_verifier(
    verifier_type: str = "heuristic",
    language: str = "python",
    **kwargs: Any,
) -> SoftVerifier:
    """Factory function to create soft verifiers.

    Args:
        verifier_type: Type of verifier ("null", "heuristic", "ensemble")
        language: Target programming language
        **kwargs: Additional arguments for specific verifier types

    Returns:
        SoftVerifier instance

    Example:
        >>> verifier = create_soft_verifier("heuristic", language="python")
        >>> score = verifier.score("def f(): pass", GenerationContext())
        >>> score.score
        0.6
    """
    if verifier_type == "null":
        return NullSoftVerifier()
    elif verifier_type == "heuristic":
        return HeuristicSoftVerifier(language=language)
    elif verifier_type == "ensemble":
        ensemble = EnsembleSoftVerifier()
        # Add default heuristic verifier
        ensemble.add_verifier(HeuristicSoftVerifier(language=language), weight=1.0)
        return ensemble
    else:
        logger.warning(f"Unknown verifier type '{verifier_type}', using null")
        return NullSoftVerifier()
