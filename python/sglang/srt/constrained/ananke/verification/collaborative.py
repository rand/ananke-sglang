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
"""Collaborative verification combining CoT reasoning with PoT execution.

This module implements collaborative verification as described in recent
research, combining:
- CoT (Chain of Thought): Reasoning-based verification
- PoT (Program of Thought): Execution-based verification

The key insight is that different verification strategies have different
strengths, and combining them improves overall accuracy.

Design Principles:
1. Combine rule-based, soft (PRM), and execution verification
2. Weighted combination based on strategy confidence
3. Fall back gracefully when strategies unavailable
4. Track strategy-specific scores for debugging

References:
    - Collaborative Verification (Oct 2024): https://arxiv.org/abs/2410.05318
    - Test-Time Compute Scaling (DeepMind, 2024)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional

from .verifier import ConstraintVerifier, VerificationResult
from .soft_verifier import (
    SoftVerifier,
    SoftScore,
    GenerationContext,
    NullSoftVerifier,
    create_soft_verifier,
)
from .execution import (
    ExecutionVerifier,
    ExecutionScore,
    TestCase,
    StaticExecutionVerifier,
    create_execution_verifier,
)

logger = logging.getLogger(__name__)


class VerificationStrategy(Enum):
    """Verification strategy types."""

    RULE_BASED = auto()  # Ananke constraint verification
    SOFT_MODEL = auto()  # PRM/learned verification
    EXECUTION = auto()  # Test execution
    COLLABORATIVE = auto()  # Combined strategies


@dataclass
class StrategyScore:
    """Score from a single verification strategy.

    Attributes:
        strategy: The verification strategy
        score: Score from 0.0 to 1.0
        confidence: Confidence in this score
        weight: Weight for combining with other strategies
        details: Strategy-specific details
    """

    strategy: VerificationStrategy
    score: float
    confidence: float = 1.0
    weight: float = 1.0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CollaborativeResult:
    """Result from collaborative verification.

    Combines scores from multiple verification strategies into
    a unified assessment with per-strategy breakdown.

    Attributes:
        code: The verified code
        overall_score: Combined score (0.0 to 1.0)
        overall_confidence: Combined confidence
        strategy_scores: Per-strategy scores
        recommendation: Recommended action (accept/reject/review)
        latency_ms: Total verification time
    """

    code: str
    overall_score: float = 0.0
    overall_confidence: float = 0.0
    strategy_scores: Dict[VerificationStrategy, StrategyScore] = field(
        default_factory=dict
    )
    recommendation: str = "review"
    latency_ms: float = 0.0

    @property
    def is_acceptable(self) -> bool:
        """Check if code is acceptable (score >= 0.7 with good confidence)."""
        return self.overall_score >= 0.7 and self.overall_confidence >= 0.5

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "code": self.code,
            "overall_score": self.overall_score,
            "overall_confidence": self.overall_confidence,
            "strategy_scores": {
                s.name: {
                    "score": ss.score,
                    "confidence": ss.confidence,
                    "weight": ss.weight,
                    "details": ss.details,
                }
                for s, ss in self.strategy_scores.items()
            },
            "recommendation": self.recommendation,
            "latency_ms": self.latency_ms,
        }


class CollaborativeVerifier:
    """Combines multiple verification strategies for improved accuracy.

    The collaborative verifier integrates:
    1. Rule-based verification (Ananke constraints)
    2. Soft verification (PRMs/heuristics)
    3. Execution-based verification (test cases)

    Each strategy contributes a weighted score, and the final score
    is a confidence-weighted combination.

    Attributes:
        rule_verifier: Ananke constraint verifier
        soft_verifier: PRM/heuristic verifier
        execution_verifier: Test execution verifier
        strategy_weights: Weights for each strategy

    Example:
        >>> verifier = CollaborativeVerifier(language="python")
        >>> result = verifier.verify(
        ...     "def add(a, b): return a + b",
        ...     context=GenerationContext(prompt="Add two numbers"),
        ...     test_cases=[TestCase("add(1, 2)", "3")],
        ... )
        >>> result.is_acceptable
        True
    """

    # Default weights for strategies
    DEFAULT_WEIGHTS = {
        VerificationStrategy.RULE_BASED: 0.5,
        VerificationStrategy.SOFT_MODEL: 0.2,
        VerificationStrategy.EXECUTION: 0.3,
    }

    def __init__(
        self,
        language: str = "python",
        rule_verifier: Optional[ConstraintVerifier] = None,
        soft_verifier: Optional[SoftVerifier] = None,
        execution_verifier: Optional[ExecutionVerifier] = None,
        strategy_weights: Optional[Dict[VerificationStrategy, float]] = None,
        enabled_strategies: Optional[set] = None,
    ):
        """Initialize collaborative verifier.

        Args:
            language: Target programming language
            rule_verifier: Ananke constraint verifier (created if None)
            soft_verifier: PRM/heuristic verifier (created if None)
            execution_verifier: Test execution verifier (created if None)
            strategy_weights: Custom weights for strategies
            enabled_strategies: Which strategies to use
        """
        self.language = language

        # Initialize verifiers
        self.rule_verifier = rule_verifier or ConstraintVerifier(language=language)
        self.soft_verifier = soft_verifier or create_soft_verifier(
            "heuristic", language=language
        )
        self.execution_verifier = execution_verifier or create_execution_verifier(
            "static", language=language
        )

        # Configure weights
        self.strategy_weights = strategy_weights or self.DEFAULT_WEIGHTS.copy()

        # Configure enabled strategies
        self.enabled_strategies = enabled_strategies or {
            VerificationStrategy.RULE_BASED,
            VerificationStrategy.SOFT_MODEL,
            VerificationStrategy.EXECUTION,
        }

    def verify(
        self,
        code: str,
        context: Optional[GenerationContext] = None,
        test_cases: Optional[List[TestCase]] = None,
        function_name: str = "",
    ) -> CollaborativeResult:
        """Verify code using collaborative strategies.

        Runs all enabled verification strategies and combines results
        using confidence-weighted averaging.

        Args:
            code: The code to verify
            context: Generation context for soft verification
            test_cases: Test cases for execution verification
            function_name: Function name for execution testing

        Returns:
            CollaborativeResult with combined assessment
        """
        import time

        start_time = time.perf_counter()
        context = context or GenerationContext(language=self.language)
        test_cases = test_cases or []

        strategy_scores: Dict[VerificationStrategy, StrategyScore] = {}

        # Rule-based verification
        if VerificationStrategy.RULE_BASED in self.enabled_strategies:
            try:
                rule_result = self.rule_verifier.verify(code)
                strategy_scores[VerificationStrategy.RULE_BASED] = StrategyScore(
                    strategy=VerificationStrategy.RULE_BASED,
                    score=rule_result.overall_score,
                    confidence=0.9 if rule_result.valid else 0.7,
                    weight=self.strategy_weights.get(
                        VerificationStrategy.RULE_BASED, 0.5
                    ),
                    details={
                        "valid": rule_result.valid,
                        "domain_scores": {
                            name: score.score
                            for name, score in rule_result.domain_scores.items()
                        },
                    },
                )
            except Exception as e:
                logger.warning(f"Rule-based verification failed: {e}")
                strategy_scores[VerificationStrategy.RULE_BASED] = StrategyScore(
                    strategy=VerificationStrategy.RULE_BASED,
                    score=0.5,
                    confidence=0.1,
                    weight=self.strategy_weights.get(
                        VerificationStrategy.RULE_BASED, 0.5
                    ),
                    details={"error": str(e)},
                )

        # Soft verification
        if VerificationStrategy.SOFT_MODEL in self.enabled_strategies:
            try:
                soft_result = self.soft_verifier.score(code, context)
                strategy_scores[VerificationStrategy.SOFT_MODEL] = StrategyScore(
                    strategy=VerificationStrategy.SOFT_MODEL,
                    score=soft_result.score,
                    confidence=soft_result.confidence,
                    weight=self.strategy_weights.get(
                        VerificationStrategy.SOFT_MODEL, 0.2
                    ),
                    details={
                        "model": soft_result.model_name,
                        "breakdown": soft_result.details,
                    },
                )
            except Exception as e:
                logger.warning(f"Soft verification failed: {e}")
                strategy_scores[VerificationStrategy.SOFT_MODEL] = StrategyScore(
                    strategy=VerificationStrategy.SOFT_MODEL,
                    score=0.5,
                    confidence=0.1,
                    weight=self.strategy_weights.get(
                        VerificationStrategy.SOFT_MODEL, 0.2
                    ),
                    details={"error": str(e)},
                )

        # Execution verification
        if (
            VerificationStrategy.EXECUTION in self.enabled_strategies
            and test_cases
        ):
            try:
                exec_result = self.execution_verifier.verify(
                    code, test_cases, function_name
                )
                strategy_scores[VerificationStrategy.EXECUTION] = StrategyScore(
                    strategy=VerificationStrategy.EXECUTION,
                    score=exec_result.score,
                    confidence=0.95 if exec_result.all_passed else 0.7,
                    weight=self.strategy_weights.get(
                        VerificationStrategy.EXECUTION, 0.3
                    ),
                    details={
                        "passed": exec_result.passed,
                        "failed": exec_result.failed,
                        "errors": exec_result.errors,
                        "total": exec_result.total,
                    },
                )
            except Exception as e:
                logger.warning(f"Execution verification failed: {e}")
                strategy_scores[VerificationStrategy.EXECUTION] = StrategyScore(
                    strategy=VerificationStrategy.EXECUTION,
                    score=0.5,
                    confidence=0.1,
                    weight=self.strategy_weights.get(
                        VerificationStrategy.EXECUTION, 0.3
                    ),
                    details={"error": str(e)},
                )

        # Combine scores
        overall_score, overall_confidence = self._combine_scores(strategy_scores)

        # Generate recommendation
        recommendation = self._generate_recommendation(
            overall_score, overall_confidence, strategy_scores
        )

        latency_ms = (time.perf_counter() - start_time) * 1000

        return CollaborativeResult(
            code=code,
            overall_score=overall_score,
            overall_confidence=overall_confidence,
            strategy_scores=strategy_scores,
            recommendation=recommendation,
            latency_ms=latency_ms,
        )

    def _combine_scores(
        self,
        strategy_scores: Dict[VerificationStrategy, StrategyScore],
    ) -> tuple[float, float]:
        """Combine strategy scores into overall score.

        Uses confidence-weighted averaging:
        overall = sum(score * weight * confidence) / sum(weight * confidence)

        Args:
            strategy_scores: Per-strategy scores

        Returns:
            (overall_score, overall_confidence) tuple
        """
        if not strategy_scores:
            return 0.5, 0.0

        total_weight = 0.0
        weighted_sum = 0.0
        confidence_sum = 0.0

        for ss in strategy_scores.values():
            effective_weight = ss.weight * ss.confidence
            weighted_sum += ss.score * effective_weight
            total_weight += effective_weight
            confidence_sum += ss.confidence

        if total_weight == 0:
            return 0.5, 0.0

        overall_score = weighted_sum / total_weight
        overall_confidence = confidence_sum / len(strategy_scores)

        return overall_score, overall_confidence

    def _generate_recommendation(
        self,
        overall_score: float,
        overall_confidence: float,
        strategy_scores: Dict[VerificationStrategy, StrategyScore],
    ) -> str:
        """Generate recommendation based on verification results.

        Args:
            overall_score: Combined score
            overall_confidence: Combined confidence
            strategy_scores: Per-strategy scores

        Returns:
            Recommendation string: "accept", "reject", or "review"
        """
        # High confidence accept/reject
        if overall_confidence >= 0.7:
            if overall_score >= 0.8:
                return "accept"
            elif overall_score < 0.3:
                return "reject"

        # Check for disagreement between strategies
        if len(strategy_scores) >= 2:
            scores = [ss.score for ss in strategy_scores.values()]
            score_range = max(scores) - min(scores)
            if score_range > 0.4:
                return "review"  # Strategies disagree significantly

        # Medium confidence decisions
        if overall_confidence >= 0.5:
            if overall_score >= 0.6:
                return "accept"
            elif overall_score < 0.4:
                return "reject"

        return "review"

    def verify_batch(
        self,
        candidates: List[str],
        context: Optional[GenerationContext] = None,
        test_cases: Optional[List[TestCase]] = None,
        function_name: str = "",
    ) -> List[CollaborativeResult]:
        """Verify multiple candidates.

        Args:
            candidates: List of code strings
            context: Generation context
            test_cases: Test cases
            function_name: Function name

        Returns:
            List of CollaborativeResult, one per candidate
        """
        return [
            self.verify(code, context, test_cases, function_name)
            for code in candidates
        ]

    def select_best(
        self,
        candidates: List[str],
        context: Optional[GenerationContext] = None,
        test_cases: Optional[List[TestCase]] = None,
        function_name: str = "",
    ) -> tuple[str, CollaborativeResult]:
        """Select the best candidate from a list.

        Args:
            candidates: List of code strings
            context: Generation context
            test_cases: Test cases
            function_name: Function name

        Returns:
            (best_code, result) tuple
        """
        if not candidates:
            raise ValueError("Cannot select from empty list")

        results = self.verify_batch(candidates, context, test_cases, function_name)

        # Select by score, with confidence as tiebreaker
        best_idx = max(
            range(len(results)),
            key=lambda i: (results[i].overall_score, results[i].overall_confidence),
        )

        return candidates[best_idx], results[best_idx]


def collaborative_verify(
    code: str,
    language: str = "python",
    context: Optional[GenerationContext] = None,
    test_cases: Optional[List[TestCase]] = None,
    **kwargs: Any,
) -> CollaborativeResult:
    """Convenience function for collaborative verification.

    Args:
        code: Code to verify
        language: Programming language
        context: Generation context
        test_cases: Test cases
        **kwargs: Additional arguments for CollaborativeVerifier

    Returns:
        CollaborativeResult with combined assessment

    Example:
        >>> result = collaborative_verify(
        ...     "def add(a, b): return a + b",
        ...     test_cases=[TestCase("add(1, 2)", "3")],
        ... )
        >>> result.recommendation
        'accept'
    """
    verifier = CollaborativeVerifier(language=language, **kwargs)
    return verifier.verify(code, context, test_cases)
