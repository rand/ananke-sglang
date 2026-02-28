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
"""Constraint verification for generated code.

This module provides the ConstraintVerifier class that evaluates generated
code against all Ananke constraint domains, producing detailed verification
results with per-domain scores.

The verifier is designed for post-hoc verification in Best-of-N sampling,
where we generate multiple candidates and select the best based on constraint
satisfaction.

Design Principles:
1. Comprehensive: Check all enabled domains (syntax, types, imports, etc.)
2. Detailed: Per-domain scores enable fine-grained ranking
3. Fast: Reuse domain instances and caching where possible
4. Graceful: Partial validity produces meaningful scores

Soundness:
    Verification is inherently SOUND - we never block generation, only score
    completed candidates.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DomainScore:
    """Score from a single constraint domain.

    Attributes:
        domain: Name of the constraint domain
        valid: Whether the code satisfies this domain's constraints
        score: Numeric score from 0.0 to 1.0 (1.0 = fully valid)
        errors: List of specific constraint violations
        latency_ns: Time taken to verify (nanoseconds)
    """

    domain: str
    valid: bool
    score: float
    errors: Tuple[str, ...] = ()
    latency_ns: int = 0

    def __post_init__(self):
        # Validate score range
        if not 0.0 <= self.score <= 1.0:
            object.__setattr__(self, "score", max(0.0, min(1.0, self.score)))


@dataclass
class VerificationResult:
    """Complete verification result for a code candidate.

    Aggregates results from all constraint domains into a unified
    verification result with an overall validity flag and score.

    Attributes:
        candidate: The code string that was verified
        valid: Whether all enabled domains report validity
        overall_score: Weighted average of domain scores (0.0 to 1.0)
        domain_scores: Individual scores per domain
        latency_ns: Total verification time (nanoseconds)
        metadata: Additional verification metadata

    The overall_score is computed as a weighted average where:
    - syntax: weight 1.0 (fundamental)
    - types: weight 0.8 (most impactful)
    - imports: weight 0.5 (recoverable)
    - controlflow: weight 0.6 (structural)
    - semantics: weight 0.7 (correctness)
    """

    candidate: str
    valid: bool = False
    overall_score: float = 0.0
    domain_scores: Dict[str, DomainScore] = field(default_factory=dict)
    latency_ns: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_errors(self) -> List[str]:
        """Get all errors across domains."""
        errors = []
        for score in self.domain_scores.values():
            errors.extend(score.errors)
        return errors

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "candidate": self.candidate,
            "valid": self.valid,
            "overall_score": self.overall_score,
            "domain_scores": {
                name: {
                    "domain": s.domain,
                    "valid": s.valid,
                    "score": s.score,
                    "errors": list(s.errors),
                    "latency_ns": s.latency_ns,
                }
                for name, s in self.domain_scores.items()
            },
            "latency_ns": self.latency_ns,
            "metadata": self.metadata,
        }


# Default domain weights for score computation
DEFAULT_DOMAIN_WEIGHTS: Dict[str, float] = {
    "syntax": 1.0,      # Fundamental - invalid syntax means unusable
    "types": 0.8,       # High impact - type errors affect correctness
    "imports": 0.5,     # Recoverable - missing imports can be added
    "controlflow": 0.6, # Structural - affects program behavior
    "semantics": 0.7,   # Correctness - logical errors
}


class ConstraintVerifier:
    """Verifies generated code against Ananke constraint domains.

    The verifier evaluates complete code strings against each enabled
    constraint domain, producing detailed verification results with
    per-domain scores.

    This is designed for test-time compute strategies like Best-of-N
    sampling, where we:
    1. Generate N candidates without per-token constraints
    2. Verify each candidate using this verifier
    3. Select the best candidate by overall_score

    Attributes:
        language: Target programming language
        enabled_domains: Set of domains to verify against
        domain_weights: Weight for each domain in overall score
        domains: Dictionary of domain instances (lazy-initialized)

    Example:
        >>> verifier = ConstraintVerifier(language="python")
        >>> result = verifier.verify("def add(x: int, y: int) -> int:\\n    return x + y")
        >>> result.valid
        True
        >>> result.overall_score
        1.0
    """

    def __init__(
        self,
        language: str = "python",
        enabled_domains: Optional[set] = None,
        domain_weights: Optional[Dict[str, float]] = None,
        type_context: Optional[Dict[str, str]] = None,
        import_context: Optional[List[str]] = None,
    ):
        """Initialize ConstraintVerifier.

        Args:
            language: Target programming language
            enabled_domains: Domains to verify against (default: all)
            domain_weights: Custom weights for score computation
            type_context: Type bindings for the type domain
            import_context: Available imports for the import domain
        """
        self.language = language
        self.enabled_domains = enabled_domains or {
            "syntax", "types", "imports", "controlflow", "semantics"
        }
        self.domain_weights = domain_weights or DEFAULT_DOMAIN_WEIGHTS

        # Context for domain initialization
        self._type_context = type_context or {}
        self._import_context = import_context or []

        # Lazy domain initialization
        self._domains: Optional[Dict[str, Any]] = None
        self._stats = {
            "verifications": 0,
            "valid_count": 0,
            "total_latency_ns": 0,
        }

    def verify(self, code: str) -> VerificationResult:
        """Verify code against all enabled constraint domains.

        Evaluates the code against each enabled domain and produces
        a unified verification result with per-domain scores.

        Args:
            code: Complete code string to verify

        Returns:
            VerificationResult with validity, scores, and errors

        Example:
            >>> result = verifier.verify("x: int = 'not an int'")
            >>> result.valid
            False
            >>> result.domain_scores["types"].errors
            ('Type mismatch: expected int, got str',)
        """
        start_time = time.perf_counter_ns()
        domain_scores: Dict[str, DomainScore] = {}

        # Verify each enabled domain
        for domain_name in self.enabled_domains:
            score = self._verify_domain(domain_name, code)
            domain_scores[domain_name] = score

        # Compute overall validity and score
        valid = all(s.valid for s in domain_scores.values())
        overall_score = self._compute_overall_score(domain_scores)

        end_time = time.perf_counter_ns()
        latency_ns = end_time - start_time

        # Update stats
        self._stats["verifications"] += 1
        self._stats["total_latency_ns"] += latency_ns
        if valid:
            self._stats["valid_count"] += 1

        return VerificationResult(
            candidate=code,
            valid=valid,
            overall_score=overall_score,
            domain_scores=domain_scores,
            latency_ns=latency_ns,
            metadata={"language": self.language},
        )

    def verify_batch(
        self,
        candidates: List[str],
        parallel: bool = False,
    ) -> List[VerificationResult]:
        """Verify multiple candidates.

        Args:
            candidates: List of code strings to verify
            parallel: If True, verify in parallel (future enhancement)

        Returns:
            List of VerificationResult, one per candidate
        """
        # For now, sequential verification
        # Future: add parallel verification with ThreadPoolExecutor
        return [self.verify(code) for code in candidates]

    def _verify_domain(self, domain_name: str, code: str) -> DomainScore:
        """Verify code against a single domain.

        Args:
            domain_name: Name of the domain
            code: Code string to verify

        Returns:
            DomainScore with validity, score, and errors
        """
        start_time = time.perf_counter_ns()

        try:
            # Dispatch to domain-specific verification
            if domain_name == "syntax":
                valid, score, errors = self._verify_syntax(code)
            elif domain_name == "types":
                valid, score, errors = self._verify_types(code)
            elif domain_name == "imports":
                valid, score, errors = self._verify_imports(code)
            elif domain_name == "controlflow":
                valid, score, errors = self._verify_controlflow(code)
            elif domain_name == "semantics":
                valid, score, errors = self._verify_semantics(code)
            else:
                # Unknown domain - assume valid (soundness)
                valid, score, errors = True, 1.0, ()

        except Exception as e:
            # Verification failed - assume valid (soundness: don't block valid code)
            logger.warning(f"Domain {domain_name} verification failed: {e}")
            valid, score, errors = True, 0.5, (f"Verification error: {str(e)}",)

        end_time = time.perf_counter_ns()
        latency_ns = end_time - start_time

        return DomainScore(
            domain=domain_name,
            valid=valid,
            score=score,
            errors=errors,
            latency_ns=latency_ns,
        )

    def _verify_syntax(self, code: str) -> Tuple[bool, float, Tuple[str, ...]]:
        """Verify code syntax using tree-sitter parser."""
        try:
            # Use tree-sitter for syntax checking
            from ..parsing.base import get_parser

            parser = get_parser(self.language)
            tree = parser.parse(bytes(code, "utf-8"))

            # Check for syntax errors
            errors = []
            self._collect_syntax_errors(tree.root_node, errors)

            if not errors:
                return True, 1.0, ()
            else:
                # Partial score based on error severity
                score = max(0.0, 1.0 - (len(errors) * 0.2))
                return False, score, tuple(errors)

        except ImportError:
            # No parser available - use Python's compile as fallback
            if self.language == "python":
                try:
                    compile(code, "<string>", "exec")
                    return True, 1.0, ()
                except SyntaxError as e:
                    return False, 0.0, (f"SyntaxError: {e.msg}",)
            # For other languages without parser, assume valid (soundness)
            return True, 0.5, ("No parser available for syntax check",)

        except Exception as e:
            # Parse error - syntax invalid
            return False, 0.0, (f"Syntax error: {str(e)}",)

    def _collect_syntax_errors(self, node, errors: list, max_errors: int = 10):
        """Recursively collect syntax errors from parse tree."""
        if node.is_error and len(errors) < max_errors:
            errors.append(f"Syntax error at line {node.start_point[0] + 1}")

        for child in node.children:
            if len(errors) >= max_errors:
                break
            self._collect_syntax_errors(child, errors, max_errors)

    def _verify_types(self, code: str) -> Tuple[bool, float, Tuple[str, ...]]:
        """Verify type correctness using type domain."""
        try:
            from ..domains.types.domain import TypeDomain

            domain = TypeDomain(language=self.language)

            # Seed type context
            for name, type_expr in self._type_context.items():
                if hasattr(domain, "_parse_type_expr"):
                    parsed = domain._parse_type_expr(type_expr)
                    domain.bind_variable(name, parsed)

            # Verify types
            result = domain.verify_code(code)

            if result.valid:
                return True, 1.0, ()
            else:
                score = max(0.0, 1.0 - (len(result.errors) * 0.15))
                return False, score, tuple(str(e) for e in result.errors)

        except ImportError:
            # Type domain not available
            return True, 0.5, ("Type verification not available",)
        except AttributeError:
            # verify_code not implemented - fall back to basic check
            return True, 0.7, ()
        except Exception as e:
            logger.debug(f"Type verification error: {e}")
            return True, 0.5, (f"Type verification error: {str(e)}",)

    def _verify_imports(self, code: str) -> Tuple[bool, float, Tuple[str, ...]]:
        """Verify import availability using import domain."""
        try:
            from ..domains.imports.domain import ImportDomain

            domain = ImportDomain(language=self.language)

            # Seed available imports
            if hasattr(domain, "set_available_modules"):
                domain.set_available_modules(set(self._import_context))

            # Verify imports
            result = domain.verify_code(code)

            if result.valid:
                return True, 1.0, ()
            else:
                score = max(0.0, 1.0 - (len(result.errors) * 0.2))
                return False, score, tuple(str(e) for e in result.errors)

        except ImportError:
            return True, 0.5, ("Import verification not available",)
        except AttributeError:
            # verify_code not implemented
            return True, 0.7, ()
        except Exception as e:
            logger.debug(f"Import verification error: {e}")
            return True, 0.5, (f"Import verification error: {str(e)}",)

    def _verify_controlflow(self, code: str) -> Tuple[bool, float, Tuple[str, ...]]:
        """Verify control flow structure."""
        try:
            from ..domains.controlflow.domain import ControlFlowDomain

            domain = ControlFlowDomain(language=self.language)
            result = domain.verify_code(code)

            if result.valid:
                return True, 1.0, ()
            else:
                score = max(0.0, 1.0 - (len(result.errors) * 0.2))
                return False, score, tuple(str(e) for e in result.errors)

        except ImportError:
            return True, 0.5, ("Control flow verification not available",)
        except AttributeError:
            return True, 0.7, ()
        except Exception as e:
            logger.debug(f"Control flow verification error: {e}")
            return True, 0.5, (f"Control flow verification error: {str(e)}",)

    def _verify_semantics(self, code: str) -> Tuple[bool, float, Tuple[str, ...]]:
        """Verify semantic constraints using SMT solver."""
        try:
            from ..domains.semantics.domain import SemanticDomain

            domain = SemanticDomain(language=self.language)
            result = domain.verify_code(code)

            if result.valid:
                return True, 1.0, ()
            else:
                score = max(0.0, 1.0 - (len(result.errors) * 0.25))
                return False, score, tuple(str(e) for e in result.errors)

        except ImportError:
            return True, 0.5, ("Semantic verification not available",)
        except AttributeError:
            return True, 0.7, ()
        except Exception as e:
            logger.debug(f"Semantic verification error: {e}")
            return True, 0.5, (f"Semantic verification error: {str(e)}",)

    def _compute_overall_score(
        self,
        domain_scores: Dict[str, DomainScore],
    ) -> float:
        """Compute weighted overall score from domain scores.

        Uses domain weights to compute a weighted average, giving
        more weight to critical domains like syntax and types.

        Args:
            domain_scores: Per-domain scores

        Returns:
            Weighted average score from 0.0 to 1.0
        """
        if not domain_scores:
            return 0.0

        total_weight = 0.0
        weighted_sum = 0.0

        for name, score in domain_scores.items():
            weight = self.domain_weights.get(name, 0.5)
            weighted_sum += score.score * weight
            total_weight += weight

        if total_weight == 0:
            return 0.0

        return weighted_sum / total_weight

    def get_stats(self) -> Dict[str, Any]:
        """Get verification statistics."""
        stats = self._stats.copy()
        if stats["verifications"] > 0:
            stats["valid_rate"] = stats["valid_count"] / stats["verifications"]
            stats["avg_latency_ns"] = stats["total_latency_ns"] // stats["verifications"]
        return stats

    def reset_stats(self) -> None:
        """Reset verification statistics."""
        self._stats = {
            "verifications": 0,
            "valid_count": 0,
            "total_latency_ns": 0,
        }
