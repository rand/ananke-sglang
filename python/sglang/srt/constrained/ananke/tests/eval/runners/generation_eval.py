# Copyright Rand Arete @ Ananke 2025
# Licensed under the Apache License, Version 2.0
"""Generation eval runner for constraint examples.

This runner evaluates Ananke-constrained generation using the 136+ validated
constraint examples from tests/fixtures/constraints/.

Usage:
    from tests.eval.runners.generation_eval import GenerationEvalRunner

    runner = GenerationEvalRunner(generate_fn)
    results = runner.run_all()
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterator, List, Optional, Any, Tuple


@dataclass
class RegexDifficulty:
    """Analysis of regex pattern difficulty."""

    pattern_length: int
    has_multiline: bool  # Contains [\s\S]* or similar
    alternation_count: int  # Number of | operators
    lookahead_count: int  # Number of (?= or (?! patterns
    lookbehind_count: int  # Number of (?<= or (?<! patterns
    is_anchored: bool  # Starts with ^
    has_backrefs: bool  # Contains \1, \2, etc.
    quantifier_count: int  # Number of *, +, ?, {n,m}
    char_class_count: int  # Number of [...] patterns
    difficulty_score: float  # 0-1 normalized score

    @classmethod
    def analyze(cls, pattern: str) -> "RegexDifficulty":
        """Analyze a regex pattern for difficulty."""
        if not pattern:
            return cls(0, False, 0, 0, 0, False, False, 0, 0, 0.0)

        # Count various regex features
        alternations = pattern.count("|")
        lookaheads = len(re.findall(r"\(\?[=!]", pattern))
        lookbehinds = len(re.findall(r"\(\?<[=!]", pattern))
        has_multiline = bool(re.search(r"\[\^?\]?[^\]]*\\s\\S[^\]]*\]|\(\?s\)", pattern))
        is_anchored = pattern.startswith("^")
        has_backrefs = bool(re.search(r"\\[1-9]", pattern))
        quantifiers = len(re.findall(r"[*+?]|\{\d+,?\d*\}", pattern))
        char_classes = len(re.findall(r"\[[^\]]+\]", pattern))

        # Calculate difficulty score (0-1)
        # Higher score = harder to match
        score = 0.0
        score += min(len(pattern) / 200, 0.3)  # Length contributes up to 0.3
        score += alternations * 0.05  # Each alternation adds 0.05
        score += (lookaheads + lookbehinds) * 0.1  # Lookarounds are hard
        score += 0.1 if has_multiline else 0  # Multiline adds complexity
        score += 0.1 if has_backrefs else 0  # Backrefs add complexity
        score -= 0.1 if is_anchored else 0  # Anchoring helps
        score = max(0.0, min(1.0, score))  # Clamp to [0, 1]

        return cls(
            pattern_length=len(pattern),
            has_multiline=has_multiline,
            alternation_count=alternations,
            lookahead_count=lookaheads,
            lookbehind_count=lookbehinds,
            is_anchored=is_anchored,
            has_backrefs=has_backrefs,
            quantifier_count=quantifiers,
            char_class_count=char_classes,
            difficulty_score=round(score, 3),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pattern_length": self.pattern_length,
            "has_multiline": self.has_multiline,
            "alternation_count": self.alternation_count,
            "lookahead_count": self.lookahead_count,
            "lookbehind_count": self.lookbehind_count,
            "is_anchored": self.is_anchored,
            "has_backrefs": self.has_backrefs,
            "quantifier_count": self.quantifier_count,
            "char_class_count": self.char_class_count,
            "difficulty_score": self.difficulty_score,
        }

from ..config import EvalConfig, get_language_config, get_domain_config
from ..judges import HAS_EBNF_SUPPORT
from ..judges.regex_judge import RegexJudge
from ..metrics import EvalMetrics, EvalResult, SatisfactionLevel


@dataclass
class ConstraintWarning:
    """Warning about potential constraint issues."""

    example_id: str
    warning_type: str
    message: str
    severity: str  # "info" | "warning" | "error"


def validate_constraint_pattern(example: "ConstraintExample") -> List[ConstraintWarning]:
    r"""Validate a constraint pattern and return warnings.

    Checks for common issues:
    - Unanchored patterns (may match unexpectedly)
    - Overly permissive patterns ([\s\S]*)
    - Very long patterns (hard to match)
    - Missing negative constraints for blacklist scenarios

    Args:
        example: The constraint example to validate

    Returns:
        List of warnings
    """
    warnings: List[ConstraintWarning] = []
    regex = example.spec.regex

    if not regex:
        return warnings

    # Check for unanchored pattern
    if not regex.startswith("^"):
        warnings.append(ConstraintWarning(
            example_id=example.id,
            warning_type="unanchored",
            message="Pattern not anchored with ^ - may match unexpected prefixes",
            severity="info",
        ))

    # Check for overly permissive patterns
    if r"[\s\S]*" in regex or r".*" in regex:
        warnings.append(ConstraintWarning(
            example_id=example.id,
            warning_type="permissive",
            message="Pattern contains [\\s\\S]* or .* - may be too permissive",
            severity="warning",
        ))

    # Check for very long patterns
    if len(regex) > 150:
        warnings.append(ConstraintWarning(
            example_id=example.id,
            warning_type="complex",
            message=f"Pattern is very long ({len(regex)} chars) - may be hard to match",
            severity="warning",
        ))

    # Check complexity expectations
    domain_config = get_domain_config(example.domain)
    if domain_config.get("expected_pass_rate", 1.0) < 0.3:
        warnings.append(ConstraintWarning(
            example_id=example.id,
            warning_type="low_expected_rate",
            message=f"Domain '{example.domain}' has low expected pass rate ({domain_config.get('expected_pass_rate')})",
            severity="info",
        ))

    return warnings

if HAS_EBNF_SUPPORT:
    from ..judges.ebnf_judge import EbnfJudge

try:
    from tests.fixtures.constraints import get_all_examples
    from tests.fixtures.constraints.base import ConstraintExample
except ImportError:
    from ...fixtures.constraints import get_all_examples
    from ...fixtures.constraints.base import ConstraintExample


@dataclass
class GenerationResult:
    """Result of a single generation eval."""

    example_id: str
    language: str
    domain: str
    complexity: str  # "syntactic" | "structural" | "semantic"

    # Generation details
    prompt: str
    constraint_spec: Dict[str, Any]
    generated_output: str
    latency_ms: float

    # Validation
    matches_constraint: bool
    constraint_type: str  # "regex" | "ebnf"
    validation_error: Optional[str] = None

    # Baseline comparison (unconstrained generation)
    baseline_output: Optional[str] = None
    baseline_matches: Optional[bool] = None
    baseline_latency_ms: Optional[float] = None

    # Regex analysis
    regex_difficulty: Optional[RegexDifficulty] = None

    # Categorization
    tags: List[str] = field(default_factory=list)

    @property
    def constraint_helped(self) -> Optional[bool]:
        """Did the constraint help produce valid output?"""
        if self.baseline_matches is None:
            return None
        return self.matches_constraint and not self.baseline_matches

    @property
    def constraint_hurt(self) -> Optional[bool]:
        """Did the constraint prevent valid output?"""
        if self.baseline_matches is None:
            return None
        return not self.matches_constraint and self.baseline_matches


class FailureCategory:
    """Failure category constants with descriptions."""

    # Output issues
    TRUNCATED = "truncated"  # Output exceeded max_tokens
    EMPTY = "empty_output"  # No output produced
    ERROR = "generation_error"  # Backend/generation error

    # Model deviation subcategories
    PARTIAL_MATCH = "partial_match"  # Starts correct, diverges
    WRONG_STRUCTURE = "wrong_structure"  # Valid code, wrong pattern
    GARBAGE = "garbage"  # Unicode artifacts, repetition, nonsense
    LANGUAGE_CONFUSION = "language_confusion"  # Wrong language syntax

    # Constraint issues
    VALIDATION_ISSUE = "validation_issue"  # Output correct but validation failed
    CONSTRAINT_TOO_TIGHT = "constraint_too_tight"  # Regex too specific

    # Unknown
    UNKNOWN = "unknown"


@dataclass
class FailureAnalysis:
    """Analysis of a generation failure."""

    example_id: str
    language: str
    domain: str
    complexity: str
    constraint_type: str

    expected_pattern: str
    actual_output: str

    failure_category: str
    details: str

    # Additional analysis
    output_starts_valid: bool = False  # First tokens match pattern
    has_unicode_artifacts: bool = False  # Contains unusual unicode
    wrong_language_detected: Optional[str] = None  # If different language detected


class GenerationEvalRunner:
    """Runner for generation eval using constraint examples.

    This runner:
    1. Takes each constraint example
    2. Generates output using the constraint spec
    3. Validates output matches the constraint
    4. Categorizes failures for error analysis
    """

    def __init__(
        self,
        generate_fn: Callable[[str, Dict[str, Any], int], str],
        config: Optional[EvalConfig] = None,
        max_tokens: int = 1024,
        temperature: float = 0.3,
        deterministic: bool = False,
        seed: int = 42,
    ):
        """Initialize the runner.

        Args:
            generate_fn: Function to generate text. Signature:
                generate_fn(prompt, constraint_spec, max_tokens) -> str
            config: Eval configuration for filtering examples
            max_tokens: Global max tokens fallback (override with per-example limits)
            temperature: Sampling temperature
            deterministic: If True, use temperature=0.0 and seed for reproducibility
            seed: Random seed for deterministic sampling
        """
        self.generate_fn = generate_fn
        self.config = config or EvalConfig.tier1_syntax()
        self.max_tokens = max_tokens
        self.temperature = 0.0 if deterministic else temperature
        self.deterministic = deterministic
        self.seed = seed

        self.regex_judge = RegexJudge()
        self.ebnf_judge = EbnfJudge() if HAS_EBNF_SUPPORT else None

    def get_examples(self) -> Iterator[ConstraintExample]:
        """Get examples matching config filters."""
        examples = get_all_examples()

        for example in examples:
            # Filter by language
            if self.config.languages and example.language not in self.config.languages:
                continue

            # Filter by domain
            if self.config.domains and example.domain not in self.config.domains:
                continue

            # Must have syntax constraint
            if not example.spec.regex and not example.spec.ebnf:
                continue

            # Skip EBNF-only if no support
            if example.spec.ebnf and not example.spec.regex:
                if not HAS_EBNF_SUPPORT:
                    continue

            yield example

    def _build_constraint_spec(self, example: ConstraintExample) -> Dict[str, Any]:
        """Build constraint spec dict from example."""
        spec = {"language": example.language}

        if example.spec.regex:
            spec["regex"] = example.spec.regex
        if example.spec.ebnf:
            spec["ebnf"] = example.spec.ebnf
        if example.spec.json_schema:
            spec["json_schema"] = example.spec.json_schema
        if example.spec.type_bindings:
            spec["type_bindings"] = [
                {"name": tb.name, "type_expr": tb.type_expr, "scope": tb.scope}
                for tb in example.spec.type_bindings
            ]
        if example.spec.imports:
            spec["imports"] = [
                {"module": ib.module, "name": ib.name}
                for ib in example.spec.imports
            ]
        if example.spec.forbidden_imports:
            spec["forbidden_imports"] = list(example.spec.forbidden_imports)
        if example.spec.available_modules:
            spec["available_modules"] = list(example.spec.available_modules)
        if example.spec.expected_type:
            spec["expected_type"] = example.spec.expected_type
        if example.spec.negative_regex:
            spec["negative_regex"] = example.spec.negative_regex

        # Pass relaxation config - critical for avoiding garbage output
        # on strict constraints like kt-sem-003
        spec["allow_relaxation"] = example.spec.allow_relaxation
        spec["relaxation_threshold"] = example.spec.relaxation_threshold

        # Pass early termination config - critical for avoiding truncation
        # on verbose examples like zig-comptime-002
        spec["enable_early_termination"] = example.spec.enable_early_termination

        return spec

    def _validate_output(
        self,
        output: str,
        example: ConstraintExample
    ) -> tuple[bool, str, Optional[str]]:
        """Validate output against constraint.

        Returns:
            (matches, constraint_type, error_message)
        """
        output_stripped = output.strip()

        # Check negative constraint first (must NOT match)
        if example.spec.negative_regex:
            try:
                if re.search(example.spec.negative_regex, output_stripped):
                    return False, "negative_regex", "Output matches forbidden pattern"
            except Exception as e:
                return False, "negative_regex", f"Invalid negative regex: {e}"

        # Prefer regex for validation (faster)
        if example.spec.regex:
            try:
                matched = self.regex_judge.matches(output_stripped, example.spec.regex)
                return matched, "regex", None
            except Exception as e:
                return False, "regex", str(e)

        # Fall back to EBNF
        if example.spec.ebnf and self.ebnf_judge:
            try:
                matched = self.ebnf_judge.matches(output_stripped, example.spec.ebnf)
                return matched, "ebnf", None
            except Exception as e:
                return False, "ebnf", str(e)

        return False, "none", "No constraint to validate against"

    def _detect_language(self, output: str) -> Optional[str]:
        """Detect programming language from output."""
        # Simple heuristics for language detection
        markers = {
            "python": (r"\bdef\s+\w+\s*\(|import\s+\w+|from\s+\w+\s+import", "python"),
            "rust": (r"\bfn\s+\w+|let\s+mut\s+|impl\s+\w+|use\s+\w+::", "rust"),
            "go": (r"\bfunc\s+\w+|package\s+\w+|import\s+\"", "go"),
            "typescript": (r":\s*(?:string|number|boolean)\b|interface\s+\w+|type\s+\w+\s*=", "typescript"),
            "kotlin": (r"\bfun\s+\w+|val\s+\w+\s*:|var\s+\w+\s*:", "kotlin"),
            "swift": (r"\bfunc\s+\w+|let\s+\w+\s*:|var\s+\w+\s*:", "swift"),
            "zig": (r"\bfn\s+\w+|const\s+\w+\s*=|@import\(", "zig"),
        }
        for lang, (pattern, name) in markers.items():
            if re.search(pattern, output):
                return name
        return None

    def _has_unicode_artifacts(self, output: str) -> bool:
        """Check for unusual unicode characters suggesting constraint issues."""
        # Check for zero-width or unusual space characters
        unusual = re.search(r'[\u200b-\u200f\u2028-\u202f\u205f-\u206f\ufeff]', output)
        # Check for excessive whitespace repetition
        excessive_space = re.search(r'\s{20,}', output)
        return bool(unusual or excessive_space)

    def _check_partial_match(self, output: str, pattern: str) -> bool:
        """Check if output starts with a valid prefix of the pattern."""
        if not pattern:
            return False
        # Try the full pattern against progressively shorter prefixes of the output
        for length in [50, 30, 20, 10]:
            prefix = output[:length] if len(output) > length else output
            if prefix.strip():
                try:
                    if re.search(pattern, prefix):
                        return True
                except re.error:
                    # Pattern is invalid, skip
                    return False
        return False

    def _categorize_failure(
        self,
        output: str,
        example: ConstraintExample,
        constraint_type: str,
        max_tokens: Optional[int] = None,
    ) -> FailureAnalysis:
        """Categorize why the generation failed with improved granularity."""

        pattern = example.spec.regex or example.spec.ebnf or ""
        has_unicode = self._has_unicode_artifacts(output)
        detected_lang = self._detect_language(output)
        starts_valid = self._check_partial_match(output, pattern)

        # Use passed max_tokens or fall back to example's effective limit
        effective_limit = max_tokens or example.get_effective_max_tokens()

        # Check for truncation (~4 chars per token average)
        if len(output) >= effective_limit * 4:
            category = FailureCategory.TRUNCATED
            details = f"Output length ({len(output)}) suggests truncation (limit={effective_limit})"

        # Check if output is empty
        elif not output.strip():
            category = FailureCategory.EMPTY
            details = "Model produced empty output"

        # Check for unicode artifacts (constraint too tight)
        elif has_unicode:
            category = FailureCategory.GARBAGE
            details = "Output contains unicode artifacts suggesting constraint mask was too restrictive"

        # Check if output contains error messages
        elif "error" in output.lower()[:200] or "exception" in output.lower()[:200]:
            category = FailureCategory.ERROR
            details = f"Output contains error text: {output[:100]}"

        # Check for wrong language
        elif detected_lang and detected_lang != example.language:
            category = FailureCategory.LANGUAGE_CONFUSION
            details = f"Expected {example.language}, detected {detected_lang}"

        # Check if output matches valid_outputs (validation issue)
        elif example.valid_outputs and output.strip() in [v.strip() for v in example.valid_outputs]:
            category = FailureCategory.VALIDATION_ISSUE
            details = "Output matches valid_outputs but failed validation"

        # Check for partial match (starts valid, diverges)
        elif starts_valid:
            category = FailureCategory.PARTIAL_MATCH
            details = "Output starts correctly but diverges from pattern"

        # Check if output looks like valid code but wrong pattern
        elif detected_lang == example.language:
            category = FailureCategory.WRONG_STRUCTURE
            details = "Valid code structure but doesn't match constraint pattern"

        else:
            category = FailureCategory.UNKNOWN
            details = "Could not categorize failure"

        return FailureAnalysis(
            example_id=example.id,
            language=example.language,
            domain=example.domain,
            complexity=example.complexity_str,
            constraint_type=constraint_type,
            expected_pattern=pattern[:200],
            actual_output=output[:500],
            failure_category=category,
            details=details,
            output_starts_valid=starts_valid,
            has_unicode_artifacts=has_unicode,
            wrong_language_detected=detected_lang if detected_lang != example.language else None,
        )

    def evaluate_example(
        self,
        example: ConstraintExample,
        with_baseline: bool = False,
    ) -> GenerationResult:
        """Evaluate a single example.

        Args:
            example: The constraint example to evaluate
            with_baseline: If True, also generate unconstrained output for comparison
        """
        # Build prompt and constraint spec
        prompt = example.get_prompt()
        constraint_spec = self._build_constraint_spec(example)

        # Get effective max_tokens (per-example or language default)
        effective_max_tokens = example.get_effective_max_tokens()

        # Generate constrained output
        start_time = time.perf_counter()
        try:
            output = self.generate_fn(prompt, constraint_spec, effective_max_tokens)
            latency_ms = (time.perf_counter() - start_time) * 1000
        except Exception as e:
            return GenerationResult(
                example_id=example.id,
                language=example.language,
                domain=example.domain,
                complexity=example.complexity_str,
                prompt=prompt,
                constraint_spec=constraint_spec,
                generated_output="",
                latency_ms=0,
                matches_constraint=False,
                constraint_type="error",
                validation_error=str(e),
                tags=example.tags,
            )

        # Validate constrained output
        matches, constraint_type, error = self._validate_output(output, example)

        # Analyze regex difficulty
        regex_difficulty = None
        if example.spec.regex:
            regex_difficulty = RegexDifficulty.analyze(example.spec.regex)

        # Generate baseline (unconstrained) if requested
        baseline_output = None
        baseline_matches = None
        baseline_latency_ms = None

        if with_baseline:
            baseline_start = time.perf_counter()
            try:
                # Generate without constraint (empty spec), same token limit
                baseline_output = self.generate_fn(prompt, {}, effective_max_tokens)
                baseline_latency_ms = (time.perf_counter() - baseline_start) * 1000
                # Validate baseline against same constraint
                baseline_matches, _, _ = self._validate_output(baseline_output, example)
            except Exception:
                baseline_output = ""
                baseline_matches = False

        return GenerationResult(
            example_id=example.id,
            language=example.language,
            domain=example.domain,
            complexity=example.complexity_str,
            prompt=prompt,
            constraint_spec=constraint_spec,
            generated_output=output,
            latency_ms=latency_ms,
            matches_constraint=matches,
            constraint_type=constraint_type,
            validation_error=error,
            baseline_output=baseline_output,
            baseline_matches=baseline_matches,
            baseline_latency_ms=baseline_latency_ms,
            regex_difficulty=regex_difficulty,
            tags=example.tags,
        )

    def run_all(
        self,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        with_baseline: bool = False,
    ) -> tuple[EvalMetrics, List[GenerationResult], List[FailureAnalysis]]:
        """Run eval on all matching examples.

        Args:
            progress_callback: Optional callback(current, total) for progress
            with_baseline: If True, also generate unconstrained baseline for comparison

        Returns:
            (metrics, all_results, failure_analyses)
        """
        examples = list(self.get_examples())
        total = len(examples)

        metrics = EvalMetrics()
        by_complexity: Dict[str, EvalMetrics] = {}
        results: List[GenerationResult] = []
        failures: List[FailureAnalysis] = []

        # Track baseline comparison stats
        baseline_stats = {
            "constraint_helped": 0,
            "constraint_hurt": 0,
            "both_passed": 0,
            "both_failed": 0,
        }

        for i, example in enumerate(examples):
            result = self.evaluate_example(example, with_baseline=with_baseline)
            results.append(result)

            # Update metrics
            eval_result = EvalResult(
                example_id=result.example_id,
                satisfied=result.matches_constraint,
                satisfaction_level=(
                    SatisfactionLevel.FULL if result.matches_constraint
                    else SatisfactionLevel.NONE
                ),
                output=result.generated_output,
                baseline_output=result.baseline_output,
                baseline_satisfied=result.baseline_matches,
                latency_ms=result.latency_ms,
                error=result.validation_error,
                metadata={
                    "constraint_type": result.constraint_type,
                    "output_length": len(result.generated_output),
                    "complexity": result.complexity,
                },
            )
            metrics.add_result(
                eval_result,
                language=result.language,
                domain=result.domain,
            )

            # Track by complexity
            if result.complexity not in by_complexity:
                by_complexity[result.complexity] = EvalMetrics()
            by_complexity[result.complexity].add_result(eval_result)

            # Update baseline comparison stats
            if with_baseline and result.baseline_matches is not None:
                if result.constraint_helped:
                    baseline_stats["constraint_helped"] += 1
                elif result.constraint_hurt:
                    baseline_stats["constraint_hurt"] += 1
                elif result.matches_constraint and result.baseline_matches:
                    baseline_stats["both_passed"] += 1
                else:
                    baseline_stats["both_failed"] += 1

            # Categorize failures
            if not result.matches_constraint:
                failure = self._categorize_failure(
                    result.generated_output,
                    example,
                    result.constraint_type,
                )
                failures.append(failure)

            if progress_callback:
                progress_callback(i + 1, total)

        # Attach complexity breakdown to metrics
        metrics.by_complexity = by_complexity
        metrics.baseline_stats = baseline_stats if with_baseline else None

        return metrics, results, failures

    def run_sample(
        self,
        n: int = 10,
        seed: Optional[int] = None,
    ) -> tuple[EvalMetrics, List[GenerationResult], List[FailureAnalysis]]:
        """Run eval on a random sample of examples.

        Args:
            n: Number of examples to sample
            seed: Random seed for reproducibility

        Returns:
            (metrics, results, failures)
        """
        import random

        if seed is not None:
            random.seed(seed)

        examples = list(self.get_examples())
        sample = random.sample(examples, min(n, len(examples)))

        # Temporarily override get_examples
        original_examples = self.get_examples
        self.get_examples = lambda: iter(sample)

        try:
            return self.run_all()
        finally:
            self.get_examples = original_examples
