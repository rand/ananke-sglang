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
"""AnankeGrammar: Multi-domain constrained decoding grammar object.

This module implements AnankeGrammar, which extends SGLang's BaseGrammarObject
to support compositional constraint checking across multiple domains (syntax,
types, imports, control flow, semantics).

AnankeGrammar wraps an underlying llguidance grammar for syntax constraints
and adds additional domain constraints via the Ananke constraint system.

Architecture:
    AnankeGrammar delegates syntax mask computation to llguidance for efficiency
    (~50μs/token), then applies additional domain masks via bitwise AND.

References:
    - llguidance: Dynamic mask computation with lazy automata
    - Hazel: Compositional constraint semantics
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch

from sglang.srt.constrained.base_grammar_backend import BaseGrammarObject

# Support both relative imports (when used as subpackage) and absolute imports (standalone testing)
try:
    from ..core.checkpoint import (
        Checkpoint,
        CheckpointManager,
        UnifiedCheckpoint,
    )
    from ..core.constraint import Satisfiability
    from ..core.domain import ConstraintDomain, GenerationContext, MaskPool
    from ..core.unified import (
        UNIFIED_TOP,
        UnifiedConstraint,
    )
    from ..masks.lazy import (
        EvaluationBudget,
        EvaluationPriority,
        LazyConstraintEvaluator,
        TieredConstraintEvaluator,
        EvaluationTier,
        DEFAULT_DOMAIN_TIERS,
        ParallelDomainEvaluator,
        AdaptiveTieredEvaluator,
    )
    from ..masks.speculative import (
        SpeculativeMaskCache,
    )
    from ..masks.relaxation import (
        MaskRelaxation,
        RelaxationPolicy,
        RelaxationResult,
        RelaxationAwareEvaluator,
        compute_mask_with_relaxation,
    )
except ImportError:
    from core.checkpoint import (
        Checkpoint,
        CheckpointManager,
        UnifiedCheckpoint,
    )
    from core.constraint import Satisfiability
    from core.domain import ConstraintDomain, GenerationContext, MaskPool
    from core.unified import (
        UNIFIED_TOP,
        UnifiedConstraint,
    )
    from masks.lazy import (
        EvaluationBudget,
        EvaluationPriority,
        LazyConstraintEvaluator,
        TieredConstraintEvaluator,
        EvaluationTier,
        DEFAULT_DOMAIN_TIERS,
        ParallelDomainEvaluator,
        AdaptiveTieredEvaluator,
    )
    from masks.speculative import (
        SpeculativeMaskCache,
    )
    from masks.relaxation import (
        MaskRelaxation,
        RelaxationPolicy,
        RelaxationResult,
        RelaxationAwareEvaluator,
        compute_mask_with_relaxation,
    )

if TYPE_CHECKING:
    from sglang.srt.constrained.llguidance_backend import GuidanceGrammar
    from ..spec.constraint_spec import ConstraintSpec
    from ..adaptive.intensity import ConstraintIntensity

logger = logging.getLogger(__name__)

from enum import Enum, auto


class EvaluationStrategy(Enum):
    """Strategy for evaluating domain constraints.

    LAZY: Budget-limited lazy evaluation with priority ordering
    TIERED: Tier-based evaluation with early termination on popcount
    ADAPTIVE: Tiered evaluation with runtime adaptation (recommended)
    PARALLEL: Parallel domain evaluation with early termination
    EAGER: Evaluate all domains sequentially (original behavior)
    """

    LAZY = auto()
    TIERED = auto()
    ADAPTIVE = auto()
    PARALLEL = auto()
    EAGER = auto()


class AnankeGrammar(BaseGrammarObject):
    """Multi-domain constrained decoding grammar for SGLang.

    AnankeGrammar wraps an underlying syntax grammar (llguidance) and adds
    compositional constraint checking across multiple domains. Token masks
    from all domains are fused via bitwise AND.

    The grammar maintains:
    - A wrapped syntax grammar for efficient CFG parsing
    - A unified constraint combining all domain constraints
    - Per-domain state for incremental constraint updates
    - Checkpoint history for rollback support

    Attributes:
        syntax_grammar: Underlying llguidance grammar for syntax
        constraint: Current unified constraint across all domains
        domains: Dictionary of constraint domains
        context: Current generation context
        checkpoint_manager: Manages rollback checkpoints
    """

    def __init__(
        self,
        syntax_grammar: Optional["GuidanceGrammar"],
        domains: Dict[str, ConstraintDomain],
        constraint: UnifiedConstraint = UNIFIED_TOP,
        vocab_size: int = 0,
        device: str = "cuda",
        tokenizer: Optional[Any] = None,
        language: str = "python",
        max_rollback_tokens: int = 200,
        checkpoint_interval: int = 1,
        mask_pool_size: int = 8,
        constraint_spec: Optional["ConstraintSpec"] = None,
        intensity: Optional["ConstraintIntensity"] = None,
        evaluation_strategy: EvaluationStrategy = EvaluationStrategy.ADAPTIVE,
        enable_speculative_cache: bool = True,
        speculative_lookahead: int = 3,
        tiered_target_popcount: int = 100,
        parallel_workers: int = 4,
        allow_relaxation: bool = True,
        relaxation_threshold: int = 10,
        enable_early_termination: bool = True,
    ):
        """Initialize AnankeGrammar.

        Args:
            syntax_grammar: Wrapped llguidance grammar for syntax constraints
            domains: Dictionary mapping domain names to domain instances
            constraint: Initial unified constraint (default: TOP)
            vocab_size: Vocabulary size for mask allocation
            device: PyTorch device for tensor operations
            tokenizer: Tokenizer for text decode operations
            language: Programming language being generated
            max_rollback_tokens: Maximum tokens for rollback (like XGrammar)
            checkpoint_interval: Create checkpoint every N tokens (default: 1).
                Set higher (e.g., 10) for better performance with sparse rollback.
            mask_pool_size: Number of pre-allocated mask tensors (default: 8).
                Set to 0 to disable pooling.
            constraint_spec: Rich constraint specification (optional).
                If provided, enables per-request language configuration,
                type context, import context, and semantic constraints.
            intensity: Constraint intensity level for this grammar.
                Determines which domains are active for mask computation.
            evaluation_strategy: Strategy for domain evaluation (default: TIERED).
                TIERED is recommended for best performance.
            enable_speculative_cache: Enable speculative mask precomputation.
            speculative_lookahead: Number of tokens to precompute ahead.
            tiered_target_popcount: Target popcount for early termination in tiered mode.
            parallel_workers: Number of workers for parallel evaluation.
            allow_relaxation: Enable progressive domain relaxation when masks
                become too tight. When enabled, domains are tried in priority
                order and skipped if applying them would drop popcount below
                the relaxation threshold.
            relaxation_threshold: Minimum popcount before relaxation triggers.
                If a domain mask would reduce popcount below this value, that
                domain is skipped (relaxed).
            enable_early_termination: Enable early termination when regex
                constraint is satisfied at a natural code boundary. This can
                prevent generation from continuing past a valid completion.
        """
        super().__init__()
        self.syntax_grammar = syntax_grammar
        self.domains = domains
        self.constraint = constraint
        self.vocab_size = vocab_size
        self.device = device
        self.tokenizer = tokenizer
        self.language = language
        self._mask_pool_size = mask_pool_size
        self.constraint_spec = constraint_spec
        self.intensity = intensity

        # Evaluation strategy configuration
        self._evaluation_strategy = evaluation_strategy
        self._tiered_target_popcount = tiered_target_popcount
        self._parallel_workers = parallel_workers
        self._speculative_lookahead = speculative_lookahead
        self._enable_speculative_cache = enable_speculative_cache

        # Pre-allocated mask tensor pool to avoid per-token CUDA allocations
        self._mask_pool: Optional[MaskPool] = None
        if vocab_size > 0 and mask_pool_size > 0:
            self._mask_pool = MaskPool(
                vocab_size=vocab_size,
                device=device,
                pool_size=mask_pool_size,
            )

        # Generation context
        self.context = GenerationContext(
            vocab_size=vocab_size,
            device=device,
            language=language,
            tokenizer=tokenizer,
            mask_pool=self._mask_pool,
        )

        # Checkpoint management for rollback
        self.checkpoint_manager = CheckpointManager(
            max_checkpoints=max_rollback_tokens
        )

        # Sparse checkpointing configuration
        self._checkpoint_interval = checkpoint_interval
        self._tokens_since_checkpoint = 0
        self._last_constraint_hash: Optional[int] = None

        # Cache for computed domain masks
        self._domain_mask_cache: Dict[str, torch.Tensor] = {}

        # Initialize evaluators based on strategy
        self._lazy_evaluator: Optional[LazyConstraintEvaluator] = None
        self._tiered_evaluator: Optional[TieredConstraintEvaluator] = None
        self._adaptive_evaluator: Optional[AdaptiveTieredEvaluator] = None
        self._parallel_evaluator: Optional[ParallelDomainEvaluator] = None
        self._speculative_cache: Optional[SpeculativeMaskCache] = None

        self._evaluation_budget = EvaluationBudget(
            max_time_ns=2_000_000,  # 2ms budget
            max_domains=5,
            min_selectivity=0.95,  # Stop if 95% of vocab blocked
        )

        # Initialize evaluator based on strategy
        if evaluation_strategy == EvaluationStrategy.LAZY:
            self._lazy_evaluator = self._init_lazy_evaluator()
        elif evaluation_strategy == EvaluationStrategy.TIERED:
            self._tiered_evaluator = self._init_tiered_evaluator()
        elif evaluation_strategy == EvaluationStrategy.ADAPTIVE:
            self._adaptive_evaluator = self._init_adaptive_evaluator()
        elif evaluation_strategy == EvaluationStrategy.PARALLEL:
            self._parallel_evaluator = self._init_parallel_evaluator()
        # EAGER uses no evaluator - direct sequential evaluation

        # Initialize speculative cache if enabled
        if enable_speculative_cache and vocab_size > 0:
            self._speculative_cache = self._init_speculative_cache()

        # Relaxation configuration
        self._allow_relaxation = allow_relaxation
        self._relaxation_threshold = relaxation_threshold
        self._relaxation_policy = RelaxationPolicy(
            enabled=allow_relaxation,
            threshold=relaxation_threshold,
        )
        self._relaxation_evaluator: Optional[RelaxationAwareEvaluator] = None
        self._last_relaxation_result: Optional[RelaxationResult] = None

        # Initialize relaxation evaluator if enabled
        if allow_relaxation:
            self._relaxation_evaluator = self._init_relaxation_evaluator()

        # Early termination configuration
        self._enable_early_termination = enable_early_termination
        self._early_termination_triggered = False

    def accept_token(self, token: int) -> None:
        """Accept a generated token, updating all domain constraints.

        This method:
        1. Updates the syntax grammar (delegation to llguidance)
        2. Updates each domain's constraint via observe_token
        3. Updates the unified constraint
        4. Optionally creates a checkpoint for rollback

        If any domain constraint becomes BOTTOM (unsatisfiable), the
        grammar is marked as finished to signal early termination.

        Args:
            token: The token ID that was generated
        """
        # Create checkpoint before mutation (for potential rollback)
        self._maybe_create_checkpoint()

        # Decode token for context update
        token_text = ""
        if self.tokenizer is not None:
            try:
                token_text = self.tokenizer.decode([token])
            except Exception:
                token_text = ""

        # Update syntax grammar
        if self.syntax_grammar is not None:
            self.syntax_grammar.accept_token(token)
            if self.syntax_grammar.finished:
                self.finished = True
                return

        # Update each domain constraint
        # Capture old constraint for change detection
        old_unified_constraint = self.constraint

        new_syntax = self.constraint.syntax
        new_types = self.constraint.types
        new_imports = self.constraint.imports
        new_controlflow = self.constraint.controlflow
        new_semantics = self.constraint.semantics

        for domain_name, domain in self.domains.items():
            domain_constraint = getattr(self.constraint, domain_name)
            new_constraint = domain.observe_token(domain_constraint, token, self.context)

            # Update the appropriate field
            if domain_name == "syntax":
                new_syntax = new_constraint
            elif domain_name == "types":
                new_types = new_constraint
            elif domain_name == "imports":
                new_imports = new_constraint
            elif domain_name == "controlflow":
                new_controlflow = new_constraint
            elif domain_name == "semantics":
                new_semantics = new_constraint

        # Update unified constraint
        self.constraint = UnifiedConstraint(
            syntax=new_syntax,
            types=new_types,
            imports=new_imports,
            controlflow=new_controlflow,
            semantics=new_semantics,
        )

        # Update generation context
        self.context = self.context.extend(token, token_text)

        # Selectively invalidate mask cache (only for domains that changed)
        # Compare old vs new constraint hashes to detect changes
        self._invalidate_changed_domain_masks(old_unified_constraint, self.constraint)

        # Check for unsatisfiability
        if self.constraint.satisfiability() == Satisfiability.UNSAT:
            logger.debug("Constraint became unsatisfiable, marking grammar finished")
            self.finished = True
            return

        # Check if regex constraint can still be satisfied
        if self._check_regex_prefix_violation():
            logger.debug("Output cannot satisfy regex constraint, marking grammar finished")
            self.finished = True
            return

        # Check for early termination (regex satisfied at natural boundary)
        if self._enable_early_termination and self._check_regex_satisfied():
            if self._is_natural_boundary():
                logger.info(
                    f"Early termination: regex satisfied at natural boundary "
                    f"(position={self.context.position}, text_len={len(self.context.generated_text)})"
                )
                self.finished = True
                self._early_termination_triggered = True

    def rollback(self, k: int) -> None:
        """Roll back k tokens, restoring previous state.

        This method implements XGrammar-style rollback by restoring
        the grammar to a checkpoint from k tokens ago.

        Args:
            k: Number of tokens to roll back
        """
        target_position = self.context.position - k
        if target_position < 0:
            target_position = 0

        checkpoint = self.checkpoint_manager.rollback_to(target_position)
        if checkpoint is None:
            logger.warning(
                f"Cannot rollback {k} tokens: no checkpoint found at position {target_position}"
            )
            return

        # Restore unified constraint
        self.constraint = checkpoint.unified_constraint

        # Restore each domain's state
        for domain_name, domain in self.domains.items():
            domain_checkpoint = checkpoint.get_domain_checkpoint(domain_name)
            if domain_checkpoint is not None:
                domain.restore(domain_checkpoint)

        # Restore context
        if checkpoint.context_snapshot:
            self.context = GenerationContext(
                generated_text=checkpoint.context_snapshot.get("generated_text", ""),
                generated_tokens=checkpoint.context_snapshot.get("generated_tokens", []),
                position=checkpoint.context_snapshot.get("position", 0),
                vocab_size=self.vocab_size,
                device=self.device,
                language=self.language,
                tokenizer=self.tokenizer,
                metadata=checkpoint.context_snapshot.get("metadata", {}),
            )

        # Clear mask cache - cached masks are invalid after constraint rollback
        # (masks were computed with post-rollback constraints, not restored ones)
        self._domain_mask_cache.clear()

        # Reset finished flag
        self.finished = False

    def is_terminated(self) -> bool:
        """Check if grammar has reached a terminal state."""
        if self.syntax_grammar is not None:
            return self.syntax_grammar.is_terminated()
        return self.finished

    def fill_vocab_mask(
        self,
        vocab_mask: torch.Tensor,
        idx: int,
        use_lazy_evaluation: bool = True,
    ) -> None:
        """Fill vocabulary mask with valid tokens for this request.

        This method:
        1. Delegates to syntax grammar for CFG-based masking
        2. Computes additional domain masks (types, imports, etc.)
        3. Applies additional masks via bitwise AND

        When use_lazy_evaluation is True, domains are evaluated in priority
        order with budget control. Expensive domains may be skipped if the
        mask is already sufficiently constrained.

        Args:
            vocab_mask: Bitmask tensor to fill [batch_size, mask_size]
            idx: Index into the batch for this request
            use_lazy_evaluation: If True, use budget-limited lazy evaluation
        """
        # Track mask popcount before syntax grammar
        before_syntax = self._count_allowed_tokens(vocab_mask, idx) if logger.isEnabledFor(logging.DEBUG) else None

        # First, delegate to syntax grammar
        if self.syntax_grammar is not None:
            self.syntax_grammar.fill_vocab_mask(vocab_mask, idx)
            if self.syntax_grammar.finished:
                self.finished = True
                return

        # Log syntax mask effect
        if before_syntax is not None:
            after_syntax = self._count_allowed_tokens(vocab_mask, idx)
            logger.debug(
                f"Syntax grammar mask: {before_syntax} -> {after_syntax} tokens allowed"
            )

        if use_lazy_evaluation:
            self._fill_vocab_mask_lazy(vocab_mask, idx)
        else:
            self._fill_vocab_mask_eager(vocab_mask, idx)

    def _count_allowed_tokens(self, vocab_mask: torch.Tensor, idx: int) -> int:
        """Count number of allowed tokens in the mask (for debugging)."""
        try:
            # Count set bits in the mask
            mask_row = vocab_mask[idx]
            return int(mask_row.sum().item() * 32)  # Approximate (each int32 has up to 32 bits)
        except Exception:
            return -1

    def _check_regex_prefix_violation(self) -> bool:
        """Check if current output can still possibly match the regex constraint.

        Uses prefix matching to detect early when the generated text cannot
        satisfy the regex, enabling early termination.

        Returns:
            True if the output definitely cannot match the regex (should terminate),
            False if it might still be possible to match.
        """
        # Only check if we have a regex constraint and generated text
        if self.constraint_spec is None or not self.constraint_spec.regex:
            return False

        generated_text = self.context.generated_text
        if not generated_text:
            return False

        # Skip check for short outputs (not enough signal)
        if len(generated_text) < 20:
            return False

        regex_pattern = self.constraint_spec.regex

        # Quick check: if pattern is anchored, verify prefix
        if regex_pattern.startswith("^"):
            # Extract the non-regex prefix if possible
            # For anchored patterns, first part should match
            import re
            try:
                # Try to match what we have so far
                # If pattern requires specific start and we don't have it, fail
                partial_pattern = regex_pattern[:min(len(regex_pattern), 100)]

                # Check if any prefix of our output matches the pattern prefix
                for length in [len(generated_text), len(generated_text) // 2, 20]:
                    prefix = generated_text[:length]
                    if re.match(partial_pattern, prefix):
                        return False  # Could still match

                # If nothing matched and we have significant output, likely violated
                if len(generated_text) > 100:
                    logger.debug(
                        f"Regex prefix violation: output '{generated_text[:50]}...' "
                        f"doesn't match anchored pattern"
                    )
                    return True

            except re.error:
                pass  # Invalid regex, don't terminate

        return False

    def _check_regex_satisfied(self) -> bool:
        """Check if the current output fully satisfies the regex constraint.

        This is used for early termination: if the regex is already satisfied,
        we can stop generation at a natural boundary instead of continuing.

        Returns:
            True if the output fully matches the regex constraint,
            False otherwise.
        """
        # Only check if we have a regex constraint and generated text
        if self.constraint_spec is None or not self.constraint_spec.regex:
            return False

        generated_text = self.context.generated_text
        if not generated_text:
            return False

        # Need at least some output to consider satisfied
        if len(generated_text) < 10:
            return False

        regex_pattern = self.constraint_spec.regex

        import re
        try:
            # Check if the full pattern matches the generated text
            # Use fullmatch for anchored patterns, search for unanchored
            if regex_pattern.startswith("^") and regex_pattern.endswith("$"):
                # Fully anchored pattern - must match entire string
                match = re.fullmatch(regex_pattern, generated_text)
            elif regex_pattern.startswith("^"):
                # Start-anchored - match at beginning
                match = re.match(regex_pattern, generated_text)
            else:
                # Unanchored - find anywhere in text
                match = re.search(regex_pattern, generated_text)

            if match:
                logger.debug(
                    f"Regex satisfied: pattern '{regex_pattern[:50]}...' matched at "
                    f"position {match.start()}-{match.end()}"
                )
                return True

        except re.error as e:
            logger.warning(f"Invalid regex pattern: {e}")

        return False

    def _is_natural_boundary(self) -> bool:
        """Check if the current position is a natural code boundary.

        Natural boundaries are positions where it makes sense to stop generation,
        such as after a complete statement, function definition, or block.

        Returns:
            True if at a natural boundary, False otherwise.
        """
        generated_text = self.context.generated_text
        if not generated_text:
            return False

        # Strip trailing whitespace for boundary detection
        text = generated_text.rstrip()
        if not text:
            return False

        # Language-specific natural boundaries
        # These are patterns where stopping makes sense
        boundaries = {
            "python": [
                "\n\n",           # Blank line (end of block)
                ":\n",            # End of header line
                "\n    pass\n",   # Pass statement
                "\nreturn ",      # Return statement start
                "\n    return ",  # Indented return
            ],
            "rust": [
                "}\n",            # End of block
                ";\n",            # End of statement
                "\n}\n",          # End of function/impl
            ],
            "go": [
                "}\n",            # End of block
                ";\n",            # End of statement (rare in Go)
                "\n}\n",          # End of function
            ],
            "typescript": [
                "}\n",            # End of block
                ";\n",            # End of statement
                "\n}\n",          # End of function/class
            ],
            "kotlin": [
                "}\n",            # End of block
                "\n}\n",          # End of function/class
            ],
            "swift": [
                "}\n",            # End of block
                "\n}\n",          # End of function/class
            ],
            "zig": [
                "}\n",            # End of block
                ";\n",            # End of statement
                "\n}\n",          # End of function/block
            ],
        }

        # Get boundaries for current language, with default fallback
        lang_boundaries = boundaries.get(self.language, ["\n\n", "}\n", ";\n"])

        # Check if text ends with any boundary pattern
        for boundary in lang_boundaries:
            # Handle both the text and boundary having trailing newlines
            boundary_stripped = boundary.rstrip()
            # Only check non-empty stripped boundaries (empty string matches everything)
            if boundary_stripped and text.endswith(boundary_stripped):
                return True

        # Additional heuristic: check for complete-looking statements
        # A line ending with certain patterns suggests completeness
        last_line = text.split("\n")[-1] if "\n" in text else text
        last_line_stripped = last_line.strip()  # Strip leading/trailing whitespace

        # Endings that the line must END WITH
        complete_endings = {
            "python": [":", "pass"],
            "rust": [";", "}", "),", ");"],
            "go": ["}", ";"],
            "typescript": [";", "}", "),", ");"],
            "kotlin": ["}", "),", ");"],
            "swift": ["}", "),", ");"],
            "zig": [";", "}", "),", ");"],
        }

        lang_endings = complete_endings.get(self.language, [";", "}"])

        for ending in lang_endings:
            if last_line_stripped.endswith(ending):
                return True

        # Keywords that the line must START WITH (Python-specific)
        # These are statement keywords that indicate a complete statement
        if self.language == "python":
            statement_keywords = ["return", "break", "continue", "raise"]
            for keyword in statement_keywords:
                if last_line_stripped.startswith(keyword):
                    return True

        return False

    def _fill_vocab_mask_eager(self, vocab_mask: torch.Tensor, idx: int) -> None:
        """Eagerly evaluate all domain masks (original behavior).

        Args:
            vocab_mask: Bitmask tensor to fill [batch_size, mask_size]
            idx: Index into the batch for this request
        """
        for domain_name, domain in self.domains.items():
            if domain_name == "syntax":
                continue  # Already handled by syntax_grammar

            domain_constraint = getattr(self.constraint, domain_name)
            if domain_constraint.is_top():
                continue  # No additional restriction

            # Compute domain mask
            domain_mask = self._get_or_compute_domain_mask(domain_name, domain)

            # Apply via bitwise AND with existing mask
            self._apply_domain_mask(vocab_mask, idx, domain_mask)

    def _fill_vocab_mask_lazy(self, vocab_mask: torch.Tensor, idx: int) -> None:
        """Evaluate domain masks using the configured evaluation strategy.

        Domains are evaluated according to the configured strategy:
        - LAZY: Budget-limited evaluation with priority ordering
        - TIERED: Tier-based evaluation with early termination on popcount
        - PARALLEL: Parallel evaluation with early termination
        - EAGER: Sequential evaluation of all domains (handled in _fill_vocab_mask_eager)

        Args:
            vocab_mask: Bitmask tensor to fill [batch_size, mask_size]
            idx: Index into the batch for this request
        """
        # Build constraints dict
        constraints = {}
        for domain_name in self.domains:
            if domain_name == "syntax":
                continue  # Already handled by syntax_grammar

            domain_constraint = getattr(self.constraint, domain_name)
            if not domain_constraint.is_top():
                constraints[domain_name] = domain_constraint

        if not constraints:
            return  # Nothing to evaluate

        fused_mask = None

        if self._evaluation_strategy == EvaluationStrategy.LAZY:
            # Budget-limited lazy evaluation
            if self._lazy_evaluator is not None:
                result = self._lazy_evaluator.evaluate(
                    constraints=constraints,
                    context=self.context,
                    budget=self._evaluation_budget,
                )
                fused_mask = result.fused_mask

        elif self._evaluation_strategy == EvaluationStrategy.TIERED:
            # Tier-based evaluation with early termination
            if self._tiered_evaluator is not None:
                result = self._tiered_evaluator.evaluate(constraints, self.context)
                fused_mask = result.fused_mask

        elif self._evaluation_strategy == EvaluationStrategy.ADAPTIVE:
            # Adaptive tiered evaluation with runtime optimization
            if self._adaptive_evaluator is not None:
                result = self._adaptive_evaluator.evaluate(constraints, self.context)
                fused_mask = result.fused_mask

        elif self._evaluation_strategy == EvaluationStrategy.PARALLEL:
            # Parallel evaluation with early termination
            if self._parallel_evaluator is not None:
                result = self._parallel_evaluator.evaluate(constraints, self.context)
                fused_mask = result.fused_mask

        else:
            # EAGER: Fall through to manual sequential (shouldn't reach here)
            for domain_name, constraint in constraints.items():
                domain = self.domains[domain_name]
                domain_mask = domain.token_mask(constraint, self.context)
                if fused_mask is None:
                    fused_mask = domain_mask.clone()
                else:
                    fused_mask &= domain_mask

        # Apply the fused mask, optionally with relaxation
        if fused_mask is not None:
            if self._allow_relaxation:
                self._apply_domain_mask_with_relaxation(
                    vocab_mask, idx, fused_mask, constraints
                )
            else:
                self._apply_domain_mask(vocab_mask, idx, fused_mask)

    def _apply_domain_mask_with_relaxation(
        self,
        vocab_mask: torch.Tensor,
        idx: int,
        fused_mask: torch.Tensor,
        constraints: Dict[str, Any],
    ) -> None:
        """Apply domain mask with relaxation support.

        If the fused mask would reduce the vocabulary mask popcount below
        the relaxation threshold, individual domain masks are applied
        progressively, skipping domains that would cause the popcount to
        drop too low.

        Args:
            vocab_mask: Bitmask tensor to fill [batch_size, mask_size]
            idx: Index into the batch for this request
            fused_mask: Pre-computed fused mask from evaluation strategy
            constraints: Dictionary of domain constraints being applied
        """
        # First, check if we can apply the full fused mask
        candidate_popcount = self._count_candidate_popcount(vocab_mask, idx, fused_mask)

        if candidate_popcount >= self._relaxation_threshold:
            # Full mask is safe to apply
            self._apply_domain_mask(vocab_mask, idx, fused_mask)
            self._last_relaxation_result = RelaxationResult(
                fused_mask=fused_mask,
                relaxation_level=MaskRelaxation.NONE,
                domains_applied=list(constraints.keys()),
                domains_relaxed=[],
                final_popcount=candidate_popcount,
                initial_popcount=self._count_allowed_tokens(vocab_mask, idx),
            )
            return

        # Full fused mask would drop too low - apply domains progressively
        logger.debug(
            f"Full mask popcount ({candidate_popcount}) below threshold "
            f"({self._relaxation_threshold}), applying progressive relaxation"
        )

        initial_popcount = self._count_allowed_tokens(vocab_mask, idx)
        domains_applied: List[str] = []
        domains_relaxed: List[str] = []
        popcount_history: Dict[str, int] = {}

        # Apply domains in priority order: types -> imports -> controlflow -> semantics
        # This order ensures more critical domains are tried first
        application_order = ["types", "imports", "controlflow", "semantics"]

        for domain_name in application_order:
            if domain_name not in constraints or domain_name not in self.domains:
                continue

            # Compute this domain's mask
            domain = self.domains[domain_name]
            domain_constraint = constraints[domain_name]
            domain_mask = domain.token_mask(domain_constraint, self.context)

            # Check candidate popcount
            candidate_popcount = self._count_candidate_popcount(
                vocab_mask, idx, domain_mask
            )

            if candidate_popcount >= self._relaxation_threshold:
                # Safe to apply
                self._apply_domain_mask(vocab_mask, idx, domain_mask)
                domains_applied.append(domain_name)
                popcount_history[domain_name] = candidate_popcount
            else:
                # Would drop below threshold - relax
                domains_relaxed.append(domain_name)
                popcount_history[f"{domain_name}_relaxed"] = candidate_popcount

                if self._relaxation_policy.log_relaxation:
                    logger.info(
                        f"Relaxing domain '{domain_name}': popcount would drop "
                        f"from {self._count_allowed_tokens(vocab_mask, idx)} to "
                        f"{candidate_popcount} (threshold={self._relaxation_threshold})"
                    )

        # Determine final relaxation level
        final_popcount = self._count_allowed_tokens(vocab_mask, idx)

        if not domains_relaxed:
            relaxation_level = MaskRelaxation.NONE
        elif len(domains_applied) == 0:
            relaxation_level = MaskRelaxation.SYNTAX_ONLY
        else:
            relaxation_level = MaskRelaxation.PARTIAL

        # Store result for introspection
        self._last_relaxation_result = RelaxationResult(
            fused_mask=fused_mask,  # Original fused mask
            relaxation_level=relaxation_level,
            domains_applied=domains_applied,
            domains_relaxed=domains_relaxed,
            final_popcount=final_popcount,
            initial_popcount=initial_popcount,
            popcount_history=popcount_history,
        )

        if domains_relaxed:
            logger.info(
                f"Relaxation: applied={domains_applied}, relaxed={domains_relaxed}, "
                f"final_popcount={final_popcount}"
            )

    def _count_candidate_popcount(
        self,
        vocab_mask: torch.Tensor,
        idx: int,
        domain_mask: torch.Tensor,
    ) -> int:
        """Count popcount if domain mask were applied (without modifying vocab_mask).

        Args:
            vocab_mask: Current bitmask tensor
            idx: Batch index
            domain_mask: Domain mask to test

        Returns:
            Popcount that would result from applying domain_mask
        """
        vocab_size = domain_mask.shape[0]
        device = domain_mask.device
        mask_size = vocab_mask.shape[1]

        # Pad domain mask to multiple of 32
        padded_size = ((vocab_size + 31) // 32) * 32
        if vocab_size < padded_size:
            padded = torch.zeros(padded_size, dtype=torch.bool, device=device)
            padded[:vocab_size] = domain_mask
        else:
            padded = domain_mask

        # Reshape to [num_words, 32] for bit packing
        reshaped = padded.view(-1, 32)

        # Get powers of 2 lookup table
        if not hasattr(self, '_powers_of_2') or self._powers_of_2.device != device:
            self._powers_of_2 = (2 ** torch.arange(32, device=device, dtype=torch.int64))

        # Pack domain mask to int32
        packed_domain = (reshaped.to(torch.int64) * self._powers_of_2).sum(dim=1).to(torch.int32)

        # Compute hypothetical AND result
        num_words = min(packed_domain.shape[0], mask_size)
        candidate = vocab_mask[idx, :num_words] & packed_domain[:num_words]

        # Count set bits using lookup table (more efficient than Python loop)
        # Convert int32 tensor to individual bits and sum
        # Unpack to uint8 and use precomputed popcount lookup
        if not hasattr(self, '_popcount_lut'):
            # Precompute popcount lookup table for bytes
            self._popcount_lut = torch.tensor(
                [bin(i).count('1') for i in range(256)],
                dtype=torch.int32,
                device=device,
            )

        candidate_bytes = candidate.view(torch.uint8)
        popcount = self._popcount_lut[candidate_bytes.long()].sum().item()

        return int(popcount)

    def allocate_vocab_mask(
        self, vocab_size: int, batch_size: int, device
    ) -> torch.Tensor:
        """Allocate vocabulary mask tensor.

        Delegates to syntax grammar if available, otherwise allocates
        a standard bitmask tensor.

        Args:
            vocab_size: Size of vocabulary
            batch_size: Number of requests in batch
            device: PyTorch device

        Returns:
            Allocated bitmask tensor
        """
        if self.syntax_grammar is not None:
            return self.syntax_grammar.allocate_vocab_mask(vocab_size, batch_size, device)

        # Fallback: allocate standard bitmask (int32, 32 tokens per element)
        # Initialize to all 1s (all tokens allowed) so domain constraints can restrict
        # Using -1 for signed int32 sets all bits to 1
        mask_size = (vocab_size + 31) // 32
        return torch.full(
            (batch_size, mask_size), fill_value=-1, dtype=torch.int32, device=device
        )

    @staticmethod
    def move_vocab_mask(vocab_mask: torch.Tensor, device) -> torch.Tensor:
        """Move vocabulary mask to device."""
        return vocab_mask.to(device, non_blocking=True)

    @staticmethod
    def apply_vocab_mask(logits: torch.Tensor, vocab_mask: torch.Tensor) -> None:
        """Apply vocabulary mask to logits tensor.

        This uses llguidance's efficient bitmask application if available.
        """
        try:
            from llguidance.torch import apply_token_bitmask_inplace

            apply_token_bitmask_inplace(logits, vocab_mask)
        except ImportError:
            # Fallback: manual mask application
            # Convert bitmask to boolean mask
            batch_size, mask_size = vocab_mask.shape
            vocab_size = logits.shape[-1]

            for b in range(batch_size):
                for i in range(vocab_size):
                    word_idx = i // 32
                    bit_idx = i % 32
                    if word_idx < mask_size:
                        if not (vocab_mask[b, word_idx] & (1 << bit_idx)):
                            logits[b, i] = float("-inf")

    def copy(self) -> "AnankeGrammar":
        """Create a copy of this grammar for parallel decoding.

        Returns:
            A new AnankeGrammar with copied state
        """
        # Copy syntax grammar
        syntax_copy = None
        if self.syntax_grammar is not None:
            syntax_copy = self.syntax_grammar.copy()

        # Create new instance with same configuration
        return AnankeGrammar(
            syntax_grammar=syntax_copy,
            domains=self.domains,  # Domains are shared (stateless per design)
            constraint=self.constraint,  # Constraints are immutable
            vocab_size=self.vocab_size,
            device=self.device,
            tokenizer=self.tokenizer,
            language=self.language,
            max_rollback_tokens=self.checkpoint_manager.max_checkpoints,
            checkpoint_interval=self._checkpoint_interval,
            mask_pool_size=self._mask_pool_size,
            constraint_spec=self.constraint_spec,
            intensity=self.intensity,
            evaluation_strategy=self._evaluation_strategy,
            enable_speculative_cache=self._enable_speculative_cache,
            speculative_lookahead=self._speculative_lookahead,
            tiered_target_popcount=self._tiered_target_popcount,
            parallel_workers=self._parallel_workers,
            allow_relaxation=self._allow_relaxation,
            relaxation_threshold=self._relaxation_threshold,
            enable_early_termination=self._enable_early_termination,
        )

    def inject_context(self, spec: "ConstraintSpec") -> None:
        """Inject fresh context from a ConstraintSpec into this grammar.

        This is called AFTER cache lookup, allowing:
        - Cached syntax grammar reuse
        - Fresh type/import/semantic context per request

        The method updates the constraint_spec and propagates context
        to each domain that supports it.

        Args:
            spec: ConstraintSpec with context to inject
        """
        self.constraint_spec = spec

        # Update language if specified in spec
        if spec.language is not None:
            self.language = spec.language
            self.context = GenerationContext(
                generated_text=self.context.generated_text,
                generated_tokens=self.context.generated_tokens,
                position=self.context.position,
                vocab_size=self.vocab_size,
                device=self.device,
                language=spec.language,
                tokenizer=self.tokenizer,
                metadata=self.context.metadata,
                mask_pool=self._mask_pool,
            )

        # Propagate context to domains
        for domain_name, domain in self.domains.items():
            if hasattr(domain, "inject_context"):
                domain.inject_context(spec)

    def try_jump_forward(self, tokenizer) -> Optional[Tuple[List[int], str]]:
        """Try to jump forward in the grammar.

        Delegates to syntax grammar if available.

        Returns:
            Jump forward helper or None if not possible
        """
        if self.syntax_grammar is not None:
            return self.syntax_grammar.try_jump_forward(tokenizer)
        return None

    def jump_forward_str_state(
        self, helper: Tuple[List[int], str]
    ) -> Tuple[str, int]:
        """Jump forward and get string state.

        Delegates to syntax grammar if available.
        """
        if self.syntax_grammar is not None:
            return self.syntax_grammar.jump_forward_str_state(helper)
        return "", -1

    def jump_and_retokenize(
        self,
        old_output_ids: List[int],
        new_output_ids: List[int],
        next_state: int,
    ) -> None:
        """Handle jump and retokenization.

        Delegates to syntax grammar if available.
        """
        if self.syntax_grammar is not None:
            self.syntax_grammar.jump_and_retokenize(
                old_output_ids, new_output_ids, next_state
            )

    def _maybe_create_checkpoint(self) -> None:
        """Create a checkpoint if conditions are met.

        Implements sparse checkpointing to reduce overhead:
        1. Create checkpoint every N tokens (default: 10)
        2. Create checkpoint when constraint changes significantly
        3. Always create checkpoint at position 0

        This reduces checkpoint overhead by ~10x while still supporting
        rollback to any position within the last N checkpoints.
        """
        self._tokens_since_checkpoint += 1

        # Condition 1: Always checkpoint at position 0
        if self.context.position == 0:
            self._create_checkpoint()
            return

        # Condition 2: Checkpoint every N tokens
        if self._tokens_since_checkpoint >= self._checkpoint_interval:
            self._create_checkpoint()
            return

        # Condition 3: Checkpoint on significant constraint change
        try:
            current_hash = hash(self.constraint)
        except TypeError:
            current_hash = id(self.constraint)

        if self._last_constraint_hash is not None and current_hash != self._last_constraint_hash:
            # Constraint changed - create checkpoint
            self._create_checkpoint()
            return

        self._last_constraint_hash = current_hash

    def _create_checkpoint(self) -> None:
        """Actually create a checkpoint (called by _maybe_create_checkpoint)."""
        self._tokens_since_checkpoint = 0

        domain_checkpoints = {}
        for domain_name, domain in self.domains.items():
            domain_checkpoints[domain_name] = domain.checkpoint()

        context_snapshot = {
            "generated_text": self.context.generated_text,
            "generated_tokens": self.context.generated_tokens.copy(),
            "position": self.context.position,
            "metadata": self.context.metadata.copy(),
        }

        self.checkpoint_manager.create_checkpoint(
            position=self.context.position,
            unified_constraint=self.constraint,
            domain_checkpoints=domain_checkpoints,
            context_snapshot=context_snapshot,
        )

        # Update last constraint hash
        try:
            self._last_constraint_hash = hash(self.constraint)
        except TypeError:
            self._last_constraint_hash = id(self.constraint)

    def _get_or_compute_domain_mask(
        self,
        domain_name: str,
        domain: ConstraintDomain,
    ) -> torch.Tensor:
        """Get cached domain mask or compute it.

        Args:
            domain_name: Name of the domain
            domain: The domain instance

        Returns:
            Boolean mask tensor for this domain
        """
        if domain_name in self._domain_mask_cache:
            return self._domain_mask_cache[domain_name]

        constraint = getattr(self.constraint, domain_name)
        mask = domain.token_mask(constraint, self.context)
        self._domain_mask_cache[domain_name] = mask
        return mask

    def _apply_domain_mask(
        self,
        vocab_mask: torch.Tensor,
        idx: int,
        domain_mask: torch.Tensor,
    ) -> None:
        """Apply a domain's boolean mask to the vocabulary bitmask.

        This performs bitwise AND between the existing bitmask and
        the domain's boolean mask using vectorized PyTorch operations.

        Complexity: O(vocab_size / 32) tensor operations instead of
        O(vocab_size) Python loop iterations.

        Args:
            vocab_mask: The vocabulary bitmask tensor [batch_size, mask_size]
            idx: Batch index for this request
            domain_mask: Boolean mask from domain [vocab_size]
        """
        vocab_size = domain_mask.shape[0]
        device = domain_mask.device
        mask_size = vocab_mask.shape[1]

        # Pad to multiple of 32 for clean reshaping
        padded_size = ((vocab_size + 31) // 32) * 32
        if vocab_size < padded_size:
            padded = torch.zeros(padded_size, dtype=torch.bool, device=device)
            padded[:vocab_size] = domain_mask
        else:
            padded = domain_mask

        # Reshape to [num_words, 32] for bit packing
        reshaped = padded.view(-1, 32)

        # Get or create powers of 2 lookup table (cached at class level)
        if not hasattr(self, '_powers_of_2') or self._powers_of_2.device != device:
            self._powers_of_2 = (2 ** torch.arange(32, device=device, dtype=torch.int64))

        # Vectorized bit packing: convert each row of 32 bools to an int32
        # Using int64 intermediate to avoid overflow, then cast to int32
        packed = (reshaped.to(torch.int64) * self._powers_of_2).sum(dim=1).to(torch.int32)

        # Apply via vectorized AND with existing mask
        num_words = min(packed.shape[0], mask_size)
        vocab_mask[idx, :num_words] &= packed[:num_words]

    def _invalidate_changed_domain_masks(
        self,
        old_constraint: UnifiedConstraint,
        new_constraint: UnifiedConstraint,
    ) -> None:
        """Selectively invalidate cached masks only for domains that changed.

        Instead of clearing the entire cache on every token, we compare
        constraint hashes to detect which domains actually changed, then
        only invalidate those.

        Args:
            old_constraint: Constraint before the token was observed
            new_constraint: Constraint after the token was observed
        """
        domain_names = ["syntax", "types", "imports", "controlflow", "semantics"]

        for domain_name in domain_names:
            old_domain = getattr(old_constraint, domain_name, None)
            new_domain = getattr(new_constraint, domain_name, None)

            # Compute hashes for comparison
            try:
                old_hash = hash(old_domain) if old_domain is not None else None
                new_hash = hash(new_domain) if new_domain is not None else None
            except TypeError:
                # Unhashable - fall back to identity comparison
                old_hash = id(old_domain)
                new_hash = id(new_domain)

            # Invalidate only if changed
            if old_hash != new_hash:
                self._domain_mask_cache.pop(domain_name, None)

    def _init_lazy_evaluator(self) -> LazyConstraintEvaluator:
        """Initialize lazy evaluator with domain priorities.

        Domains are registered with priorities based on their typical
        selectivity and computation cost:
        - syntax: CRITICAL (always evaluate, fundamental constraint)
        - types: HIGH (usually highly selective, moderate cost)
        - imports: NORMAL (moderately selective)
        - controlflow: NORMAL (depends on code structure)
        - semantics: LOW (expensive, often redundant with others)

        Returns:
            Configured LazyConstraintEvaluator
        """
        evaluator = LazyConstraintEvaluator()

        # Domain priorities and estimated times (nanoseconds)
        domain_config = {
            "syntax": (EvaluationPriority.CRITICAL, 50_000),      # 50μs
            "types": (EvaluationPriority.HIGH, 500_000),          # 500μs
            "imports": (EvaluationPriority.NORMAL, 200_000),      # 200μs
            "controlflow": (EvaluationPriority.NORMAL, 100_000),  # 100μs
            "semantics": (EvaluationPriority.LOW, 1_000_000),     # 1ms
        }

        for domain_name, domain in self.domains.items():
            if domain_name in domain_config:
                priority, est_time = domain_config[domain_name]
            else:
                priority, est_time = EvaluationPriority.NORMAL, 500_000

            evaluator.register(
                domain=domain_name,
                compute_fn=domain.token_mask,
                priority=priority,
                estimated_time_ns=est_time,
            )

        return evaluator

    def _init_tiered_evaluator(self) -> TieredConstraintEvaluator:
        """Initialize tiered constraint evaluator.

        Creates a tiered evaluator that organizes domains into tiers
        based on evaluation cost and selectivity. Uses early termination
        when the mask popcount falls below the target threshold.

        Returns:
            Configured TieredConstraintEvaluator
        """
        evaluator = TieredConstraintEvaluator(target_popcount=self._tiered_target_popcount)

        # Assign domains to tiers based on typical cost/selectivity
        tier_assignments = {
            "syntax": EvaluationTier.FAST,      # ~50μs, fundamental
            "controlflow": EvaluationTier.FAST, # ~100μs, structural
            "imports": EvaluationTier.MEDIUM,   # ~200μs, moderate
            "types": EvaluationTier.SLOW,       # ~500μs, type checking
            "semantics": EvaluationTier.OPTIONAL, # ~1ms, expensive
        }

        for domain_name, domain in self.domains.items():
            tier = tier_assignments.get(domain_name, EvaluationTier.MEDIUM)
            evaluator.register(domain_name, domain.token_mask, tier)

        return evaluator

    def _init_adaptive_evaluator(self) -> AdaptiveTieredEvaluator:
        """Initialize adaptive tiered constraint evaluator.

        Creates an adaptive evaluator that dynamically optimizes domain
        evaluation based on runtime statistics:
        1. Tracks latency and selectivity per domain
        2. Reorders domains within tiers by efficiency (selectivity/latency)
        3. Skips domains with low usefulness rate (<10%)
        4. Adapts early termination threshold based on popcount history

        This is the recommended evaluation strategy for best performance.

        Returns:
            Configured AdaptiveTieredEvaluator
        """
        evaluator = AdaptiveTieredEvaluator(
            target_popcount=self._tiered_target_popcount,
            max_time_ns=self._evaluation_budget.max_time_ns,
            enable_reordering=True,
            enable_skip_prediction=True,
            enable_adaptive_threshold=True,
        )

        # Assign domains to tiers based on typical cost/selectivity
        tier_assignments = {
            "syntax": EvaluationTier.FAST,      # ~50μs, fundamental
            "controlflow": EvaluationTier.FAST, # ~100μs, structural
            "imports": EvaluationTier.MEDIUM,   # ~200μs, moderate
            "types": EvaluationTier.SLOW,       # ~500μs, type checking
            "semantics": EvaluationTier.OPTIONAL, # ~1ms, expensive
        }

        # Initial latency estimates (nanoseconds)
        initial_latencies = {
            "syntax": 50_000,       # 50μs
            "controlflow": 100_000, # 100μs
            "imports": 200_000,     # 200μs
            "types": 500_000,       # 500μs
            "semantics": 1_000_000, # 1ms
        }

        for domain_name, domain in self.domains.items():
            tier = tier_assignments.get(domain_name, EvaluationTier.MEDIUM)
            latency = initial_latencies.get(domain_name)
            evaluator.register(domain_name, domain.token_mask, tier, latency)

        return evaluator

    def _init_parallel_evaluator(self) -> ParallelDomainEvaluator:
        """Initialize parallel domain evaluator.

        Creates a parallel evaluator that runs domain mask computations
        concurrently using a thread pool with early termination when
        the fused mask becomes empty.

        Returns:
            Configured ParallelDomainEvaluator
        """
        evaluator = ParallelDomainEvaluator(max_workers=self._parallel_workers)

        for domain_name, domain in self.domains.items():
            if domain_name != "syntax":  # Syntax handled separately
                evaluator.register(domain_name, domain.token_mask)

        return evaluator

    def _init_relaxation_evaluator(self) -> RelaxationAwareEvaluator:
        """Initialize relaxation-aware constraint evaluator.

        Creates an evaluator that applies domain constraints progressively
        with relaxation support. When applying a domain mask would reduce
        the popcount below the threshold, that domain is skipped (relaxed).

        Relaxation order (most dispensable first):
        - semantics: Expensive, often redundant with other domains
        - controlflow: Can be violated gracefully
        - imports: Least critical
        - types: Usually important but can relax as last resort

        Syntax is NEVER relaxed.

        Returns:
            Configured RelaxationAwareEvaluator
        """
        evaluator = RelaxationAwareEvaluator(policy=self._relaxation_policy)

        # Register all non-syntax domains
        for domain_name, domain in self.domains.items():
            if domain_name != "syntax":  # Syntax handled separately, never relaxed
                evaluator.register(domain_name, domain.token_mask)

        return evaluator

    def _init_speculative_cache(self) -> SpeculativeMaskCache:
        """Initialize speculative mask cache.

        Creates a speculative cache that precomputes masks for likely
        next tokens while the model is computing logits. This hides
        mask computation latency.

        Returns:
            Configured SpeculativeMaskCache
        """
        def compute_mask(state: int, context: GenerationContext) -> torch.Tensor:
            """Compute fused domain mask for a state."""
            mask = context.create_mask()
            for domain_name, domain in self.domains.items():
                if domain_name == "syntax":
                    continue
                domain_constraint = getattr(self.constraint, domain_name, None)
                if domain_constraint is not None and not domain_constraint.is_top():
                    domain_mask = domain.token_mask(domain_constraint, context)
                    mask &= domain_mask
            return mask

        def state_transition(current_state: int, token: int) -> int:
            """Compute next state after token (simple hash-based)."""
            return hash((current_state, token)) & 0x7FFFFFFF

        return SpeculativeMaskCache(
            compute_fn=compute_mask,
            lookahead_depth=self._speculative_lookahead,
            max_workers=self._parallel_workers,
            max_cache_size=64,
            state_transition_fn=state_transition,
        )

    # =========================================================================
    # Cache Metrics
    # =========================================================================

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache hit rate statistics.

        Returns a dictionary with:
        - hits: Number of cache hits
        - misses: Number of cache misses
        - hit_rate_percent: Hit rate as percentage
        - evictions: Number of entries evicted

        Returns:
            Dictionary of cache statistics
        """
        cache_size = len(self._domain_mask_cache)
        # Note: _domain_mask_cache is a simple dict, not MaskCache
        # For more detailed stats, this would need MaskCache integration
        return {
            "cache_size": cache_size,
            "domains_cached": list(self._domain_mask_cache.keys()),
        }

    def get_constraint_status(self) -> Dict[str, Any]:
        """Get diagnostic information about which constraints are active.

        Returns a dictionary with:
        - domains_registered: List of registered domain names
        - constraints_active: Dict mapping domain name to whether constraint is non-TOP
        - constraint_details: Details about each constraint
        - spec_summary: Summary of constraint_spec if present

        Returns:
            Dictionary of constraint status information
        """
        status: Dict[str, Any] = {
            "domains_registered": list(self.domains.keys()),
            "constraints_active": {},
            "constraint_details": {},
        }

        # Check each domain's constraint status
        for domain_name in self.domains:
            if domain_name == "syntax":
                # Syntax is handled separately
                status["constraints_active"][domain_name] = (
                    self.syntax_grammar is not None
                )
                continue

            domain_constraint = getattr(self.constraint, domain_name, None)
            if domain_constraint is not None:
                is_active = not domain_constraint.is_top()
                status["constraints_active"][domain_name] = is_active
                status["constraint_details"][domain_name] = {
                    "is_top": domain_constraint.is_top(),
                    "repr": repr(domain_constraint)[:100],
                }

        # Add spec summary if available
        if self.constraint_spec is not None:
            spec = self.constraint_spec
            status["spec_summary"] = {
                "language": spec.language,
                "has_regex": spec.regex is not None,
                "has_ebnf": spec.ebnf is not None,
                "has_json_schema": spec.json_schema is not None,
                "has_expected_type": spec.expected_type is not None,
                "type_bindings_count": len(spec.type_bindings),
                "forbidden_imports_count": (
                    len(spec.forbidden_imports) if spec.forbidden_imports else 0
                ),
                "function_signatures_count": len(spec.function_signatures),
            }

        return status

    def get_evaluator_stats(self) -> Dict[str, Any]:
        """Get evaluator statistics based on current strategy.

        Returns statistics about domain evaluation from the active evaluator.

        Returns:
            Dictionary of evaluation statistics
        """
        stats: Dict[str, Any] = {
            "evaluation_strategy": self._evaluation_strategy.name,
        }

        if self._evaluation_strategy == EvaluationStrategy.LAZY:
            if self._lazy_evaluator is not None and hasattr(self._lazy_evaluator, 'stats'):
                stats["lazy_stats"] = self._lazy_evaluator.stats
        elif self._evaluation_strategy == EvaluationStrategy.TIERED:
            if self._tiered_evaluator is not None and hasattr(self._tiered_evaluator, 'get_stats'):
                stats["tiered_stats"] = self._tiered_evaluator.get_stats()
        elif self._evaluation_strategy == EvaluationStrategy.ADAPTIVE:
            if self._adaptive_evaluator is not None:
                stats["adaptive_stats"] = self._adaptive_evaluator.get_stats()
        elif self._evaluation_strategy == EvaluationStrategy.PARALLEL:
            if self._parallel_evaluator is not None and hasattr(self._parallel_evaluator, 'get_stats'):
                stats["parallel_stats"] = self._parallel_evaluator.get_stats()

        if self._speculative_cache is not None:
            stats["speculative_stats"] = self._speculative_cache.get_stats().__dict__

        return stats

    def get_lazy_evaluator_stats(self) -> Dict[str, Any]:
        """Get lazy evaluator statistics (deprecated, use get_evaluator_stats).

        Returns statistics about domain evaluation from the lazy evaluator.

        Returns:
            Dictionary of lazy evaluation statistics
        """
        return self.get_evaluator_stats()

    def log_cache_summary(self) -> None:
        """Log cache performance summary.

        Logs cache hit rate and performance metrics at DEBUG level.
        """
        stats = self.get_cache_stats()
        logger.debug(f"AnankeGrammar cache stats: {stats}")

    # =========================================================================
    # Speculative Decoding Support
    # =========================================================================

    def verify_draft_tokens(
        self,
        draft_tokens: List[int],
        return_first_invalid: bool = True,
    ) -> Tuple[int, Optional[UnifiedConstraint]]:
        """Verify draft tokens against constraints for speculative decoding.

        This method efficiently verifies a sequence of draft tokens from
        a draft model without permanently updating grammar state. It enables
        speculative decoding by determining how many draft tokens are valid.

        The verification is performed by:
        1. Saving current state
        2. Processing tokens one by one
        3. Checking constraint validity after each token
        4. Restoring original state after verification

        Args:
            draft_tokens: List of draft token IDs to verify
            return_first_invalid: If True, return immediately on first invalid token.
                If False, continue to find all valid tokens.

        Returns:
            Tuple of (num_valid_tokens, constraint_at_rejection):
            - num_valid_tokens: Number of tokens that pass constraint checking
            - constraint_at_rejection: Constraint state when first invalid token
              was encountered, or None if all tokens valid

        Example:
            >>> grammar = AnankeGrammar(...)
            >>> draft_tokens = [1234, 5678, 9012]  # From draft model
            >>> num_valid, rejection_constraint = grammar.verify_draft_tokens(draft_tokens)
            >>> # Accept first num_valid tokens, discard rest
        """
        if not draft_tokens:
            return 0, None

        # Save current state for restoration
        saved_constraint = self.constraint
        saved_context = GenerationContext(
            generated_text=self.context.generated_text,
            generated_tokens=self.context.generated_tokens.copy(),
            position=self.context.position,
            vocab_size=self.vocab_size,
            device=self.device,
            language=self.language,
            tokenizer=self.tokenizer,
            metadata=self.context.metadata.copy(),
            mask_pool=self._mask_pool,
        )
        saved_mask_cache = self._domain_mask_cache.copy()
        saved_finished = self.finished

        # Track syntax grammar state if available
        syntax_state_saved = False
        if self.syntax_grammar is not None and hasattr(self.syntax_grammar, 'copy'):
            # Some grammar objects support copying for state preservation
            syntax_grammar_copy = self.syntax_grammar.copy()
            syntax_state_saved = True

        num_valid = 0
        rejection_constraint: Optional[UnifiedConstraint] = None

        try:
            for i, token in enumerate(draft_tokens):
                # Check if token would be valid before accepting
                if not self._is_token_valid_for_draft(token):
                    rejection_constraint = self.constraint
                    break

                # Accept token (updates state)
                self.accept_token(token)

                # Check if constraint became unsatisfiable
                if self.finished or self.constraint.satisfiability() == Satisfiability.UNSAT:
                    rejection_constraint = self.constraint
                    break

                num_valid += 1

                if return_first_invalid and self.finished:
                    break

        finally:
            # Restore original state
            self.constraint = saved_constraint
            self.context = saved_context
            self._domain_mask_cache = saved_mask_cache
            self.finished = saved_finished

            # Restore syntax grammar state if we saved it
            if syntax_state_saved:
                self.syntax_grammar = syntax_grammar_copy

        return num_valid, rejection_constraint

    def _is_token_valid_for_draft(self, token: int) -> bool:
        """Check if a token is valid under current constraints.

        This is a lightweight check for speculative decoding that doesn't
        compute full masks. It checks if the token is allowed by each
        domain's constraint.

        Args:
            token: Token ID to check

        Returns:
            True if token is valid under all constraints
        """
        # Check syntax grammar first (if available)
        if self.syntax_grammar is not None:
            # Get syntax mask
            try:
                # Allocate a small mask for single-token check
                mask_size = (self.vocab_size + 31) // 32
                vocab_mask = torch.zeros(
                    (1, mask_size), dtype=torch.int32, device=self.device
                )
                self.syntax_grammar.fill_vocab_mask(vocab_mask, 0)

                # Check if token is allowed
                word_idx = token // 32
                bit_idx = token % 32
                if word_idx < mask_size:
                    if not (vocab_mask[0, word_idx] & (1 << bit_idx)):
                        return False
            except Exception as e:
                logger.debug(f"Syntax check failed for token {token}: {e}")
                # On error, assume valid (soundness)
                pass

        # Check each domain constraint
        for domain_name, domain in self.domains.items():
            if domain_name == "syntax":
                continue  # Already checked above

            # Get domain constraint (safely handle unknown domains)
            domain_constraint = getattr(self.constraint, domain_name, None)
            if domain_constraint is None:
                # Unknown domain - compute mask directly
                try:
                    mask = domain.token_mask(None, self.context)
                    if mask is not None and token < len(mask) and not mask[token]:
                        return False
                except Exception as e:
                    logger.debug(f"Domain {domain_name} mask computation failed: {e}")
                continue

            if domain_constraint.is_top():
                continue  # No constraint

            try:
                # Get domain mask
                mask = domain.token_mask(domain_constraint, self.context)
                if mask is not None and token < len(mask) and not mask[token]:
                    return False
            except Exception as e:
                logger.debug(f"Domain {domain_name} check failed for token {token}: {e}")
                # On error, assume valid (soundness: don't block valid tokens)
                pass

        return True

    def verify_draft_batch(
        self,
        draft_sequences: List[List[int]],
    ) -> List[Tuple[int, Optional[UnifiedConstraint]]]:
        """Verify multiple draft sequences in batch.

        Convenience method for verifying multiple draft sequences.
        Currently processes sequentially; future optimization could
        parallelize across sequences.

        Args:
            draft_sequences: List of draft token sequences

        Returns:
            List of (num_valid, rejection_constraint) tuples
        """
        results = []
        for draft_tokens in draft_sequences:
            result = self.verify_draft_tokens(draft_tokens)
            results.append(result)
        return results

    def get_speculative_stats(self) -> Dict[str, Any]:
        """Get statistics about speculative decoding verification.

        Returns:
            Dictionary with speculative decoding statistics
        """
        return {
            "supported": True,
            "syntax_grammar_available": self.syntax_grammar is not None,
            "active_domains": list(self.domains.keys()),
        }
