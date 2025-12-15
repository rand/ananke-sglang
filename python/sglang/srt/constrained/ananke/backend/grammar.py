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
    from ..core.domain import ConstraintDomain, GenerationContext
    from ..core.unified import (
        UNIFIED_TOP,
        UnifiedConstraint,
    )
except ImportError:
    from core.checkpoint import (
        Checkpoint,
        CheckpointManager,
        UnifiedCheckpoint,
    )
    from core.constraint import Satisfiability
    from core.domain import ConstraintDomain, GenerationContext
    from core.unified import (
        UNIFIED_TOP,
        UnifiedConstraint,
    )

if TYPE_CHECKING:
    from sglang.srt.constrained.llguidance_backend import GuidanceGrammar

logger = logging.getLogger(__name__)


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
        syntax_grammar: Optional[GuidanceGrammar],
        domains: Dict[str, ConstraintDomain],
        constraint: UnifiedConstraint = UNIFIED_TOP,
        vocab_size: int = 0,
        device: str = "cuda",
        tokenizer: Optional[Any] = None,
        language: str = "python",
        max_rollback_tokens: int = 200,
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
        """
        super().__init__()
        self.syntax_grammar = syntax_grammar
        self.domains = domains
        self.constraint = constraint
        self.vocab_size = vocab_size
        self.device = device
        self.tokenizer = tokenizer
        self.language = language

        # Generation context
        self.context = GenerationContext(
            vocab_size=vocab_size,
            device=device,
            language=language,
            tokenizer=tokenizer,
        )

        # Checkpoint management for rollback
        self.checkpoint_manager = CheckpointManager(
            max_checkpoints=max_rollback_tokens
        )

        # Cache for computed domain masks
        self._domain_mask_cache: Dict[str, torch.Tensor] = {}

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

        # Clear mask cache
        self._domain_mask_cache.clear()

        # Reset finished flag
        self.finished = False

    def is_terminated(self) -> bool:
        """Check if grammar has reached a terminal state."""
        if self.syntax_grammar is not None:
            return self.syntax_grammar.is_terminated()
        return self.finished

    def fill_vocab_mask(self, vocab_mask: torch.Tensor, idx: int) -> None:
        """Fill vocabulary mask with valid tokens for this request.

        This method:
        1. Delegates to syntax grammar for CFG-based masking
        2. Computes additional domain masks (types, imports, etc.)
        3. Applies additional masks via bitwise AND

        Args:
            vocab_mask: Bitmask tensor to fill [batch_size, mask_size]
            idx: Index into the batch for this request
        """
        # First, delegate to syntax grammar
        if self.syntax_grammar is not None:
            self.syntax_grammar.fill_vocab_mask(vocab_mask, idx)
            if self.syntax_grammar.finished:
                self.finished = True
                return

        # Compute and apply additional domain masks
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
        mask_size = (vocab_size + 31) // 32
        return torch.zeros(
            (batch_size, mask_size), dtype=torch.int32, device=device
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

    def copy(self) -> AnankeGrammar:
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
        )

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
        """Create a checkpoint if conditions are met."""
        # For now, checkpoint every token (can be optimized later)
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
