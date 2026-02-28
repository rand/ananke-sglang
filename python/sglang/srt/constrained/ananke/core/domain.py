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
"""Constraint domain abstraction for the Ananke compositional constraint system.

This module defines the ConstraintDomain abstract base class and GenerationContext
dataclass that enable token-level constraint checking across different domains
(syntax, types, imports, control flow, semantics).

Each domain maintains its own constraint state and provides:
1. Token mask computation - which tokens satisfy current constraints
2. Token observation - updating constraint state when a token is generated
3. Checkpointing - saving/restoring state for rollback

References:
    - Hazel: "Statically Contextualizing LLMs with Typed Holes" (OOPSLA 2024)
    - Hazel: "Live Functional Programming with Typed Holes" (ICFP 2019)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Generator, Generic, List, Optional, Tuple, TypeVar

import torch

from .constraint import Constraint, Satisfiability


class MaskPool:
    """Pre-allocated tensor pool to avoid per-token CUDA allocations.

    Token mask computation is called on every token. Creating new tensors
    each time adds allocation overhead (~100μs on GPU). This pool maintains
    a fixed set of pre-allocated tensors that can be reused.

    Usage:
        pool = MaskPool(vocab_size=32000, device="cuda", pool_size=8)
        mask, handle = pool.acquire()
        # use mask...
        pool.release(handle)

    Thread Safety:
        This class is NOT thread-safe. Each thread should have its own pool,
        or external synchronization should be used.
    """

    def __init__(self, vocab_size: int, device: str, pool_size: int = 8):
        """Initialize the mask pool.

        Args:
            vocab_size: Size of vocabulary (mask dimension)
            device: PyTorch device ("cpu", "cuda", etc.)
            pool_size: Number of tensors to pre-allocate
        """
        self._vocab_size = vocab_size
        self._device = device
        self._pool: List[torch.Tensor] = [
            torch.ones(vocab_size, dtype=torch.bool, device=device)
            for _ in range(pool_size)
        ]
        self._available: List[int] = list(range(pool_size))

    def acquire(self, fill_value: bool = True) -> Tuple[torch.Tensor, int]:
        """Acquire a mask tensor from the pool.

        Args:
            fill_value: Initial fill value (True for all-valid, False for all-blocked)

        Returns:
            Tuple of (tensor, handle). Use handle with release().
            Handle is -1 if a fallback allocation was used.
        """
        if not self._available:
            # Pool exhausted - allocate new tensor (rare case)
            return (
                torch.full(
                    (self._vocab_size,),
                    fill_value,
                    dtype=torch.bool,
                    device=self._device,
                ),
                -1,
            )

        handle = self._available.pop()
        mask = self._pool[handle]
        mask.fill_(fill_value)
        return mask, handle

    def release(self, handle: int) -> None:
        """Return a mask tensor to the pool.

        Args:
            handle: Handle from acquire(). Ignored if -1 (fallback allocation).
        """
        if handle >= 0 and handle not in self._available:
            self._available.append(handle)

    @property
    def available_count(self) -> int:
        """Number of tensors currently available in the pool."""
        return len(self._available)

    @property
    def pool_size(self) -> int:
        """Total size of the pool."""
        return len(self._pool)

if TYPE_CHECKING:
    from .checkpoint import Checkpoint


# Type variable for concrete constraint types
C = TypeVar("C", bound=Constraint)


@dataclass
class GenerationContext:
    """Context information available during token generation.

    This dataclass captures the current state of generation, providing
    domains with the information they need to compute valid token masks
    and update their constraints.

    Following the ChatLSP protocol from Hazel, the context provides rich
    type information that can guide generation toward valid completions.

    Attributes:
        generated_text: The text generated so far
        generated_tokens: List of token IDs generated so far
        position: Current position in the generated sequence
        vocab_size: Size of the tokenizer vocabulary
        device: PyTorch device for tensor operations
        language: Programming language being generated (e.g., "python", "rust")
        tokenizer: Reference to the tokenizer (for decode operations)
        metadata: Additional domain-specific metadata
        mask_pool: Optional pre-allocated tensor pool for mask computation
    """

    generated_text: str = ""
    generated_tokens: List[int] = field(default_factory=list)
    position: int = 0
    vocab_size: int = 0
    device: str = "cuda"
    language: str = "python"
    tokenizer: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    mask_pool: Optional[MaskPool] = None

    def extend(self, token_id: int, token_text: str) -> GenerationContext:
        """Create a new context with an additional token.

        This method is immutable - it returns a new context rather than
        modifying the existing one, enabling safe checkpointing and rollback.

        Args:
            token_id: The token ID that was generated
            token_text: The decoded text of the token

        Returns:
            A new GenerationContext with the token appended
        """
        return GenerationContext(
            generated_text=self.generated_text + token_text,
            generated_tokens=self.generated_tokens + [token_id],
            position=self.position + 1,
            vocab_size=self.vocab_size,
            device=self.device,
            language=self.language,
            tokenizer=self.tokenizer,
            metadata=self.metadata.copy(),
            mask_pool=self.mask_pool,  # Share the pool across contexts
        )

    def acquire_mask(self, fill_value: bool = True) -> Tuple[torch.Tensor, int]:
        """Acquire a mask tensor, using pool if available.

        Args:
            fill_value: Initial fill value for the mask

        Returns:
            Tuple of (tensor, handle). Handle is -1 if no pool or fallback.
        """
        if self.mask_pool is not None:
            return self.mask_pool.acquire(fill_value)

        # No pool - allocate directly
        return (
            torch.full(
                (self.vocab_size,),
                fill_value,
                dtype=torch.bool,
                device=self.device,
            ),
            -1,
        )

    def release_mask(self, handle: int) -> None:
        """Release a mask tensor back to the pool.

        Args:
            handle: Handle from acquire_mask(). Ignored if -1.
        """
        if self.mask_pool is not None and handle >= 0:
            self.mask_pool.release(handle)

    @contextmanager
    def borrowed_mask(self, fill_value: bool = True) -> Generator[torch.Tensor, None, None]:
        """Context manager for borrowing a mask from the pool.

        Automatically releases the mask when the context exits, ensuring
        no memory leaks even if exceptions occur.

        Args:
            fill_value: Initial fill value (True for all-valid, False for all-blocked)

        Yields:
            A mask tensor that will be automatically returned to the pool.

        Example:
            with context.borrowed_mask() as mask:
                mask[blocked_tokens] = False
                return mask.clone()  # Return a copy if needed beyond the context
        """
        mask, handle = self.acquire_mask(fill_value)
        try:
            yield mask
        finally:
            self.release_mask(handle)

    def create_mask(self, fill_value: bool = True) -> torch.Tensor:
        """Create a mask tensor, using pool if available.

        This is a convenience method that acquires a mask and returns it
        without tracking the handle. Use this when you need to return
        the mask and can't use the context manager.

        NOTE: When using this method, the mask will be returned to the pool
        when the pool runs out of tensors and allocates new ones. This is
        safe but may cause additional allocations if called frequently.

        For best performance, prefer borrowed_mask() when possible.

        Args:
            fill_value: Initial fill value

        Returns:
            A mask tensor
        """
        mask, _ = self.acquire_mask(fill_value)
        return mask


class ConstraintDomain(ABC, Generic[C]):
    """Abstract base class for constraint domains in Ananke.

    A constraint domain represents a particular aspect of code validity
    (syntax, types, imports, control flow, semantics) and provides:

    1. Token mask computation: Given current constraints and context,
       determine which tokens can be generated while maintaining validity.

    2. Token observation: When a token is generated, update the domain's
       internal state and constraint to reflect the new information.

    3. State management: Support checkpointing and rollback for speculative
       decoding and backtracking.

    The domain architecture follows Hazel's principle of "totality" - every
    partial program state should have a well-defined constraint, even if
    that constraint indicates an error.

    Type Parameters:
        C: The constraint type for this domain (must extend Constraint)

    Example:
        >>> class TypeDomain(ConstraintDomain[TypeConstraint]):
        ...     def token_mask(self, constraint, context):
        ...         # Return boolean mask of valid tokens
        ...         ...
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """The unique name of this domain (e.g., 'syntax', 'types')."""
        raise NotImplementedError

    @property
    @abstractmethod
    def top(self) -> C:
        """The top (unconstrained) element for this domain."""
        raise NotImplementedError

    @property
    @abstractmethod
    def bottom(self) -> C:
        """The bottom (unsatisfiable) element for this domain."""
        raise NotImplementedError

    @abstractmethod
    def token_mask(
        self,
        constraint: C,
        context: GenerationContext,
    ) -> torch.Tensor:
        """Compute a boolean mask of valid tokens given current constraints.

        This is the core operation for constrained decoding - it determines
        which tokens can be generated while satisfying the domain's constraints.

        The mask should be conservative: if there's uncertainty about whether
        a token is valid, it should be marked as valid (True) to avoid
        over-constraining generation.

        Performance target: <500μs for type domain, ~50μs for syntax domain.

        Args:
            constraint: The current constraint for this domain
            context: The generation context with text/tokens so far

        Returns:
            A boolean tensor of shape (vocab_size,) where True indicates
            the token is valid according to this domain's constraints
        """
        raise NotImplementedError

    @abstractmethod
    def observe_token(
        self,
        constraint: C,
        token_id: int,
        context: GenerationContext,
    ) -> C:
        """Update the constraint after observing a generated token.

        This method is called after a token is selected and added to the
        generation. It updates the domain's constraint to reflect the new
        information. The returned constraint may be:

        - Refined (more restrictive) based on new information
        - Unchanged if the token doesn't affect this domain
        - BOTTOM if the token creates an unsatisfiable state

        Following Hazel's incremental typing approach, this should perform
        minimal recomputation - only updating what changed.

        Args:
            constraint: The current constraint for this domain
            token_id: The token ID that was generated
            context: The generation context BEFORE adding this token

        Returns:
            The updated constraint after observing the token
        """
        raise NotImplementedError

    @abstractmethod
    def checkpoint(self) -> Checkpoint:
        """Create a checkpoint of the current domain state.

        Checkpoints enable rollback for speculative decoding and
        backtracking when generation reaches an unsatisfiable state.

        The checkpoint should capture all mutable state needed to
        restore the domain to its current state.

        Returns:
            A Checkpoint object that can be passed to restore()
        """
        raise NotImplementedError

    @abstractmethod
    def restore(self, checkpoint: Checkpoint) -> None:
        """Restore domain state from a checkpoint.

        This method is called when rolling back generation to a
        previous state, typically because the current path led to
        an unsatisfiable constraint.

        Args:
            checkpoint: A checkpoint previously created by checkpoint()
        """
        raise NotImplementedError

    def satisfiability(self, constraint: C) -> Satisfiability:
        """Check the satisfiability of a constraint.

        Default implementation delegates to the constraint's own method.
        Domains may override for more efficient or domain-specific checks.

        Args:
            constraint: The constraint to check

        Returns:
            The satisfiability status
        """
        return constraint.satisfiability()

    def propagate_from(
        self,
        source_domain: str,
        source_constraint: Constraint,
        current_constraint: C,
        context: GenerationContext,
    ) -> C:
        """Handle constraint propagation from another domain.

        Cross-domain propagation allows constraints in one domain to
        influence constraints in another. For example:
        - Type constraints can restrict valid syntactic forms
        - Import constraints affect available types

        Default implementation returns the current constraint unchanged.
        Domains should override to handle relevant propagation edges.

        Args:
            source_domain: Name of the domain sending the constraint
            source_constraint: The constraint from the source domain
            current_constraint: This domain's current constraint
            context: The generation context

        Returns:
            The updated constraint after propagation
        """
        return current_constraint


class PassthroughDomain(ConstraintDomain[C]):
    """A domain that imposes no constraints (always returns all-True mask).

    This is useful for:
    - Placeholder domains during incremental development
    - Domains that are disabled by configuration
    - Testing other components in isolation

    Type Parameters:
        C: The constraint type (for compatibility with domain interfaces)
    """

    def __init__(self, domain_name: str, top_constraint: C, bottom_constraint: C):
        """Initialize a passthrough domain.

        Args:
            domain_name: Name to identify this domain
            top_constraint: The TOP element for the constraint type
            bottom_constraint: The BOTTOM element for the constraint type
        """
        self._name = domain_name
        self._top = top_constraint
        self._bottom = bottom_constraint

    @property
    def name(self) -> str:
        return self._name

    @property
    def top(self) -> C:
        return self._top

    @property
    def bottom(self) -> C:
        return self._bottom

    def token_mask(
        self,
        constraint: C,
        context: GenerationContext,
    ) -> torch.Tensor:
        """Return all-True mask (no constraints)."""
        return context.create_mask(fill_value=True)

    def observe_token(
        self,
        constraint: C,
        token_id: int,
        context: GenerationContext,
    ) -> C:
        """Return constraint unchanged."""
        return constraint

    def checkpoint(self) -> Checkpoint:
        """Return empty checkpoint."""
        from .checkpoint import Checkpoint

        return Checkpoint(domain_name=self._name, state={})

    def restore(self, checkpoint: Checkpoint) -> None:
        """No-op restore."""
        pass
