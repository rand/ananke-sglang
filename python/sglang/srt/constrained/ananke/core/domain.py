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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Generic, List, Optional, TypeVar

import torch

from .constraint import Constraint, Satisfiability

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
    """

    generated_text: str = ""
    generated_tokens: List[int] = field(default_factory=list)
    position: int = 0
    vocab_size: int = 0
    device: str = "cuda"
    language: str = "python"
    tokenizer: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

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
        )


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
        return torch.ones(context.vocab_size, dtype=torch.bool, device=context.device)

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
