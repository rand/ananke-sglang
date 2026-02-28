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
"""Syntax domain for grammar-based structural constraints.

This module implements SyntaxDomain, which wraps llguidance for efficient
CFG parsing and constraint checking. The domain delegates token mask
computation to llguidance (~50μs/token) while integrating with Ananke's
constraint composition system.

The syntax domain is special in that it:
1. Wraps an existing llguidance grammar object
2. Delegates mask computation entirely to llguidance
3. Tracks constraint state via hashing for checkpointing

References:
    - llguidance: Dynamic mask computation with lazy automata
    - XGrammar: Grammar rollback semantics
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

import torch

# Support both relative imports (when used as subpackage) and absolute imports (standalone testing)
try:
    from ...core.checkpoint import Checkpoint
    from ...core.constraint import Satisfiability
    from ...core.domain import ConstraintDomain, GenerationContext
except ImportError:
    from core.checkpoint import Checkpoint
    from core.constraint import Satisfiability
    from core.domain import ConstraintDomain, GenerationContext

from .constraint import (
    SYNTAX_BOTTOM,
    SYNTAX_TOP,
    GrammarType,
    SyntaxConstraint,
)

if TYPE_CHECKING:
    from sglang.srt.constrained.base_grammar_backend import BaseGrammarObject

logger = logging.getLogger(__name__)


class SyntaxDomain(ConstraintDomain[SyntaxConstraint]):
    """Domain for grammar-based structural constraints.

    SyntaxDomain wraps an llguidance grammar object, delegating all
    parsing and mask computation to the highly optimized llguidance
    implementation (~50μs/token).

    The domain's role is to:
    1. Integrate llguidance with Ananke's constraint algebra
    2. Track state for checkpointing and rollback
    3. Provide consistent interface with other domains

    Note: Unlike other domains that compute their own token masks,
    SyntaxDomain relies on the underlying grammar object. The token_mask()
    method returns an all-True mask since actual masking is done by
    AnankeGrammar.fill_vocab_mask() which delegates to llguidance.

    Attributes:
        grammar_object: The wrapped llguidance grammar object
        grammar_type: Type of grammar (JSON, regex, EBNF, structural)
        grammar_string: The grammar specification string
        _state_counter: Counter for generating unique state hashes
    """

    def __init__(
        self,
        grammar_object: Optional[BaseGrammarObject] = None,
        grammar_type: Optional[GrammarType] = None,
        grammar_string: Optional[str] = None,
    ):
        """Initialize the syntax domain.

        Args:
            grammar_object: The wrapped llguidance grammar object
            grammar_type: Type of grammar (for constraint creation)
            grammar_string: The grammar specification string
        """
        self._grammar_object = grammar_object
        self._grammar_type = grammar_type
        self._grammar_string = grammar_string
        self._state_counter = 0

        # Create initial constraint
        if grammar_type and grammar_string:
            self._current_constraint = SyntaxConstraint(
                grammar_type=grammar_type,
                grammar_string=grammar_string,
                state_hash=self._state_counter,
            )
        else:
            self._current_constraint = SYNTAX_TOP

    @property
    def name(self) -> str:
        """The unique name of this domain."""
        return "syntax"

    @property
    def top(self) -> SyntaxConstraint:
        """The top (unconstrained) element for this domain."""
        return SYNTAX_TOP

    @property
    def bottom(self) -> SyntaxConstraint:
        """The bottom (unsatisfiable) element for this domain."""
        return SYNTAX_BOTTOM

    @property
    def grammar_object(self) -> Optional[BaseGrammarObject]:
        """The wrapped llguidance grammar object."""
        return self._grammar_object

    def token_mask(
        self,
        constraint: SyntaxConstraint,
        context: GenerationContext,
    ) -> torch.Tensor:
        """Compute a boolean mask of valid tokens.

        For the syntax domain, we return an all-True mask since actual
        masking is handled by the wrapped grammar object via
        AnankeGrammar.fill_vocab_mask().

        This design choice avoids duplicating the llguidance bitmask
        computation and keeps the mask fusion logic efficient.

        Args:
            constraint: The current syntax constraint
            context: The generation context

        Returns:
            All-True boolean mask (actual masking via grammar object)
        """
        # Return all-True mask - actual masking is done by grammar object
        return context.create_mask(fill_value=True)

    def observe_token(
        self,
        constraint: SyntaxConstraint,
        token_id: int,
        context: GenerationContext,
    ) -> SyntaxConstraint:
        """Update the constraint after observing a generated token.

        The actual parsing state is maintained by the wrapped grammar
        object. This method updates the constraint's state hash to
        track the logical state for checkpointing.

        If the grammar becomes unsatisfiable (e.g., syntax error),
        returns SYNTAX_BOTTOM.

        Args:
            constraint: The current syntax constraint
            token_id: The token ID that was generated
            context: The generation context

        Returns:
            The updated constraint
        """
        if constraint.is_bottom():
            return SYNTAX_BOTTOM

        if constraint.is_top():
            # No grammar constraint - nothing to update
            return SYNTAX_TOP

        # Check if grammar object indicates unsatisfiable state
        if self._grammar_object is not None:
            if self._grammar_object.finished:
                # Grammar reached terminal state
                return constraint.with_complete(True)

        # Update state counter for checkpointing
        self._state_counter += 1

        return constraint.with_state_hash(self._state_counter)

    def checkpoint(self) -> Checkpoint:
        """Create a checkpoint of the current domain state.

        Returns:
            A Checkpoint capturing the current state
        """
        return Checkpoint(
            domain_name=self.name,
            state={
                "state_counter": self._state_counter,
                "constraint": self._current_constraint,
            },
            position=self._state_counter,
        )

    def restore(self, checkpoint: Checkpoint) -> None:
        """Restore domain state from a checkpoint.

        Args:
            checkpoint: A checkpoint previously created by checkpoint()
        """
        self._state_counter = checkpoint.state.get("state_counter", 0)
        self._current_constraint = checkpoint.state.get(
            "constraint", self._current_constraint
        )

    def satisfiability(self, constraint: SyntaxConstraint) -> Satisfiability:
        """Check the satisfiability of a syntax constraint.

        Args:
            constraint: The constraint to check

        Returns:
            The satisfiability status
        """
        return constraint.satisfiability()

    def update_grammar_object(self, grammar_object: BaseGrammarObject) -> None:
        """Update the wrapped grammar object.

        This is called when a new grammar is created (e.g., via dispatch).

        Args:
            grammar_object: The new grammar object
        """
        self._grammar_object = grammar_object
        self._state_counter = 0

    def create_constraint_for_grammar(
        self,
        grammar_type: GrammarType,
        grammar_string: str,
    ) -> SyntaxConstraint:
        """Create a syntax constraint for a grammar specification.

        Args:
            grammar_type: Type of grammar
            grammar_string: The grammar specification string

        Returns:
            A new SyntaxConstraint
        """
        self._grammar_type = grammar_type
        self._grammar_string = grammar_string
        self._state_counter = 0

        self._current_constraint = SyntaxConstraint(
            grammar_type=grammar_type,
            grammar_string=grammar_string,
            state_hash=0,
        )

        return self._current_constraint
