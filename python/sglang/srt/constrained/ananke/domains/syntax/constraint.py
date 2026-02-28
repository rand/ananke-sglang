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
"""Syntax constraint representing grammar-based structural requirements.

This module defines SyntaxConstraint, which wraps a grammar specification
(JSON schema, regex, EBNF, or structural tag) and delegates actual parsing
to llguidance for efficient (~50μs/token) mask computation.

The syntax constraint is primarily a container for the grammar specification,
with the actual state management handled by the underlying llguidance grammar
object. The constraint supports the semilattice interface required by Ananke's
constraint composition system.

References:
    - llguidance: Dynamic mask computation with lazy automata
    - XGrammar: Grammar intersection and rollback semantics
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

# Support both relative imports (when used as subpackage) and absolute imports (standalone testing)
try:
    from ...core.constraint import Constraint, Satisfiability
except ImportError:
    from core.constraint import Constraint, Satisfiability


class GrammarType(Enum):
    """Type of grammar specification."""

    JSON_SCHEMA = auto()
    REGEX = auto()
    EBNF = auto()
    STRUCTURAL_TAG = auto()


@dataclass(frozen=True, slots=True)
class SyntaxConstraint(Constraint["SyntaxConstraint"]):
    """Constraint representing a grammar specification.

    SyntaxConstraint wraps a grammar string (JSON schema, regex, EBNF, or
    structural tag) and tracks the current parsing state via a state hash.
    Actual parsing is delegated to llguidance.

    The constraint forms a semilattice where:
    - TOP (SYNTAX_TOP) represents no syntactic constraint
    - BOTTOM (SYNTAX_BOTTOM) represents unsatisfiable syntax
    - meet() of two constraints returns BOTTOM if they're incompatible

    Note: Full grammar intersection is undecidable for CFGs, so meet()
    conservatively returns BOTTOM if grammars differ. In practice, Ananke
    uses a single grammar per generation, so this is rarely needed.

    Attributes:
        grammar_type: Type of grammar (JSON, regex, EBNF, structural)
        grammar_string: The grammar specification string
        state_hash: Hash of the current parsing state (for checkpointing)
        is_complete: Whether the grammar has been fully matched
        _is_top: True if this is the TOP element
        _is_bottom: True if this is the BOTTOM element
    """

    grammar_type: Optional[GrammarType] = None
    grammar_string: Optional[str] = None
    state_hash: int = 0
    is_complete: bool = False
    _is_top: bool = False
    _is_bottom: bool = False

    def meet(self, other: SyntaxConstraint) -> SyntaxConstraint:
        """Compute the meet of two syntax constraints.

        Grammar intersection is undecidable for CFGs, so we use a
        conservative approximation:
        - If either is BOTTOM, return BOTTOM
        - If either is TOP, return the other
        - If grammars are identical, return self (with merged state)
        - Otherwise, return BOTTOM (conservative)

        Args:
            other: The constraint to combine with

        Returns:
            The conjunction of both constraints
        """
        # Handle BOTTOM (annihilation)
        if self._is_bottom or other._is_bottom:
            return SYNTAX_BOTTOM

        # Handle TOP (identity)
        if self._is_top:
            return other
        if other._is_top:
            return self

        # Same grammar - merge the constraints
        # Note: We compare grammar strings for equality
        if (
            self.grammar_type == other.grammar_type
            and self.grammar_string == other.grammar_string
        ):
            # For commutativity: return a canonical merged constraint
            # Take the higher state_hash (more advanced state)
            # and is_complete is True only if both are complete
            merged_hash = max(self.state_hash, other.state_hash)
            merged_complete = self.is_complete and other.is_complete

            return SyntaxConstraint(
                grammar_type=self.grammar_type,
                grammar_string=self.grammar_string,
                state_hash=merged_hash,
                is_complete=merged_complete,
            )

        # Different grammars - conservatively return BOTTOM
        # (Full intersection would require grammar analysis)
        return SYNTAX_BOTTOM

    def is_top(self) -> bool:
        """Check if this is the TOP (unconstrained) element."""
        return self._is_top

    def is_bottom(self) -> bool:
        """Check if this is the BOTTOM (unsatisfiable) element."""
        return self._is_bottom

    def satisfiability(self) -> Satisfiability:
        """Determine the satisfiability status.

        Returns:
            SAT if the constraint can be satisfied
            UNSAT if the constraint cannot be satisfied
        """
        if self._is_bottom:
            return Satisfiability.UNSAT
        return Satisfiability.SAT

    def with_state_hash(self, state_hash: int) -> SyntaxConstraint:
        """Create a new constraint with updated state hash."""
        return SyntaxConstraint(
            grammar_type=self.grammar_type,
            grammar_string=self.grammar_string,
            state_hash=state_hash,
            is_complete=self.is_complete,
            _is_top=self._is_top,
            _is_bottom=self._is_bottom,
        )

    def with_complete(self, is_complete: bool) -> SyntaxConstraint:
        """Create a new constraint marked as complete or incomplete."""
        return SyntaxConstraint(
            grammar_type=self.grammar_type,
            grammar_string=self.grammar_string,
            state_hash=self.state_hash,
            is_complete=is_complete,
            _is_top=self._is_top,
            _is_bottom=self._is_bottom,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SyntaxConstraint):
            return NotImplemented
        return (
            self._is_top == other._is_top
            and self._is_bottom == other._is_bottom
            and self.grammar_type == other.grammar_type
            and self.grammar_string == other.grammar_string
            and self.state_hash == other.state_hash
            and self.is_complete == other.is_complete
        )

    def __hash__(self) -> int:
        if self._is_top:
            return hash("SYNTAX_TOP")
        if self._is_bottom:
            return hash("SYNTAX_BOTTOM")
        return hash((self.grammar_type, self.grammar_string, self.state_hash))

    def __repr__(self) -> str:
        if self._is_top:
            return "SYNTAX_TOP"
        if self._is_bottom:
            return "SYNTAX_BOTTOM"
        type_name = self.grammar_type.name if self.grammar_type else "NONE"
        return f"SyntaxConstraint({type_name}, hash={self.state_hash})"


# Singleton instances
SYNTAX_TOP = SyntaxConstraint(_is_top=True)
SYNTAX_BOTTOM = SyntaxConstraint(_is_bottom=True)


def syntax_from_json_schema(schema: str) -> SyntaxConstraint:
    """Create a syntax constraint from a JSON schema."""
    return SyntaxConstraint(
        grammar_type=GrammarType.JSON_SCHEMA,
        grammar_string=schema,
    )


def syntax_from_regex(pattern: str) -> SyntaxConstraint:
    """Create a syntax constraint from a regex pattern."""
    return SyntaxConstraint(
        grammar_type=GrammarType.REGEX,
        grammar_string=pattern,
    )


def syntax_from_ebnf(grammar: str) -> SyntaxConstraint:
    """Create a syntax constraint from an EBNF grammar."""
    return SyntaxConstraint(
        grammar_type=GrammarType.EBNF,
        grammar_string=grammar,
    )


def syntax_from_structural_tag(tag: str) -> SyntaxConstraint:
    """Create a syntax constraint from a structural tag."""
    return SyntaxConstraint(
        grammar_type=GrammarType.STRUCTURAL_TAG,
        grammar_string=tag,
    )
