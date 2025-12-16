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
"""Incremental parse state caching for efficient rollback and re-parsing.

This module provides caching and optimization for incremental parsing:
- Position-based checkpoint caching for fast rollback
- Minimal diff computation for targeted re-parsing
- Integration with tree-sitter's O(log n) incremental updates

Key insight: During constrained generation, we often need to roll back
to previous states when tokens are rejected. Caching parse states at
strategic positions (e.g., every N tokens) enables O(1) rollback.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Generic, TypeVar
import time
import copy

from parsing.base import IncrementalParser, ParseResult, ParseState


@dataclass
class ParseCheckpoint:
    """A cached parse state at a specific position.

    Attributes:
        position: Character position in source
        token_count: Number of tokens processed
        parser_state: Opaque checkpoint from the parser
        source_hash: Hash of source up to this position
        result: The parse result at this position
        created_at_ns: When this checkpoint was created
    """

    position: int
    token_count: int
    parser_state: Any
    source_hash: int
    result: ParseResult
    created_at_ns: int = 0


@dataclass
class CacheStats:
    """Statistics for the incremental parse cache.

    Attributes:
        checkpoints_created: Number of checkpoints created
        checkpoints_used: Number of times a checkpoint was used for restore
        checkpoints_evicted: Number of checkpoints evicted
        avg_restore_time_ns: Average time to restore from checkpoint
        avg_parse_time_ns: Average time for incremental parse
    """

    checkpoints_created: int = 0
    checkpoints_used: int = 0
    checkpoints_evicted: int = 0
    avg_restore_time_ns: float = 0.0
    avg_parse_time_ns: float = 0.0

    @property
    def restore_efficiency(self) -> float:
        """Ratio of checkpoint uses to creations."""
        if self.checkpoints_created == 0:
            return 0.0
        return self.checkpoints_used / self.checkpoints_created


class IncrementalParseCache:
    """Caches parse states for efficient rollback during generation.

    During constrained generation, tokens may be rejected by constraint
    checkers, requiring rollback to a previous state. This cache maintains
    checkpoints at strategic positions to enable fast rollback.

    Checkpointing strategy:
    - Create checkpoint every N tokens (configurable)
    - Create checkpoint at significant syntax boundaries
    - LRU eviction when cache is full

    Usage:
        cache = IncrementalParseCache(parser, checkpoint_interval=10)

        # Normal operation - cache handles checkpointing
        result = cache.extend_with_token(token)

        # Rollback to a position
        cache.rollback_to_position(100)  # Restores nearest checkpoint

        # Rollback by N tokens
        cache.rollback_tokens(5)  # Go back 5 tokens

    Thread Safety:
        This class is NOT thread-safe. Each thread should have its own cache.
    """

    def __init__(
        self,
        parser: IncrementalParser,
        checkpoint_interval: int = 10,
        max_checkpoints: int = 50,
        checkpoint_at_boundaries: bool = True,
    ) -> None:
        """Initialize the incremental parse cache.

        Args:
            parser: The underlying incremental parser
            checkpoint_interval: Create checkpoint every N tokens
            max_checkpoints: Maximum checkpoints to cache
            checkpoint_at_boundaries: Create checkpoints at syntax boundaries
        """
        self._parser = parser
        self._checkpoint_interval = checkpoint_interval
        self._max_checkpoints = max_checkpoints
        self._checkpoint_at_boundaries = checkpoint_at_boundaries

        # Checkpoints ordered by position
        self._checkpoints: OrderedDict[int, ParseCheckpoint] = OrderedDict()
        self._token_count = 0
        self._stats = CacheStats()

        # Syntax boundary detection
        self._boundary_tokens = {
            "{", "}", "(", ")", "[", "]",  # Delimiters
            ";", "\n",  # Statement endings
            "def ", "class ", "fn ", "func ", "function ",  # Definitions
        }

    @property
    def parser(self) -> IncrementalParser:
        """Get the underlying parser."""
        return self._parser

    @property
    def position(self) -> int:
        """Current position in source."""
        return self._parser.current_position

    @property
    def token_count(self) -> int:
        """Number of tokens processed."""
        return self._token_count

    def extend_with_token(self, token: Any) -> ParseResult:
        """Extend with a token, possibly creating a checkpoint.

        Args:
            token: Token to add (TokenInfo from parser.base)

        Returns:
            ParseResult from the parser
        """
        start_ns = time.perf_counter_ns()

        # Maybe create checkpoint before this token
        should_checkpoint = (
            self._token_count > 0 and
            self._token_count % self._checkpoint_interval == 0
        )

        if should_checkpoint:
            self._create_checkpoint()

        # Parse the token
        result = self._parser.extend_with_token(token)
        self._token_count += 1

        # Check for syntax boundary checkpoint
        if self._checkpoint_at_boundaries and self._is_boundary_token(token.text):
            self._create_checkpoint()

        elapsed_ns = time.perf_counter_ns() - start_ns
        self._update_avg_parse_time(elapsed_ns)

        return result

    def extend_with_text(self, text: str) -> ParseResult:
        """Extend with raw text.

        Note: This doesn't track individual tokens, so checkpointing
        is done at text boundaries.

        Args:
            text: Text to add

        Returns:
            ParseResult from the parser
        """
        start_ns = time.perf_counter_ns()

        # Create checkpoint before significant text additions
        if len(text) > 10:
            self._create_checkpoint()

        result = self._parser.extend_with_text(text)

        elapsed_ns = time.perf_counter_ns() - start_ns
        self._update_avg_parse_time(elapsed_ns)

        return result

    def rollback_to_position(self, position: int) -> Optional[ParseResult]:
        """Rollback to the nearest checkpoint at or before position.

        Args:
            position: Target character position

        Returns:
            ParseResult at the restored position, or None if no checkpoint
        """
        start_ns = time.perf_counter_ns()

        # Find nearest checkpoint at or before position
        checkpoint = self._find_checkpoint_before(position)
        if checkpoint is None:
            return None

        # Restore parser state
        self._parser.restore(checkpoint.parser_state)
        self._token_count = checkpoint.token_count

        # Remove checkpoints after this position
        self._remove_checkpoints_after(checkpoint.position)

        self._stats.checkpoints_used += 1
        elapsed_ns = time.perf_counter_ns() - start_ns
        self._update_avg_restore_time(elapsed_ns)

        return checkpoint.result

    def rollback_tokens(self, n: int) -> Optional[ParseResult]:
        """Rollback by N tokens.

        Args:
            n: Number of tokens to roll back

        Returns:
            ParseResult at the restored position, or None if not possible
        """
        target_token_count = max(0, self._token_count - n)

        # Find checkpoint with token_count <= target
        for pos in reversed(self._checkpoints.keys()):
            checkpoint = self._checkpoints[pos]
            if checkpoint.token_count <= target_token_count:
                return self.rollback_to_position(pos)

        return None

    def _create_checkpoint(self) -> None:
        """Create a checkpoint at current position."""
        position = self._parser.current_position
        source = self._parser.current_source

        checkpoint = ParseCheckpoint(
            position=position,
            token_count=self._token_count,
            parser_state=self._parser.checkpoint(),
            source_hash=hash(source),
            result=ParseResult(
                state=ParseState.VALID,
                ast=self._parser.get_ast(),
                position=position,
            ),
            created_at_ns=time.perf_counter_ns(),
        )

        self._checkpoints[position] = checkpoint
        self._stats.checkpoints_created += 1

        # Evict if needed
        self._evict_if_needed()

    def _find_checkpoint_before(self, position: int) -> Optional[ParseCheckpoint]:
        """Find the nearest checkpoint at or before position."""
        best: Optional[ParseCheckpoint] = None

        for pos, checkpoint in self._checkpoints.items():
            if pos <= position:
                if best is None or pos > best.position:
                    best = checkpoint

        return best

    def _remove_checkpoints_after(self, position: int) -> None:
        """Remove all checkpoints after position."""
        to_remove = [pos for pos in self._checkpoints if pos > position]
        for pos in to_remove:
            del self._checkpoints[pos]

    def _evict_if_needed(self) -> None:
        """Evict oldest checkpoints if cache is full."""
        while len(self._checkpoints) > self._max_checkpoints:
            # Remove oldest (first) checkpoint
            oldest_pos = next(iter(self._checkpoints.keys()))
            del self._checkpoints[oldest_pos]
            self._stats.checkpoints_evicted += 1

    def _is_boundary_token(self, text: str) -> bool:
        """Check if token represents a syntax boundary."""
        return any(b in text for b in self._boundary_tokens)

    def _update_avg_parse_time(self, elapsed_ns: int) -> None:
        """Update average parse time statistic."""
        alpha = 0.1
        self._stats.avg_parse_time_ns = (
            alpha * elapsed_ns + (1 - alpha) * self._stats.avg_parse_time_ns
        )

    def _update_avg_restore_time(self, elapsed_ns: int) -> None:
        """Update average restore time statistic."""
        alpha = 0.1
        self._stats.avg_restore_time_ns = (
            alpha * elapsed_ns + (1 - alpha) * self._stats.avg_restore_time_ns
        )

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return CacheStats(
            checkpoints_created=self._stats.checkpoints_created,
            checkpoints_used=self._stats.checkpoints_used,
            checkpoints_evicted=self._stats.checkpoints_evicted,
            avg_restore_time_ns=self._stats.avg_restore_time_ns,
            avg_parse_time_ns=self._stats.avg_parse_time_ns,
        )

    def clear_checkpoints(self) -> None:
        """Clear all checkpoints."""
        self._checkpoints.clear()

    def reset(self) -> None:
        """Reset to initial state."""
        self._parser.reset()
        self._checkpoints.clear()
        self._token_count = 0


@dataclass
class TextEdit:
    """A single text edit operation.

    Attributes:
        start: Start position in source
        end: End position (exclusive)
        new_text: Replacement text
    """

    start: int
    end: int
    new_text: str

    @property
    def is_insert(self) -> bool:
        """Whether this is a pure insertion."""
        return self.start == self.end

    @property
    def is_delete(self) -> bool:
        """Whether this is a pure deletion."""
        return len(self.new_text) == 0

    @property
    def length_delta(self) -> int:
        """Change in length from this edit."""
        return len(self.new_text) - (self.end - self.start)


class DiffBasedUpdater:
    """Computes and applies minimal updates to parsed ASTs.

    Instead of re-parsing the entire source after each change, this
    class computes the minimal edit and uses tree-sitter's incremental
    update capability for O(log n) updates.

    This is particularly useful for:
    - Token insertions at the end (most common during generation)
    - Small corrections/edits
    - Rollback operations

    Usage:
        updater = DiffBasedUpdater(parser)
        edit = TextEdit(start=50, end=50, new_text=" + 1")
        result = updater.apply_edit(edit)
    """

    def __init__(self, parser: IncrementalParser) -> None:
        """Initialize the diff-based updater.

        Args:
            parser: The underlying incremental parser
        """
        self._parser = parser
        self._last_source = ""

    @property
    def parser(self) -> IncrementalParser:
        """Get the underlying parser."""
        return self._parser

    def apply_edit(self, edit: TextEdit) -> ParseResult:
        """Apply a single edit and return updated parse result.

        Args:
            edit: The edit to apply

        Returns:
            ParseResult after the edit
        """
        # Get current source
        source = self._parser.current_source

        # Apply edit to source
        new_source = source[:edit.start] + edit.new_text + source[edit.end:]

        # For tree-sitter compatible parsers, we could use the incremental
        # edit API here. For now, we use extend_with_text for appends.
        if edit.is_insert and edit.start == len(source):
            # Append at end - most common case
            return self._parser.extend_with_text(edit.new_text)
        else:
            # Full re-parse needed
            return self._parser.parse_initial(new_source)

    def apply_edits(self, edits: List[TextEdit]) -> ParseResult:
        """Apply multiple edits in order.

        Args:
            edits: Edits to apply (should be in order by position)

        Returns:
            ParseResult after all edits
        """
        result: Optional[ParseResult] = None

        # Apply edits from back to front to preserve positions
        sorted_edits = sorted(edits, key=lambda e: e.start, reverse=True)

        for edit in sorted_edits:
            result = self.apply_edit(edit)

        return result or ParseResult(
            state=ParseState.VALID,
            ast=self._parser.get_ast(),
            position=self._parser.current_position,
        )

    def compute_diff(self, old_source: str, new_source: str) -> List[TextEdit]:
        """Compute minimal edits to transform old to new source.

        Uses a simple algorithm optimized for the common case of
        appending text at the end.

        Args:
            old_source: Original source
            new_source: Target source

        Returns:
            List of TextEdit operations
        """
        # Common prefix
        prefix_len = 0
        min_len = min(len(old_source), len(new_source))
        while prefix_len < min_len and old_source[prefix_len] == new_source[prefix_len]:
            prefix_len += 1

        # If only appending at end (most common case)
        if prefix_len == len(old_source):
            return [TextEdit(
                start=prefix_len,
                end=prefix_len,
                new_text=new_source[prefix_len:],
            )]

        # Common suffix
        suffix_len = 0
        while (suffix_len < min_len - prefix_len and
               old_source[-(suffix_len + 1)] == new_source[-(suffix_len + 1)]):
            suffix_len += 1

        # Compute edit
        old_end = len(old_source) - suffix_len
        new_end = len(new_source) - suffix_len

        return [TextEdit(
            start=prefix_len,
            end=old_end,
            new_text=new_source[prefix_len:new_end],
        )]


def create_cached_parser(
    parser: IncrementalParser,
    checkpoint_interval: int = 10,
    max_checkpoints: int = 50,
) -> IncrementalParseCache:
    """Factory function to create an IncrementalParseCache.

    Args:
        parser: The underlying parser
        checkpoint_interval: Create checkpoint every N tokens
        max_checkpoints: Maximum checkpoints to cache

    Returns:
        New IncrementalParseCache instance
    """
    return IncrementalParseCache(
        parser=parser,
        checkpoint_interval=checkpoint_interval,
        max_checkpoints=max_checkpoints,
    )
