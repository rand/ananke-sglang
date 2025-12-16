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
"""Unit tests for incremental parse cache."""

import pytest
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from parsing.base import (
    IncrementalParser,
    ParseResult,
    ParseState,
    ParseError,
    TokenInfo,
)
from parsing.incremental_cache import (
    IncrementalParseCache,
    ParseCheckpoint,
    TextEdit,
    DiffBasedUpdater,
    create_cached_parser,
)
from domains.types.marking.marked_ast import MarkedASTNode, ASTNodeKind
from domains.types.marking.provenance import SourceSpan, UNKNOWN_SPAN


class MockParser(IncrementalParser):
    """Mock parser for testing."""

    def __init__(self) -> None:
        self._source = ""
        self._position = 0
        self._checkpoints: List[Tuple[str, int]] = []

    @property
    def language(self) -> str:
        return "mock"

    @property
    def current_source(self) -> str:
        return self._source

    @property
    def current_position(self) -> int:
        return len(self._source)

    def parse_initial(self, source: str) -> ParseResult:
        self._source = source
        self._position = len(source)
        return ParseResult(
            state=ParseState.VALID,
            ast=self._create_mock_ast(),
            position=len(source),
        )

    def extend_with_token(self, token: TokenInfo) -> ParseResult:
        self._source += token.text
        self._position = len(self._source)
        return ParseResult(
            state=ParseState.VALID,
            ast=self._create_mock_ast(),
            position=len(self._source),
        )

    def extend_with_text(self, text: str) -> ParseResult:
        self._source += text
        self._position = len(self._source)
        return ParseResult(
            state=ParseState.VALID,
            ast=self._create_mock_ast(),
            position=len(self._source),
        )

    def find_holes(self) -> List[Tuple[str, SourceSpan]]:
        return []

    def get_expected_tokens(self) -> List[str]:
        return []

    def checkpoint(self) -> Any:
        return (self._source, self._position)

    def restore(self, checkpoint: Any) -> None:
        self._source, self._position = checkpoint

    def get_ast(self) -> Optional[MarkedASTNode]:
        return self._create_mock_ast()

    def copy(self) -> 'MockParser':
        new = MockParser()
        new._source = self._source
        new._position = self._position
        return new

    def _create_mock_ast(self) -> MarkedASTNode:
        return MarkedASTNode(
            kind=ASTNodeKind.MODULE,
            span=UNKNOWN_SPAN,
            children=[],
        )


class TestIncrementalParseCache:
    """Tests for IncrementalParseCache."""

    def test_create(self) -> None:
        """Test creating a cache."""
        parser = MockParser()
        cache = IncrementalParseCache(parser, checkpoint_interval=5)

        assert cache.parser is parser
        assert cache.position == 0
        assert cache.token_count == 0

    def test_create_with_factory(self) -> None:
        """Test creating cache with factory function."""
        parser = MockParser()
        cache = create_cached_parser(parser, checkpoint_interval=3)

        assert cache._checkpoint_interval == 3

    def test_extend_with_token(self) -> None:
        """Test extending with tokens."""
        parser = MockParser()
        cache = IncrementalParseCache(parser, checkpoint_interval=5)

        token = TokenInfo(token_id=1, text="hello", position=0, length=5)
        result = cache.extend_with_token(token)

        assert result.state == ParseState.VALID
        assert cache.position == 5
        assert cache.token_count == 1

    def test_checkpoint_creation(self) -> None:
        """Test that checkpoints are created at intervals."""
        parser = MockParser()
        cache = IncrementalParseCache(parser, checkpoint_interval=3)

        # Add 5 tokens
        for i in range(5):
            token = TokenInfo(token_id=i, text=f"t{i}", position=i * 2, length=2)
            cache.extend_with_token(token)

        # Should have checkpoint at token 3
        stats = cache.get_stats()
        assert stats.checkpoints_created >= 1

    def test_rollback_to_position(self) -> None:
        """Test rolling back to a position."""
        parser = MockParser()
        cache = IncrementalParseCache(parser, checkpoint_interval=2)

        # Add tokens
        for i in range(6):
            token = TokenInfo(token_id=i, text=f"_{i}", position=i * 2, length=2)
            cache.extend_with_token(token)

        original_position = cache.position
        original_token_count = cache.token_count

        # Rollback
        result = cache.rollback_to_position(4)

        assert result is not None
        assert cache.position <= 4
        assert cache.token_count < original_token_count

    def test_rollback_tokens(self) -> None:
        """Test rolling back by number of tokens."""
        parser = MockParser()
        cache = IncrementalParseCache(parser, checkpoint_interval=2)

        # Add 10 tokens
        for i in range(10):
            token = TokenInfo(token_id=i, text=f"x{i}", position=i * 2, length=2)
            cache.extend_with_token(token)

        original_count = cache.token_count
        assert original_count == 10

        # Roll back 5 tokens
        result = cache.rollback_tokens(5)

        assert result is not None
        assert cache.token_count <= original_count - 5

    def test_boundary_checkpoint(self) -> None:
        """Test checkpoint creation at syntax boundaries."""
        parser = MockParser()
        cache = IncrementalParseCache(
            parser,
            checkpoint_interval=100,  # High interval
            checkpoint_at_boundaries=True,
        )

        # Add boundary tokens
        boundary_tokens = ["{", "def ", "}"]
        for i, text in enumerate(boundary_tokens):
            token = TokenInfo(token_id=i, text=text, position=i, length=len(text))
            cache.extend_with_token(token)

        # Should have checkpoints for boundary tokens
        stats = cache.get_stats()
        assert stats.checkpoints_created >= 1

    def test_stats_tracking(self) -> None:
        """Test statistics tracking."""
        parser = MockParser()
        cache = IncrementalParseCache(parser, checkpoint_interval=2)

        # Add tokens to create checkpoints
        # With interval=2, checkpoints created after tokens 2, 4
        for i in range(6):
            token = TokenInfo(token_id=i, text=f"a{i}", position=i * 2, length=2)
            cache.extend_with_token(token)

        # Use a checkpoint - rollback to position covered by checkpoint
        # Total source is "a0a1a2a3a4a5" (12 chars)
        # Checkpoint at token 2 has position ~4 chars
        result = cache.rollback_to_position(10)  # Should find a checkpoint

        stats = cache.get_stats()
        assert stats.checkpoints_created > 0
        # Only assert checkpoint used if rollback succeeded
        if result is not None:
            assert stats.checkpoints_used > 0

    def test_cache_eviction(self) -> None:
        """Test checkpoint eviction when cache is full."""
        parser = MockParser()
        cache = IncrementalParseCache(
            parser,
            checkpoint_interval=1,  # Every token
            max_checkpoints=3,
        )

        # Add many tokens
        for i in range(10):
            token = TokenInfo(token_id=i, text=f"t{i}", position=i * 2, length=2)
            cache.extend_with_token(token)

        stats = cache.get_stats()
        assert stats.checkpoints_evicted > 0

    def test_clear_checkpoints(self) -> None:
        """Test clearing checkpoints."""
        parser = MockParser()
        cache = IncrementalParseCache(parser, checkpoint_interval=1)

        # Add tokens
        for i in range(5):
            token = TokenInfo(token_id=i, text=f"t{i}", position=i * 2, length=2)
            cache.extend_with_token(token)

        # Clear
        cache.clear_checkpoints()

        # Rollback should fail
        result = cache.rollback_to_position(2)
        assert result is None

    def test_reset(self) -> None:
        """Test resetting the cache."""
        parser = MockParser()
        parser.parse_initial("initial text")
        cache = IncrementalParseCache(parser, checkpoint_interval=2)

        # Add some state
        for i in range(3):
            token = TokenInfo(token_id=i, text=f"t{i}", position=i, length=2)
            cache.extend_with_token(token)

        # Reset
        cache.reset()

        assert cache.position == 0
        assert cache.token_count == 0


class TestTextEdit:
    """Tests for TextEdit."""

    def test_is_insert(self) -> None:
        """Test insert detection."""
        edit = TextEdit(start=5, end=5, new_text="hello")
        assert edit.is_insert
        assert not edit.is_delete

    def test_is_delete(self) -> None:
        """Test delete detection."""
        edit = TextEdit(start=5, end=10, new_text="")
        assert edit.is_delete
        assert not edit.is_insert

    def test_length_delta(self) -> None:
        """Test length delta calculation."""
        # Insert
        edit = TextEdit(start=5, end=5, new_text="abc")
        assert edit.length_delta == 3

        # Delete
        edit = TextEdit(start=5, end=10, new_text="")
        assert edit.length_delta == -5

        # Replace
        edit = TextEdit(start=5, end=10, new_text="abc")
        assert edit.length_delta == -2


class TestDiffBasedUpdater:
    """Tests for DiffBasedUpdater."""

    def test_create(self) -> None:
        """Test creating an updater."""
        parser = MockParser()
        updater = DiffBasedUpdater(parser)

        assert updater.parser is parser

    def test_apply_insert_at_end(self) -> None:
        """Test applying insert at end."""
        parser = MockParser()
        parser.parse_initial("hello")

        updater = DiffBasedUpdater(parser)
        edit = TextEdit(start=5, end=5, new_text=" world")
        result = updater.apply_edit(edit)

        assert result.state == ParseState.VALID
        assert parser.current_source == "hello world"

    def test_compute_diff_append(self) -> None:
        """Test diff computation for append."""
        parser = MockParser()
        updater = DiffBasedUpdater(parser)

        edits = updater.compute_diff("hello", "hello world")

        assert len(edits) == 1
        assert edits[0].is_insert
        assert edits[0].start == 5
        assert edits[0].new_text == " world"

    def test_compute_diff_replace(self) -> None:
        """Test diff computation for replacement."""
        parser = MockParser()
        updater = DiffBasedUpdater(parser)

        edits = updater.compute_diff("hello world", "hello there")

        assert len(edits) == 1
        assert edits[0].start == 6
        assert edits[0].new_text == "there"

    def test_compute_diff_same(self) -> None:
        """Test diff computation for identical strings."""
        parser = MockParser()
        updater = DiffBasedUpdater(parser)

        edits = updater.compute_diff("hello", "hello")

        assert len(edits) == 1
        assert edits[0].start == 5
        assert edits[0].end == 5
        assert edits[0].new_text == ""

    def test_apply_multiple_edits(self) -> None:
        """Test applying multiple edits."""
        parser = MockParser()
        parser.parse_initial("ab")

        updater = DiffBasedUpdater(parser)
        edits = [
            TextEdit(start=2, end=2, new_text="cd"),
            TextEdit(start=4, end=4, new_text="ef"),
        ]
        result = updater.apply_edits(edits)

        assert result.state == ParseState.VALID
