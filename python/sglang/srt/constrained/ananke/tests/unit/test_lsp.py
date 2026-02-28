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
"""Tests for LSP integration module.

Tests the Language Server Protocol integration for IDE-grade type checking.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch

try:
    from ...lsp.protocol import (
        LSPConfig,
        LSPRequest,
        LSPResponse,
        LSPError,
        CompletionItem,
        CompletionItemKind,
        CompletionList,
        Diagnostic,
        DiagnosticSeverity,
        Position,
        Range,
        TextDocumentIdentifier,
        VersionedTextDocumentIdentifier,
        TextDocumentContentChangeEvent,
        TextEdit,
    )
    from ...lsp.client import (
        LSPClient,
        BatchedLSPClient,
        CachedLSPClient,
        SubprocessLSPClient,
        create_lsp_client,
    )
    from ...lsp.integration import (
        LSPTypeDomain,
        LSPCompletionProvider,
        LSPDiagnosticProvider,
        LSPIntegrationStats,
        LSPCompletionContext,
        create_lsp_type_domain,
    )
except ImportError:
    from lsp.protocol import (
        LSPConfig,
        LSPRequest,
        LSPResponse,
        LSPError,
        CompletionItem,
        CompletionItemKind,
        CompletionList,
        Diagnostic,
        DiagnosticSeverity,
        Position,
        Range,
        TextDocumentIdentifier,
        VersionedTextDocumentIdentifier,
        TextDocumentContentChangeEvent,
        TextEdit,
    )
    from lsp.client import (
        LSPClient,
        BatchedLSPClient,
        CachedLSPClient,
        SubprocessLSPClient,
        create_lsp_client,
    )
    from lsp.integration import (
        LSPTypeDomain,
        LSPCompletionProvider,
        LSPDiagnosticProvider,
        LSPIntegrationStats,
        LSPCompletionContext,
        create_lsp_type_domain,
    )


# =============================================================================
# Protocol Tests
# =============================================================================


class TestPosition:
    """Tests for Position class."""

    def test_create_position(self) -> None:
        """Test position creation."""
        pos = Position(line=10, character=5)
        assert pos.line == 10
        assert pos.character == 5

    def test_position_to_dict(self) -> None:
        """Test position serialization."""
        pos = Position(line=10, character=5)
        d = pos.to_dict()
        assert d == {"line": 10, "character": 5}

    def test_position_from_dict(self) -> None:
        """Test position deserialization."""
        d = {"line": 10, "character": 5}
        pos = Position.from_dict(d)
        assert pos.line == 10
        assert pos.character == 5

    def test_position_comparison(self) -> None:
        """Test position comparison."""
        p1 = Position(line=1, character=0)
        p2 = Position(line=2, character=0)
        p3 = Position(line=1, character=5)
        assert p1 < p2
        assert p1 < p3
        assert p2 > p1


class TestRange:
    """Tests for Range class."""

    def test_create_range(self) -> None:
        """Test range creation."""
        r = Range(
            start=Position(0, 0),
            end=Position(10, 5),
        )
        assert r.start.line == 0
        assert r.end.line == 10

    def test_range_to_dict(self) -> None:
        """Test range serialization."""
        r = Range(
            start=Position(0, 0),
            end=Position(10, 5),
        )
        d = r.to_dict()
        assert d["start"] == {"line": 0, "character": 0}
        assert d["end"] == {"line": 10, "character": 5}

    def test_range_contains(self) -> None:
        """Test range contains position."""
        r = Range(
            start=Position(5, 0),
            end=Position(10, 0),
        )
        assert r.contains(Position(7, 5))
        assert not r.contains(Position(3, 0))
        assert not r.contains(Position(12, 0))


class TestCompletionItem:
    """Tests for CompletionItem class."""

    def test_create_completion_item(self) -> None:
        """Test completion item creation."""
        item = CompletionItem(
            label="print",
            kind=CompletionItemKind.Function,
            detail="(x: Any) -> None",
        )
        assert item.label == "print"
        assert item.kind == CompletionItemKind.Function
        assert item.detail == "(x: Any) -> None"

    def test_completion_item_to_dict(self) -> None:
        """Test completion item serialization."""
        item = CompletionItem(
            label="print",
            kind=CompletionItemKind.Function,
        )
        d = item.to_dict()
        assert d["label"] == "print"
        assert d["kind"] == 3  # Function

    def test_completion_item_from_dict(self) -> None:
        """Test completion item deserialization."""
        d = {
            "label": "print",
            "kind": 3,
            "detail": "builtin",
        }
        item = CompletionItem.from_dict(d)
        assert item.label == "print"
        assert item.kind == CompletionItemKind.Function
        assert item.detail == "builtin"

    def test_effective_insert_text_from_text_edit(self) -> None:
        """Test effective insert text from text edit."""
        item = CompletionItem(
            label="foo",
            text_edit=TextEdit(
                range=Range(Position(0, 0), Position(0, 0)),
                new_text="foobar",
            ),
        )
        assert item.effective_insert_text == "foobar"

    def test_effective_insert_text_from_insert_text(self) -> None:
        """Test effective insert text from insert_text field."""
        item = CompletionItem(
            label="foo",
            insert_text="foobar",
        )
        assert item.effective_insert_text == "foobar"

    def test_effective_insert_text_from_label(self) -> None:
        """Test effective insert text fallback to label."""
        item = CompletionItem(label="foo")
        assert item.effective_insert_text == "foo"


class TestCompletionList:
    """Tests for CompletionList class."""

    def test_create_completion_list(self) -> None:
        """Test completion list creation."""
        items = [
            CompletionItem(label="a"),
            CompletionItem(label="b"),
        ]
        lst = CompletionList(is_incomplete=False, items=items)
        assert len(lst) == 2
        assert not lst.is_incomplete

    def test_completion_list_iteration(self) -> None:
        """Test completion list iteration."""
        items = [
            CompletionItem(label="a"),
            CompletionItem(label="b"),
        ]
        lst = CompletionList(is_incomplete=False, items=items)
        labels = [item.label for item in lst]
        assert labels == ["a", "b"]


class TestDiagnostic:
    """Tests for Diagnostic class."""

    def test_create_diagnostic(self) -> None:
        """Test diagnostic creation."""
        d = Diagnostic(
            range=Range(Position(0, 0), Position(0, 10)),
            message="Undefined variable 'x'",
            severity=DiagnosticSeverity.Error,
        )
        assert d.message == "Undefined variable 'x'"
        assert d.is_error

    def test_diagnostic_is_warning(self) -> None:
        """Test warning detection."""
        d = Diagnostic(
            range=Range(Position(0, 0), Position(0, 10)),
            message="Unused variable",
            severity=DiagnosticSeverity.Warning,
        )
        assert d.is_warning
        assert not d.is_error


class TestLSPConfig:
    """Tests for LSPConfig class."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        config = LSPConfig()
        assert config.python == "pyright-langserver"
        assert config.timeout_ms == 100
        assert config.cache_size == 1000

    def test_get_server_for_language(self) -> None:
        """Test language server lookup."""
        config = LSPConfig()
        assert config.get_server_for_language("python") == "pyright-langserver"
        assert config.get_server_for_language("typescript") == "typescript-language-server"
        assert config.get_server_for_language("unknown") is None


class TestLSPError:
    """Tests for LSPError class."""

    def test_error_codes(self) -> None:
        """Test error code constants."""
        assert LSPError.PARSE_ERROR == -32700
        assert LSPError.REQUEST_CANCELLED == -32800

    def test_is_timeout(self) -> None:
        """Test timeout detection."""
        err = LSPError(code=-32800, message="Request timeout")
        assert err.is_timeout


# =============================================================================
# Mock Client for Testing
# =============================================================================


class MockLSPClient(LSPClient):
    """Mock LSP client for testing."""

    def __init__(
        self,
        completions: Optional[List[CompletionItem]] = None,
        hover_text: Optional[str] = None,
        should_fail: bool = False,
    ):
        super().__init__(language="python")
        self._completions = completions or []
        self._hover_text = hover_text
        self._should_fail = should_fail
        self._requests: List[LSPRequest] = []

    async def start(self) -> bool:
        return not self._should_fail

    async def stop(self) -> None:
        pass

    async def send_request(self, request: LSPRequest) -> LSPResponse:
        self._requests.append(request)
        if self._should_fail:
            return LSPResponse(
                id=request.id or 0,
                error=LSPError(code=-32603, message="Mock error"),
            )
        return LSPResponse(id=request.id or 0, result={})

    async def get_completions(
        self,
        uri: str,
        position: Position,
    ) -> Optional[CompletionList]:
        if self._should_fail:
            return None
        return CompletionList(is_incomplete=False, items=self._completions)

    async def get_hover(
        self,
        uri: str,
        position: Position,
    ) -> Optional[str]:
        if self._should_fail:
            return None
        return self._hover_text


# =============================================================================
# Client Tests
# =============================================================================


class TestCachedLSPClient:
    """Tests for CachedLSPClient."""

    def test_cache_hit(self) -> None:
        """Test cache hit returns cached response."""
        async def run_test():
            inner = MockLSPClient()
            cached = CachedLSPClient(inner, cache_size=10)

            # First request
            request = LSPRequest(method="test", params={"a": 1}, id=1)
            await cached.send_request(request)

            # Second identical request
            request2 = LSPRequest(method="test", params={"a": 1}, id=2)
            await cached.send_request(request2)

            # Should only have one actual request
            assert len(inner._requests) == 1
            assert cached.cache_hit_rate > 0

        asyncio.run(run_test())

    def test_cache_invalidation(self) -> None:
        """Test cache invalidation by document."""
        async def run_test():
            inner = MockLSPClient()
            cached = CachedLSPClient(inner, cache_size=10)

            # Add some cached entries
            request = LSPRequest(
                method="textDocument/completion",
                params={"textDocument": {"uri": "file:///test.py"}},
                id=1,
            )
            await cached.send_request(request)

            # Invalidate
            cached.invalidate_document("file:///test.py")

            # Cache should be cleared for that document
            assert len(cached._cache) == 0

        asyncio.run(run_test())


class TestBatchedLSPClient:
    """Tests for BatchedLSPClient."""

    def test_notifications_not_batched(self) -> None:
        """Test that notifications bypass batching."""
        async def run_test():
            inner = MockLSPClient()
            batched = BatchedLSPClient(inner, batch_interval_ms=100)

            # Notification (no id)
            request = LSPRequest(method="notify", params={}, id=None)
            await batched.send_request(request)

            # Should be sent immediately
            assert len(inner._requests) == 1

        asyncio.run(run_test())


# =============================================================================
# Integration Tests
# =============================================================================


class TestLSPDiagnosticProvider:
    """Tests for LSPDiagnosticProvider."""

    def test_no_diagnostics_full_score(self) -> None:
        """Test that no diagnostics gives full score."""
        provider = LSPDiagnosticProvider()
        score = provider.get_score("file:///test.py")
        assert score == 1.0

    def test_error_diagnostic_zero_score(self) -> None:
        """Test that error diagnostic gives zero score."""
        provider = LSPDiagnosticProvider()
        provider.update_diagnostics(
            "file:///test.py",
            [
                Diagnostic(
                    range=Range(Position(0, 0), Position(0, 10)),
                    message="Error",
                    severity=DiagnosticSeverity.Error,
                )
            ],
        )
        score = provider.get_score("file:///test.py")
        assert score == 0.0

    def test_warning_diagnostic_partial_score(self) -> None:
        """Test that warning diagnostic gives partial score."""
        provider = LSPDiagnosticProvider()
        provider.update_diagnostics(
            "file:///test.py",
            [
                Diagnostic(
                    range=Range(Position(0, 0), Position(0, 10)),
                    message="Warning",
                    severity=DiagnosticSeverity.Warning,
                )
            ],
        )
        score = provider.get_score("file:///test.py")
        assert 0.0 < score < 1.0

    def test_clear_diagnostics(self) -> None:
        """Test clearing diagnostics."""
        provider = LSPDiagnosticProvider()
        provider.update_diagnostics(
            "file:///test.py",
            [
                Diagnostic(
                    range=Range(Position(0, 0), Position(0, 10)),
                    message="Error",
                    severity=DiagnosticSeverity.Error,
                )
            ],
        )
        provider.clear_diagnostics("file:///test.py")
        score = provider.get_score("file:///test.py")
        assert score == 1.0


class TestLSPCompletionProvider:
    """Tests for LSPCompletionProvider."""

    def test_get_completion_tokens(self) -> None:
        """Test getting tokens from completions."""
        async def run_test():
            completions = [
                CompletionItem(label="print"),
                CompletionItem(label="pass"),
            ]
            client = MockLSPClient(completions=completions)

            # Mock tokenizer
            tokenizer = MagicMock()
            tokenizer.encode.side_effect = lambda text, add_special_tokens: [hash(text) % 1000]

            provider = LSPCompletionProvider(
                client=client,
                tokenizer=tokenizer,
                vocab_size=32000,
            )

            context = LSPCompletionContext(
                uri="file:///test.py",
                position=Position(0, 0),
                prefix_text="",
                language="python",
            )

            tokens = await provider.get_completion_tokens(context)
            assert len(tokens) == 2  # print and pass

        asyncio.run(run_test())

    def test_completion_failure_returns_empty(self) -> None:
        """Test that completion failure returns empty set."""
        async def run_test():
            client = MockLSPClient(should_fail=True)
            tokenizer = MagicMock()

            provider = LSPCompletionProvider(
                client=client,
                tokenizer=tokenizer,
                vocab_size=32000,
            )

            context = LSPCompletionContext(
                uri="file:///test.py",
                position=Position(0, 0),
                prefix_text="",
                language="python",
            )

            tokens = await provider.get_completion_tokens(context)
            assert len(tokens) == 0

        asyncio.run(run_test())

    def test_get_completion_mask(self) -> None:
        """Test creating mask from tokens."""
        client = MockLSPClient()
        tokenizer = MagicMock()

        provider = LSPCompletionProvider(
            client=client,
            tokenizer=tokenizer,
            vocab_size=1000,
        )

        tokens = {10, 20, 30}
        mask = provider.get_completion_mask(tokens)

        assert mask.shape == (1000,)
        assert mask[10]
        assert mask[20]
        assert mask[30]
        assert not mask[5]


class TestLSPTypeDomain:
    """Tests for LSPTypeDomain."""

    def test_start_and_stop(self) -> None:
        """Test starting and stopping the LSP domain."""
        async def run_test():
            client = MockLSPClient()
            tokenizer = MagicMock()

            domain = LSPTypeDomain(
                client=client,
                tokenizer=tokenizer,
                vocab_size=32000,
            )

            success = await domain.start("file:///workspace")
            assert success

            await domain.stop()

        asyncio.run(run_test())

    def test_get_token_mask_with_completions(self) -> None:
        """Test token mask with LSP completions."""
        async def run_test():
            completions = [
                CompletionItem(label="print"),
            ]
            client = MockLSPClient(completions=completions)

            tokenizer = MagicMock()
            tokenizer.encode.return_value = [42]  # print -> token 42

            domain = LSPTypeDomain(
                client=client,
                tokenizer=tokenizer,
                vocab_size=1000,
            )

            # Open a document first
            await domain.open_document(
                uri="file:///test.py",
                language_id="python",
                content="",
            )

            mask = await domain.get_token_mask(
                uri="file:///test.py",
                position=Position(0, 0),
            )

            assert mask[42]  # Token for "print" should be allowed

        asyncio.run(run_test())

    def test_fallback_on_failure(self) -> None:
        """Test that failure falls back to all-allowed."""
        async def run_test():
            client = MockLSPClient(should_fail=True)
            tokenizer = MagicMock()

            domain = LSPTypeDomain(
                client=client,
                tokenizer=tokenizer,
                vocab_size=1000,
                fallback_enabled=True,
            )

            await domain.open_document(
                uri="file:///test.py",
                language_id="python",
                content="",
            )

            mask = await domain.get_token_mask(
                uri="file:///test.py",
                position=Position(0, 0),
            )

            # All tokens should be allowed (fallback)
            assert mask.all()

        asyncio.run(run_test())

    def test_document_lifecycle(self) -> None:
        """Test opening, updating, and closing documents."""
        async def run_test():
            client = MockLSPClient()
            tokenizer = MagicMock()

            domain = LSPTypeDomain(
                client=client,
                tokenizer=tokenizer,
                vocab_size=32000,
            )

            # Open
            await domain.open_document(
                uri="file:///test.py",
                language_id="python",
                content="x = 1",
            )
            assert "file:///test.py" in domain._open_documents

            # Update
            await domain.update_document(
                uri="file:///test.py",
                content="x = 2",
            )
            assert domain._document_content["file:///test.py"] == "x = 2"

            # Close
            await domain.close_document("file:///test.py")
            assert "file:///test.py" not in domain._open_documents

        asyncio.run(run_test())


class TestLSPIntegrationStats:
    """Tests for LSPIntegrationStats."""

    def test_success_rate(self) -> None:
        """Test success rate calculation."""
        stats = LSPIntegrationStats(
            total_queries=10,
            successful_queries=8,
        )
        assert stats.success_rate == 0.8

    def test_success_rate_zero_queries(self) -> None:
        """Test success rate with zero queries."""
        stats = LSPIntegrationStats()
        assert stats.success_rate == 0.0

    def test_completion_use_rate(self) -> None:
        """Test completion use rate calculation."""
        stats = LSPIntegrationStats(
            total_completions=100,
            completions_used=75,
        )
        assert stats.completion_use_rate == 0.75


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_lsp_client(self) -> None:
        """Test LSP client factory."""
        client = create_lsp_client(
            language="python",
            enable_batching=True,
            enable_caching=True,
        )
        # Should be wrapped in batching and caching layers
        assert isinstance(client, BatchedLSPClient)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
