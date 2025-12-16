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
"""Async LSP client with batching and caching.

This module provides LSP client implementations for communicating with
language servers. It supports:
- Async communication via subprocess
- Request batching to reduce round-trips
- Response caching for repeated queries
- Timeout handling with graceful fallback

Design based on ChatLSP (OOPSLA 2024) principles:
- Fast timeouts (100ms default) to not block generation
- Soft responses (never hard-block on LSP failure)
- Incremental document sync for streaming generation

Supported Operations:
- textDocument/completion: Get completions at position
- textDocument/hover: Get type information at position
- textDocument/signatureHelp: Get function signatures
- textDocument/publishDiagnostics: Receive diagnostics

References:
    - LSP Specification: https://microsoft.github.io/language-server-protocol/
    - ChatLSP (OOPSLA 2024): https://arxiv.org/abs/2409.00921
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .protocol import (
    LSPConfig,
    LSPRequest,
    LSPResponse,
    LSPError,
    CompletionItem,
    CompletionList,
    Diagnostic,
    Position,
    Range,
    TextDocumentIdentifier,
    VersionedTextDocumentIdentifier,
    TextDocumentContentChangeEvent,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Base Client Interface
# =============================================================================


class LSPClient(ABC):
    """Abstract base class for LSP clients.

    Provides a common interface for communicating with language servers.
    Implementations handle transport (subprocess, socket, etc.).

    Attributes:
        config: LSP configuration
        language: Language this client handles
        is_initialized: Whether the server is initialized
    """

    def __init__(self, config: Optional[LSPConfig] = None, language: str = "python"):
        """Initialize the LSP client.

        Args:
            config: LSP configuration
            language: Language this client handles
        """
        self.config = config or LSPConfig()
        self.language = language
        self.is_initialized = False
        self._request_id = 0

    def _next_id(self) -> int:
        """Get next request ID."""
        self._request_id += 1
        return self._request_id

    @abstractmethod
    async def start(self) -> bool:
        """Start the language server.

        Returns:
            True if server started successfully
        """
        ...

    @abstractmethod
    async def stop(self) -> None:
        """Stop the language server."""
        ...

    @abstractmethod
    async def send_request(self, request: LSPRequest) -> LSPResponse:
        """Send a request and wait for response.

        Args:
            request: The LSP request

        Returns:
            LSP response
        """
        ...

    async def send_notification(self, method: str, params: Dict[str, Any]) -> None:
        """Send a notification (no response expected).

        Args:
            method: LSP method name
            params: Notification parameters
        """
        request = LSPRequest(method=method, params=params, id=None)
        await self.send_request(request)

    # -------------------------------------------------------------------------
    # High-level LSP Operations
    # -------------------------------------------------------------------------

    async def initialize(
        self,
        root_uri: str,
        capabilities: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Initialize the language server.

        Args:
            root_uri: Root workspace URI
            capabilities: Client capabilities

        Returns:
            True if initialization succeeded
        """
        if capabilities is None:
            capabilities = self._default_capabilities()

        request = LSPRequest(
            method="initialize",
            params={
                "processId": None,
                "rootUri": root_uri,
                "capabilities": capabilities,
            },
            id=self._next_id(),
        )

        try:
            response = await self.send_request(request)
            if response.is_success:
                # Send initialized notification
                await self.send_notification("initialized", {})
                self.is_initialized = True
                return True
            else:
                logger.warning(f"LSP initialize failed: {response.error}")
                return False
        except Exception as e:
            logger.warning(f"LSP initialize error: {e}")
            return False

    async def shutdown(self) -> None:
        """Shutdown the language server gracefully."""
        if not self.is_initialized:
            return

        try:
            request = LSPRequest(
                method="shutdown",
                params={},
                id=self._next_id(),
            )
            await self.send_request(request)
            await self.send_notification("exit", {})
        except Exception as e:
            logger.debug(f"LSP shutdown error (expected): {e}")
        finally:
            self.is_initialized = False

    async def open_document(
        self,
        uri: str,
        language_id: str,
        version: int,
        text: str,
    ) -> None:
        """Notify server that a document was opened.

        Args:
            uri: Document URI
            language_id: Language identifier
            version: Document version
            text: Document content
        """
        await self.send_notification(
            "textDocument/didOpen",
            {
                "textDocument": {
                    "uri": uri,
                    "languageId": language_id,
                    "version": version,
                    "text": text,
                }
            },
        )

    async def update_document(
        self,
        uri: str,
        version: int,
        changes: List[TextDocumentContentChangeEvent],
    ) -> None:
        """Notify server of document changes.

        Args:
            uri: Document URI
            version: New document version
            changes: Content changes
        """
        await self.send_notification(
            "textDocument/didChange",
            {
                "textDocument": {"uri": uri, "version": version},
                "contentChanges": [c.to_dict() for c in changes],
            },
        )

    async def close_document(self, uri: str) -> None:
        """Notify server that a document was closed.

        Args:
            uri: Document URI
        """
        await self.send_notification(
            "textDocument/didClose",
            {"textDocument": {"uri": uri}},
        )

    async def get_completions(
        self,
        uri: str,
        position: Position,
    ) -> Optional[CompletionList]:
        """Get completions at a position.

        Args:
            uri: Document URI
            position: Cursor position

        Returns:
            CompletionList or None on error
        """
        request = LSPRequest(
            method="textDocument/completion",
            params={
                "textDocument": {"uri": uri},
                "position": position.to_dict(),
            },
            id=self._next_id(),
        )

        try:
            response = await self.send_request(request)
            if response.is_success and response.result is not None:
                # Result can be CompletionItem[] or CompletionList
                if isinstance(response.result, list):
                    return CompletionList(
                        is_incomplete=False,
                        items=[CompletionItem.from_dict(item) for item in response.result],
                    )
                else:
                    return CompletionList.from_dict(response.result)
            return None
        except Exception as e:
            logger.debug(f"LSP completion error: {e}")
            return None

    async def get_hover(
        self,
        uri: str,
        position: Position,
    ) -> Optional[str]:
        """Get hover information at a position.

        Args:
            uri: Document URI
            position: Cursor position

        Returns:
            Hover content string or None
        """
        request = LSPRequest(
            method="textDocument/hover",
            params={
                "textDocument": {"uri": uri},
                "position": position.to_dict(),
            },
            id=self._next_id(),
        )

        try:
            response = await self.send_request(request)
            if response.is_success and response.result is not None:
                contents = response.result.get("contents")
                if isinstance(contents, str):
                    return contents
                elif isinstance(contents, dict):
                    return contents.get("value", str(contents))
                elif isinstance(contents, list):
                    return "\n".join(
                        c.get("value", str(c)) if isinstance(c, dict) else str(c)
                        for c in contents
                    )
            return None
        except Exception as e:
            logger.debug(f"LSP hover error: {e}")
            return None

    async def get_signature_help(
        self,
        uri: str,
        position: Position,
    ) -> Optional[Dict[str, Any]]:
        """Get signature help at a position.

        Args:
            uri: Document URI
            position: Cursor position

        Returns:
            Signature help dict or None
        """
        request = LSPRequest(
            method="textDocument/signatureHelp",
            params={
                "textDocument": {"uri": uri},
                "position": position.to_dict(),
            },
            id=self._next_id(),
        )

        try:
            response = await self.send_request(request)
            if response.is_success:
                return response.result
            return None
        except Exception as e:
            logger.debug(f"LSP signature help error: {e}")
            return None

    def _default_capabilities(self) -> Dict[str, Any]:
        """Get default client capabilities."""
        return {
            "textDocument": {
                "completion": {
                    "completionItem": {
                        "snippetSupport": False,
                        "commitCharactersSupport": True,
                        "documentationFormat": ["plaintext"],
                        "deprecatedSupport": True,
                        "preselectSupport": True,
                    },
                    "completionItemKind": {
                        "valueSet": list(range(1, 26)),
                    },
                    "contextSupport": True,
                },
                "hover": {
                    "contentFormat": ["plaintext"],
                },
                "signatureHelp": {
                    "signatureInformation": {
                        "documentationFormat": ["plaintext"],
                        "parameterInformation": {
                            "labelOffsetSupport": True,
                        },
                    },
                },
                "publishDiagnostics": {
                    "relatedInformation": True,
                    "versionSupport": True,
                },
                "synchronization": {
                    "willSave": False,
                    "willSaveWaitUntil": False,
                    "didSave": True,
                },
            },
            "workspace": {
                "workspaceFolders": True,
            },
        }


# =============================================================================
# Subprocess-based Client
# =============================================================================


class SubprocessLSPClient(LSPClient):
    """LSP client using subprocess communication.

    Communicates with the language server via stdin/stdout using
    the JSON-RPC protocol.

    Attributes:
        process: Subprocess running the language server
        _pending: Pending request futures
    """

    def __init__(
        self,
        config: Optional[LSPConfig] = None,
        language: str = "python",
        server_command: Optional[List[str]] = None,
    ):
        """Initialize subprocess client.

        Args:
            config: LSP configuration
            language: Language this client handles
            server_command: Command to start the server
        """
        super().__init__(config, language)
        self._server_command = server_command
        self._process: Optional[asyncio.subprocess.Process] = None
        self._pending: Dict[int, asyncio.Future] = {}
        self._reader_task: Optional[asyncio.Task] = None

    async def start(self) -> bool:
        """Start the language server subprocess."""
        if self._process is not None:
            return True

        command = self._server_command
        if command is None:
            server = self.config.get_server_for_language(self.language)
            if server is None:
                logger.warning(f"No server configured for {self.language}")
                return False
            command = [server, "--stdio"]

        try:
            self._process = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            self._reader_task = asyncio.create_task(self._read_responses())
            return True
        except Exception as e:
            logger.warning(f"Failed to start LSP server: {e}")
            return False

    async def stop(self) -> None:
        """Stop the language server subprocess."""
        if self._reader_task is not None:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass
            self._reader_task = None

        if self._process is not None:
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                self._process.kill()
            self._process = None

        # Cancel pending requests
        for future in self._pending.values():
            future.cancel()
        self._pending.clear()

    async def send_request(self, request: LSPRequest) -> LSPResponse:
        """Send a request via subprocess stdin."""
        if self._process is None or self._process.stdin is None:
            return LSPResponse(
                id=request.id or 0,
                error=LSPError(
                    code=LSPError.SERVER_NOT_INITIALIZED,
                    message="Server not started",
                ),
            )

        # Encode request
        content = json.dumps(request.to_dict())
        message = f"Content-Length: {len(content)}\r\n\r\n{content}"

        try:
            self._process.stdin.write(message.encode())
            await self._process.stdin.drain()
        except Exception as e:
            return LSPResponse(
                id=request.id or 0,
                error=LSPError(code=LSPError.INTERNAL_ERROR, message=str(e)),
            )

        # For notifications (no id), return immediately
        if request.id is None:
            return LSPResponse(id=0)

        # Wait for response with timeout
        future: asyncio.Future = asyncio.Future()
        self._pending[request.id] = future

        try:
            response = await asyncio.wait_for(
                future,
                timeout=self.config.timeout_ms / 1000.0,
            )
            return response
        except asyncio.TimeoutError:
            self._pending.pop(request.id, None)
            return LSPResponse(
                id=request.id,
                error=LSPError(
                    code=LSPError.REQUEST_CANCELLED,
                    message=f"Request timeout ({self.config.timeout_ms}ms)",
                ),
            )

    async def _read_responses(self) -> None:
        """Read responses from subprocess stdout."""
        if self._process is None or self._process.stdout is None:
            return

        buffer = b""

        while True:
            try:
                # Read headers
                chunk = await self._process.stdout.read(1024)
                if not chunk:
                    break

                buffer += chunk

                while True:
                    # Parse Content-Length header
                    header_end = buffer.find(b"\r\n\r\n")
                    if header_end == -1:
                        break

                    header = buffer[:header_end].decode()
                    content_length = 0
                    for line in header.split("\r\n"):
                        if line.lower().startswith("content-length:"):
                            content_length = int(line.split(":")[1].strip())
                            break

                    if content_length == 0:
                        buffer = buffer[header_end + 4:]
                        continue

                    # Check if we have full content
                    message_start = header_end + 4
                    message_end = message_start + content_length

                    if len(buffer) < message_end:
                        break

                    # Parse message
                    content = buffer[message_start:message_end].decode()
                    buffer = buffer[message_end:]

                    try:
                        data = json.loads(content)
                        self._handle_message(data)
                    except json.JSONDecodeError:
                        logger.debug(f"Invalid JSON from LSP: {content[:100]}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"LSP reader error: {e}")
                break

    def _handle_message(self, data: Dict[str, Any]) -> None:
        """Handle an incoming message."""
        if "id" in data and "method" not in data:
            # Response to a request
            request_id = data["id"]
            if request_id in self._pending:
                future = self._pending.pop(request_id)
                if not future.done():
                    future.set_result(LSPResponse.from_dict(data))
        elif "method" in data:
            # Server notification or request
            method = data["method"]
            if method == "textDocument/publishDiagnostics":
                # Handle diagnostics
                pass
            elif method.startswith("$/"):
                # Internal notification, ignore
                pass


# =============================================================================
# Batched Client
# =============================================================================


class BatchedLSPClient(LSPClient):
    """LSP client with request batching.

    Batches multiple requests to reduce round-trip overhead.
    Useful when making many completion queries in quick succession.

    Attributes:
        inner: Underlying LSP client
        batch_interval_ms: Batching window in milliseconds
    """

    def __init__(
        self,
        inner: LSPClient,
        batch_interval_ms: Optional[int] = None,
    ):
        """Initialize batched client.

        Args:
            inner: Underlying LSP client
            batch_interval_ms: Batching interval (default from config)
        """
        super().__init__(inner.config, inner.language)
        self._inner = inner
        self._batch_interval_ms = batch_interval_ms or inner.config.batch_interval_ms
        self._pending_batch: List[Tuple[LSPRequest, asyncio.Future]] = []
        self._batch_task: Optional[asyncio.Task] = None
        self._batch_lock = asyncio.Lock()

    async def start(self) -> bool:
        """Start the underlying client."""
        return await self._inner.start()

    async def stop(self) -> None:
        """Stop the client and flush pending batch."""
        if self._batch_task is not None:
            self._batch_task.cancel()
            try:
                await self._batch_task
            except asyncio.CancelledError:
                pass

        # Flush remaining batch
        await self._flush_batch()

        await self._inner.stop()

    async def send_request(self, request: LSPRequest) -> LSPResponse:
        """Queue request for batching."""
        if request.id is None:
            # Notifications go through immediately
            return await self._inner.send_request(request)

        future: asyncio.Future = asyncio.Future()

        async with self._batch_lock:
            self._pending_batch.append((request, future))
            self._schedule_batch()

        return await future

    def _schedule_batch(self) -> None:
        """Schedule batch flush."""
        if self._batch_task is None or self._batch_task.done():
            self._batch_task = asyncio.create_task(self._batch_timer())

    async def _batch_timer(self) -> None:
        """Wait for batch interval then flush."""
        await asyncio.sleep(self._batch_interval_ms / 1000.0)
        await self._flush_batch()

    async def _flush_batch(self) -> None:
        """Send all pending requests."""
        async with self._batch_lock:
            batch = self._pending_batch
            self._pending_batch = []

        if not batch:
            return

        # Send requests concurrently
        tasks = []
        for request, future in batch:
            task = asyncio.create_task(self._send_and_resolve(request, future))
            tasks.append(task)

        await asyncio.gather(*tasks, return_exceptions=True)

    async def _send_and_resolve(
        self,
        request: LSPRequest,
        future: asyncio.Future,
    ) -> None:
        """Send request and resolve its future."""
        try:
            response = await self._inner.send_request(request)
            if not future.done():
                future.set_result(response)
        except Exception as e:
            if not future.done():
                future.set_exception(e)


# =============================================================================
# Cached Client
# =============================================================================


class CachedLSPClient(LSPClient):
    """LSP client with response caching.

    Caches responses for repeated queries at the same position.
    Uses an LRU eviction strategy.

    Attributes:
        inner: Underlying LSP client
        cache_size: Maximum cache entries
    """

    def __init__(
        self,
        inner: LSPClient,
        cache_size: Optional[int] = None,
    ):
        """Initialize cached client.

        Args:
            inner: Underlying LSP client
            cache_size: Maximum cache entries (default from config)
        """
        super().__init__(inner.config, inner.language)
        self._inner = inner
        self._cache_size = cache_size or inner.config.cache_size
        self._cache: OrderedDict[str, LSPResponse] = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0

    async def start(self) -> bool:
        """Start the underlying client."""
        return await self._inner.start()

    async def stop(self) -> None:
        """Stop the client."""
        await self._inner.stop()

    async def send_request(self, request: LSPRequest) -> LSPResponse:
        """Send request with caching."""
        if request.id is None:
            # Notifications bypass cache
            return await self._inner.send_request(request)

        # Generate cache key
        cache_key = self._make_cache_key(request)

        # Check cache
        if cache_key in self._cache:
            self._cache_hits += 1
            # Move to end (LRU)
            self._cache.move_to_end(cache_key)
            # Return cached response with new id
            cached = self._cache[cache_key]
            return LSPResponse(
                id=request.id,
                result=cached.result,
                error=cached.error,
            )

        # Cache miss
        self._cache_misses += 1
        response = await self._inner.send_request(request)

        # Only cache successful responses
        if response.is_success:
            self._cache[cache_key] = response
            # Evict oldest if over capacity
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)

        return response

    def _make_cache_key(self, request: LSPRequest) -> str:
        """Generate cache key for request."""
        # Key based on method and serialized params
        return f"{request.method}:{json.dumps(request.params, sort_keys=True)}"

    def clear_cache(self) -> None:
        """Clear the cache."""
        self._cache.clear()

    def invalidate_document(self, uri: str) -> None:
        """Invalidate all cache entries for a document.

        Args:
            uri: Document URI to invalidate
        """
        keys_to_remove = [
            key for key in self._cache if uri in key
        ]
        for key in keys_to_remove:
            del self._cache[key]

    @property
    def cache_hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self._cache_hits + self._cache_misses
        if total == 0:
            return 0.0
        return self._cache_hits / total


# =============================================================================
# Factory Function
# =============================================================================


def create_lsp_client(
    language: str,
    config: Optional[LSPConfig] = None,
    enable_batching: bool = True,
    enable_caching: bool = True,
    server_command: Optional[List[str]] = None,
) -> LSPClient:
    """Create an LSP client for a language.

    Args:
        language: Programming language
        config: LSP configuration
        enable_batching: Whether to enable request batching
        enable_caching: Whether to enable response caching
        server_command: Custom server command

    Returns:
        Configured LSP client
    """
    config = config or LSPConfig()

    # Create base client
    client: LSPClient = SubprocessLSPClient(
        config=config,
        language=language,
        server_command=server_command,
    )

    # Add caching layer
    if enable_caching:
        client = CachedLSPClient(client, cache_size=config.cache_size)

    # Add batching layer
    if enable_batching:
        client = BatchedLSPClient(client, batch_interval_ms=config.batch_interval_ms)

    return client
