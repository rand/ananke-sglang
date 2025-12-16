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
"""LSP integration with Ananke type domain.

This module integrates Language Server Protocol clients with the Ananke
constrained generation system, providing IDE-grade type checking and
completion suggestions.

Based on ChatLSP (OOPSLA 2024):
- LSP provides accurate type context from real language servers
- Completions inform token mask generation
- Diagnostics provide soft verification signals

Key Design Decisions:
1. LSP responses are SOFT hints - never hard-block tokens
2. Fallback to internal type domain on LSP timeout/error
3. Async operation to minimize generation latency
4. Document sync for incremental generation

Integration Points:
- LSPTypeDomain: Wraps type domain with LSP enhancement
- LSPCompletionProvider: Converts LSP completions to token masks
- LSPDiagnosticProvider: Uses diagnostics for verification

References:
    - ChatLSP (OOPSLA 2024): https://arxiv.org/abs/2409.00921
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple

import torch

from .protocol import (
    LSPConfig,
    CompletionItem,
    CompletionItemKind,
    CompletionList,
    Diagnostic,
    DiagnosticSeverity,
    Position,
    Range,
    TextDocumentContentChangeEvent,
)
from .client import LSPClient, create_lsp_client

logger = logging.getLogger(__name__)


# =============================================================================
# Integration Statistics
# =============================================================================


@dataclass
class LSPIntegrationStats:
    """Statistics for LSP integration.

    Attributes:
        total_queries: Total LSP queries made
        successful_queries: Queries that returned results
        timeout_queries: Queries that timed out
        error_queries: Queries that errored
        cache_hits: Cache hits (if using cached client)
        avg_latency_ms: Average query latency
        total_completions: Total completions received
        completions_used: Completions used in mask
    """

    total_queries: int = 0
    successful_queries: int = 0
    timeout_queries: int = 0
    error_queries: int = 0
    cache_hits: int = 0
    avg_latency_ms: float = 0.0
    total_completions: int = 0
    completions_used: int = 0

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_queries == 0:
            return 0.0
        return self.successful_queries / self.total_queries

    @property
    def completion_use_rate(self) -> float:
        """Calculate completion use rate."""
        if self.total_completions == 0:
            return 0.0
        return self.completions_used / self.total_completions


# =============================================================================
# LSP Completion Provider
# =============================================================================


@dataclass
class LSPCompletionContext:
    """Context for LSP completion query.

    Attributes:
        uri: Document URI
        position: Cursor position
        prefix_text: Text before cursor
        language: Programming language
        expected_type: Expected type at position (if known)
    """

    uri: str
    position: Position
    prefix_text: str
    language: str
    expected_type: Optional[str] = None


class LSPCompletionProvider:
    """Provides completions from LSP for mask generation.

    Converts LSP completion items into token masks that allow
    tokens matching the completions.

    Attributes:
        client: LSP client
        tokenizer: Tokenizer for converting text to tokens
        vocab_size: Vocabulary size
        stats: Query statistics
    """

    def __init__(
        self,
        client: LSPClient,
        tokenizer: Any,
        vocab_size: int = 32000,
    ):
        """Initialize completion provider.

        Args:
            client: LSP client
            tokenizer: Tokenizer instance
            vocab_size: Vocabulary size
        """
        self._client = client
        self._tokenizer = tokenizer
        self._vocab_size = vocab_size
        self.stats = LSPIntegrationStats()
        self._completion_cache: Dict[str, List[str]] = {}

    async def get_completion_tokens(
        self,
        context: LSPCompletionContext,
    ) -> Set[int]:
        """Get token IDs that match LSP completions.

        Args:
            context: Completion context

        Returns:
            Set of allowed token IDs
        """
        start_time = time.perf_counter()
        self.stats.total_queries += 1

        try:
            completions = await self._client.get_completions(
                uri=context.uri,
                position=context.position,
            )

            if completions is None or len(completions) == 0:
                self.stats.error_queries += 1
                return set()

            self.stats.successful_queries += 1
            self.stats.total_completions += len(completions.items)

            # Convert completions to token IDs
            tokens = self._completions_to_tokens(completions, context)
            self.stats.completions_used += len(tokens)

            # Update latency
            latency = (time.perf_counter() - start_time) * 1000
            self._update_avg_latency(latency)

            return tokens

        except asyncio.TimeoutError:
            self.stats.timeout_queries += 1
            return set()
        except Exception as e:
            logger.debug(f"LSP completion error: {e}")
            self.stats.error_queries += 1
            return set()

    def _completions_to_tokens(
        self,
        completions: CompletionList,
        context: LSPCompletionContext,
    ) -> Set[int]:
        """Convert LSP completions to token IDs.

        Args:
            completions: LSP completion list
            context: Completion context

        Returns:
            Set of token IDs matching completions
        """
        tokens: Set[int] = set()

        for item in completions.items:
            # Get the text to be inserted
            insert_text = item.effective_insert_text

            # Skip empty or very long insertions
            if not insert_text or len(insert_text) > 100:
                continue

            # Tokenize the insertion text
            try:
                # Get tokens for the completion text
                item_tokens = self._tokenizer.encode(insert_text, add_special_tokens=False)
                if item_tokens:
                    # Add the first token (what would be generated next)
                    tokens.add(item_tokens[0])
            except Exception:
                # Tokenization failed, skip this completion
                continue

        return tokens

    def _update_avg_latency(self, latency_ms: float) -> None:
        """Update rolling average latency."""
        n = self.stats.successful_queries
        if n == 1:
            self.stats.avg_latency_ms = latency_ms
        else:
            # Exponential moving average
            alpha = 0.1
            self.stats.avg_latency_ms = (
                alpha * latency_ms + (1 - alpha) * self.stats.avg_latency_ms
            )

    def get_completion_mask(
        self,
        allowed_tokens: Set[int],
    ) -> torch.Tensor:
        """Create a mask tensor from allowed tokens.

        Args:
            allowed_tokens: Set of allowed token IDs

        Returns:
            Boolean mask tensor (True = allowed)
        """
        mask = torch.zeros(self._vocab_size, dtype=torch.bool)
        for token in allowed_tokens:
            if 0 <= token < self._vocab_size:
                mask[token] = True
        return mask


# =============================================================================
# LSP Diagnostic Provider
# =============================================================================


class LSPDiagnosticProvider:
    """Provides diagnostics from LSP for verification.

    Uses LSP diagnostics (errors/warnings) as soft signals
    for code quality verification.

    Attributes:
        diagnostics: Current diagnostics by document
        severity_weights: Scoring weights by severity
    """

    def __init__(self):
        """Initialize diagnostic provider."""
        self._diagnostics: Dict[str, List[Diagnostic]] = {}
        self.severity_weights = {
            DiagnosticSeverity.Error: 0.0,      # Errors reduce score to 0
            DiagnosticSeverity.Warning: 0.5,   # Warnings reduce by half
            DiagnosticSeverity.Information: 0.9,  # Info slight reduction
            DiagnosticSeverity.Hint: 0.95,     # Hints minimal impact
        }

    def update_diagnostics(self, uri: str, diagnostics: List[Diagnostic]) -> None:
        """Update diagnostics for a document.

        Args:
            uri: Document URI
            diagnostics: New diagnostics
        """
        self._diagnostics[uri] = diagnostics

    def clear_diagnostics(self, uri: str) -> None:
        """Clear diagnostics for a document.

        Args:
            uri: Document URI
        """
        self._diagnostics.pop(uri, None)

    def get_score(self, uri: str, range_: Optional[Range] = None) -> float:
        """Get verification score based on diagnostics.

        Args:
            uri: Document URI
            range_: Optional range to check (None = whole document)

        Returns:
            Score from 0.0 to 1.0
        """
        diagnostics = self._diagnostics.get(uri, [])
        if not diagnostics:
            return 1.0

        # Filter by range if specified
        if range_ is not None:
            diagnostics = [
                d for d in diagnostics
                if self._ranges_overlap(d.range, range_)
            ]

        if not diagnostics:
            return 1.0

        # Calculate score based on severity
        score = 1.0
        for diagnostic in diagnostics:
            severity = diagnostic.severity or DiagnosticSeverity.Information
            weight = self.severity_weights.get(severity, 0.9)
            score = min(score, weight)

        return score

    def has_errors(self, uri: str) -> bool:
        """Check if document has errors.

        Args:
            uri: Document URI

        Returns:
            True if document has error diagnostics
        """
        diagnostics = self._diagnostics.get(uri, [])
        return any(d.is_error for d in diagnostics)

    def _ranges_overlap(self, r1: Range, r2: Range) -> bool:
        """Check if two ranges overlap."""
        # Ranges overlap if neither is completely before or after the other
        if r1.end < r2.start or r2.end < r1.start:
            return False
        return True


# =============================================================================
# LSP Type Domain
# =============================================================================


class LSPTypeDomain:
    """Type domain enhanced with LSP integration.

    Wraps an internal type domain and enhances it with LSP-based
    type checking. LSP responses are treated as soft hints that
    can further constrain (but never override) the internal domain.

    Design: LSP is advisory only. If LSP suggests tokens A,B,C but
    internal domain allows A,B,C,D,E, we use A,B,C. But if LSP fails
    or times out, we fall back to A,B,C,D,E (soundness preserved).

    Attributes:
        client: LSP client
        completion_provider: Completion to token converter
        diagnostic_provider: Diagnostic scorer
        fallback_enabled: Whether to fall back on LSP failure
    """

    def __init__(
        self,
        client: LSPClient,
        tokenizer: Any,
        vocab_size: int = 32000,
        fallback_enabled: bool = True,
    ):
        """Initialize LSP type domain.

        Args:
            client: LSP client
            tokenizer: Tokenizer instance
            vocab_size: Vocabulary size
            fallback_enabled: Whether to fall back on failure
        """
        self._client = client
        self._tokenizer = tokenizer
        self._vocab_size = vocab_size
        self._fallback_enabled = fallback_enabled

        self.completion_provider = LSPCompletionProvider(
            client=client,
            tokenizer=tokenizer,
            vocab_size=vocab_size,
        )
        self.diagnostic_provider = LSPDiagnosticProvider()

        # Document state
        self._open_documents: Dict[str, int] = {}  # uri -> version
        self._document_content: Dict[str, str] = {}

    async def start(self, root_uri: str) -> bool:
        """Start the LSP client.

        Args:
            root_uri: Workspace root URI

        Returns:
            True if started successfully
        """
        if not await self._client.start():
            return False
        return await self._client.initialize(root_uri)

    async def stop(self) -> None:
        """Stop the LSP client."""
        await self._client.shutdown()
        await self._client.stop()

    async def open_document(
        self,
        uri: str,
        language_id: str,
        content: str,
    ) -> None:
        """Open a document for LSP tracking.

        Args:
            uri: Document URI
            language_id: Language identifier
            content: Initial content
        """
        version = 1
        self._open_documents[uri] = version
        self._document_content[uri] = content

        await self._client.open_document(
            uri=uri,
            language_id=language_id,
            version=version,
            text=content,
        )

    async def update_document(
        self,
        uri: str,
        content: str,
    ) -> None:
        """Update document content.

        Args:
            uri: Document URI
            content: New content
        """
        if uri not in self._open_documents:
            return

        version = self._open_documents[uri] + 1
        self._open_documents[uri] = version
        self._document_content[uri] = content

        # Full sync for simplicity
        change = TextDocumentContentChangeEvent(text=content)
        await self._client.update_document(
            uri=uri,
            version=version,
            changes=[change],
        )

        # Clear diagnostics (will be re-received)
        self.diagnostic_provider.clear_diagnostics(uri)

    async def close_document(self, uri: str) -> None:
        """Close a document.

        Args:
            uri: Document URI
        """
        if uri in self._open_documents:
            await self._client.close_document(uri)
            del self._open_documents[uri]
            del self._document_content[uri]
            self.diagnostic_provider.clear_diagnostics(uri)

    async def get_token_mask(
        self,
        uri: str,
        position: Position,
        base_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get token mask enhanced with LSP completions.

        Args:
            uri: Document URI
            position: Cursor position
            base_mask: Base mask from internal type domain

        Returns:
            Token mask (True = allowed)
        """
        # Start with base mask or all-allowed
        if base_mask is not None:
            mask = base_mask.clone()
        else:
            mask = torch.ones(self._vocab_size, dtype=torch.bool)

        # Get LSP completions
        context = LSPCompletionContext(
            uri=uri,
            position=position,
            prefix_text=self._document_content.get(uri, ""),
            language=self._client.language,
        )

        completion_tokens = await self.completion_provider.get_completion_tokens(context)

        if completion_tokens:
            # Create LSP mask
            lsp_mask = self.completion_provider.get_completion_mask(completion_tokens)

            # Intersect with base mask (LSP can only further constrain)
            mask &= lsp_mask
        elif not self._fallback_enabled:
            # LSP failed and fallback disabled - block everything
            mask.fill_(False)

        return mask

    async def get_type_at_position(
        self,
        uri: str,
        position: Position,
    ) -> Optional[str]:
        """Get type information at a position.

        Args:
            uri: Document URI
            position: Cursor position

        Returns:
            Type string or None
        """
        hover = await self._client.get_hover(uri, position)
        if hover:
            # Extract type from hover (language-specific parsing)
            return self._extract_type_from_hover(hover)
        return None

    def _extract_type_from_hover(self, hover: str) -> Optional[str]:
        """Extract type annotation from hover text.

        Args:
            hover: Hover text from LSP

        Returns:
            Type string or None
        """
        # Look for common patterns like "def foo() -> int" or "x: int"
        import re

        # Pattern: ": Type" or "-> Type"
        patterns = [
            r":\s*([A-Za-z_]\w*(?:\[[^\]]+\])?)",  # : Type or : Type[...]
            r"->\s*([A-Za-z_]\w*(?:\[[^\]]+\])?)",  # -> Type
            r"type:\s*([A-Za-z_]\w*(?:\[[^\]]+\])?)",  # type: Type
        ]

        for pattern in patterns:
            match = re.search(pattern, hover)
            if match:
                return match.group(1)

        return None

    def get_verification_score(self, uri: str) -> float:
        """Get verification score for a document.

        Args:
            uri: Document URI

        Returns:
            Score from 0.0 to 1.0
        """
        return self.diagnostic_provider.get_score(uri)

    @property
    def stats(self) -> LSPIntegrationStats:
        """Get integration statistics."""
        return self.completion_provider.stats


# =============================================================================
# Factory Function
# =============================================================================


def create_lsp_type_domain(
    language: str,
    tokenizer: Any,
    vocab_size: int = 32000,
    config: Optional[LSPConfig] = None,
    enable_batching: bool = True,
    enable_caching: bool = True,
) -> LSPTypeDomain:
    """Create an LSP-enhanced type domain.

    Args:
        language: Programming language
        tokenizer: Tokenizer instance
        vocab_size: Vocabulary size
        config: LSP configuration
        enable_batching: Whether to enable request batching
        enable_caching: Whether to enable response caching

    Returns:
        Configured LSPTypeDomain
    """
    client = create_lsp_client(
        language=language,
        config=config,
        enable_batching=enable_batching,
        enable_caching=enable_caching,
    )

    return LSPTypeDomain(
        client=client,
        tokenizer=tokenizer,
        vocab_size=vocab_size,
    )
