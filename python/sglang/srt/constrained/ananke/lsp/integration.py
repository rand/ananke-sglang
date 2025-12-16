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


@dataclass
class DiagnosticConfidence:
    """Confidence score for a diagnostic.

    High confidence diagnostics (>= threshold) can trigger hard blocking.
    """
    confidence: float  # 0.0 to 1.0
    is_type_error: bool  # True if this is a type-related error
    source: str  # Source of the diagnostic (e.g., "pyright", "typescript")
    can_hard_block: bool  # Whether this diagnostic can trigger hard blocking


class LSPDiagnosticProvider:
    """Provides diagnostics from LSP for verification with confidence scoring.

    Phase 2.2 enhancement: Implements confidence-based hard blocking for
    high-confidence type errors. When LSP reports a type error with
    confidence >= 0.95, we hard-block tokens that would cause that error.

    Uses LSP diagnostics (errors/warnings) as signals for code quality
    verification, with confidence scoring for hard vs soft blocking.

    Attributes:
        diagnostics: Current diagnostics by document
        severity_weights: Scoring weights by severity
        hard_block_threshold: Confidence threshold for hard blocking
        enable_hard_blocking: Whether hard blocking is enabled
    """

    # Error codes that indicate type errors (language-specific)
    TYPE_ERROR_CODES = frozenset({
        # Python (Pyright/Pylsp)
        "reportGeneralTypeIssues",
        "reportArgumentType",
        "reportReturnType",
        "reportAssignmentType",
        "reportIndexIssue",
        "reportCallIssue",
        # TypeScript
        "2322",  # Type 'X' is not assignable to type 'Y'
        "2345",  # Argument of type 'X' is not assignable to parameter of type 'Y'
        "2339",  # Property 'X' does not exist on type 'Y'
        "2304",  # Cannot find name 'X'
        # Rust (rust-analyzer)
        "E0308",  # mismatched types
        "E0277",  # trait bound not satisfied
        "E0382",  # use of moved value
    })

    # Sources with high reliability for type errors
    HIGH_CONFIDENCE_SOURCES = frozenset({
        "pyright",
        "typescript",
        "rust-analyzer",
        "gopls",
    })

    def __init__(
        self,
        hard_block_threshold: float = 0.95,
        enable_hard_blocking: bool = True,
    ):
        """Initialize diagnostic provider.

        Args:
            hard_block_threshold: Confidence threshold for hard blocking
            enable_hard_blocking: Whether to enable hard blocking
        """
        self._diagnostics: Dict[str, List[Diagnostic]] = {}
        self._confidence_cache: Dict[str, List[DiagnosticConfidence]] = {}
        self.hard_block_threshold = hard_block_threshold
        self.enable_hard_blocking = enable_hard_blocking

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
        # Compute confidence scores for new diagnostics
        self._confidence_cache[uri] = [
            self._compute_confidence(d) for d in diagnostics
        ]

    def clear_diagnostics(self, uri: str) -> None:
        """Clear diagnostics for a document.

        Args:
            uri: Document URI
        """
        self._diagnostics.pop(uri, None)
        self._confidence_cache.pop(uri, None)

    def _compute_confidence(self, diagnostic: Diagnostic) -> DiagnosticConfidence:
        """Compute confidence score for a diagnostic.

        Confidence is based on:
        - Source reliability (pyright, typescript > generic)
        - Error code specificity
        - Severity (errors > warnings)
        - Presence of type information in message

        Args:
            diagnostic: The diagnostic to score

        Returns:
            DiagnosticConfidence with computed scores
        """
        confidence = 0.5  # Base confidence
        source = diagnostic.source or "unknown"
        is_type_error = False

        # Boost confidence for reliable sources
        if source.lower() in self.HIGH_CONFIDENCE_SOURCES:
            confidence += 0.3

        # Check if this is a type error
        code = str(diagnostic.code) if diagnostic.code else ""
        if code in self.TYPE_ERROR_CODES:
            is_type_error = True
            confidence += 0.15

        # Check message for type-related keywords
        message_lower = diagnostic.message.lower() if diagnostic.message else ""
        type_keywords = ["type", "expected", "assignable", "cannot be", "incompatible"]
        if any(kw in message_lower for kw in type_keywords):
            is_type_error = True
            confidence += 0.1

        # Severity boost
        if diagnostic.severity == DiagnosticSeverity.Error:
            confidence += 0.1
        elif diagnostic.severity == DiagnosticSeverity.Warning:
            confidence += 0.05

        # Cap at 1.0
        confidence = min(1.0, confidence)

        # Determine if this can hard-block
        can_hard_block = (
            self.enable_hard_blocking
            and is_type_error
            and confidence >= self.hard_block_threshold
            and diagnostic.severity == DiagnosticSeverity.Error
        )

        return DiagnosticConfidence(
            confidence=confidence,
            is_type_error=is_type_error,
            source=source,
            can_hard_block=can_hard_block,
        )

    def get_high_confidence_type_errors(
        self,
        uri: str,
        range_: Optional[Range] = None,
    ) -> List[Tuple[Diagnostic, DiagnosticConfidence]]:
        """Get high-confidence type errors that can trigger hard blocking.

        Args:
            uri: Document URI
            range_: Optional range to filter

        Returns:
            List of (diagnostic, confidence) tuples for hard-blocking errors
        """
        diagnostics = self._diagnostics.get(uri, [])
        confidences = self._confidence_cache.get(uri, [])

        if not diagnostics or not confidences:
            return []

        result = []
        for diag, conf in zip(diagnostics, confidences):
            if not conf.can_hard_block:
                continue

            # Filter by range if specified
            if range_ is not None:
                if not self._ranges_overlap(diag.range, range_):
                    continue

            result.append((diag, conf))

        return result

    def should_hard_block(
        self,
        uri: str,
        range_: Optional[Range] = None,
    ) -> bool:
        """Check if we should hard-block based on diagnostics.

        Returns True if there's at least one high-confidence type error
        in the given range.

        Args:
            uri: Document URI
            range_: Optional range to check

        Returns:
            True if hard blocking should be triggered
        """
        return len(self.get_high_confidence_type_errors(uri, range_)) > 0

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

    Phase 2.2 Enhancements:
    - Confidence-based hard blocking for high-confidence type errors
    - Async prefetch for next-token prediction
    - Type refinement from hover information
    - Configurable hard-block threshold (default 0.95)

    Design: LSP is advisory only EXCEPT for high-confidence type errors.
    When LSP reports a type error with confidence >= 0.95 from a reliable
    source (pyright, typescript, etc.), we hard-block tokens that would
    cause that error.

    Attributes:
        client: LSP client
        completion_provider: Completion to token converter
        diagnostic_provider: Diagnostic scorer with confidence tracking
        fallback_enabled: Whether to fall back on LSP failure
        hard_block_enabled: Whether to enable hard blocking for type errors
        prefetch_enabled: Whether to enable async prefetch
    """

    def __init__(
        self,
        client: LSPClient,
        tokenizer: Any,
        vocab_size: int = 32000,
        fallback_enabled: bool = True,
        hard_block_enabled: bool = True,
        hard_block_threshold: float = 0.95,
        prefetch_enabled: bool = True,
    ):
        """Initialize LSP type domain.

        Args:
            client: LSP client
            tokenizer: Tokenizer instance
            vocab_size: Vocabulary size
            fallback_enabled: Whether to fall back on failure
            hard_block_enabled: Whether to enable hard blocking
            hard_block_threshold: Confidence threshold for hard blocking
            prefetch_enabled: Whether to enable async prefetch
        """
        self._client = client
        self._tokenizer = tokenizer
        self._vocab_size = vocab_size
        self._fallback_enabled = fallback_enabled
        self._hard_block_enabled = hard_block_enabled
        self._prefetch_enabled = prefetch_enabled

        self.completion_provider = LSPCompletionProvider(
            client=client,
            tokenizer=tokenizer,
            vocab_size=vocab_size,
        )
        self.diagnostic_provider = LSPDiagnosticProvider(
            hard_block_threshold=hard_block_threshold,
            enable_hard_blocking=hard_block_enabled,
        )

        # Document state
        self._open_documents: Dict[str, int] = {}  # uri -> version
        self._document_content: Dict[str, str] = {}

        # Prefetch cache
        self._prefetch_cache: Dict[str, Tuple[Position, Set[int]]] = {}
        self._prefetch_task: Optional[asyncio.Task] = None

        # Type refinement cache
        self._type_cache: Dict[Tuple[str, int, int], str] = {}  # (uri, line, col) -> type

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
        check_diagnostics: bool = True,
    ) -> torch.Tensor:
        """Get token mask enhanced with LSP completions and hard blocking.

        Phase 2.2 enhancement: Now checks for high-confidence type errors
        and can hard-block tokens that would cause those errors.

        Args:
            uri: Document URI
            position: Cursor position
            base_mask: Base mask from internal type domain
            check_diagnostics: Whether to check diagnostics for hard blocking

        Returns:
            Token mask (True = allowed)
        """
        # Check for prefetch cache hit
        prefetch_key = uri
        if prefetch_key in self._prefetch_cache:
            cached_pos, cached_tokens = self._prefetch_cache[prefetch_key]
            if cached_pos.line == position.line and cached_pos.character == position.character:
                # Cache hit - use prefetched tokens
                del self._prefetch_cache[prefetch_key]
                if cached_tokens:
                    mask = self.completion_provider.get_completion_mask(cached_tokens)
                    if base_mask is not None:
                        mask &= base_mask
                    return mask

        # Start with base mask or all-allowed
        if base_mask is not None:
            mask = base_mask.clone()
        else:
            mask = torch.ones(self._vocab_size, dtype=torch.bool)

        # Phase 2.2: Check for high-confidence type errors
        if check_diagnostics and self._hard_block_enabled:
            cursor_range = Range(start=position, end=position)
            if self.diagnostic_provider.should_hard_block(uri, cursor_range):
                # Get high-confidence type errors at this position
                errors = self.diagnostic_provider.get_high_confidence_type_errors(uri, cursor_range)
                for diag, conf in errors:
                    logger.info(
                        f"Hard-blocking due to type error (confidence={conf.confidence:.2f}): "
                        f"{diag.message}"
                    )
                # Hard block by returning empty mask
                mask.fill_(False)
                return mask

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

        # Start prefetch for next position if enabled
        if self._prefetch_enabled:
            next_position = Position(line=position.line, character=position.character + 1)
            self._start_prefetch(uri, next_position)

        return mask

    def _start_prefetch(self, uri: str, position: Position) -> None:
        """Start async prefetch for next position.

        Args:
            uri: Document URI
            position: Position to prefetch for
        """
        # Cancel existing prefetch if any
        if self._prefetch_task is not None and not self._prefetch_task.done():
            self._prefetch_task.cancel()

        async def prefetch():
            context = LSPCompletionContext(
                uri=uri,
                position=position,
                prefix_text=self._document_content.get(uri, ""),
                language=self._client.language,
            )
            try:
                tokens = await self.completion_provider.get_completion_tokens(context)
                self._prefetch_cache[uri] = (position, tokens)
            except asyncio.CancelledError:
                pass  # Expected if we start a new prefetch
            except Exception as e:
                logger.debug(f"Prefetch failed: {e}")

        self._prefetch_task = asyncio.create_task(prefetch())

    async def get_type_at_position(
        self,
        uri: str,
        position: Position,
        use_cache: bool = True,
    ) -> Optional[str]:
        """Get type information at a position with caching.

        Args:
            uri: Document URI
            position: Cursor position
            use_cache: Whether to use type cache

        Returns:
            Type string or None
        """
        # Check cache
        cache_key = (uri, position.line, position.character)
        if use_cache and cache_key in self._type_cache:
            return self._type_cache[cache_key]

        hover = await self._client.get_hover(uri, position)
        if hover:
            # Extract type from hover (language-specific parsing)
            type_str = self._extract_type_from_hover(hover)
            if type_str:
                self._type_cache[cache_key] = type_str
            return type_str
        return None

    async def refine_type_from_lsp(
        self,
        uri: str,
        position: Position,
        current_type: Optional[str] = None,
    ) -> Optional[str]:
        """Refine type using LSP hover information.

        Uses LSP hover to get more specific type information that
        can help constrain generation.

        Args:
            uri: Document URI
            position: Cursor position
            current_type: Current type from internal domain

        Returns:
            Refined type string or current_type if no refinement
        """
        lsp_type = await self.get_type_at_position(uri, position)
        if lsp_type is None:
            return current_type

        # If no current type, use LSP type
        if current_type is None:
            return lsp_type

        # If current type is generic (Any, object), prefer LSP type
        generic_types = {"Any", "object", "unknown", "any"}
        if current_type in generic_types:
            return lsp_type

        # If LSP type is more specific, prefer it
        # (e.g., current_type="int" but LSP says "Literal[0, 1]")
        if "[" in lsp_type and "[" not in current_type:
            return lsp_type

        return current_type

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
