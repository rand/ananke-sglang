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
"""LSP integration for IDE-grade type checking.

This module implements ChatLSP protocol support for integrating external
language servers (pyright, tsserver, rust-analyzer, gopls) with the
Ananke constrained generation system.

Based on: Hazel ChatLSP (OOPSLA 2024) - "Statically Contextualizing LLMs
for Code Synthesis"

Key Components:
- protocol.py: LSP message types and protocol definitions
- client.py: Async LSP client with batching and caching
- integration.py: TypeDomain integration layer

Design Principles:
1. LSP responses are soft hints, never hard-block tokens
2. Fallback to internal type domain on LSP timeout/error
3. Response caching for repeated position queries
4. Async batching to minimize round-trip overhead

Supported Language Servers:
- Python: pyright, pylsp
- TypeScript: tsserver
- Rust: rust-analyzer
- Go: gopls
- Kotlin: kotlin-language-server
- Swift: sourcekit-lsp
- Zig: zls
"""

from __future__ import annotations

from .protocol import (
    LSPConfig,
    LSPRequest,
    LSPResponse,
    LSPError,
    CompletionItem,
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
from .client import (
    LSPClient,
    BatchedLSPClient,
    CachedLSPClient,
    create_lsp_client,
)
from .integration import (
    LSPTypeDomain,
    LSPCompletionProvider,
    LSPDiagnosticProvider,
    create_lsp_type_domain,
)

__all__ = [
    # Protocol types
    "LSPConfig",
    "LSPRequest",
    "LSPResponse",
    "LSPError",
    "CompletionItem",
    "CompletionList",
    "Diagnostic",
    "DiagnosticSeverity",
    "Position",
    "Range",
    "TextDocumentIdentifier",
    "VersionedTextDocumentIdentifier",
    "TextDocumentContentChangeEvent",
    "TextEdit",
    # Client classes
    "LSPClient",
    "BatchedLSPClient",
    "CachedLSPClient",
    "create_lsp_client",
    # Integration
    "LSPTypeDomain",
    "LSPCompletionProvider",
    "LSPDiagnosticProvider",
    "create_lsp_type_domain",
]
