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
"""LSP protocol message types.

This module defines the data structures for Language Server Protocol
communication, following the LSP 3.17 specification.

Key Types:
- Position, Range: Text location types
- TextDocument*: Document identification and versioning
- CompletionItem, CompletionList: Completion results
- Diagnostic: Error/warning messages

References:
    - LSP Specification: https://microsoft.github.io/language-server-protocol/
    - ChatLSP (OOPSLA 2024): https://arxiv.org/abs/2409.00921
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any, Dict, List, Literal, Optional, Union


# =============================================================================
# Basic Types
# =============================================================================


@dataclass(frozen=True)
class Position:
    """Position in a text document (0-indexed).

    Attributes:
        line: Line number (0-indexed)
        character: Character offset in line (0-indexed, UTF-16 code units)
    """

    line: int
    character: int

    def to_dict(self) -> Dict[str, int]:
        """Convert to LSP JSON format."""
        return {"line": self.line, "character": self.character}

    @classmethod
    def from_dict(cls, data: Dict[str, int]) -> Position:
        """Create from LSP JSON format."""
        return cls(line=data["line"], character=data["character"])

    def __lt__(self, other: Position) -> bool:
        """Compare positions."""
        if self.line != other.line:
            return self.line < other.line
        return self.character < other.character

    def __le__(self, other: Position) -> bool:
        """Compare positions (less than or equal)."""
        return self == other or self < other

    def __gt__(self, other: Position) -> bool:
        """Compare positions (greater than)."""
        return not self <= other

    def __ge__(self, other: Position) -> bool:
        """Compare positions (greater than or equal)."""
        return not self < other


@dataclass(frozen=True)
class Range:
    """Range in a text document.

    Attributes:
        start: Start position (inclusive)
        end: End position (exclusive)
    """

    start: Position
    end: Position

    def to_dict(self) -> Dict[str, Any]:
        """Convert to LSP JSON format."""
        return {"start": self.start.to_dict(), "end": self.end.to_dict()}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Range:
        """Create from LSP JSON format."""
        return cls(
            start=Position.from_dict(data["start"]),
            end=Position.from_dict(data["end"]),
        )

    def contains(self, pos: Position) -> bool:
        """Check if position is within range."""
        return self.start <= pos < self.end


# =============================================================================
# Text Document Types
# =============================================================================


@dataclass(frozen=True)
class TextDocumentIdentifier:
    """Identifies a text document.

    Attributes:
        uri: Document URI (file:// or virtual)
    """

    uri: str

    def to_dict(self) -> Dict[str, str]:
        """Convert to LSP JSON format."""
        return {"uri": self.uri}

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> TextDocumentIdentifier:
        """Create from LSP JSON format."""
        return cls(uri=data["uri"])


@dataclass(frozen=True)
class VersionedTextDocumentIdentifier(TextDocumentIdentifier):
    """Identifies a specific version of a text document.

    Attributes:
        uri: Document URI
        version: Document version number
    """

    version: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to LSP JSON format."""
        return {"uri": self.uri, "version": self.version}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> VersionedTextDocumentIdentifier:
        """Create from LSP JSON format."""
        return cls(uri=data["uri"], version=data["version"])


@dataclass
class TextDocumentContentChangeEvent:
    """Change event for a text document.

    Attributes:
        text: New content (full or incremental)
        range: Range being replaced (None for full sync)
        range_length: Length of replaced range (deprecated)
    """

    text: str
    range: Optional[Range] = None
    range_length: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to LSP JSON format."""
        result: Dict[str, Any] = {"text": self.text}
        if self.range is not None:
            result["range"] = self.range.to_dict()
        if self.range_length is not None:
            result["rangeLength"] = self.range_length
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TextDocumentContentChangeEvent:
        """Create from LSP JSON format."""
        range_data = data.get("range")
        return cls(
            text=data["text"],
            range=Range.from_dict(range_data) if range_data else None,
            range_length=data.get("rangeLength"),
        )


@dataclass
class TextEdit:
    """A text edit to apply.

    Attributes:
        range: Range to replace
        new_text: Replacement text
    """

    range: Range
    new_text: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to LSP JSON format."""
        return {"range": self.range.to_dict(), "newText": self.new_text}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TextEdit:
        """Create from LSP JSON format."""
        return cls(
            range=Range.from_dict(data["range"]),
            new_text=data["newText"],
        )


# =============================================================================
# Completion Types
# =============================================================================


class CompletionItemKind(IntEnum):
    """Completion item kinds (LSP enum)."""

    Text = 1
    Method = 2
    Function = 3
    Constructor = 4
    Field = 5
    Variable = 6
    Class = 7
    Interface = 8
    Module = 9
    Property = 10
    Unit = 11
    Value = 12
    Enum = 13
    Keyword = 14
    Snippet = 15
    Color = 16
    File = 17
    Reference = 18
    Folder = 19
    EnumMember = 20
    Constant = 21
    Struct = 22
    Event = 23
    Operator = 24
    TypeParameter = 25


@dataclass
class CompletionItem:
    """A completion suggestion.

    Attributes:
        label: Display label
        kind: Item kind (function, variable, etc.)
        detail: Additional detail (type signature)
        documentation: Documentation string
        insert_text: Text to insert
        sort_text: Sort key
        filter_text: Filter key
        text_edit: Edit to apply
        additional_text_edits: Additional edits (imports, etc.)
        data: Custom data for resolve
    """

    label: str
    kind: Optional[CompletionItemKind] = None
    detail: Optional[str] = None
    documentation: Optional[str] = None
    insert_text: Optional[str] = None
    sort_text: Optional[str] = None
    filter_text: Optional[str] = None
    text_edit: Optional[TextEdit] = None
    additional_text_edits: Optional[List[TextEdit]] = None
    data: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to LSP JSON format."""
        result: Dict[str, Any] = {"label": self.label}
        if self.kind is not None:
            result["kind"] = self.kind.value
        if self.detail is not None:
            result["detail"] = self.detail
        if self.documentation is not None:
            result["documentation"] = self.documentation
        if self.insert_text is not None:
            result["insertText"] = self.insert_text
        if self.sort_text is not None:
            result["sortText"] = self.sort_text
        if self.filter_text is not None:
            result["filterText"] = self.filter_text
        if self.text_edit is not None:
            result["textEdit"] = self.text_edit.to_dict()
        if self.additional_text_edits is not None:
            result["additionalTextEdits"] = [
                e.to_dict() for e in self.additional_text_edits
            ]
        if self.data is not None:
            result["data"] = self.data
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CompletionItem:
        """Create from LSP JSON format."""
        text_edit_data = data.get("textEdit")
        additional_edits = data.get("additionalTextEdits")
        return cls(
            label=data["label"],
            kind=CompletionItemKind(data["kind"]) if "kind" in data else None,
            detail=data.get("detail"),
            documentation=data.get("documentation"),
            insert_text=data.get("insertText"),
            sort_text=data.get("sortText"),
            filter_text=data.get("filterText"),
            text_edit=TextEdit.from_dict(text_edit_data) if text_edit_data else None,
            additional_text_edits=[TextEdit.from_dict(e) for e in additional_edits]
            if additional_edits
            else None,
            data=data.get("data"),
        )

    @property
    def effective_insert_text(self) -> str:
        """Get the text that would be inserted."""
        if self.text_edit is not None:
            return self.text_edit.new_text
        if self.insert_text is not None:
            return self.insert_text
        return self.label


@dataclass
class CompletionList:
    """List of completion items.

    Attributes:
        is_incomplete: Whether more items are available
        items: Completion items
    """

    is_incomplete: bool
    items: List[CompletionItem]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to LSP JSON format."""
        return {
            "isIncomplete": self.is_incomplete,
            "items": [item.to_dict() for item in self.items],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CompletionList:
        """Create from LSP JSON format."""
        return cls(
            is_incomplete=data.get("isIncomplete", False),
            items=[CompletionItem.from_dict(item) for item in data.get("items", [])],
        )

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)


# =============================================================================
# Diagnostic Types
# =============================================================================


class DiagnosticSeverity(IntEnum):
    """Diagnostic severity levels."""

    Error = 1
    Warning = 2
    Information = 3
    Hint = 4


@dataclass
class Diagnostic:
    """A diagnostic (error/warning).

    Attributes:
        range: Location of the diagnostic
        severity: Error, warning, info, or hint
        code: Diagnostic code
        source: Source of the diagnostic
        message: Diagnostic message
        related_information: Related diagnostics
    """

    range: Range
    message: str
    severity: Optional[DiagnosticSeverity] = None
    code: Optional[Union[int, str]] = None
    source: Optional[str] = None
    related_information: Optional[List[Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to LSP JSON format."""
        result: Dict[str, Any] = {
            "range": self.range.to_dict(),
            "message": self.message,
        }
        if self.severity is not None:
            result["severity"] = self.severity.value
        if self.code is not None:
            result["code"] = self.code
        if self.source is not None:
            result["source"] = self.source
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Diagnostic:
        """Create from LSP JSON format."""
        return cls(
            range=Range.from_dict(data["range"]),
            message=data["message"],
            severity=DiagnosticSeverity(data["severity"])
            if "severity" in data
            else None,
            code=data.get("code"),
            source=data.get("source"),
            related_information=data.get("relatedInformation"),
        )

    @property
    def is_error(self) -> bool:
        """Check if this is an error."""
        return self.severity == DiagnosticSeverity.Error

    @property
    def is_warning(self) -> bool:
        """Check if this is a warning."""
        return self.severity == DiagnosticSeverity.Warning


# =============================================================================
# Request/Response Types
# =============================================================================


@dataclass
class LSPConfig:
    """Configuration for LSP clients.

    Attributes:
        python: Python language server command
        typescript: TypeScript language server command
        rust: Rust language server command
        go: Go language server command
        kotlin: Kotlin language server command
        swift: Swift language server command
        zig: Zig language server command
        timeout_ms: Request timeout in milliseconds
        batch_interval_ms: Batching interval in milliseconds
        cache_size: Maximum cache entries
    """

    python: str = "pyright-langserver"
    typescript: str = "typescript-language-server"
    rust: str = "rust-analyzer"
    go: str = "gopls"
    kotlin: str = "kotlin-language-server"
    swift: str = "sourcekit-lsp"
    zig: str = "zls"
    timeout_ms: int = 100
    batch_interval_ms: int = 10
    cache_size: int = 1000

    def get_server_for_language(self, language: str) -> Optional[str]:
        """Get server command for a language."""
        mapping = {
            "python": self.python,
            "typescript": self.typescript,
            "javascript": self.typescript,
            "rust": self.rust,
            "go": self.go,
            "kotlin": self.kotlin,
            "swift": self.swift,
            "zig": self.zig,
        }
        return mapping.get(language.lower())


@dataclass
class LSPRequest:
    """An LSP request.

    Attributes:
        method: LSP method name
        params: Request parameters
        id: Request ID
    """

    method: str
    params: Dict[str, Any]
    id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-RPC format."""
        result: Dict[str, Any] = {
            "jsonrpc": "2.0",
            "method": self.method,
            "params": self.params,
        }
        if self.id is not None:
            result["id"] = self.id
        return result


@dataclass
class LSPResponse:
    """An LSP response.

    Attributes:
        id: Request ID this responds to
        result: Response result (if success)
        error: Error information (if failure)
    """

    id: int
    result: Optional[Any] = None
    error: Optional[LSPError] = None

    @property
    def is_success(self) -> bool:
        """Check if response indicates success."""
        return self.error is None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> LSPResponse:
        """Create from JSON-RPC format."""
        error_data = data.get("error")
        return cls(
            id=data.get("id", 0),
            result=data.get("result"),
            error=LSPError.from_dict(error_data) if error_data else None,
        )


@dataclass
class LSPError:
    """An LSP error.

    Attributes:
        code: Error code
        message: Error message
        data: Additional error data
    """

    code: int
    message: str
    data: Optional[Any] = None

    # Standard LSP error codes
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
    SERVER_NOT_INITIALIZED = -32002
    UNKNOWN_ERROR_CODE = -32001
    REQUEST_CANCELLED = -32800
    CONTENT_MODIFIED = -32801

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-RPC format."""
        result: Dict[str, Any] = {"code": self.code, "message": self.message}
        if self.data is not None:
            result["data"] = self.data
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> LSPError:
        """Create from JSON-RPC format."""
        return cls(
            code=data["code"],
            message=data["message"],
            data=data.get("data"),
        )

    @property
    def is_timeout(self) -> bool:
        """Check if this is a timeout error."""
        return "timeout" in self.message.lower()
