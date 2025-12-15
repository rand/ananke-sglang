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
"""Multi-format constraint specification parser for Ananke.

This module provides the ConstraintSpecParser class that handles parsing
constraint specifications from multiple input formats:

1. JSON inline - Dict or JSON string in request body
2. External URI - Reference to external constraint specification
3. Dense binary - MessagePack or Protobuf encoded binary data
4. Legacy - Traditional json_schema/regex/ebnf/structural_tag parameters

The parser follows a priority order: binary > uri > inline > legacy

Example:
    >>> parser = ConstraintSpecParser()
    >>> spec = parser.parse(
    ...     constraint_spec={"json_schema": "{...}", "language": "python"}
    ... )
    >>> # Or from legacy params
    >>> spec = parser.parse(json_schema="{...}")
"""

from __future__ import annotations

import base64
import json
import logging
from typing import Any, Dict, Optional, Union
from urllib.parse import urlparse

from .constraint_spec import (
    CacheScope,
    ConstraintSource,
    ConstraintSpec,
    LanguageDetection,
)

logger = logging.getLogger(__name__)


class ConstraintSpecParser:
    """Multi-format parser for constraint specifications.

    Parses ConstraintSpec from various input formats with automatic format
    detection and validation.

    Supported formats:
    - JSON inline: Dict or JSON string
    - URI reference: ananke://, http://, https://, file://
    - Binary: MessagePack or Protobuf encoded
    - Legacy: Individual json_schema/regex/ebnf/structural_tag params

    Attributes:
        uri_cache: Cache for fetched URI specifications
        strict: Whether to raise on validation errors

    Example:
        >>> parser = ConstraintSpecParser()
        >>> spec = parser.parse(
        ...     constraint_spec={"json_schema": "{...}", "language": "python"}
        ... )
    """

    SUPPORTED_URI_SCHEMES = frozenset({"ananke", "http", "https", "file"})
    SUPPORTED_BINARY_FORMATS = frozenset({"msgpack", "protobuf", "json"})

    def __init__(
        self,
        strict: bool = False,
        enable_uri_fetch: bool = True,
        uri_cache_size: int = 100,
    ) -> None:
        """Initialize the parser.

        Args:
            strict: If True, raise on validation errors. If False, log warnings.
            enable_uri_fetch: If True, fetch external URIs. If False, return None.
            uri_cache_size: Maximum number of URIs to cache.
        """
        self._strict = strict
        self._enable_uri_fetch = enable_uri_fetch
        self._uri_cache: Dict[str, ConstraintSpec] = {}
        self._uri_cache_size = uri_cache_size

    def parse(
        self,
        constraint_spec: Optional[Union[str, Dict[str, Any]]] = None,
        constraint_uri: Optional[str] = None,
        constraint_bytes: Optional[Union[bytes, str]] = None,
        constraint_bytes_format: str = "msgpack",
        # Legacy parameters for backward compatibility
        json_schema: Optional[str] = None,
        regex: Optional[str] = None,
        ebnf: Optional[str] = None,
        structural_tag: Optional[str] = None,
    ) -> Optional[ConstraintSpec]:
        """Parse constraint specification from any supported format.

        Priority order: binary > uri > inline > legacy

        Args:
            constraint_spec: JSON dict or string containing constraint spec
            constraint_uri: URI reference to external constraint spec
            constraint_bytes: Binary encoded constraint spec (bytes or base64 string)
            constraint_bytes_format: Format of binary data ("msgpack", "protobuf", "json")
            json_schema: Legacy JSON schema string
            regex: Legacy regex pattern
            ebnf: Legacy EBNF grammar
            structural_tag: Legacy structural tag

        Returns:
            Parsed ConstraintSpec or None if no constraint specified
        """
        # Priority: binary > uri > inline > legacy
        if constraint_bytes is not None:
            return self._parse_binary(constraint_bytes, constraint_bytes_format)

        if constraint_uri is not None:
            return self._fetch_and_parse_uri(constraint_uri)

        if constraint_spec is not None:
            return self._parse_inline(constraint_spec)

        if any([json_schema, regex, ebnf, structural_tag]):
            return ConstraintSpec.from_legacy(
                json_schema=json_schema,
                regex=regex,
                ebnf=ebnf,
                structural_tag=structural_tag,
            )

        return None

    def _parse_inline(
        self,
        spec: Union[str, Dict[str, Any]],
    ) -> Optional[ConstraintSpec]:
        """Parse inline constraint specification.

        Args:
            spec: JSON string or dict

        Returns:
            Parsed ConstraintSpec
        """
        if isinstance(spec, str):
            try:
                spec = json.loads(spec)
            except json.JSONDecodeError as e:
                self._handle_error(f"Invalid JSON in constraint_spec: {e}")
                return None

        if not isinstance(spec, dict):
            self._handle_error(f"constraint_spec must be dict, got {type(spec)}")
            return None

        try:
            result = ConstraintSpec.from_dict(spec)
            result.source = ConstraintSource.INLINE
            return result
        except (KeyError, ValueError, TypeError) as e:
            self._handle_error(f"Failed to parse constraint_spec: {e}")
            return None

    def _parse_binary(
        self,
        data: Union[bytes, str],
        format: str,
    ) -> Optional[ConstraintSpec]:
        """Parse binary encoded constraint specification.

        Args:
            data: Binary data or base64 encoded string
            format: Encoding format ("msgpack", "protobuf", "json")

        Returns:
            Parsed ConstraintSpec
        """
        if format not in self.SUPPORTED_BINARY_FORMATS:
            self._handle_error(
                f"Unsupported binary format: {format}. "
                f"Supported: {self.SUPPORTED_BINARY_FORMATS}"
            )
            return None

        # Decode base64 if string
        if isinstance(data, str):
            try:
                data = base64.b64decode(data)
            except Exception as e:
                self._handle_error(f"Failed to decode base64 constraint_bytes: {e}")
                return None

        try:
            if format == "msgpack":
                return self._parse_msgpack(data)
            elif format == "protobuf":
                return self._parse_protobuf(data)
            elif format == "json":
                return self._parse_inline(data.decode("utf-8"))
        except Exception as e:
            self._handle_error(f"Failed to parse binary constraint ({format}): {e}")
            return None

        return None

    def _parse_msgpack(self, data: bytes) -> Optional[ConstraintSpec]:
        """Parse MessagePack encoded data.

        Args:
            data: MessagePack bytes

        Returns:
            Parsed ConstraintSpec
        """
        try:
            import msgpack  # type: ignore
        except ImportError:
            self._handle_error(
                "msgpack not installed. Install with: pip install msgpack"
            )
            return None

        try:
            unpacked = msgpack.unpackb(data, raw=False)
            result = ConstraintSpec.from_dict(unpacked)
            result.source = ConstraintSource.BINARY
            return result
        except Exception as e:
            self._handle_error(f"Failed to parse MessagePack: {e}")
            return None

    def _parse_protobuf(self, data: bytes) -> Optional[ConstraintSpec]:
        """Parse Protobuf encoded data.

        Args:
            data: Protobuf bytes

        Returns:
            Parsed ConstraintSpec

        Note:
            Protobuf support requires generated message classes.
            Currently returns None with a warning.
        """
        self._handle_error(
            "Protobuf support not yet implemented. Use msgpack or json format."
        )
        return None

    def _fetch_and_parse_uri(self, uri: str) -> Optional[ConstraintSpec]:
        """Fetch and parse constraint specification from URI.

        Args:
            uri: URI to fetch (ananke://, http://, https://, file://)

        Returns:
            Parsed ConstraintSpec
        """
        if not self._enable_uri_fetch:
            self._handle_error("URI fetching is disabled")
            return None

        # Check cache
        if uri in self._uri_cache:
            logger.debug(f"URI cache hit: {uri}")
            return self._uri_cache[uri]

        parsed = urlparse(uri)
        scheme = parsed.scheme.lower()

        if scheme not in self.SUPPORTED_URI_SCHEMES:
            self._handle_error(
                f"Unsupported URI scheme: {scheme}. "
                f"Supported: {self.SUPPORTED_URI_SCHEMES}"
            )
            return None

        try:
            if scheme == "ananke":
                spec = self._fetch_ananke_uri(uri, parsed)
            elif scheme in ("http", "https"):
                spec = self._fetch_http_uri(uri)
            elif scheme == "file":
                spec = self._fetch_file_uri(parsed.path)
            else:
                return None

            if spec is not None:
                spec.source = ConstraintSource.URI
                spec.source_uri = uri
                self._cache_uri(uri, spec)

            return spec

        except Exception as e:
            self._handle_error(f"Failed to fetch URI {uri}: {e}")
            return None

    def _fetch_ananke_uri(
        self,
        uri: str,
        parsed: Any,
    ) -> Optional[ConstraintSpec]:
        """Fetch from ananke:// URI scheme.

        Args:
            uri: Full URI
            parsed: Parsed URL components

        Returns:
            Parsed ConstraintSpec

        Note:
            The ananke:// scheme is for integration with the ananke CLI tool.
            Currently returns None with a warning.
        """
        self._handle_error(
            f"ananke:// URI scheme not yet implemented: {uri}"
        )
        return None

    def _fetch_http_uri(self, uri: str) -> Optional[ConstraintSpec]:
        """Fetch from HTTP/HTTPS URI.

        Args:
            uri: HTTP(S) URL

        Returns:
            Parsed ConstraintSpec
        """
        try:
            import urllib.request
        except ImportError:
            self._handle_error("urllib not available")
            return None

        try:
            with urllib.request.urlopen(uri, timeout=10) as response:
                data = response.read()
                content_type = response.headers.get("Content-Type", "")

                if "msgpack" in content_type:
                    return self._parse_msgpack(data)
                elif "protobuf" in content_type:
                    return self._parse_protobuf(data)
                else:
                    # Default to JSON
                    return self._parse_inline(data.decode("utf-8"))

        except Exception as e:
            self._handle_error(f"HTTP fetch failed: {e}")
            return None

    def _fetch_file_uri(self, path: str) -> Optional[ConstraintSpec]:
        """Fetch from file:// URI.

        Args:
            path: File path

        Returns:
            Parsed ConstraintSpec
        """
        try:
            with open(path, "rb") as f:
                data = f.read()

            # Detect format from extension
            if path.endswith(".msgpack"):
                return self._parse_msgpack(data)
            elif path.endswith(".pb") or path.endswith(".proto"):
                return self._parse_protobuf(data)
            else:
                # Default to JSON
                return self._parse_inline(data.decode("utf-8"))

        except FileNotFoundError:
            self._handle_error(f"File not found: {path}")
            return None
        except Exception as e:
            self._handle_error(f"Failed to read file {path}: {e}")
            return None

    def _cache_uri(self, uri: str, spec: ConstraintSpec) -> None:
        """Cache a fetched URI specification.

        Args:
            uri: URI key
            spec: Parsed specification
        """
        if len(self._uri_cache) >= self._uri_cache_size:
            # Simple LRU: remove oldest entry
            oldest = next(iter(self._uri_cache))
            del self._uri_cache[oldest]

        self._uri_cache[uri] = spec

    def _handle_error(self, message: str) -> None:
        """Handle a parse error.

        Args:
            message: Error message
        """
        if self._strict:
            raise ValueError(message)
        else:
            logger.warning(f"ConstraintSpecParser: {message}")

    def clear_cache(self) -> None:
        """Clear the URI cache."""
        self._uri_cache.clear()


def apply_convenience_overrides(
    spec: ConstraintSpec,
    constraint_language: Optional[str] = None,
    type_bindings: Optional[Dict[str, str]] = None,
    expected_type: Optional[str] = None,
    cache_scope: Optional[str] = None,
) -> ConstraintSpec:
    """Apply convenience parameter overrides to a ConstraintSpec.

    This function allows users to specify common overrides without
    providing a full constraint_spec dict.

    Args:
        spec: Base ConstraintSpec to modify
        constraint_language: Override language
        type_bindings: Simple dict of name -> type_expr
        expected_type: Override expected type
        cache_scope: Override cache scope string

    Returns:
        New ConstraintSpec with overrides applied
    """
    from .constraint_spec import TypeBinding

    updates: Dict[str, Any] = {}

    if constraint_language is not None:
        updates["language"] = constraint_language
        updates["language_detection"] = LanguageDetection.EXPLICIT

    if type_bindings is not None:
        # Convert simple dict to TypeBinding list
        bindings = [
            TypeBinding(name=name, type_expr=type_expr).to_dict()
            for name, type_expr in type_bindings.items()
        ]
        # Merge with existing
        existing = [b.to_dict() for b in spec.type_bindings]
        updates["type_bindings"] = existing + bindings

    if expected_type is not None:
        updates["expected_type"] = expected_type

    if cache_scope is not None:
        updates["cache_scope"] = cache_scope

    if updates:
        return spec.copy(**updates)

    return spec
