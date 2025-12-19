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
"""HTTP transport layer for Ananke client SDK.

Provides low-level HTTP client implementations for sync and async communication
with SGLang servers. Handles connection management, request formatting, and
response parsing.
"""

from __future__ import annotations

import json
import time
from typing import Any, AsyncIterator, Dict, Iterator, Optional, Union

# Optional async support - aiohttp may not be installed
try:
    import aiohttp

    ASYNC_AVAILABLE = True
except ImportError:
    aiohttp = None  # type: ignore
    ASYNC_AVAILABLE = False

# Optional sync support - httpx or requests
try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    httpx = None  # type: ignore
    HTTPX_AVAILABLE = False

try:
    import requests

    REQUESTS_AVAILABLE = True
except ImportError:
    requests = None  # type: ignore
    REQUESTS_AVAILABLE = False


class HttpTransportError(Exception):
    """Error during HTTP transport."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response_body: Optional[str] = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class SyncHttpTransport:
    """Synchronous HTTP transport using httpx or requests.

    Attributes:
        base_url: Base URL of the SGLang server
        timeout: Request timeout in seconds
        headers: Default headers for all requests
    """

    def __init__(
        self,
        base_url: str,
        timeout: float = 60.0,
        headers: Optional[Dict[str, str]] = None,
    ):
        """Initialize sync HTTP transport.

        Args:
            base_url: Base URL of SGLang server (e.g., "http://localhost:8000")
            timeout: Request timeout in seconds
            headers: Additional headers for all requests

        Raises:
            ImportError: If neither httpx nor requests is available
        """
        if not HTTPX_AVAILABLE and not REQUESTS_AVAILABLE:
            raise ImportError(
                "Neither httpx nor requests is available. "
                "Install with: pip install httpx  or  pip install requests"
            )

        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.headers = {"Content-Type": "application/json"}
        if headers:
            self.headers.update(headers)

        self._client: Optional[Any] = None

    def _get_client(self) -> Any:
        """Get or create HTTP client."""
        if self._client is None:
            if HTTPX_AVAILABLE:
                self._client = httpx.Client(
                    base_url=self.base_url,
                    timeout=self.timeout,
                    headers=self.headers,
                )
            else:
                # requests doesn't have a persistent client, use session
                self._client = requests.Session()
                self._client.headers.update(self.headers)
        return self._client

    def post(
        self,
        path: str,
        data: Dict[str, Any],
        timeout: Optional[float] = None,
    ) -> tuple[Dict[str, Any], float]:
        """Make POST request and return (response_json, latency_ms).

        Args:
            path: API endpoint path (e.g., "/v1/completions")
            data: Request body as dict
            timeout: Optional timeout override

        Returns:
            Tuple of (response dict, latency in milliseconds)

        Raises:
            HttpTransportError: On HTTP errors
        """
        url = f"{self.base_url}{path}"
        effective_timeout = timeout or self.timeout
        start_time = time.perf_counter()

        try:
            if HTTPX_AVAILABLE:
                client = self._get_client()
                response = client.post(
                    path,
                    json=data,
                    timeout=effective_timeout,
                )
                latency_ms = (time.perf_counter() - start_time) * 1000

                if response.status_code >= 400:
                    raise HttpTransportError(
                        f"HTTP {response.status_code}: {response.text}",
                        status_code=response.status_code,
                        response_body=response.text,
                    )

                return response.json(), latency_ms
            else:
                # requests fallback
                response = self._get_client().post(
                    url,
                    json=data,
                    timeout=effective_timeout,
                )
                latency_ms = (time.perf_counter() - start_time) * 1000

                if response.status_code >= 400:
                    raise HttpTransportError(
                        f"HTTP {response.status_code}: {response.text}",
                        status_code=response.status_code,
                        response_body=response.text,
                    )

                return response.json(), latency_ms

        except (httpx.HTTPError if HTTPX_AVAILABLE else Exception) as e:
            if isinstance(e, HttpTransportError):
                raise
            latency_ms = (time.perf_counter() - start_time) * 1000
            raise HttpTransportError(f"Request failed: {e}") from e

    def post_stream(
        self,
        path: str,
        data: Dict[str, Any],
        timeout: Optional[float] = None,
    ) -> Iterator[Dict[str, Any]]:
        """Make streaming POST request and yield SSE events.

        Args:
            path: API endpoint path
            data: Request body as dict
            timeout: Optional timeout override

        Yields:
            Parsed JSON data from each SSE event

        Raises:
            HttpTransportError: On HTTP errors
        """
        url = f"{self.base_url}{path}"
        effective_timeout = timeout or self.timeout

        try:
            if HTTPX_AVAILABLE:
                client = self._get_client()
                with client.stream(
                    "POST",
                    path,
                    json=data,
                    timeout=effective_timeout,
                ) as response:
                    if response.status_code >= 400:
                        response.read()
                        raise HttpTransportError(
                            f"HTTP {response.status_code}: {response.text}",
                            status_code=response.status_code,
                            response_body=response.text,
                        )

                    for line in response.iter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str.strip() == "[DONE]":
                                break
                            try:
                                yield json.loads(data_str)
                            except json.JSONDecodeError:
                                continue
            else:
                # requests fallback
                response = self._get_client().post(
                    url,
                    json=data,
                    timeout=effective_timeout,
                    stream=True,
                )

                if response.status_code >= 400:
                    raise HttpTransportError(
                        f"HTTP {response.status_code}: {response.text}",
                        status_code=response.status_code,
                        response_body=response.text,
                    )

                for line in response.iter_lines():
                    if line:
                        line_str = line.decode("utf-8") if isinstance(line, bytes) else line
                        if line_str.startswith("data: "):
                            data_str = line_str[6:]
                            if data_str.strip() == "[DONE]":
                                break
                            try:
                                yield json.loads(data_str)
                            except json.JSONDecodeError:
                                continue

        except (httpx.HTTPError if HTTPX_AVAILABLE else Exception) as e:
            if isinstance(e, HttpTransportError):
                raise
            raise HttpTransportError(f"Streaming request failed: {e}") from e

    def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            if HTTPX_AVAILABLE:
                self._client.close()
            # requests.Session doesn't need explicit close
            self._client = None


class AsyncHttpTransport:
    """Asynchronous HTTP transport using aiohttp.

    Attributes:
        base_url: Base URL of the SGLang server
        timeout: Request timeout in seconds
        headers: Default headers for all requests
    """

    def __init__(
        self,
        base_url: str,
        timeout: float = 60.0,
        headers: Optional[Dict[str, str]] = None,
    ):
        """Initialize async HTTP transport.

        Args:
            base_url: Base URL of SGLang server
            timeout: Request timeout in seconds
            headers: Additional headers for all requests

        Raises:
            ImportError: If aiohttp is not available
        """
        if not ASYNC_AVAILABLE:
            raise ImportError(
                "aiohttp is required for async client. "
                "Install with: pip install aiohttp"
            )

        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.headers = {"Content-Type": "application/json"}
        if headers:
            self.headers.update(headers)

        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(
                base_url=self.base_url,
                timeout=timeout,
                headers=self.headers,
            )
        return self._session

    async def post(
        self,
        path: str,
        data: Dict[str, Any],
        timeout: Optional[float] = None,
    ) -> tuple[Dict[str, Any], float]:
        """Make async POST request and return (response_json, latency_ms).

        Args:
            path: API endpoint path
            data: Request body as dict
            timeout: Optional timeout override

        Returns:
            Tuple of (response dict, latency in milliseconds)

        Raises:
            HttpTransportError: On HTTP errors
        """
        session = await self._get_session()
        start_time = time.perf_counter()

        try:
            timeout_obj = (
                aiohttp.ClientTimeout(total=timeout) if timeout else None
            )

            async with session.post(path, json=data, timeout=timeout_obj) as response:
                latency_ms = (time.perf_counter() - start_time) * 1000

                if response.status >= 400:
                    body = await response.text()
                    raise HttpTransportError(
                        f"HTTP {response.status}: {body}",
                        status_code=response.status,
                        response_body=body,
                    )

                result = await response.json()
                return result, latency_ms

        except aiohttp.ClientError as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            raise HttpTransportError(f"Request failed: {e}") from e

    async def post_stream(
        self,
        path: str,
        data: Dict[str, Any],
        timeout: Optional[float] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Make async streaming POST request and yield SSE events.

        Args:
            path: API endpoint path
            data: Request body as dict
            timeout: Optional timeout override

        Yields:
            Parsed JSON data from each SSE event

        Raises:
            HttpTransportError: On HTTP errors
        """
        session = await self._get_session()

        try:
            timeout_obj = (
                aiohttp.ClientTimeout(total=timeout) if timeout else None
            )

            async with session.post(path, json=data, timeout=timeout_obj) as response:
                if response.status >= 400:
                    body = await response.text()
                    raise HttpTransportError(
                        f"HTTP {response.status}: {body}",
                        status_code=response.status,
                        response_body=body,
                    )

                async for line in response.content:
                    line_str = line.decode("utf-8").strip()
                    if line_str.startswith("data: "):
                        data_str = line_str[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            yield json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

        except aiohttp.ClientError as e:
            raise HttpTransportError(f"Streaming request failed: {e}") from e

    async def close(self) -> None:
        """Close the aiohttp session."""
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None
