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
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Dict, Iterator, Optional, Set, Union

logger = logging.getLogger(__name__)

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
        is_retryable: bool = False,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body
        self.is_retryable = is_retryable


# Default retryable HTTP status codes
DEFAULT_RETRYABLE_STATUS_CODES: Set[int] = {
    408,  # Request Timeout
    429,  # Too Many Requests
    500,  # Internal Server Error
    502,  # Bad Gateway
    503,  # Service Unavailable
    504,  # Gateway Timeout
}


@dataclass
class LoggingConfig:
    """Configuration for request/response logging.

    Attributes:
        enabled: Enable request/response logging
        log_request_body: Log full request body (may contain sensitive data)
        log_response_body: Log full response body
        log_headers: Log request headers (may contain auth tokens)
        truncate_body: Maximum characters to log from bodies (0 = unlimited)
        log_level: Logging level for requests (DEBUG, INFO, etc.)
    """

    enabled: bool = True
    log_request_body: bool = False
    log_response_body: bool = False
    log_headers: bool = False
    truncate_body: int = 500
    log_level: int = logging.DEBUG

    def log_request(
        self,
        method: str,
        url: str,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        """Log an outgoing request."""
        if not self.enabled:
            return

        parts = [f"{method} {url}"]

        if self.log_headers and headers:
            # Redact authorization headers
            safe_headers = {
                k: ("***" if k.lower() in ("authorization", "x-api-key") else v)
                for k, v in headers.items()
            }
            parts.append(f"headers={safe_headers}")

        if self.log_request_body and data:
            body_str = json.dumps(data)
            if self.truncate_body > 0 and len(body_str) > self.truncate_body:
                body_str = body_str[: self.truncate_body] + "..."
            parts.append(f"body={body_str}")

        logger.log(self.log_level, "Request: %s", " | ".join(parts))

    def log_response(
        self,
        status_code: int,
        latency_ms: float,
        body: Optional[Any] = None,
        error: Optional[str] = None,
    ) -> None:
        """Log a response or error."""
        if not self.enabled:
            return

        parts = [f"status={status_code}", f"latency={latency_ms:.1f}ms"]

        if error:
            parts.append(f"error={error}")
        elif self.log_response_body and body:
            body_str = json.dumps(body) if isinstance(body, dict) else str(body)
            if self.truncate_body > 0 and len(body_str) > self.truncate_body:
                body_str = body_str[: self.truncate_body] + "..."
            parts.append(f"body={body_str}")

        level = logging.WARNING if error or status_code >= 400 else self.log_level
        logger.log(level, "Response: %s", " | ".join(parts))


@dataclass
class RetryConfig:
    """Configuration for retry behavior with exponential backoff.

    Attributes:
        max_retries: Maximum number of retry attempts (0 = no retries)
        initial_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        exponential_base: Base for exponential backoff (delay *= base^attempt)
        jitter: Add random jitter to delays (0.0-1.0, fraction of delay)
        retryable_status_codes: HTTP status codes that trigger retry
        retry_on_timeout: Whether to retry on timeout errors
    """

    max_retries: int = 3
    initial_delay: float = 0.5
    max_delay: float = 30.0
    exponential_base: float = 2.0
    jitter: float = 0.1
    retryable_status_codes: Set[int] = field(
        default_factory=lambda: DEFAULT_RETRYABLE_STATUS_CODES.copy()
    )
    retry_on_timeout: bool = True

    def compute_delay(self, attempt: int) -> float:
        """Compute delay for given attempt number with exponential backoff.

        Args:
            attempt: Current attempt number (0-indexed)

        Returns:
            Delay in seconds with optional jitter applied
        """
        delay = self.initial_delay * (self.exponential_base ** attempt)
        delay = min(delay, self.max_delay)

        if self.jitter > 0:
            jitter_amount = delay * self.jitter * random.random()
            delay += jitter_amount

        return delay

    def should_retry(
        self,
        attempt: int,
        error: Optional[Exception] = None,
        status_code: Optional[int] = None,
    ) -> bool:
        """Determine if a request should be retried.

        Args:
            attempt: Current attempt number (0-indexed)
            error: Exception that occurred (if any)
            status_code: HTTP status code (if available)

        Returns:
            True if the request should be retried
        """
        if attempt >= self.max_retries:
            return False

        # Check status code
        if status_code is not None and status_code in self.retryable_status_codes:
            return True

        # Check for timeout errors
        if self.retry_on_timeout and error is not None:
            error_str = str(error).lower()
            if "timeout" in error_str or "timed out" in error_str:
                return True

        # Check for connection errors (generally retryable)
        if error is not None:
            error_str = str(error).lower()
            if any(
                term in error_str
                for term in ["connection", "connect", "reset", "refused"]
            ):
                return True

        return False


class SyncHttpTransport:
    """Synchronous HTTP transport using httpx or requests.

    Attributes:
        base_url: Base URL of the SGLang server
        timeout: Request timeout in seconds
        headers: Default headers for all requests
        retry_config: Configuration for retry behavior
        logging_config: Configuration for request/response logging
    """

    def __init__(
        self,
        base_url: str,
        timeout: float = 60.0,
        headers: Optional[Dict[str, str]] = None,
        retry_config: Optional[RetryConfig] = None,
        logging_config: Optional[LoggingConfig] = None,
    ):
        """Initialize sync HTTP transport.

        Args:
            base_url: Base URL of SGLang server (e.g., "http://localhost:8000")
            timeout: Request timeout in seconds
            headers: Additional headers for all requests
            retry_config: Retry configuration (default: RetryConfig with 3 retries)
            logging_config: Logging configuration (default: LoggingConfig enabled)

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

        self.retry_config = retry_config or RetryConfig()
        self.logging_config = logging_config or LoggingConfig()
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
        """Make POST request with retry support, return (response_json, latency_ms).

        Args:
            path: API endpoint path (e.g., "/v1/completions")
            data: Request body as dict
            timeout: Optional timeout override

        Returns:
            Tuple of (response dict, latency in milliseconds)

        Raises:
            HttpTransportError: On HTTP errors after all retries exhausted
        """
        url = f"{self.base_url}{path}"
        effective_timeout = timeout or self.timeout
        last_error: Optional[Exception] = None

        # Log the request (only on first attempt to avoid spam during retries)
        self.logging_config.log_request("POST", url, data, self.headers)

        for attempt in range(self.retry_config.max_retries + 1):
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
                        is_retryable = response.status_code in self.retry_config.retryable_status_codes
                        self.logging_config.log_response(
                            response.status_code, latency_ms, error=response.text[:200]
                        )
                        error = HttpTransportError(
                            f"HTTP {response.status_code}: {response.text}",
                            status_code=response.status_code,
                            response_body=response.text,
                            is_retryable=is_retryable,
                        )

                        if self.retry_config.should_retry(attempt, status_code=response.status_code):
                            delay = self.retry_config.compute_delay(attempt)
                            logger.warning(
                                f"Request to {path} failed with status {response.status_code}, "
                                f"retrying in {delay:.2f}s (attempt {attempt + 1}/{self.retry_config.max_retries + 1})"
                            )
                            time.sleep(delay)
                            last_error = error
                            continue

                        raise error

                    # Log successful response
                    result = response.json()
                    self.logging_config.log_response(response.status_code, latency_ms, result)
                    return result, latency_ms
                else:
                    # requests fallback
                    response = self._get_client().post(
                        url,
                        json=data,
                        timeout=effective_timeout,
                    )
                    latency_ms = (time.perf_counter() - start_time) * 1000

                    if response.status_code >= 400:
                        is_retryable = response.status_code in self.retry_config.retryable_status_codes
                        self.logging_config.log_response(
                            response.status_code, latency_ms, error=response.text[:200]
                        )
                        error = HttpTransportError(
                            f"HTTP {response.status_code}: {response.text}",
                            status_code=response.status_code,
                            response_body=response.text,
                            is_retryable=is_retryable,
                        )

                        if self.retry_config.should_retry(attempt, status_code=response.status_code):
                            delay = self.retry_config.compute_delay(attempt)
                            logger.warning(
                                f"Request to {path} failed with status {response.status_code}, "
                                f"retrying in {delay:.2f}s (attempt {attempt + 1}/{self.retry_config.max_retries + 1})"
                            )
                            time.sleep(delay)
                            last_error = error
                            continue

                        raise error

                    # Log successful response
                    result = response.json()
                    self.logging_config.log_response(response.status_code, latency_ms, result)
                    return result, latency_ms

            except HttpTransportError:
                raise
            except Exception as e:
                latency_ms = (time.perf_counter() - start_time) * 1000
                self.logging_config.log_response(0, latency_ms, error=str(e)[:200])

                if self.retry_config.should_retry(attempt, error=e):
                    delay = self.retry_config.compute_delay(attempt)
                    logger.warning(
                        f"Request to {path} failed with {type(e).__name__}: {e}, "
                        f"retrying in {delay:.2f}s (attempt {attempt + 1}/{self.retry_config.max_retries + 1})"
                    )
                    time.sleep(delay)
                    last_error = e
                    continue

                raise HttpTransportError(f"Request failed: {e}", is_retryable=False) from e

        # All retries exhausted
        if last_error:
            if isinstance(last_error, HttpTransportError):
                raise last_error
            raise HttpTransportError(
                f"Request failed after {self.retry_config.max_retries + 1} attempts: {last_error}",
                is_retryable=False,
            ) from last_error

        # Should not reach here, but just in case
        raise HttpTransportError("Request failed: unknown error", is_retryable=False)

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
        retry_config: Configuration for retry behavior
        logging_config: Configuration for request/response logging
    """

    def __init__(
        self,
        base_url: str,
        timeout: float = 60.0,
        headers: Optional[Dict[str, str]] = None,
        retry_config: Optional[RetryConfig] = None,
        logging_config: Optional[LoggingConfig] = None,
    ):
        """Initialize async HTTP transport.

        Args:
            base_url: Base URL of SGLang server
            timeout: Request timeout in seconds
            headers: Additional headers for all requests
            retry_config: Retry configuration (default: RetryConfig with 3 retries)
            logging_config: Logging configuration (default: LoggingConfig enabled)

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

        self.retry_config = retry_config or RetryConfig()
        self.logging_config = logging_config or LoggingConfig()
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
        """Make async POST request with retry support, return (response_json, latency_ms).

        Args:
            path: API endpoint path
            data: Request body as dict
            timeout: Optional timeout override

        Returns:
            Tuple of (response dict, latency in milliseconds)

        Raises:
            HttpTransportError: On HTTP errors after all retries exhausted
        """
        import asyncio

        url = f"{self.base_url}{path}"
        session = await self._get_session()
        last_error: Optional[Exception] = None

        # Log the request
        self.logging_config.log_request("POST", url, data, self.headers)

        for attempt in range(self.retry_config.max_retries + 1):
            start_time = time.perf_counter()

            try:
                timeout_obj = (
                    aiohttp.ClientTimeout(total=timeout) if timeout else None
                )

                async with session.post(path, json=data, timeout=timeout_obj) as response:
                    latency_ms = (time.perf_counter() - start_time) * 1000

                    if response.status >= 400:
                        body = await response.text()
                        is_retryable = response.status in self.retry_config.retryable_status_codes
                        self.logging_config.log_response(
                            response.status, latency_ms, error=body[:200]
                        )
                        error = HttpTransportError(
                            f"HTTP {response.status}: {body}",
                            status_code=response.status,
                            response_body=body,
                            is_retryable=is_retryable,
                        )

                        if self.retry_config.should_retry(attempt, status_code=response.status):
                            delay = self.retry_config.compute_delay(attempt)
                            logger.warning(
                                f"Async request to {path} failed with status {response.status}, "
                                f"retrying in {delay:.2f}s (attempt {attempt + 1}/{self.retry_config.max_retries + 1})"
                            )
                            await asyncio.sleep(delay)
                            last_error = error
                            continue

                        raise error

                    # Log successful response
                    result = await response.json()
                    self.logging_config.log_response(response.status, latency_ms, result)
                    return result, latency_ms

            except HttpTransportError:
                raise
            except aiohttp.ClientError as e:
                latency_ms = (time.perf_counter() - start_time) * 1000
                self.logging_config.log_response(0, latency_ms, error=str(e)[:200])

                if self.retry_config.should_retry(attempt, error=e):
                    delay = self.retry_config.compute_delay(attempt)
                    logger.warning(
                        f"Async request to {path} failed with {type(e).__name__}: {e}, "
                        f"retrying in {delay:.2f}s (attempt {attempt + 1}/{self.retry_config.max_retries + 1})"
                    )
                    await asyncio.sleep(delay)
                    last_error = e
                    continue

                raise HttpTransportError(f"Request failed: {e}", is_retryable=False) from e

        # All retries exhausted
        if last_error:
            if isinstance(last_error, HttpTransportError):
                raise last_error
            raise HttpTransportError(
                f"Request failed after {self.retry_config.max_retries + 1} attempts: {last_error}",
                is_retryable=False,
            ) from last_error

        # Should not reach here
        raise HttpTransportError("Request failed: unknown error", is_retryable=False)

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
