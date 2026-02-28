# Copyright Rand Arete @ Ananke 2025
# Licensed under the Apache License, Version 2.0
"""Unit tests for HTTP transport layer with retry and logging support."""

from __future__ import annotations

import logging
import time
from unittest.mock import MagicMock, patch

import pytest

from client.http import (
    DEFAULT_RETRYABLE_STATUS_CODES,
    HttpTransportError,
    LoggingConfig,
    RetryConfig,
)


class TestRetryConfig:
    """Tests for RetryConfig dataclass."""

    def test_default_values(self):
        """Test default retry configuration."""
        config = RetryConfig()
        assert config.max_retries == 3
        assert config.initial_delay == 0.5
        assert config.max_delay == 30.0
        assert config.exponential_base == 2.0
        assert config.jitter == 0.1
        assert config.retry_on_timeout is True
        assert config.retryable_status_codes == DEFAULT_RETRYABLE_STATUS_CODES

    def test_custom_values(self):
        """Test custom retry configuration."""
        config = RetryConfig(
            max_retries=5,
            initial_delay=1.0,
            max_delay=60.0,
            exponential_base=3.0,
            jitter=0.2,
        )
        assert config.max_retries == 5
        assert config.initial_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 3.0
        assert config.jitter == 0.2

    def test_compute_delay_exponential_growth(self):
        """Test exponential backoff delay computation."""
        config = RetryConfig(initial_delay=1.0, exponential_base=2.0, jitter=0.0)

        # Without jitter, delays should be exact
        assert config.compute_delay(0) == 1.0  # 1 * 2^0
        assert config.compute_delay(1) == 2.0  # 1 * 2^1
        assert config.compute_delay(2) == 4.0  # 1 * 2^2
        assert config.compute_delay(3) == 8.0  # 1 * 2^3

    def test_compute_delay_respects_max(self):
        """Test that delay is capped at max_delay."""
        config = RetryConfig(
            initial_delay=1.0,
            max_delay=5.0,
            exponential_base=2.0,
            jitter=0.0,
        )

        assert config.compute_delay(0) == 1.0
        assert config.compute_delay(2) == 4.0
        assert config.compute_delay(3) == 5.0  # Capped at max
        assert config.compute_delay(10) == 5.0  # Still capped

    def test_compute_delay_with_jitter(self):
        """Test that jitter adds randomness to delays."""
        config = RetryConfig(initial_delay=1.0, jitter=0.5)

        # With jitter, delays should vary
        delays = [config.compute_delay(0) for _ in range(10)]

        # All delays should be >= base delay
        for d in delays:
            assert d >= 1.0
            assert d <= 1.5  # Max jitter is 0.5 * 1.0 = 0.5

        # At least some variation expected (very unlikely all same)
        assert len(set(delays)) > 1

    def test_should_retry_respects_max_retries(self):
        """Test that should_retry respects max_retries limit."""
        config = RetryConfig(max_retries=3)

        assert config.should_retry(0, status_code=503) is True
        assert config.should_retry(1, status_code=503) is True
        assert config.should_retry(2, status_code=503) is True
        assert config.should_retry(3, status_code=503) is False  # Exceeded

    def test_should_retry_retryable_status_codes(self):
        """Test retry on specific status codes."""
        config = RetryConfig()

        # Should retry on 5xx and 429
        assert config.should_retry(0, status_code=500) is True
        assert config.should_retry(0, status_code=502) is True
        assert config.should_retry(0, status_code=503) is True
        assert config.should_retry(0, status_code=504) is True
        assert config.should_retry(0, status_code=429) is True
        assert config.should_retry(0, status_code=408) is True

        # Should NOT retry on 4xx (except 408, 429)
        assert config.should_retry(0, status_code=400) is False
        assert config.should_retry(0, status_code=401) is False
        assert config.should_retry(0, status_code=403) is False
        assert config.should_retry(0, status_code=404) is False

    def test_should_retry_on_timeout_error(self):
        """Test retry on timeout errors."""
        config = RetryConfig(retry_on_timeout=True)

        timeout_error = Exception("Connection timed out")
        assert config.should_retry(0, error=timeout_error) is True

        timeout_error2 = Exception("Request timeout exceeded")
        assert config.should_retry(0, error=timeout_error2) is True

    def test_should_retry_on_connection_error(self):
        """Test retry on connection errors."""
        config = RetryConfig()

        conn_error = Exception("Connection refused")
        assert config.should_retry(0, error=conn_error) is True

        reset_error = Exception("Connection reset by peer")
        assert config.should_retry(0, error=reset_error) is True

    def test_should_retry_disabled_timeout(self):
        """Test that timeout retry can be disabled."""
        config = RetryConfig(retry_on_timeout=False)

        # Pure timeout error (no "connection" in message)
        timeout_error = Exception("Request timed out after 30s")
        # Should not retry on timeout alone when disabled
        assert config.should_retry(0, error=timeout_error) is False
        # But still retry on status codes
        assert config.should_retry(0, status_code=503) is True
        # And still retry on connection errors (separate from timeout)
        assert config.should_retry(0, error=Exception("Connection refused")) is True

    def test_custom_retryable_status_codes(self):
        """Test custom retryable status codes."""
        config = RetryConfig(retryable_status_codes={418, 503})

        assert config.should_retry(0, status_code=418) is True
        assert config.should_retry(0, status_code=503) is True
        assert config.should_retry(0, status_code=500) is False
        assert config.should_retry(0, status_code=429) is False


class TestLoggingConfig:
    """Tests for LoggingConfig dataclass."""

    def test_default_values(self):
        """Test default logging configuration."""
        config = LoggingConfig()
        assert config.enabled is True
        assert config.log_request_body is False
        assert config.log_response_body is False
        assert config.log_headers is False
        assert config.truncate_body == 500
        assert config.log_level == logging.DEBUG

    def test_log_request_basic(self, caplog):
        """Test basic request logging."""
        config = LoggingConfig(enabled=True, log_level=logging.INFO)

        with caplog.at_level(logging.INFO):
            config.log_request("POST", "http://localhost:8000/test", None, None)

        assert "Request: POST http://localhost:8000/test" in caplog.text

    def test_log_request_with_body(self, caplog):
        """Test request logging with body."""
        config = LoggingConfig(enabled=True, log_request_body=True, log_level=logging.INFO)

        with caplog.at_level(logging.INFO):
            config.log_request("POST", "/test", {"prompt": "hello"}, None)

        assert "body=" in caplog.text
        assert "prompt" in caplog.text

    def test_log_request_body_truncation(self, caplog):
        """Test request body truncation."""
        config = LoggingConfig(
            enabled=True,
            log_request_body=True,
            truncate_body=20,
            log_level=logging.INFO,
        )

        long_body = {"data": "x" * 100}
        with caplog.at_level(logging.INFO):
            config.log_request("POST", "/test", long_body, None)

        assert "..." in caplog.text

    def test_log_request_header_redaction(self, caplog):
        """Test that sensitive headers are redacted."""
        config = LoggingConfig(enabled=True, log_headers=True, log_level=logging.INFO)

        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer secret-token",
            "X-Api-Key": "my-api-key",
        }

        with caplog.at_level(logging.INFO):
            config.log_request("POST", "/test", None, headers)

        assert "Content-Type" in caplog.text
        assert "application/json" in caplog.text
        assert "secret-token" not in caplog.text
        assert "my-api-key" not in caplog.text
        assert "***" in caplog.text

    def test_log_response_success(self, caplog):
        """Test successful response logging."""
        config = LoggingConfig(enabled=True, log_level=logging.INFO)

        with caplog.at_level(logging.INFO):
            config.log_response(200, 150.5, {"result": "ok"})

        assert "status=200" in caplog.text
        assert "latency=150.5ms" in caplog.text

    def test_log_response_with_body(self, caplog):
        """Test response logging with body."""
        config = LoggingConfig(
            enabled=True,
            log_response_body=True,
            log_level=logging.INFO,
        )

        with caplog.at_level(logging.INFO):
            config.log_response(200, 100.0, {"text": "generated"})

        assert "body=" in caplog.text
        assert "generated" in caplog.text

    def test_log_response_error_uses_warning(self, caplog):
        """Test that error responses use WARNING level."""
        config = LoggingConfig(enabled=True)

        with caplog.at_level(logging.DEBUG):
            config.log_response(500, 50.0, error="Server Error")

        # Check that WARNING was used
        assert any(r.levelno == logging.WARNING for r in caplog.records)
        assert "error=Server Error" in caplog.text

    def test_log_response_4xx_uses_warning(self, caplog):
        """Test that 4xx responses use WARNING level."""
        config = LoggingConfig(enabled=True)

        with caplog.at_level(logging.DEBUG):
            config.log_response(404, 30.0)

        assert any(r.levelno == logging.WARNING for r in caplog.records)

    def test_logging_disabled(self, caplog):
        """Test that disabled logging produces no output."""
        config = LoggingConfig(enabled=False)

        with caplog.at_level(logging.DEBUG):
            config.log_request("POST", "/test", {"data": "secret"}, {"Auth": "token"})
            config.log_response(200, 100.0, {"result": "data"})

        assert caplog.text == ""


class TestHttpTransportError:
    """Tests for HttpTransportError exception."""

    def test_basic_error(self):
        """Test basic error creation."""
        error = HttpTransportError("Request failed")
        assert str(error) == "Request failed"
        assert error.status_code is None
        assert error.response_body is None
        assert error.is_retryable is False

    def test_error_with_status(self):
        """Test error with HTTP status code."""
        error = HttpTransportError(
            "HTTP 500",
            status_code=500,
            response_body="Internal Server Error",
        )
        assert error.status_code == 500
        assert error.response_body == "Internal Server Error"

    def test_error_retryable_flag(self):
        """Test retryable flag."""
        retryable = HttpTransportError("Temp error", is_retryable=True)
        non_retryable = HttpTransportError("Perm error", is_retryable=False)

        assert retryable.is_retryable is True
        assert non_retryable.is_retryable is False


class TestRetryBehavior:
    """Integration-style tests for retry behavior."""

    def test_exponential_backoff_timing(self):
        """Test that exponential backoff computes reasonable delays."""
        config = RetryConfig(
            max_retries=5,
            initial_delay=0.1,
            max_delay=10.0,
            exponential_base=2.0,
            jitter=0.0,
        )

        delays = [config.compute_delay(i) for i in range(6)]

        # Verify exponential growth
        assert delays[0] == 0.1
        assert delays[1] == 0.2
        assert delays[2] == 0.4
        assert delays[3] == 0.8
        assert delays[4] == 1.6
        assert delays[5] == 3.2

    def test_retry_decision_matrix(self):
        """Test retry decisions across various scenarios."""
        config = RetryConfig(max_retries=2)

        # Matrix of (attempt, status_code, error, expected_result)
        cases = [
            (0, 503, None, True),   # First attempt, retryable status
            (1, 503, None, True),   # Second attempt, retryable status
            (2, 503, None, False),  # Exceeded max retries
            (0, 400, None, False),  # Non-retryable status
            (0, None, Exception("timeout"), True),  # Timeout error
            (0, None, Exception("refused"), True),  # Connection error
            (0, None, Exception("unknown"), False), # Unknown error
        ]

        for attempt, status, error, expected in cases:
            result = config.should_retry(attempt, error=error, status_code=status)
            assert result == expected, f"Failed for {(attempt, status, error)}"
