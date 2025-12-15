"""Tests for healthcheck.py utilities."""

import json
import unittest
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread
from unittest.mock import patch, MagicMock
import sys
import os

# Add deploy/scripts to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from healthcheck import (
    check_health,
    check_health_generate,
    check_models,
    full_health_check,
)


class MockHandler(BaseHTTPRequestHandler):
    """Mock HTTP handler for testing."""

    def log_message(self, format, *args):
        pass  # Suppress logging

    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"status": "healthy"}')
        elif self.path == '/health_generate':
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"status": "ready"}')
        elif self.path == '/v1/models':
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({
                "data": [{"id": "test-model"}]
            }).encode())
        else:
            self.send_response(404)
            self.end_headers()


class TestHealthCheck(unittest.TestCase):
    """Test health check functions."""

    @classmethod
    def setUpClass(cls):
        """Start mock server."""
        cls.server = HTTPServer(('localhost', 0), MockHandler)
        cls.port = cls.server.server_address[1]
        cls.base_url = f"http://localhost:{cls.port}"
        cls.thread = Thread(target=cls.server.serve_forever)
        cls.thread.daemon = True
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        """Stop mock server."""
        cls.server.shutdown()

    def test_check_health_success(self):
        """Test successful health check."""
        result = check_health(self.base_url, timeout=5)
        self.assertTrue(result)

    def test_check_health_failure(self):
        """Test failed health check."""
        result = check_health("http://localhost:1", timeout=1)
        self.assertFalse(result)

    def test_check_health_generate_success(self):
        """Test successful readiness check."""
        result = check_health_generate(self.base_url, timeout=5)
        self.assertTrue(result)

    def test_check_models_success(self):
        """Test models endpoint."""
        result = check_models(self.base_url, timeout=5)
        self.assertIsNotNone(result)
        self.assertIn("data", result)
        self.assertEqual(result["data"][0]["id"], "test-model")

    def test_check_models_failure(self):
        """Test models endpoint failure."""
        result = check_models("http://localhost:1", timeout=1)
        self.assertIsNone(result)

    def test_full_health_check(self):
        """Test comprehensive health check."""
        result = full_health_check(self.base_url, timeout=5, check_ananke_backend=False)
        self.assertTrue(result["healthy"])
        self.assertTrue(result["checks"]["liveness"])
        self.assertTrue(result["checks"]["readiness"])
        self.assertTrue(result["checks"]["models"])
        self.assertIn("timestamp", result)


class TestHealthCheckEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def test_check_health_invalid_url(self):
        """Test with invalid URL."""
        result = check_health("not-a-valid-url", timeout=1)
        self.assertFalse(result)

    def test_check_health_timeout(self):
        """Test timeout handling."""
        # Very short timeout should fail
        result = check_health("http://10.255.255.1:30000", timeout=0.1)
        self.assertFalse(result)

    def test_full_health_check_server_down(self):
        """Test full health check when server is down."""
        result = full_health_check("http://localhost:1", timeout=1)
        self.assertFalse(result["healthy"])
        self.assertFalse(result["checks"]["liveness"])


if __name__ == '__main__':
    unittest.main()
