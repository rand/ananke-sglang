"""
SGLang Ananke SageMaker Inference Handler

Custom inference handler for SageMaker endpoints that provides:
- OpenAI-compatible API endpoints
- Ananke constraint specification support
- Health checks compatible with SageMaker

This module can be used with SageMaker's Python SDK for custom inference.
"""

import json
import os
from typing import Any

import requests


class SGLangHandler:
    """SageMaker inference handler for SGLang Ananke."""

    def __init__(self):
        self.server_url = os.getenv("SGLANG_URL", "http://localhost:8080")
        self.timeout = int(os.getenv("INFERENCE_TIMEOUT", "120"))

    def ping(self) -> bool:
        """Health check for SageMaker."""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=10)
            return response.status_code == 200
        except Exception:
            return False

    def handle(self, data: dict, context: Any = None) -> dict:
        """Handle inference request."""
        # Determine request type
        if "messages" in data:
            return self._chat_completion(data)
        elif "prompt" in data:
            return self._completion(data)
        else:
            return {"error": "Invalid request format"}

    def _chat_completion(self, data: dict) -> dict:
        """Handle chat completion request."""
        response = requests.post(
            f"{self.server_url}/v1/chat/completions",
            json=data,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def _completion(self, data: dict) -> dict:
        """Handle text completion request."""
        response = requests.post(
            f"{self.server_url}/v1/completions",
            json=data,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()


# Global handler instance
_handler = None


def model_fn(model_dir: str):
    """Load model - SGLang loads model on server startup."""
    global _handler
    _handler = SGLangHandler()
    return _handler


def input_fn(request_body: str, request_content_type: str = "application/json") -> dict:
    """Deserialize input data."""
    if request_content_type == "application/json":
        return json.loads(request_body)
    raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data: dict, model: SGLangHandler) -> dict:
    """Run inference."""
    return model.handle(input_data)


def output_fn(prediction: dict, accept: str = "application/json") -> str:
    """Serialize output data."""
    if accept == "application/json":
        return json.dumps(prediction)
    raise ValueError(f"Unsupported accept type: {accept}")
