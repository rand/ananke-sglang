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
"""Unit tests for Ananke Client SDK.

Tests cover:
- GenerationConfig serialization and factory methods
- GenerationResult parsing from API responses
- ConstraintBuilder fluent API
- ConstraintSpec factory methods
- HTTP transport error handling (mocked)
"""

from __future__ import annotations

import json
import pytest
from typing import Any, Dict
from unittest.mock import MagicMock, patch

# Import client components
from ananke.client import (
    AnankeClient,
    ConstraintBuilder,
    GenerationConfig,
    GenerationResult,
    FinishReason,
    RelaxationInfo,
    StreamChunk,
)
from ananke.client.http import HttpTransportError
from ananke.spec.constraint_spec import ConstraintSpec, TypeBinding


# =============================================================================
# GenerationConfig Tests
# =============================================================================


class TestGenerationConfig:
    """Tests for GenerationConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = GenerationConfig()

        assert config.max_tokens == 256
        assert config.temperature == 0.7
        assert config.top_p == 1.0
        assert config.top_k == -1
        assert config.seed is None
        assert config.allow_relaxation is True
        assert config.relaxation_threshold == 10
        assert config.enable_early_termination is True
        assert config.stream is False
        assert config.timeout_seconds == 60.0

    def test_to_dict_minimal(self) -> None:
        """Test serialization with default values."""
        config = GenerationConfig()
        d = config.to_dict()

        assert d["max_new_tokens"] == 256
        assert d["temperature"] == 0.7
        assert d["top_p"] == 1.0
        # Optional fields should not be present
        assert "top_k" not in d
        assert "seed" not in d

    def test_to_dict_with_all_options(self) -> None:
        """Test serialization with all options set."""
        config = GenerationConfig(
            max_tokens=512,
            temperature=0.0,
            top_p=0.9,
            top_k=50,
            seed=42,
            allow_relaxation=False,
            relaxation_threshold=5,
            enable_early_termination=False,
        )
        d = config.to_dict()

        assert d["max_new_tokens"] == 512
        assert d["temperature"] == 0.0
        assert d["top_p"] == 0.9
        assert d["top_k"] == 50
        assert d["seed"] == 42
        assert d["constraint_relaxation"] is False
        assert d["relaxation_threshold"] == 5
        assert d["early_termination"] is False

    def test_deterministic_factory(self) -> None:
        """Test deterministic configuration factory."""
        config = GenerationConfig.deterministic(seed=123, max_tokens=100)

        assert config.temperature == 0.0
        assert config.seed == 123
        assert config.max_tokens == 100


# =============================================================================
# GenerationResult Tests
# =============================================================================


class TestGenerationResult:
    """Tests for GenerationResult parsing."""

    def test_from_simple_response(self) -> None:
        """Test parsing simple completion response."""
        response = {
            "text": "return n * 2",
            "finish_reason": "stop",
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
            },
        }

        result = GenerationResult.from_response(response, latency_ms=150.0)

        assert result.text == "return n * 2"
        assert result.finish_reason == FinishReason.STOP
        assert result.tokens_generated == 5
        assert result.prompt_tokens == 10
        assert result.constraint_satisfied is True
        assert result.latency_ms == 150.0

    def test_from_choices_response(self) -> None:
        """Test parsing OpenAI-style choices response."""
        response = {
            "choices": [
                {
                    "text": "    if n <= 1:",
                    "finish_reason": "length",
                }
            ],
            "usage": {
                "completion_tokens": 10,
            },
        }

        result = GenerationResult.from_response(response, latency_ms=200.0)

        assert result.text == "    if n <= 1:"
        assert result.finish_reason == FinishReason.LENGTH
        assert result.tokens_generated == 10

    def test_from_message_response(self) -> None:
        """Test parsing chat message response."""
        response = {
            "choices": [
                {
                    "message": {"content": "Generated text here"},
                    "finish_reason": "stop",
                }
            ],
        }

        result = GenerationResult.from_response(response, latency_ms=100.0)

        assert result.text == "Generated text here"

    def test_with_constraint_info(self) -> None:
        """Test parsing response with constraint metadata."""
        response = {
            "text": "result",
            "finish_reason": "constraint_satisfied",
            "constraint_info": {
                "satisfied": True,
                "relaxation_occurred": True,
                "domains_relaxed": ["semantics"],
                "relaxation_count": 2,
                "final_popcount": 150,
            },
        }

        result = GenerationResult.from_response(response, latency_ms=50.0)

        assert result.finish_reason == FinishReason.CONSTRAINT_SATISFIED
        assert result.constraint_satisfied is True
        assert result.relaxation.occurred is True
        assert result.relaxation.domains_relaxed == ["semantics"]
        assert result.relaxation.relaxation_count == 2
        assert result.relaxation.final_popcount == 150

    def test_constraint_violation(self) -> None:
        """Test parsing constraint violation response."""
        response = {
            "text": "invalid output",
            "finish_reason": "constraint_violation",
        }

        result = GenerationResult.from_response(response, latency_ms=30.0)

        assert result.finish_reason == FinishReason.CONSTRAINT_VIOLATION
        assert result.constraint_satisfied is False


# =============================================================================
# StreamChunk Tests
# =============================================================================


class TestStreamChunk:
    """Tests for StreamChunk parsing."""

    def test_from_sse_data(self) -> None:
        """Test parsing SSE event data."""
        data = {
            "text": "def ",
            "usage": {"completion_tokens": 1},
        }

        chunk = StreamChunk.from_sse_data(data)

        assert chunk.text == "def "
        assert chunk.is_final is False
        assert chunk.tokens_so_far == 1
        assert chunk.finish_reason is None

    def test_from_final_sse_data(self) -> None:
        """Test parsing final SSE event."""
        data = {
            "text": "",
            "finish_reason": "stop",
            "usage": {"completion_tokens": 10},
        }

        chunk = StreamChunk.from_sse_data(data)

        assert chunk.is_final is True
        assert chunk.finish_reason == FinishReason.STOP

    def test_from_delta_format(self) -> None:
        """Test parsing OpenAI delta format."""
        data = {
            "choices": [{"delta": {"content": "hello"}}],
        }

        chunk = StreamChunk.from_sse_data(data)

        assert chunk.text == "hello"


# =============================================================================
# ConstraintBuilder Tests
# =============================================================================


class TestConstraintBuilder:
    """Tests for ConstraintBuilder fluent API."""

    def test_simple_regex(self) -> None:
        """Test building simple regex constraint."""
        spec = (
            ConstraintBuilder()
            .language("python")
            .regex(r"^return\s+")
            .build()
        )

        assert spec.language == "python"
        assert spec.regex == r"^return\s+"

    def test_full_builder(self) -> None:
        """Test building complete constraint spec."""
        spec = (
            ConstraintBuilder()
            .language("python")
            .regex(r"^return\s+")
            .type_binding("x", "int", scope="parameter")
            .type_binding("y", "str", scope="parameter")
            .expected_type("Union[int, str]")
            .forbidden_import("os")
            .forbidden_import("subprocess")
            .in_function("process", return_type="Union[int, str]")
            .precondition("x >= 0", scope="process")
            .build()
        )

        assert spec.language == "python"
        assert spec.regex == r"^return\s+"
        assert len(spec.type_bindings) == 2
        assert spec.type_bindings[0].name == "x"
        assert spec.type_bindings[0].type_expr == "int"
        assert spec.expected_type == "Union[int, str]"
        assert "os" in spec.forbidden_imports
        assert "subprocess" in spec.forbidden_imports
        assert spec.control_flow is not None
        assert spec.control_flow.function_name == "process"
        assert len(spec.semantic_constraints) == 1
        assert spec.semantic_constraints[0].kind == "precondition"

    def test_to_dict(self) -> None:
        """Test getting builder config as dict."""
        builder = (
            ConstraintBuilder()
            .language("rust")
            .regex(r"^let\s+")
        )

        d = builder.to_dict()

        assert d["language"] == "rust"
        assert d["regex"] == r"^let\s+"

    def test_json_schema_dict(self) -> None:
        """Test building JSON schema constraint from dict."""
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}

        spec = (
            ConstraintBuilder()
            .json_schema(schema)
            .build()
        )

        # Schema should be serialized to JSON string
        assert spec.json_schema is not None
        parsed = json.loads(spec.json_schema)
        assert parsed["type"] == "object"

    def test_loop_context(self) -> None:
        """Test building constraint with loop context."""
        spec = (
            ConstraintBuilder()
            .language("python")
            .in_loop(depth=2, variables=["i", "j"])
            .build()
        )

        assert spec.control_flow is not None
        assert spec.control_flow.loop_depth == 2
        assert spec.control_flow.loop_variables == ("i", "j")

    def test_intensity(self) -> None:
        """Test setting constraint intensity."""
        spec = (
            ConstraintBuilder()
            .language("python")
            .intensity("full")
            .build()
        )

        assert spec.intensity == "full"


# =============================================================================
# ConstraintSpec Factory Tests
# =============================================================================


class TestConstraintSpecFactories:
    """Tests for ConstraintSpec factory methods."""

    def test_regex_factory(self) -> None:
        """Test regex factory method."""
        spec = ConstraintSpec.from_regex(r"^return\s+\d+")

        assert spec.regex == r"^return\s+\d+"
        assert spec.language is None
        assert spec.json_schema is None

    def test_regex_with_language(self) -> None:
        """Test regex factory with language hint."""
        spec = ConstraintSpec.from_regex(
            r"^return\s+",
            language="python",
            expected_type="int",
        )

        assert spec.regex == r"^return\s+"
        assert spec.language == "python"
        assert spec.expected_type == "int"

    def test_json_schema_factory_string(self) -> None:
        """Test JSON schema factory with string."""
        schema = '{"type": "string"}'
        spec = ConstraintSpec.from_json_schema(schema)

        assert spec.json_schema == schema

    def test_json_schema_factory_dict(self) -> None:
        """Test JSON schema factory with dict."""
        schema = {"type": "number", "minimum": 0}
        spec = ConstraintSpec.from_json_schema(schema)

        assert spec.json_schema is not None
        parsed = json.loads(spec.json_schema)
        assert parsed["type"] == "number"
        assert parsed["minimum"] == 0

    def test_ebnf_factory(self) -> None:
        """Test EBNF factory method."""
        grammar = 'root ::= "hello" | "world"'
        spec = ConstraintSpec.from_ebnf(grammar, language="text")

        assert spec.ebnf == grammar
        assert spec.language == "text"

    def test_for_completion_factory(self) -> None:
        """Test for_completion factory method."""
        spec = ConstraintSpec.for_completion(
            language="python",
            expected_type="int",
            in_function="fibonacci",
            return_type="int",
            type_bindings=[TypeBinding("n", "int", scope="parameter")],
        )

        assert spec.language == "python"
        assert spec.expected_type == "int"
        assert spec.control_flow is not None
        assert spec.control_flow.function_name == "fibonacci"
        assert spec.control_flow.expected_return_type == "int"
        assert len(spec.type_bindings) == 1


# =============================================================================
# AnankeClient Tests (Mocked)
# =============================================================================


class TestAnankeClientMocked:
    """Tests for AnankeClient with mocked HTTP transport."""

    @patch("ananke.client.SyncHttpTransport")
    def test_generate_simple(self, mock_transport_cls: MagicMock) -> None:
        """Test simple generation request."""
        mock_transport = MagicMock()
        mock_transport.post.return_value = (
            {"text": "generated text", "finish_reason": "stop"},
            100.0,
        )
        mock_transport_cls.return_value = mock_transport

        client = AnankeClient("http://localhost:8000")
        result = client.generate("Hello, ")

        assert result.text == "generated text"
        assert result.finish_reason == FinishReason.STOP

        # Verify the request was made
        mock_transport.post.assert_called_once()
        call_args = mock_transport.post.call_args
        assert call_args[0][0] == "/v1/completions"
        request_body = call_args[0][1]
        assert request_body["prompt"] == "Hello, "

    @patch("ananke.client.SyncHttpTransport")
    def test_generate_with_constraint(self, mock_transport_cls: MagicMock) -> None:
        """Test generation with constraint specification."""
        mock_transport = MagicMock()
        mock_transport.post.return_value = (
            {"text": "return n", "finish_reason": "stop"},
            50.0,
        )
        mock_transport_cls.return_value = mock_transport

        client = AnankeClient("http://localhost:8000")
        spec = ConstraintSpec.from_regex(r"^return\s+", language="python")
        result = client.generate("def double(n):\n    ", constraint=spec)

        assert result.text == "return n"

        # Verify constraint was included
        call_args = mock_transport.post.call_args
        request_body = call_args[0][1]
        assert "constraint_spec" in request_body
        assert request_body["constraint_spec"]["regex"] == r"^return\s+"
        assert request_body["constraint_spec"]["language"] == "python"

    @patch("ananke.client.SyncHttpTransport")
    def test_generate_with_string_constraint(self, mock_transport_cls: MagicMock) -> None:
        """Test generation with string regex shorthand."""
        mock_transport = MagicMock()
        mock_transport.post.return_value = (
            {"text": "test", "finish_reason": "stop"},
            25.0,
        )
        mock_transport_cls.return_value = mock_transport

        client = AnankeClient("http://localhost:8000")
        result = client.generate("prompt", constraint=r"^test")

        # String should be treated as regex
        call_args = mock_transport.post.call_args
        request_body = call_args[0][1]
        assert request_body["constraint_spec"]["regex"] == r"^test"

    @patch("ananke.client.SyncHttpTransport")
    def test_generate_with_config_overrides(self, mock_transport_cls: MagicMock) -> None:
        """Test generation with config overrides via kwargs."""
        mock_transport = MagicMock()
        mock_transport.post.return_value = (
            {"text": "output", "finish_reason": "stop"},
            30.0,
        )
        mock_transport_cls.return_value = mock_transport

        client = AnankeClient("http://localhost:8000")
        result = client.generate(
            "prompt",
            max_tokens=512,
            temperature=0.0,
            seed=42,
        )

        call_args = mock_transport.post.call_args
        request_body = call_args[0][1]
        assert request_body["max_new_tokens"] == 512
        assert request_body["temperature"] == 0.0
        assert request_body["seed"] == 42

    @patch("ananke.client.SyncHttpTransport")
    def test_context_manager(self, mock_transport_cls: MagicMock) -> None:
        """Test client as context manager."""
        mock_transport = MagicMock()
        mock_transport.post.return_value = ({"text": "x", "finish_reason": "stop"}, 10.0)
        mock_transport_cls.return_value = mock_transport

        with AnankeClient("http://localhost:8000") as client:
            result = client.generate("test")
            assert result.text == "x"

        # Verify close was called
        mock_transport.close.assert_called_once()


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    def test_http_transport_error(self) -> None:
        """Test HttpTransportError attributes."""
        error = HttpTransportError(
            "Request failed",
            status_code=500,
            response_body="Internal Server Error",
        )

        assert str(error) == "Request failed"
        assert error.status_code == 500
        assert error.response_body == "Internal Server Error"


# =============================================================================
# Integration Patterns Tests
# =============================================================================


class TestIntegrationPatterns:
    """Tests demonstrating common usage patterns."""

    def test_complete_workflow_pattern(self) -> None:
        """Test complete workflow pattern (no actual HTTP)."""
        # This demonstrates the intended API usage pattern
        # without making actual HTTP requests

        # Pattern 1: Simple regex
        spec1 = ConstraintSpec.from_regex(r"^\s+if n <= 1:")

        assert spec1.has_syntax_constraint()
        assert not spec1.has_domain_context()

        # Pattern 2: Builder for complex constraints
        spec2 = (
            ConstraintBuilder()
            .language("python")
            .regex(r"^return\s+")
            .type_binding("n", "int", scope="parameter")
            .expected_type("int")
            .in_function("fibonacci", return_type="int")
            .build()
        )

        assert spec2.has_syntax_constraint()
        assert spec2.has_domain_context()
        assert spec2.language == "python"

        # Pattern 3: JSON schema for structured output
        spec3 = ConstraintSpec.from_json_schema({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "value": {"type": "number"},
            },
            "required": ["name", "value"],
        })

        assert spec3.json_schema is not None

        # Pattern 4: Code completion context
        spec4 = ConstraintSpec.for_completion(
            language="rust",
            expected_type="Result<Config, Error>",
            in_function="load_config",
            return_type="Result<Config, Error>",
        )

        assert spec4.control_flow is not None
        assert spec4.control_flow.function_name == "load_config"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
