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
"""Data models for Ananke client SDK.

This module defines the core data structures for generation requests and results,
including:
- GenerationConfig: Configuration for generation requests
- GenerationResult: Result of a constrained generation
- FinishReason: Why generation stopped
- StreamChunk: Chunk of streamed generation output
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class FinishReason(Enum):
    """Reason why generation finished.

    Attributes:
        STOP: Normal stop token reached
        LENGTH: Maximum tokens reached
        CONSTRAINT_VIOLATION: Generation violated constraint
        CONSTRAINT_SATISFIED: Constraint fully satisfied (early termination)
        ERROR: Generation error occurred
    """

    STOP = "stop"
    LENGTH = "length"
    CONSTRAINT_VIOLATION = "constraint_violation"
    CONSTRAINT_SATISFIED = "constraint_satisfied"
    ERROR = "error"

    def __str__(self) -> str:
        return self.value


@dataclass
class GenerationConfig:
    """Configuration for constrained generation requests.

    Controls sampling parameters, constraint behavior, and resource limits.

    Attributes:
        max_tokens: Maximum tokens to generate (default: 256)
        temperature: Sampling temperature (default: 0.7, 0.0 for deterministic)
        top_p: Nucleus sampling parameter (default: 1.0)
        top_k: Top-k sampling parameter (default: -1, disabled)
        seed: Random seed for reproducibility (optional)
        allow_relaxation: Whether to relax constraints if too tight (default: True)
        relaxation_threshold: Minimum mask popcount before relaxation (default: 10)
        enable_early_termination: Stop when regex satisfied (default: True)
        stream: Enable streaming response (default: False)
        timeout_seconds: Request timeout in seconds (default: 60.0)

    Example:
        >>> config = GenerationConfig(
        ...     max_tokens=512,
        ...     temperature=0.0,  # Deterministic
        ...     seed=42,
        ...     enable_early_termination=True,
        ... )
    """

    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = -1
    seed: Optional[int] = None
    allow_relaxation: bool = True
    relaxation_threshold: int = 10
    enable_early_termination: bool = True
    stream: bool = False
    timeout_seconds: float = 60.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API request."""
        d: Dict[str, Any] = {
            "max_new_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        if self.top_k > 0:
            d["top_k"] = self.top_k
        if self.seed is not None:
            d["seed"] = self.seed
        if not self.allow_relaxation:
            d["constraint_relaxation"] = False
        if self.relaxation_threshold != 10:
            d["relaxation_threshold"] = self.relaxation_threshold
        if not self.enable_early_termination:
            d["early_termination"] = False
        return d

    @classmethod
    def deterministic(cls, seed: int = 42, **kwargs: Any) -> "GenerationConfig":
        """Create deterministic config for reproducible generation.

        Args:
            seed: Random seed (default: 42)
            **kwargs: Additional config overrides

        Returns:
            GenerationConfig with temperature=0.0 and specified seed
        """
        return cls(temperature=0.0, seed=seed, **kwargs)


@dataclass
class RelaxationInfo:
    """Information about constraint relaxation during generation.

    Attributes:
        occurred: Whether any relaxation occurred
        domains_relaxed: List of domain names that were relaxed
        relaxation_count: Number of times relaxation was triggered
        final_popcount: Token mask popcount at completion
    """

    occurred: bool = False
    domains_relaxed: List[str] = field(default_factory=list)
    relaxation_count: int = 0
    final_popcount: Optional[int] = None


@dataclass
class GenerationResult:
    """Result of a constrained generation request.

    Contains the generated text along with metadata about the generation process,
    including constraint satisfaction, timing, and any relaxation that occurred.

    Attributes:
        text: Generated text output
        finish_reason: Why generation stopped
        tokens_generated: Number of tokens generated
        constraint_satisfied: Whether the constraint was fully satisfied
        latency_ms: Total request latency in milliseconds
        relaxation: Relaxation information (if applicable)
        early_termination: Whether early termination was triggered
        prompt_tokens: Number of prompt tokens (if available)
        usage: Full token usage dict from API (if available)
        error: Error message if finish_reason is ERROR

    Example:
        >>> result = client.generate("def fib(n):", constraint)
        >>> if result.constraint_satisfied:
        ...     print(f"Generated {result.tokens_generated} tokens in {result.latency_ms}ms")
        ...     print(result.text)
    """

    text: str
    finish_reason: FinishReason
    tokens_generated: int
    constraint_satisfied: bool
    latency_ms: float
    relaxation: RelaxationInfo = field(default_factory=RelaxationInfo)
    early_termination: bool = False
    prompt_tokens: Optional[int] = None
    usage: Optional[Dict[str, int]] = None
    error: Optional[str] = None

    @classmethod
    def from_response(
        cls,
        response: Dict[str, Any],
        latency_ms: float,
    ) -> "GenerationResult":
        """Create from SGLang API response.

        Args:
            response: Raw API response dict
            latency_ms: Measured request latency

        Returns:
            Parsed GenerationResult
        """
        # Extract text from various response formats
        text = ""
        if "text" in response:
            text = response["text"]
        elif "choices" in response and response["choices"]:
            choice = response["choices"][0]
            if "text" in choice:
                text = choice["text"]
            elif "message" in choice:
                text = choice["message"].get("content", "")

        # Parse finish reason
        finish_reason_str = response.get("finish_reason", "stop")
        if "choices" in response and response["choices"]:
            finish_reason_str = response["choices"][0].get("finish_reason", "stop")

        try:
            finish_reason = FinishReason(finish_reason_str)
        except ValueError:
            finish_reason = (
                FinishReason.STOP if finish_reason_str == "stop" else FinishReason.ERROR
            )

        # Extract token counts
        tokens_generated = 0
        prompt_tokens = None
        usage = response.get("usage")
        if usage:
            tokens_generated = usage.get("completion_tokens", 0)
            prompt_tokens = usage.get("prompt_tokens")

        # Parse relaxation info
        relaxation = RelaxationInfo()
        if "constraint_info" in response:
            constraint_info = response["constraint_info"]
            relaxation = RelaxationInfo(
                occurred=constraint_info.get("relaxation_occurred", False),
                domains_relaxed=constraint_info.get("domains_relaxed", []),
                relaxation_count=constraint_info.get("relaxation_count", 0),
                final_popcount=constraint_info.get("final_popcount"),
            )

        # Determine constraint satisfaction
        constraint_satisfied = finish_reason not in (
            FinishReason.CONSTRAINT_VIOLATION,
            FinishReason.ERROR,
        )
        if "constraint_info" in response:
            constraint_satisfied = response["constraint_info"].get(
                "satisfied", constraint_satisfied
            )

        return cls(
            text=text,
            finish_reason=finish_reason,
            tokens_generated=tokens_generated,
            constraint_satisfied=constraint_satisfied,
            latency_ms=latency_ms,
            relaxation=relaxation,
            early_termination=response.get("early_termination", False),
            prompt_tokens=prompt_tokens,
            usage=usage,
            error=response.get("error"),
        )


@dataclass
class StreamChunk:
    """A chunk of streamed generation output.

    Attributes:
        text: Text content of this chunk
        is_final: Whether this is the final chunk
        tokens_so_far: Cumulative tokens generated so far
        finish_reason: Finish reason (only set on final chunk)
    """

    text: str
    is_final: bool = False
    tokens_so_far: int = 0
    finish_reason: Optional[FinishReason] = None

    @classmethod
    def from_sse_data(cls, data: Dict[str, Any]) -> "StreamChunk":
        """Parse from SSE event data.

        Args:
            data: Parsed JSON data from SSE event

        Returns:
            Parsed StreamChunk
        """
        text = ""
        if "text" in data:
            text = data["text"]
        elif "choices" in data and data["choices"]:
            delta = data["choices"][0].get("delta", {})
            text = delta.get("content", "")

        is_final = data.get("finish_reason") is not None
        finish_reason = None
        if is_final:
            try:
                finish_reason = FinishReason(data.get("finish_reason", "stop"))
            except ValueError:
                finish_reason = FinishReason.STOP

        return cls(
            text=text,
            is_final=is_final,
            tokens_so_far=data.get("usage", {}).get("completion_tokens", 0),
            finish_reason=finish_reason,
        )
