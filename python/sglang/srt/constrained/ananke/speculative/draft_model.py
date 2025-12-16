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
"""Draft model protocol for speculative decoding.

This module defines the protocol for draft models used in CDSL-style
speculative decoding. Draft models generate candidate token sequences
that are then verified against full constraint systems.

Key Insight (from CDSL paper, NAACL 2025):
    Using a smaller/faster draft model to generate candidates, then
    verifying against full constraints, achieves 2.2x-12.15x speedup
    while maintaining output quality.

Design Principles:
1. Draft models should be fast (smaller model or cached predictions)
2. Draft tokens use relaxed constraints (syntax-only) for speed
3. Verification uses full domain stack for correctness
4. Accept longest valid prefix from draft sequence

References:
    - CDSL: Constrained Decoding with Speculative Lookaheads (NAACL 2025)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, runtime_checkable

import torch

logger = logging.getLogger(__name__)


@dataclass
class DraftContext:
    """Context for draft generation.

    Provides information about the current generation state to help
    the draft model produce better candidates.

    Attributes:
        prefix_tokens: Tokens generated so far
        prefix_text: Text generated so far
        prompt: Original prompt/instruction
        temperature: Sampling temperature
        language: Programming language
        expected_type: Expected type at current position (if known)
        metadata: Additional context
    """

    prefix_tokens: List[int] = field(default_factory=list)
    prefix_text: str = ""
    prompt: str = ""
    temperature: float = 1.0
    language: str = "python"
    expected_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DraftResult:
    """Result from draft model generation.

    Attributes:
        tokens: Draft token sequence
        log_probs: Log probabilities for each token (optional)
        scores: Quality scores for each token (optional)
        latency_ms: Generation latency
    """

    tokens: List[int]
    log_probs: Optional[List[float]] = None
    scores: Optional[List[float]] = None
    latency_ms: float = 0.0

    @property
    def length(self) -> int:
        """Number of draft tokens."""
        return len(self.tokens)


@runtime_checkable
class DraftModel(Protocol):
    """Protocol for draft models in speculative decoding.

    Draft models generate candidate token sequences that are verified
    against full constraints. They should be faster than the main model,
    either by being smaller or by using cached predictions.

    Implementations:
    - GreedyDraftModel: Uses greedy sampling (fastest, lowest quality)
    - SamplingDraftModel: Uses temperature sampling
    - CachedDraftModel: Uses cached predictions from previous runs
    - DistilledDraftModel: Uses a distilled smaller model

    Example:
        class MyDraftModel:
            def generate_draft(
                self,
                context: DraftContext,
                lookahead_length: int = 5,
            ) -> DraftResult:
                # Generate draft tokens
                tokens = self._generate(context, lookahead_length)
                return DraftResult(tokens=tokens)
    """

    def generate_draft(
        self,
        context: DraftContext,
        lookahead_length: int = 5,
    ) -> DraftResult:
        """Generate draft token sequence.

        Args:
            context: Current generation context
            lookahead_length: Number of tokens to generate

        Returns:
            DraftResult with generated tokens
        """
        ...


class NullDraftModel:
    """Null draft model that returns empty drafts.

    Used as a fallback when no draft model is configured.
    """

    def generate_draft(
        self,
        context: DraftContext,
        lookahead_length: int = 5,
    ) -> DraftResult:
        """Return empty draft."""
        return DraftResult(tokens=[])


class GreedyDraftModel:
    """Draft model using greedy sampling from logits.

    Generates draft tokens by taking the argmax of logits at each step.
    Fast but may not produce diverse candidates.

    Attributes:
        logits_fn: Function that returns logits for next token
        syntax_mask_fn: Optional function that returns syntax-only mask
    """

    def __init__(
        self,
        logits_fn: Callable[[List[int]], torch.Tensor],
        syntax_mask_fn: Optional[Callable[[List[int]], torch.Tensor]] = None,
        vocab_size: int = 32000,
    ):
        """Initialize greedy draft model.

        Args:
            logits_fn: Function(prefix_tokens) -> logits tensor
            syntax_mask_fn: Optional function(prefix_tokens) -> mask tensor
            vocab_size: Vocabulary size
        """
        self._logits_fn = logits_fn
        self._syntax_mask_fn = syntax_mask_fn
        self._vocab_size = vocab_size

    def generate_draft(
        self,
        context: DraftContext,
        lookahead_length: int = 5,
    ) -> DraftResult:
        """Generate draft using greedy sampling."""
        import time

        start_time = time.perf_counter()

        tokens: List[int] = []
        log_probs: List[float] = []
        prefix = context.prefix_tokens.copy()

        for _ in range(lookahead_length):
            # Get logits for next token
            logits = self._logits_fn(prefix)

            # Apply syntax mask if available (relaxed constraints for speed)
            if self._syntax_mask_fn is not None:
                mask = self._syntax_mask_fn(prefix)
                # Set masked positions to -inf
                logits = logits.clone()
                logits[~mask] = float("-inf")

            # Greedy selection
            probs = torch.softmax(logits, dim=-1)
            token = int(torch.argmax(logits).item())

            # Check for valid token
            if token >= self._vocab_size or probs[token].item() < 1e-10:
                break

            tokens.append(token)
            log_probs.append(float(torch.log(probs[token]).item()))
            prefix.append(token)

        latency_ms = (time.perf_counter() - start_time) * 1000

        return DraftResult(
            tokens=tokens,
            log_probs=log_probs,
            latency_ms=latency_ms,
        )


class SamplingDraftModel:
    """Draft model using temperature sampling from logits.

    Generates draft tokens by sampling from the softmax distribution.
    Produces more diverse candidates than greedy sampling.

    Attributes:
        logits_fn: Function that returns logits for next token
        syntax_mask_fn: Optional function that returns syntax-only mask
        temperature: Sampling temperature (higher = more diverse)
        top_k: Number of top tokens to sample from (0 = no limit)
        top_p: Cumulative probability threshold (0 = no limit)
    """

    def __init__(
        self,
        logits_fn: Callable[[List[int]], torch.Tensor],
        syntax_mask_fn: Optional[Callable[[List[int]], torch.Tensor]] = None,
        vocab_size: int = 32000,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
    ):
        """Initialize sampling draft model.

        Args:
            logits_fn: Function(prefix_tokens) -> logits tensor
            syntax_mask_fn: Optional function(prefix_tokens) -> mask tensor
            vocab_size: Vocabulary size
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
        """
        self._logits_fn = logits_fn
        self._syntax_mask_fn = syntax_mask_fn
        self._vocab_size = vocab_size
        self._temperature = temperature
        self._top_k = top_k
        self._top_p = top_p

    def generate_draft(
        self,
        context: DraftContext,
        lookahead_length: int = 5,
    ) -> DraftResult:
        """Generate draft using temperature sampling."""
        import time

        start_time = time.perf_counter()

        tokens: List[int] = []
        log_probs: List[float] = []
        prefix = context.prefix_tokens.copy()

        # Use context temperature if different from default
        temperature = context.temperature if context.temperature > 0 else self._temperature

        for _ in range(lookahead_length):
            # Get logits for next token
            logits = self._logits_fn(prefix)

            # Apply syntax mask if available
            if self._syntax_mask_fn is not None:
                mask = self._syntax_mask_fn(prefix)
                logits = logits.clone()
                logits[~mask] = float("-inf")

            # Apply temperature
            logits = logits / temperature

            # Apply top-k
            if self._top_k > 0:
                top_k_values, top_k_indices = torch.topk(logits, min(self._top_k, logits.size(-1)))
                logits = torch.full_like(logits, float("-inf"))
                logits.scatter_(-1, top_k_indices, top_k_values)

            # Apply top-p (nucleus sampling)
            if self._top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > self._top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False

                indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float("-inf")

            # Sample
            probs = torch.softmax(logits, dim=-1)
            token = int(torch.multinomial(probs, num_samples=1).item())

            # Check for valid token
            if token >= self._vocab_size or probs[token].item() < 1e-10:
                break

            tokens.append(token)
            log_probs.append(float(torch.log(probs[token]).item()))
            prefix.append(token)

        latency_ms = (time.perf_counter() - start_time) * 1000

        return DraftResult(
            tokens=tokens,
            log_probs=log_probs,
            latency_ms=latency_ms,
        )


class CachedDraftModel:
    """Draft model using cached predictions.

    Uses cached token sequences from previous generations to avoid
    recomputation. Useful when generating similar code patterns.

    Attributes:
        cache: Dictionary mapping context hashes to cached sequences
        fallback: Fallback draft model if cache miss
    """

    def __init__(
        self,
        fallback: DraftModel,
        max_cache_size: int = 1000,
    ):
        """Initialize cached draft model.

        Args:
            fallback: Fallback model for cache misses
            max_cache_size: Maximum number of cached sequences
        """
        self._fallback = fallback
        self._max_cache_size = max_cache_size
        self._cache: Dict[int, DraftResult] = {}

    def generate_draft(
        self,
        context: DraftContext,
        lookahead_length: int = 5,
    ) -> DraftResult:
        """Generate draft from cache or fallback."""
        import time

        start_time = time.perf_counter()

        # Create cache key
        cache_key = self._make_key(context)

        # Check cache
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            # Truncate if needed
            if cached.length >= lookahead_length:
                return DraftResult(
                    tokens=cached.tokens[:lookahead_length],
                    log_probs=cached.log_probs[:lookahead_length] if cached.log_probs else None,
                    latency_ms=(time.perf_counter() - start_time) * 1000,
                )

        # Cache miss - use fallback
        result = self._fallback.generate_draft(context, lookahead_length)

        # Cache result
        self._cache[cache_key] = result

        # Evict oldest if cache is full
        while len(self._cache) > self._max_cache_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        return result

    def _make_key(self, context: DraftContext) -> int:
        """Create cache key from context."""
        # Hash based on prefix tokens and expected type
        key_tuple = (
            tuple(context.prefix_tokens[-20:]),  # Last 20 tokens
            context.expected_type,
            context.language,
        )
        return hash(key_tuple)

    def clear_cache(self) -> None:
        """Clear the cache."""
        self._cache.clear()


def create_draft_model(
    model_type: str = "null",
    logits_fn: Optional[Callable[[List[int]], torch.Tensor]] = None,
    syntax_mask_fn: Optional[Callable[[List[int]], torch.Tensor]] = None,
    **kwargs: Any,
) -> DraftModel:
    """Factory function to create draft models.

    Args:
        model_type: Type of model ("null", "greedy", "sampling", "cached")
        logits_fn: Function to get logits (required for greedy/sampling)
        syntax_mask_fn: Optional function to get syntax mask
        **kwargs: Additional arguments for specific model types

    Returns:
        DraftModel instance
    """
    if model_type == "null":
        return NullDraftModel()
    elif model_type == "greedy":
        if logits_fn is None:
            raise ValueError("greedy model requires logits_fn")
        return GreedyDraftModel(logits_fn, syntax_mask_fn, **kwargs)
    elif model_type == "sampling":
        if logits_fn is None:
            raise ValueError("sampling model requires logits_fn")
        return SamplingDraftModel(logits_fn, syntax_mask_fn, **kwargs)
    elif model_type == "cached":
        fallback = create_draft_model(
            kwargs.pop("fallback_type", "null"),
            logits_fn,
            syntax_mask_fn,
            **kwargs,
        )
        return CachedDraftModel(fallback, **kwargs)
    else:
        logger.warning(f"Unknown draft model type '{model_type}', using null")
        return NullDraftModel()
