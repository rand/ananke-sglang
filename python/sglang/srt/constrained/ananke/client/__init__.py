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
"""Ananke Client SDK for constrained generation.

This module provides ergonomic Python clients for SGLang with Ananke constraints.

Classes:
    AnankeClient: Synchronous client for constrained generation
    AnankeAsyncClient: Asynchronous client for constrained generation
    GenerationConfig: Configuration for generation requests
    GenerationResult: Result of a generation request
    ConstraintBuilder: Fluent builder for constraint specifications

Example:
    >>> from ananke.client import AnankeClient
    >>> from ananke.spec.constraint_spec import ConstraintSpec
    >>>
    >>> client = AnankeClient("http://localhost:8000")
    >>>
    >>> # Simple regex constraint
    >>> result = client.generate(
    ...     prompt="def fibonacci(n):",
    ...     constraint=ConstraintSpec.from_regex(r"^\\s+if n <= 1"),
    ...     max_tokens=256,
    ... )
    >>>
    >>> # Rich constraint with type context
    >>> from ananke.client import ConstraintBuilder
    >>> spec = (
    ...     ConstraintBuilder()
    ...     .language("python")
    ...     .regex(r"^return.*")
    ...     .type_binding("n", "int")
    ...     .expected_type("int")
    ...     .build()
    ... )
    >>> result = client.generate(prompt, constraint=spec)
    >>>
    >>> # Streaming
    >>> async for chunk in async_client.generate_stream(prompt, constraint=spec):
    ...     print(chunk.text, end="")
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    Optional,
    Union,
)

from .http import (
    AsyncHttpTransport,
    HttpTransportError,
    LoggingConfig,
    RetryConfig,
    SyncHttpTransport,
)
from .models import (
    FinishReason,
    GenerationConfig,
    GenerationResult,
    RelaxationInfo,
    StreamChunk,
)

if TYPE_CHECKING:
    from ..spec.constraint_spec import ConstraintSpec

__all__ = [
    "AnankeClient",
    "AnankeAsyncClient",
    "GenerationConfig",
    "GenerationResult",
    "FinishReason",
    "RelaxationInfo",
    "StreamChunk",
    "ConstraintBuilder",
    "HttpTransportError",
    "RetryConfig",
    "LoggingConfig",
]


class AnankeClient:
    """Synchronous client for Ananke-constrained generation.

    Provides a simple, ergonomic interface for constrained text generation
    with SGLang servers running the Ananke backend.

    Attributes:
        base_url: Base URL of the SGLang server
        model: Default model to use for generation
        default_config: Default generation configuration

    Example:
        >>> client = AnankeClient("http://localhost:8000")
        >>> result = client.generate(
        ...     prompt="def hello():",
        ...     constraint=ConstraintSpec.from_regex(r'^\\s+return "'),
        ...     max_tokens=100,
        ... )
        >>> print(result.text)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        model: Optional[str] = None,
        default_config: Optional[GenerationConfig] = None,
        timeout: float = 60.0,
        headers: Optional[Dict[str, str]] = None,
        retry_config: Optional[RetryConfig] = None,
        logging_config: Optional[LoggingConfig] = None,
    ):
        """Initialize the Ananke client.

        Args:
            base_url: SGLang server URL (default: localhost:8000)
            model: Default model name (optional, uses server default)
            default_config: Default generation config
            timeout: Request timeout in seconds
            headers: Additional HTTP headers
            retry_config: Retry configuration for transient failures
            logging_config: Logging configuration for debugging
        """
        self.base_url = base_url
        self.model = model
        self.default_config = default_config or GenerationConfig()
        self._transport = SyncHttpTransport(
            base_url, timeout, headers, retry_config, logging_config
        )

    def generate(
        self,
        prompt: str,
        constraint: Optional[Union["ConstraintSpec", Dict[str, Any], str]] = None,
        config: Optional[GenerationConfig] = None,
        **kwargs: Any,
    ) -> GenerationResult:
        """Generate text with optional constraints.

        Args:
            prompt: Input prompt text
            constraint: Constraint specification (ConstraintSpec, dict, or regex string)
            config: Generation configuration (uses default if not provided)
            **kwargs: Override config fields (max_tokens, temperature, etc.)

        Returns:
            GenerationResult with generated text and metadata

        Example:
            >>> # Simple generation
            >>> result = client.generate("Hello, ")
            >>>
            >>> # With regex constraint
            >>> result = client.generate(
            ...     "def add(a, b):",
            ...     constraint=ConstraintSpec.regex(r"^\\s+return a \\+ b"),
            ... )
            >>>
            >>> # With full constraint spec
            >>> result = client.generate(
            ...     prompt,
            ...     constraint=spec,
            ...     max_tokens=512,
            ...     temperature=0.0,
            ... )
        """
        effective_config = self._merge_config(config, kwargs)
        request_body = self._build_request(prompt, constraint, effective_config)

        response, latency_ms = self._transport.post(
            "/v1/completions",
            request_body,
            timeout=effective_config.timeout_seconds,
        )

        return GenerationResult.from_response(response, latency_ms)

    def generate_stream(
        self,
        prompt: str,
        constraint: Optional[Union["ConstraintSpec", Dict[str, Any], str]] = None,
        config: Optional[GenerationConfig] = None,
        **kwargs: Any,
    ) -> Iterator[StreamChunk]:
        """Generate text with streaming output.

        Args:
            prompt: Input prompt text
            constraint: Constraint specification
            config: Generation configuration
            **kwargs: Override config fields

        Yields:
            StreamChunk for each token/chunk generated

        Example:
            >>> for chunk in client.generate_stream(prompt, constraint):
            ...     print(chunk.text, end="", flush=True)
        """
        effective_config = self._merge_config(config, kwargs)
        effective_config.stream = True
        request_body = self._build_request(prompt, constraint, effective_config)

        for event in self._transport.post_stream(
            "/v1/completions",
            request_body,
            timeout=effective_config.timeout_seconds,
        ):
            yield StreamChunk.from_sse_data(event)

    def health_check(self) -> bool:
        """Check if the server is healthy.

        Returns:
            True if server responds successfully
        """
        try:
            response, _ = self._transport.post("/health", {}, timeout=5.0)
            return True
        except HttpTransportError:
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model.

        Returns:
            Dict with model information

        Raises:
            HttpTransportError: If request fails
        """
        response, _ = self._transport.post("/v1/models", {}, timeout=10.0)
        return response

    def close(self) -> None:
        """Close the client and release resources."""
        self._transport.close()

    def __enter__(self) -> "AnankeClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def _merge_config(
        self,
        config: Optional[GenerationConfig],
        overrides: Dict[str, Any],
    ) -> GenerationConfig:
        """Merge config with defaults and overrides."""
        base = config or self.default_config

        # Apply overrides from kwargs
        if overrides:
            return GenerationConfig(
                max_tokens=overrides.get("max_tokens", base.max_tokens),
                temperature=overrides.get("temperature", base.temperature),
                top_p=overrides.get("top_p", base.top_p),
                top_k=overrides.get("top_k", base.top_k),
                seed=overrides.get("seed", base.seed),
                allow_relaxation=overrides.get("allow_relaxation", base.allow_relaxation),
                relaxation_threshold=overrides.get(
                    "relaxation_threshold", base.relaxation_threshold
                ),
                enable_early_termination=overrides.get(
                    "enable_early_termination", base.enable_early_termination
                ),
                stream=overrides.get("stream", base.stream),
                timeout_seconds=overrides.get("timeout_seconds", base.timeout_seconds),
            )
        return base

    def _build_request(
        self,
        prompt: str,
        constraint: Optional[Union["ConstraintSpec", Dict[str, Any], str]],
        config: GenerationConfig,
    ) -> Dict[str, Any]:
        """Build the API request body."""
        request: Dict[str, Any] = {
            "prompt": prompt,
            **config.to_dict(),
        }

        if self.model:
            request["model"] = self.model

        if config.stream:
            request["stream"] = True

        # Handle constraint formats
        if constraint is not None:
            constraint_spec = self._normalize_constraint(constraint)
            request["constraint_spec"] = constraint_spec

        return request

    def _normalize_constraint(
        self,
        constraint: Union["ConstraintSpec", Dict[str, Any], str],
    ) -> Dict[str, Any]:
        """Normalize constraint to dict format."""
        if isinstance(constraint, str):
            # Treat string as regex
            return {"regex": constraint}
        elif isinstance(constraint, dict):
            return constraint
        else:
            # ConstraintSpec
            return constraint.to_dict()


class AnankeAsyncClient:
    """Asynchronous client for Ananke-constrained generation.

    Provides an async interface for constrained text generation with
    SGLang servers running the Ananke backend.

    Example:
        >>> async with AnankeAsyncClient("http://localhost:8000") as client:
        ...     result = await client.generate(
        ...         prompt="def hello():",
        ...         constraint=ConstraintSpec.from_regex(r'^\\s+return "'),
        ...     )
        ...     print(result.text)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        model: Optional[str] = None,
        default_config: Optional[GenerationConfig] = None,
        timeout: float = 60.0,
        headers: Optional[Dict[str, str]] = None,
        retry_config: Optional[RetryConfig] = None,
        logging_config: Optional[LoggingConfig] = None,
    ):
        """Initialize the async Ananke client.

        Args:
            base_url: SGLang server URL
            model: Default model name
            default_config: Default generation config
            timeout: Request timeout in seconds
            headers: Additional HTTP headers
            retry_config: Retry configuration for transient failures
            logging_config: Logging configuration for debugging
        """
        self.base_url = base_url
        self.model = model
        self.default_config = default_config or GenerationConfig()
        self._transport = AsyncHttpTransport(
            base_url, timeout, headers, retry_config, logging_config
        )

    async def generate(
        self,
        prompt: str,
        constraint: Optional[Union["ConstraintSpec", Dict[str, Any], str]] = None,
        config: Optional[GenerationConfig] = None,
        **kwargs: Any,
    ) -> GenerationResult:
        """Generate text with optional constraints (async).

        Args:
            prompt: Input prompt text
            constraint: Constraint specification
            config: Generation configuration
            **kwargs: Override config fields

        Returns:
            GenerationResult with generated text and metadata
        """
        effective_config = self._merge_config(config, kwargs)
        request_body = self._build_request(prompt, constraint, effective_config)

        response, latency_ms = await self._transport.post(
            "/v1/completions",
            request_body,
            timeout=effective_config.timeout_seconds,
        )

        return GenerationResult.from_response(response, latency_ms)

    async def generate_stream(
        self,
        prompt: str,
        constraint: Optional[Union["ConstraintSpec", Dict[str, Any], str]] = None,
        config: Optional[GenerationConfig] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Generate text with streaming output (async).

        Args:
            prompt: Input prompt text
            constraint: Constraint specification
            config: Generation configuration
            **kwargs: Override config fields

        Yields:
            StreamChunk for each token/chunk generated
        """
        effective_config = self._merge_config(config, kwargs)
        effective_config.stream = True
        request_body = self._build_request(prompt, constraint, effective_config)

        async for event in self._transport.post_stream(
            "/v1/completions",
            request_body,
            timeout=effective_config.timeout_seconds,
        ):
            yield StreamChunk.from_sse_data(event)

    async def health_check(self) -> bool:
        """Check if the server is healthy (async)."""
        try:
            await self._transport.post("/health", {}, timeout=5.0)
            return True
        except HttpTransportError:
            return False

    async def close(self) -> None:
        """Close the client and release resources."""
        await self._transport.close()

    async def __aenter__(self) -> "AnankeAsyncClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    def _merge_config(
        self,
        config: Optional[GenerationConfig],
        overrides: Dict[str, Any],
    ) -> GenerationConfig:
        """Merge config with defaults and overrides."""
        base = config or self.default_config

        if overrides:
            return GenerationConfig(
                max_tokens=overrides.get("max_tokens", base.max_tokens),
                temperature=overrides.get("temperature", base.temperature),
                top_p=overrides.get("top_p", base.top_p),
                top_k=overrides.get("top_k", base.top_k),
                seed=overrides.get("seed", base.seed),
                allow_relaxation=overrides.get("allow_relaxation", base.allow_relaxation),
                relaxation_threshold=overrides.get(
                    "relaxation_threshold", base.relaxation_threshold
                ),
                enable_early_termination=overrides.get(
                    "enable_early_termination", base.enable_early_termination
                ),
                stream=overrides.get("stream", base.stream),
                timeout_seconds=overrides.get("timeout_seconds", base.timeout_seconds),
            )
        return base

    def _build_request(
        self,
        prompt: str,
        constraint: Optional[Union["ConstraintSpec", Dict[str, Any], str]],
        config: GenerationConfig,
    ) -> Dict[str, Any]:
        """Build the API request body."""
        request: Dict[str, Any] = {
            "prompt": prompt,
            **config.to_dict(),
        }

        if self.model:
            request["model"] = self.model

        if config.stream:
            request["stream"] = True

        if constraint is not None:
            constraint_spec = self._normalize_constraint(constraint)
            request["constraint_spec"] = constraint_spec

        return request

    def _normalize_constraint(
        self,
        constraint: Union["ConstraintSpec", Dict[str, Any], str],
    ) -> Dict[str, Any]:
        """Normalize constraint to dict format."""
        if isinstance(constraint, str):
            return {"regex": constraint}
        elif isinstance(constraint, dict):
            return constraint
        else:
            return constraint.to_dict()


class ConstraintBuilder:
    """Fluent builder for constructing ConstraintSpec objects.

    Provides an ergonomic way to build complex constraint specifications
    with type safety and IDE autocompletion.

    Example:
        >>> spec = (
        ...     ConstraintBuilder()
        ...     .language("python")
        ...     .regex(r"^return\\s+")
        ...     .type_binding("x", "int")
        ...     .type_binding("y", "str")
        ...     .expected_type("Union[int, str]")
        ...     .forbidden_import("os")
        ...     .build()
        ... )
    """

    def __init__(self) -> None:
        """Initialize empty builder."""
        self._data: Dict[str, Any] = {}

    # === Syntax Constraints ===

    def regex(self, pattern: str) -> "ConstraintBuilder":
        """Set regex constraint pattern.

        Args:
            pattern: Regular expression pattern

        Returns:
            self for chaining
        """
        self._data["regex"] = pattern
        return self

    def negative_regex(self, pattern: str) -> "ConstraintBuilder":
        """Set negative regex pattern (output must NOT match).

        Args:
            pattern: Regular expression pattern to exclude

        Returns:
            self for chaining
        """
        self._data["negative_regex"] = pattern
        return self

    def json_schema(self, schema: Union[str, Dict[str, Any]]) -> "ConstraintBuilder":
        """Set JSON schema constraint.

        Args:
            schema: JSON schema as string or dict

        Returns:
            self for chaining
        """
        if isinstance(schema, dict):
            import json
            schema = json.dumps(schema)
        self._data["json_schema"] = schema
        return self

    def ebnf(self, grammar: str) -> "ConstraintBuilder":
        """Set EBNF grammar constraint.

        Args:
            grammar: EBNF grammar string

        Returns:
            self for chaining
        """
        self._data["ebnf"] = grammar
        return self

    def structural_tag(self, tag: str) -> "ConstraintBuilder":
        """Set structural tag constraint.

        Args:
            tag: Structural tag identifier

        Returns:
            self for chaining
        """
        self._data["structural_tag"] = tag
        return self

    # === Language Configuration ===

    def language(self, lang: str) -> "ConstraintBuilder":
        """Set target programming language.

        Args:
            lang: Language identifier (python, rust, typescript, etc.)

        Returns:
            self for chaining
        """
        self._data["language"] = lang
        return self

    def language_detection(self, mode: str) -> "ConstraintBuilder":
        """Set language detection mode.

        Args:
            mode: "auto", "explicit", or "stack"

        Returns:
            self for chaining
        """
        self._data["language_detection"] = mode
        return self

    # === Type Context ===

    def type_binding(
        self,
        name: str,
        type_expr: str,
        scope: Optional[str] = None,
        mutable: bool = True,
    ) -> "ConstraintBuilder":
        """Add a type binding to the context.

        Args:
            name: Variable name
            type_expr: Type expression (e.g., "List[int]")
            scope: Scope identifier ("local", "parameter", "global")
            mutable: Whether the binding is mutable

        Returns:
            self for chaining
        """
        if "type_bindings" not in self._data:
            self._data["type_bindings"] = []

        binding = {"name": name, "type_expr": type_expr}
        if scope:
            binding["scope"] = scope
        if not mutable:
            binding["mutable"] = mutable

        self._data["type_bindings"].append(binding)
        return self

    def expected_type(self, type_expr: str) -> "ConstraintBuilder":
        """Set the expected type of the generated expression.

        Args:
            type_expr: Expected type expression

        Returns:
            self for chaining
        """
        self._data["expected_type"] = type_expr
        return self

    def type_alias(self, name: str, definition: str) -> "ConstraintBuilder":
        """Add a type alias.

        Args:
            name: Alias name
            definition: Type definition

        Returns:
            self for chaining
        """
        if "type_aliases" not in self._data:
            self._data["type_aliases"] = {}
        self._data["type_aliases"][name] = definition
        return self

    def function_signature(
        self,
        name: str,
        params: list[tuple[str, str]],
        return_type: str,
        is_async: bool = False,
    ) -> "ConstraintBuilder":
        """Add a function signature to the context.

        Args:
            name: Function name
            params: List of (name, type) tuples
            return_type: Return type expression
            is_async: Whether function is async

        Returns:
            self for chaining
        """
        if "function_signatures" not in self._data:
            self._data["function_signatures"] = []

        sig = {
            "name": name,
            "params": [{"name": n, "type_expr": t} for n, t in params],
            "return_type": return_type,
        }
        if is_async:
            sig["is_async"] = True

        self._data["function_signatures"].append(sig)
        return self

    # === Import Context ===

    def import_binding(
        self,
        module: str,
        name: Optional[str] = None,
        alias: Optional[str] = None,
    ) -> "ConstraintBuilder":
        """Add an import binding.

        Args:
            module: Module path
            name: Imported name (for 'from x import name')
            alias: Import alias

        Returns:
            self for chaining
        """
        if "imports" not in self._data:
            self._data["imports"] = []

        binding = {"module": module}
        if name:
            binding["name"] = name
        if alias:
            binding["alias"] = alias

        self._data["imports"].append(binding)
        return self

    def available_module(self, module: str) -> "ConstraintBuilder":
        """Mark a module as available for import.

        Args:
            module: Module name

        Returns:
            self for chaining
        """
        if "available_modules" not in self._data:
            self._data["available_modules"] = []
        self._data["available_modules"].append(module)
        return self

    def forbidden_import(self, module: str) -> "ConstraintBuilder":
        """Mark a module as forbidden.

        Args:
            module: Module name to forbid

        Returns:
            self for chaining
        """
        if "forbidden_imports" not in self._data:
            self._data["forbidden_imports"] = []
        self._data["forbidden_imports"].append(module)
        return self

    # === Control Flow Context ===

    def in_function(
        self,
        name: str,
        return_type: Optional[str] = None,
        is_async: bool = False,
    ) -> "ConstraintBuilder":
        """Set control flow context as being inside a function.

        Args:
            name: Function name
            return_type: Expected return type
            is_async: Whether function is async

        Returns:
            self for chaining
        """
        if "control_flow" not in self._data:
            self._data["control_flow"] = {}

        self._data["control_flow"]["function_name"] = name
        if return_type:
            self._data["control_flow"]["expected_return_type"] = return_type
        if is_async:
            self._data["control_flow"]["in_async_context"] = True

        return self

    def in_loop(self, depth: int = 1, variables: Optional[list[str]] = None) -> "ConstraintBuilder":
        """Set control flow context as being inside a loop.

        Args:
            depth: Loop nesting depth
            variables: Loop variables

        Returns:
            self for chaining
        """
        if "control_flow" not in self._data:
            self._data["control_flow"] = {}

        self._data["control_flow"]["loop_depth"] = depth
        if variables:
            self._data["control_flow"]["loop_variables"] = variables

        return self

    def in_try_block(self, exception_types: Optional[list[str]] = None) -> "ConstraintBuilder":
        """Set control flow context as being inside a try block.

        Args:
            exception_types: Exception types being caught

        Returns:
            self for chaining
        """
        if "control_flow" not in self._data:
            self._data["control_flow"] = {}

        self._data["control_flow"]["in_try_block"] = True
        if exception_types:
            self._data["control_flow"]["exception_types"] = exception_types

        return self

    # === Semantic Constraints ===

    def precondition(self, expression: str, scope: Optional[str] = None) -> "ConstraintBuilder":
        """Add a precondition constraint.

        Args:
            expression: Boolean expression
            scope: Scope where it applies

        Returns:
            self for chaining
        """
        return self._add_semantic_constraint("precondition", expression, scope)

    def postcondition(self, expression: str, scope: Optional[str] = None) -> "ConstraintBuilder":
        """Add a postcondition constraint.

        Args:
            expression: Boolean expression
            scope: Scope where it applies

        Returns:
            self for chaining
        """
        return self._add_semantic_constraint("postcondition", expression, scope)

    def invariant(self, expression: str, scope: Optional[str] = None) -> "ConstraintBuilder":
        """Add an invariant constraint.

        Args:
            expression: Boolean expression
            scope: Scope where it applies

        Returns:
            self for chaining
        """
        return self._add_semantic_constraint("invariant", expression, scope)

    def _add_semantic_constraint(
        self,
        kind: str,
        expression: str,
        scope: Optional[str],
    ) -> "ConstraintBuilder":
        """Add a semantic constraint."""
        if "semantic_constraints" not in self._data:
            self._data["semantic_constraints"] = []

        constraint = {"kind": kind, "expression": expression}
        if scope:
            constraint["scope"] = scope

        self._data["semantic_constraints"].append(constraint)
        return self

    # === Domain Configuration ===

    def enable_domain(self, domain: str) -> "ConstraintBuilder":
        """Enable a specific domain.

        Args:
            domain: Domain name (syntax, types, imports, controlflow, semantics)

        Returns:
            self for chaining
        """
        if "enabled_domains" not in self._data:
            self._data["enabled_domains"] = []
        self._data["enabled_domains"].append(domain)
        return self

    def disable_domain(self, domain: str) -> "ConstraintBuilder":
        """Disable a specific domain.

        Args:
            domain: Domain name

        Returns:
            self for chaining
        """
        if "disabled_domains" not in self._data:
            self._data["disabled_domains"] = []
        self._data["disabled_domains"].append(domain)
        return self

    def intensity(self, level: str) -> "ConstraintBuilder":
        """Set constraint intensity level.

        Args:
            level: "none", "syntax_only", "standard", "full", "exhaustive", or "auto"

        Returns:
            self for chaining
        """
        self._data["intensity"] = level
        return self

    # === Cache Control ===

    def cache_scope(self, scope: str) -> "ConstraintBuilder":
        """Set cache scope for this constraint.

        Args:
            scope: "syntax_only", "syntax_and_lang", or "full_context"

        Returns:
            self for chaining
        """
        self._data["cache_scope"] = scope
        return self

    # === Building ===

    def build(self) -> "ConstraintSpec":
        """Build the ConstraintSpec from accumulated configuration.

        Returns:
            Configured ConstraintSpec instance
        """
        from ..spec.constraint_spec import ConstraintSpec
        return ConstraintSpec.from_dict(self._data)

    def to_dict(self) -> Dict[str, Any]:
        """Return the accumulated configuration as a dict.

        Useful when you don't want to import ConstraintSpec directly.

        Returns:
            Configuration dictionary
        """
        return self._data.copy()
