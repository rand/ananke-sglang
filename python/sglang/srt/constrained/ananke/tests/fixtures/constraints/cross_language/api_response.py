# Copyright Rand Arete @ Ananke 2025
# Licensed under the Apache License, Version 2.0
"""Cross-language API response handling examples.

Demonstrates the same API response handling workflow across all 7 languages,
showing how type constraints, JSON schema, and error handling adapt to
each language's idioms.
"""

from __future__ import annotations

from typing import List

from ..base import ConstraintExample

# Support both relative imports (when used as subpackage) and absolute imports
try:
    from .....spec.constraint_spec import (
        ConstraintSpec,
        TypeBinding,
        FunctionSignature,
        SemanticConstraint,
    )
except ImportError:
    from spec.constraint_spec import (
        ConstraintSpec,
        TypeBinding,
        FunctionSignature,
        SemanticConstraint,
    )


# Common JSON schema for API response across all languages
API_RESPONSE_SCHEMA = """{
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["success", "error"]},
        "data": {"type": "object"},
        "error": {
            "type": "object",
            "properties": {
                "code": {"type": "integer"},
                "message": {"type": "string"}
            }
        },
        "timestamp": {"type": "string", "format": "date-time"}
    },
    "required": ["status", "timestamp"],
    "if": {"properties": {"status": {"const": "success"}}},
    "then": {"required": ["data"]},
    "else": {"required": ["error"]}
}"""


CROSS_LANGUAGE_API_EXAMPLES: List[ConstraintExample] = [
    # Python: Pydantic-style response handling
    ConstraintExample(
        id="cross-api-python",
        name="Python API Response Handler",
        description="Parse and validate API response with Pydantic model and type hints",
        scenario="Developer writing an async API client that deserializes JSON responses into typed models",
        prompt="""Parse an HTTP response into a typed ApiResponse model. The response must have status ("success"/"error"),
timestamp, and either data (on success) or error object (on failure).

async def parse_response(response: httpx.Response, model_cls: Type[T]) -> ApiResponse[T]:
    """,
        spec=ConstraintSpec(
            language="python",
            json_schema=API_RESPONSE_SCHEMA,
            expected_type="ApiResponse[T]",
            type_bindings=[
                TypeBinding(name="response", type_expr="httpx.Response", scope="parameter"),
                TypeBinding(name="model_cls", type_expr="Type[T]", scope="parameter"),
            ],
            function_signatures=[
                FunctionSignature(
                    name="parse_response",
                    params=(
                        TypeBinding(name="response", type_expr="httpx.Response"),
                        TypeBinding(name="model_cls", type_expr="Type[T]"),
                    ),
                    return_type="ApiResponse[T]",
                    type_params=("T",),
                    is_async=True,
                )
            ],
            semantic_constraints=[
                SemanticConstraint(
                    kind="postcondition",
                    expression="result.status in ('success', 'error')",
                    scope="parse_response",
                ),
            ],
        ),
        expected_effect="Ensures response conforms to ApiResponse schema with proper status field",
        valid_outputs=[
            'return ApiResponse(status="success", data=model_cls(**json_data), timestamp=datetime.now())',
            "return ApiResponse.model_validate(response.json())",
        ],
        invalid_outputs=[
            'return {"status": "ok"}',  # wrong status enum
            "return json_data",  # not wrapped in ApiResponse
        ],
        tags=["api", "json", "async", "generics", "cross-language"],
        language="python",
        domain="syntax",
    ),

    # Rust: serde-based response handling
    ConstraintExample(
        id="cross-api-rust",
        name="Rust API Response Handler",
        description="Deserialize API response with serde and Result error handling",
        scenario="Developer writing an async HTTP client with typed JSON deserialization",
        prompt="""Deserialize an HTTP response into ApiResponse<T> using serde. Use the ? operator for error propagation.
Return Result with proper error handling - no unwrap().

async fn parse_response<T: DeserializeOwned>(response: reqwest::Response) -> Result<ApiResponse<T>, ApiError> {
    """,
        spec=ConstraintSpec(
            language="rust",
            json_schema=API_RESPONSE_SCHEMA,
            expected_type="Result<ApiResponse<T>, ApiError>",
            type_bindings=[
                TypeBinding(name="response", type_expr="reqwest::Response", scope="parameter"),
            ],
            function_signatures=[
                FunctionSignature(
                    name="parse_response",
                    params=(TypeBinding(name="response", type_expr="reqwest::Response"),),
                    return_type="Result<ApiResponse<T>, ApiError>",
                    type_params=("T: DeserializeOwned",),
                    is_async=True,
                )
            ],
            semantic_constraints=[
                SemanticConstraint(
                    kind="postcondition",
                    expression="result.is_ok() implies result.unwrap().status.is_valid()",
                    scope="parse_response",
                ),
            ],
        ),
        expected_effect="Ensures proper Result wrapping and serde deserialization",
        valid_outputs=[
            "let data: ApiResponse<T> = response.json().await?; Ok(data)",
            "response.json::<ApiResponse<T>>().await.map_err(ApiError::from)",
        ],
        invalid_outputs=[
            "response.json().await.unwrap()",  # no error handling
            "Ok(response.text().await?)",  # wrong return type
        ],
        tags=["api", "json", "async", "result", "cross-language"],
        language="rust",
        domain="syntax",
    ),

    # TypeScript: fetch with type guards
    ConstraintExample(
        id="cross-api-typescript",
        name="TypeScript API Response Handler",
        description="Fetch API response with type guards and discriminated unions",
        scenario="Developer writing a typed fetch wrapper with proper error discrimination",
        prompt="""Parse a fetch Response into typed ApiResponse<T>. Ensure the return type is properly typed
and the response conforms to the API schema with status discrimination.

async function parseResponse<T>(response: Response): Promise<ApiResponse<T>> {
    """,
        spec=ConstraintSpec(
            language="typescript",
            json_schema=API_RESPONSE_SCHEMA,
            expected_type="Promise<ApiResponse<T>>",
            type_bindings=[
                TypeBinding(name="response", type_expr="Response", scope="parameter"),
            ],
            function_signatures=[
                FunctionSignature(
                    name="parseResponse",
                    params=(TypeBinding(name="response", type_expr="Response"),),
                    return_type="Promise<ApiResponse<T>>",
                    type_params=("T",),
                    is_async=True,
                )
            ],
        ),
        expected_effect="Ensures proper async/await and type narrowing for response status",
        valid_outputs=[
            "const json = await response.json(); return validateApiResponse<T>(json);",
            "return response.json() as Promise<ApiResponse<T>>;",
        ],
        invalid_outputs=[
            "return response.json();",  # untyped
            "return { status: 'ok' };",  # wrong schema
        ],
        tags=["api", "json", "async", "generics", "cross-language"],
        language="typescript",
        domain="syntax",
    ),

    # Go: struct unmarshaling with error handling
    ConstraintExample(
        id="cross-api-go",
        name="Go API Response Handler",
        description="Unmarshal JSON response with proper error handling idioms",
        scenario="Developer writing an HTTP client with json.Unmarshal and error checks",
        prompt="""Parse an HTTP response body into ApiResponse[T]. Use defer for body cleanup.
Check errors immediately with if err != nil pattern.

func ParseResponse[T any](resp *http.Response) (*ApiResponse[T], error) {
    """,
        spec=ConstraintSpec(
            language="go",
            json_schema=API_RESPONSE_SCHEMA,
            expected_type="(*ApiResponse[T], error)",
            type_bindings=[
                TypeBinding(name="resp", type_expr="*http.Response", scope="parameter"),
            ],
            function_signatures=[
                FunctionSignature(
                    name="ParseResponse",
                    params=(TypeBinding(name="resp", type_expr="*http.Response"),),
                    return_type="(*ApiResponse[T], error)",
                    type_params=("T",),
                )
            ],
            semantic_constraints=[
                SemanticConstraint(
                    kind="postcondition",
                    expression="err == nil implies result.Status != ''",
                    scope="ParseResponse",
                ),
            ],
        ),
        expected_effect="Ensures proper error return pattern and defer body.Close()",
        valid_outputs=[
            "defer resp.Body.Close(); var result ApiResponse[T]; if err := json.NewDecoder(resp.Body).Decode(&result); err != nil { return nil, err }; return &result, nil",
        ],
        invalid_outputs=[
            "json.Unmarshal(body, &result)",  # no error check
            "return result",  # missing error return
        ],
        tags=["api", "json", "error-handling", "cross-language"],
        language="go",
        domain="syntax",
    ),

    # Zig: JSON parsing with error unions
    ConstraintExample(
        id="cross-api-zig",
        name="Zig API Response Handler",
        description="Parse JSON response with error unions and allocator management",
        scenario="Developer writing an HTTP client with std.json parsing",
        prompt="""Parse JSON body into ApiResponse(T) using std.json. Use try for error propagation
and pass the allocator explicitly (don't use global allocators).

fn parseResponse(comptime T: type, allocator: std.mem.Allocator, body: []const u8) !ApiResponse(T) {
    """,
        spec=ConstraintSpec(
            language="zig",
            json_schema=API_RESPONSE_SCHEMA,
            expected_type="!ApiResponse(T)",
            type_bindings=[
                TypeBinding(name="allocator", type_expr="std.mem.Allocator", scope="parameter"),
                TypeBinding(name="body", type_expr="[]const u8", scope="parameter"),
            ],
            function_signatures=[
                FunctionSignature(
                    name="parseResponse",
                    params=(
                        TypeBinding(name="allocator", type_expr="std.mem.Allocator"),
                        TypeBinding(name="body", type_expr="[]const u8"),
                    ),
                    return_type="!ApiResponse(T)",
                    type_params=("T",),
                )
            ],
        ),
        expected_effect="Ensures proper error union handling and allocator usage",
        valid_outputs=[
            "const parsed = try std.json.parseFromSlice(ApiResponse(T), allocator, body, .{}); return parsed.value;",
        ],
        invalid_outputs=[
            "std.json.parseFromSlice(ApiResponse(T), allocator, body, .{}).value",  # no try
        ],
        tags=["api", "json", "error-union", "allocator", "cross-language"],
        language="zig",
        domain="syntax",
    ),

    # Kotlin: kotlinx.serialization with coroutines
    ConstraintExample(
        id="cross-api-kotlin",
        name="Kotlin API Response Handler",
        description="Parse JSON response with kotlinx.serialization in suspend context",
        scenario="Developer writing a Ktor client with typed JSON deserialization",
        prompt="""Parse an HTTP response into ApiResponse<T> using kotlinx.serialization.
This is a suspend function - use proper coroutine patterns.

suspend inline fun <reified T> parseResponse(response: HttpResponse): ApiResponse<T> {
    """,
        spec=ConstraintSpec(
            language="kotlin",
            json_schema=API_RESPONSE_SCHEMA,
            expected_type="ApiResponse<T>",
            type_bindings=[
                TypeBinding(name="response", type_expr="HttpResponse", scope="parameter"),
            ],
            function_signatures=[
                FunctionSignature(
                    name="parseResponse",
                    params=(TypeBinding(name="response", type_expr="HttpResponse"),),
                    return_type="ApiResponse<T>",
                    type_params=("T",),
                    is_async=True,  # suspend function
                )
            ],
        ),
        expected_effect="Ensures suspend function usage and proper serialization",
        valid_outputs=[
            "return response.body<ApiResponse<T>>()",
            "val json = response.bodyAsText(); return Json.decodeFromString<ApiResponse<T>>(json)",
        ],
        invalid_outputs=[
            "return response.body()",  # untyped
            "Json.decodeFromString(json)",  # no type parameter
        ],
        tags=["api", "json", "coroutines", "serialization", "cross-language"],
        language="kotlin",
        domain="syntax",
    ),

    # Swift: Codable with async/await
    ConstraintExample(
        id="cross-api-swift",
        name="Swift API Response Handler",
        description="Decode JSON response with Codable protocol in async context",
        scenario="Developer writing a URLSession-based client with typed decoding",
        prompt="""Decode JSON data into ApiResponse<T> using JSONDecoder. T must conform to Decodable.
Use try for error handling - this is a throwing function.

func parseResponse<T: Decodable>(data: Data) throws -> ApiResponse<T> {
    """,
        spec=ConstraintSpec(
            language="swift",
            json_schema=API_RESPONSE_SCHEMA,
            expected_type="ApiResponse<T>",
            type_bindings=[
                TypeBinding(name="data", type_expr="Data", scope="parameter"),
            ],
            function_signatures=[
                FunctionSignature(
                    name="parseResponse",
                    params=(TypeBinding(name="data", type_expr="Data"),),
                    return_type="ApiResponse<T>",
                    type_params=("T: Decodable",),
                    is_async=True,
                )
            ],
        ),
        expected_effect="Ensures proper Codable usage and throwing function pattern",
        valid_outputs=[
            "let decoder = JSONDecoder(); return try decoder.decode(ApiResponse<T>.self, from: data)",
            "return try JSONDecoder().decode(ApiResponse<T>.self, from: data)",
        ],
        invalid_outputs=[
            "JSONDecoder().decode(ApiResponse.self, from: data)",  # missing try
            "return data as? ApiResponse<T>",  # wrong casting
        ],
        tags=["api", "json", "async", "codable", "cross-language"],
        language="swift",
        domain="syntax",
    ),
]
