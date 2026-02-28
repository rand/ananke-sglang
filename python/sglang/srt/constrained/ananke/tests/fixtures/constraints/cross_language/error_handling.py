# Copyright Rand Arete @ Ananke 2025
# Licensed under the Apache License, Version 2.0
"""Cross-language error handling examples.

Demonstrates idiomatic error handling patterns across all 7 languages,
showing how each language's error model affects constraint specifications.
"""

from __future__ import annotations

from typing import List

from ..base import ConstraintExample

# Support both relative imports (when used as subpackage) and absolute imports
try:
    from .....spec.constraint_spec import (
        ConstraintSpec,
        ControlFlowContext,
        FunctionSignature,
        SemanticConstraint,
        TypeBinding,
    )
except ImportError:
    from spec.constraint_spec import (
        ConstraintSpec,
        ControlFlowContext,
        FunctionSignature,
        SemanticConstraint,
        TypeBinding,
    )


CROSS_LANGUAGE_ERROR_EXAMPLES: List[ConstraintExample] = [
    # Python: Exception handling with context managers
    ConstraintExample(
        id="cross-error-python",
        name="Python Resource Error Handler",
        description="Handle file operations with context managers and exception chaining",
        scenario="Developer writing a file processor with proper resource cleanup",
        prompt="""Complete this Python function that reads a config file safely:

def read_config(path: Path) -> Optional[str]:
    with open(path) as f:
        """,
        spec=ConstraintSpec(
            language="python",
            # Regex enforces return f.read() pattern inside context manager
            regex=r"^return\s+f\.read\(\)",
            expected_type="Optional[str]",
            type_bindings=[
                TypeBinding(name="path", type_expr="Path", scope="parameter"),
            ],
            control_flow=ControlFlowContext(
                function_name="read_config",
                expected_return_type="Optional[str]",
                in_try_block=True,
                exception_types=("FileNotFoundError", "PermissionError", "json.JSONDecodeError"),
            ),
            semantic_constraints=[
                SemanticConstraint(
                    kind="postcondition",
                    expression="result is None or isinstance(result, str)",
                    scope="read_config",
                ),
            ],
        ),
        expected_effect="Ensures proper exception handling and resource cleanup",
        valid_outputs=[
            "return f.read()",
            "return f.read().strip()",
        ],
        invalid_outputs=[
            "return open(path).read()",  # no context manager
            "f = open(path); return f.read()",  # no cleanup
        ],
        tags=["error-handling", "context-manager", "exceptions", "cross-language"],
        language="python",
        domain="controlflow",
    ),

    # Rust: Result<T, E> with ? operator
    ConstraintExample(
        id="cross-error-rust",
        name="Rust Result Error Handler",
        description="Propagate errors with ? operator and custom error types",
        scenario="Developer writing a config loader with proper error propagation",
        prompt="""Complete this Rust function that reads a config file. Use the ? operator for error propagation:

fn read_config(path: &Path) -> Result<Config, ConfigError> {
    """,
        spec=ConstraintSpec(
            language="rust",
            # Regex enforces ? operator or Result combinators, blocks unwrap/expect
            regex=r"^(?:let\s+\w+\s*=\s*.*\?|fs::\w+\([^)]+\)\.(?:map_err|and_then))",
            expected_type="Result<Config, ConfigError>",
            type_bindings=[
                TypeBinding(name="path", type_expr="&Path", scope="parameter"),
            ],
            control_flow=ControlFlowContext(
                function_name="read_config",
                expected_return_type="Result<Config, ConfigError>",
            ),
            semantic_constraints=[
                SemanticConstraint(
                    kind="invariant",
                    expression="!contains_call('unwrap') && !contains_call('expect')",
                    scope="read_config",
                    variables=(),
                ),
            ],
            forbidden_imports={"std::panic"},
        ),
        expected_effect="Ensures ? operator usage, no unwrap/expect, proper From impls",
        valid_outputs=[
            "let content = fs::read_to_string(path)?; let config: Config = serde_json::from_str(&content)?; Ok(config)",
            "fs::read_to_string(path).map_err(ConfigError::Io).and_then(|s| serde_json::from_str(&s).map_err(ConfigError::Parse))",
        ],
        invalid_outputs=[
            "fs::read_to_string(path).unwrap()",  # unwrap
            "fs::read_to_string(path).expect(\"file\")",  # expect
        ],
        tags=["error-handling", "result", "question-mark", "cross-language"],
        language="rust",
        domain="controlflow",
    ),

    # TypeScript: Discriminated union errors
    ConstraintExample(
        id="cross-error-typescript",
        name="TypeScript Result Type Handler",
        description="Handle errors with discriminated union Result type pattern",
        scenario="Developer implementing a typed Result pattern for type-safe error handling",
        spec=ConstraintSpec(
            language="typescript",
            # Regex enforces try/catch with Result discriminator return
            regex=r"try\s*\{.*return\s*\{\s*ok:\s*true",
            expected_type="Result<Config, ConfigError>",
            type_bindings=[
                TypeBinding(name="path", type_expr="string", scope="parameter"),
            ],
            function_signatures=[
                FunctionSignature(
                    name="readConfig",
                    params=(TypeBinding(name="path", type_expr="string"),),
                    return_type="Promise<Result<Config, ConfigError>>",
                    is_async=True,
                )
            ],
        ),
        expected_effect="Ensures proper Result type usage with ok/err discriminator",
        valid_outputs=[
            "try { const content = await fs.readFile(path, 'utf-8'); return { ok: true, value: JSON.parse(content) }; } catch (e) { return { ok: false, error: new ConfigError(e) }; }",
        ],
        invalid_outputs=[
            "return JSON.parse(await fs.readFile(path, 'utf-8'))",  # throws
            "return { value: config }",  # missing discriminator
        ],
        tags=["error-handling", "result-type", "discriminated-union", "cross-language"],
        language="typescript",
        domain="controlflow",
        max_tokens=4096,  # TypeScript try/catch with Result type is verbose
    ),

    # Go: Multi-return error pattern
    ConstraintExample(
        id="cross-error-go",
        name="Go Error Return Handler",
        description="Handle errors with Go's idiomatic (value, error) return pattern",
        scenario="Developer writing a config loader following Go error conventions",
        spec=ConstraintSpec(
            language="go",
            # Regex enforces (value, err :=) pattern with error check
            regex=r"^(?:\w+,\s*err\s*:=.*;\s*if\s+err\s*!=\s*nil)",
            expected_type="(*Config, error)",
            type_bindings=[
                TypeBinding(name="path", type_expr="string", scope="parameter"),
            ],
            function_signatures=[
                FunctionSignature(
                    name="ReadConfig",
                    params=(TypeBinding(name="path", type_expr="string"),),
                    return_type="(*Config, error)",
                )
            ],
            semantic_constraints=[
                SemanticConstraint(
                    kind="invariant",
                    expression="all_errors_checked_immediately()",
                    scope="ReadConfig",
                    variables=(),
                ),
            ],
        ),
        expected_effect="Ensures if err != nil pattern and proper error wrapping",
        valid_outputs=[
            'data, err := os.ReadFile(path); if err != nil { return nil, fmt.Errorf("reading config: %w", err) }; var cfg Config; if err := json.Unmarshal(data, &cfg); err != nil { return nil, err }; return &cfg, nil',
        ],
        invalid_outputs=[
            "data, _ := os.ReadFile(path)",  # ignored error
            "return &cfg",  # missing error return
        ],
        tags=["error-handling", "multi-return", "error-wrapping", "cross-language"],
        language="go",
        domain="controlflow",
    ),

    # Zig: Error union with errdefer
    ConstraintExample(
        id="cross-error-zig",
        name="Zig Error Union Handler",
        description="Handle errors with error unions and errdefer cleanup",
        scenario="Developer writing a config loader with proper resource cleanup on error",
        spec=ConstraintSpec(
            language="zig",
            # Regex enforces try with errdefer pattern
            regex=r"^const\s+\w+\s*=\s*try\s+.*;\s*errdefer",
            expected_type="ConfigError!Config",
            type_bindings=[
                TypeBinding(name="allocator", type_expr="std.mem.Allocator", scope="parameter"),
                TypeBinding(name="path", type_expr="[]const u8", scope="parameter"),
            ],
            control_flow=ControlFlowContext(
                function_name="readConfig",
                expected_return_type="ConfigError!Config",
            ),
        ),
        expected_effect="Ensures try/catch usage, errdefer for cleanup, proper error propagation",
        valid_outputs=[
            "const file = try std.fs.cwd().openFile(path, .{}); errdefer file.close(); const content = try file.readToEndAlloc(allocator, max_size); return parseConfig(content);",
        ],
        invalid_outputs=[
            "std.fs.cwd().openFile(path, .{}).file",  # no try
            "const file = try std.fs.cwd().openFile(path, .{});",  # no errdefer
        ],
        tags=["error-handling", "error-union", "errdefer", "cross-language"],
        language="zig",
        domain="controlflow",
        max_tokens=4096,  # Zig error handling with comments is very verbose
    ),

    # Kotlin: Sealed class Result with runCatching
    ConstraintExample(
        id="cross-error-kotlin",
        name="Kotlin Result Error Handler",
        description="Handle errors with Kotlin's Result type and runCatching",
        scenario="Developer writing a config loader with functional error handling",
        spec=ConstraintSpec(
            language="kotlin",
            # Regex enforces runCatching or Result.success/failure pattern
            regex=r"^return\s+(?:runCatching|Result\.(?:success|failure))",
            expected_type="Result<Config>",
            type_bindings=[
                TypeBinding(name="path", type_expr="Path", scope="parameter"),
            ],
            function_signatures=[
                FunctionSignature(
                    name="readConfig",
                    params=(TypeBinding(name="path", type_expr="Path"),),
                    return_type="Result<Config>",
                )
            ],
        ),
        expected_effect="Ensures Result type usage with map/recover chains",
        valid_outputs=[
            "return runCatching { path.readText() }.mapCatching { Json.decodeFromString<Config>(it) }",
            "return runCatching { Json.decodeFromString<Config>(path.readText()) }",
        ],
        invalid_outputs=[
            "return Config(path.readText())",  # throws
            "try { return Result.success(parse(path)) } catch (e: Exception) { throw e }",  # rethrows
        ],
        tags=["error-handling", "result-type", "runCatching", "cross-language"],
        language="kotlin",
        domain="controlflow",
    ),

    # Swift: throws with Result type
    ConstraintExample(
        id="cross-error-swift",
        name="Swift Result Error Handler",
        description="Handle errors with Swift's Result type and throwing functions",
        scenario="Developer writing a config loader with proper error propagation",
        prompt="""Complete this Swift function that reads a config file. Return a Result type:

func readConfig(path: URL) -> Result<Config, ConfigError> {
    """,
        spec=ConstraintSpec(
            language="swift",
            # Regex enforces Result pattern with mapError/flatMap or do/catch
            regex=r"^(?:return\s+Result\s*\{|do\s*\{\s*let\s+\w+\s*=\s*try)",
            expected_type="Result<Config, ConfigError>",
            type_bindings=[
                TypeBinding(name="path", type_expr="URL", scope="parameter"),
            ],
            function_signatures=[
                FunctionSignature(
                    name="readConfig",
                    params=(TypeBinding(name="path", type_expr="URL"),),
                    return_type="Result<Config, ConfigError>",
                )
            ],
        ),
        expected_effect="Ensures Result type with mapError and flatMap",
        valid_outputs=[
            "return Result { try Data(contentsOf: path) }.flatMap { data in Result { try JSONDecoder().decode(Config.self, from: data) } }.mapError { ConfigError.underlying($0) }",
            "do { let data = try Data(contentsOf: path); let config = try JSONDecoder().decode(Config.self, from: data); return .success(config) } catch { return .failure(.underlying(error)) }",
        ],
        invalid_outputs=[
            "return .success(try JSONDecoder().decode(Config.self, from: Data(contentsOf: path)))",  # try in success
            "try! Data(contentsOf: path)",  # force try
        ],
        tags=["error-handling", "result-type", "throwing", "cross-language"],
        language="swift",
        domain="controlflow",
    ),
]
