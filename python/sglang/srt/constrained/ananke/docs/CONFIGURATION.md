# Ananke Configuration Guide

This document describes all configuration options for the Ananke constrained generation system.

## Table of Contents

- [Quick Start](#quick-start)
- [ConstraintSpec](#constraintspec)
- [Backend Configuration](#backend-configuration)
- [Grammar Configuration](#grammar-configuration)
- [Domain Configuration](#domain-configuration)
- [Client SDK Configuration](#client-sdk-configuration)
- [Performance Tuning](#performance-tuning)

## Quick Start

```python
from sglang.srt.constrained.ananke.spec.constraint_spec import ConstraintSpec
from sglang.srt.constrained.ananke.backend import AnankeBackend

# Create a simple constraint
spec = ConstraintSpec(
    regex=r'\{"name": "[a-zA-Z]+", "age": \d+\}',
    language="python",
)

# Or use JSON schema
spec = ConstraintSpec(
    json_schema='{"type": "object", "properties": {"name": {"type": "string"}}}',
    language="python",
)
```

## ConstraintSpec

`ConstraintSpec` is the core configuration object for constrained generation.

### Core Syntax Constraints

Exactly one of these must be specified:

| Field | Type | Description |
|-------|------|-------------|
| `json_schema` | `str` | JSON Schema string for structured output |
| `regex` | `str` | Regular expression pattern |
| `ebnf` | `str` | EBNF grammar string |
| `structural_tag` | `str` | Structural tag identifier |

### Language Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `language` | `str` | `None` | Explicit language (python, typescript, rust, go, kotlin, swift, zig) |
| `language_detection` | `LanguageDetection` | `AUTO` | Detection strategy |
| `language_stack` | `List[LanguageFrame]` | `[]` | For polyglot generation |

**LanguageDetection Options:**
- `AUTO`: Tree-sitter based detection from generated text (default)
- `EXPLICIT`: Use the `language` field only
- `STACK`: Use `language_stack` for polyglot generation

### Type Context

| Field | Type | Description |
|-------|------|-------------|
| `type_bindings` | `List[TypeBinding]` | Variable type bindings in scope |
| `function_signatures` | `List[FunctionSignature]` | Function signatures in scope |
| `class_definitions` | `List[ClassDefinition]` | Class definitions in scope |
| `expected_type` | `str` | Expected type of generated expression |
| `type_aliases` | `Dict[str, str]` | Type alias mappings |

Example:
```python
spec = ConstraintSpec(
    regex=r'.*',
    language="python",
    type_bindings=[
        TypeBinding("x", "int"),
        TypeBinding("y", "List[str]"),
    ],
    expected_type="Dict[str, Any]",
)
```

### Import Context

| Field | Type | Description |
|-------|------|-------------|
| `imports` | `List[ImportBinding]` | Import bindings in scope |
| `available_modules` | `Set[str]` | Available module names |
| `forbidden_imports` | `Set[str]` | Forbidden module names |
| `module_stubs` | `List[ModuleStub]` | Type stubs for modules |

### Domain Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled_domains` | `Set[str]` | `None` | Domains to enable (overrides backend) |
| `disabled_domains` | `Set[str]` | `None` | Domains to disable |
| `domain_configs` | `Dict[str, Any]` | `{}` | Per-domain configuration |

Available domains: `syntax`, `types`, `imports`, `controlflow`, `semantics`

### Relaxation Settings

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `allow_relaxation` | `bool` | `True` | Enable progressive domain relaxation |
| `relaxation_threshold` | `int` | `100` | Minimum popcount before relaxation triggers |
| `relaxation_domains` | `Set[str]` | `None` | Domains that can be relaxed |

### Early Termination

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enable_early_termination` | `bool` | `True` | Stop when constraint satisfied at language boundary |
| `max_tokens` | `int` | `None` | Per-constraint token limit |

### Cache Control

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `cache_scope` | `CacheScope` | `SYNTAX_ONLY` | What to include in cache key |

**CacheScope Options:**
- `SYNTAX_ONLY`: Maximum cache reuse
- `SYNTAX_AND_LANG`: Include language in key
- `FULL_CONTEXT`: Include all context in key

## Backend Configuration

```python
backend = AnankeBackend.create(
    language="python",              # Default language
    enable_types=True,              # Enable type domain
    enable_imports=True,            # Enable import domain
    enable_controlflow=True,        # Enable control flow domain
    enable_semantics=False,         # Enable semantic domain (requires Z3)
    allow_relaxation=True,          # Enable progressive relaxation
    relaxation_threshold=100,       # Minimum popcount threshold
)
```

## Grammar Configuration

```python
grammar = AnankeGrammar(
    syntax_grammar=llguidance_grammar,  # Underlying syntax grammar
    domains={"types": type_domain},     # Domain instances
    vocab_size=32000,                   # Vocabulary size
    device="cuda",                      # Device for tensors
    language="python",                  # Target language
    max_rollback_tokens=200,            # Checkpoint history size
    checkpoint_interval=10,             # Sparse checkpointing interval
    mask_pool_size=8,                   # Pre-allocated tensor pool size
    allow_relaxation=True,              # Enable progressive relaxation
    relaxation_threshold=100,           # Minimum popcount threshold
    tiered_target_popcount=100,         # Target for tiered evaluation
)
```

### Evaluation Modes

| Mode | Description |
|------|-------------|
| `SEQUENTIAL` | Evaluate domains in order |
| `TIERED` | Tier-based evaluation with early termination on popcount |
| `ADAPTIVE` | Select evaluation strategy based on context |

## Client SDK Configuration

### RetryConfig

```python
from sglang.srt.constrained.ananke.client import AnankeClient, RetryConfig

retry_config = RetryConfig(
    max_retries=3,              # Maximum retry attempts (0 = no retries)
    initial_delay=0.5,          # Initial delay in seconds
    max_delay=30.0,             # Maximum delay in seconds
    exponential_base=2.0,       # Backoff multiplier
    jitter=0.1,                 # Random jitter fraction (0.0-1.0)
    retry_on_timeout=True,      # Retry on timeout errors
    retryable_status_codes={    # HTTP status codes to retry
        408, 429, 500, 502, 503, 504
    },
)

client = AnankeClient(
    base_url="http://localhost:8000",
    retry_config=retry_config,
)
```

### LoggingConfig

```python
from sglang.srt.constrained.ananke.client import LoggingConfig
import logging

logging_config = LoggingConfig(
    enabled=True,               # Enable request/response logging
    log_request_body=False,     # Log request body (may contain sensitive data)
    log_response_body=False,    # Log response body
    log_headers=False,          # Log headers (auth tokens redacted)
    truncate_body=500,          # Max chars to log (0 = unlimited)
    log_level=logging.DEBUG,    # Logging level
)

client = AnankeClient(
    base_url="http://localhost:8000",
    logging_config=logging_config,
)
```

### GenerationConfig

```python
from sglang.srt.constrained.ananke.client import GenerationConfig

config = GenerationConfig(
    max_tokens=256,             # Maximum tokens to generate
    temperature=0.7,            # Sampling temperature
    top_p=0.9,                  # Nucleus sampling threshold
    top_k=50,                   # Top-k sampling
    stop=["\n\n", "```"],       # Stop sequences
)

result = client.generate(
    prompt="def hello():",
    constraint=spec,
    config=config,
)
```

## Performance Tuning

### Lazy Evaluation Budget

```python
from sglang.srt.constrained.ananke.masks.lazy import EvaluationBudget

budget = EvaluationBudget(
    max_time_ns=2_000_000,    # 2ms maximum
    max_domains=5,            # Skip expensive domains if exceeded
    min_selectivity=0.95,     # Stop if mask is highly selective
)
```

### Zig Native Acceleration

When available, the Zig native library provides SIMD-accelerated mask fusion:

```python
from sglang.srt.constrained.ananke.zig.ffi import is_native_available

if is_native_available():
    print("Using SIMD-accelerated mask fusion (~50x speedup)")
```

### TokenClassifier Caching

TokenClassifier uses disk-based caching for faster warm starts:

```python
import os

# Cache directory (defaults to ~/.cache/ananke/token_classifiers)
os.environ["ANANKE_CACHE_DIR"] = "/path/to/cache"

# Disable caching
os.environ["ANANKE_DISABLE_CACHE"] = "1"
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ANANKE_CACHE_DIR` | Cache directory for token classifiers | `~/.cache/ananke` |
| `ANANKE_DISABLE_CACHE` | Disable disk caching | `0` |
| `ANANKE_LOG_LEVEL` | Logging level | `INFO` |
| `ANANKE_ZIG_LIB_PATH` | Path to Zig native library | Auto-detected |

## See Also

- [README.md](../README.md) - Overview and quick start
- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture
- [CONSTRAINT_SPEC.md](CONSTRAINT_SPEC.md) - Constraint specification format
