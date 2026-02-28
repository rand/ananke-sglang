# Migration Guide: XGrammar to Ananke

This guide helps you migrate from XGrammar to Ananke for constrained decoding in SGLang.

## Overview

Ananke is a multi-domain constrained decoding system that extends XGrammar's syntax-only
constraints with additional domains (types, imports, control flow, semantics). It provides
a superset of XGrammar's functionality while adding:

- Type-aware code generation
- Import validation and module resolution
- Control flow analysis
- Progressive constraint relaxation
- Early termination optimization

## Quick Comparison

| Feature | XGrammar | Ananke |
|---------|----------|--------|
| JSON Schema | ✅ | ✅ |
| Regex | ✅ | ✅ |
| EBNF | ✅ | ✅ |
| Type constraints | ❌ | ✅ |
| Import validation | ❌ | ✅ |
| Control flow | ❌ | ✅ |
| Multi-language | ❌ | ✅ (7 languages) |
| Progressive relaxation | ❌ | ✅ |
| Early termination | ❌ | ✅ |

## API Changes

### Backend Selection

**XGrammar:**
```python
# In SGLang server configuration
--constrained-json-backend xgrammar
```

**Ananke:**
```python
# In SGLang server configuration
--constrained-json-backend ananke

# Or with full domain support
--constrained-json-backend ananke --ananke-enable-types --ananke-enable-imports
```

### Constraint Specification

**XGrammar (legacy):**
```python
# Using json_schema parameter
response = requests.post(
    "http://localhost:8000/v1/completions",
    json={
        "prompt": "Generate a person:",
        "json_schema": '{"type": "object", "properties": {"name": {"type": "string"}}}',
    }
)

# Using regex
response = requests.post(
    "http://localhost:8000/v1/completions",
    json={
        "prompt": "Generate a number:",
        "regex": r"\d+",
    }
)
```

**Ananke (new):**
```python
# Using constraint_spec (recommended)
response = requests.post(
    "http://localhost:8000/v1/completions",
    json={
        "prompt": "Generate a person:",
        "constraint_spec": {
            "json_schema": '{"type": "object", "properties": {"name": {"type": "string"}}}',
            "language": "python",
        }
    }
)

# Legacy parameters still work (backward compatible)
response = requests.post(
    "http://localhost:8000/v1/completions",
    json={
        "prompt": "Generate a number:",
        "regex": r"\d+",  # Still supported
    }
)
```

### Python SDK

**XGrammar:**
```python
import sglang as sgl

@sgl.function
def generate_json(s):
    s += sgl.gen("output", json_schema=schema)
```

**Ananke:**
```python
import sglang as sgl
from sglang.srt.constrained.ananke.spec import ConstraintSpec

# Option 1: Using ConstraintSpec (recommended for rich constraints)
@sgl.function
def generate_code(s):
    spec = ConstraintSpec(
        json_schema=schema,
        language="python",
        type_bindings=[TypeBinding("x", "int")],
    )
    s += sgl.gen("output", constraint_spec=spec.to_dict())

# Option 2: Using legacy parameters (simpler, backward compatible)
@sgl.function
def generate_json(s):
    s += sgl.gen("output", json_schema=schema)  # Still works!
```

### Client SDK

**XGrammar:**
No dedicated client SDK.

**Ananke:**
```python
from sglang.srt.constrained.ananke.client import (
    AnankeClient,
    GenerationConfig,
    RetryConfig,
)

client = AnankeClient(
    base_url="http://localhost:8000",
    retry_config=RetryConfig(max_retries=3),
)

# Using ConstraintSpec
result = client.generate(
    prompt="def hello():",
    constraint=ConstraintSpec(
        regex=r'\s+return ".*"',
        language="python",
    ),
)

# Using simple regex (backward compatible)
result = client.generate(
    prompt="Generate a number:",
    constraint=r"\d+",  # String interpreted as regex
)
```

## Feature Migration

### JSON Schema Constraints

**No changes required.** Ananke is fully backward compatible:

```python
# Both work identically
json_schema = '{"type": "object", "properties": {"name": {"type": "string"}}}'

# XGrammar style
response = client.post("/v1/completions", json={"json_schema": json_schema})

# Ananke style (equivalent)
response = client.post("/v1/completions", json={
    "constraint_spec": {"json_schema": json_schema}
})
```

### Regex Constraints

**No changes required.** Same behavior:

```python
# XGrammar style
response = client.post("/v1/completions", json={"regex": r"\d{3}-\d{4}"})

# Ananke style (equivalent)
response = client.post("/v1/completions", json={
    "constraint_spec": {"regex": r"\d{3}-\d{4}"}
})
```

### EBNF Grammar

**No changes required:**

```python
ebnf = """
root ::= "hello" " " name
name ::= [a-zA-Z]+
"""

response = client.post("/v1/completions", json={"ebnf": ebnf})
```

## New Features in Ananke

### Type-Aware Generation

```python
from sglang.srt.constrained.ananke.spec import ConstraintSpec, TypeBinding

spec = ConstraintSpec(
    regex=r'.*',  # Allow any syntax
    language="python",
    type_bindings=[
        TypeBinding("x", "int"),
        TypeBinding("y", "List[str]"),
    ],
    expected_type="Dict[str, int]",
)
```

### Import Validation

```python
from sglang.srt.constrained.ananke.spec import ConstraintSpec, ImportBinding

spec = ConstraintSpec(
    regex=r'.*',
    language="python",
    imports=[
        ImportBinding("json", "json"),
        ImportBinding("List", "typing", "List"),
    ],
    forbidden_imports={"os", "subprocess", "eval"},
)
```

### Progressive Relaxation

If constraints are too strict, Ananke can progressively relax them:

```python
spec = ConstraintSpec(
    regex=r'.*',
    language="python",
    allow_relaxation=True,
    relaxation_threshold=100,  # Minimum viable tokens
)
```

### Early Termination

Stop generation when the constraint is satisfied:

```python
spec = ConstraintSpec(
    regex=r'return \d+',  # Match "return" followed by a number
    language="python",
    enable_early_termination=True,  # Stop when regex matches at statement boundary
)
```

## Configuration Differences

### Server Configuration

**XGrammar:**
```bash
python -m sglang.launch_server \
    --model-path meta-llama/Llama-2-7b \
    --constrained-json-backend xgrammar
```

**Ananke (minimal):**
```bash
python -m sglang.launch_server \
    --model-path meta-llama/Llama-2-7b \
    --constrained-json-backend ananke
```

**Ananke (full features):**
```bash
python -m sglang.launch_server \
    --model-path meta-llama/Llama-2-7b \
    --constrained-json-backend ananke \
    --ananke-enable-types \
    --ananke-enable-imports \
    --ananke-enable-controlflow
```

## Performance Considerations

### Cold Start

Ananke has slightly longer cold start due to domain initialization, but this is
amortized over requests. Use caching to minimize impact:

```python
spec = ConstraintSpec(
    json_schema=schema,
    cache_scope=CacheScope.SYNTAX_ONLY,  # Maximum cache reuse
)
```

### Throughput

For syntax-only constraints (JSON Schema, regex), Ananke performs similarly to XGrammar.
When using additional domains (types, imports), there's modest overhead:

| Scenario | XGrammar | Ananke |
|----------|----------|--------|
| JSON Schema only | 1.0x | 1.0x |
| + Type constraints | N/A | 1.1x |
| + Import validation | N/A | 1.15x |
| + All domains | N/A | 1.25x |

### Memory

Ananke uses slightly more memory for domain state:
- Base: ~50MB additional for token classifiers
- Per domain: ~10-20MB

## Troubleshooting

### "Unknown backend: ananke"

Ensure you're using SGLang version with Ananke support:
```bash
pip install sglang[ananke]
```

### Legacy parameters not working

Check that the backend is set to `ananke`:
```python
# In your request, verify backend selection
print(response.json().get("backend"))  # Should be "ananke"
```

### Type constraints not applied

Ensure types domain is enabled:
```bash
--ananke-enable-types
```

Or in ConstraintSpec:
```python
spec = ConstraintSpec(
    enabled_domains={"syntax", "types"},
    ...
)
```

## Getting Help

- [Configuration Guide](CONFIGURATION.md)
- [Architecture Overview](ARCHITECTURE.md)
- [GitHub Issues](https://github.com/rand/ananke-sglang/issues)
