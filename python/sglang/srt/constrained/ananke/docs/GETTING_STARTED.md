# Getting Started with Ananke Constrained Generation

This guide walks you through using the Ananke multi-domain constraint system with SGLang for type-safe, semantically-guided code generation.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Server Configuration](#server-configuration)
4. [Making Requests](#making-requests)
5. [Constraint Types](#constraint-types)
6. [Configuration Options](#configuration-options)
7. [Performance Tuning](#performance-tuning)
8. [Troubleshooting](#troubleshooting)

---

## Overview

Ananke extends SGLang's constrained generation with multi-domain constraint checking:

```
Request with constraint (JSON schema, regex, EBNF)
    ↓
┌─────────────────────────────────────────────────────────────┐
│                     AnankeBackend                           │
│  ┌──────────────────┐    ┌────────────────────────────────┐ │
│  │ Syntax Domain    │    │    Semantic Domains            │ │
│  │ (llguidance)     │    │  ┌──────┐ ┌───────┐ ┌────────┐ │ │
│  │  • CFG parsing   │    │  │Types │ │Imports│ │Control │ │ │
│  │  • ~50μs/token   │    │  │      │ │       │ │ Flow   │ │ │
│  └────────┬─────────┘    │  └──┬───┘ └───┬───┘ └───┬────┘ │ │
│           │              │     └─────────┴─────────┘      │ │
│           └──────────────┼──────────────┬─────────────────┘ │
│                          │              ▼                   │
│                          │    Mask Fusion (SIMD)            │
└──────────────────────────┼──────────────────────────────────┘
                           ↓
                   Valid Token Mask
```

**Key Benefits:**
- **Type Safety**: Bidirectional type inference prevents type errors
- **Import Awareness**: Tracks module dependencies and availability
- **Control Flow Analysis**: Ensures reachability and termination
- **Composable Constraints**: Domains combine via semilattice algebra

---

## Quick Start

### 1. Start the SGLang Server with Ananke

```bash
python -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --grammar-backend ananke \
    --port 30000
```

### 2. Make a Constrained Generation Request

```python
import requests

response = requests.post(
    "http://localhost:30000/generate",
    json={
        "text": "Write a Python function that adds two numbers:\n```python\n",
        "sampling_params": {
            "max_new_tokens": 256,
            "temperature": 0,
            "json_schema": None,  # Or use regex/ebnf below
            "regex": r"def add\(\w+: int, \w+: int\) -> int:\n    return \w+ \+ \w+",
        }
    }
)

print(response.json()["text"])
```

### 3. Using the Python Client

```python
import sglang as sgl

# Connect to server
sgl.set_default_backend(sgl.RuntimeEndpoint("http://localhost:30000"))

@sgl.function
def generate_function(s, prompt):
    s += prompt
    s += sgl.gen(
        "code",
        max_tokens=256,
        regex=r"def \w+\([^)]*\):[^\n]*\n(    [^\n]+\n)*",
    )

result = generate_function.run(prompt="# Function to calculate factorial\n")
print(result["code"])
```

---

## Server Configuration

### Command-Line Arguments

```bash
python -m sglang.launch_server \
    --model-path <MODEL_PATH> \
    --grammar-backend ananke \
    # Ananke-specific options (if exposed):
    # --ananke-language python          # Target language
    # --ananke-max-rollback-tokens 200  # Rollback history size
    # JSON whitespace options:
    --constrained-json-whitespace-pattern "[ \\t\\n]*"
```

### Available Grammar Backends

| Backend | Description | Best For |
|---------|-------------|----------|
| `ananke` | Multi-domain constraint checking | Code generation with type safety |
| `llguidance` | Fast CFG-based masking | General structured output |
| `xgrammar` | Efficient EBNF compilation | Complex grammars |
| `outlines` | Regex/JSON schema | Simple patterns |
| `none` | Disable constraints | Unconstrained generation |

### Programmatic Backend Creation

```python
from sglang.srt.constrained.ananke.backend.backend import AnankeBackend

backend = AnankeBackend(
    tokenizer=tokenizer,
    vocab_size=32000,
    language="python",                    # Target programming language
    enabled_domains={"types", "imports"}, # Selectively enable domains
    max_rollback_tokens=200,              # Checkpoint history
)
```

---

## Making Requests

### HTTP API

All constraint types are specified in `sampling_params`. **Only one constraint type per request.**

```python
import requests

def generate_with_constraint(prompt, constraint_type, constraint_value):
    """Generate text with a constraint.

    Args:
        prompt: Input text
        constraint_type: One of "json_schema", "regex", "ebnf", "structural_tag"
        constraint_value: The constraint specification
    """
    sampling_params = {
        "max_new_tokens": 512,
        "temperature": 0,
        constraint_type: constraint_value,
    }

    response = requests.post(
        "http://localhost:30000/generate",
        json={
            "text": prompt,
            "sampling_params": sampling_params,
        }
    )
    return response.json()["text"]

# Example: Generate JSON matching a schema
result = generate_with_constraint(
    prompt="Generate a user profile:\n",
    constraint_type="json_schema",
    constraint_value='''{
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "email": {"type": "string", "format": "email"}
        },
        "required": ["name", "age"]
    }'''
)
```

### Batch Requests

```python
import requests

response = requests.post(
    "http://localhost:30000/generate",
    json={
        "text": [
            "Write a function to add numbers:\n",
            "Write a function to multiply numbers:\n",
        ],
        "sampling_params": {
            "max_new_tokens": 256,
            "regex": r"def \w+\([^)]*\) -> \w+:\n    .*",
        }
    }
)

for text in response.json()["text"]:
    print(text)
    print("---")
```

### OpenAI-Compatible API

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:30000/v1",
    api_key="not-needed",
)

response = client.chat.completions.create(
    model="default",
    messages=[
        {"role": "user", "content": "Generate a Python function signature:"}
    ],
    extra_body={
        "regex": r"def \w+\([^)]*\) -> \w+:",
    }
)

print(response.choices[0].message.content)
```

---

## Constraint Types

### 1. JSON Schema

Constrains output to valid JSON matching a schema.

```python
schema = {
    "type": "object",
    "properties": {
        "function_name": {"type": "string"},
        "parameters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string", "enum": ["int", "str", "float", "bool"]}
                }
            }
        },
        "return_type": {"type": "string"}
    },
    "required": ["function_name", "parameters", "return_type"]
}

response = requests.post(url, json={
    "text": "Describe a function to calculate area:\n",
    "sampling_params": {
        "json_schema": json.dumps(schema),
        "max_new_tokens": 256,
    }
})
```

### 2. Regular Expression

Constrains output to match a regex pattern.

```python
# Python function signature
regex = r"def [a-z_][a-z0-9_]*\([^)]*\)( -> [a-zA-Z_][a-zA-Z0-9_\[\], ]*)?: ..."

# Variable assignment
regex = r"[a-z_][a-z0-9_]* = (True|False|\d+|\"[^\"]*\")"

# Import statement
regex = r"(from [a-z_.]+ import [a-z_,\s]+|import [a-z_.]+)"

response = requests.post(url, json={
    "text": "Define a variable:\n",
    "sampling_params": {
        "regex": regex,
        "max_new_tokens": 64,
    }
})
```

### 3. EBNF Grammar

Constrains output to match an EBNF grammar specification.

```python
ebnf = '''
root ::= function_def

function_def ::= "def " identifier "(" params? ")" return_type? ":" body

identifier ::= [a-zA-Z_][a-zA-Z0-9_]*

params ::= param ("," param)*
param ::= identifier ": " type_hint

type_hint ::= "int" | "str" | "float" | "bool" | "List[" type_hint "]"

return_type ::= " -> " type_hint

body ::= "\\n" indent statement+
indent ::= "    "
statement ::= (return_stmt | expr_stmt) "\\n"
return_stmt ::= "return " expr
expr_stmt ::= identifier " = " expr
expr ::= identifier | number | identifier " + " expr
number ::= [0-9]+
'''

response = requests.post(url, json={
    "text": "# Add two numbers\n",
    "sampling_params": {
        "ebnf": ebnf,
        "max_new_tokens": 256,
    }
})
```

### 4. Structural Tags

Pre-defined structural patterns (if configured).

```python
response = requests.post(url, json={
    "text": "Generate code:\n",
    "sampling_params": {
        "structural_tag": "python_function",
        "max_new_tokens": 256,
    }
})
```

---

## Configuration Options

### Domain Configuration

Ananke supports five constraint domains:

| Domain | Description | Cost | Default |
|--------|-------------|------|---------|
| `syntax` | CFG-based parsing (via llguidance) | ~50μs | Enabled |
| `types` | Bidirectional type inference | ~500μs | Enabled |
| `imports` | Module/package tracking | ~200μs | Enabled |
| `controlflow` | CFG reachability analysis | ~100μs | Enabled |
| `semantics` | SMT constraint solving | ~1ms | Enabled |

To selectively enable domains:

```python
backend = AnankeBackend(
    tokenizer=tokenizer,
    vocab_size=32000,
    enabled_domains={"syntax", "types", "imports"},  # Disable controlflow, semantics
)
```

### Language Selection

Ananke supports multiple programming languages:

| Language | Type System | Import Resolution | Full Support |
|----------|-------------|-------------------|--------------|
| Python | Full | pip/stdlib | Yes |
| TypeScript | Full | npm/stdlib | Yes |
| Go | Full | go modules | Yes |
| Rust | Full | cargo | Yes |
| Kotlin | Full | maven/gradle | Yes |
| Swift | Full | SPM | Yes |
| Zig | Full | build.zig | Yes |

```python
backend = AnankeBackend(
    tokenizer=tokenizer,
    vocab_size=32000,
    language="typescript",  # or "python", "go", "rust", etc.
)
```

### Grammar Object Options

```python
from sglang.srt.constrained.ananke.backend.grammar import AnankeGrammar

grammar = AnankeGrammar(
    syntax_grammar=llguidance_grammar,
    domains=domains,
    vocab_size=32000,
    device="cuda",
    language="python",

    # Performance options:
    max_rollback_tokens=200,   # Checkpoint history (default: 200)
    checkpoint_interval=1,      # Checkpoint every N tokens (default: 1)
    mask_pool_size=8,          # Pre-allocated mask tensors (default: 8)
)
```

---

## Performance Tuning

### Lazy Evaluation Budget

Control how much time is spent on domain evaluation per token:

```python
from sglang.srt.constrained.ananke.masks.lazy import EvaluationBudget

# Fast mode: Skip expensive domains
fast_budget = EvaluationBudget(
    max_time_ns=1_000_000,    # 1ms max
    max_domains=3,            # Skip semantics
    min_selectivity=0.90,     # Stop early if mask is selective
)

# Thorough mode: Evaluate all domains
thorough_budget = EvaluationBudget(
    max_time_ns=5_000_000,    # 5ms max
    max_domains=5,            # All domains
    min_selectivity=0.99,     # Only stop if nearly all blocked
)
```

### Sparse Checkpointing

For scenarios with infrequent rollback:

```python
grammar = AnankeGrammar(
    checkpoint_interval=10,  # Checkpoint every 10 tokens (10x less overhead)
    max_rollback_tokens=200,
    ...
)
```

### Native SIMD Acceleration

Check if Zig native library is available:

```python
try:
    from sglang.srt.constrained.ananke.zig.ffi import is_native_available
    if is_native_available():
        print("SIMD mask fusion enabled (~50x faster)")
except ImportError:
    print("Using vectorized Python (still fast)")
```

### Cache Performance Monitoring

```python
# Get cache statistics
stats = grammar.get_cache_stats()
print(f"Cache size: {stats['cache_size']}")
print(f"Domains cached: {stats['domains_cached']}")

# Log cache summary
grammar.log_cache_summary()
```

---

## Troubleshooting

### Common Issues

#### 1. "ananke" not recognized as grammar backend

The Ananke backend must be imported before use:

```python
# Ensure the backend is registered
from sglang.srt.constrained.ananke.backend.backend import AnankeBackend
```

#### 2. Constraint too restrictive (no valid tokens)

If all tokens are masked, the grammar becomes unsatisfiable:

```python
# Check if grammar is finished/unsatisfiable
if grammar.is_terminated():
    print("Grammar reached terminal state")
    print(f"Constraint satisfiability: {grammar.constraint.satisfiability()}")
```

#### 3. Slow mask computation

Enable lazy evaluation and check domain timing:

```python
# Use lazy evaluation (default)
grammar.fill_vocab_mask(vocab_mask, idx, use_lazy_evaluation=True)

# Check lazy evaluator stats
stats = grammar.get_lazy_evaluator_stats()
```

#### 4. Rollback failures

Ensure checkpoint history is large enough:

```python
# Increase rollback capacity
grammar = AnankeGrammar(
    max_rollback_tokens=500,  # Increase from default 200
    ...
)
```

### Debug Logging

Enable debug logging for detailed information:

```python
import logging
logging.getLogger("sglang.srt.constrained.ananke").setLevel(logging.DEBUG)
```

### Verifying Constraint Compilation

```python
# Check if grammar compiled successfully
grammar = backend.dispatch_json(json_schema)
if grammar is None:
    print("Grammar compilation failed")
elif grammar == INVALID_GRAMMAR_OBJ:
    print("Invalid grammar specification")
else:
    print(f"Grammar ready, constraint: {grammar.constraint}")
```

---

## Next Steps

- [Constraint Specification Guide](./CONSTRAINT_SPEC.md) - Rich context passing (types, imports, semantics)
- [Reference Documentation](./REFERENCE.md) - Detailed API reference
- [Architecture Guide](./ARCHITECTURE.md) - System design and internals
- [Contributing](./CONTRIBUTING.md) - How to extend Ananke

---

## Examples

See the `tests/integration/` folder for complete examples:

```bash
# Run integration tests as examples
PYTHONPATH="$PWD/python:$PYTHONPATH" python -m pytest \
    python/sglang/srt/constrained/ananke/tests/integration/ -v -s
```
