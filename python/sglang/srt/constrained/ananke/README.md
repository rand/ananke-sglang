# Ananke: Compositional Constraint System for LLM Code Generation

Ananke is a multi-domain constrained decoding framework that enables type-safe,
semantically-guided code generation for large language models. It implements
compositional constraint checking across syntax, types, imports, control flow,
and semantic domains.

## Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│                              AnankeGrammar                                 │
│                                                                            │
│  ┌─────────────────┐   ┌─────────────────────────────────────────────────┐ │
│  │  Syntax Domain  │   │              Constraint Domains                 │ │
│  │   (llguidance)  │   │                                                 │ │
│  │                 │   │  ┌─────────┐ ┌─────────┐ ┌───────────────────┐  │ │
│  │  CFG Parsing    │   │  │  Types  │ │ Imports │ │   ControlFlow     │  │ │
│  │  JSON Schema    │   │  │ <500μs  │ │ <200μs  │ │     <100μs        │  │ │
│  │  Regex/EBNF     │   │  └────┬────┘ └────┬────┘ └─────────┬─────────┘  │ │
│  │    ~50μs        │   │       └───────────┴────────────────┘            │ │
│  └────────┬────────┘   │                   │                             │ │
│           │            │       ┌───────────▼────────────┐                │ │
│           │            │       │   Semantics Domain     │                │ │
│           │            │       │   (SMT/Z3, optional)   │                │ │
│           │            │       └────────────┬───────────┘                │ │
│           │            └────────────────────┼────────────────────────────┘ │
│           │                                 │                              │
│           │     ┌───────────────────────────▼────────────────────────────┐ │
│           └─────►          Mask Fusion (Zig SIMD ~50x faster)            │ │
│                 │  Selectivity-ordered · Early termination · Lazy eval   │ │
│                 └───────────────────────────┬────────────────────────────┘ │
│                                             ▼                              │
│  ┌──────────────────────┐    ┌─────────────────────────────────────────┐   │
│  │  Checkpoint System   │    │           Final Token Mask              │   │
│  │  Sparse + O(1) ops   │    │    Boolean tensor for valid tokens      │   │
│  └──────────────────────┘    └─────────────────────────────────────────┘   │
│                                                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Typed Holes System (Hazel)                       │   │
│  │  Environment capture · Fill-and-resume · Totality guarantees        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────────┘
```

For detailed architecture documentation, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Key Features

### Multi-Domain Constraint Checking
- **Syntax Domain**: CFG-based parsing via llguidance
- **Type Domain**: Bidirectional type inference with hole typing
- **Import Domain**: Module dependency tracking
- **ControlFlow Domain**: CFG construction and path analysis
- **Semantic Domain**: SMT-based constraint solving

### Performance Optimizations

| Optimization | Impact |
|--------------|--------|
| Vectorized mask application | 455x speedup |
| TokenClassifier disk cache | 38x faster warm start |
| Pre-allocated mask tensor pool | ~100μs/token savings |
| Incremental constraint computation | 50%+ cache hits |
| Lazy evaluation with budget control | Bounded latency |
| Immutable CFG with structural sharing | O(1) checkpoint |
| Sparse checkpointing | 10x fewer checkpoint ops |
| Zig SIMD mask fusion | ~50x vs vectorized Python |

## Usage

### Basic Integration

```python
from sglang.srt.constrained.ananke.backend import AnankeBackend

# Create backend with desired configuration
backend = AnankeBackend.create(
    language="python",
    enable_types=True,
    enable_imports=True,
    enable_controlflow=True,
)

# Create grammar for a generation request
grammar = backend.create_grammar(
    tokenizer=tokenizer,
    vocab_size=32000,
    device="cuda",
)

# Fill vocabulary mask for constrained decoding
grammar.fill_vocab_mask(vocab_mask, idx=0)

# Accept generated tokens
grammar.accept_token(token_id)
```

### Configuration Options

```python
AnankeGrammar(
    syntax_grammar=llguidance_grammar,  # Underlying syntax grammar
    domains={"types": type_domain, ...},  # Domain instances
    vocab_size=32000,
    device="cuda",
    language="python",
    max_rollback_tokens=200,  # Checkpoint history size
    checkpoint_interval=10,   # Sparse checkpointing interval
    mask_pool_size=8,         # Pre-allocated tensor pool size
)
```

## Supported Languages

All 7 languages have **full support** including parsing, types, imports, and control flow:

| Language | Parsing | Types | Imports | ControlFlow |
|----------|---------|-------|---------|-------------|
| Python | Complete | Complete | Complete | Complete |
| TypeScript | Complete | Complete | Complete | Complete |
| Go | Complete | Complete | Complete | Complete |
| Rust | Complete | Complete | Complete | Complete |
| Kotlin | Complete | Complete | Complete | Complete |
| Swift | Complete | Complete | Complete | Complete |
| Zig | Complete | Complete | Complete | Complete |

See [docs/LANGUAGE_IMPLEMENTATION_STATUS.md](docs/LANGUAGE_IMPLEMENTATION_STATUS.md) for detailed implementation information.

## Performance Tuning

### Lazy Evaluation Budget

```python
from sglang.srt.constrained.ananke.masks.lazy import EvaluationBudget

# Configure budget for latency-sensitive scenarios
budget = EvaluationBudget(
    max_time_ns=2_000_000,    # 2ms maximum
    max_domains=5,            # Skip expensive domains if budget exceeded
    min_selectivity=0.95,     # Stop if mask is highly selective
)
```

### Sparse Checkpointing

For scenarios with infrequent rollback:

```python
grammar = AnankeGrammar(
    checkpoint_interval=10,  # Checkpoint every 10 tokens
    ...
)
```

### Native Zig Acceleration

When the Zig native library is available, mask fusion uses SIMD instructions:

```python
from sglang.srt.constrained.ananke.zig.ffi import is_native_available

if is_native_available():
    print("Using SIMD-accelerated mask fusion")
```

## Testing

```bash
# Run all tests
PYTHONPATH="$PWD/python:$PYTHONPATH" python -m pytest \
    python/sglang/srt/constrained/ananke/tests/ -v

# Run benchmarks
PYTHONPATH="$PWD/python:$PYTHONPATH" python -m pytest \
    python/sglang/srt/constrained/ananke/tests/benchmark/ -v
```

## Documentation

| Document | Description |
|----------|-------------|
| [GETTING_STARTED.md](docs/GETTING_STARTED.md) | Quick start and usage guide |
| [REFERENCE.md](docs/REFERENCE.md) | Complete API reference |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | System architecture overview |
| [ARCHITECTURE_DEEP_DIVE.md](docs/ARCHITECTURE_DEEP_DIVE.md) | Mathematical foundations |
| [CONTRIBUTING.md](docs/CONTRIBUTING.md) | Development guide |
| [DEPLOYMENT.md](docs/DEPLOYMENT.md) | Deployment options |
| [CONSTRAINT_SPEC.md](docs/CONSTRAINT_SPEC.md) | Constraint specification format |
| [LANGUAGE_IMPLEMENTATION_STATUS.md](docs/LANGUAGE_IMPLEMENTATION_STATUS.md) | Language support details |

## References

- [Hazel](https://hazel.org): "Statically Contextualizing LLMs with Typed Holes" (OOPSLA 2024)
- [llguidance](https://github.com/microsoft/llguidance): Dynamic token masking
- [XGrammar](https://arxiv.org/abs/2404.14366): Grammar-constrained decoding

## License

Apache License 2.0
