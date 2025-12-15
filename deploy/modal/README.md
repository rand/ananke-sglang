# SGLang Ananke Deployment on Modal.com

Deploy SGLang with the Ananke constrained generation backend on Modal's serverless GPU infrastructure.

## Overview

This directory contains Modal deployment configurations for SGLang with Ananke:

| File | Description |
|------|-------------|
| `sglang_ananke.py` | Generic deployment for any model with Ananke |
| `qwen3_coder_ananke.py` | Production deployment for Qwen3-Coder-30B-A3B-Instruct |
| `tests/test_grammar_backends.py` | Comparative tests: unconstrained vs Ananke |
| `tests/test_qwen3_coder.py` | E2E tests for Qwen3-Coder deployment |

## Quick Start

### Prerequisites

1. Modal.com account with GPU access (A100-80GB recommended)
2. HuggingFace token for model access

```bash
# Install Modal CLI
pip install modal

# Set up Modal authentication
modal setup

# Create HuggingFace secret
modal secret create huggingface HF_TOKEN=<your-token>
```

### Deploy Qwen3-Coder

```bash
# Pre-download model (recommended, ~60GB)
modal run deploy/modal/qwen3_coder_ananke.py::download_model

# Deploy the service
modal deploy deploy/modal/qwen3_coder_ananke.py

# Run basic tests
modal run deploy/modal/qwen3_coder_ananke.py
```

### Run Grammar Backend Comparison

```bash
modal run deploy/modal/tests/test_grammar_backends.py
```

## API Usage

### Unconstrained Generation

```python
result = server.generate.remote(
    prompt="def fibonacci(n: int) -> int:",
    max_tokens=200,
)
```

### Constrained Generation with Ananke

```python
# constraint_spec is a TOP-LEVEL field in the API
result = server.generate_constrained.remote(
    prompt="def fibonacci(n: int) -> int:",
    constraint_spec={
        "language": "python",
        "domains": ["syntax"],
    },
    max_tokens=200,
)
```

### Chat Completion

```python
result = server.chat.remote(
    messages=[{"role": "user", "content": "Write a prime check function"}],
    constraint_spec={"language": "python", "domains": ["syntax"]},
    max_tokens=200,
)
```

## Constraint Specification

The `constraint_spec` field supports:

```python
{
    "language": "python",  # Target language
    "domains": ["syntax", "types", "imports"],  # Constraint domains
    "typing_context": {...},  # Optional type context
}
```

**Important**: `constraint_spec` must be a top-level field in API requests, NOT nested under `extra_body`.

## Test Results

Comparative testing on Qwen3-Coder-30B-A3B-Instruct (A100-80GB):

### Overall Syntax Validity
| Backend | Rate |
|---------|------|
| Unconstrained | 62% |
| Ananke | 62% |

### High-Temperature Stress Test (temp=0.9)
| Backend | Rate |
|---------|------|
| Unconstrained | 10% |
| Ananke | **30%** |

**Key Findings**:
- Ananke is **3x better** under high uncertainty (high temperature)
- Ananke produces more compact, complete code (fewer tokens)
- ~22% latency overhead for grammar enforcement

## Configuration

### GPU Options

| GPU | VRAM | Recommended For |
|-----|------|-----------------|
| A100-80GB | 80GB | Qwen3-Coder-30B (MoE) |
| A100-40GB | 40GB | Models up to 13B |
| A10G | 24GB | Models up to 7B |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODAL_GPU_TYPE` | `a100` | GPU type (a100, a10g, h100) |
| `MODAL_GPU_MEMORY` | `40GB` | GPU memory (40GB or 80GB) |
| `ANANKE_LANGUAGE` | `python` | Default constraint language |
| `ANANKE_MAX_ROLLBACK_TOKENS` | `200` | Max rollback on constraint violation |

## Troubleshooting

### Model fails to load
- Ensure HuggingFace secret is set: `modal secret list`
- Check GPU has sufficient VRAM
- Pre-download model to volume first

### Constraint spec not applied
- Verify `constraint_spec` is a top-level field (not in `extra_body`)
- Check server started with `--grammar-backend ananke`

### High latency on first request
- Grammar initialization takes 5-10s on first constrained request
- Subsequent requests are much faster

## License

Apache 2.0 - See repository root for details.
