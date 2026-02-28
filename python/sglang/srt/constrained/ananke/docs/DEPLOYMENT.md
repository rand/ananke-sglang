# Deploying Ananke

This guide covers deployment options for SGLang with the Ananke constrained generation backend.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Modal.com Deployment](#modalcom-deployment)
3. [Docker Deployment](#docker-deployment)
4. [Production Configuration](#production-configuration)
5. [Monitoring and Observability](#monitoring-and-observability)
6. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Local Development Server

```bash
# Start SGLang with Ananke backend
python -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --grammar-backend ananke \
    --port 30000

# Test the endpoint
curl -X POST http://localhost:30000/generate \
    -H "Content-Type: application/json" \
    -d '{
        "text": "def add(a: int, b: int) -> int:\n",
        "sampling_params": {
            "max_new_tokens": 50,
            "regex": "    return [a-z]+ \\+ [a-z]+"
        }
    }'
```

---

## Modal.com Deployment

Modal.com provides serverless GPU infrastructure ideal for Ananke deployments.

### Overview

The Modal deployment is located at `deploy/modal/` in the repository root:

```
deploy/modal/
├── sglang_ananke.py          # Generic Ananke deployment
├── qwen3_coder_ananke.py     # Production Qwen3-Coder deployment
└── tests/
    ├── test_grammar_backends.py  # Backend comparison tests
    └── test_qwen3_coder.py       # E2E tests
```

### Prerequisites

```bash
# Install Modal CLI
pip install modal

# Authenticate
modal setup

# Create HuggingFace secret (for model access)
modal secret create huggingface HF_TOKEN=<your-token>
```

### Deploy

```bash
# Pre-download model (recommended for large models)
modal run deploy/modal/qwen3_coder_ananke.py::download_model

# Deploy the service
modal deploy deploy/modal/qwen3_coder_ananke.py

# Run basic tests
modal run deploy/modal/qwen3_coder_ananke.py
```

### API Usage

```python
# Unconstrained generation
result = server.generate.remote(
    prompt="def fibonacci(n: int) -> int:",
    max_tokens=200,
)

# Constrained generation with Ananke
result = server.generate_constrained.remote(
    prompt="def fibonacci(n: int) -> int:",
    constraint_spec={
        "language": "python",
        "domains": ["syntax", "types"],
    },
    max_tokens=200,
)
```

**Important**: `constraint_spec` must be a top-level field, NOT nested under `extra_body`.

### GPU Configuration

| GPU | VRAM | Recommended For |
|-----|------|-----------------|
| A100-80GB | 80GB | Qwen3-Coder-30B, large MoE models |
| A100-40GB | 40GB | Models up to 13B |
| A10G | 24GB | Models up to 7B |

For detailed Modal deployment documentation, see [deploy/modal/README.md](../../../../deploy/modal/README.md).

---

## Docker Deployment

### Basic Dockerfile

```dockerfile
FROM nvidia/cuda:12.1-devel-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y python3.11 python3-pip

# Install SGLang with Ananke
RUN pip install sglang[all]

# Optional: Install Z3 for semantic constraints
RUN apt-get install -y z3 libz3-dev

# Copy Ananke configuration
COPY ananke_config.yaml /app/config.yaml

# Start server
CMD ["python", "-m", "sglang.launch_server", \
     "--model-path", "${MODEL_PATH}", \
     "--grammar-backend", "ananke", \
     "--port", "30000"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  sglang-ananke:
    build: .
    ports:
      - "30000:30000"
    environment:
      - MODEL_PATH=meta-llama/Meta-Llama-3.1-8B-Instruct
      - ANANKE_LANGUAGE=python
      - ANANKE_MAX_ROLLBACK_TOKENS=200
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    volumes:
      - model-cache:/root/.cache/huggingface

volumes:
  model-cache:
```

### Run with Docker

```bash
# Build
docker build -t sglang-ananke .

# Run
docker run --gpus all -p 30000:30000 \
    -e MODEL_PATH=meta-llama/Meta-Llama-3.1-8B-Instruct \
    sglang-ananke
```

---

## Production Configuration

### Server Arguments

```bash
python -m sglang.launch_server \
    --model-path <MODEL_PATH> \
    --grammar-backend ananke \
    --port 30000 \
    # Ananke-specific options
    --ananke-language python \
    --ananke-max-rollback-tokens 200 \
    --ananke-enabled-domains "syntax,types,imports" \
    # Performance options
    --tensor-parallel-size 2 \
    --max-running-requests 32 \
    --mem-fraction-static 0.8
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ANANKE_LANGUAGE` | `python` | Default target language |
| `ANANKE_MAX_ROLLBACK_TOKENS` | `200` | Maximum rollback depth |
| `ANANKE_ENABLED_DOMAINS` | all | Comma-separated domain list |
| `ANANKE_CHECKPOINT_INTERVAL` | `1` | Checkpoint every N tokens |
| `ANANKE_MASK_POOL_SIZE` | `8` | Pre-allocated mask tensors |

### Performance Tuning

#### For Latency-Sensitive Workloads

```bash
# Fewer domains, faster response
--ananke-enabled-domains "syntax,types"

# Sparse checkpointing (if rollback is rare)
# Configure via code:
# grammar = AnankeGrammar(checkpoint_interval=10, ...)
```

#### For Throughput-Oriented Workloads

```bash
# All domains for maximum constraint coverage
--ananke-enabled-domains "syntax,types,imports,controlflow,semantics"

# Increase concurrent requests
--max-running-requests 64
```

---

## Monitoring and Observability

### Logging

```python
import logging

# Enable Ananke debug logging
logging.getLogger("sglang.srt.constrained.ananke").setLevel(logging.DEBUG)
```

### Cache Statistics

```python
# After generation, get cache stats
stats = grammar.get_cache_stats()
print(f"Cache size: {stats['cache_size']}")
print(f"Domains cached: {stats['domains_cached']}")

# Log summary
grammar.log_cache_summary()
```

### Metrics to Monitor

| Metric | Description | Target |
|--------|-------------|--------|
| `tokens_per_second` | Generation throughput | Model-dependent |
| `constraint_check_latency_ms` | Per-token constraint time | <2-3ms |
| `cache_hit_rate` | Mask cache effectiveness | >50% |
| `rollback_count` | Constraint violation recovery | Low |
| `domain_latency_ms` | Per-domain breakdown | See targets below |

### Domain Latency Targets

| Domain | Target | Notes |
|--------|--------|-------|
| Syntax | ~50μs | llguidance |
| Types | <500μs | Incremental |
| Imports | <200μs | Statement detection |
| ControlFlow | <100μs | CFG analysis |
| Semantics | <1ms | Z3 dependent |

---

## Troubleshooting

### "ananke" not recognized as grammar backend

Ensure the backend is registered:

```python
from sglang.srt.constrained.ananke.backend.backend import AnankeBackend
```

### Constraint spec not applied

- Verify `constraint_spec` is a **top-level field** (not in `extra_body`)
- Check server started with `--grammar-backend ananke`

### High latency on first request

Grammar initialization takes 5-10 seconds on first constrained request. Subsequent requests are faster due to caching.

### Out of memory with large vocabularies

Reduce mask pool size:
```python
grammar = AnankeGrammar(mask_pool_size=4, ...)  # Default is 8
```

### Type errors not detected

Ensure types domain is enabled:
```bash
--ananke-enabled-domains "syntax,types"
```

### Z3 not available

The semantics domain works without Z3 but with reduced functionality. Install Z3:
```bash
pip install z3-solver
```

---

## See Also

- [Modal Deployment README](../../../../deploy/modal/README.md) - Detailed Modal guide
- [GETTING_STARTED.md](./GETTING_STARTED.md) - Usage guide
- [REFERENCE.md](./REFERENCE.md) - API reference
- [ARCHITECTURE.md](./ARCHITECTURE.md) - System overview
