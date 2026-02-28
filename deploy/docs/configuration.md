# SGLang Ananke Configuration Reference

Complete reference for all configuration options when deploying SGLang with the Ananke backend.

## Table of Contents

- [Environment Variables](#environment-variables)
- [CLI Arguments](#cli-arguments)
- [Constraint Specification](#constraint-specification)
- [Platform-Specific Configuration](#platform-specific-configuration)

## Environment Variables

### Model Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | - | HuggingFace model path or local path (required) |
| `HF_TOKEN` | - | HuggingFace API token for gated models |
| `MODEL_REVISION` | `main` | Model revision/branch to use |

### Server Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `SGLANG_HOST` | `0.0.0.0` | Server bind address |
| `SGLANG_PORT` | `30000` | Server port |
| `SGLANG_TP_SIZE` | `1` | Tensor parallelism (GPUs for model sharding) |
| `SGLANG_PP_SIZE` | `1` | Pipeline parallelism |
| `SGLANG_DP_SIZE` | `1` | Data parallelism (model replicas) |
| `SGLANG_MEM_FRACTION` | `0.9` | GPU memory utilization fraction |
| `SGLANG_LOG_LEVEL` | `INFO` | Log level (DEBUG, INFO, WARNING, ERROR) |
| `SGLANG_ENABLE_METRICS` | `true` | Enable Prometheus metrics |

### Ananke Backend Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `SGLANG_GRAMMAR_BACKEND` | `ananke` | Grammar backend (ananke, xgrammar, outlines, llguidance, none) |
| `ANANKE_LANGUAGE` | `python` | Target language (python, typescript, go, rust, kotlin, swift, zig) |
| `ANANKE_MAX_ROLLBACK_TOKENS` | `200` | Maximum rollback history for speculative decoding |
| `ANANKE_ENABLED_DOMAINS` | (all) | Comma-separated domain list |

### Ananke Domains

Available constraint domains:

| Domain | Description |
|--------|-------------|
| `syntax` | Grammar and syntax validation |
| `types` | Type checking and inference |
| `imports` | Module and import resolution |
| `controlflow` | Control flow analysis |
| `semantics` | Semantic analysis (variable scoping, etc.) |

Example: `ANANKE_ENABLED_DOMAINS=syntax,types,imports`

## CLI Arguments

When starting SGLang directly:

```bash
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --host 0.0.0.0 \
    --port 30000 \
    --grammar-backend ananke \
    --ananke-language python \
    --ananke-max-rollback-tokens 200 \
    --ananke-enabled-domains syntax,types,imports
```

### Full Argument Reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-path` | - | Model path (required) |
| `--host` | `127.0.0.1` | Server host |
| `--port` | `30000` | Server port |
| `--tp-size` | `1` | Tensor parallelism |
| `--grammar-backend` | `xgrammar` | Grammar backend |
| `--ananke-language` | `python` | Target programming language |
| `--ananke-max-rollback-tokens` | `200` | Rollback history size |
| `--ananke-enabled-domains` | (all) | Enabled constraint domains |

## Constraint Specification

The Ananke backend supports rich constraint specifications via the API.

### Basic Syntax Constraints

```json
{
    "constraint_spec": {
        "language": "python",
        "domains": ["syntax"]
    }
}
```

### Type-Aware Constraints

```json
{
    "constraint_spec": {
        "language": "python",
        "domains": ["syntax", "types"],
        "context": {
            "type_context": {
                "x": "int",
                "y": "str",
                "return": "bool"
            }
        }
    }
}
```

### Import Resolution

```json
{
    "constraint_spec": {
        "language": "python",
        "domains": ["syntax", "imports"],
        "context": {
            "imports": [
                {"module": "typing", "names": ["Optional", "List"]},
                {"module": "dataclasses", "names": ["dataclass"]}
            ]
        }
    }
}
```

### Full Constraint Spec

```json
{
    "constraint_spec": {
        "language": "python",
        "domains": ["syntax", "types", "imports", "controlflow", "semantics"],
        "context": {
            "imports": [...],
            "type_context": {...},
            "scope": {...}
        },
        "options": {
            "strict_mode": true,
            "allow_partial": false
        }
    }
}
```

## Platform-Specific Configuration

### Docker

```bash
# Environment file
cp deploy/.env.example deploy/.env
# Edit .env with your configuration

# Run with Docker Compose
docker compose -f deploy/docker/compose.ananke.yaml up -d
```

### Kubernetes (Helm)

```yaml
# values.yaml
model:
  path: "meta-llama/Llama-3.1-8B-Instruct"

ananke:
  enabled: true
  language: "python"
  maxRollbackTokens: 200
  enabledDomains: "syntax,types,imports"

resources:
  limits:
    nvidia.com/gpu: 1
```

```bash
helm install sglang deploy/helm/sglang-ananke -f values.yaml
```

### Modal.com

```python
# Environment variables in Modal
MODEL_PATH = "meta-llama/Llama-3.1-8B-Instruct"
ANANKE_LANGUAGE = "python"
ANANKE_MAX_ROLLBACK_TOKENS = 200
```

### AWS SageMaker

```python
# Container environment
environment = {
    "SGLANG_GRAMMAR_BACKEND": "ananke",
    "ANANKE_LANGUAGE": "python",
    "ANANKE_MAX_ROLLBACK_TOKENS": "200",
}
```

### GCP Vertex AI

```python
# Container environment
serving_container_environment_variables = {
    "SGLANG_GRAMMAR_BACKEND": "ananke",
    "ANANKE_LANGUAGE": "python",
    "ANANKE_MAX_ROLLBACK_TOKENS": "200",
}
```

## Performance Tuning

### Memory Configuration

| Model Size | Recommended `MEM_FRACTION` | Min GPU Memory |
|------------|---------------------------|----------------|
| 7B | 0.9 | 16GB |
| 13B | 0.9 | 24GB |
| 34B | 0.85 | 40GB (or 2x24GB TP=2) |
| 70B | 0.85 | 80GB (or 2x40GB TP=2) |

### Rollback Token Configuration

| Use Case | Recommended `MAX_ROLLBACK_TOKENS` |
|----------|-----------------------------------|
| Short functions | 100 |
| General code | 200 (default) |
| Long functions | 500 |
| Complex classes | 1000 |

### Domain Configuration by Use Case

| Use Case | Recommended Domains |
|----------|---------------------|
| Quick syntax check | `syntax` |
| Type-safe code | `syntax,types` |
| Full Python | `syntax,types,imports` |
| Maximum validation | `syntax,types,imports,controlflow,semantics` |
