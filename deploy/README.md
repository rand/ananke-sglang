# SGLang Ananke Deployment Infrastructure

Deployment tools and configurations for SGLang with the Ananke constrained generation backend.

## Quick Start

### Local Development (Docker Compose)

```bash
# Copy and configure environment
cp .env.example .env
# Edit .env with your HF_TOKEN and model path

# Start with Ananke backend
docker compose -f docker/compose.ananke.yaml up -d

# With monitoring stack
docker compose -f docker/compose.ananke.yaml -f docker/compose.monitoring.yaml up -d
```

### Kubernetes (Helm)

```bash
# Install with default values
helm install sglang helm/sglang-ananke

# Install with cloud-specific values
helm install sglang helm/sglang-ananke -f helm/sglang-ananke/values-gcp.yaml
helm install sglang helm/sglang-ananke -f helm/sglang-ananke/values-aws.yaml
```

### Cloud Platforms

```bash
# Modal.com
modal deploy modal/sglang_ananke.py

# AWS SageMaker
python aws/sagemaker/deploy.py --model meta-llama/Llama-3.1-8B-Instruct

# GCP Vertex AI
python gcp/vertexai/deploy.py --model meta-llama/Llama-3.1-8B-Instruct
```

## Directory Structure

```
deploy/
├── .env.example              # Environment configuration template
├── docker/
│   ├── Dockerfile.ananke     # Full Ananke image (all optional deps)
│   ├── Dockerfile.ananke-minimal  # Core Ananke only
│   ├── compose.ananke.yaml   # Local development
│   └── compose.monitoring.yaml    # Prometheus + Grafana overlay
├── helm/
│   └── sglang-ananke/        # Kubernetes Helm chart
├── scripts/
│   ├── deploy.py             # Unified deployment CLI
│   ├── healthcheck.py        # Health check utilities
│   └── build-zig-native.sh   # Build Zig native library
├── modal/                    # Modal.com deployment
├── aws/                      # AWS SageMaker/EKS
├── gcp/                      # GCP Vertex AI/GKE
└── monitoring/               # Prometheus alerts & Grafana dashboards
```

## Image Variants

| Variant | Tag | Description |
|---------|-----|-------------|
| Full | `sglang-ananke:latest` | All Ananke optional dependencies (z3, tree-sitter, immutables) |
| Minimal | `sglang-ananke:minimal` | Core Ananke only (llguidance base) |
| Native | `sglang-ananke:native` | Full + pre-built Zig SIMD library |

## Ananke Configuration

All Ananke features are configurable via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `SGLANG_GRAMMAR_BACKEND` | `ananke` | Grammar backend to use |
| `ANANKE_LANGUAGE` | `python` | Target language (python, typescript, go, rust, kotlin, swift, zig) |
| `ANANKE_MAX_ROLLBACK_TOKENS` | `200` | Maximum rollback history for speculative decoding |
| `ANANKE_ENABLED_DOMAINS` | (all) | Comma-separated: syntax,types,imports,controlflow,semantics |

See [docs/configuration.md](docs/configuration.md) for complete configuration reference.

## Health Checks

All deployments expose standard health endpoints:

- `/health` - Basic liveness check
- `/health_generate` - Readiness check (model loaded)
- `/v1/models` - List available models

Initial delay: 120 seconds (model loading time varies by size).

## Monitoring

Prometheus metrics exposed at `/metrics`:

- `sglang_requests_total` - Total request count
- `sglang_request_duration_seconds` - Request latency histogram
- `sglang_tokens_generated_total` - Total tokens generated
- `sglang_ananke_constraint_violations_total` - Constraint violation count
- `sglang_ananke_rollbacks_total` - Rollback operation count

Pre-built Grafana dashboards available in `monitoring/grafana/dashboards/`.
