# Quick Start Guide

Get SGLang with Ananke backend running in under 5 minutes.

## Prerequisites

- Docker with NVIDIA Container Toolkit
- At least one NVIDIA GPU with 16GB+ memory
- HuggingFace account (for gated models)

## Option 1: Docker Compose (Recommended)

```bash
# Clone the repository
git clone https://github.com/sgl-project/sglang.git
cd sglang

# Configure environment
cp deploy/.env.example deploy/.env

# Edit .env and set your HF_TOKEN
# HF_TOKEN=hf_xxxxx

# Start the server
cd deploy
docker compose -f docker/compose.ananke.yaml up -d

# Check logs
docker compose -f docker/compose.ananke.yaml logs -f

# Wait for "Server ready" message (2-5 minutes for model loading)
```

## Option 2: Docker Run

```bash
docker run -d \
    --name sglang-ananke \
    --gpus all \
    -p 30000:30000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -e HF_TOKEN=hf_xxxxx \
    --shm-size 10g \
    --ipc host \
    lmsysorg/sglang:latest \
    python3 -m sglang.launch_server \
        --model-path meta-llama/Llama-3.1-8B-Instruct \
        --host 0.0.0.0 \
        --port 30000 \
        --grammar-backend ananke
```

## Test the Server

### Health Check

```bash
curl http://localhost:30000/health
# {"status": "healthy"}
```

### Simple Completion

```bash
curl -X POST http://localhost:30000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "default",
        "prompt": "def fibonacci(n):",
        "max_tokens": 100
    }'
```

### Constrained Generation (Ananke)

```bash
curl -X POST http://localhost:30000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "default",
        "prompt": "def calculate_sum(numbers: list[int]) -> int:",
        "max_tokens": 100,
        "extra_body": {
            "constraint_spec": {
                "language": "python",
                "domains": ["syntax", "types"]
            }
        }
    }'
```

## Next Steps

- [Configuration Reference](../configuration.md) - Full configuration options
- [Kubernetes Deployment](kubernetes.md) - Deploy on K8s with Helm
- [Cloud Deployment](cloud-platforms.md) - Deploy on Modal, AWS, or GCP
- [Monitoring Setup](monitoring.md) - Set up Prometheus and Grafana
