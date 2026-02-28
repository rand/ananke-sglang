# Cloud Platform Deployment Guide

Deploy SGLang with Ananke backend to Modal.com, AWS SageMaker, and GCP Vertex AI.

## Table of Contents

- [Modal.com](#modalcom)
- [AWS SageMaker](#aws-sagemaker)
- [GCP Vertex AI](#gcp-vertex-ai)
- [Comparison](#comparison)

## Modal.com

Modal provides serverless GPU infrastructure with automatic scaling.

### Prerequisites

- Modal account: https://modal.com
- Modal CLI installed: `pip install modal`
- Modal token configured: `modal token new`

### Deploy

```bash
cd deploy/modal

# Deploy the service
modal deploy sglang_ananke.py

# Test locally first
modal run sglang_ananke.py

# Development with hot-reload
modal serve sglang_ananke.py
```

### Configuration

Environment variables in `sglang_ananke.py`:

```python
MODEL_PATH = "meta-llama/Llama-3.1-8B-Instruct"
ANANKE_LANGUAGE = "python"
ANANKE_MAX_ROLLBACK_TOKENS = 200
GPU_TYPE = "a100"  # a100, a10g, h100, t4
GPU_COUNT = 1
```

### Using the API

```python
import modal

# Connect to deployed app
app = modal.App.lookup("sglang-ananke")
SGLangAnanke = modal.Cls.lookup("sglang-ananke", "SGLangAnanke")

sglang = SGLangAnanke()

# Simple generation
result = sglang.generate.remote(
    prompt="def fibonacci(n):",
    max_tokens=100,
)

# Constrained generation
result = sglang.generate_constrained.remote(
    prompt="def add(a: int, b: int) -> int:",
    constraint_spec={
        "language": "python",
        "domains": ["syntax", "types"]
    },
    max_tokens=100,
)
```

### Pre-download Models

For faster cold starts:

```bash
modal run sglang_ananke.py::download_model --model-path meta-llama/Llama-3.1-8B-Instruct
```

## AWS SageMaker

Deploy as a SageMaker real-time inference endpoint.

### Prerequisites

- AWS account with SageMaker access
- AWS CLI configured: `aws configure`
- ECR repository for container images
- SageMaker execution role

### Build and Push Image

```bash
# Set variables
export AWS_REGION=us-west-2
export AWS_ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
export ECR_REPO=$AWS_ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com/sglang-ananke

# Create ECR repository
aws ecr create-repository --repository-name sglang-ananke

# Login to ECR
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ECR_REPO

# Build and push
cd deploy/aws/sagemaker
docker build -t sglang-ananke-sagemaker .
docker tag sglang-ananke-sagemaker:latest $ECR_REPO:latest
docker push $ECR_REPO:latest
```

### Deploy Endpoint

```bash
cd deploy/aws/sagemaker

# Create model
python deploy.py create-model \
    --model-name sglang-ananke \
    --image-uri $ECR_REPO:latest

# Deploy endpoint
python deploy.py deploy \
    --endpoint-name sglang-ananke \
    --model-name sglang-ananke \
    --instance-type ml.g5.xlarge
```

### Test Endpoint

```bash
python deploy.py test \
    --endpoint-name sglang-ananke \
    --prompt "def hello():"
```

### Instance Types

| Instance | GPU | Memory | Use Case |
|----------|-----|--------|----------|
| ml.g5.xlarge | A10G (24GB) | 16GB | 7B models |
| ml.g5.2xlarge | A10G (24GB) | 32GB | 7B models with more memory |
| ml.p4d.24xlarge | 8x A100 (40GB) | 1152GB | 70B+ models |

### Clean Up

```bash
python deploy.py delete --endpoint-name sglang-ananke
```

## GCP Vertex AI

Deploy as a Vertex AI custom prediction endpoint.

### Prerequisites

- GCP project with Vertex AI API enabled
- gcloud CLI configured: `gcloud auth login`
- Artifact Registry repository
- Service account with Vertex AI permissions

### Build and Push Image

```bash
# Set variables
export GCP_PROJECT=your-project-id
export GCP_REGION=us-central1
export AR_REPO=$GCP_REGION-docker.pkg.dev/$GCP_PROJECT/sglang/sglang-ananke

# Create Artifact Registry repository
gcloud artifacts repositories create sglang \
    --repository-format=docker \
    --location=$GCP_REGION

# Configure Docker for Artifact Registry
gcloud auth configure-docker $GCP_REGION-docker.pkg.dev

# Build and push
cd deploy/docker
docker build -f Dockerfile.ananke -t sglang-ananke ../../
docker tag sglang-ananke:latest $AR_REPO:latest
docker push $AR_REPO:latest
```

### Deploy Endpoint

```bash
cd deploy/gcp/vertexai

# Upload model
python deploy.py upload-model \
    --model-name sglang-ananke \
    --image-uri $AR_REPO:latest

# Deploy endpoint
python deploy.py deploy \
    --endpoint-name sglang-ananke \
    --model-name sglang-ananke \
    --gpu-type l4 \
    --create-endpoint
```

### Test Endpoint

```bash
python deploy.py test \
    --endpoint-name sglang-ananke \
    --prompt "def hello():"
```

### GPU Types

| Type | GPU | Memory | Availability |
|------|-----|--------|--------------|
| l4 | NVIDIA L4 | 24GB | Most regions |
| t4 | NVIDIA T4 | 16GB | All regions |
| a100-40gb | A100 (40GB) | 40GB | Limited regions |
| a100-80gb | A100 (80GB) | 80GB | Very limited |

### Clean Up

```bash
python deploy.py delete --endpoint-name sglang-ananke --force
```

## Comparison

| Feature | Modal.com | AWS SageMaker | GCP Vertex AI |
|---------|-----------|---------------|---------------|
| **Setup Complexity** | Low | Medium | Medium |
| **Auto-scaling** | Built-in | Manual/HPA | Built-in |
| **Cold Start** | Fast (GPU pool) | Slow (5-10min) | Slow (5-10min) |
| **GPU Options** | T4, A10G, A100, H100 | T4, V100, A10G, A100 | T4, L4, A100 |
| **Pricing Model** | Per-second | Per-hour | Per-hour |
| **Best For** | Development, bursty traffic | Production, enterprise | GCP-native apps |

### Cost Estimates (A100 40GB)

| Platform | Hourly Cost | Monthly (24/7) |
|----------|------------|----------------|
| Modal.com | ~$3.50/hr active | Pay per use |
| AWS SageMaker | ~$5.67/hr | ~$4,080 |
| GCP Vertex AI | ~$5.07/hr | ~$3,650 |

*Prices approximate and subject to change. Check current pricing.*

## Unified CLI

Use the unified CLI for any platform:

```bash
# Modal
python deploy/scripts/deploy.py modal deploy

# SageMaker
python deploy/scripts/deploy.py sagemaker deploy --model-name sglang-ananke

# Vertex AI
python deploy/scripts/deploy.py vertex deploy --model-name sglang-ananke
```
