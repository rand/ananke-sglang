# Kubernetes Deployment Guide

Deploy SGLang with Ananke backend on Kubernetes using Helm.

## Prerequisites

- Kubernetes cluster with GPU nodes
- NVIDIA GPU Operator installed
- Helm 3.x
- kubectl configured

## Quick Start

```bash
# Add Helm repository (when published)
# helm repo add sglang https://sgl-project.github.io/charts
# helm repo update

# Or install from local chart
helm install sglang deploy/helm/sglang-ananke \
    --set model.hfToken=hf_xxxxx \
    --set model.path=meta-llama/Llama-3.1-8B-Instruct
```

## Configuration

### Basic Values

```yaml
# values.yaml
image:
  repository: lmsysorg/sglang
  tag: latest

model:
  path: "meta-llama/Llama-3.1-8B-Instruct"
  hfToken: ""  # Or use existingSecret

ananke:
  enabled: true
  language: "python"
  maxRollbackTokens: 200

resources:
  limits:
    nvidia.com/gpu: 1
    memory: 40Gi
  requests:
    nvidia.com/gpu: 1
    memory: 16Gi
```

### Using Secrets

```yaml
# Create secret first
kubectl create secret generic hf-token --from-literal=token=hf_xxxxx

# values.yaml
model:
  existingSecret:
    name: hf-token
    key: token
```

### Persistent Storage

```yaml
modelVolume:
  enabled: true
  type: pvc
  pvc:
    size: 100Gi
    storageClassName: fast-ssd
```

## Cloud-Specific Configurations

### GKE (Google Kubernetes Engine)

```bash
helm install sglang deploy/helm/sglang-ananke \
    -f deploy/helm/sglang-ananke/values-gcp.yaml \
    --set model.hfToken=hf_xxxxx
```

Key GKE features:
- Node selector for A100/L4 GPUs
- GCS-backed storage via Filestore
- GKE Ingress with managed certificates
- Workload Identity for GCS access

### EKS (Amazon Elastic Kubernetes Service)

```bash
helm install sglang deploy/helm/sglang-ananke \
    -f deploy/helm/sglang-ananke/values-aws.yaml \
    --set model.hfToken=hf_xxxxx
```

Key EKS features:
- Node selector for p4d/g5 instances
- EBS GP3 storage
- AWS ALB Ingress
- IRSA for S3 access

## Scaling

### Horizontal Pod Autoscaler

```yaml
autoscaling:
  enabled: true
  minReplicas: 1
  maxReplicas: 4
  targetCPUUtilizationPercentage: 80
```

### Manual Scaling

```bash
kubectl scale deployment sglang-sglang-ananke --replicas=3
```

## Monitoring

### Enable ServiceMonitor

```yaml
monitoring:
  serviceMonitor:
    enabled: true
    namespace: monitoring
```

### Port Forward for Testing

```bash
kubectl port-forward svc/sglang-sglang-ananke 30000:80
```

## Troubleshooting

### Check Pod Status

```bash
kubectl get pods -l app.kubernetes.io/name=sglang-ananke
kubectl describe pod <pod-name>
```

### View Logs

```bash
kubectl logs -f -l app.kubernetes.io/name=sglang-ananke
```

### Common Issues

**Pod stuck in Pending:**
- Check GPU availability: `kubectl describe nodes | grep nvidia`
- Verify GPU tolerations in values.yaml

**OOMKilled:**
- Increase memory limits
- Reduce model size or use tensor parallelism

**Model loading timeout:**
- Increase `healthCheck.liveness.initialDelaySeconds`
- Pre-download model to PVC
