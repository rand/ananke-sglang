# Monitoring Setup Guide

Set up Prometheus and Grafana monitoring for SGLang with Ananke backend.

## Quick Start with Docker Compose

```bash
cd deploy

# Start SGLang with monitoring stack
docker compose -f docker/compose.ananke.yaml -f docker/compose.monitoring.yaml up -d

# Access dashboards
# Grafana: http://localhost:3000 (admin/admin)
# Prometheus: http://localhost:9090
```

## Components

### Prometheus

Collects and stores metrics:
- SGLang server metrics
- Ananke constraint engine metrics
- GPU metrics (via nvidia-exporter)
- Host metrics (via node-exporter)

### Grafana

Visualization and dashboards:
- Pre-built SGLang Ananke dashboard
- GPU utilization panels
- Request latency graphs
- Constraint violation tracking

## Metrics Reference

### SGLang Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `sglang_requests_total` | Counter | Total requests by status |
| `sglang_request_duration_seconds` | Histogram | Request latency distribution |
| `sglang_tokens_generated_total` | Counter | Total tokens generated |
| `sglang_requests_queued` | Gauge | Current queue depth |

### Ananke Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `sglang_ananke_constraint_violations_total` | Counter | Constraint violations by domain |
| `sglang_ananke_rollbacks_total` | Counter | Rollback operations |
| `sglang_ananke_cache_hits_total` | Counter | Grammar cache hits |
| `sglang_ananke_cache_misses_total` | Counter | Grammar cache misses |

### GPU Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `nvidia_gpu_utilization` | Gauge | GPU utilization percentage |
| `nvidia_gpu_memory_used_bytes` | Gauge | GPU memory used |
| `nvidia_gpu_memory_total_bytes` | Gauge | Total GPU memory |
| `nvidia_gpu_temperature_celsius` | Gauge | GPU temperature |

## Alerting

Pre-configured alerts in `monitoring/prometheus/alerts/ananke-alerts.yaml`:

### Critical Alerts

| Alert | Condition | Description |
|-------|-----------|-------------|
| `SGLangServerDown` | Server unreachable for 1min | Immediate response needed |
| `GPUMemoryHigh` | Memory >95% for 5min | Risk of OOM errors |

### Warning Alerts

| Alert | Condition | Description |
|-------|-----------|-------------|
| `SGLangHighLatency` | P95 >30s for 5min | Performance degradation |
| `SGLangErrorRateHigh` | Error rate >5% | Application issues |
| `AnankeConstraintViolationsHigh` | >10/s violations | Check grammar configs |
| `AnankeRollbacksHigh` | >20/s rollbacks | Increase rollback tokens |

## Kubernetes Integration

### ServiceMonitor (Prometheus Operator)

Enable in Helm values:

```yaml
monitoring:
  serviceMonitor:
    enabled: true
    namespace: monitoring
    interval: 30s
```

### PodMonitor Alternative

```yaml
apiVersion: monitoring.coreos.com/v1
kind: PodMonitor
metadata:
  name: sglang-ananke
spec:
  selector:
    matchLabels:
      app.kubernetes.io/name: sglang-ananke
  podMetricsEndpoints:
    - port: http
      path: /metrics
```

## Custom Dashboards

### Import Pre-built Dashboard

1. Open Grafana (http://localhost:3000)
2. Go to Dashboards → Import
3. Upload `monitoring/grafana/dashboards/sglang-ananke.json`

### Create Custom Panels

Example PromQL queries:

```promql
# Request rate
rate(sglang_requests_total[5m])

# P95 latency
histogram_quantile(0.95, rate(sglang_request_duration_seconds_bucket[5m]))

# Error rate
rate(sglang_requests_total{status="error"}[5m]) / rate(sglang_requests_total[5m])

# Ananke cache hit rate
rate(sglang_ananke_cache_hits_total[5m]) / (rate(sglang_ananke_cache_hits_total[5m]) + rate(sglang_ananke_cache_misses_total[5m]))

# GPU memory utilization
nvidia_gpu_memory_used_bytes / nvidia_gpu_memory_total_bytes * 100
```

## Alertmanager Integration

Add alertmanager for notifications:

```yaml
# prometheus.yml
alerting:
  alertmanagers:
    - static_configs:
        - targets: ['alertmanager:9093']
```

Configure routes:

```yaml
# alertmanager.yml
route:
  receiver: 'slack'
  routes:
    - match:
        severity: critical
      receiver: 'pagerduty'

receivers:
  - name: 'slack'
    slack_configs:
      - api_url: 'https://hooks.slack.com/services/...'
        channel: '#alerts'
  - name: 'pagerduty'
    pagerduty_configs:
      - service_key: '...'
```

## Troubleshooting

### No Metrics from SGLang

1. Check server is running: `curl http://localhost:30000/health`
2. Verify metrics endpoint: `curl http://localhost:30000/metrics`
3. Check Prometheus targets: http://localhost:9090/targets

### Grafana Dashboard Empty

1. Verify Prometheus datasource in Grafana
2. Check time range (default: last 1 hour)
3. Ensure metrics are being collected

### High GPU Memory but Low Utilization

This is normal during batch processing or when queue is empty. GPU memory is pre-allocated for KV cache.
