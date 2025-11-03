---
name: "Deployment Readiness"
description: "Production deployment validation for Deep Research SOP Pipeline H ensuring models ready for real-world deployment. Use before deploying to production, creating deployment plans, or validating infrastructure requirements. Validates performance benchmarks, monitoring setup, incident response plans, rollback strategies, and infrastructure scalability for Quality Gate 3."
---

# Deployment Readiness

Validate ML models and systems for production deployment, ensuring operational readiness across performance, monitoring, security, and incident management dimensions.

## Overview

**Purpose**: Validate production deployment readiness

**When to Use**:
- Before deploying models to production (Phase 3)
- Quality Gate 3 validation required
- Creating deployment plans
- Infrastructure capacity planning
- Production environment testing

**Quality Gate**: Required for Quality Gate 3 APPROVED status

**Prerequisites**:
- Model trained and evaluated (Gate 2 APPROVED)
- Reproducibility audit passed
- Production environment available for testing
- Infrastructure requirements documented

**Outputs**:
- Deployment readiness report (PASS/FAIL)
- Infrastructure requirements specification
- Monitoring plan with alerts and dashboards
- Incident response plan
- Rollback strategy
- Performance benchmarks (production environment)
- Deployment checklist

**Time Estimate**: 1-2 weeks
- Infrastructure setup: 2-3 days
- Performance benchmarking: 1-2 days
- Monitoring setup: 2-3 days
- Security validation: 1-2 days
- Documentation: 1-2 days

**Agents Used**: tester, archivist

---

## Quick Start

### 1. Infrastructure Requirements
```yaml
# deployment/infrastructure_requirements.yaml

compute:
  gpu:
    type: "NVIDIA A100"
    count: 2
    memory: "80GB each"
  cpu:
    cores: 32
    memory: "256GB"

storage:
  model_weights: "50GB"
  datasets: "500GB"
  logs: "100GB"

network:
  ingress_bandwidth: "10Gbps"
  egress_bandwidth: "10Gbps"
  latency_target: "<100ms p95"

scalability:
  min_instances: 2
  max_instances: 10
  autoscaling_metric: "requests_per_second"
  target_utilization: 70%
```

### 2. Performance Benchmarking
```bash
# Benchmark in production environment
python scripts/production_benchmarks.py \
  --model deployment/model.pth \
  --environment production \
  --metrics "latency,throughput,memory,cpu" \
  --duration 3600 \
  --output deployment/benchmarks.json
```

### 3. Monitoring Setup
```bash
# Deploy monitoring stack (Prometheus + Grafana)
docker-compose -f deployment/monitoring/docker-compose.yml up -d

# Configure alerts
kubectl apply -f deployment/monitoring/alerts.yaml

# Test alert pipeline
python scripts/test_alerts.py --alert-manager http://localhost:9093
```

### 4. Deployment Plan
```bash
# Generate deployment plan
python scripts/generate_deployment_plan.py \
  --model deployment/model.pth \
  --infrastructure deployment/infrastructure_requirements.yaml \
  --output deployment/deployment_plan.md
```

### 5. Validate Deployment Readiness
```bash
# Run comprehensive readiness checks
python scripts/validate_deployment_readiness.py \
  --deployment-plan deployment/deployment_plan.md \
  --benchmarks deployment/benchmarks.json \
  --monitoring-config deployment/monitoring/ \
  --output deployment/readiness_report.md
```

---

## Detailed Instructions

### Phase 1: Infrastructure Validation (2-3 days)

**Objective**: Validate production infrastructure meets requirements

**Steps**:

#### 1.1 Capacity Planning
```python
# scripts/capacity_planning.py

def estimate_capacity_requirements(model, workload):
    """Estimate infrastructure requirements."""
    # GPU requirements
    gpu_memory_per_batch = estimate_gpu_memory(model, batch_size=32)
    num_gpus = math.ceil(gpu_memory_per_batch * target_throughput / gpu_capacity)

    # CPU requirements
    cpu_cores = estimate_cpu_usage(model, workload)

    # Storage requirements
    storage_model = model_size_gb
    storage_data = dataset_size_gb
    storage_logs = estimated_logs_per_day_gb * retention_days

    return {
        "gpu": {"count": num_gpus, "memory_per_gpu": gpu_capacity},
        "cpu": {"cores": cpu_cores},
        "storage": {
            "total": storage_model + storage_data + storage_logs
        }
    }

# Run capacity planning
requirements = estimate_capacity_requirements(model, expected_workload)
print(f"Infrastructure Requirements: {requirements}")
```

**Deliverable**: Infrastructure requirements specification

---

#### 1.2 Environment Setup
```bash
# Setup production environment
# Using Kubernetes for orchestration

# 1. Create namespace
kubectl create namespace ml-production

# 2. Deploy model serving (TorchServe, TensorFlow Serving, or custom)
kubectl apply -f deployment/kubernetes/model-serving.yaml

# 3. Deploy load balancer
kubectl apply -f deployment/kubernetes/load-balancer.yaml

# 4. Verify deployment
kubectl get pods -n ml-production
kubectl get services -n ml-production
```

**Deliverable**: Production environment deployed

---

### Phase 2: Performance Benchmarking (1-2 days)

**Objective**: Measure performance in production environment

**Steps**:

#### 2.1 Latency Benchmarking
```python
# scripts/benchmark_latency.py
import time
import numpy as np

def benchmark_latency(model, test_inputs, num_runs=1000):
    """Benchmark inference latency."""
    latencies = []

    for _ in range(num_runs):
        start = time.perf_counter()
        output = model(test_inputs)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # Convert to ms

    results = {
        "mean": np.mean(latencies),
        "std": np.std(latencies),
        "p50": np.percentile(latencies, 50),
        "p95": np.percentile(latencies, 95),
        "p99": np.percentile(latencies, 99)
    }

    print(f"Latency Results (ms):")
    print(f"  Mean: {results['mean']:.2f}")
    print(f"  P50: {results['p50']:.2f}")
    print(f"  P95: {results['p95']:.2f}")
    print(f"  P99: {results['p99']:.2f}")

    # Check against SLA (e.g., P95 < 100ms)
    sla_p95 = 100.0
    if results['p95'] > sla_p95:
        print(f"⚠️  WARNING: P95 latency {results['p95']:.2f}ms exceeds SLA {sla_p95}ms")
        return False
    else:
        print(f"✅ PASS: P95 latency {results['p95']:.2f}ms within SLA")
        return True

# Run benchmark
benchmark_latency(model, test_inputs)
```

**Deliverable**: Latency benchmarks

---

#### 2.2 Throughput Benchmarking
```python
# scripts/benchmark_throughput.py

def benchmark_throughput(model, duration_seconds=3600):
    """Benchmark queries per second (QPS)."""
    start_time = time.time()
    requests_processed = 0

    while time.time() - start_time < duration_seconds:
        # Simulate request
        output = model(test_input)
        requests_processed += 1

    elapsed = time.time() - start_time
    qps = requests_processed / elapsed

    print(f"Throughput: {qps:.2f} QPS")

    # Check against target (e.g., 100 QPS)
    target_qps = 100.0
    if qps < target_qps:
        print(f"⚠️  WARNING: Throughput {qps:.2f} QPS below target {target_qps}")
        return False
    else:
        print(f"✅ PASS: Throughput {qps:.2f} QPS meets target")
        return True

# Run benchmark
benchmark_throughput(model)
```

**Deliverable**: Throughput benchmarks

---

#### 2.3 Resource Utilization
```bash
# Monitor GPU/CPU/Memory utilization during load test
# Using NVIDIA SMI for GPUs
nvidia-smi dmon -s pucvmet -c 3600 > deployment/gpu_utilization.log &

# Using psutil for CPU/Memory
python scripts/monitor_resources.py --duration 3600 --output deployment/resource_utilization.json &

# Run load test
python scripts/load_test.py --requests-per-second 100 --duration 3600

# Analyze utilization
python scripts/analyze_utilization.py \
  --gpu deployment/gpu_utilization.log \
  --cpu deployment/resource_utilization.json \
  --target-utilization 70 \
  --output deployment/utilization_report.md
```

**Deliverable**: Resource utilization report

---

### Phase 3: Monitoring & Observability (2-3 days)

**Objective**: Set up comprehensive monitoring

**Steps**:

#### 3.1 Metrics Collection
```yaml
# deployment/monitoring/prometheus.yml

global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'model-serving'
    static_configs:
      - targets: ['model-serving:8080']

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']

  - job_name: 'gpu-exporter'
    static_configs:
      - targets: ['dcgm-exporter:9400']
```

**Key Metrics**:
- **Inference Metrics**: Latency (P50, P95, P99), throughput (QPS), error rate
- **Resource Metrics**: GPU utilization, CPU utilization, memory usage
- **Business Metrics**: Requests per user, predictions per day, model drift

---

#### 3.2 Alerting
```yaml
# deployment/monitoring/alerts.yaml

groups:
  - name: model_serving_alerts
    interval: 30s
    rules:
      # High latency alert
      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(inference_duration_seconds_bucket[5m])) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High inference latency"
          description: "P95 latency {{ $value }}s exceeds 100ms threshold"

      # Low throughput alert
      - alert: LowThroughput
        expr: rate(inference_requests_total[5m]) < 50
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Low throughput"
          description: "QPS {{ $value }} below 50 threshold"

      # High error rate alert
      - alert: HighErrorRate
        expr: rate(inference_errors_total[5m]) / rate(inference_requests_total[5m]) > 0.05
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "High error rate"
          description: "Error rate {{ $value | humanizePercentage }} exceeds 5%"

      # GPU out of memory alert
      - alert: GPUOutOfMemory
        expr: DCGM_FI_DEV_FB_FREE / DCGM_FI_DEV_FB_USED < 0.1
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "GPU out of memory"
          description: "GPU memory usage > 90%"
```

**Deliverable**: Alerting configuration

---

#### 3.3 Dashboards
```json
// deployment/monitoring/grafana_dashboard.json

{
  "dashboard": {
    "title": "ML Model Production Monitoring",
    "panels": [
      {
        "title": "Inference Latency (P95)",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(inference_duration_seconds_bucket[5m]))"
          }
        ]
      },
      {
        "title": "Requests Per Second",
        "targets": [
          {
            "expr": "rate(inference_requests_total[1m])"
          }
        ]
      },
      {
        "title": "Error Rate",
        "targets": [
          {
            "expr": "rate(inference_errors_total[5m]) / rate(inference_requests_total[5m])"
          }
        ]
      },
      {
        "title": "GPU Utilization",
        "targets": [
          {
            "expr": "DCGM_FI_DEV_GPU_UTIL"
          }
        ]
      }
    ]
  }
}
```

**Deliverable**: Monitoring dashboards

---

### Phase 4: Incident Response (1-2 days)

**Objective**: Prepare incident response plan

**Steps**:

#### 4.1 Incident Response Plan
```markdown
# Incident Response Plan

## Severity Levels

### P0 - Critical (Production Down)
- **Response Time**: 15 minutes
- **Resolution Time**: 2 hours
- **Escalation**: Immediate page on-call engineer

### P1 - High (Degraded Performance)
- **Response Time**: 30 minutes
- **Resolution Time**: 4 hours
- **Escalation**: Email + Slack alert

### P2 - Medium (Minor Issues)
- **Response Time**: 2 hours
- **Resolution Time**: 24 hours
- **Escalation**: Create ticket

## Runbooks

### High Latency Runbook
1. Check current load (QPS)
2. Check GPU/CPU utilization
3. Scale up instances if utilization >80%
4. Check for model drift (retrain if needed)
5. Roll back to previous version if issue persists

### High Error Rate Runbook
1. Check error logs
2. Identify error type (input validation, OOM, model error)
3. If input validation: Update input schema
4. If OOM: Reduce batch size or add GPU
5. If model error: Roll back to previous version

### GPU Out of Memory Runbook
1. Reduce batch size
2. Enable gradient checkpointing
3. Use mixed precision (FP16)
4. Scale up to larger GPU (A100 80GB)
5. Implement model parallelism
```

**Deliverable**: Incident response plan

---

#### 4.2 Rollback Strategy
```bash
# deployment/rollback.sh

#!/bin/bash
set -e

# Rollback strategy: Blue-Green Deployment

echo "Starting rollback to previous version..."

# 1. Verify previous version exists
if [ ! -f "deployment/previous_version.yaml" ]; then
    echo "ERROR: Previous version not found"
    exit 1
fi

# 2. Deploy previous version (green)
kubectl apply -f deployment/previous_version.yaml

# 3. Wait for deployment to be ready
kubectl wait --for=condition=available --timeout=300s deployment/model-serving-green

# 4. Switch traffic to green (previous version)
kubectl patch service model-serving -p '{"spec":{"selector":{"version":"green"}}}'

# 5. Verify rollback successful
python scripts/verify_deployment.py --expected-version green

# 6. Terminate blue (failed version)
kubectl delete deployment model-serving-blue

echo "✅ Rollback completed successfully"
```

**Deliverable**: Rollback strategy

---

### Phase 5: Security Validation (1-2 days)

**Objective**: Validate security posture

**Criteria**:

#### 5.1 Authentication & Authorization
- [ ] API requires authentication (API keys, OAuth)
- [ ] Role-based access control (RBAC) implemented
- [ ] Rate limiting configured (prevent abuse)

#### 5.2 Data Security
- [ ] Data encrypted in transit (TLS 1.3)
- [ ] Data encrypted at rest (AES-256)
- [ ] PII handling compliant with GDPR/HIPAA

#### 5.3 Model Security
- [ ] Model weights access controlled
- [ ] Adversarial input detection enabled
- [ ] Input validation implemented

#### 5.4 Infrastructure Security
- [ ] Network policies configured (Kubernetes NetworkPolicy)
- [ ] Container security scanning enabled (Trivy, Aqua)
- [ ] Secrets management (Vault, Kubernetes Secrets)

**Deliverable**: Security validation checklist

---

### Phase 6: Documentation (1-2 days)

**Objective**: Document deployment procedures

**Deliverables**:

#### 6.1 Deployment Checklist
```markdown
# Deployment Checklist

## Pre-Deployment
- [ ] Model trained and Gate 2 APPROVED
- [ ] Reproducibility audit passed
- [ ] Performance benchmarks meet SLA
- [ ] Monitoring configured and tested
- [ ] Alerts configured and tested
- [ ] Incident response plan documented
- [ ] Rollback strategy tested
- [ ] Security validation passed

## Deployment
- [ ] Deploy to staging environment
- [ ] Run smoke tests in staging
- [ ] Deploy to production (canary or blue-green)
- [ ] Monitor metrics for 24 hours
- [ ] Gradually ramp traffic (10% → 50% → 100%)

## Post-Deployment
- [ ] Verify all metrics within SLA
- [ ] Check error logs
- [ ] Confirm alerts working
- [ ] Update documentation
- [ ] Notify stakeholders
```

#### 6.2 Operations Manual
- Deployment procedures
- Scaling procedures
- Monitoring procedures
- Troubleshooting guide
- Runbooks for common issues

**Deliverable**: Complete deployment documentation

---

## Integration with Deep Research SOP

### Pipeline Integration
- **Pipeline H (Deployment Readiness)**: This skill validates production deployment readiness
- **Quality Gate 3**: Deployment readiness PASS required for Gate 3 APPROVED

### Agent Coordination
```
tester agent performs performance benchmarking and monitoring setup
  ↓
archivist agent documents deployment procedures
  ↓
evaluator agent validates Gate 3
```

---

## Troubleshooting

### Issue: High latency (>100ms P95)
**Solution**: Scale up instances, optimize model (quantization, pruning), use faster hardware

### Issue: Low throughput (<100 QPS)
**Solution**: Increase batch size, use model parallelism, optimize data loading

### Issue: Gate 3 validation fails
**Solution**: Ensure all deployment readiness criteria met (performance, monitoring, incident response)

---

## Related Skills and Commands

### Prerequisites
- `holistic-evaluation` - Performance evaluation complete
- `reproducibility-audit` - Reproducibility validated

### Next Steps
- `research-publication` - Academic publication
- `gate-validation --gate 3` - Gate 3 validation

---

## References

### Deployment Best Practices
- Google SRE Handbook
- AWS Well-Architected Framework
- Kubernetes Best Practices

### Monitoring Standards
- Prometheus Best Practices
- OpenTelemetry
- The Four Golden Signals (Latency, Traffic, Errors, Saturation)
