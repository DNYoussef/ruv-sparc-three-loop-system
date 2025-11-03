# Example 2: Advanced Neural Cluster Management & Scaling

## Scenario Overview

**Objective**: Build a production-grade neural cluster with dynamic scaling, health monitoring, and fault tolerance for continuous model training.

**Use Case**: A machine learning platform needs to support multiple concurrent training jobs across diverse models (image classification, NLP, time series) with auto-scaling based on workload.

**Requirements**:
- Support 3+ simultaneous training jobs
- Auto-scale workers from 4 to 32 based on demand
- 99.9% uptime with automatic failover
- Real-time health monitoring & alerts
- Cost optimization with spot instance support

---

## Architecture Design

```
┌───────────────────────────────────────────────────────────────┐
│              Cluster Orchestration Layer                       │
│   (Queen Coordinator + DAA Autonomous Management)             │
└─────────────────┬─────────────────────────────────────────────┘
                  │
     ┌────────────┼────────────┬────────────┐
     │            │            │            │
┌────▼───┐  ┌────▼───┐  ┌────▼───┐  ┌────▼───┐
│Training│  │Training│  │Training│  │Training│
│ Job 1  │  │ Job 2  │  │ Job 3  │  │ Job 4  │
│(ResNet)│  │ (BERT) │  │(LSTM)  │  │(GAN)   │
└────┬───┘  └────┬───┘  └────┬───┘  └────┬───┘
     │            │            │            │
     │       ┌────┴────────────┴────────────┤
     │       │                              │
┌────▼───────▼─────────────────────────────▼────┐
│          Worker Pool (Auto-Scaling)            │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐          │
│  │  W1  │ │  W2  │ │  W3  │ │  W4  │ (min: 4) │
│  └──────┘ └──────┘ └──────┘ └──────┘          │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐          │
│  │  W5  │ │  W6  │ │  W7  │ │  W8  │          │
│  └──────┘ └──────┘ └──────┘ └──────┘          │
│        ... (scale up to 32 workers)            │
└────────────────────────────────────────────────┘
     │            │            │            │
┌────▼────┐  ┌───▼────┐  ┌───▼────┐  ┌───▼────┐
│Parameter│  │Health  │  │Metrics │  │Backup  │
│Server 1 │  │Monitor │  │Collector│ │Manager │
└─────────┘  └────────┘  └────────┘  └────────┘
```

**Key Components**:
1. **Orchestration Layer**: DAA-powered autonomous cluster management
2. **Multi-Tenant Training**: Concurrent isolated training jobs
3. **Auto-Scaling Pool**: Dynamic worker allocation (4-32 nodes)
4. **Health Monitor**: Real-time node health checks & failover
5. **Metrics Collector**: Prometheus-compatible metrics export

---

## Step-by-Step Walkthrough

### Phase 1: Production Cluster Setup (5 minutes)

**Step 1.1**: Initialize production-grade cluster with hierarchical topology

```bash
# Create hierarchical cluster for large-scale coordination
npx flow-nexus@latest neural cluster init \
  --name "ml-platform-prod-cluster" \
  --topology hierarchical \
  --architecture hybrid \
  --daa-enabled true \
  --consensus raft \
  --wasm-optimization true \
  --auto-scaling true \
  --min-workers 4 \
  --max-workers 32 \
  --health-check-interval 30
```

**Expected Output**:
```json
{
  "cluster_id": "cls_prod_xyz123",
  "name": "ml-platform-prod-cluster",
  "topology": "hierarchical",
  "status": "initializing",
  "config": {
    "architecture": "hybrid",
    "daa_enabled": true,
    "consensus": "raft",
    "auto_scaling": {
      "enabled": true,
      "min_workers": 4,
      "max_workers": 32,
      "scale_up_threshold": 0.8,
      "scale_down_threshold": 0.3
    },
    "health_monitoring": {
      "enabled": true,
      "interval_seconds": 30,
      "failure_threshold": 3,
      "auto_heal": true
    }
  },
  "tier": "production",
  "sla": "99.9% uptime"
}
```

**Why Hierarchical?**
- Supports 32+ workers efficiently (mesh becomes inefficient at scale)
- Leader-based coordination via Raft consensus
- Fault-tolerant leader election
- Optimized for multi-tenant workloads

---

**Step 1.2**: Deploy initial worker pool (minimum viable cluster)

```bash
# Deploy 4 baseline workers for immediate availability
npx flow-nexus@latest neural cluster scale \
  --cluster-id cls_prod_xyz123 \
  --target-workers 4 \
  --wait true
```

**Expected Output**:
```json
{
  "cluster_id": "cls_prod_xyz123",
  "scaling_operation": "scale_to_4",
  "status": "in_progress",
  "current_workers": 0,
  "target_workers": 4,
  "nodes": [
    {
      "node_id": "node_w1_aaa111",
      "status": "provisioning",
      "sandbox_id": "sbx_e2b_001",
      "role": "worker"
    },
    {
      "node_id": "node_w2_bbb222",
      "status": "provisioning",
      "sandbox_id": "sbx_e2b_002",
      "role": "worker"
    },
    {
      "node_id": "node_w3_ccc333",
      "status": "provisioning",
      "sandbox_id": "sbx_e2b_003",
      "role": "worker"
    },
    {
      "node_id": "node_w4_ddd444",
      "status": "provisioning",
      "sandbox_id": "sbx_e2b_004",
      "role": "worker"
    }
  ],
  "estimated_completion": "2m 30s"
}
```

---

**Step 1.3**: Deploy supporting infrastructure

```bash
# Deploy redundant parameter servers (active-passive for HA)
npx flow-nexus@latest neural node deploy \
  --cluster-id cls_prod_xyz123 \
  --node-type parameter_server \
  --role aggregator \
  --model large \
  --autonomy 0.95 \
  --capabilities aggregation,synchronization,checkpointing \
  --replicas 2 \
  --ha-mode active-passive

# Deploy health monitor
npx flow-nexus@latest neural node deploy \
  --cluster-id cls_prod_xyz123 \
  --node-type validator \
  --role health-monitor \
  --autonomy 0.9 \
  --capabilities health-checks,failover,alerting

# Deploy metrics collector
npx flow-nexus@latest neural node deploy \
  --cluster-id cls_prod_xyz123 \
  --node-type coordinator \
  --role metrics-collector \
  --autonomy 0.85 \
  --capabilities metrics,prometheus-export,logging
```

---

### Phase 2: Multi-Tenant Training Jobs (10 minutes)

**Step 2.1**: Submit concurrent training jobs

**Job 1: Image Classification (ResNet-50)**
```bash
npx flow-nexus@latest neural train distributed \
  --cluster-id cls_prod_xyz123 \
  --job-name "image-classification-resnet50" \
  --dataset gs://ml-platform/datasets/imagenet-subset \
  --architecture cnn \
  --model resnet50 \
  --epochs 100 \
  --batch-size 64 \
  --learning-rate 0.01 \
  --optimizer sgd \
  --workers 8 \
  --priority high \
  --gpu-required true
```

**Job 2: NLP Sentiment Analysis (BERT-base)**
```bash
npx flow-nexus@latest neural train distributed \
  --cluster-id cls_prod_xyz123 \
  --job-name "nlp-sentiment-bert" \
  --dataset gs://ml-platform/datasets/imdb-reviews \
  --architecture transformer \
  --model bert-base \
  --epochs 50 \
  --batch-size 32 \
  --learning-rate 0.0001 \
  --optimizer adam \
  --workers 6 \
  --priority medium
```

**Job 3: Time Series Forecasting (LSTM)**
```bash
npx flow-nexus@latest neural train distributed \
  --cluster-id cls_prod_xyz123 \
  --job-name "timeseries-lstm-forecast" \
  --dataset gs://ml-platform/datasets/stock-prices \
  --architecture rnn \
  --model lstm \
  --epochs 200 \
  --batch-size 16 \
  --learning-rate 0.001 \
  --optimizer adam \
  --workers 4 \
  --priority low
```

**Expected Output (Cluster Status)**:
```json
{
  "cluster_id": "cls_prod_xyz123",
  "status": "running",
  "workers": {
    "total": 18,
    "available": 0,
    "allocated": 18
  },
  "auto_scaling": {
    "triggered": true,
    "reason": "utilization > 80%",
    "action": "scaling_up",
    "target_workers": 24
  },
  "training_jobs": [
    {
      "job_id": "job_resnet_001",
      "name": "image-classification-resnet50",
      "status": "training",
      "workers_allocated": 8,
      "epoch": 12,
      "total_epochs": 100,
      "current_loss": 1.82,
      "priority": "high"
    },
    {
      "job_id": "job_bert_002",
      "name": "nlp-sentiment-bert",
      "status": "training",
      "workers_allocated": 6,
      "epoch": 8,
      "total_epochs": 50,
      "current_accuracy": 0.73,
      "priority": "medium"
    },
    {
      "job_id": "job_lstm_003",
      "name": "timeseries-lstm-forecast",
      "status": "queued",
      "workers_allocated": 0,
      "message": "Waiting for auto-scaling to complete",
      "priority": "low"
    }
  ]
}
```

**What Happens**:
- Job 1 (high priority) gets 8 workers immediately
- Job 2 (medium priority) gets 6 workers (18 total used)
- Job 3 (low priority) queued, triggers auto-scaling to 24 workers
- DAA orchestrator manages resource allocation based on priority

---

### Phase 3: Auto-Scaling in Action (3 minutes)

**Step 3.1**: Monitor auto-scaling event

```bash
# Watch cluster scale up in real-time
npx flow-nexus@latest neural cluster status \
  --cluster-id cls_prod_xyz123 \
  --watch true \
  --interval 10
```

**Expected Output (Time: 0s)**:
```json
{
  "cluster_id": "cls_prod_xyz123",
  "auto_scaling_event": {
    "event_id": "scale_evt_123",
    "timestamp": "2025-11-02T15:10:00Z",
    "trigger": "utilization > 80%",
    "action": "scale_up",
    "current_workers": 18,
    "target_workers": 24,
    "status": "provisioning",
    "nodes_provisioning": [
      "node_w19_eee555",
      "node_w20_fff666",
      "node_w21_ggg777",
      "node_w22_hhh888",
      "node_w23_iii999",
      "node_w24_jjj000"
    ]
  }
}
```

**Expected Output (Time: 2m 15s)**:
```json
{
  "cluster_id": "cls_prod_xyz123",
  "auto_scaling_event": {
    "event_id": "scale_evt_123",
    "timestamp": "2025-11-02T15:12:15Z",
    "action": "scale_up",
    "current_workers": 24,
    "target_workers": 24,
    "status": "completed",
    "nodes_ready": 6,
    "duration": "2m 15s"
  },
  "training_jobs": [
    {
      "job_id": "job_lstm_003",
      "name": "timeseries-lstm-forecast",
      "status": "training",
      "workers_allocated": 4,
      "epoch": 1,
      "message": "Training started after scale-up"
    }
  ]
}
```

**What Happens**:
- Auto-scaler detects high utilization (18/18 workers allocated)
- Provisions 6 additional workers (18 → 24)
- Job 3 automatically starts training once new workers are ready
- Total cluster utilization: 18/24 (75%, within optimal range)

---

**Step 3.2**: Trigger scale-down after job completion

```bash
# Simulate Job 1 completion (or wait for actual completion)
# Auto-scaler will detect idle workers and scale down

# Check cluster status 10 minutes after Job 1 finishes
npx flow-nexus@latest neural cluster status \
  --cluster-id cls_prod_xyz123
```

**Expected Output**:
```json
{
  "cluster_id": "cls_prod_xyz123",
  "workers": {
    "total": 12,
    "available": 2,
    "allocated": 10
  },
  "auto_scaling_event": {
    "event_id": "scale_evt_124",
    "timestamp": "2025-11-02T15:25:00Z",
    "trigger": "utilization < 30% for 10 minutes",
    "action": "scale_down",
    "current_workers": 24,
    "target_workers": 12,
    "status": "completed",
    "nodes_terminated": 12,
    "cost_savings": "$14.40/hour"
  },
  "training_jobs": [
    {
      "job_id": "job_resnet_001",
      "status": "completed",
      "final_accuracy": 0.89
    },
    {
      "job_id": "job_bert_002",
      "status": "training",
      "workers_allocated": 6
    },
    {
      "job_id": "job_lstm_003",
      "status": "training",
      "workers_allocated": 4
    }
  ]
}
```

**What Happens**:
- Job 1 completes, freeing 8 workers
- Cluster utilization drops to 10/24 (41%)
- After 10-minute cooldown period, auto-scaler triggers scale-down
- Cluster scales down to 12 workers (50% reduction)
- $14.40/hour cost savings while maintaining active jobs

---

### Phase 4: Health Monitoring & Fault Tolerance (Continuous)

**Step 4.1**: Simulate worker failure

```bash
# Manually terminate a worker to test failover
npx flow-nexus@latest sandbox stop \
  --sandbox-id sbx_e2b_002 \
  --force true
```

**Expected Output (Health Monitor)**:
```json
{
  "cluster_id": "cls_prod_xyz123",
  "health_event": {
    "event_id": "health_evt_456",
    "timestamp": "2025-11-02T15:30:00Z",
    "type": "worker_failure",
    "node_id": "node_w2_bbb222",
    "sandbox_id": "sbx_e2b_002",
    "failure_reason": "sandbox_terminated",
    "detection_time": "15s",
    "actions_taken": [
      "Mark node as unhealthy",
      "Reallocate training tasks to healthy workers",
      "Provision replacement worker",
      "Restore training state from last checkpoint"
    ]
  },
  "recovery_status": {
    "replacement_node": "node_w25_kkk111",
    "sandbox_id": "sbx_e2b_025",
    "status": "provisioned",
    "training_resumed": true,
    "data_loss": "0 epochs (checkpoint at epoch 8)"
  }
}
```

**What Happens**:
1. Health monitor detects worker failure within 30 seconds
2. Training tasks redistributed to healthy workers
3. New replacement worker auto-provisioned
4. Training state restored from last checkpoint (epoch 8)
5. Zero data loss, minimal downtime (<2 minutes)

---

**Step 4.2**: Configure alerting & monitoring

```bash
# Set up Prometheus metrics endpoint
npx flow-nexus@latest neural cluster configure \
  --cluster-id cls_prod_xyz123 \
  --enable-metrics true \
  --metrics-endpoint "http://prometheus.example.com:9090" \
  --alert-webhook "https://alerts.example.com/webhook"

# Configure alert rules
npx flow-nexus@latest neural cluster alerts \
  --cluster-id cls_prod_xyz123 \
  --add-rule "worker_failure" \
  --condition "node_status == 'unhealthy'" \
  --severity critical \
  --notification slack,email

npx flow-nexus@latest neural cluster alerts \
  --cluster-id cls_prod_xyz123 \
  --add-rule "high_utilization" \
  --condition "worker_utilization > 90%" \
  --severity warning \
  --notification slack
```

**Metrics Exported**:
```
# Worker utilization
flow_nexus_worker_utilization_percent{cluster="cls_prod_xyz123"} 75.0

# Training job metrics
flow_nexus_training_job_epoch{job="job_bert_002"} 15
flow_nexus_training_job_loss{job="job_bert_002"} 0.42
flow_nexus_training_job_accuracy{job="job_bert_002"} 0.81

# Cluster health
flow_nexus_cluster_workers_total{cluster="cls_prod_xyz123"} 12
flow_nexus_cluster_workers_healthy{cluster="cls_prod_xyz123"} 11
flow_nexus_cluster_workers_unhealthy{cluster="cls_prod_xyz123"} 1

# Auto-scaling events
flow_nexus_autoscaling_events_total{cluster="cls_prod_xyz123",action="scale_up"} 3
flow_nexus_autoscaling_events_total{cluster="cls_prod_xyz123",action="scale_down"} 2

# Cost metrics
flow_nexus_cluster_cost_per_hour{cluster="cls_prod_xyz123"} 28.80
```

---

## Complete Code Example

**Production Cluster Management Script** (`manage_cluster.sh`):

```bash
#!/bin/bash
set -euo pipefail

# Configuration
CLUSTER_NAME="ml-platform-prod-cluster"
MIN_WORKERS=4
MAX_WORKERS=32
HEALTH_CHECK_INTERVAL=30
METRICS_ENDPOINT="http://prometheus.internal:9090"
ALERT_WEBHOOK="https://alerts.internal/webhook"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
  echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
  echo -e "${RED}[ERROR]${NC} $1" >&2
}

warn() {
  echo -e "${YELLOW}[WARN]${NC} $1"
}

# Function: Initialize cluster
init_cluster() {
  log "🚀 Initializing production neural cluster..."

  CLUSTER_ID=$(npx flow-nexus@latest neural cluster init \
    --name "$CLUSTER_NAME" \
    --topology hierarchical \
    --architecture hybrid \
    --daa-enabled true \
    --consensus raft \
    --wasm-optimization true \
    --auto-scaling true \
    --min-workers $MIN_WORKERS \
    --max-workers $MAX_WORKERS \
    --health-check-interval $HEALTH_CHECK_INTERVAL \
    --format json | jq -r '.cluster_id')

  if [ -z "$CLUSTER_ID" ]; then
    error "Failed to create cluster"
    exit 1
  fi

  log "✅ Cluster created: $CLUSTER_ID"
  echo "$CLUSTER_ID" > .cluster_id
}

# Function: Deploy infrastructure
deploy_infrastructure() {
  local CLUSTER_ID=$1
  log "🔧 Deploying cluster infrastructure..."

  # Initial worker pool
  log "  Scaling to $MIN_WORKERS workers..."
  npx flow-nexus@latest neural cluster scale \
    --cluster-id "$CLUSTER_ID" \
    --target-workers $MIN_WORKERS \
    --wait true

  # Parameter servers (HA)
  log "  Deploying redundant parameter servers..."
  npx flow-nexus@latest neural node deploy \
    --cluster-id "$CLUSTER_ID" \
    --node-type parameter_server \
    --role aggregator \
    --model large \
    --autonomy 0.95 \
    --capabilities aggregation,synchronization,checkpointing \
    --replicas 2 \
    --ha-mode active-passive

  # Health monitor
  log "  Deploying health monitor..."
  npx flow-nexus@latest neural node deploy \
    --cluster-id "$CLUSTER_ID" \
    --node-type validator \
    --role health-monitor \
    --autonomy 0.9 \
    --capabilities health-checks,failover,alerting

  # Metrics collector
  log "  Deploying metrics collector..."
  npx flow-nexus@latest neural node deploy \
    --cluster-id "$CLUSTER_ID" \
    --node-type coordinator \
    --role metrics-collector \
    --autonomy 0.85 \
    --capabilities metrics,prometheus-export,logging

  log "✅ Infrastructure deployed"
}

# Function: Configure monitoring
configure_monitoring() {
  local CLUSTER_ID=$1
  log "📊 Configuring monitoring & alerting..."

  # Enable Prometheus metrics
  npx flow-nexus@latest neural cluster configure \
    --cluster-id "$CLUSTER_ID" \
    --enable-metrics true \
    --metrics-endpoint "$METRICS_ENDPOINT" \
    --alert-webhook "$ALERT_WEBHOOK"

  # Alert rules
  npx flow-nexus@latest neural cluster alerts \
    --cluster-id "$CLUSTER_ID" \
    --add-rule "worker_failure" \
    --condition "node_status == 'unhealthy'" \
    --severity critical \
    --notification slack,email

  npx flow-nexus@latest neural cluster alerts \
    --cluster-id "$CLUSTER_ID" \
    --add-rule "high_utilization" \
    --condition "worker_utilization > 90%" \
    --severity warning \
    --notification slack

  log "✅ Monitoring configured"
}

# Function: Submit training job
submit_job() {
  local CLUSTER_ID=$1
  local JOB_NAME=$2
  local DATASET=$3
  local ARCHITECTURE=$4
  local EPOCHS=$5
  local WORKERS=$6

  log "🎓 Submitting training job: $JOB_NAME"

  JOB_ID=$(npx flow-nexus@latest neural train distributed \
    --cluster-id "$CLUSTER_ID" \
    --job-name "$JOB_NAME" \
    --dataset "$DATASET" \
    --architecture "$ARCHITECTURE" \
    --epochs "$EPOCHS" \
    --workers "$WORKERS" \
    --format json | jq -r '.job_id')

  log "✅ Job submitted: $JOB_ID"
  echo "$JOB_ID"
}

# Function: Monitor cluster health
monitor_health() {
  local CLUSTER_ID=$1

  log "🩺 Starting health monitoring (press Ctrl+C to stop)..."

  while true; do
    STATUS=$(npx flow-nexus@latest neural cluster status \
      --cluster-id "$CLUSTER_ID" \
      --format json)

    TOTAL_WORKERS=$(echo "$STATUS" | jq -r '.workers.total')
    HEALTHY_WORKERS=$(echo "$STATUS" | jq -r '.workers.healthy // .workers.total')
    UTILIZATION=$(echo "$STATUS" | jq -r '.workers.utilization_percent // 0')

    if [ "$HEALTHY_WORKERS" -lt "$TOTAL_WORKERS" ]; then
      warn "Unhealthy workers detected: $HEALTHY_WORKERS/$TOTAL_WORKERS"
    fi

    log "Workers: $HEALTHY_WORKERS/$TOTAL_WORKERS healthy | Utilization: $UTILIZATION%"

    sleep $HEALTH_CHECK_INTERVAL
  done
}

# Function: Cleanup cluster
cleanup() {
  local CLUSTER_ID=$1
  log "🧹 Terminating cluster..."

  npx flow-nexus@latest neural cluster terminate \
    --cluster-id "$CLUSTER_ID" \
    --force true

  rm -f .cluster_id
  log "✅ Cluster terminated"
}

# Main script
main() {
  case "${1:-}" in
    init)
      init_cluster
      ;;
    deploy)
      CLUSTER_ID=$(cat .cluster_id 2>/dev/null || echo "")
      if [ -z "$CLUSTER_ID" ]; then
        error "No cluster ID found. Run 'init' first."
        exit 1
      fi
      deploy_infrastructure "$CLUSTER_ID"
      configure_monitoring "$CLUSTER_ID"
      ;;
    submit)
      CLUSTER_ID=$(cat .cluster_id 2>/dev/null || echo "")
      submit_job "$CLUSTER_ID" "${2:-}" "${3:-}" "${4:-}" "${5:-}" "${6:-}"
      ;;
    monitor)
      CLUSTER_ID=$(cat .cluster_id 2>/dev/null || echo "")
      monitor_health "$CLUSTER_ID"
      ;;
    cleanup)
      CLUSTER_ID=$(cat .cluster_id 2>/dev/null || echo "")
      cleanup "$CLUSTER_ID"
      ;;
    *)
      echo "Usage: $0 {init|deploy|submit|monitor|cleanup}"
      exit 1
      ;;
  esac
}

main "$@"
```

**Usage Examples**:
```bash
# Initialize cluster
./manage_cluster.sh init

# Deploy infrastructure
./manage_cluster.sh deploy

# Submit training job
./manage_cluster.sh submit \
  "image-classifier" \
  "gs://datasets/imagenet" \
  "cnn" \
  "100" \
  "8"

# Monitor cluster health
./manage_cluster.sh monitor

# Cleanup
./manage_cluster.sh cleanup
```

---

## Outcomes & Results

### Cluster Performance Metrics

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| **Uptime SLA** | 99.9% | 99.95% | ✅ Exceeded |
| **Auto-Scale Speed** | <5 min | 2m 15s | ✅ Met |
| **Failover Time** | <5 min | 1m 45s | ✅ Met |
| **Max Concurrent Jobs** | 3+ | 4 | ✅ Exceeded |
| **Worker Utilization** | 70-85% | 78% | ✅ Optimal |

### Cost Optimization

- **Baseline Cost** (24 workers, 24/7): $172.80/day
- **Auto-Scaled Cost** (avg 14 workers): $100.80/day
- **Cost Savings**: $72.00/day (41.7% reduction)
- **Monthly Savings**: $2,160

### Scalability Achievements

- **Peak Workload**: 32 workers, 4 concurrent jobs
- **Scale-Up Events**: 5 events, avg 2m 20s duration
- **Scale-Down Events**: 3 events, avg 1m 10s duration
- **Worker Failures**: 2 failures, 100% auto-recovery

---

## Tips & Best Practices

### 1. Auto-Scaling Configuration

**Optimal Thresholds**:
```bash
--scale-up-threshold 0.8    # Scale up when 80% utilized
--scale-down-threshold 0.3  # Scale down when <30% utilized
--cooldown-period 600       # 10-minute cooldown before scaling again
```

### 2. Consensus Algorithm Selection

| Consensus | Best For | Pros | Cons |
|-----------|----------|------|------|
| **Raft** | Production clusters | Strong consistency, leader election | Single leader bottleneck |
| **Gossip** | Large clusters (>50 nodes) | Decentralized, scalable | Eventual consistency |
| **Byzantine** | Security-critical | Fault-tolerant against malicious nodes | High overhead |
| **Proof-of-Learning** | Federated learning | Merit-based consensus | Complex verification |

### 3. Health Check Best Practices

**Configure aggressive health checks**:
```bash
--health-check-interval 30  # Check every 30 seconds
--failure-threshold 3       # Mark unhealthy after 3 failed checks
--auto-heal true            # Auto-replace unhealthy nodes
```

### 4. Cost Optimization Strategies

**Use spot instances for non-critical jobs**:
```bash
--worker-instance-type spot
--spot-max-price 0.05
--fallback-to-on-demand true
```

**Schedule batch jobs during off-peak hours**:
```bash
--schedule "0 2 * * *"  # Run at 2 AM daily
--max-duration 4h       # Complete within 4 hours
```

### 5. Multi-Tenancy Isolation

**Resource quotas per job**:
```bash
--max-workers-per-job 12
--max-cpu-per-job 48
--max-memory-per-job 96GB
```

### 6. Disaster Recovery

**Regular checkpointing**:
```bash
--checkpoint-interval 600  # Checkpoint every 10 minutes
--checkpoint-storage gs://backups/checkpoints
--max-checkpoints 5        # Keep last 5 checkpoints
```

---

## Next Steps

1. **Advanced Monitoring**: Integrate Grafana dashboards
2. **Multi-Region**: Deploy across AWS + GCP for redundancy
3. **GPU Acceleration**: Enable GPU workers for deep learning
4. **Custom Topologies**: Build hybrid hierarchical-mesh topologies
5. **CI/CD Integration**: Automate model training in pipelines

---

## Related Resources

- [Example 1: Distributed Training Basics](./example-1-distributed-training.md)
- [Example 3: Model Deployment & Serving](./example-3-model-deployment.md)
- [Flow Nexus Auto-Scaling Guide](https://flow-nexus.ruv.io/docs/auto-scaling)
- [E2B Sandbox Pricing](https://e2b.dev/pricing)
