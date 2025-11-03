# Example 1: Distributed Neural Network Training with Flow Nexus

## Scenario Overview

**Objective**: Train a sentiment analysis model across multiple E2B sandboxes using distributed training with federated learning capabilities.

**Use Case**: A startup needs to train a custom NLP model on customer feedback data while maintaining data privacy and leveraging distributed compute resources.

**Requirements**:
- Train a transformer-based sentiment classifier
- Distribute training across 4 worker nodes
- Use federated learning to keep data localized
- Achieve 90%+ accuracy on validation set
- Complete training in under 30 minutes

---

## Architecture Design

```
┌─────────────────────────────────────────────────────────────┐
│                    Neural Cluster Manager                    │
│                  (Flow Nexus Orchestrator)                   │
└─────────────────────┬───────────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┬─────────────┐
        │             │             │             │
   ┌────▼───┐   ┌────▼───┐   ┌────▼───┐   ┌────▼───┐
   │Worker 1│   │Worker 2│   │Worker 3│   │Worker 4│
   │E2B Box │   │E2B Box │   │E2B Box │   │E2B Box │
   │        │   │        │   │        │   │        │
   │Dataset │   │Dataset │   │Dataset │   │Dataset │
   │Shard 1 │   │Shard 2 │   │Shard 3 │   │Shard 4 │
   └────────┘   └────────┘   └────────┘   └────────┘
        │             │             │             │
        └─────────────┼─────────────┴─────────────┘
                      │
              ┌───────▼────────┐
              │Parameter Server│
              │  (Aggregator)  │
              └────────────────┘
```

**Components**:
1. **Neural Cluster Manager**: Orchestrates distributed training workflow
2. **Worker Nodes**: 4 E2B sandboxes processing data shards in parallel
3. **Parameter Server**: Aggregates gradients and updates global model
4. **Federated Learning**: Data never leaves local sandboxes

---

## Step-by-Step Walkthrough

### Phase 1: Cluster Initialization (2 minutes)

**Step 1.1**: Initialize distributed neural cluster with mesh topology

```bash
# Initialize cluster with mesh topology for peer-to-peer communication
npx flow-nexus@latest neural cluster init \
  --name "sentiment-analyzer-cluster" \
  --topology mesh \
  --architecture transformer \
  --daa-enabled true \
  --consensus proof-of-learning \
  --wasm-optimization true
```

**Expected Output**:
```json
{
  "cluster_id": "cls_9x7k3m2p",
  "name": "sentiment-analyzer-cluster",
  "topology": "mesh",
  "status": "initializing",
  "created_at": "2025-11-02T14:30:00Z",
  "config": {
    "architecture": "transformer",
    "daa_enabled": true,
    "consensus": "proof-of-learning",
    "wasm_optimization": true
  }
}
```

**What Happens**:
- Flow Nexus provisions cluster infrastructure
- Initializes DAA (Decentralized Autonomous Agents) coordination
- Sets up WASM acceleration for faster tensor operations
- Configures proof-of-learning consensus mechanism

---

### Phase 2: Worker Node Deployment (3 minutes)

**Step 2.1**: Deploy 4 worker nodes with training capabilities

```bash
# Deploy worker nodes in parallel
for i in {1..4}; do
  npx flow-nexus@latest neural node deploy \
    --cluster-id cls_9x7k3m2p \
    --node-type worker \
    --role worker \
    --model base \
    --template nodejs \
    --autonomy 0.8 \
    --capabilities training,inference &
done
wait
```

**Expected Output (per node)**:
```json
{
  "node_id": "node_w1_abc123",
  "cluster_id": "cls_9x7k3m2p",
  "sandbox_id": "sbx_e2b_xyz789",
  "status": "running",
  "role": "worker",
  "capabilities": ["training", "inference"],
  "autonomy_level": 0.8,
  "resources": {
    "cpu_cores": 4,
    "memory_gb": 8,
    "gpu": "shared"
  }
}
```

**What Happens**:
- Each worker spawns in dedicated E2B sandbox
- Node.js runtime with TensorFlow.js installed
- WASM-accelerated tensor operations enabled
- DAA autonomy at 80% (high self-governance)

---

**Step 2.2**: Deploy parameter server for gradient aggregation

```bash
# Deploy central parameter server
npx flow-nexus@latest neural node deploy \
  --cluster-id cls_9x7k3m2p \
  --node-type parameter_server \
  --role aggregator \
  --model large \
  --template nodejs \
  --autonomy 0.9 \
  --capabilities aggregation,synchronization
```

**Expected Output**:
```json
{
  "node_id": "node_ps_def456",
  "cluster_id": "cls_9x7k3m2p",
  "sandbox_id": "sbx_e2b_pqr321",
  "status": "running",
  "role": "aggregator",
  "capabilities": ["aggregation", "synchronization"],
  "autonomy_level": 0.9
}
```

---

### Phase 3: Network Connectivity (1 minute)

**Step 3.1**: Connect nodes in mesh topology

```bash
# Establish peer-to-peer connections between all nodes
npx flow-nexus@latest neural cluster connect \
  --cluster-id cls_9x7k3m2p \
  --topology mesh
```

**Expected Output**:
```json
{
  "cluster_id": "cls_9x7k3m2p",
  "topology": "mesh",
  "connections": [
    {"from": "node_w1_abc123", "to": "node_w2_bcd234"},
    {"from": "node_w1_abc123", "to": "node_w3_cde345"},
    {"from": "node_w1_abc123", "to": "node_w4_def456"},
    {"from": "node_w2_bcd234", "to": "node_w3_cde345"},
    {"from": "node_w2_bcd234", "to": "node_w4_def456"},
    {"from": "node_w3_cde345", "to": "node_w4_def456"},
    {"from": "node_ps_def456", "to": "all"}
  ],
  "status": "connected"
}
```

**What Happens**:
- Mesh topology creates full connectivity (each node talks to every other)
- Parameter server broadcasts to all workers
- Low-latency communication channels established
- DAA consensus protocol activated

---

### Phase 4: Distributed Training Execution (20 minutes)

**Step 4.1**: Upload training dataset and initiate federated training

```bash
# Upload sharded dataset to each worker sandbox
for i in {1..4}; do
  npx flow-nexus@latest sandbox upload \
    --sandbox-id sbx_e2b_w${i} \
    --file-path /data/train_shard_${i}.json \
    --content "$(cat customer_feedback_shard_${i}.json)" &
done
wait

# Start distributed training with federated learning
npx flow-nexus@latest neural train distributed \
  --cluster-id cls_9x7k3m2p \
  --dataset /data/train_*.json \
  --epochs 50 \
  --batch-size 32 \
  --learning-rate 0.001 \
  --optimizer adam \
  --federated true
```

**Expected Output**:
```json
{
  "training_id": "train_fed_gh789",
  "cluster_id": "cls_9x7k3m2p",
  "status": "training",
  "config": {
    "epochs": 50,
    "batch_size": 32,
    "learning_rate": 0.001,
    "optimizer": "adam",
    "federated": true
  },
  "workers": [
    "node_w1_abc123",
    "node_w2_bcd234",
    "node_w3_cde345",
    "node_w4_def456"
  ],
  "aggregator": "node_ps_def456"
}
```

---

**Step 4.2**: Monitor training progress in real-time

```bash
# Watch training metrics every 30 seconds
watch -n 30 'npx flow-nexus@latest neural cluster status --cluster-id cls_9x7k3m2p'
```

**Expected Output (Epoch 10/50)**:
```json
{
  "cluster_id": "cls_9x7k3m2p",
  "status": "training",
  "training_sessions": [
    {
      "training_id": "train_fed_gh789",
      "epoch": 10,
      "total_epochs": 50,
      "metrics": {
        "global_loss": 0.42,
        "global_accuracy": 0.78,
        "validation_accuracy": 0.75,
        "convergence_rate": 0.85
      },
      "workers": {
        "node_w1_abc123": {"local_loss": 0.39, "samples": 2500},
        "node_w2_bcd234": {"local_loss": 0.41, "samples": 2500},
        "node_w3_cde345": {"local_loss": 0.44, "samples": 2500},
        "node_w4_def456": {"local_loss": 0.43, "samples": 2500}
      },
      "time_elapsed": "8m 30s",
      "estimated_completion": "12m 15s"
    }
  ]
}
```

**What Happens**:
- Each worker trains on local data shard independently
- Gradients computed locally without sharing raw data
- Parameter server aggregates gradients via proof-of-learning consensus
- Global model updated and broadcasted back to workers
- Federated averaging ensures privacy-preserving training

---

### Phase 5: Model Validation & Deployment (5 minutes)

**Step 5.1**: Run distributed inference on test set

```bash
# Upload test dataset
npx flow-nexus@latest sandbox upload \
  --sandbox-id sbx_e2b_ps \
  --file-path /data/test.json \
  --content "$(cat customer_feedback_test.json)"

# Run distributed inference with ensemble aggregation
npx flow-nexus@latest neural predict distributed \
  --cluster-id cls_9x7k3m2p \
  --input-data /data/test.json \
  --aggregation ensemble
```

**Expected Output**:
```json
{
  "prediction_id": "pred_ens_ij012",
  "cluster_id": "cls_9x7k3m2p",
  "aggregation": "ensemble",
  "results": {
    "accuracy": 0.92,
    "precision": 0.91,
    "recall": 0.93,
    "f1_score": 0.92,
    "predictions": [
      {"text": "Great product!", "sentiment": "positive", "confidence": 0.97},
      {"text": "Terrible experience", "sentiment": "negative", "confidence": 0.95},
      {"text": "It's okay", "sentiment": "neutral", "confidence": 0.78}
    ]
  },
  "inference_time": "1.2s",
  "nodes_used": 4
}
```

**What Happens**:
- Test data distributed across all 4 workers
- Each worker runs inference independently
- Ensemble aggregation combines predictions (majority voting + confidence weighting)
- Final predictions achieve 92% accuracy (exceeds 90% target)

---

**Step 5.2**: Export trained model for production

```bash
# Export global model from parameter server
npx flow-nexus@latest sandbox execute \
  --sandbox-id sbx_e2b_ps \
  --code "
    const model = await tf.loadLayersModel('file:///model/global_model.json');
    await model.save('file:///export/sentiment_model_v1.0');
    console.log('Model exported successfully');
  "

# Download exported model
npx flow-nexus@latest sandbox download \
  --sandbox-id sbx_e2b_ps \
  --remote-path /export/sentiment_model_v1.0 \
  --local-path ./production/sentiment_model_v1.0
```

---

## Complete Code Example

**Automated Training Script** (`train_distributed.sh`):

```bash
#!/bin/bash
set -e

# Configuration
CLUSTER_NAME="sentiment-analyzer-cluster"
WORKERS=4
EPOCHS=50
BATCH_SIZE=32
LEARNING_RATE=0.001

echo "🚀 Initializing distributed neural cluster..."

# Step 1: Create cluster
CLUSTER_ID=$(npx flow-nexus@latest neural cluster init \
  --name "$CLUSTER_NAME" \
  --topology mesh \
  --architecture transformer \
  --daa-enabled true \
  --consensus proof-of-learning \
  --wasm-optimization true \
  --format json | jq -r '.cluster_id')

echo "✅ Cluster created: $CLUSTER_ID"

# Step 2: Deploy workers
echo "🔧 Deploying $WORKERS worker nodes..."
WORKER_IDS=()
for i in $(seq 1 $WORKERS); do
  WORKER_ID=$(npx flow-nexus@latest neural node deploy \
    --cluster-id "$CLUSTER_ID" \
    --node-type worker \
    --role worker \
    --model base \
    --template nodejs \
    --autonomy 0.8 \
    --capabilities training,inference \
    --format json | jq -r '.node_id')
  WORKER_IDS+=("$WORKER_ID")
  echo "  Worker $i deployed: $WORKER_ID"
done

# Step 3: Deploy parameter server
echo "🔧 Deploying parameter server..."
PS_ID=$(npx flow-nexus@latest neural node deploy \
  --cluster-id "$CLUSTER_ID" \
  --node-type parameter_server \
  --role aggregator \
  --model large \
  --template nodejs \
  --autonomy 0.9 \
  --capabilities aggregation,synchronization \
  --format json | jq -r '.node_id')

echo "✅ Parameter server deployed: $PS_ID"

# Step 4: Connect nodes
echo "🌐 Connecting nodes in mesh topology..."
npx flow-nexus@latest neural cluster connect \
  --cluster-id "$CLUSTER_ID" \
  --topology mesh

# Step 5: Upload data shards
echo "📤 Uploading training data shards..."
for i in $(seq 1 $WORKERS); do
  npx flow-nexus@latest sandbox upload \
    --sandbox-id "${WORKER_IDS[$i-1]}" \
    --file-path "/data/train_shard_${i}.json" \
    --content "$(cat data/customer_feedback_shard_${i}.json)" &
done
wait

# Step 6: Start distributed training
echo "🎓 Starting federated training..."
TRAINING_ID=$(npx flow-nexus@latest neural train distributed \
  --cluster-id "$CLUSTER_ID" \
  --dataset /data/train_*.json \
  --epochs $EPOCHS \
  --batch-size $BATCH_SIZE \
  --learning-rate $LEARNING_RATE \
  --optimizer adam \
  --federated true \
  --format json | jq -r '.training_id')

echo "✅ Training started: $TRAINING_ID"

# Step 7: Monitor training
echo "📊 Monitoring training progress..."
while true; do
  STATUS=$(npx flow-nexus@latest neural cluster status \
    --cluster-id "$CLUSTER_ID" \
    --format json)

  EPOCH=$(echo "$STATUS" | jq -r '.training_sessions[0].epoch')
  ACCURACY=$(echo "$STATUS" | jq -r '.training_sessions[0].metrics.validation_accuracy')

  echo "  Epoch $EPOCH/$EPOCHS | Validation Accuracy: $ACCURACY"

  if [ "$EPOCH" -eq "$EPOCHS" ]; then
    echo "✅ Training complete!"
    break
  fi

  sleep 30
done

# Step 8: Run validation
echo "🧪 Running distributed inference on test set..."
npx flow-nexus@latest neural predict distributed \
  --cluster-id "$CLUSTER_ID" \
  --input-data /data/test.json \
  --aggregation ensemble

echo "🎉 Distributed training workflow complete!"
echo "Cluster ID: $CLUSTER_ID"
echo "Training ID: $TRAINING_ID"
```

---

## Outcomes & Results

### Performance Metrics

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| **Validation Accuracy** | 90%+ | 92.1% | ✅ Exceeded |
| **Training Time** | <30 min | 24m 15s | ✅ Met |
| **Inference Latency** | <2s | 1.2s | ✅ Met |
| **Data Privacy** | Federated | Yes | ✅ Met |
| **Scalability** | 4 nodes | 4 nodes | ✅ Met |

### Cost Analysis

- **E2B Sandbox Cost**: $0.05/min × 4 workers × 25 min = $5.00
- **Parameter Server Cost**: $0.08/min × 25 min = $2.00
- **Total Training Cost**: $7.00
- **Cost per Epoch**: $0.14
- **Cost per Accuracy Point**: $0.76

### Business Value

- **Time Savings**: 4× faster than single-node training
- **Privacy Compliance**: GDPR-compliant federated learning
- **Scalability**: Can scale to 100+ workers for larger datasets
- **Production Ready**: Model exported and ready for deployment

---

## Tips & Best Practices

### 1. Optimal Worker Count

**Rule of Thumb**: `workers = dataset_size_gb / 2`

- **Small datasets** (<10GB): 2-4 workers
- **Medium datasets** (10-100GB): 4-16 workers
- **Large datasets** (>100GB): 16-64 workers

### 2. Topology Selection

| Topology | Best For | Communication Pattern |
|----------|----------|----------------------|
| **Mesh** | Small clusters (<10 nodes) | All-to-all, low latency |
| **Ring** | Sequential processing | Node-to-next, ordered |
| **Star** | Centralized aggregation | Hub-and-spoke, simple |
| **Hierarchical** | Large clusters (>20 nodes) | Multi-tier, scalable |

### 3. Federated Learning Tips

**Enable federated mode when**:
- Data privacy is critical (GDPR, HIPAA)
- Data cannot leave local environments
- Multiple organizations collaborate on training

**Disable federated mode when**:
- Speed is priority over privacy
- All data centrally available
- Training on public datasets

### 4. Monitoring & Debugging

**Watch for red flags**:
- **Divergent losses**: Workers have wildly different loss values (data imbalance)
- **Slow convergence**: Global accuracy stagnates (learning rate too low/high)
- **Node failures**: Sandboxes crash (resource exhaustion)

**Debugging commands**:
```bash
# Check node health
npx flow-nexus@latest sandbox status --sandbox-id sbx_e2b_w1

# View sandbox logs
npx flow-nexus@latest sandbox logs --sandbox-id sbx_e2b_w1 --lines 100

# Test connectivity
npx flow-nexus@latest neural cluster status --cluster-id cls_9x7k3m2p
```

### 5. Optimization Strategies

**WASM Acceleration**: 2-3× speedup for tensor ops
```bash
--wasm-optimization true
```

**Gradient Compression**: Reduce network bandwidth by 50%
```bash
--gradient-compression true --compression-ratio 0.5
```

**Mixed Precision Training**: 2× faster with minimal accuracy loss
```bash
--mixed-precision true
```

### 6. Cleanup & Cost Management

**Always terminate clusters when done**:
```bash
# Stop training
npx flow-nexus@latest neural cluster terminate --cluster-id cls_9x7k3m2p

# Verify all sandboxes stopped
npx flow-nexus@latest sandbox list --status running
```

**Cost-saving tips**:
- Use spot instances for non-critical training
- Schedule training during off-peak hours
- Enable auto-shutdown on idle (30 min timeout)

---

## Next Steps

1. **Scale Up**: Increase to 16 workers for larger datasets
2. **Advanced Models**: Try GPT-style architectures with `--architecture transformer-xl`
3. **Hyperparameter Tuning**: Use `neural_train` with grid search
4. **Production Deployment**: Integrate with CI/CD pipeline
5. **Monitoring**: Set up Prometheus + Grafana for real-time metrics

---

## Related Resources

- [Example 2: Neural Cluster Management](./example-2-neural-cluster.md)
- [Example 3: Model Deployment & Serving](./example-3-model-deployment.md)
- [Flow Nexus Neural Documentation](https://flow-nexus.ruv.io/docs/neural)
- [E2B Sandboxes Guide](https://e2b.dev/docs)
