# Flow Nexus Neural - Resources

Supporting scripts, templates, and assets for distributed neural network training with Flow Nexus.

## 📁 Directory Structure

```
resources/
├── scripts/          # Executable training scripts
├── templates/        # Model configuration templates
└── README.md         # This file
```

---

## 🔧 Scripts

Executable Node.js scripts for neural network training workflows.

### 1. train_distributed_cluster.js

**Purpose**: Train neural networks across multiple E2B sandbox nodes with distributed computing.

**Features**:
- Multi-node cluster initialization (mesh, ring, star, hierarchical topologies)
- Parameter server + worker architecture
- Gradient aggregation and synchronization
- Federated learning support (data stays local)
- Byzantine fault tolerance via proof-of-learning consensus
- Real-time monitoring and checkpointing

**Usage**:
```bash
node scripts/train_distributed_cluster.js \
  --config path/to/config.json \
  --dataset imagenet \
  --nodes 8 \
  --topology mesh \
  --federated \
  --epochs 100 \
  --batch-size 64 \
  --learning-rate 0.001 \
  --checkpoint-interval 10 \
  --verbose
```

**Options**:
- `-c, --config <path>`: Path to training configuration JSON (required)
- `-d, --dataset <id>`: Dataset identifier or path (required)
- `-n, --nodes <count>`: Number of worker nodes (default: 4)
- `-t, --topology <type>`: Cluster topology - mesh/ring/star/hierarchical (default: mesh)
- `-f, --federated`: Enable federated learning mode (default: false)
- `-e, --epochs <count>`: Number of training epochs (default: 100)
- `-b, --batch-size <size>`: Batch size per worker (default: 32)
- `-l, --learning-rate <rate>`: Learning rate (default: 0.001)
- `--checkpoint-interval <epochs>`: Save checkpoint every N epochs (default: 10)
- `--verbose`: Enable verbose logging

**Environment Variables**:
- `FLOW_NEXUS_API`: Flow Nexus API endpoint (default: https://api.flow-nexus.ruv.io)
- `FLOW_NEXUS_API_KEY`: API authentication key (required)

**Example**:
```bash
export FLOW_NEXUS_API_KEY="your-api-key"
node scripts/train_distributed_cluster.js \
  --config resources/templates/transformer-nlp.json \
  --dataset custom_text_data \
  --nodes 6 \
  --topology hierarchical \
  --epochs 50 \
  --batch-size 16
```

---

### 2. train_single_model.js

**Purpose**: Train neural networks on single E2B sandbox with custom architectures.

**Features**:
- Support for 5 architectures (feedforward, LSTM, GAN, transformer, autoencoder)
- Divergent thinking patterns (lateral, quantum, chaotic, associative, evolutionary)
- Real-time training metrics
- Model checkpointing
- Hyperparameter tuning support
- 5 training tiers (nano, mini, small, medium, large)

**Usage**:
```bash
node scripts/train_single_model.js \
  --config path/to/config.json \
  --tier small \
  --epochs 100 \
  --batch-size 32 \
  --learning-rate 0.001 \
  --optimizer adam \
  --divergent \
  --pattern lateral \
  --checkpoint-interval 10 \
  --verbose
```

**Options**:
- `-c, --config <path>`: Path to model configuration JSON (required)
- `-t, --tier <size>`: Training tier - nano/mini/small/medium/large (default: small)
- `-e, --epochs <count>`: Number of training epochs (default: 100)
- `-b, --batch-size <size>`: Batch size (default: 32)
- `-l, --learning-rate <rate>`: Learning rate (default: 0.001)
- `--optimizer <type>`: Optimizer - adam/sgd/rmsprop/adagrad (default: adam)
- `--divergent`: Enable divergent thinking patterns (default: false)
- `--pattern <type>`: Divergent pattern - lateral/quantum/chaotic/associative/evolutionary (default: lateral)
- `--checkpoint-interval <epochs>`: Save checkpoint every N epochs (default: 10)
- `--verbose`: Enable verbose logging

**Environment Variables**:
- `FLOW_NEXUS_API`: Flow Nexus API endpoint
- `FLOW_NEXUS_API_KEY`: API authentication key (required)
- `FLOW_NEXUS_USER_ID`: User identifier (required)

**Example**:
```bash
export FLOW_NEXUS_API_KEY="your-api-key"
export FLOW_NEXUS_USER_ID="user-123"
node scripts/train_single_model.js \
  --config resources/templates/lstm-timeseries.json \
  --tier medium \
  --epochs 150 \
  --divergent \
  --pattern quantum
```

---

### 3. deploy_template.js

**Purpose**: Deploy pre-trained models from the Flow Nexus marketplace.

**Features**:
- Search marketplace by category, tags, or keywords
- Filter by tier (free/paid) and accuracy metrics
- Deploy with custom training configurations
- Rate and review deployed templates
- List all available categories

**Usage**:
```bash
# Search templates
node scripts/deploy_template.js \
  --search "sentiment analysis" \
  --category nlp \
  --tier free \
  --limit 20

# Deploy template
node scripts/deploy_template.js \
  --template sentiment-analysis-v2 \
  --config custom-config.json \
  --verbose

# List categories
node scripts/deploy_template.js --list-categories
```

**Options**:
- `-s, --search <query>`: Search templates by keyword
- `-t, --template <id>`: Template ID to deploy
- `-c, --config <path>`: Custom configuration JSON (optional)
- `--category <type>`: Filter by category (classification/regression/nlp/vision/timeseries/anomaly/generative)
- `--tier <type>`: Filter by tier - free/paid (default: free)
- `--limit <count>`: Maximum number of search results (default: 20)
- `--list-categories`: List all available categories
- `--verbose`: Enable verbose logging

**Environment Variables**:
- `FLOW_NEXUS_API`: Flow Nexus API endpoint
- `FLOW_NEXUS_API_KEY`: API authentication key (required)
- `FLOW_NEXUS_USER_ID`: User identifier (required)

**Example**:
```bash
export FLOW_NEXUS_API_KEY="your-api-key"
export FLOW_NEXUS_USER_ID="user-123"

# Search for NLP templates
node scripts/deploy_template.js --search "text classification" --category nlp

# Deploy with custom config
node scripts/deploy_template.js \
  --template bert-classifier-v3 \
  --config my-custom-config.json
```

---

### 4. benchmark_model.js

**Purpose**: Run comprehensive performance benchmarks on trained models.

**Features**:
- Inference latency benchmarking (p50, p95, p99 percentiles)
- Throughput testing (queries per second)
- Memory profiling (peak usage, efficiency)
- GPU utilization monitoring
- Accuracy validation on test sets
- Export results to JSON/CSV
- Baseline comparisons

**Usage**:
```bash
node scripts/benchmark_model.js \
  --model model-abc123 \
  --type comprehensive \
  --iterations 1000 \
  --warmup 100 \
  --export json \
  --output benchmark-results.json \
  --verbose
```

**Options**:
- `-m, --model <id>`: Model ID to benchmark (required)
- `-t, --type <type>`: Benchmark type - inference/throughput/memory/comprehensive (default: comprehensive)
- `-i, --iterations <count>`: Number of benchmark iterations (default: 1000)
- `--warmup <count>`: Warmup iterations before benchmarking (default: 100)
- `--export <format>`: Export format - json/csv (default: json)
- `--output <path>`: Output file path (optional)
- `--verbose`: Enable verbose logging

**Environment Variables**:
- `FLOW_NEXUS_API`: Flow Nexus API endpoint
- `FLOW_NEXUS_API_KEY`: API authentication key (required)

**Example**:
```bash
export FLOW_NEXUS_API_KEY="your-api-key"
node scripts/benchmark_model.js \
  --model model-transformer-001 \
  --type comprehensive \
  --iterations 5000 \
  --export csv \
  --output benchmark-transformer.csv
```

**Output Example**:
```
⚡ Running performance benchmarks...
   Model: model-abc123
   Type: comprehensive
   Iterations: 1000
   Warmup: 100

📊 Benchmark Results

🚀 Inference Latency:
  Mean Latency: 12.50 ms
  P50 Latency:  11.20 ms
  P95 Latency:  18.30 ms
  P99 Latency:  25.10 ms

📈 Throughput:
  Queries Per Second: 8000 QPS

💾 Memory Usage:
  Peak Memory: 245.00 MB

🎮 GPU Metrics:
  GPU Utilization: 78.00%

🎯 Accuracy Metrics:
  Accuracy:  92.00%
  F1 Score:  89.00%

📊 Performance vs. Baselines:
  ✓ inference_latency_ms: 12.50 ≤ 50.00 (baseline)
  ✓ throughput_qps: 8000 ≥ 1000 (baseline)
  ✓ accuracy: 0.92 ≥ 0.90 (baseline)

💾 Results exported to: benchmark-model-abc123-1635789012345.json
```

---

## 📋 Templates

Pre-configured model architectures for common use cases.

### 1. feedforward-classifier.json

**Description**: Standard feedforward neural network for multi-class classification with dropout regularization.

**Architecture**:
- 4 hidden layers (256 → 128 → 64 → 10)
- Batch normalization after each layer
- Dropout regularization (30%, 20%, 10%)
- Softmax output activation
- He normal kernel initialization

**Training Configuration**:
- Epochs: 100
- Batch size: 32
- Learning rate: 0.001
- Optimizer: Adam
- Loss: Categorical crossentropy

**Use Cases**:
- Image classification (MNIST, CIFAR-10)
- Tabular data classification
- Feature vector classification
- Multi-class prediction problems

**Performance Targets**:
- Accuracy: ≥90%
- Inference latency: ≤10ms
- Memory usage: ≤200MB

**Example Usage**:
```bash
node scripts/train_single_model.js \
  --config resources/templates/feedforward-classifier.json \
  --tier small
```

---

### 2. lstm-timeseries.json

**Description**: LSTM network optimized for time series forecasting with bidirectional layers and attention mechanism.

**Architecture**:
- Bidirectional LSTM (128 units)
- Multi-head attention (4 heads)
- 2 additional LSTM layers (64, 32 units)
- Dense output layer with linear activation
- L2 regularization

**Training Configuration**:
- Epochs: 150
- Batch size: 64
- Learning rate: 0.001
- Optimizer: Adam
- Loss: Mean squared error (MSE)
- Metrics: MAE, MSE, MAPE

**Sequence Configuration**:
- Lookback window: 60 timesteps
- Forecast horizon: 10 timesteps
- Normalization: Enabled
- Stride: 1

**Use Cases**:
- Stock price forecasting
- Weather prediction
- Energy consumption forecasting
- Traffic flow prediction
- Sensor data analysis

**Performance Targets**:
- MAE: ≤0.05
- MSE: ≤0.01
- Inference latency: ≤25ms
- Memory usage: ≤350MB

**Example Usage**:
```bash
node scripts/train_single_model.js \
  --config resources/templates/lstm-timeseries.json \
  --tier medium \
  --divergent \
  --pattern quantum
```

---

### 3. transformer-nlp.json

**Description**: Transformer architecture for NLP tasks with multi-head attention and positional encoding.

**Architecture**:
- Word embedding (vocab: 10,000, dim: 512)
- Positional encoding (max length: 512)
- 6-layer transformer encoder (8 heads, 2048 FF dim)
- Global average pooling
- Dense classification layers (256 → 128 → 2)
- Attention + hidden dropout (10%)

**Training Configuration**:
- Epochs: 50
- Batch size: 16
- Learning rate: 0.0001
- Optimizer: Adam
- Loss: Categorical crossentropy
- Warmup schedule: 1000 steps

**Text Configuration**:
- Max sequence length: 512
- Tokenizer: WordPiece
- Padding: Post
- Truncation: Enabled

**Use Cases**:
- Sentiment analysis
- Text classification
- Named entity recognition (NER)
- Question answering
- Document categorization
- Intent detection

**Performance Targets**:
- Accuracy: ≥94%
- F1 Score: ≥92%
- Inference latency: ≤50ms
- Memory usage: ≤800MB

**Example Usage**:
```bash
node scripts/train_single_model.js \
  --config resources/templates/transformer-nlp.json \
  --tier large \
  --divergent \
  --pattern associative
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
npm install axios commander cli-table3
```

### 2. Set Environment Variables

```bash
export FLOW_NEXUS_API_KEY="your-api-key-here"
export FLOW_NEXUS_USER_ID="your-user-id"
```

### 3. Train a Model

```bash
# Single-node training
node resources/scripts/train_single_model.js \
  --config resources/templates/feedforward-classifier.json \
  --tier small

# Distributed training
node resources/scripts/train_distributed_cluster.js \
  --config resources/templates/transformer-nlp.json \
  --dataset custom_text_data \
  --nodes 4 \
  --topology mesh
```

### 4. Deploy from Marketplace

```bash
# Search templates
node resources/scripts/deploy_template.js --search "image classification"

# Deploy template
node resources/scripts/deploy_template.js --template resnet-classifier-v2
```

### 5. Benchmark Model

```bash
node resources/scripts/benchmark_model.js \
  --model model-abc123 \
  --type comprehensive \
  --export json
```

---

## 📚 Additional Resources

- **Flow Nexus Docs**: https://flow-nexus.ruv.io/docs
- **Neural Network Guide**: https://flow-nexus.ruv.io/docs/neural
- **Template Marketplace**: https://flow-nexus.ruv.io/templates
- **API Reference**: https://flow-nexus.ruv.io/api

---

## 🔒 Authentication

All scripts require Flow Nexus authentication. Register and login:

```bash
# Register (one-time)
npx flow-nexus@latest register

# Login
npx flow-nexus@latest login

# Get API key
npx flow-nexus@latest api-key
```

---

## ⚠️ Troubleshooting

### Import Errors

**Problem**: `Error: Cannot find module 'axios'`

**Solution**:
```bash
npm install axios commander cli-table3
```

### Authentication Errors

**Problem**: `Error: FLOW_NEXUS_API_KEY environment variable required`

**Solution**:
```bash
export FLOW_NEXUS_API_KEY="your-api-key"
export FLOW_NEXUS_USER_ID="your-user-id"
```

### Permission Denied

**Problem**: `Permission denied when executing scripts`

**Solution**:
```bash
chmod +x resources/scripts/*.js
```

---

## 📝 License

MIT License - See main skill documentation for details.
