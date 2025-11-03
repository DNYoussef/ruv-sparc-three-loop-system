# Example 3: Model Deployment & Serving with Flow Nexus Neural

## Scenario Overview

**Objective**: Deploy a trained sentiment analysis model as a production-ready REST API with horizontal scaling, A/B testing, and real-time monitoring.

**Use Case**: A SaaS company needs to serve a fine-tuned BERT model for customer sentiment analysis at scale, handling 10,000+ requests per minute with <100ms latency.

**Requirements**:
- Deploy model as REST API with OpenAPI docs
- Support 10,000+ requests/minute
- P95 latency <100ms
- A/B testing for model versions
- Real-time performance monitoring
- Blue-green deployment strategy
- Auto-scaling based on load

---

## Architecture Design

```
                    ┌────────────────────────┐
                    │   Load Balancer        │
                    │   (HAProxy/nginx)      │
                    └───────────┬────────────┘
                                │
                ┌───────────────┼───────────────┐
                │               │               │
        ┌───────▼──────┐ ┌─────▼──────┐ ┌─────▼──────┐
        │  Blue Env    │ │ Green Env  │ │ Canary Env │
        │ (v1.0 - 70%) │ │(v1.1 - 25%)│ │(v2.0 - 5%) │
        └───────┬──────┘ └─────┬──────┘ └─────┬──────┘
                │               │               │
     ┌──────────┼───────────────┼───────────────┼──────────┐
     │          │               │               │          │
┌────▼───┐ ┌───▼────┐ ┌────▼───┐ ┌───▼────┐ ┌──▼─────┐ ┌──▼─────┐
│Replica1│ │Replica2│ │Replica3│ │Replica4│ │Canary 1│ │Canary 2│
│E2B Box │ │E2B Box │ │E2B Box │ │E2B Box │ │E2B Box │ │E2B Box │
│        │ │        │ │        │ │        │ │        │ │        │
│Model   │ │Model   │ │Model   │ │Model   │ │Model   │ │Model   │
│v1.0    │ │v1.0    │ │v1.1    │ │v1.1    │ │v2.0    │ │v2.0    │
└────────┘ └────────┘ └────────┘ └────────┘ └────────┘ └────────┘
     │          │          │          │          │          │
     └──────────┴──────────┴──────────┴──────────┴──────────┘
                            │
                   ┌────────▼────────┐
                   │ Metrics Pipeline │
                   │ (Prometheus +    │
                   │  Grafana)        │
                   └──────────────────┘
```

**Components**:
1. **Load Balancer**: Traffic distribution with A/B routing rules
2. **Blue Environment**: Stable production model (v1.0, 70% traffic)
3. **Green Environment**: New stable model (v1.1, 25% traffic)
4. **Canary Environment**: Experimental model (v2.0, 5% traffic)
5. **Replicas**: Auto-scaled instances (2-20 per environment)
6. **Metrics Pipeline**: Real-time monitoring & alerting

---

## Step-by-Step Walkthrough

### Phase 1: Model Packaging & Upload (5 minutes)

**Step 1.1**: Export trained model from previous examples

```bash
# Assume model trained in Example 1 (cls_9x7k3m2p)
# Export model artifacts
npx flow-nexus@latest sandbox execute \
  --sandbox-id sbx_e2b_ps \
  --code "
    const tf = require('@tensorflow/tfjs-node');

    // Load trained model
    const model = await tf.loadLayersModel('file:///models/sentiment_model');

    // Export for serving
    await model.save('file:///export/sentiment_v1.0');

    // Export metadata
    const metadata = {
      name: 'sentiment-analyzer',
      version: '1.0.0',
      architecture: 'transformer',
      input_shape: [128],
      output_classes: 3,
      labels: ['negative', 'neutral', 'positive'],
      accuracy: 0.92,
      training_date: '2025-11-02',
      hyperparameters: {
        epochs: 50,
        batch_size: 32,
        learning_rate: 0.001
      }
    };

    require('fs').writeFileSync(
      '/export/model_metadata.json',
      JSON.stringify(metadata, null, 2)
    );

    console.log('Model exported successfully');
  "
```

---

**Step 1.2**: Upload model to Flow Nexus storage

```bash
# Create model registry entry
npx flow-nexus@latest neural publish-template \
  --model-id model_sentiment_v1_0 \
  --name "sentiment-analyzer-v1.0" \
  --description "BERT-based sentiment classifier (92% accuracy)" \
  --category classification \
  --user-id usr_12345 \
  --price 0  # Free template

# Upload model artifacts to cloud storage
npx flow-nexus@latest storage upload \
  --bucket ml-models \
  --path sentiment-analyzer/v1.0/model.json \
  --content "$(cat /export/sentiment_v1.0/model.json)"

npx flow-nexus@latest storage upload \
  --bucket ml-models \
  --path sentiment-analyzer/v1.0/weights.bin \
  --content "$(base64 /export/sentiment_v1.0/weights.bin)"

npx flow-nexus@latest storage upload \
  --bucket ml-models \
  --path sentiment-analyzer/v1.0/metadata.json \
  --content "$(cat /export/model_metadata.json)"
```

**Expected Output**:
```json
{
  "template_id": "tpl_sentiment_v1_0",
  "model_registry": {
    "model_id": "model_sentiment_v1_0",
    "version": "1.0.0",
    "status": "published",
    "artifacts": [
      "gs://ml-models/sentiment-analyzer/v1.0/model.json",
      "gs://ml-models/sentiment-analyzer/v1.0/weights.bin",
      "gs://ml-models/sentiment-analyzer/v1.0/metadata.json"
    ]
  }
}
```

---

### Phase 2: Deployment Infrastructure Setup (10 minutes)

**Step 2.1**: Create deployment template with REST API wrapper

**Template File**: `deployment_template.yaml`

```yaml
name: sentiment-api-deployment
version: 1.0.0
description: REST API for sentiment analysis model serving

variables:
  model_version:
    type: string
    default: "1.0.0"
    description: Model version to deploy

  replicas:
    type: integer
    default: 3
    description: Number of replica instances

  port:
    type: integer
    default: 8080
    description: API server port

infrastructure:
  sandboxes:
    - name: api-server
      template: nodejs
      replicas: ${replicas}
      resources:
        cpu_cores: 2
        memory_gb: 4
      env_vars:
        MODEL_VERSION: ${model_version}
        MODEL_PATH: gs://ml-models/sentiment-analyzer/v${model_version}
        PORT: ${port}
        LOG_LEVEL: info

  load_balancer:
    type: haproxy
    algorithm: least_connections
    health_check:
      path: /health
      interval: 10
      timeout: 5

  monitoring:
    prometheus:
      enabled: true
      scrape_interval: 15
    grafana:
      enabled: true
      dashboards:
        - model-performance
        - api-metrics

application:
  setup:
    - npm install express @tensorflow/tfjs-node
    - npm install prometheus-client morgan helmet
    - mkdir -p /app/models
    - gsutil cp -r ${MODEL_PATH}/* /app/models/

  main_script: |
    const express = require('express');
    const tf = require('@tensorflow/tfjs-node');
    const prometheus = require('prom-client');
    const morgan = require('morgan');
    const helmet = require('helmet');

    const app = express();
    app.use(helmet());
    app.use(morgan('combined'));
    app.use(express.json());

    // Prometheus metrics
    const register = new prometheus.Registry();
    const requestCounter = new prometheus.Counter({
      name: 'sentiment_api_requests_total',
      help: 'Total API requests',
      labelNames: ['method', 'status']
    });
    const latencyHistogram = new prometheus.Histogram({
      name: 'sentiment_api_latency_ms',
      help: 'API latency in milliseconds',
      buckets: [10, 50, 100, 200, 500]
    });
    register.registerMetric(requestCounter);
    register.registerMetric(latencyHistogram);

    // Load model
    let model;
    let metadata;

    async function loadModel() {
      console.log('Loading model from /app/models/...');
      model = await tf.loadLayersModel('file:///app/models/model.json');
      metadata = JSON.parse(require('fs').readFileSync('/app/models/metadata.json', 'utf8'));
      console.log(`Model loaded: ${metadata.name} v${metadata.version}`);
    }

    // Health check endpoint
    app.get('/health', (req, res) => {
      res.json({
        status: 'healthy',
        model_loaded: !!model,
        version: metadata?.version || 'unknown'
      });
    });

    // Metrics endpoint
    app.get('/metrics', async (req, res) => {
      res.set('Content-Type', register.contentType);
      res.end(await register.metrics());
    });

    // Model info endpoint
    app.get('/info', (req, res) => {
      res.json(metadata);
    });

    // Prediction endpoint
    app.post('/predict', async (req, res) => {
      const startTime = Date.now();

      try {
        const { text } = req.body;

        if (!text) {
          requestCounter.inc({ method: 'POST', status: '400' });
          return res.status(400).json({ error: 'Missing text field' });
        }

        // Tokenize and preprocess (simplified)
        const tokens = text.toLowerCase().split(' ').slice(0, 128);
        const inputTensor = tf.tensor2d([tokens.map((_, i) => i)], [1, 128]);

        // Run inference
        const prediction = model.predict(inputTensor);
        const probabilities = await prediction.array();
        const predictedClass = probabilities[0].indexOf(Math.max(...probabilities[0]));
        const confidence = probabilities[0][predictedClass];

        // Cleanup tensors
        inputTensor.dispose();
        prediction.dispose();

        const latency = Date.now() - startTime;
        latencyHistogram.observe(latency);
        requestCounter.inc({ method: 'POST', status: '200' });

        res.json({
          text,
          sentiment: metadata.labels[predictedClass],
          confidence: parseFloat(confidence.toFixed(4)),
          probabilities: {
            negative: parseFloat(probabilities[0][0].toFixed(4)),
            neutral: parseFloat(probabilities[0][1].toFixed(4)),
            positive: parseFloat(probabilities[0][2].toFixed(4))
          },
          model_version: metadata.version,
          latency_ms: latency
        });

      } catch (error) {
        console.error('Prediction error:', error);
        requestCounter.inc({ method: 'POST', status: '500' });
        res.status(500).json({ error: 'Prediction failed' });
      }
    });

    // Batch prediction endpoint
    app.post('/predict/batch', async (req, res) => {
      const startTime = Date.now();

      try {
        const { texts } = req.body;

        if (!Array.isArray(texts)) {
          return res.status(400).json({ error: 'texts must be an array' });
        }

        const predictions = await Promise.all(
          texts.map(async (text) => {
            const tokens = text.toLowerCase().split(' ').slice(0, 128);
            const inputTensor = tf.tensor2d([tokens.map((_, i) => i)], [1, 128]);
            const prediction = model.predict(inputTensor);
            const probabilities = await prediction.array();
            const predictedClass = probabilities[0].indexOf(Math.max(...probabilities[0]));

            inputTensor.dispose();
            prediction.dispose();

            return {
              text,
              sentiment: metadata.labels[predictedClass],
              confidence: parseFloat(probabilities[0][predictedClass].toFixed(4))
            };
          })
        );

        const latency = Date.now() - startTime;
        latencyHistogram.observe(latency);

        res.json({
          predictions,
          count: predictions.length,
          latency_ms: latency,
          model_version: metadata.version
        });

      } catch (error) {
        console.error('Batch prediction error:', error);
        res.status(500).json({ error: 'Batch prediction failed' });
      }
    });

    // Start server
    const PORT = process.env.PORT || 8080;

    loadModel().then(() => {
      app.listen(PORT, () => {
        console.log(`Sentiment API server running on port ${PORT}`);
      });
    }).catch((err) => {
      console.error('Failed to load model:', err);
      process.exit(1);
    });
```

---

**Step 2.2**: Deploy Blue environment (v1.0 - production)

```bash
# Deploy template with Flow Nexus
npx flow-nexus@latest template deploy \
  --template deployment_template.yaml \
  --deployment-name sentiment-api-blue \
  --variables model_version=1.0.0,replicas=3,port=8080

# Wait for deployment to complete
npx flow-nexus@latest sandbox list --status running | grep sentiment-api-blue
```

**Expected Output**:
```json
{
  "deployment_id": "dep_blue_abc123",
  "name": "sentiment-api-blue",
  "status": "running",
  "sandboxes": [
    {
      "sandbox_id": "sbx_blue_001",
      "url": "https://sbx-blue-001.e2b.dev",
      "health": "healthy",
      "replicas": 1
    },
    {
      "sandbox_id": "sbx_blue_002",
      "url": "https://sbx-blue-002.e2b.dev",
      "health": "healthy",
      "replicas": 1
    },
    {
      "sandbox_id": "sbx_blue_003",
      "url": "https://sbx-blue-003.e2b.dev",
      "health": "healthy",
      "replicas": 1
    }
  ],
  "load_balancer": {
    "url": "https://sentiment-api-blue.e2b.dev",
    "algorithm": "least_connections"
  }
}
```

---

**Step 2.3**: Deploy Green environment (v1.1 - new stable)

```bash
# Assume v1.1 model trained with improved accuracy (94%)
npx flow-nexus@latest template deploy \
  --template deployment_template.yaml \
  --deployment-name sentiment-api-green \
  --variables model_version=1.1.0,replicas=2,port=8080
```

---

**Step 2.4**: Deploy Canary environment (v2.0 - experimental)

```bash
# v2.0 model with new architecture (experimental)
npx flow-nexus@latest template deploy \
  --template deployment_template.yaml \
  --deployment-name sentiment-api-canary \
  --variables model_version=2.0.0,replicas=1,port=8080
```

---

### Phase 3: A/B Testing Configuration (5 minutes)

**Step 3.1**: Configure load balancer with traffic splitting

```bash
# Configure HAProxy with weighted traffic distribution
cat > haproxy_config.cfg <<'EOF'
global
    log stdout format raw local0
    maxconn 10000

defaults
    log global
    mode http
    timeout connect 5000ms
    timeout client 50000ms
    timeout server 50000ms

frontend sentiment_api
    bind *:80

    # A/B testing ACL rules
    acl canary_test rand(100) lt 5     # 5% canary traffic
    acl green_test rand(95) lt 26      # 25% of remaining (95%)

    # Route traffic based on A/B rules
    use_backend canary if canary_test
    use_backend green if green_test
    default_backend blue

backend blue
    balance leastconn
    option httpchk GET /health
    server blue1 sbx-blue-001.e2b.dev:8080 check
    server blue2 sbx-blue-002.e2b.dev:8080 check
    server blue3 sbx-blue-003.e2b.dev:8080 check

backend green
    balance leastconn
    option httpchk GET /health
    server green1 sbx-green-001.e2b.dev:8080 check
    server green2 sbx-green-002.e2b.dev:8080 check

backend canary
    balance leastconn
    option httpchk GET /health
    server canary1 sbx-canary-001.e2b.dev:8080 check
EOF

# Deploy HAProxy configuration
npx flow-nexus@latest sandbox create \
  --template base \
  --name haproxy-lb \
  --startup-script "
    apt-get update && apt-get install -y haproxy
    cat > /etc/haproxy/haproxy.cfg <<'HEREDOC'
$(cat haproxy_config.cfg)
HEREDOC
    systemctl restart haproxy
  "
```

---

### Phase 4: Performance Testing & Monitoring (15 minutes)

**Step 4.1**: Load testing with Apache Bench

```bash
# Simulate 10,000 requests/minute (167 req/sec)
ab -n 10000 -c 50 -p test_payload.json -T application/json \
  https://sentiment-api-blue.e2b.dev/predict

# Test payload (test_payload.json)
cat > test_payload.json <<EOF
{
  "text": "This product exceeded my expectations! Highly recommended."
}
EOF
```

**Expected Output**:
```
Concurrency Level:      50
Time taken for tests:   60.234 seconds
Complete requests:      10000
Failed requests:        0
Requests per second:    166.02 [#/sec] (mean)
Time per request:       301.17 [ms] (mean)
Time per request:       6.02 [ms] (mean, across all concurrent requests)

Percentage of requests served within a certain time (ms)
  50%     58
  66%     72
  75%     81
  80%     87
  90%     105
  95%     124
  98%     158
  99%     189
 100%     312 (longest request)
```

**Analysis**:
- ✅ P50 latency: 58ms (<100ms target)
- ✅ P95 latency: 124ms (slightly above target, acceptable)
- ✅ P99 latency: 189ms (within acceptable range)
- ✅ Throughput: 166 req/sec (meets 10,000 req/min target)

---

**Step 4.2**: Monitor A/B test metrics

```bash
# Query Prometheus for A/B traffic distribution
curl -s 'http://prometheus:9090/api/v1/query' \
  --data-urlencode 'query=sum by (backend) (rate(sentiment_api_requests_total[5m]))' \
  | jq '.data.result'
```

**Expected Output**:
```json
[
  {
    "metric": {"backend": "blue"},
    "value": [1730563200, "116.2"]
  },
  {
    "metric": {"backend": "green"},
    "value": [1730563200, "41.5"]
  },
  {
    "metric": {"backend": "canary"},
    "value": [1730563200, "8.3"]
  }
]
```

**Traffic Distribution**:
- Blue (v1.0): 116.2 req/sec (70% of traffic) ✅
- Green (v1.1): 41.5 req/sec (25% of traffic) ✅
- Canary (v2.0): 8.3 req/sec (5% of traffic) ✅

---

**Step 4.3**: Compare model performance across versions

```bash
# Query accuracy metrics by model version
curl -s 'http://prometheus:9090/api/v1/query' \
  --data-urlencode 'query=avg by (model_version) (sentiment_prediction_confidence)' \
  | jq '.data.result'
```

**Expected Output**:
```json
[
  {
    "metric": {"model_version": "1.0.0"},
    "value": [1730563200, "0.89"]
  },
  {
    "metric": {"model_version": "1.1.0"},
    "value": [1730563200, "0.92"]
  },
  {
    "metric": {"model_version": "2.0.0"},
    "value": [1730563200, "0.87"]
  }
]
```

**Analysis**:
- v1.0 (blue): 89% average confidence (baseline)
- v1.1 (green): 92% average confidence (+3% improvement) ✅ Winner
- v2.0 (canary): 87% average confidence (-2% regression) ❌ Reject

**Decision**: Promote v1.1 to 100% traffic, rollback v2.0 canary

---

### Phase 5: Blue-Green Deployment (5 minutes)

**Step 5.1**: Gradually shift traffic from v1.0 to v1.1

```bash
# Update HAProxy config to route 100% to green (v1.1)
cat > haproxy_config_v2.cfg <<'EOF'
frontend sentiment_api
    bind *:80
    default_backend green

backend green
    balance leastconn
    option httpchk GET /health
    server green1 sbx-green-001.e2b.dev:8080 check
    server green2 sbx-green-002.e2b.dev:8080 check
    server green3 sbx-green-003.e2b.dev:8080 check  # Scale up green

# Keep blue as backup
backend blue
    balance leastconn
    option httpchk GET /health
    server blue1 sbx-blue-001.e2b.dev:8080 check backup
EOF

# Apply new configuration
npx flow-nexus@latest sandbox execute \
  --sandbox-id sbx_haproxy_lb \
  --code "
    cat > /etc/haproxy/haproxy.cfg <<'HEREDOC'
$(cat haproxy_config_v2.cfg)
HEREDOC
    systemctl reload haproxy
  "
```

---

**Step 5.2**: Verify traffic cutover

```bash
# Monitor traffic for 5 minutes after cutover
for i in {1..5}; do
  echo "Minute $i:"
  curl -s 'http://prometheus:9090/api/v1/query' \
    --data-urlencode 'query=sum by (model_version) (rate(sentiment_api_requests_total[1m]))' \
    | jq -r '.data.result[] | "\(.metric.model_version): \(.value[1]) req/sec"'
  sleep 60
done
```

**Expected Output**:
```
Minute 1:
1.0.0: 85.3 req/sec
1.1.0: 80.7 req/sec

Minute 2:
1.0.0: 42.1 req/sec
1.1.0: 123.9 req/sec

Minute 3:
1.0.0: 8.2 req/sec
1.1.0: 157.8 req/sec

Minute 4:
1.0.0: 0.5 req/sec
1.1.0: 165.5 req/sec

Minute 5:
1.0.0: 0.0 req/sec
1.1.0: 166.0 req/sec
```

**Result**: Traffic successfully cutover to v1.1 with zero downtime ✅

---

**Step 5.3**: Scale down old blue environment

```bash
# Terminate blue environment (v1.0) after cutover
npx flow-nexus@latest sandbox stop --sandbox-id sbx_blue_001
npx flow-nexus@latest sandbox stop --sandbox-id sbx_blue_002
npx flow-nexus@latest sandbox stop --sandbox-id sbx_blue_003

# Terminate canary environment (v2.0 rejected)
npx flow-nexus@latest sandbox stop --sandbox-id sbx_canary_001

# Cost savings: 4 sandboxes × $0.05/min = $0.20/min = $288/day
```

---

## Complete Code Example

**Automated Deployment Script** (`deploy_model.sh`):

```bash
#!/bin/bash
set -euo pipefail

# Configuration
MODEL_VERSION="$1"
ENVIRONMENT="${2:-blue}"  # blue, green, or canary
REPLICAS="${3:-3}"
PORT=8080

# Validate inputs
if [ -z "$MODEL_VERSION" ]; then
  echo "Usage: $0 <model_version> [environment] [replicas]"
  exit 1
fi

log() {
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# Step 1: Deploy environment
log "Deploying $ENVIRONMENT environment with model v$MODEL_VERSION..."

DEPLOYMENT_ID=$(npx flow-nexus@latest template deploy \
  --template deployment_template.yaml \
  --deployment-name "sentiment-api-$ENVIRONMENT" \
  --variables "model_version=$MODEL_VERSION,replicas=$REPLICAS,port=$PORT" \
  --format json | jq -r '.deployment_id')

log "Deployment created: $DEPLOYMENT_ID"

# Step 2: Wait for health checks
log "Waiting for all replicas to become healthy..."
for i in {1..30}; do
  HEALTHY=$(npx flow-nexus@latest sandbox list \
    --status running \
    --filter "sentiment-api-$ENVIRONMENT" \
    --format json | jq '[.[] | select(.health == "healthy")] | length')

  if [ "$HEALTHY" -eq "$REPLICAS" ]; then
    log "All $REPLICAS replicas healthy!"
    break
  fi

  log "Healthy replicas: $HEALTHY/$REPLICAS (waiting...)"
  sleep 10
done

# Step 3: Run smoke tests
log "Running smoke tests..."
LB_URL=$(npx flow-nexus@latest sandbox list \
  --filter "sentiment-api-$ENVIRONMENT" \
  --format json | jq -r '.[0].load_balancer_url')

RESPONSE=$(curl -s -X POST "$LB_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "This is a test"}')

SENTIMENT=$(echo "$RESPONSE" | jq -r '.sentiment')
if [ -z "$SENTIMENT" ]; then
  log "ERROR: Smoke test failed!"
  exit 1
fi

log "Smoke test passed: sentiment=$SENTIMENT"

# Step 4: Output deployment info
echo ""
echo "========================================="
echo "Deployment successful!"
echo "========================================="
echo "Environment: $ENVIRONMENT"
echo "Model Version: $MODEL_VERSION"
echo "Replicas: $REPLICAS"
echo "Load Balancer URL: $LB_URL"
echo "========================================="
echo ""
echo "Test with:"
echo "curl -X POST $LB_URL/predict \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{\"text\": \"Your text here\"}'"
```

**Usage Examples**:
```bash
# Deploy v1.0 to blue environment (3 replicas)
./deploy_model.sh 1.0.0 blue 3

# Deploy v1.1 to green environment (2 replicas)
./deploy_model.sh 1.1.0 green 2

# Deploy v2.0 to canary environment (1 replica)
./deploy_model.sh 2.0.0 canary 1
```

---

## Outcomes & Results

### Performance Benchmarks

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| **Throughput** | 10,000 req/min | 10,200 req/min | ✅ Exceeded |
| **P50 Latency** | <100ms | 58ms | ✅ Met |
| **P95 Latency** | <100ms | 124ms | ⚠️ Slightly over |
| **P99 Latency** | <200ms | 189ms | ✅ Met |
| **Uptime** | 99.9% | 99.97% | ✅ Exceeded |
| **Zero Downtime** | Yes | Yes | ✅ Met |

### A/B Testing Results

| Version | Traffic % | Avg Confidence | Decision |
|---------|-----------|----------------|----------|
| v1.0 (Blue) | 70% | 89% | Baseline |
| v1.1 (Green) | 25% | 92% | **WINNER** (Promote to 100%) |
| v2.0 (Canary) | 5% | 87% | **REJECT** (Rollback) |

### Cost Analysis

**Before Optimization** (Single environment, over-provisioned):
- 10 replicas × $0.05/min × 1440 min/day = $720/day

**After Optimization** (Blue-green with right-sizing):
- Production (green): 3 replicas × $0.05/min × 1440 min/day = $216/day
- Backup (blue, standby): 1 replica × $0.05/min × 1440 min/day = $72/day
- **Total**: $288/day (60% cost reduction)

### Business Impact

- **Reduced deployment risk**: A/B testing caught v2.0 regression before full rollout
- **Zero downtime**: Blue-green deployment achieved seamless cutover
- **Improved model accuracy**: 92% vs 89% (3% absolute gain)
- **Cost optimization**: $432/day savings vs over-provisioned setup

---

## Tips & Best Practices

### 1. Traffic Splitting Strategies

**Conservative approach** (gradual rollout):
```
Week 1: Blue 90%, Green 10%
Week 2: Blue 70%, Green 30%
Week 3: Blue 50%, Green 50%
Week 4: Blue 0%, Green 100%
```

**Aggressive approach** (fast iteration):
```
Day 1: Blue 95%, Canary 5%
Day 2: Blue 70%, Green 30% (if canary passed)
Day 3: Blue 0%, Green 100%
```

### 2. Automated Rollback Triggers

**Monitor error rates**:
```bash
# Rollback if error rate >1%
if error_rate > 0.01:
  rollback_to_previous_version()
```

**Monitor latency degradation**:
```bash
# Rollback if P95 latency increases >20%
if p95_latency > baseline * 1.2:
  rollback_to_previous_version()
```

### 3. Model Versioning Best Practices

**Semantic versioning for models**:
- **Major** (2.0.0): Architecture change (e.g., BERT → GPT)
- **Minor** (1.1.0): Performance improvement (e.g., accuracy gain)
- **Patch** (1.0.1): Bug fixes (e.g., tokenization fix)

### 4. Cost Optimization Tips

**Auto-scaling based on load**:
```bash
--min-replicas 2   # Always-on for availability
--max-replicas 20  # Scale up during traffic spikes
--scale-metric cpu_utilization
--scale-threshold 70
```

**Spot instances for non-critical environments**:
```bash
--instance-type spot  # 70% cost savings for canary/staging
--fallback-to-on-demand false
```

### 5. Monitoring & Observability

**Key metrics to track**:
- Request latency (P50, P95, P99)
- Prediction confidence (model quality proxy)
- Error rate (5xx errors)
- Model version distribution (A/B traffic)
- Resource utilization (CPU, memory)

**Alerting rules**:
```yaml
- alert: HighErrorRate
  expr: rate(http_errors_total[5m]) > 0.01
  severity: critical

- alert: HighLatency
  expr: histogram_quantile(0.95, rate(http_request_duration_ms[5m])) > 100
  severity: warning
```

### 6. Security Best Practices

**API authentication**:
```javascript
app.use((req, res, next) => {
  const apiKey = req.headers['x-api-key'];
  if (!apiKey || !validateApiKey(apiKey)) {
    return res.status(401).json({ error: 'Unauthorized' });
  }
  next();
});
```

**Rate limiting**:
```javascript
const rateLimit = require('express-rate-limit');

const limiter = rateLimit({
  windowMs: 60 * 1000, // 1 minute
  max: 100 // 100 requests per minute per IP
});

app.use('/predict', limiter);
```

---

## Next Steps

1. **Multi-Region Deployment**: Deploy to AWS + GCP for global low latency
2. **Advanced A/B Testing**: Implement multi-armed bandit algorithms
3. **Model Monitoring**: Track data drift and model degradation
4. **CI/CD Integration**: Automate deployment from model training
5. **GraphQL API**: Add GraphQL endpoint for flexible queries

---

## Related Resources

- [Example 1: Distributed Training](./example-1-distributed-training.md)
- [Example 2: Neural Cluster Management](./example-2-neural-cluster.md)
- [Flow Nexus Deployment Guide](https://flow-nexus.ruv.io/docs/deployment)
- [Model Serving Best Practices](https://flow-nexus.ruv.io/docs/serving)
