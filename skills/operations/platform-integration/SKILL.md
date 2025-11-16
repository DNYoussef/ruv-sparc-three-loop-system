---
name: platform-integration
description: Enterprise platform integration orchestration with API connectivity,
  webhook automation, data synchronization, and multi-platform coordination. Supports
  REST, GraphQL, WebSockets, message queues, and event-driven architectures.
tags:
- platform
- integration
- api
- webhooks
- sync
- essential
- tier-2
version: 2.0.0
category: operations
author: ruv
---

# Platform Integration

## Purpose

Execute enterprise-grade platform integration with comprehensive API connectivity, webhook automation, data synchronization, and event-driven coordination across multiple platforms and services.

## Specialist Agent

I am a platform integration architect specializing in multi-platform connectivity.

**Methodology** (Integration Lifecycle Pattern):
1. Discovery & mapping (API/webhook analysis)
2. Authentication strategy (OAuth, API keys, JWT)
3. Integration architecture design
4. Connector implementation (API clients, SDKs)
5. Webhook handler development
6. Data transformation pipelines
7. Synchronization engine setup
8. Error handling & retry logic
9. Monitoring & observability
10. Testing & validation
11. Documentation generation
12. Production deployment

**Supported Platforms**:
- **Cloud**: AWS, Azure, GCP, Salesforce, HubSpot
- **DevOps**: GitHub, GitLab, Jira, CircleCI, Jenkins
- **Commerce**: Stripe, Shopify, WooCommerce, PayPal
- **Communication**: Slack, Discord, Twilio, SendGrid
- **Databases**: PostgreSQL, MongoDB, Redis, Elasticsearch
- **Message Queues**: RabbitMQ, Kafka, AWS SQS, Azure Service Bus

**Integration Patterns**:
- **API-First**: RESTful, GraphQL, gRPC
- **Event-Driven**: Webhooks, WebSockets, SSE
- **Message Queues**: Pub/Sub, AMQP, MQTT
- **Data Sync**: ETL, CDC, batch/real-time
- **Orchestration**: Workflow engines, state machines

## Input Contract

```yaml
input:
  platforms: array[object] # Platform configurations
    - name: string (e.g., "salesforce", "stripe")
      type: string (api|webhook|sync)
      auth: object (credentials, tokens)
      endpoints: array[string]
  integration_type: string # "api_integration" | "webhook_automation" | "data_sync" | "full"
  sync_direction: string # "bidirectional" | "source_to_target" | "target_to_source"
  sync_frequency: string # "real_time" | "hourly" | "daily" | "on_demand"
  error_strategy: string # "retry" | "dead_letter" | "alert" | "fallback"
  monitoring: boolean # Enable observability (default: true)
```

## Output Contract

```yaml
output:
  artifacts:
    connectors: directory # API client implementations
    handlers: directory # Webhook handlers
    sync_engine: directory # Data synchronization
    tests: directory # Integration test suite
    configs: array[file] # Configuration files
    docs: markdown # Integration documentation
  metrics:
    api_latency_p95: number # ms
    webhook_success_rate: number # percentage
    sync_throughput: number # records/sec
    error_rate: number # percentage
  endpoints:
    api_base_url: string
    webhook_url: string
    health_check: string
  monitoring:
    dashboard_url: string
    alerts: array[object]
```

## Execution Flow

```bash
#!/bin/bash
set -e

INTEGRATION_TYPE="${1:-full}"
PLATFORMS_CONFIG="$2"
OUTPUT_DIR="platform-integration-$(date +%s)"

mkdir -p "$OUTPUT_DIR"/{connectors,handlers,sync,tests,configs,docs,monitoring}

echo "================================================================"
echo "Platform Integration Orchestration"
echo "Type: $INTEGRATION_TYPE | Platforms: $PLATFORMS_CONFIG"
echo "================================================================"

# STAGE 1: Discovery & Mapping
echo "[1/12] Discovering platform capabilities..."
python3 resources/scripts/api-connector.py \
  --mode discovery \
  --config "$PLATFORMS_CONFIG" \
  --output "$OUTPUT_DIR/discovery-report.json"

# STAGE 2: Authentication Strategy
echo "[2/12] Configuring authentication..."
cat > "$OUTPUT_DIR/configs/auth-config.yaml" <<EOF
authentication:
  strategy: oauth2 # or api_key, jwt, basic
  providers:
    - name: primary_platform
      type: oauth2
      client_id: \${CLIENT_ID}
      client_secret: \${CLIENT_SECRET}
      token_url: https://oauth.example.com/token
      scopes: [read, write]

  token_management:
    refresh_strategy: automatic
    expiry_buffer: 300 # seconds before expiry to refresh
    storage: redis # or memory, database
EOF

# STAGE 3: Integration Architecture Design
echo "[3/12] Designing integration architecture..."
cat > "$OUTPUT_DIR/docs/architecture.md" <<EOF
# Integration Architecture

## System Overview
\`\`\`
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│   Source    │─────▶│  Integration │─────▶│   Target    │
│  Platform   │◀─────│    Layer     │◀─────│  Platform   │
└─────────────┘      └──────────────┘      └─────────────┘
       │                    │                      │
       │                    ▼                      │
       │              ┌──────────┐                 │
       └──────────────│ Message  │─────────────────┘
                      │  Queue   │
                      └──────────┘
\`\`\`

## Components
1. **API Connectors**: Platform-specific clients
2. **Webhook Handlers**: Event processing pipeline
3. **Sync Engine**: Bidirectional data synchronization
4. **Error Handler**: Retry logic + dead letter queue
5. **Monitoring**: Metrics, logging, alerts

## Data Flow
- Inbound: Webhook → Validation → Transform → Queue → Process
- Outbound: Trigger → Fetch → Transform → API Call → Confirm
EOF

# STAGE 4: API Connector Implementation
echo "[4/12] Implementing API connectors..."
python3 resources/scripts/api-connector.py \
  --mode generate \
  --config "$PLATFORMS_CONFIG" \
  --output "$OUTPUT_DIR/connectors/" \
  --template resources/templates/api-connector-template.py

# Generate connector for each platform
PLATFORMS=$(cat "$PLATFORMS_CONFIG" | jq -r '.platforms[].name')
for PLATFORM in $PLATFORMS; do
  echo "  - Generating connector for: $PLATFORM"
  python3 resources/scripts/api-connector.py \
    --platform "$PLATFORM" \
    --output "$OUTPUT_DIR/connectors/${PLATFORM}_connector.py"
done

# STAGE 5: Webhook Handler Development
echo "[5/12] Developing webhook handlers..."
node resources/scripts/webhook-handler.js \
  --config "$PLATFORMS_CONFIG" \
  --output "$OUTPUT_DIR/handlers/" \
  --framework express # or fastify, koa

# Create webhook verification
cat > "$OUTPUT_DIR/handlers/webhook-verifier.js" <<'EOF'
const crypto = require('crypto');

class WebhookVerifier {
  constructor(secret) {
    this.secret = secret;
  }

  verifySignature(payload, signature, algorithm = 'sha256') {
    const hmac = crypto.createHmac(algorithm, this.secret);
    const expectedSignature = hmac.update(JSON.stringify(payload)).digest('hex');
    return crypto.timingSafeEqual(
      Buffer.from(signature),
      Buffer.from(expectedSignature)
    );
  }

  verifyTimestamp(timestamp, maxAge = 300) {
    const now = Math.floor(Date.now() / 1000);
    return Math.abs(now - timestamp) <= maxAge;
  }
}

module.exports = WebhookVerifier;
EOF

# STAGE 6: Data Transformation Pipelines
echo "[6/12] Setting up data transformation..."
cat > "$OUTPUT_DIR/sync/transformer.py" <<'EOF'
from typing import Dict, Any, List, Callable
import json
from datetime import datetime

class DataTransformer:
    def __init__(self, mapping_config: Dict[str, Any]):
        self.mapping = mapping_config
        self.transforms = {
            'uppercase': str.upper,
            'lowercase': str.lower,
            'timestamp': lambda x: datetime.now().isoformat(),
            'boolean': lambda x: str(x).lower() in ['true', '1', 'yes'],
        }

    def transform(self, source_data: Dict[str, Any]) -> Dict[str, Any]:
        result = {}
        for target_field, mapping in self.mapping.items():
            if isinstance(mapping, dict):
                source_field = mapping.get('source')
                transform = mapping.get('transform')
                default = mapping.get('default')

                value = source_data.get(source_field, default)

                if transform and transform in self.transforms:
                    value = self.transforms[transform](value)

                result[target_field] = value
            else:
                result[target_field] = source_data.get(mapping)

        return result

    def bulk_transform(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self.transform(record) for record in records]

EOF

# STAGE 7: Synchronization Engine Setup
echo "[7/12] Implementing synchronization engine..."
python3 resources/scripts/sync-manager.py \
  --config "$PLATFORMS_CONFIG" \
  --output "$OUTPUT_DIR/sync/sync_engine.py" \
  --strategy bidirectional

# Create sync configuration
cat > "$OUTPUT_DIR/configs/sync-config.yaml" <<EOF
synchronization:
  direction: bidirectional # or unidirectional
  frequency: real_time # or cron: "0 */6 * * *"
  batch_size: 1000

  conflict_resolution: # When records differ
    strategy: last_write_wins # or source_wins, target_wins, merge
    merge_fields: [updated_at, status]

  filters:
    include: # Only sync records matching criteria
      - field: status
        operator: in
        values: [active, pending]
    exclude:
      - field: deleted
        operator: eq
        value: true

  transformations:
    source_to_target:
      customer_id: user_id
      email_address: email
      phone_number: phone
    target_to_source:
      user_id: customer_id
      email: email_address
EOF

# STAGE 8: Error Handling & Retry Logic
echo "[8/12] Implementing error handling..."
cat > "$OUTPUT_DIR/handlers/error-handler.js" <<'EOF'
class ErrorHandler {
  constructor(config) {
    this.maxRetries = config.maxRetries || 3;
    this.retryDelay = config.retryDelay || 1000;
    this.deadLetterQueue = config.deadLetterQueue;
  }

  async handleWithRetry(operation, context = {}) {
    let lastError;

    for (let attempt = 1; attempt <= this.maxRetries; attempt++) {
      try {
        return await operation();
      } catch (error) {
        lastError = error;
        console.error(\`Attempt \${attempt} failed:', error.message);

        if (attempt < this.maxRetries) {
          const delay = this.retryDelay * Math.pow(2, attempt - 1); // Exponential backoff
          await this.sleep(delay);
        }
      }
    }

    // All retries failed - send to dead letter queue
    await this.sendToDeadLetter({
      error: lastError.message,
      context,
      timestamp: new Date().toISOString(),
      attempts: this.maxRetries
    });

    throw lastError;
  }

  async sendToDeadLetter(payload) {
    if (this.deadLetterQueue) {
      await this.deadLetterQueue.send(payload);
    }
  }

  sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

module.exports = ErrorHandler;
EOF

# STAGE 9: Monitoring & Observability
echo "[9/12] Setting up monitoring..."
cat > "$OUTPUT_DIR/monitoring/metrics.js" <<'EOF'
const prometheus = require('prom-client');

const register = new prometheus.Registry();

// API Metrics
const apiLatency = new prometheus.Histogram({
  name: 'api_request_duration_seconds',
  help: 'API request duration in seconds',
  labelNames: ['platform', 'endpoint', 'status'],
  buckets: [0.1, 0.5, 1, 2, 5, 10]
});

const apiErrors = new prometheus.Counter({
  name: 'api_errors_total',
  help: 'Total API errors',
  labelNames: ['platform', 'error_type']
});

// Webhook Metrics
const webhookReceived = new prometheus.Counter({
  name: 'webhooks_received_total',
  help: 'Total webhooks received',
  labelNames: ['platform', 'event_type']
});

const webhookProcessingTime = new prometheus.Histogram({
  name: 'webhook_processing_duration_seconds',
  help: 'Webhook processing duration',
  labelNames: ['platform', 'event_type']
});

// Sync Metrics
const syncRecords = new prometheus.Counter({
  name: 'sync_records_total',
  help: 'Total records synchronized',
  labelNames: ['direction', 'status']
});

const syncLatency = new prometheus.Histogram({
  name: 'sync_duration_seconds',
  help: 'Synchronization duration',
  buckets: [1, 5, 10, 30, 60, 300]
});

[apiLatency, apiErrors, webhookReceived, webhookProcessingTime, syncRecords, syncLatency]
  .forEach(metric => register.registerMetric(metric));

module.exports = { register, apiLatency, apiErrors, webhookReceived, webhookProcessingTime, syncRecords, syncLatency };
EOF

# STAGE 10: Testing & Validation
echo "[10/12] Creating integration tests..."
bash resources/scripts/integration-tester.sh \
  --connectors "$OUTPUT_DIR/connectors/" \
  --handlers "$OUTPUT_DIR/handlers/" \
  --output "$OUTPUT_DIR/tests/"

# STAGE 11: Documentation Generation
echo "[11/12] Generating documentation..."
cat > "$OUTPUT_DIR/docs/INTEGRATION-GUIDE.md" <<EOF
# Platform Integration Guide

## Overview
This integration connects multiple platforms with bidirectional data synchronization.

## Setup

### Prerequisites
- Node.js 18+
- Python 3.9+
- Redis (for caching/queues)
- PostgreSQL (optional, for state tracking)

### Installation
\`\`\`bash
npm install
pip install -r requirements.txt
\`\`\`

### Configuration
1. Copy \`.env.example\` to \`.env\`
2. Add platform credentials
3. Configure sync rules in \`configs/sync-config.yaml\`

## API Connectors

### Available Connectors
$(ls -1 "$OUTPUT_DIR/connectors/" | sed 's/^/- /')

### Usage Example
\`\`\`python
from connectors.salesforce_connector import SalesforceConnector

connector = SalesforceConnector(
    client_id=os.getenv('SF_CLIENT_ID'),
    client_secret=os.getenv('SF_CLIENT_SECRET')
)

# Fetch accounts
accounts = connector.get_accounts(limit=100)

# Create contact
contact = connector.create_contact({
    'FirstName': 'John',
    'LastName': 'Doe',
    'Email': 'john@example.com'
})
\`\`\`

## Webhook Handlers

### Registering Webhooks
\`\`\`bash
node handlers/register-webhooks.js \\
  --platform stripe \\
  --url https://your-domain.com/webhooks/stripe \\
  --events payment_intent.succeeded,charge.failed
\`\`\`

### Handling Events
Webhooks are automatically verified and processed by event type.

## Data Synchronization

### Manual Sync
\`\`\`bash
python sync/sync_engine.py \\
  --source salesforce \\
  --target hubspot \\
  --entity contacts
\`\`\`

### Automated Sync
Runs based on \`sync-config.yaml\` frequency.

## Monitoring

### Metrics Endpoint
\`\`\`
GET /metrics
\`\`\`

### Health Check
\`\`\`
GET /health
\`\`\`

### Key Metrics
- API latency (p50, p95, p99)
- Webhook success rate
- Sync throughput
- Error rate by platform

## Troubleshooting

### Common Issues
1. **Authentication Failed**: Check credentials in \`.env\`
2. **Webhook Verification Failed**: Verify webhook secret matches
3. **Sync Conflicts**: Review conflict resolution strategy
4. **Rate Limiting**: Implement backoff in connector

### Logs
\`\`\`bash
tail -f logs/integration.log
\`\`\`

---
🤖 Generated by Platform Integration Skill
EOF

# STAGE 12: Production Deployment
echo "[12/12] Preparing production deployment..."

# Create Docker configuration
cat > "$OUTPUT_DIR/Dockerfile" <<'EOF'
FROM node:18-alpine

WORKDIR /app

COPY package*.json ./
RUN npm ci --only=production

COPY . .

ENV NODE_ENV=production
EXPOSE 3000

CMD ["node", "server.js"]
EOF

cat > "$OUTPUT_DIR/docker-compose.yml" <<'EOF'
version: '3.8'

services:
  integration:
    build: .
    ports:
      - "3000:3000"
    environment:
      - NODE_ENV=production
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data

volumes:
  redis-data:
EOF

# Production readiness check
echo ""
echo "================================================================"
echo "Platform Integration Complete!"
echo "================================================================"
echo ""
echo "Artifacts in: $OUTPUT_DIR/"
echo "- Connectors: connectors/ ($(ls -1 "$OUTPUT_DIR/connectors/" | wc -l) platforms)"
echo "- Handlers: handlers/ (webhook processing)"
echo "- Sync Engine: sync/ (bidirectional sync)"
echo "- Tests: tests/ (integration suite)"
echo "- Configs: configs/ (YAML configurations)"
echo "- Documentation: docs/INTEGRATION-GUIDE.md"
echo "- Monitoring: monitoring/ (Prometheus metrics)"
echo ""
echo "Next Steps:"
echo "1. Configure credentials in .env"
echo "2. Run tests: npm test"
echo "3. Start services: docker-compose up -d"
echo "4. Register webhooks: node handlers/register-webhooks.js"
echo "5. Monitor: http://localhost:3000/metrics"
echo ""
```

## Integration Points

### Cascades
- Part of `/enterprise-integration` cascade
- Used by `/api-orchestration` cascade
- Invokes `/monitoring-setup` cascade

### Commands
- Uses: `/api-connector`, `/webhook-handler`
- Uses: `/sync-manager`, `/integration-tester`
- Creates: `/health-check`, `/metrics-endpoint`

### Other Skills
- Invokes: `security` (for auth/secrets)
- Invokes: `monitoring` (for observability)
- Output to: `documentation` (for API docs)

## Usage Example

```bash
# Full platform integration
platform-integration full platforms-config.json

# API-only integration
platform-integration api_integration platforms-config.json

# Webhook automation
platform-integration webhook_automation platforms-config.json

# Data synchronization only
platform-integration data_sync platforms-config.json
```

## Failure Modes

- **Auth failure**: Verify credentials, check token expiry
- **Webhook verification failed**: Check secret configuration
- **Sync conflicts**: Review conflict resolution strategy
- **Rate limiting**: Implement exponential backoff
- **Network timeout**: Increase timeout, check connectivity
- **Data transformation error**: Validate mapping configuration

## Advanced Features

### Multi-Region Deployment
```yaml
regions:
  - name: us-east-1
    primary: true
  - name: eu-west-1
    failover: true
```

### Event Streaming
```yaml
streaming:
  provider: kafka
  topics:
    - platform-events
    - sync-updates
```

### Custom Transformations
```python
def custom_transform(data):
    # Your transformation logic
    return transformed_data

transformer.register_custom('my_transform', custom_transform)
```