# Example 2: Multi-Repository Swarm Review

## Scenario Overview

**Context**: A microservices architecture requires coordinated changes across 3 repositories for a new payment processing feature. Changes must be reviewed together to ensure API contract compatibility, data consistency, and security compliance.

**Repositories Involved**:
1. `company/payment-service` (Node.js/Express) - New payment processing logic
2. `company/order-service` (Python/FastAPI) - Order status integration
3. `company/notification-service` (Go) - Payment confirmation emails

**Challenge**: Traditional single-repo reviews miss cross-service issues like:
- API contract mismatches (schema drift)
- Distributed transaction failures
- Inconsistent error handling
- Security vulnerabilities across boundaries

**Solution**: Multi-repository swarm review with mesh topology for cross-service validation.

---

## Architecture: Mesh Swarm Topology

### Swarm Design

```
                    ┌─────────────────────┐
                    │  Mesh Coordinator   │
                    │  (Hierarchical)     │
                    └──────────┬──────────┘
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
    ┌─────▼─────┐       ┌─────▼─────┐       ┌─────▼─────┐
    │  Payment  │◄─────►│   Order   │◄─────►│   Notify  │
    │  Reviewer │       │  Reviewer │       │  Reviewer │
    │  Swarm    │       │   Swarm   │       │   Swarm   │
    └─────┬─────┘       └─────┬─────┘       └─────┬─────┘
          │                   │                    │
    ┌─────┴─────┐       ┌─────┴─────┐       ┌─────┴─────┐
    │ Security  │       │ Contract  │       │   E2E     │
    │ Analyst   │       │ Validator │       │  Tester   │
    └───────────┘       └───────────┘       └───────────┘
```

**Key Features**:
- **Mesh Topology**: Each service swarm can communicate directly (peer-to-peer)
- **Shared Memory**: Cross-service context via Memory MCP
- **Contract Validation**: Automatic OpenAPI schema drift detection
- **E2E Testing**: Full workflow validation across services

---

## Step 1: Initialize Multi-Repo Swarm

### Repository Setup

```bash
# Clone all repositories
mkdir ~/projects/payment-feature-review
cd ~/projects/payment-feature-review

git clone https://github.com/company/payment-service
git clone https://github.com/company/order-service
git clone https://github.com/company/notification-service

# Checkout feature branches
cd payment-service && git checkout feature/stripe-integration && cd ..
cd order-service && git checkout feature/payment-status-webhook && cd ..
cd notification-service && git checkout feature/payment-emails && cd ..
```

### Swarm Initialization

```bash
# Initialize mesh swarm for multi-repo coordination
npx claude-flow@alpha swarm init \
  --topology mesh \
  --max-agents 12 \
  --strategy adaptive \
  --session-id payment-feature-multi-repo

# Expected Output:
# ✓ Swarm initialized: swarm-payment-multi-repo-20251102
# ✓ Topology: Mesh (12 agents, peer-to-peer communication)
# ✓ Cross-repo context enabled via Memory MCP
# ✓ Session ID: payment-feature-multi-repo
```

---

## Step 2: Spawn Repository-Specific Swarms

### Payment Service Reviewers (4 agents)

```javascript
[Single Message - Parallel Agent Execution]:

  // Payment service security specialist
  Task("Payment Security Auditor",
    "Review Stripe integration security in payment-service. Check API key management, PCI DSS compliance, webhook signature verification, and HTTPS enforcement. Store security checklist in memory with key 'security/payment-service'.",
    "analyst")

  // Payment API contract validator
  Task("Payment API Validator",
    "Extract OpenAPI schema from payment-service endpoints (/process-payment, /refund). Store schema in memory with key 'contracts/payment-service'. Validate against prior version for breaking changes.",
    "researcher")

  // Payment code quality reviewer
  Task("Payment Code Reviewer",
    "Review code quality in payment-service: error handling, retry logic, idempotency keys, transaction logging. Check async/await patterns and database transaction management.",
    "analyst")

  // Payment performance analyst
  Task("Payment Performance Analyst",
    "Benchmark payment processing latency. Check Stripe API call timeout configuration, database query optimization for payment logs, and memory usage during concurrent payments.",
    "optimizer")
```

### Order Service Reviewers (4 agents)

```javascript
[Single Message - Parallel Agent Execution]:

  // Order webhook handler specialist
  Task("Order Webhook Specialist",
    "Review webhook handler in order-service for payment status updates. Verify webhook authentication, duplicate event handling (idempotency), and database update atomicity. Store integration points in memory with key 'webhooks/order-service'.",
    "analyst")

  // Order API contract validator
  Task("Order API Validator",
    "Extract OpenAPI schema from order-service endpoints (/orders/:id/payment-status). Store schema in memory with key 'contracts/order-service'. Cross-reference with payment-service schema for consistency.",
    "researcher")

  // Order data consistency analyst
  Task("Order Data Consistency Analyst",
    "Review distributed transaction handling between payment and order services. Check for race conditions, eventual consistency patterns, and rollback mechanisms on payment failure.",
    "analyst")

  // Order testing specialist
  Task("Order Test Specialist",
    "Review test coverage for payment webhook scenarios: successful payment, failed payment, refund, duplicate webhook. Verify integration tests with payment-service mocks.",
    "tester")
```

### Notification Service Reviewers (4 agents)

```javascript
[Single Message - Parallel Agent Execution]:

  // Notification template reviewer
  Task("Email Template Specialist",
    "Review payment confirmation email templates. Check for sensitive data exposure (no full card numbers), proper formatting, and localization support. Verify SMTP security (TLS).",
    "analyst")

  // Notification API validator
  Task("Notification API Validator",
    "Extract API schema for notification-service trigger endpoint (/send-payment-confirmation). Store in memory with key 'contracts/notification-service'. Validate expected payload format.",
    "researcher")

  // Notification reliability analyst
  Task("Notification Reliability Analyst",
    "Review email delivery reliability: retry logic for SMTP failures, queue-based async processing, dead-letter queue for failed emails, and delivery status tracking.",
    "optimizer")

  // Notification testing specialist
  Task("Notification Test Specialist",
    "Review test coverage: email content validation, SMTP connection mocking, delivery confirmation, and error handling for network failures.",
    "tester")
```

---

## Step 3: Cross-Service Contract Validation

### Automated Schema Extraction and Comparison

**Agent**: Cross-Service Contract Validator
**Type**: Mesh coordinator agent (accesses all repos)

```bash
# Agent executes in payment-service repo
cd ~/projects/payment-feature-review/payment-service

# Extract OpenAPI schema from code annotations
npx swagger-jsdoc -d swaggerDef.js src/routes/*.js > schemas/payment-api.json

# Store in shared memory
npx claude-flow@alpha memory store \
  --key "contracts/payment-service/v2.3.0" \
  --value "$(cat schemas/payment-api.json)"

# Agent executes in order-service repo
cd ~/projects/payment-feature-review/order-service

# Extract FastAPI schema
python -c "from app.main import app; import json; print(json.dumps(app.openapi()))" > schemas/order-api.json

# Store in shared memory
npx claude-flow@alpha memory store \
  --key "contracts/order-service/v1.8.0" \
  --value "$(cat schemas/order-api.json)"

# Agent executes in notification-service repo
cd ~/projects/payment-feature-review/notification-service

# Extract Go API schema
go run scripts/generate-openapi.go > schemas/notification-api.json

# Store in shared memory
npx claude-flow@alpha memory store \
  --key "contracts/notification-service/v1.2.0" \
  --value "$(cat schemas/notification-api.json)"
```

### Contract Drift Analysis

**Agent Output**: Contract Validator Report

```markdown
## Cross-Service API Contract Analysis

### Schema Compatibility Matrix

| Source Service | Endpoint | Target Service | Endpoint | Status | Issues |
|----------------|----------|----------------|----------|--------|--------|
| payment-service | POST /process-payment | order-service | POST /orders/:id/payment-status | ⚠️ DRIFT | Field mismatch |
| order-service | GET /orders/:id | notification-service | POST /send-payment-confirmation | ✅ COMPATIBLE | None |
| payment-service | POST /refund | order-service | POST /orders/:id/payment-status | ✅ COMPATIBLE | None |

---

### ⚠️ CRITICAL: Schema Drift Detected

**Issue**: Payment service sends `transaction_id` but order service expects `payment_transaction_id`

**Payment Service Schema** (`payment-service/schemas/payment-api.json`):
```json
{
  "paths": {
    "/process-payment": {
      "post": {
        "requestBody": {
          "content": {
            "application/json": {
              "schema": {
                "type": "object",
                "properties": {
                  "order_id": { "type": "string" },
                  "amount": { "type": "number" },
                  "transaction_id": { "type": "string" }  // ← MISMATCH
                }
              }
            }
          }
        }
      }
    }
  }
}
```

**Order Service Expected Schema** (`order-service/schemas/order-api.json`):
```json
{
  "paths": {
    "/orders/{id}/payment-status": {
      "post": {
        "requestBody": {
          "content": {
            "application/json": {
              "schema": {
                "type": "object",
                "properties": {
                  "order_id": { "type": "string" },
                  "amount": { "type": "number" },
                  "payment_transaction_id": { "type": "string" }  // ← EXPECTS THIS
                }
              }
            }
          }
        }
      }
    }
  }
}
```

**Impact**: Order service will reject payment webhooks → payment processed but order not updated

**Fix Required**:
1. **Option A**: Update payment-service to send `payment_transaction_id`
2. **Option B**: Update order-service to accept `transaction_id` and map internally
3. **Option C**: Create API gateway adapter layer (recommended for legacy compatibility)

**Recommendation**: Option B (order-service update) - payment-service naming is simpler and used by 3 other services

---

### Auto-Fix Suggestion

**File**: `order-service/app/routes/orders.py`

```diff
# Webhook handler for payment status updates
@router.post("/orders/{order_id}/payment-status")
async def update_payment_status(order_id: str, payload: PaymentStatusPayload):
+   # Map incoming field names for backward compatibility
+   if "transaction_id" in payload and "payment_transaction_id" not in payload:
+       payload.payment_transaction_id = payload.transaction_id
+
    order = await db.orders.find_one({"_id": order_id})
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")

-   order.payment_transaction_id = payload.payment_transaction_id
+   order.payment_transaction_id = payload.payment_transaction_id or payload.transaction_id
    order.payment_status = payload.status
    await db.orders.update_one({"_id": order_id}, {"$set": order.dict()})

    return {"status": "updated"}
```

**Testing**: Add integration test to verify both field names work

```python
# tests/test_payment_webhook.py
@pytest.mark.asyncio
async def test_payment_webhook_with_transaction_id():
    """Test webhook accepts 'transaction_id' field (legacy compatibility)"""
    response = await client.post(
        "/orders/ORD-12345/payment-status",
        json={
            "order_id": "ORD-12345",
            "amount": 99.99,
            "transaction_id": "txn_abc123",  # Old field name
            "status": "completed"
        }
    )
    assert response.status_code == 200

    order = await db.orders.find_one({"_id": "ORD-12345"})
    assert order["payment_transaction_id"] == "txn_abc123"
```
```

---

## Step 4: End-to-End Workflow Testing

### E2E Test Agent Execution

**Agent**: E2E Testing Specialist
**Task**: Validate complete payment flow across all 3 services

```bash
# Agent spawns E2B sandbox with all services running
npx claude-flow@alpha e2b-sandbox create \
  --template docker-compose \
  --services payment,order,notification \
  --network bridge

# Start services in sandbox
docker-compose up -d

# Wait for health checks
./scripts/wait-for-services.sh

# Run E2E test workflow
npm run test:e2e:payment-flow
```

### E2E Test Scenario

```javascript
// tests/e2e/payment-flow.test.js

describe('Complete Payment Flow - Multi-Service Integration', () => {

  test('Successful payment triggers order update and email', async () => {
    // Step 1: Create order via order-service
    const orderResponse = await axios.post('http://order-service:3001/orders', {
      customer_id: 'CUST-123',
      items: [{ sku: 'WIDGET-001', quantity: 2, price: 49.99 }],
      total: 99.98
    });
    const orderId = orderResponse.data.order_id;
    expect(orderId).toBeDefined();

    // Step 2: Process payment via payment-service
    const paymentResponse = await axios.post('http://payment-service:3000/process-payment', {
      order_id: orderId,
      amount: 99.98,
      payment_method: 'card',
      card_token: 'tok_visa_test'
    });
    expect(paymentResponse.status).toBe(200);
    const transactionId = paymentResponse.data.transaction_id;

    // Step 3: Verify order status updated (via webhook)
    await sleep(2000); // Wait for webhook processing
    const orderStatusResponse = await axios.get(`http://order-service:3001/orders/${orderId}`);
    expect(orderStatusResponse.data.payment_status).toBe('completed');
    expect(orderStatusResponse.data.payment_transaction_id).toBe(transactionId);

    // Step 4: Verify email sent (check notification-service logs)
    const notificationLogs = await getServiceLogs('notification-service');
    expect(notificationLogs).toContain(`Email sent to customer CUST-123`);
    expect(notificationLogs).toContain(`Transaction: ${transactionId}`);

    // Step 5: Verify idempotency - duplicate webhook ignored
    await axios.post(`http://order-service:3001/orders/${orderId}/payment-status`, {
      order_id: orderId,
      transaction_id: transactionId,
      status: 'completed'
    });
    const duplicateCheckLogs = await getServiceLogs('order-service');
    expect(duplicateCheckLogs).toContain('Duplicate webhook ignored');
  });

  test('Failed payment does NOT update order or send email', async () => {
    const orderResponse = await axios.post('http://order-service:3001/orders', {
      customer_id: 'CUST-456',
      items: [{ sku: 'WIDGET-002', quantity: 1, price: 29.99 }],
      total: 29.99
    });
    const orderId = orderResponse.data.order_id;

    // Trigger payment failure with test card
    try {
      await axios.post('http://payment-service:3000/process-payment', {
        order_id: orderId,
        amount: 29.99,
        payment_method: 'card',
        card_token: 'tok_chargeDeclined'  // Test card that always fails
      });
    } catch (error) {
      expect(error.response.status).toBe(402); // Payment Required
    }

    // Verify order status remains 'pending'
    const orderStatusResponse = await axios.get(`http://order-service:3001/orders/${orderId}`);
    expect(orderStatusResponse.data.payment_status).toBe('pending');

    // Verify NO email sent
    const notificationLogs = await getServiceLogs('notification-service');
    expect(notificationLogs).not.toContain(`customer CUST-456`);
  });

});
```

### E2E Test Results

```markdown
## E2E Test Report - Payment Flow Integration

**Test Suite**: Multi-Service Payment Feature
**Services**: payment-service, order-service, notification-service
**Environment**: E2B Docker Compose Sandbox
**Duration**: 8m 23s

### Results Summary

✅ **PASSED**: 12/12 tests (100%)

| Test Case | Duration | Status |
|-----------|----------|--------|
| Successful payment triggers order update and email | 4.2s | ✅ PASS |
| Failed payment does NOT update order or send email | 2.8s | ✅ PASS |
| Refund updates order status and sends refund email | 3.1s | ✅ PASS |
| Duplicate webhook is idempotent (no duplicate updates) | 1.9s | ✅ PASS |
| Payment timeout triggers retry logic | 5.7s | ✅ PASS |
| Concurrent payments to same order are serialized | 6.4s | ✅ PASS |
| SMTP failure queues email for retry | 3.6s | ✅ PASS |
| Webhook signature validation rejects tampered requests | 1.2s | ✅ PASS |
| Database transaction rollback on payment failure | 2.3s | ✅ PASS |
| Order status query returns correct payment state | 0.8s | ✅ PASS |
| Email template contains correct transaction details | 1.4s | ✅ PASS |
| API rate limiting prevents payment spam | 4.1s | ✅ PASS |

### Performance Metrics

- **Average Response Time**: 287ms (payment processing)
- **P95 Latency**: 412ms
- **Throughput**: 34 payments/second (stress test)
- **Error Rate**: 0.02% (2 failures in 10,000 requests)

### Coverage

- **Code Coverage**: 89% (across all 3 services)
- **API Contract Coverage**: 100% (all endpoints tested)
- **Error Scenarios**: 8/8 covered

**Recommendation**: ✅ **ALL TESTS PASSED - READY FOR DEPLOYMENT**
```

---

## Step 5: Consolidated Multi-Repo Review Report

### GitHub Multi-PR Summary Comment

```markdown
## 🤖 Multi-Repository AI Swarm Review - Payment Feature

**Swarm Topology**: Mesh (12 agents across 3 repos)
**Review Duration**: 14m 37s
**Services Reviewed**: payment-service, order-service, notification-service
**Overall Status**: ⚠️ **CHANGES REQUESTED**

---

### 🔴 CRITICAL Cross-Service Issue

**Schema Drift: Payment Transaction ID Field Mismatch**

- **Repos Affected**: `payment-service` ↔️ `order-service`
- **Issue**: Field name inconsistency (`transaction_id` vs `payment_transaction_id`)
- **Impact**: Order updates will fail when payment completes
- **Fix Applied**: Auto-fix committed to `order-service` (backward compatible mapping)
- **PR**: [order-service#789 - Fix payment field compatibility](https://github.com/company/order-service/pull/789)

---

### ⚠️ Security Issues by Repository

#### Payment Service (3 issues)
1. ⚠️ **Stripe API Key in Environment** - Use secret manager (AWS Secrets Manager / HashiCorp Vault)
2. ⚠️ **Webhook Signature Not Verified** - Add Stripe signature validation
3. ℹ️ **Idempotency Key Not Persisted** - Store keys in Redis to prevent duplicate charges

#### Order Service (1 issue)
1. ⚠️ **Webhook Endpoint Not Rate Limited** - Add rate limiting (10 req/min per IP)

#### Notification Service (2 issues)
1. ⚠️ **SMTP Password in Config File** - Move to environment variable
2. ℹ️ **Email Templates Not Sanitized** - Add HTML escape for user data

---

### ✅ E2E Integration Tests: PASSED (12/12)

- [x] Successful payment flow (payment → order → email)
- [x] Failed payment handling (no order update, no email)
- [x] Refund flow
- [x] Duplicate webhook idempotency
- [x] Concurrent payment handling
- [x] Database transaction rollback
- [x] SMTP retry logic
- [x] Webhook signature validation

**Performance**: 287ms avg, 34 payments/sec throughput

---

### 📊 Service-Specific Metrics

| Service | Security | Quality | Tests | Performance | Status |
|---------|----------|---------|-------|-------------|--------|
| payment-service | 7/10 | 9/10 | 91% | 8/10 | ⚠️ Security fixes needed |
| order-service | 8/10 | 10/10 | 87% | 9/10 | ✅ Good after field fix |
| notification-service | 7/10 | 9/10 | 84% | 10/10 | ⚠️ SMTP security |

---

### 🔧 Auto-Fix Commits Created

1. **order-service**: [PR #789](https://github.com/company/order-service/pull/789) - Backward compatible field mapping
2. **payment-service**: [PR #234](https://github.com/company/payment-service/pull/234) - Webhook signature verification
3. **notification-service**: [PR #456](https://github.com/company/notification-service/pull/456) - Move SMTP password to env

---

### 📝 Next Steps

1. **Developers**: Review and merge auto-fix PRs (#789, #234, #456)
2. **DevOps**: Configure secret managers for API keys (payment + notification services)
3. **QA**: Re-run E2E tests after security fixes applied
4. **Release Manager**: Coordinate deployment (order-service first, then payment, then notification)

**Estimated Time to Production**: 1-2 days (after security fixes)

---

*Generated by Multi-Repo AI Swarm Review | Mesh Topology | 12 Specialist Agents*
```

---

## Outcomes and Benefits

### Multi-Repository Coordination

**Traditional Approach** (Manual):
- 3 separate PR reviews by different engineers
- Schema mismatches discovered in QA/staging (week 3)
- Integration failures in production (rollback required)
- Total time: 3-4 weeks

**AI Swarm Approach**:
- Simultaneous review of all 3 repos in 14 minutes
- Schema drift detected before merge
- E2E tests validate integration early
- Total time: 2 days (with fixes)

### Cost Savings

- **Engineering Time**: 3 senior engineers × 2 hours each = 6 hours → **14 minutes**
- **QA Time**: 4 hours integration testing → **8 minutes** (automated E2E)
- **Rollback Cost**: $0 (issues caught before production)
- **ROI**: 95% time reduction, ~$5,000 saved per feature

### Quality Improvements

- **Cross-Service Issues Caught**: 1 CRITICAL schema drift (would have caused production incident)
- **Security Issues Caught**: 6 issues across 3 services
- **Test Coverage**: 89% average (vs 72% before AI review)
- **E2E Integration**: 12/12 tests passing before merge

---

## Advanced Tips for Multi-Repo Reviews

### 1. Shared Memory for Context

```bash
# Store cross-repo requirements in shared memory
npx claude-flow@alpha memory store \
  --key "multi-repo/payment-feature/requirements" \
  --value '{
    "api_contracts": ["payment-service", "order-service", "notification-service"],
    "security_standards": "PCI DSS Level 1",
    "performance_target": "< 300ms P95",
    "deployment_order": ["order-service", "payment-service", "notification-service"]
  }'

# Agents auto-retrieve during review
```

### 2. Mesh vs Hierarchical Topology

```bash
# Use MESH for peer-to-peer coordination (3-5 repos)
npx claude-flow@alpha swarm init --topology mesh

# Use HIERARCHICAL for large-scale reviews (6+ repos)
npx claude-flow@alpha swarm init --topology hierarchical
```

### 3. Contract-First Development

```bash
# Generate OpenAPI schemas from code before review
npm run generate:openapi

# Store schemas in git for version tracking
git add schemas/*.json
git commit -m "chore: update API schemas for v2.3.0"

# AI agents compare schema versions automatically
```

### 4. Distributed Tracing Integration

```javascript
// Add OpenTelemetry to E2E tests for request flow visualization
const { trace } = require('@opentelemetry/api');

test('Payment flow with distributed tracing', async () => {
  const tracer = trace.getTracer('e2e-tests');

  await tracer.startActiveSpan('payment-flow-e2e', async (span) => {
    // Create order (trace ID propagates across services)
    const order = await createOrder({ traceId: span.spanContext().traceId });

    // Process payment
    const payment = await processPayment(order.id);

    // Verify order updated
    const updatedOrder = await getOrder(order.id);

    span.end();
  });

  // Export trace to Jaeger/Zipkin for visualization
});
```

---

## Conclusion

Multi-repository swarm reviews solve the critical challenge of **coordinated changes across microservices**. By using mesh topology and shared memory, AI agents can:

1. Detect cross-service contract drift
2. Validate end-to-end integration flows
3. Apply consistent security standards
4. Generate auto-fix commits across repos

This approach is essential for modern distributed architectures where a single feature spans multiple codebases. The 95% time reduction and early issue detection make it a game-changer for teams building microservices.
