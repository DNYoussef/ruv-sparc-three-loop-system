---
name: integration-test
category: testing
version: 1.0.0
---

# /integration-test

Integration testing for API contracts, service interactions, and external dependencies.

## Usage
```bash
/integration-test [scope] [options]
```

## Parameters
- `scope` - Test scope: all|api|database|external|services (default: all)
- `--environment` - Test environment: local|staging|production (default: staging)
- `--parallel` - Run tests in parallel (default: true)
- `--timeout` - Test timeout in seconds (default: 30)
- `--retry` - Retry failed tests (default: 2)
- `--mock-external` - Mock external services (default: false)
- `--report-format` - Output format: html|json|junit (default: html)

## What It Does

**Comprehensive Integration Testing**:
1. 🔗 **API Integration**: REST/GraphQL endpoint testing
2. 🗄️ **Database Integration**: CRUD operations, transactions
3. 🌐 **External Services**: Third-party API integration
4. 📨 **Message Queues**: Event-driven communication
5. 🔐 **Authentication**: OAuth, JWT, session management
6. 💾 **Cache Integration**: Redis, Memcached testing
7. 📊 **Contract Testing**: API schema validation
8. 🔄 **State Management**: Multi-service workflows

**Testing Patterns**:
- **Consumer Contract**: Verify API responses match expectations
- **Provider Contract**: Ensure API adheres to contracts
- **Transaction Testing**: Multi-step database operations
- **Idempotency**: Repeated requests produce same result
- **Circuit Breaker**: Failure handling and recovery

## Examples

```bash
# Full integration test suite
/integration-test

# API integration tests only
/integration-test api

# Database integration with retry
/integration-test database --retry 3

# Test against staging environment
/integration-test --environment staging

# Mock external services for faster tests
/integration-test --mock-external true

# Parallel execution with custom timeout
/integration-test --parallel true --timeout 60

# External service integration
/integration-test external --environment production
```

## Output

```
🔗 Integration Test Suite Started

Environment: staging
Scope: all
Parallel: 4 workers
Mock External: false

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
API Integration Tests (45 endpoints)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Authentication & Authorization:
    ✅ POST /api/auth/register: 201 Created
       Request: { email, password, name }
       Response: { user, token }
       Duration: 234ms

    ✅ POST /api/auth/login: 200 OK
       Contract: Valid JWT token in response
       Duration: 156ms

    ✅ POST /api/auth/refresh: 200 OK
       Refresh token rotation: Working
       Duration: 89ms

    ❌ POST /api/auth/forgot-password: 500 Internal Server Error
       Expected: 200 OK
       Actual: Database connection timeout
       Error: ECONNREFUSED postgresql://localhost:5432
       Retrying (1/2)...
       ✅ Retry successful: 200 OK (567ms)

  User Management:
    ✅ GET /api/users: 200 OK
       Pagination: Working (page 1/5, 20 items)
       Duration: 78ms

    ✅ GET /api/users/{id}: 200 OK
       Contract validation: ✅ Schema matches
       Duration: 45ms

    ✅ PUT /api/users/{id}: 200 OK
       Idempotency: ✅ Multiple requests identical
       Duration: 123ms

    ✅ DELETE /api/users/{id}: 204 No Content
       Cascade delete: ✅ Related records removed
       Duration: 167ms

  Order Management:
    ✅ POST /api/orders: 201 Created
       Transaction integrity: ✅ Rollback on payment failure
       Duration: 345ms

    ✅ GET /api/orders/{id}: 200 OK
       Contract: ✅ All required fields present
       Duration: 89ms

  Result: 44/45 passed (97.8%)
  Duration: 12.4s

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Database Integration Tests (28 operations)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Connection Pool:
    ✅ PostgreSQL: Connected (pool: 10/20)
    ✅ Redis: Connected (1 replica)
    ✅ MongoDB: Connected (3-node replica set)

  CRUD Operations:
    ✅ INSERT: User created successfully
    ✅ SELECT: Query returns correct data
    ✅ UPDATE: Row updated, version incremented
    ✅ DELETE: Soft delete working (deleted_at set)

  Transaction Testing:
    ✅ COMMIT: Multi-table insert committed
    ✅ ROLLBACK: Failed transaction rolled back
    ✅ SAVEPOINT: Nested transaction working
    ✅ ISOLATION: Read committed isolation level

  Performance:
    ✅ Bulk insert: 1,000 rows in 234ms
    ✅ Complex join: 5-table join in 89ms
    ✅ Index usage: Query using index (not seq scan)

  Data Integrity:
    ✅ Foreign keys: Constraint violations handled
    ✅ Unique constraints: Duplicates rejected
    ✅ Check constraints: Invalid data rejected
    ✅ Triggers: Audit log populated

  Result: 28/28 passed (100%)
  Duration: 8.7s

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
External Service Integration (12 services)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Payment Gateway (Stripe):
    ✅ Create payment intent: Success
       Amount: $99.99
       Response time: 456ms
    ✅ Confirm payment: Success
    ✅ Refund payment: Success (partial refund)
    ✅ Webhook handling: Verified signature

  Email Service (SendGrid):
    ✅ Send transactional email: Delivered
       To: test@example.com
       Template: welcome-email
       Response time: 234ms
    ✅ Batch send: 100 emails queued

  Cloud Storage (AWS S3):
    ✅ Upload file: Success
       Size: 2.3 MB
       URL: https://s3.amazonaws.com/bucket/file.jpg
    ✅ Download file: Success
    ✅ Delete file: Success
    ✅ Presigned URL: Valid (expires in 1h)

  SMS Service (Twilio):
    ⚠️  Send SMS: Rate limit hit (429 Too Many Requests)
       Retry after: 60 seconds
       Circuit breaker: OPEN (will retry after cooldown)

  Analytics (Segment):
    ✅ Track event: Success
    ✅ Identify user: Success

  Result: 11/12 passed (91.7%)
  Duration: 15.3s

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Message Queue Integration (8 patterns)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  RabbitMQ:
    ✅ Publish message: Success
    ✅ Consume message: Received and processed
    ✅ Dead letter queue: Failed messages routed
    ✅ Message TTL: Expired messages cleaned up

  Redis Pub/Sub:
    ✅ Publish event: Success
    ✅ Subscribe to channel: Received event
    ✅ Pattern subscription: Wildcard working

  Kafka:
    ✅ Produce message: Acknowledged
    ✅ Consume from topic: Message received
    ⚠️  Consumer lag: 234 messages behind
       Recommend: Scale consumers

  Result: 8/8 passed (100%)
  Duration: 6.2s

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Service-to-Service Integration (15 flows)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  User Service → Order Service:
    ✅ Create order for user: Success
    ✅ User validation: Verified
    ✅ Authorization: User has permission

  Order Service → Payment Service:
    ✅ Process payment: Success
    ✅ Transaction recorded: Both services updated
    ✅ Idempotency: Duplicate request rejected

  Order Service → Inventory Service:
    ✅ Reserve inventory: Success
    ✅ Release on failure: Compensating transaction

  Notification Service:
    ✅ Order confirmation email: Sent
    ✅ SMS notification: Sent
    ✅ Push notification: Delivered

  Analytics Service:
    ✅ Event tracking: Success
    ✅ User metrics updated: Success

  Result: 15/15 passed (100%)
  Duration: 18.9s

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Cache Integration Tests (10 scenarios)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Redis:
    ✅ SET/GET: Value cached and retrieved
    ✅ SETEX: TTL working (expires after 60s)
    ✅ DEL: Cache invalidation successful
    ✅ INCR: Atomic increment working
    ✅ HSET/HGET: Hash operations working

  Cache Strategies:
    ✅ Cache-aside: Read-through working
    ✅ Write-through: Cache updated on write
    ✅ Write-behind: Async write queued
    ✅ Cache warming: Data preloaded
    ✅ Eviction policy: LRU working

  Result: 10/10 passed (100%)
  Duration: 4.1s

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Authentication Integration (12 flows)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  JWT Authentication:
    ✅ Token generation: Valid JWT
    ✅ Token validation: Signature verified
    ✅ Token refresh: New token issued
    ✅ Token revocation: Blacklist working

  OAuth 2.0:
    ✅ Authorization code flow: Success
    ✅ Token exchange: Access token received
    ✅ Refresh token: New access token issued

  Session Management:
    ✅ Session creation: Cookie set
    ✅ Session validation: User authenticated
    ✅ Session expiration: Auto logout
    ✅ Session persistence: Stored in Redis

  Multi-factor Auth:
    ✅ TOTP generation: 6-digit code
    ✅ TOTP validation: Code verified

  Result: 12/12 passed (100%)
  Duration: 9.8s

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Integration Test Summary
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Status: ✅ SUCCESS (with warnings)
Total Tests: 130
Passed: 128 (98.5%)
Failed: 0
Warnings: 2

Test Breakdown:
  ✅ API Integration: 44/45 (97.8%)
  ✅ Database: 28/28 (100%)
  ✅ External Services: 11/12 (91.7%)
  ✅ Message Queues: 8/8 (100%)
  ✅ Service-to-Service: 15/15 (100%)
  ✅ Cache: 10/10 (100%)
  ✅ Authentication: 12/12 (100%)

Warnings:
  ⚠️  Twilio SMS: Rate limit hit (circuit breaker active)
  ⚠️  Kafka consumer: High lag detected (234 messages)

Performance Metrics:
  Average response time: 187ms
  Slowest test: Order creation (345ms)
  Fastest test: User GET (45ms)
  Total duration: 1m 15s

Retry Statistics:
  Tests retried: 1
  Successful retries: 1 (100%)

Report Generated:
  HTML: reports/integration-test-2025-11-01.html
  JUnit: reports/integration-test-2025-11-01.xml
  JSON: reports/integration-test-2025-11-01.json

Recommendations:
  1. Investigate Twilio rate limit (increase quota or add backoff)
  2. Scale Kafka consumers to reduce lag
  3. Monitor database connection timeouts
  4. Consider caching frequently accessed endpoints

✅ Integration Tests Complete!
```

## Chains With

```bash
# Integration → E2E → regression
/integration-test && /e2e-test && /regression-test

# Test → deploy pipeline
/integration-test && /k8s-deploy --environment staging

# Full test suite
/integration-test && /load-test && /smoke-test

# Pre-production validation
/integration-test --environment production --mock-external false
```

## See Also
- `/e2e-test` - End-to-end testing
- `/regression-test` - Regression testing
- `/load-test` - Load and stress testing
- `/smoke-test` - Quick smoke tests
- `/production-readiness` - Pre-deployment validation
