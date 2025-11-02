---
name: smoke-test
category: testing
version: 1.0.0
---

# /smoke-test

Quick smoke tests to verify critical functionality and system health after deployment.

## Usage
```bash
/smoke-test [environment] [options]
```

## Parameters
- `environment` - Target environment: local|dev|staging|production (default: current)
- `--endpoints` - Specific endpoints to test (comma-separated, default: all critical)
- `--timeout` - Request timeout in seconds (default: 10)
- `--retry` - Retry failed requests (default: 3)
- `--parallel` - Run tests in parallel (default: true)
- `--report-format` - Output format: table|json|junit (default: table)
- `--fail-fast` - Stop on first failure (default: true)

## What It Does

**Quick Health Check (< 60 seconds)**:
1. 🏥 **Health Endpoints**: /health, /ready, /status
2. 🔐 **Authentication**: Login, token validation
3. 📊 **Critical APIs**: Core business endpoints
4. 🗄️ **Database**: Connection and basic query
5. 💾 **Cache**: Redis/Memcached connectivity
6. 🌐 **External Services**: Third-party API availability
7. 🔍 **DNS/Network**: Service discovery, load balancer
8. 📈 **Basic Performance**: Response time validation

**Test Coverage**:
- **Must-Work**: Critical paths that must always work
- **Shallow**: Surface-level validation (not deep E2E)
- **Fast**: Complete in under 60 seconds
- **Idempotent**: Safe to run in production
- **Non-Destructive**: Read-only or safe writes

## Examples

```bash
# Quick smoke test (current environment)
/smoke-test

# Test staging environment
/smoke-test staging

# Test specific endpoints
/smoke-test --endpoints "/api/health,/api/auth/login,/api/users"

# Production smoke test after deployment
/smoke-test production --fail-fast true

# Parallel execution with custom timeout
/smoke-test --parallel true --timeout 5

# Generate JUnit report for CI/CD
/smoke-test --report-format junit

# Local smoke test with retries disabled
/smoke-test local --retry 0
```

## Output

```
💨 Smoke Test Started

Environment: staging
Target: https://api.staging.example.com
Endpoints: 15 critical paths
Timeout: 10s per request
Parallel: Enabled
Fail Fast: Yes

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Health & Readiness Checks
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ✅ GET /health
     Status: 200 OK
     Response: { "status": "healthy", "uptime": 345678 }
     Duration: 45ms

  ✅ GET /ready
     Status: 200 OK
     Response: { "ready": true, "checks": { "db": "ok", "cache": "ok" } }
     Duration: 67ms

  ✅ GET /status
     Status: 200 OK
     Version: v1.3.0
     Environment: staging
     Duration: 34ms

  Result: 3/3 passed ✅
  Duration: 146ms

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Authentication & Authorization
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ✅ POST /api/auth/login
     Request: { "email": "smoke-test@example.com", "password": "***" }
     Status: 200 OK
     Response: { "token": "eyJ...", "expires": 3600 }
     Token Format: Valid JWT
     Duration: 234ms

  ✅ GET /api/auth/verify
     Headers: Authorization: Bearer eyJ...
     Status: 200 OK
     Response: { "valid": true, "user": { "id": "123", "email": "..." } }
     Duration: 89ms

  ✅ POST /api/auth/refresh
     Status: 200 OK
     New Token: eyJ... (different from original)
     Duration: 123ms

  Result: 3/3 passed ✅
  Duration: 446ms

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Critical API Endpoints
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ✅ GET /api/users
     Status: 200 OK
     Response: Array of 20 users
     Schema: Valid (all required fields present)
     Duration: 78ms

  ✅ GET /api/products
     Status: 200 OK
     Response: Array of 24 products
     Pagination: Working (page 1/10)
     Duration: 123ms

  ✅ POST /api/orders (test order)
     Request: { "product_id": "test-123", "quantity": 1, "test": true }
     Status: 201 Created
     Response: { "id": "ord-456", "status": "pending" }
     Test Flag: Honored (order marked for cleanup)
     Duration: 345ms

  ✅ GET /api/orders/ord-456
     Status: 200 OK
     Response: Order details match creation
     Duration: 67ms

  ✅ DELETE /api/orders/ord-456 (cleanup)
     Status: 204 No Content
     Test order cleaned up successfully
     Duration: 89ms

  Result: 5/5 passed ✅
  Duration: 702ms

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Database Connectivity
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ✅ PostgreSQL Connection
     Host: postgres.staging.svc.cluster.local:5432
     Database: myapp_staging
     Status: Connected
     Pool: 5/20 connections active
     Duration: 34ms

  ✅ Simple Query: SELECT 1
     Result: 1
     Duration: 12ms

  ✅ Table Access: SELECT COUNT(*) FROM users
     Result: 12,345 users
     Duration: 23ms

  Result: 3/3 passed ✅
  Duration: 69ms

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Cache Connectivity
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ✅ Redis Connection
     Host: redis.staging.svc.cluster.local:6379
     Status: Connected
     Role: master
     Replicas: 1
     Duration: 28ms

  ✅ SET/GET Test
     SET smoke-test-key: "test-value-1234"
     GET smoke-test-key: "test-value-1234" ✅
     DEL smoke-test-key: 1 (cleaned up)
     Duration: 45ms

  ✅ Cache Hit Test
     GET frequent-key: Hit (value exists)
     Cache working correctly
     Duration: 15ms

  Result: 3/3 passed ✅
  Duration: 88ms

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
External Service Connectivity
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ✅ Stripe API
     Endpoint: https://api.stripe.com/v1/charges
     Status: 200 OK
     API Version: 2023-10-16
     Duration: 234ms

  ✅ SendGrid API
     Endpoint: https://api.sendgrid.com/v3/mail/send
     Status: 202 Accepted (test mode, no email sent)
     Duration: 189ms

  ✅ AWS S3
     Bucket: myapp-staging-assets
     Test: HEAD bucket
     Status: 200 OK (bucket accessible)
     Duration: 123ms

  Result: 3/3 passed ✅
  Duration: 546ms

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DNS & Network
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ✅ DNS Resolution
     api.staging.example.com → 10.0.1.45
     Duration: 23ms

  ✅ Load Balancer
     Health check: 3/3 targets healthy
     Status: Operational
     Duration: 45ms

  ✅ SSL Certificate
     Issuer: Let's Encrypt
     Valid until: 2025-12-01
     Days remaining: 30
     Status: Valid ✅
     Duration: 67ms

  Result: 3/3 passed ✅
  Duration: 135ms

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Performance Baseline
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Response Times:
    /health: 45ms ✅ (SLA: <100ms)
    /api/auth/login: 234ms ✅ (SLA: <500ms)
    /api/users: 78ms ✅ (SLA: <200ms)
    /api/products: 123ms ✅ (SLA: <200ms)
    /api/orders (POST): 345ms ✅ (SLA: <1000ms)

  All endpoints within SLA ✅

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Smoke Test Summary
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Status: ✅ ALL TESTS PASSED
Environment: staging
Total Tests: 26
Passed: 26 (100%)
Failed: 0
Skipped: 0

Test Breakdown:
  ✅ Health Checks: 3/3
  ✅ Authentication: 3/3
  ✅ API Endpoints: 5/5
  ✅ Database: 3/3
  ✅ Cache: 3/3
  ✅ External Services: 3/3
  ✅ Network: 3/3
  ✅ Performance: 5/5 within SLA

Performance Summary:
  Total Duration: 2.1s
  Average Response Time: 112ms
  Slowest Endpoint: POST /api/orders (345ms)
  Fastest Endpoint: SELECT 1 (12ms)

System Health:
  ✅ All critical paths operational
  ✅ Database connections healthy
  ✅ Cache operational
  ✅ External services reachable
  ✅ Performance within SLA
  ✅ No errors or warnings

Report Generated:
  📊 reports/smoke-test-staging-2025-11-01.html
  📋 reports/smoke-test-staging-2025-11-01.junit.xml

✅ Environment ready for use!
```

## Chains With

```bash
# Deploy → smoke test → monitor
/k8s-deploy && /smoke-test && /monitoring-configure

# Smoke test → full test suite
/smoke-test && /integration-test && /e2e-test

# Quick validation after release
/github-release && /docker-deploy && /smoke-test production

# Pre-production smoke test
/smoke-test staging && /production-readiness
```

## See Also
- `/integration-test` - Integration testing
- `/e2e-test` - End-to-end testing
- `/production-readiness` - Pre-deployment validation
- `/k8s-deploy` - Kubernetes deployment
- `/monitoring-configure` - Setup monitoring
