# Production Pre-Launch Checklist Example

This comprehensive example demonstrates a complete production readiness validation workflow for a Node.js/Express API before launching to production.

## Project: E-Commerce API v2.0

**Environment**: Production
**Target Launch Date**: 2025-12-01
**Deployment Strategy**: Blue-Green
**Traffic**: 10,000 req/min expected

---

## Phase 1: Automated Quality Gates (60 minutes)

### Gate 1: Test Suite Validation

```bash
# Run complete test suite with coverage
npm run test:all -- --coverage --verbose

# Expected output:
# ✅ Unit tests: 245/245 passed
# ✅ Integration tests: 89/89 passed
# ✅ E2E tests: 34/34 passed
# ✅ Coverage: 87.3% (threshold: 80%)
```

**Validation Script**:
```bash
#!/bin/bash
# Automated test validation

COVERAGE_THRESHOLD=80
TEST_RESULTS=$(npm test -- --coverage --json)

# Parse results
TOTAL_TESTS=$(echo "$TEST_RESULTS" | jq '.numTotalTests')
PASSED_TESTS=$(echo "$TEST_RESULTS" | jq '.numPassedTests')
COVERAGE=$(echo "$TEST_RESULTS" | jq '.coverageMap.total.lines.pct')

if [ "$PASSED_TESTS" -eq "$TOTAL_TESTS" ] && \
   [ "$(echo "$COVERAGE >= $COVERAGE_THRESHOLD" | bc)" -eq 1 ]; then
    echo "✅ GATE 1 PASSED: All tests passing with $COVERAGE% coverage"
    exit 0
else
    echo "❌ GATE 1 FAILED: Tests: $PASSED_TESTS/$TOTAL_TESTS, Coverage: $COVERAGE%"
    exit 1
fi
```

**Results**:
- ✅ All 368 tests passing
- ✅ Coverage: 87.3% (lines), 84.2% (branches), 91.1% (functions)
- ✅ No skipped or disabled tests
- ✅ Average test duration: 2.3s (threshold: 5s)

---

### Gate 2: Code Quality Audit

```bash
# Run ESLint with zero-tolerance
npx eslint . --max-warnings 0 --format json > code-quality-report.json

# Run TypeScript compiler check
npx tsc --noEmit

# Check cyclomatic complexity
npx complexity-report --format json src/ > complexity-report.json
```

**Quality Metrics**:
```json
{
  "quality_score": 92,
  "linting": {
    "errors": 0,
    "warnings": 0,
    "files_checked": 127
  },
  "typescript": {
    "errors": 0,
    "warnings": 3,
    "strict_mode": true
  },
  "complexity": {
    "max_cyclomatic": 8,
    "max_cognitive": 12,
    "avg_function_length": 23,
    "max_file_length": 387
  }
}
```

**Results**:
- ✅ Quality Score: 92/100 (threshold: 85)
- ✅ Zero linting errors
- ✅ TypeScript strict mode enabled
- ✅ All functions below complexity threshold (10)
- ⚠️ 3 files need refactoring (390-420 lines, threshold: 500)

---

### Gate 3: Security Deep-Dive

```bash
# Run comprehensive security audit
node resources/security-audit.js . --deep > security-report.json

# Additional security scans
npm audit --json
npx snyk test --json > snyk-report.json
```

**Security Scan Results**:
```json
{
  "timestamp": "2025-11-02T10:30:00Z",
  "summary": {
    "critical": 0,
    "high": 0,
    "medium": 2,
    "low": 5,
    "total": 7
  },
  "findings": {
    "critical": [],
    "high": [],
    "medium": [
      {
        "type": "Missing CSRF Protection",
        "recommendation": "Implement CSRF tokens for state-changing operations",
        "files": ["src/routes/orders.js", "src/routes/users.js"]
      },
      {
        "type": "Missing Rate Limiting",
        "recommendation": "Add rate limiting to authentication endpoints",
        "affected_endpoints": ["/api/login", "/api/register"]
      }
    ]
  },
  "dependencies": {
    "total": 156,
    "outdated": 8,
    "vulnerable": 0
  },
  "passed": true
}
```

**Security Checklist**:
- ✅ Zero critical/high vulnerabilities
- ✅ No hardcoded secrets detected
- ✅ All security headers configured (helmet.js)
- ✅ JWT tokens with 256-bit secrets
- ✅ Bcrypt password hashing (12 rounds)
- ✅ SQL injection protection (parameterized queries)
- ✅ XSS protection (input sanitization)
- ⚠️ CSRF protection needed for 2 endpoints
- ⚠️ Rate limiting recommended for auth endpoints

**Action Items**:
1. Implement CSRF tokens (csurf middleware) - ETA: 2 hours
2. Add rate limiting (express-rate-limit) - ETA: 1 hour
3. Update 8 outdated dependencies - ETA: 30 minutes

---

### Gate 4: Performance Benchmarking

```bash
# Run performance validation
bash resources/performance-validator.sh . production

# Load testing with autocannon
autocannon -c 100 -d 60 -p 10 \
  --json http://localhost:3000/api/products > load-test-results.json
```

**Performance Metrics**:
```json
{
  "response_times": {
    "avg_ms": 142,
    "p50_ms": 118,
    "p95_ms": 387,
    "p99_ms": 824,
    "max_ms": 1243
  },
  "throughput": {
    "requests_per_second": 287,
    "total_requests": 17220,
    "total_duration_seconds": 60
  },
  "resource_usage": {
    "cpu_percent": 52,
    "memory_mb": 387,
    "heap_used_mb": 224
  },
  "bottlenecks": {
    "n_plus_one_queries": 0,
    "blocking_operations": 0,
    "missing_indexes": 1
  },
  "sla_compliance": {
    "avg_response_time": "PASS",
    "p95_response_time": "PASS",
    "throughput": "PASS",
    "error_rate": "PASS"
  }
}
```

**SLA Validation**:
- ✅ Average Response Time: 142ms (SLA: <200ms)
- ✅ P95 Response Time: 387ms (SLA: <500ms)
- ⚠️ P99 Response Time: 824ms (SLA: <1000ms, close to threshold)
- ✅ Throughput: 287 req/s (SLA: >100 req/s)
- ✅ Error Rate: 0.02% (SLA: <1%)
- ✅ CPU Usage: 52% (SLA: <70%)
- ✅ Memory Usage: 387MB/1GB (SLA: <80%)

**Optimization Opportunities**:
1. Add database index on `products.category_id` (found 1 missing index)
2. Implement Redis caching for product catalog (hot data)
3. Enable gzip compression (reduce payload size by 60%)
4. Optimize P99 response time with connection pooling tuning

---

### Gate 5: Documentation Completeness

```bash
# Validate required documentation
./scripts/check-documentation.sh
```

**Documentation Audit**:
- ✅ README.md (1,247 lines, includes all sections)
- ✅ docs/API.md (892 lines, OpenAPI 3.0 spec)
- ✅ docs/deployment.md (567 lines, step-by-step guide)
- ✅ docs/rollback.md (234 lines, <5 min rollback SLA)
- ✅ docs/monitoring.md (412 lines, Datadog setup)
- ✅ docs/troubleshooting.md (678 lines, common issues)
- ✅ .env.example (47 variables documented)
- ✅ CHANGELOG.md (full version history)
- ✅ CONTRIBUTING.md (developer guidelines)

**API Documentation Coverage**:
- Total Endpoints: 47
- Documented: 47 (100%)
- With Examples: 47 (100%)
- With Auth Details: 34 (100% of protected endpoints)

---

### Gate 6: Infrastructure Readiness

```bash
# Run deployment verification
python3 resources/deployment-verifier.py . production
```

**Infrastructure Checklist**:
```yaml
Environment Configuration:
  - ✅ NODE_ENV=production configured
  - ✅ All 47 environment variables set
  - ✅ Secrets in AWS Secrets Manager
  - ✅ Environment-specific config validated

Database:
  - ✅ PostgreSQL 14.5 (production cluster)
  - ✅ Connection pooling (max: 20, min: 5)
  - ✅ Read replicas configured (2x)
  - ✅ Automated backups (daily, 30-day retention)
  - ✅ Database migrations tested (v1.8.2 → v2.0.0)

Caching:
  - ✅ Redis 7.0 cluster (3 nodes)
  - ✅ Cache hit ratio: 87% (target: >80%)
  - ✅ TTL configured for all keys

Load Balancer:
  - ✅ Application Load Balancer (AWS)
  - ✅ Health checks configured (/health every 30s)
  - ✅ SSL/TLS certificates valid (expires: 2026-03-15)
  - ✅ HTTP/2 enabled

Monitoring:
  - ✅ Datadog APM configured
  - ✅ Custom metrics collection (38 metrics)
  - ✅ Error tracking (Sentry)
  - ✅ Log aggregation (CloudWatch Logs)
  - ✅ Alerting rules (12 critical, 23 warning)

Scaling:
  - ✅ Auto-scaling group (min: 3, max: 12)
  - ✅ Scale-up trigger: CPU >70% for 5 min
  - ✅ Scale-down trigger: CPU <30% for 15 min
  - ✅ Container health checks configured
```

---

## Phase 2: Manual Quality Validation (120 minutes)

### Smoke Testing

```bash
# Run smoke test suite in staging
npm run test:smoke -- --env=staging

# Critical user flows:
# 1. User registration → Email verification → Login
# 2. Browse products → Add to cart → Checkout → Payment
# 3. Order tracking → Order history
# 4. Password reset flow
```

**Smoke Test Results**:
- ✅ User registration: 100% success (50 iterations)
- ✅ Product browsing: <150ms avg response time
- ✅ Cart operations: 100% success
- ✅ Checkout flow: 98% success (1 timeout, retried successfully)
- ✅ Payment processing: 100% success (Stripe test mode)
- ✅ Order tracking: 100% success
- ✅ Password reset: 100% success (email delivery confirmed)

---

### Load Testing (Staging Environment)

```bash
# Gradual ramp-up load test
k6 run --vus 10 --duration 5m load-test.js   # Warm-up
k6 run --vus 50 --duration 10m load-test.js  # Normal load
k6 run --vus 200 --duration 5m load-test.js  # Peak load
k6 run --vus 500 --duration 2m load-test.js  # Stress test
```

**Load Test Results**:
| VUs | RPS | Avg Latency | P95 Latency | Error Rate |
|-----|-----|-------------|-------------|------------|
| 10  | 287 | 142ms       | 387ms       | 0%         |
| 50  | 1,234 | 198ms     | 512ms       | 0.01%      |
| 200 | 4,567 | 342ms     | 876ms       | 0.04%      |
| 500 | 8,932 | 892ms     | 2,341ms     | 2.3%       |

**Observations**:
- ✅ Handles expected load (10,000 req/min = 167 req/s) with ease
- ✅ Graceful degradation under stress
- ⚠️ Error rate spikes at 500 concurrent users (2.3%)
- ✅ Auto-scaling triggered correctly at 200 VUs (3 → 6 instances)

---

### Security Penetration Testing

```bash
# OWASP ZAP automated scan
docker run -t owasp/zap2docker-stable zap-baseline.py \
  -t https://staging-api.example.com \
  -r security-scan-report.html
```

**Penetration Test Results**:
- ✅ No critical vulnerabilities
- ✅ No high-risk vulnerabilities
- ⚠️ 2 medium-risk findings (see action items)
- ℹ️ 5 low-risk/informational findings

**Action Items**:
1. Add `X-Content-Type-Options: nosniff` header (missing on 3 endpoints)
2. Implement stricter CORS policy (currently allows all origins in staging)

---

## Phase 3: Production Deployment (30 minutes)

### Pre-Deployment Checklist

```bash
# Final pre-deployment validation
./scripts/pre-deployment-check.sh production
```

**Pre-Deployment Sign-offs**:
- ✅ Development Lead: Code review approved (@tech-lead)
- ✅ QA Lead: All tests passing (@qa-manager)
- ✅ Security Team: Security audit approved (@security-team)
- ✅ DevOps: Infrastructure ready (@devops-lead)
- ✅ Product Owner: Features approved (@product-owner)
- ✅ On-call Engineer: Ready for deployment (@oncall-engineer)

---

### Deployment Execution

```bash
# Blue-Green Deployment
# Step 1: Deploy to green environment
terraform apply -var="environment=green"

# Step 2: Run health checks
./scripts/health-check.sh green

# Step 3: Smoke tests on green
npm run test:smoke -- --env=green

# Step 4: Switch traffic (10% canary)
./scripts/switch-traffic.sh --canary 10 --target green

# Step 5: Monitor for 15 minutes
./scripts/monitor-deployment.sh --duration 15m

# Step 6: Full traffic switch
./scripts/switch-traffic.sh --full --target green
```

**Deployment Timeline**:
- 09:00 - Deploy to green environment (8 minutes)
- 09:08 - Health checks pass (2 minutes)
- 09:10 - Smoke tests on green (5 minutes)
- 09:15 - 10% canary release (monitoring started)
- 09:30 - Metrics stable, 50% traffic to green
- 09:45 - Metrics stable, 100% traffic to green
- 09:50 - Blue environment kept for 24h (rollback safety)
- 10:00 - **DEPLOYMENT COMPLETE**

---

## Phase 4: Post-Deployment Validation (60 minutes)

### Health Check Validation

```bash
# Validate all health endpoints
curl -f https://api.example.com/health
curl -f https://api.example.com/health/database
curl -f https://api.example.com/health/redis
curl -f https://api.example.com/health/external-services
```

**Health Status**:
- ✅ API: Healthy (200 OK)
- ✅ Database: Healthy (latency: 3ms)
- ✅ Redis: Healthy (hit rate: 89%)
- ✅ External Services: All reachable

---

### Metrics Monitoring (First Hour)

**Key Metrics**:
```
10:00 - Request Rate: 8,234 req/min (expected: 10,000 req/min)
10:15 - Request Rate: 11,567 req/min (traffic ramping up)
10:30 - Request Rate: 14,892 req/min (above expected, holding steady)
10:45 - Request Rate: 13,234 req/min (normalized)

Error Rate: 0.01% (SLA: <1%) ✅
Avg Response Time: 156ms (SLA: <200ms) ✅
P95 Response Time: 412ms (SLA: <500ms) ✅
P99 Response Time: 789ms (SLA: <1000ms) ✅

CPU Usage: 58% (6 instances) ✅
Memory Usage: 64% ✅
Database Connections: 47/100 ✅
```

---

### Business Metrics Validation

**First Hour KPIs**:
- ✅ Total Orders: 1,247 (matching historical average)
- ✅ Conversion Rate: 3.2% (baseline: 3.1%)
- ✅ Cart Abandonment: 68% (baseline: 69%)
- ✅ Average Order Value: $87.34 (baseline: $86.12)
- ✅ Payment Success Rate: 99.2% (baseline: 99.1%)

---

## Rollback Plan (If Needed)

**Rollback Triggers**:
1. Error rate > 5% for 5 consecutive minutes
2. P95 response time > 2x SLA for 10 minutes
3. Payment processing failures > 1%
4. Database connection errors > 5%
5. Manual decision by on-call engineer

**Rollback Procedure** (< 5 minutes):
```bash
# Immediate traffic switch back to blue
./scripts/switch-traffic.sh --full --target blue --emergency

# Verify blue environment health
./scripts/health-check.sh blue

# Monitor recovery
./scripts/monitor-deployment.sh --duration 30m

# Investigate root cause
./scripts/collect-logs.sh --since deployment --severity error
```

---

## Final Checklist Summary

**Quality Gates**: 6/6 Passed ✅
**Manual Validation**: Complete ✅
**Deployment**: Successful ✅
**Post-Deployment**: All metrics nominal ✅

**Next Steps**:
1. Monitor for 24 hours before decommissioning blue environment
2. Address medium-priority security findings within 7 days
3. Implement performance optimizations identified during testing
4. Update runbooks with lessons learned

---

**Deployment Status**: ✅ **PRODUCTION LAUNCH SUCCESSFUL**

**Deployed Version**: v2.0.0
**Deployment Time**: 2025-11-02 10:00 UTC
**Total Downtime**: 0 seconds (blue-green deployment)
**Rollback Available**: Yes (blue environment active for 24h)
