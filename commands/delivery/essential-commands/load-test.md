---
name: load-test
category: testing
version: 1.0.0
---

# /load-test

Load and stress testing to validate system performance under expected and peak traffic.

## Usage
```bash
/load-test [scenario] [options]
```

## Parameters
- `scenario` - Test scenario: baseline|peak|stress|spike|soak (default: baseline)
- `--users` - Concurrent users (default: 100)
- `--duration` - Test duration in minutes (default: 5)
- `--ramp-up` - Ramp-up time in seconds (default: 60)
- `--target-rps` - Target requests per second (optional)
- `--endpoints` - Endpoints to test (comma-separated, default: all critical)
- `--report-format` - Output format: html|json|csv (default: html)
- `--thresholds` - Performance thresholds (default: 95th percentile <1s)

## What It Does

**Comprehensive Load Testing**:
1. 📊 **Baseline**: Normal traffic patterns
2. 🚀 **Peak**: Expected peak load (2-3x baseline)
3. ⚡ **Stress**: Beyond capacity to find breaking point
4. 📈 **Spike**: Sudden traffic surge
5. ⏱️ **Soak**: Extended duration (endurance testing)
6. 🎯 **Scenario-Based**: Real user behavior simulation
7. 📉 **Bottleneck Detection**: Identify performance issues
8. 📊 **Real-Time Monitoring**: Live metrics dashboard

**Metrics Collected**:
- **Response Time**: Min, max, avg, p50, p95, p99
- **Throughput**: Requests per second (RPS)
- **Error Rate**: Percentage of failed requests
- **Resource Usage**: CPU, memory, network, disk
- **Database**: Query time, connection pool, locks
- **Cache**: Hit ratio, eviction rate
- **Network**: Latency, bandwidth utilization

## Examples

```bash
# Baseline load test (100 users)
/load-test baseline

# Peak load test (1000 users)
/load-test peak --users 1000 --duration 10

# Stress test to find breaking point
/load-test stress --users 5000 --ramp-up 300

# Spike test (sudden 10x traffic)
/load-test spike --users 10000 --ramp-up 10

# Soak test (24-hour endurance)
/load-test soak --users 500 --duration 1440

# Custom load test with specific RPS
/load-test --target-rps 5000 --duration 15

# Test specific endpoints
/load-test --endpoints "/api/users,/api/orders,/api/products"
```

## Output

```
⚡ Load Test Started

Scenario: peak
Users: 1,000 concurrent
Duration: 10 minutes
Ramp-up: 60 seconds
Target RPS: Auto (best effort)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Test Configuration
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Target: https://api.staging.example.com
  Virtual Users: 1,000
  Test Duration: 10m 0s
  Ramp-up: 60s (16.67 users/sec)

  Test Endpoints:
    GET /api/users (40% of traffic)
    POST /api/orders (30% of traffic)
    GET /api/products (20% of traffic)
    PUT /api/profile (10% of traffic)

  Performance Thresholds:
    p95 response time: <1,000ms
    p99 response time: <2,000ms
    Error rate: <1%
    RPS: >4,000

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Ramp-Up Phase (60 seconds)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  [00:10] Users: 167 | RPS: 687 | Avg: 89ms | Errors: 0%
  [00:20] Users: 333 | RPS: 1,245 | Avg: 92ms | Errors: 0%
  [00:30] Users: 500 | RPS: 1,987 | Avg: 98ms | Errors: 0.1%
  [00:40] Users: 667 | RPS: 2,634 | Avg: 125ms | Errors: 0.2%
  [00:50] Users: 833 | RPS: 3,298 | Avg: 156ms | Errors: 0.3%
  [00:60] Users: 1,000 | RPS: 3,945 | Avg: 187ms | Errors: 0.5%

  ✅ Ramp-up complete

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Sustained Load Phase (9 minutes)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  [02:00] RPS: 4,234 | p95: 234ms | p99: 456ms | Errors: 0.4% ✅
  [03:00] RPS: 4,187 | p95: 245ms | p99: 478ms | Errors: 0.6% ✅
  [04:00] RPS: 4,298 | p95: 267ms | p99: 512ms | Errors: 0.8% ✅
  [05:00] RPS: 4,156 | p95: 289ms | p99: 534ms | Errors: 1.2% ⚠️
  [06:00] RPS: 4,089 | p95: 312ms | p99: 589ms | Errors: 1.5% ⚠️
  [07:00] RPS: 3,967 | p95: 345ms | p99: 623ms | Errors: 2.1% ❌
  [08:00] RPS: 3,845 | p95: 398ms | p99: 712ms | Errors: 3.4% ❌
  [09:00] RPS: 3,712 | p95: 456ms | p99: 834ms | Errors: 4.8% ❌
  [10:00] RPS: 3,589 | p95: 523ms | p99: 967ms | Errors: 6.2% ❌

  ⚠️  Performance degradation detected after 5 minutes
  ❌ Error rate threshold exceeded at 7 minutes

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Response Time Distribution
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  GET /api/users:
    Min: 23ms
    Avg: 187ms
    p50: 156ms
    p90: 289ms
    p95: 345ms ✅ (threshold: 1,000ms)
    p99: 623ms ✅ (threshold: 2,000ms)
    Max: 3,456ms

  POST /api/orders:
    Min: 45ms
    Avg: 234ms
    p50: 198ms
    p90: 378ms
    p95: 489ms ✅
    p99: 756ms ✅
    Max: 4,123ms

  GET /api/products:
    Min: 34ms
    Avg: 145ms
    p50: 123ms
    p90: 234ms
    p95: 298ms ✅
    p99: 456ms ✅
    Max: 2,345ms

  PUT /api/profile:
    Min: 56ms
    Avg: 298ms
    p50: 245ms
    p90: 456ms
    p95: 567ms ✅
    p99: 834ms ✅
    Max: 5,678ms

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Throughput Analysis
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Total Requests: 2,456,789
  Successful: 2,332,145 (94.9%)
  Failed: 124,644 (5.1%)

  Requests per Second:
    Min: 3,589 RPS
    Avg: 4,094 RPS ✅ (threshold: 4,000)
    Max: 4,298 RPS
    Target: 4,000 RPS

  Error Breakdown:
    500 Internal Server Error: 78,234 (62.8%)
    503 Service Unavailable: 34,567 (27.7%)
    504 Gateway Timeout: 9,843 (7.9%)
    429 Too Many Requests: 2,000 (1.6%)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Resource Utilization
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  API Servers (3 instances):
    CPU: 78% avg (peak: 94%)
    Memory: 82% avg (peak: 91%)
    Network: 450 Mbps avg (peak: 680 Mbps)

  Database (PostgreSQL):
    CPU: 89% avg (peak: 98%) ⚠️
    Memory: 76% avg (peak: 84%)
    Connections: 95/100 (95%) ⚠️
    Active queries: 87 avg (peak: 112)
    Slow queries: 234 (>1s)

  Cache (Redis):
    CPU: 45% avg (peak: 67%)
    Memory: 3.2 GB / 8 GB (40%)
    Hit ratio: 76.3% (target: >80%) ⚠️
    Evictions: 12,345

  Load Balancer:
    Throughput: 4,094 req/s avg
    Healthy targets: 3/3
    Unhealthy: 0
    Connection errors: 234

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Bottleneck Analysis
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  🔴 Critical Bottlenecks:
    1. Database Connection Pool Saturation
       - 95/100 connections used (95%)
       - 234 connection wait timeouts
       - Recommend: Increase pool size to 150

    2. High Database CPU Usage
       - 89% average, 98% peak
       - 234 slow queries (>1s)
       - Recommend: Add indexes, optimize queries

    3. Increasing Error Rate Over Time
       - 0.4% → 6.2% over 10 minutes
       - Indicates memory leak or resource exhaustion
       - Recommend: Investigate API server memory growth

  ⚠️  Warning Issues:
    4. Low Cache Hit Ratio (76.3%)
       - Target: >80%
       - Recommend: Increase cache TTL, warm cache

    5. API Server Memory Growth
       - Linear memory increase detected
       - Potential memory leak in session management
       - Recommend: Profile memory usage

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Load Test Summary
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Status: ⚠️  PASSED WITH WARNINGS
Test Duration: 10m 0s
Total Requests: 2,456,789
Success Rate: 94.9%
Error Rate: 5.1% ❌ (threshold: <1%)

Performance Metrics:
  Average RPS: 4,094 ✅ (target: 4,000)
  p95 response time: 345ms ✅ (threshold: <1,000ms)
  p99 response time: 623ms ✅ (threshold: <2,000ms)

Threshold Results:
  ✅ p95 response time: PASS
  ✅ p99 response time: PASS
  ❌ Error rate: FAIL (5.1% > 1%)
  ✅ RPS: PASS (4,094 > 4,000)

System Capacity:
  Current: 1,000 concurrent users @ 4,094 RPS
  Recommended: 800 concurrent users @ 3,200 RPS (with <1% errors)
  Breaking point: ~1,200 users (database saturation)

Reports Generated:
  📊 HTML: reports/load-test-peak-2025-11-01.html
  📈 Grafana: http://grafana.example.com/d/load-test
  📉 CSV: reports/load-test-metrics.csv

Recommendations:
  🔴 CRITICAL:
    1. Increase database connection pool (100 → 150)
    2. Optimize slow database queries (234 queries >1s)
    3. Investigate API server memory leak

  ⚠️  HIGH PRIORITY:
    4. Scale database CPU (consider read replicas)
    5. Improve cache hit ratio (76% → 85%+)
    6. Add database query caching

  ✅ MEDIUM PRIORITY:
    7. Add horizontal pod autoscaling (HPA)
    8. Implement circuit breakers for external services
    9. Add rate limiting per user

⚠️  RECOMMENDATION:
System can handle peak load but with elevated error rates.
Address database bottlenecks before production deployment.
```

## Chains With

```bash
# Load test → analyze → optimize
/load-test peak && /bottleneck-detect && /performance-report

# Pre-deployment load test
/load-test baseline && /production-readiness

# Full performance suite
/load-test stress && /e2e-test && /regression-test

# Continuous load testing
/load-test soak --duration 1440 && /monitoring-configure
```

## See Also
- `/bottleneck-detect` - Bottleneck detection
- `/performance-report` - Performance analysis
- `/integration-test` - Integration testing
- `/monitoring-configure` - Setup monitoring
- `/production-readiness` - Pre-deployment validation
