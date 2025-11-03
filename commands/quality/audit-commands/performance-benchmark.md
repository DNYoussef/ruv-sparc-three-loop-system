---
name: performance-benchmark
description: Comprehensive performance benchmarking for APIs, frontends, and databases
category: audit-commands
version: 2.0.0
requires_mcp: false
---

# ⚡ Performance Benchmark - Comprehensive Performance Analysis

**Command**: Performance Benchmarking | **Category**: Quality Audits

Complete performance analysis covering API endpoints, frontend metrics, database queries, and load testing.

## Benchmark Execution

```bash
#!/bin/bash
# Complete Performance Benchmark

echo "⚡ PERFORMANCE BENCHMARK"
echo "======================="

# Phase 1: API Benchmarking
npx autocannon http://localhost:3000/api/users \
  -c 100 -d 30 -p 10 \
  --json > benchmarks/api-users.json

# Phase 2: Frontend Performance (Lighthouse)
npx lighthouse http://localhost:3000 \
  --output json \
  --output-path benchmarks/lighthouse.json \
  --chrome-flags="--headless"

# Phase 3: Database Query Performance
cat > benchmarks/db-queries.sql <<'EOF'
EXPLAIN ANALYZE
SELECT * FROM users
WHERE status = 'active'
ORDER BY created_at DESC
LIMIT 20;
EOF

psql $DATABASE_URL -f benchmarks/db-queries.sql > benchmarks/query-plan.txt

# Phase 4: Load Testing (k6)
k6 run tests/load-test.js --out json=benchmarks/k6-results.json

# Phase 5: Generate Report
cat > benchmarks/report.md <<EOF
# Performance Benchmark Report
Generated: $(date)

## API Performance
- Throughput: $(jq '.requests.average' benchmarks/api-users.json) req/s
- P50: $(jq '.latency.p50' benchmarks/api-users.json)ms
- P95: $(jq '.latency.p95' benchmarks/api-users.json)ms
- P99: $(jq '.latency.p99' benchmarks/api-users.json)ms

## Frontend Performance
- LCP: $(jq '.audits["largest-contentful-paint"].numericValue' benchmarks/lighthouse.json)ms
- FID: $(jq '.audits["max-potential-fid"].numericValue' benchmarks/lighthouse.json)ms
- CLS: $(jq '.audits["cumulative-layout-shift"].numericValue' benchmarks/lighthouse.json)
- Performance Score: $(jq '.categories.performance.score * 100' benchmarks/lighthouse.json)

## Thresholds
- API P95 < 200ms: $([ $(jq '.latency.p95' benchmarks/api-users.json) -lt 200 ] && echo "✅ PASS" || echo "❌ FAIL")
- LCP < 2500ms: $([ $(jq '.audits["largest-contentful-paint"].numericValue' benchmarks/lighthouse.json | cut -d. -f1) -lt 2500 ] && echo "✅ PASS" || echo "❌ FAIL")
EOF

echo "✅ Benchmark complete: benchmarks/report.md"
```

**Agent Status**: Production-Ready
**Version**: 2.0.0
