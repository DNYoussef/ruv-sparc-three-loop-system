#!/bin/bash
#
# Performance Validation Script
# Validates application performance against production SLAs
#

set -e

TARGET_PATH="${1:-.}"
ENVIRONMENT="${2:-production}"
OUTPUT_DIR="performance-validation-$(date +%s)"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# SLA thresholds
RESPONSE_TIME_AVG_THRESHOLD=200     # milliseconds
RESPONSE_TIME_P95_THRESHOLD=500     # milliseconds
RESPONSE_TIME_P99_THRESHOLD=1000    # milliseconds
THROUGHPUT_MIN_THRESHOLD=100        # requests per second
ERROR_RATE_MAX_THRESHOLD=1          # percentage
CPU_MAX_THRESHOLD=70                # percentage
MEMORY_MAX_THRESHOLD=80             # percentage

echo "========================================"
echo "Performance Validation"
echo "Environment: $ENVIRONMENT"
echo "Target: $TARGET_PATH"
echo "========================================"
echo ""

# Initialize results
GATES_PASSED=0
GATES_TOTAL=5

# GATE 1: Response Time Benchmarks
echo "[1/5] Running response time benchmarks..."

# Check if server is running
if ! pgrep -f "node.*server" > /dev/null; then
    echo -e "${YELLOW}Warning: No node server detected. Start server first.${NC}"
    echo "Skipping performance tests..."
    exit 0
fi

# Get server port (assume 3000 or check package.json)
SERVER_PORT=3000

# Run basic load test with autocannon (if available)
if command -v autocannon &> /dev/null; then
    echo "Running load test with autocannon..."

    autocannon -c 10 -d 30 -p 1 \
        --json \
        http://localhost:$SERVER_PORT > "$OUTPUT_DIR/load-test.json" 2>&1 || true

    # Parse results
    if [ -f "$OUTPUT_DIR/load-test.json" ]; then
        AVG_LATENCY=$(cat "$OUTPUT_DIR/load-test.json" | jq -r '.latency.mean // 0')
        P95_LATENCY=$(cat "$OUTPUT_DIR/load-test.json" | jq -r '.latency.p95 // 0')
        P99_LATENCY=$(cat "$OUTPUT_DIR/load-test.json" | jq -r '.latency.p99 // 0')
        THROUGHPUT=$(cat "$OUTPUT_DIR/load-test.json" | jq -r '.requests.mean // 0')
        ERROR_RATE=$(cat "$OUTPUT_DIR/load-test.json" | jq -r '.errors // 0')

        echo "  Average latency: ${AVG_LATENCY}ms"
        echo "  P95 latency: ${P95_LATENCY}ms"
        echo "  P99 latency: ${P99_LATENCY}ms"
        echo "  Throughput: ${THROUGHPUT} req/s"
        echo "  Error rate: ${ERROR_RATE}%"

        # Validate against SLAs
        if (( $(echo "$AVG_LATENCY < $RESPONSE_TIME_AVG_THRESHOLD" | bc -l) )) && \
           (( $(echo "$P95_LATENCY < $RESPONSE_TIME_P95_THRESHOLD" | bc -l) )); then
            echo -e "${GREEN}✅ GATE 1: Response times within SLA${NC}"
            ((GATES_PASSED++))
        else
            echo -e "${RED}❌ GATE 1: Response times exceed SLA${NC}"
        fi
    fi
else
    echo -e "${YELLOW}autocannon not installed. Install with: npm install -g autocannon${NC}"
fi

# GATE 2: Bottleneck Detection
echo ""
echo "[2/5] Detecting performance bottlenecks..."

# Check for N+1 queries
echo "Scanning for N+1 query patterns..."
N_PLUS_ONE_COUNT=$(grep -r "\.map.*await\|\.forEach.*await" "$TARGET_PATH" \
    --include="*.js" --include="*.ts" \
    | grep -v "test" | grep -v "node_modules" | wc -l || echo "0")

if [ "$N_PLUS_ONE_COUNT" -gt 0 ]; then
    echo -e "${YELLOW}⚠️  Found $N_PLUS_ONE_COUNT potential N+1 query patterns${NC}"
fi

# Check for synchronous blocking operations
echo "Scanning for blocking operations..."
BLOCKING_OPS=$(grep -r "readFileSync\|execSync\|readdirSync" "$TARGET_PATH" \
    --include="*.js" --include="*.ts" \
    | grep -v "test" | grep -v "node_modules" | wc -l || echo "0")

if [ "$BLOCKING_OPS" -gt 0 ]; then
    echo -e "${YELLOW}⚠️  Found $BLOCKING_OPS synchronous blocking operations${NC}"
fi

# Check for missing database indexes
echo "Checking database query patterns..."
UNINDEXED_QUERIES=$(grep -r "WHERE.*=" "$TARGET_PATH" \
    --include="*.js" --include="*.ts" \
    | grep -v "test" | grep -v "node_modules" | wc -l || echo "0")

if [ "$UNINDEXED_QUERIES" -gt 5 ]; then
    echo -e "${YELLOW}⚠️  Found $UNINDEXED_QUERIES WHERE clauses - verify indexes${NC}"
fi

if [ "$N_PLUS_ONE_COUNT" -eq 0 ] && [ "$BLOCKING_OPS" -eq 0 ]; then
    echo -e "${GREEN}✅ GATE 2: No major bottlenecks detected${NC}"
    ((GATES_PASSED++))
else
    echo -e "${YELLOW}⚠️  GATE 2: Potential bottlenecks found${NC}"
fi

# GATE 3: Memory Usage Analysis
echo ""
echo "[3/5] Analyzing memory usage..."

# Check for memory leaks patterns
MEMORY_LEAK_PATTERNS=$(grep -r "setInterval\|setTimeout" "$TARGET_PATH" \
    --include="*.js" --include="*.ts" \
    | grep -v "clearInterval\|clearTimeout" \
    | grep -v "test" | grep -v "node_modules" | wc -l || echo "0")

if [ "$MEMORY_LEAK_PATTERNS" -gt 0 ]; then
    echo -e "${YELLOW}⚠️  Found $MEMORY_LEAK_PATTERNS potential memory leak patterns${NC}"
fi

# Check for large object allocations
LARGE_ARRAYS=$(grep -r "new Array\|\.fill(" "$TARGET_PATH" \
    --include="*.js" --include="*.ts" \
    | grep -v "test" | grep -v "node_modules" | wc -l || echo "0")

if [ "$LARGE_ARRAYS" -gt 0 ]; then
    echo "  Found $LARGE_ARRAYS large array allocations"
fi

# Check for streaming usage for large data
HAS_STREAMING=$(grep -r "stream\|pipe\|createReadStream" "$TARGET_PATH" \
    --include="*.js" --include="*.ts" \
    | grep -v "test" | grep -v "node_modules" | wc -l || echo "0")

echo "  Streaming patterns detected: $HAS_STREAMING"

if [ "$MEMORY_LEAK_PATTERNS" -eq 0 ]; then
    echo -e "${GREEN}✅ GATE 3: No memory leak patterns detected${NC}"
    ((GATES_PASSED++))
else
    echo -e "${YELLOW}⚠️  GATE 3: Potential memory issues found${NC}"
fi

# GATE 4: Caching Strategy
echo ""
echo "[4/5] Validating caching strategy..."

# Check for caching implementation
HAS_REDIS=$(grep -r "redis\|cache" "$TARGET_PATH" \
    --include="*.js" --include="*.ts" \
    | grep -v "test" | grep -v "node_modules" | wc -l || echo "0")

HAS_MEMOIZATION=$(grep -r "memoize\|useMemo\|useCallback" "$TARGET_PATH" \
    --include="*.js" --include="*.ts" \
    | grep -v "test" | grep -v "node_modules" | wc -l || echo "0")

HAS_HTTP_CACHE=$(grep -r "Cache-Control\|ETag\|max-age" "$TARGET_PATH" \
    --include="*.js" --include="*.ts" \
    | grep -v "test" | grep -v "node_modules" | wc -l || echo "0")

echo "  Redis/Cache: $HAS_REDIS occurrences"
echo "  Memoization: $HAS_MEMOIZATION occurrences"
echo "  HTTP caching: $HAS_HTTP_CACHE occurrences"

TOTAL_CACHING=$((HAS_REDIS + HAS_MEMOIZATION + HAS_HTTP_CACHE))

if [ "$TOTAL_CACHING" -gt 5 ]; then
    echo -e "${GREEN}✅ GATE 4: Caching strategy implemented${NC}"
    ((GATES_PASSED++))
else
    echo -e "${YELLOW}⚠️  GATE 4: Limited caching detected${NC}"
fi

# GATE 5: Database Query Optimization
echo ""
echo "[5/5] Checking database query optimization..."

# Check for SELECT * queries
SELECT_STAR=$(grep -r "SELECT \*" "$TARGET_PATH" \
    --include="*.js" --include="*.ts" \
    | grep -v "test" | grep -v "node_modules" | wc -l || echo "0")

# Check for missing pagination
MISSING_PAGINATION=$(grep -r "\.find(\|\.findAll(" "$TARGET_PATH" \
    --include="*.js" --include="*.ts" \
    | grep -v "limit\|take" \
    | grep -v "test" | grep -v "node_modules" | wc -l || echo "0")

# Check for eager loading
HAS_EAGER_LOADING=$(grep -r "include\|populate\|with" "$TARGET_PATH" \
    --include="*.js" --include="*.ts" \
    | grep -v "test" | grep -v "node_modules" | wc -l || echo "0")

# Check for connection pooling
HAS_POOLING=$(grep -r "pool\|maxConnections" "$TARGET_PATH" \
    --include="*.js" --include="*.ts" \
    | grep -v "test" | grep -v "node_modules" | wc -l || echo "0")

echo "  SELECT * queries: $SELECT_STAR"
echo "  Missing pagination: $MISSING_PAGINATION"
echo "  Eager loading: $HAS_EAGER_LOADING"
echo "  Connection pooling: $HAS_POOLING"

if [ "$SELECT_STAR" -lt 3 ] && [ "$MISSING_PAGINATION" -lt 3 ] && [ "$HAS_POOLING" -gt 0 ]; then
    echo -e "${GREEN}✅ GATE 5: Database queries optimized${NC}"
    ((GATES_PASSED++))
else
    echo -e "${YELLOW}⚠️  GATE 5: Database query optimization needed${NC}"
fi

# Generate performance report
echo ""
echo "========================================"
echo "Performance Validation Report"
echo "========================================"
echo ""
echo "Gates Passed: $GATES_PASSED/$GATES_TOTAL"
echo ""

cat > "$OUTPUT_DIR/performance-report.json" <<EOF
{
  "timestamp": "$(date -Iseconds)",
  "environment": "$ENVIRONMENT",
  "gates_passed": $GATES_PASSED,
  "gates_total": $GATES_TOTAL,
  "metrics": {
    "avg_latency_ms": ${AVG_LATENCY:-0},
    "p95_latency_ms": ${P95_LATENCY:-0},
    "p99_latency_ms": ${P99_LATENCY:-0},
    "throughput_rps": ${THROUGHPUT:-0},
    "error_rate_pct": ${ERROR_RATE:-0}
  },
  "bottlenecks": {
    "n_plus_one_queries": $N_PLUS_ONE_COUNT,
    "blocking_operations": $BLOCKING_OPS,
    "unindexed_queries": $UNINDEXED_QUERIES
  },
  "memory": {
    "leak_patterns": $MEMORY_LEAK_PATTERNS,
    "large_allocations": $LARGE_ARRAYS,
    "streaming_usage": $HAS_STREAMING
  },
  "caching": {
    "redis_cache": $HAS_REDIS,
    "memoization": $HAS_MEMOIZATION,
    "http_cache": $HAS_HTTP_CACHE
  },
  "database": {
    "select_star_queries": $SELECT_STAR,
    "missing_pagination": $MISSING_PAGINATION,
    "eager_loading": $HAS_EAGER_LOADING,
    "connection_pooling": $HAS_POOLING
  },
  "passed": $([ "$GATES_PASSED" -ge 4 ] && echo "true" || echo "false")
}
EOF

echo "Report saved to: $OUTPUT_DIR/performance-report.json"
echo ""

# Recommendations
echo "Recommendations:"
echo ""

if [ "$N_PLUS_ONE_COUNT" -gt 0 ]; then
    echo "- Fix N+1 queries with eager loading or batch queries"
fi

if [ "$BLOCKING_OPS" -gt 0 ]; then
    echo "- Replace synchronous operations with async alternatives"
fi

if [ "$MEMORY_LEAK_PATTERNS" -gt 0 ]; then
    echo "- Clear intervals/timeouts on cleanup"
fi

if [ "$TOTAL_CACHING" -lt 5 ]; then
    echo "- Implement caching strategy (Redis, HTTP cache headers)"
fi

if [ "$SELECT_STAR" -gt 2 ]; then
    echo "- Avoid SELECT * queries, specify only needed columns"
fi

if [ "$MISSING_PAGINATION" -gt 2 ]; then
    echo "- Add pagination to list queries (limit/offset)"
fi

if [ "$HAS_POOLING" -eq 0 ]; then
    echo "- Configure database connection pooling"
fi

echo ""
echo "========================================"

# Exit code based on gates passed
if [ "$GATES_PASSED" -ge 4 ]; then
    echo -e "${GREEN}✅ PERFORMANCE VALIDATION PASSED${NC}"
    echo "========================================"
    exit 0
else
    echo -e "${RED}❌ PERFORMANCE VALIDATION FAILED${NC}"
    echo "========================================"
    exit 1
fi
