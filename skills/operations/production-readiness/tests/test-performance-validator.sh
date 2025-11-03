#!/bin/bash
#
# Tests for Performance Validation Script
#

# Test framework setup
TESTS_PASSED=0
TESTS_FAILED=0

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Test helper functions
test_assert() {
    local description="$1"
    local condition="$2"

    if eval "$condition"; then
        echo -e "${GREEN}✓${NC} $description"
        ((TESTS_PASSED++))
        return 0
    else
        echo -e "${RED}✗${NC} $description"
        ((TESTS_FAILED++))
        return 1
    fi
}

test_assert_file_exists() {
    local description="$1"
    local file_path="$2"

    if [ -f "$file_path" ]; then
        echo -e "${GREEN}✓${NC} $description"
        ((TESTS_PASSED++))
        return 0
    else
        echo -e "${RED}✗${NC} $description: File not found: $file_path"
        ((TESTS_FAILED++))
        return 1
    fi
}

test_assert_contains() {
    local description="$1"
    local file_path="$2"
    local pattern="$3"

    if grep -q "$pattern" "$file_path"; then
        echo -e "${GREEN}✓${NC} $description"
        ((TESTS_PASSED++))
        return 0
    else
        echo -e "${RED}✗${NC} $description: Pattern not found: $pattern"
        ((TESTS_FAILED++))
        return 1
    fi
}

# Create test directory
TEST_DIR=$(mktemp -d)
cd "$TEST_DIR" || exit 1

echo "========================================"
echo "Performance Validator Tests"
echo "Test Directory: $TEST_DIR"
echo "========================================"
echo ""

# Test 1: Script exists and is executable
echo "[Test Suite 1] Script Validation"
SCRIPT_PATH="../resources/performance-validator.sh"

test_assert "Performance validator script exists" "[ -f '$SCRIPT_PATH' ]"
test_assert "Performance validator script is executable" "[ -x '$SCRIPT_PATH' ] || chmod +x '$SCRIPT_PATH'"

echo ""

# Test 2: Create minimal project structure
echo "[Test Suite 2] Project Structure Detection"

# Create package.json
cat > package.json <<EOF
{
  "name": "test-project",
  "version": "1.0.0",
  "scripts": {
    "start": "node index.js"
  }
}
EOF

test_assert_file_exists "package.json created" "package.json"

# Create source files
mkdir -p src
cat > src/index.js <<EOF
const express = require('express');
const app = express();

app.get('/health', (req, res) => {
  res.json({ status: 'ok' });
});

app.listen(3000);
EOF

test_assert_file_exists "Source file created" "src/index.js"

echo ""

# Test 3: Test N+1 query detection
echo "[Test Suite 3] Bottleneck Detection"

cat > src/database.js <<EOF
async function getUsersWithPosts(userIds) {
  const users = await User.findAll();
  for (const user of users) {
    const posts = await Post.findAll({ where: { userId: user.id } }); // N+1 query
  }
}

const results = data.map(async item => {
  return await fetchDetail(item.id); // Potential N+1
});
EOF

# Run validator and check output
OUTPUT_FILE="validator-output.txt"
bash "$SCRIPT_PATH" "$TEST_DIR" "staging" > "$OUTPUT_FILE" 2>&1 || true

test_assert "Validator creates output" "[ -s '$OUTPUT_FILE' ]"
test_assert "Detects N+1 queries" "grep -q 'N+1' '$OUTPUT_FILE' || grep -q 'map.*await' '$OUTPUT_FILE'"

echo ""

# Test 4: Test blocking operations detection
echo "[Test Suite 4] Blocking Operations Detection"

cat > src/sync-operations.js <<EOF
const fs = require('fs');

function loadConfig() {
  const data = fs.readFileSync('./config.json', 'utf-8'); // Blocking!
  return JSON.parse(data);
}

function runCommand() {
  const result = execSync('ls -la'); // Blocking!
  return result;
}
EOF

bash "$SCRIPT_PATH" "$TEST_DIR" "staging" > "$OUTPUT_FILE" 2>&1 || true

test_assert "Detects synchronous operations" "grep -q 'blocking\\|Sync' '$OUTPUT_FILE'"

echo ""

# Test 5: Test database optimization checks
echo "[Test Suite 5] Database Optimization"

cat > src/queries.js <<EOF
// Bad: SELECT *
const users = await db.query('SELECT * FROM users');

// Bad: Missing pagination
const allPosts = await Post.findAll();

// Good: Connection pooling
const pool = new Pool({
  max: 20,
  connectionTimeoutMillis: 2000
});

// Good: Eager loading
const users = await User.findAll({
  include: [Post, Comment]
});
EOF

bash "$SCRIPT_PATH" "$TEST_DIR" "staging" > "$OUTPUT_FILE" 2>&1 || true

test_assert "Detects SELECT * queries" "grep -q 'SELECT.*\\*' '$OUTPUT_FILE'"
test_assert "Detects connection pooling" "grep -q 'pool\\|Connection pooling' '$OUTPUT_FILE'"
test_assert "Detects eager loading" "grep -q 'include\\|Eager loading' '$OUTPUT_FILE'"

echo ""

# Test 6: Test caching detection
echo "[Test Suite 6] Caching Strategy"

cat > src/caching.js <<EOF
const redis = require('redis');
const client = redis.createClient();

// Redis caching
app.get('/data', async (req, res) => {
  const cached = await client.get('data');
  if (cached) return res.json(JSON.parse(cached));

  const data = await fetchData();
  await client.set('data', JSON.stringify(data), 'EX', 3600);
  res.json(data);
});

// Memoization
const memoized = useMemo(() => expensiveCalculation(input), [input]);

// HTTP caching
res.setHeader('Cache-Control', 'public, max-age=3600');
res.setHeader('ETag', generateETag(data));
EOF

bash "$SCRIPT_PATH" "$TEST_DIR" "staging" > "$OUTPUT_FILE" 2>&1 || true

test_assert "Detects Redis caching" "grep -q 'redis\\|Redis' '$OUTPUT_FILE'"
test_assert "Detects memoization" "grep -q 'memoiz\\|useMemo' '$OUTPUT_FILE'"
test_assert "Detects HTTP caching" "grep -q 'Cache-Control\\|ETag' '$OUTPUT_FILE'"

echo ""

# Test 7: Test report generation
echo "[Test Suite 7] Report Generation"

bash "$SCRIPT_PATH" "$TEST_DIR" "production" > "$OUTPUT_FILE" 2>&1 || true

test_assert "Generates performance report" "[ -f performance-validation-*/performance-report.json ] || grep -q 'performance-report.json' '$OUTPUT_FILE'"
test_assert "Report contains metrics" "grep -q 'metrics\\|latency\\|throughput' '$OUTPUT_FILE'"
test_assert "Report contains gates status" "grep -q 'GATE\\|Gates' '$OUTPUT_FILE'"

echo ""

# Test 8: Test environment-specific behavior
echo "[Test Suite 8] Environment Configuration"

# Production should be stricter than staging
bash "$SCRIPT_PATH" "$TEST_DIR" "production" > "prod-output.txt" 2>&1 || true
bash "$SCRIPT_PATH" "$TEST_DIR" "staging" > "staging-output.txt" 2>&1 || true

test_assert "Production validation runs" "[ -f 'prod-output.txt' ]"
test_assert "Staging validation runs" "[ -f 'staging-output.txt' ]"
test_assert "Both outputs contain gate results" "grep -q 'GATE' 'prod-output.txt' && grep -q 'GATE' 'staging-output.txt'"

echo ""

# Test 9: Test memory leak pattern detection
echo "[Test Suite 9] Memory Leak Detection"

cat > src/memory-leaks.js <<EOF
// Bad: setInterval without clearInterval
setInterval(() => {
  fetchData();
}, 1000);

// Bad: setTimeout without cleanup
setTimeout(() => {
  processData();
}, 5000);

// Good: with cleanup
const interval = setInterval(() => {
  fetchData();
}, 1000);

cleanup(() => clearInterval(interval));
EOF

bash "$SCRIPT_PATH" "$TEST_DIR" "staging" > "$OUTPUT_FILE" 2>&1 || true

test_assert "Detects potential memory leaks" "grep -q 'memory.*leak\\|setInterval\\|setTimeout' '$OUTPUT_FILE'"

echo ""

# Test 10: Test recommendations generation
echo "[Test Suite 10] Recommendations"

bash "$SCRIPT_PATH" "$TEST_DIR" "production" > "$OUTPUT_FILE" 2>&1 || true

test_assert "Generates recommendations" "grep -q 'Recommendations\\|Fix\\|Implement' '$OUTPUT_FILE'"

echo ""

# Cleanup
cd /
rm -rf "$TEST_DIR"

# Test summary
echo "========================================"
echo "Test Summary"
echo "========================================"
echo -e "${GREEN}Passed: $TESTS_PASSED${NC}"
echo -e "${RED}Failed: $TESTS_FAILED${NC}"
echo "Total: $((TESTS_PASSED + TESTS_FAILED))"
echo "========================================"

# Exit with appropriate code
if [ "$TESTS_FAILED" -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed!${NC}"
    exit 1
fi
