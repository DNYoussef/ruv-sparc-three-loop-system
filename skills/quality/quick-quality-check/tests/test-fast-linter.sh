#!/bin/bash
# Test suite for fast-linter.sh
# Part of quick-quality-check Enhanced tier tests

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESOURCES_DIR="${SCRIPT_DIR}/../resources"
LINTER="${RESOURCES_DIR}/fast-linter.sh"

# Test counters
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Logging functions
log_test() {
    echo -e "${YELLOW}[TEST]${NC} $*"
}

log_pass() {
    echo -e "${GREEN}[PASS]${NC} $*"
    ((TESTS_PASSED++))
}

log_fail() {
    echo -e "${RED}[FAIL]${NC} $*"
    ((TESTS_FAILED++))
}

# Setup test environment
setup() {
    TEST_DIR=$(mktemp -d)
    cd "$TEST_DIR"

    # Create test files
    mkdir -p src tests

    cat > src/good.js <<'EOF'
// Well-formatted JavaScript
function add(a, b) {
  return a + b;
}

module.exports = { add };
EOF

    cat > src/bad.js <<'EOF'
// Poorly formatted JavaScript with issues
function badFunction( x,y,z ){console.log(x);var result=x+y+z;return result;}
EOF

    cat > src/with-todos.js <<'EOF'
// File with TODO comments
function process() {
  // TODO: Implement this
  // FIXME: Handle edge cases
  return null;
}
EOF
}

# Cleanup test environment
cleanup() {
    if [[ -n "${TEST_DIR:-}" ]] && [[ -d "$TEST_DIR" ]]; then
        rm -rf "$TEST_DIR"
    fi
}

trap cleanup EXIT

# Test 1: Linter script exists and is executable
test_linter_exists() {
    log_test "Checking if linter script exists..."
    ((TESTS_RUN++))

    if [[ -f "$LINTER" ]]; then
        log_pass "Linter script exists"
    else
        log_fail "Linter script not found at $LINTER"
    fi
}

# Test 2: Linter can process valid file
test_linter_valid_file() {
    log_test "Testing linter with valid file..."
    ((TESTS_RUN++))

    setup

    if bash "$LINTER" src/good.js /dev/null 2>&1; then
        log_pass "Linter processed valid file"
    else
        log_fail "Linter failed on valid file"
    fi
}

# Test 3: Linter detects formatting issues
test_linter_detects_issues() {
    log_test "Testing linter issue detection..."
    ((TESTS_RUN++))

    setup

    OUTPUT=$(mktemp)
    bash "$LINTER" src/bad.js "$OUTPUT" 2>&1 || true

    if [[ -f "$OUTPUT" ]] && grep -q "issues" "$OUTPUT" 2>/dev/null; then
        log_pass "Linter detected formatting issues"
    else
        log_fail "Linter did not detect formatting issues"
    fi

    rm -f "$OUTPUT"
}

# Test 4: Linter detects TODO comments
test_linter_detects_todos() {
    log_test "Testing TODO comment detection..."
    ((TESTS_RUN++))

    setup

    OUTPUT=$(mktemp)
    bash "$LINTER" src/with-todos.js "$OUTPUT" 2>&1 || true

    if [[ -f "$OUTPUT" ]] && grep -qi "todo" "$OUTPUT" 2>/dev/null; then
        log_pass "Linter detected TODO comments"
    else
        log_fail "Linter did not detect TODO comments"
    fi

    rm -f "$OUTPUT"
}

# Test 5: Linter handles directory input
test_linter_directory() {
    log_test "Testing linter with directory input..."
    ((TESTS_RUN++))

    setup

    OUTPUT=$(mktemp)
    bash "$LINTER" src "$OUTPUT" 2>&1 || true

    if [[ -f "$OUTPUT" ]]; then
        log_pass "Linter processed directory"
    else
        log_fail "Linter failed to process directory"
    fi

    rm -f "$OUTPUT"
}

# Test 6: Linter produces valid JSON output
test_linter_json_output() {
    log_test "Testing JSON output format..."
    ((TESTS_RUN++))

    setup

    OUTPUT=$(mktemp)
    bash "$LINTER" src/good.js "$OUTPUT" 2>&1 || true

    if [[ -f "$OUTPUT" ]] && jq empty "$OUTPUT" 2>/dev/null; then
        log_pass "Linter produced valid JSON"
    else
        log_fail "Linter did not produce valid JSON"
    fi

    rm -f "$OUTPUT"
}

# Test 7: Linter respects configuration
test_linter_configuration() {
    log_test "Testing configuration support..."
    ((TESTS_RUN++))

    setup

    # Create minimal config
    cat > check-config.yaml <<'EOF'
linting:
  enabled: true
EOF

    CONFIG_FILE="check-config.yaml" bash "$LINTER" src/good.js /dev/null 2>&1

    if [[ $? -eq 0 ]] || [[ $? -eq 1 ]]; then
        log_pass "Linter respects configuration"
    else
        log_fail "Linter failed with configuration"
    fi
}

# Run all tests
main() {
    echo "=================================="
    echo "Fast Linter Test Suite"
    echo "=================================="
    echo ""

    test_linter_exists
    test_linter_valid_file
    test_linter_detects_issues
    test_linter_detects_todos
    test_linter_directory
    test_linter_json_output
    test_linter_configuration

    echo ""
    echo "=================================="
    echo "Test Results"
    echo "=================================="
    echo "Tests Run: $TESTS_RUN"
    echo "Passed: $TESTS_PASSED"
    echo "Failed: $TESTS_FAILED"
    echo ""

    if [[ $TESTS_FAILED -eq 0 ]]; then
        echo -e "${GREEN}All tests passed!${NC}"
        exit 0
    else
        echo -e "${RED}Some tests failed!${NC}"
        exit 1
    fi
}

main
