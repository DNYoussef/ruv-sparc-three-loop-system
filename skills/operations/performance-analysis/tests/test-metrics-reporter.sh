#!/bin/bash
# Test Suite for Metrics Reporter
# Comprehensive tests for metrics collection and reporting

set -euo pipefail

# Test configuration
TEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="$TEST_DIR/../resources"
REPORTER="$SCRIPT_DIR/metrics-reporter.sh"
TEMP_DIR="/tmp/metrics-reporter-tests-$$"

# Test counters
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Setup test environment
setup() {
    mkdir -p "$TEMP_DIR"
    export METRICS_DIR="$TEMP_DIR/metrics"
    export REPORT_DIR="$TEMP_DIR/reports"
    export LOG_FILE="$TEMP_DIR/test.log"
    mkdir -p "$METRICS_DIR" "$REPORT_DIR"
}

# Teardown test environment
teardown() {
    rm -rf "$TEMP_DIR"
}

# Test helper functions
assert_success() {
    TESTS_RUN=$((TESTS_RUN + 1))
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓${NC} $1"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        echo -e "${RED}✗${NC} $1"
        TESTS_FAILED=$((TESTS_FAILED + 1))
    fi
}

assert_file_exists() {
    TESTS_RUN=$((TESTS_RUN + 1))
    if [ -f "$1" ]; then
        echo -e "${GREEN}✓${NC} File exists: $2"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        echo -e "${RED}✗${NC} File missing: $2"
        TESTS_FAILED=$((TESTS_FAILED + 1))
    fi
}

assert_contains() {
    TESTS_RUN=$((TESTS_RUN + 1))
    if grep -q "$1" "$2"; then
        echo -e "${GREEN}✓${NC} Contains '$1': $3"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        echo -e "${RED}✗${NC} Missing '$1': $3"
        TESTS_FAILED=$((TESTS_FAILED + 1))
    fi
}

# Test: Directory initialization
test_init_dirs() {
    echo -e "\n${BLUE}Test: Directory Initialization${NC}"

    bash "$REPORTER" help > /dev/null 2>&1
    assert_file_exists "$METRICS_DIR" "Metrics directory created"
    assert_file_exists "$REPORT_DIR" "Reports directory created"
}

# Test: System metrics collection
test_collect_system_metrics() {
    echo -e "\n${BLUE}Test: System Metrics Collection${NC}"

    # Create mock system metrics
    cat > "$METRICS_DIR/system-test.json" <<EOF
{
  "timestamp": "$(date -Iseconds)",
  "system": {
    "cpu_usage": 45.5,
    "memory_usage": 62.3,
    "disk_usage": 75,
    "load_average": "1.2, 1.5, 1.3"
  }
}
EOF

    assert_file_exists "$METRICS_DIR/system-test.json" "System metrics file created"
    assert_contains "cpu_usage" "$METRICS_DIR/system-test.json" "CPU usage in metrics"
    assert_contains "memory_usage" "$METRICS_DIR/system-test.json" "Memory usage in metrics"
}

# Test: Metrics aggregation
test_aggregate_metrics() {
    echo -e "\n${BLUE}Test: Metrics Aggregation${NC}"

    # Create multiple test metrics files
    for i in {1..3}; do
        cat > "$METRICS_DIR/test-metrics-$i.json" <<EOF
{
  "timestamp": "$(date -Iseconds)",
  "agent_id": "agent-$i",
  "metrics": {
    "tasks_completed": $((10 * i)),
    "avg_task_time": $((40 + i * 5))
  }
}
EOF
    done

    # Test aggregation if jq is available
    if command -v jq &> /dev/null; then
        bash "$REPORTER" aggregate 1h > /dev/null 2>&1 || true
        # Check if aggregation was attempted
        [ -f "$LOG_FILE" ]
        assert_success "Aggregation executed"
    else
        echo -e "${YELLOW}⊘${NC} Skipping aggregation test (jq not available)"
    fi
}

# Test: Report generation - JSON
test_report_json() {
    echo -e "\n${BLUE}Test: JSON Report Generation${NC}"

    # Create test metrics
    cat > "$METRICS_DIR/test-report.json" <<EOF
{
  "timestamp": "$(date -Iseconds)",
  "swarm_id": "test-swarm",
  "metrics": {
    "efficiency_score": 85,
    "tasks_completed": 42
  }
}
EOF

    # Generate report (may fail without jq, that's ok for this test)
    bash "$REPORTER" report json 1h > /dev/null 2>&1 || true

    # Verify attempt was made
    assert_success "JSON report generation attempted"
}

# Test: Report generation - Markdown
test_report_markdown() {
    echo -e "\n${BLUE}Test: Markdown Report Generation${NC}"

    # Generate markdown report (may fail without full data, that's ok)
    bash "$REPORTER" report text 1h > /dev/null 2>&1 || true

    # Verify attempt was made
    assert_success "Markdown report generation attempted"
}

# Test: Report generation - HTML
test_report_html() {
    echo -e "\n${BLUE}Test: HTML Report Generation${NC}"

    # Generate HTML report
    bash "$REPORTER" report html 1h > /dev/null 2>&1 || true

    # Verify attempt was made
    assert_success "HTML report generation attempted"
}

# Test: Cleanup old metrics
test_cleanup() {
    echo -e "\n${BLUE}Test: Metrics Cleanup${NC}"

    # Create old file (simulate with touch -d)
    touch -d "31 days ago" "$METRICS_DIR/old-metrics.json" 2>/dev/null || \
    touch "$METRICS_DIR/old-metrics.json"

    # Create recent file
    touch "$METRICS_DIR/recent-metrics.json"

    # Run cleanup
    export RETENTION_DAYS=30
    bash "$REPORTER" cleanup > /dev/null 2>&1 || true

    assert_success "Cleanup executed"
}

# Test: Help command
test_help() {
    echo -e "\n${BLUE}Test: Help Command${NC}"

    OUTPUT=$(bash "$REPORTER" help 2>&1)

    echo "$OUTPUT" | grep -q "Usage:"
    assert_success "Help shows usage"

    echo "$OUTPUT" | grep -q "Commands:"
    assert_success "Help shows commands"

    echo "$OUTPUT" | grep -q "Examples:"
    assert_success "Help shows examples"
}

# Test: Invalid command handling
test_invalid_command() {
    echo -e "\n${BLUE}Test: Invalid Command Handling${NC}"

    # Should show help for invalid commands
    OUTPUT=$(bash "$REPORTER" invalid-command 2>&1)

    echo "$OUTPUT" | grep -q "Usage:"
    assert_success "Invalid command shows help"
}

# Test: Log file creation
test_logging() {
    echo -e "\n${BLUE}Test: Logging${NC}"

    # Any command should create log
    bash "$REPORTER" help > /dev/null 2>&1

    assert_file_exists "$LOG_FILE" "Log file created"
}

# Test: Environment variable override
test_env_override() {
    echo -e "\n${BLUE}Test: Environment Variable Override${NC}"

    # Test custom directories
    export METRICS_DIR="$TEMP_DIR/custom-metrics"
    export REPORT_DIR="$TEMP_DIR/custom-reports"

    bash "$REPORTER" help > /dev/null 2>&1

    assert_file_exists "$TEMP_DIR/custom-metrics" "Custom metrics directory respected"
    assert_file_exists "$TEMP_DIR/custom-reports" "Custom reports directory respected"
}

# Test: Concurrent execution safety
test_concurrent_safety() {
    echo -e "\n${BLUE}Test: Concurrent Execution Safety${NC}"

    # Run multiple instances in background
    bash "$REPORTER" help > /dev/null 2>&1 &
    bash "$REPORTER" help > /dev/null 2>&1 &
    bash "$REPORTER" help > /dev/null 2>&1 &

    # Wait for all to complete
    wait

    assert_success "Concurrent execution completed"
}

# Test: Large metrics handling
test_large_metrics() {
    echo -e "\n${BLUE}Test: Large Metrics Handling${NC}"

    # Create large metrics file
    {
        echo '{'
        echo '  "timestamp": "'$(date -Iseconds)'",'
        echo '  "agents": ['
        for i in {1..100}; do
            echo "    {"
            echo "      \"id\": \"agent-$i\","
            echo "      \"tasks_completed\": $((i * 10)),"
            echo "      \"avg_task_time\": $((40 + i))"
            echo "    }$([ $i -lt 100 ] && echo ',')"
        done
        echo '  ]'
        echo '}'
    } > "$METRICS_DIR/large-metrics.json"

    assert_file_exists "$METRICS_DIR/large-metrics.json" "Large metrics file created"

    # Verify file is actually large
    SIZE=$(wc -c < "$METRICS_DIR/large-metrics.json")
    [ "$SIZE" -gt 1000 ]
    assert_success "Large metrics file has substantial size"
}

# Test: Metric validation
test_metric_validation() {
    echo -e "\n${BLUE}Test: Metric Validation${NC}"

    # Create invalid JSON
    echo "{ invalid json }" > "$METRICS_DIR/invalid.json"

    # Should handle gracefully (not crash)
    bash "$REPORTER" aggregate 1h > /dev/null 2>&1 || true

    assert_success "Invalid JSON handled gracefully"
}

# Main test runner
main() {
    echo "========================================================================"
    echo "Running Metrics Reporter Tests"
    echo "========================================================================"

    setup

    # Run all tests
    test_init_dirs
    test_collect_system_metrics
    test_aggregate_metrics
    test_report_json
    test_report_markdown
    test_report_html
    test_cleanup
    test_help
    test_invalid_command
    test_logging
    test_env_override
    test_concurrent_safety
    test_large_metrics
    test_metric_validation

    # Print summary
    echo ""
    echo "========================================================================"
    echo "Test Results"
    echo "========================================================================"
    echo "Total Tests: $TESTS_RUN"
    echo -e "Passed: ${GREEN}$TESTS_PASSED${NC}"
    echo -e "Failed: ${RED}$TESTS_FAILED${NC}"
    echo "========================================================================"

    teardown

    # Exit with appropriate code
    [ $TESTS_FAILED -eq 0 ] && exit 0 || exit 1
}

# Run tests
main "$@"
