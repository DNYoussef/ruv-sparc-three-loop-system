#!/bin/bash

###############################################################################
# Test suite for trajectory tracking functionality
# Tests trajectory initialization, step recording, completion, and analysis.
###############################################################################

set -euo pipefail

# Test configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRACKER_SCRIPT="${SCRIPT_DIR}/../resources/trajectory-tracker.sh"
TEST_DIR="${SCRIPT_DIR}/../../../tests/temp/trajectory-tests"
export TRAJECTORY_LOG="${TEST_DIR}/test-trajectories.jsonl"
export AGENTDB_PATH="${TEST_DIR}/test.db"
export DOMAIN="testing"
export AGENT_NAME="test-agent"

# Test results
TESTS_PASSED=0
TESTS_FAILED=0
declare -a FAILED_TESTS

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Setup test environment
setup() {
    log_info "Setting up test environment..."
    mkdir -p "${TEST_DIR}"
    rm -f "${TRAJECTORY_LOG}"

    # Make tracker script executable
    chmod +x "${TRACKER_SCRIPT}"
}

# Cleanup test environment
cleanup() {
    log_info "Cleaning up test environment..."
    rm -rf "${TEST_DIR}"
}

# Run a single test
run_test() {
    local test_name="$1"
    local test_fn="$2"

    echo ""
    echo "Running test: ${test_name}"

    if $test_fn; then
        echo -e "✓ ${test_name} ${GREEN}PASSED${NC}"
        ((TESTS_PASSED++))
    else
        echo -e "✗ ${test_name} ${RED}FAILED${NC}"
        ((TESTS_FAILED++))
        FAILED_TESTS+=("${test_name}")
        return 1
    fi
}

# Test 1: Trajectory initialization
test_init_trajectory() {
    local traj_id
    traj_id=$("${TRACKER_SCRIPT}" init "test-task-1" "Test task description" 2>/dev/null)

    if [[ -z "${traj_id}" ]]; then
        log_error "Trajectory ID should not be empty"
        return 1
    fi

    if [[ ! "${traj_id}" =~ ^traj_test-task-1_ ]]; then
        log_error "Trajectory ID format incorrect: ${traj_id}"
        return 1
    fi

    if [[ ! -f "/tmp/${traj_id}.json" ]]; then
        log_error "Trajectory file not created"
        return 1
    fi

    # Verify trajectory metadata
    if ! grep -q "\"task\": \"Test task description\"" "/tmp/${traj_id}.json"; then
        log_error "Trajectory metadata incorrect"
        return 1
    fi

    # Cleanup
    rm -f "/tmp/${traj_id}.json"

    return 0
}

# Test 2: Step recording
test_record_step() {
    # Initialize trajectory
    local traj_id
    traj_id=$("${TRACKER_SCRIPT}" init "test-task-2" "Step recording test" 2>/dev/null)

    # Record step
    "${TRACKER_SCRIPT}" step "${traj_id}" "test-action" "test-result" 2>/dev/null

    # Verify step was recorded
    if ! grep -q "\"action\": \"test-action\"" "/tmp/${traj_id}.json"; then
        log_error "Step action not recorded"
        rm -f "/tmp/${traj_id}.json"
        return 1
    fi

    if ! grep -q "\"result\": \"test-result\"" "/tmp/${traj_id}.json"; then
        log_error "Step result not recorded"
        rm -f "/tmp/${traj_id}.json"
        return 1
    fi

    # Cleanup
    rm -f "/tmp/${traj_id}.json"

    return 0
}

# Test 3: Multiple steps
test_multiple_steps() {
    local traj_id
    traj_id=$("${TRACKER_SCRIPT}" init "test-task-3" "Multiple steps test" 2>/dev/null)

    # Record multiple steps
    "${TRACKER_SCRIPT}" step "${traj_id}" "step-1" "result-1" 2>/dev/null
    "${TRACKER_SCRIPT}" step "${traj_id}" "step-2" "result-2" 2>/dev/null
    "${TRACKER_SCRIPT}" step "${traj_id}" "step-3" "result-3" 2>/dev/null

    # Verify all steps recorded
    local step_count
    step_count=$(grep -o "\"action\":" "/tmp/${traj_id}.json" | wc -l)

    if [[ ${step_count} -ne 3 ]]; then
        log_error "Expected 3 steps, found ${step_count}"
        rm -f "/tmp/${traj_id}.json"
        return 1
    fi

    # Cleanup
    rm -f "/tmp/${traj_id}.json"

    return 0
}

# Test 4: Trajectory completion
test_complete_trajectory() {
    local traj_id
    traj_id=$("${TRACKER_SCRIPT}" init "test-task-4" "Completion test" 2>/dev/null)

    # Record steps
    "${TRACKER_SCRIPT}" step "${traj_id}" "action-1" "result-1" 2>/dev/null
    "${TRACKER_SCRIPT}" step "${traj_id}" "action-2" "result-2" 2>/dev/null

    # Complete trajectory
    "${TRACKER_SCRIPT}" complete "${traj_id}" "success" '{"time": 100}' 2>/dev/null

    # Verify completion in log
    if [[ ! -f "${TRAJECTORY_LOG}" ]]; then
        log_error "Trajectory log not created"
        return 1
    fi

    if ! grep -q "\"outcome\": \"success\"" "${TRAJECTORY_LOG}"; then
        log_error "Outcome not recorded in log"
        return 1
    fi

    if ! grep -q "\"time\": 100" "${TRAJECTORY_LOG}"; then
        log_error "Metrics not recorded"
        return 1
    fi

    # Verify cleanup
    if [[ -f "/tmp/${traj_id}.json" ]]; then
        log_error "Trajectory file not cleaned up"
        return 1
    fi

    return 0
}

# Test 5: Success tracking
test_success_tracking() {
    # Create successful trajectory
    local traj_id_1
    traj_id_1=$("${TRACKER_SCRIPT}" init "success-1" "Success test" 2>/dev/null)
    "${TRACKER_SCRIPT}" step "${traj_id_1}" "action" "result" 2>/dev/null
    "${TRACKER_SCRIPT}" complete "${traj_id_1}" "success" 2>/dev/null

    # Create failed trajectory
    local traj_id_2
    traj_id_2=$("${TRACKER_SCRIPT}" init "failure-1" "Failure test" 2>/dev/null)
    "${TRACKER_SCRIPT}" step "${traj_id_2}" "action" "error" 2>/dev/null
    "${TRACKER_SCRIPT}" complete "${traj_id_2}" "failure" 2>/dev/null

    # Check trajectory log
    local success_count
    success_count=$(grep -c "\"outcome\": \"success\"" "${TRAJECTORY_LOG}" || echo "0")

    local failure_count
    failure_count=$(grep -c "\"outcome\": \"failure\"" "${TRAJECTORY_LOG}" || echo "0")

    if [[ ${success_count} -ne 1 ]]; then
        log_error "Expected 1 success, found ${success_count}"
        return 1
    fi

    if [[ ${failure_count} -ne 1 ]]; then
        log_error "Expected 1 failure, found ${failure_count}"
        return 1
    fi

    return 0
}

# Test 6: Analysis
test_analyze_patterns() {
    # Create multiple trajectories
    for i in {1..5}; do
        local traj_id
        traj_id=$("${TRACKER_SCRIPT}" init "task-${i}" "Analysis test ${i}" 2>/dev/null)
        "${TRACKER_SCRIPT}" step "${traj_id}" "common-action" "result-${i}" 2>/dev/null

        if [[ $i -le 3 ]]; then
            "${TRACKER_SCRIPT}" complete "${traj_id}" "success" 2>/dev/null
        else
            "${TRACKER_SCRIPT}" complete "${traj_id}" "failure" 2>/dev/null
        fi
    done

    # Run analysis
    local analysis_output
    analysis_output=$("${TRACKER_SCRIPT}" analyze 2>/dev/null)

    # Verify analysis output
    if [[ ! "${analysis_output}" =~ "Total Trajectories: 5" ]]; then
        log_error "Analysis should show 5 trajectories"
        return 1
    fi

    if [[ ! "${analysis_output}" =~ "Successful: 3" ]]; then
        log_error "Analysis should show 3 successes"
        return 1
    fi

    return 0
}

# Test 7: Export
test_export_trajectories() {
    local export_file="${TEST_DIR}/export.json"

    # Export trajectories
    "${TRACKER_SCRIPT}" export "${export_file}" 2>/dev/null

    if [[ ! -f "${export_file}" ]]; then
        log_error "Export file not created"
        return 1
    fi

    # Verify export has content
    if [[ ! -s "${export_file}" ]]; then
        log_error "Export file is empty"
        return 1
    fi

    return 0
}

# Test 8: Cleanup old trajectories
test_cleanup_old() {
    local before_count
    before_count=$(grep -c "\"id\":" "${TRAJECTORY_LOG}" || echo "0")

    # Cleanup trajectories older than 0 days (should remove all)
    "${TRACKER_SCRIPT}" cleanup 0 2>/dev/null

    local after_count
    after_count=$(grep -c "\"id\":" "${TRAJECTORY_LOG}" 2>/dev/null || echo "0")

    if [[ ${after_count} -ge ${before_count} ]]; then
        log_error "Cleanup did not remove trajectories"
        return 1
    fi

    return 0
}

# Main test execution
main() {
    echo "========================================"
    echo "Trajectory Tracking Test Suite"
    echo "========================================"

    setup

    run_test "Trajectory initialization" test_init_trajectory
    run_test "Step recording" test_record_step
    run_test "Multiple steps" test_multiple_steps
    run_test "Trajectory completion" test_complete_trajectory
    run_test "Success tracking" test_success_tracking
    run_test "Pattern analysis" test_analyze_patterns
    run_test "Export trajectories" test_export_trajectories
    run_test "Cleanup old trajectories" test_cleanup_old

    cleanup

    echo ""
    echo "========================================"
    echo "Test Summary:"
    echo "  Passed: ${TESTS_PASSED}"
    echo "  Failed: ${TESTS_FAILED}"

    if [[ ${TESTS_FAILED} -gt 0 ]]; then
        echo ""
        echo "Failed tests:"
        for test in "${FAILED_TESTS[@]}"; do
            echo "  - ${test}"
        done
    fi
    echo "========================================"

    if [[ ${TESTS_FAILED} -eq 0 ]]; then
        exit 0
    else
        exit 1
    fi
}

main
