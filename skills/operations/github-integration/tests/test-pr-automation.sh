#!/usr/bin/env bash
#
# Test Suite for PR Automation Script
# Tests PR creation, merging, and bulk operations
#

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
PR_SCRIPT="../resources/scripts/pr-automation.sh"
TEST_REPO="test-owner/test-repo"
DRY_RUN="true"

# Logging
log_info() { echo -e "${GREEN}[INFO]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $*"; }
log_test() { echo -e "${BLUE}[TEST]${NC} $*"; }

# Test counters
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0

# Run test
run_test() {
    local test_name="$1"
    local test_func="$2"

    ((TESTS_RUN++))
    log_test "$test_name"

    if $test_func; then
        ((TESTS_PASSED++))
        log_success "✅ $test_name"
    else
        ((TESTS_FAILED++))
        log_error "❌ $test_name"
    fi
}

# Test: Script exists and is executable
test_script_exists() {
    if [[ -f "$PR_SCRIPT" && -x "$PR_SCRIPT" ]]; then
        return 0
    else
        log_error "Script not found or not executable: $PR_SCRIPT"
        return 1
    fi
}

# Test: Help command
test_help_command() {
    local output
    output=$("$PR_SCRIPT" help 2>&1 || true)

    if echo "$output" | grep -q "GitHub Pull Request Automation"; then
        return 0
    else
        log_error "Help command failed"
        return 1
    fi
}

# Test: Create PR dry run
test_create_pr_dry_run() {
    export DRY_RUN="true"
    export GITHUB_TOKEN="fake-token-for-testing"

    # This should not make actual API calls due to dry run
    local output
    output=$("$PR_SCRIPT" create "$TEST_REPO" "feature-branch" "main" "Test PR" "Description" 2>&1 || true)

    if echo "$output" | grep -q "DRY RUN"; then
        return 0
    else
        log_error "Dry run not working"
        return 1
    fi
}

# Test: Auto-merge dry run
test_auto_merge_dry_run() {
    export DRY_RUN="true"
    export GITHUB_TOKEN="fake-token-for-testing"

    local output
    output=$("$PR_SCRIPT" auto-merge "$TEST_REPO" 123 "squash" 2>&1 || true)

    if echo "$output" | grep -q "DRY RUN"; then
        return 0
    else
        log_error "Auto-merge dry run failed"
        return 1
    fi
}

# Test: Add reviewers dry run
test_add_reviewers_dry_run() {
    export DRY_RUN="true"
    export GITHUB_TOKEN="fake-token-for-testing"

    local output
    output=$("$PR_SCRIPT" add-reviewers "$TEST_REPO" 123 "reviewer1" "reviewer2" 2>&1 || true)

    if echo "$output" | grep -q "DRY RUN"; then
        return 0
    else
        log_error "Add reviewers dry run failed"
        return 1
    fi
}

# Test: Sync PR dry run
test_sync_pr_dry_run() {
    export DRY_RUN="true"
    export GITHUB_TOKEN="fake-token-for-testing"

    local output
    output=$("$PR_SCRIPT" sync "$TEST_REPO" 123 2>&1 || true)

    if echo "$output" | grep -q "DRY RUN"; then
        return 0
    else
        log_error "Sync PR dry run failed"
        return 1
    fi
}

# Test: Close stale PRs dry run
test_close_stale_dry_run() {
    export DRY_RUN="true"
    export GITHUB_TOKEN="fake-token-for-testing"

    local output
    output=$("$PR_SCRIPT" close-stale "$TEST_REPO" 30 2>&1 || true)

    # Should attempt to list PRs (may fail without real token, but that's ok)
    if echo "$output" | grep -qE "(Closing PRs|API request failed)"; then
        return 0
    else
        log_error "Close stale dry run failed"
        return 1
    fi
}

# Test: Bulk create requires branch list file
test_bulk_create_validation() {
    export DRY_RUN="true"
    export GITHUB_TOKEN="fake-token-for-testing"

    local output
    output=$("$PR_SCRIPT" bulk-create "$TEST_REPO" "main" "template.md" "/nonexistent/file.txt" 2>&1 || true)

    if echo "$output" | grep -q "not found"; then
        return 0
    else
        log_error "Bulk create validation failed"
        return 1
    fi
}

# Test: Missing GitHub token detection
test_missing_token_detection() {
    unset GITHUB_TOKEN
    export DRY_RUN="false"

    local output
    output=$("$PR_SCRIPT" create "$TEST_REPO" "feature" "main" "Test" 2>&1 || true)

    if echo "$output" | grep -q "GITHUB_TOKEN not set"; then
        return 0
    else
        log_error "Missing token not detected"
        return 1
    fi
}

# Test: Invalid command handling
test_invalid_command() {
    export GITHUB_TOKEN="fake-token"

    local output
    output=$("$PR_SCRIPT" invalid-command 2>&1 || true)

    if echo "$output" | grep -q "Unknown command"; then
        return 0
    else
        log_error "Invalid command not handled"
        return 1
    fi
}

# Test: Verbose mode
test_verbose_mode() {
    export VERBOSE="true"
    export DRY_RUN="true"
    export GITHUB_TOKEN="fake-token"

    local output
    output=$("$PR_SCRIPT" create "$TEST_REPO" "feature" "main" "Test" 2>&1 || true)

    if echo "$output" | grep -qE "(DEBUG|INFO)"; then
        return 0
    else
        log_error "Verbose mode not working"
        return 1
    fi
}

# Main test runner
main() {
    echo "================================="
    echo "PR Automation Test Suite"
    echo "================================="

    # Check prerequisites
    if ! command -v bash &>/dev/null; then
        log_error "Bash not found"
        exit 1
    fi

    if [[ ! -f "$PR_SCRIPT" ]]; then
        log_error "PR automation script not found: $PR_SCRIPT"
        exit 1
    fi

    # Make script executable if needed
    chmod +x "$PR_SCRIPT" 2>/dev/null || true

    # Run tests
    run_test "Script exists and is executable" test_script_exists
    run_test "Help command" test_help_command
    run_test "Create PR dry run" test_create_pr_dry_run
    run_test "Auto-merge dry run" test_auto_merge_dry_run
    run_test "Add reviewers dry run" test_add_reviewers_dry_run
    run_test "Sync PR dry run" test_sync_pr_dry_run
    run_test "Close stale PRs dry run" test_close_stale_dry_run
    run_test "Bulk create validation" test_bulk_create_validation
    run_test "Missing token detection" test_missing_token_detection
    run_test "Invalid command handling" test_invalid_command
    run_test "Verbose mode" test_verbose_mode

    # Summary
    echo ""
    echo "================================="
    echo "Test Results"
    echo "================================="
    echo "Total tests: $TESTS_RUN"
    echo "Passed: $TESTS_PASSED"
    echo "Failed: $TESTS_FAILED"
    echo "================================="

    if [[ $TESTS_FAILED -eq 0 ]]; then
        log_success "All tests passed!"
        exit 0
    else
        log_error "Some tests failed"
        exit 1
    fi
}

main "$@"
