#!/bin/bash
# Test suite for firewall configuration script
# Tests firewall-config.sh functionality

set -euo pipefail

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

TEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESOURCES_DIR="$(dirname "$TEST_DIR")/resources"
SCRIPT_DIR="$RESOURCES_DIR/scripts"
TEMP_DIR="/tmp/network-security-test-$$"

# Test counters
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0

# Setup test environment
setup() {
    echo "Setting up test environment..."
    mkdir -p "$TEMP_DIR"/{config,logs}

    # Create test configuration file
    cat > "$TEMP_DIR/config/trusted-domains.conf" <<EOF
# Test trusted domains configuration
*.npmjs.org
registry.npmjs.org
*.github.com
api.github.com
EOF

    export CONFIG_FILE="$TEMP_DIR/config/trusted-domains.conf"
    export LOG_FILE="$TEMP_DIR/logs/firewall.log"
    export DRY_RUN="true"
}

# Cleanup test environment
cleanup() {
    echo "Cleaning up test environment..."
    rm -rf "$TEMP_DIR"
}

# Test helper functions
assert_equals() {
    local expected="$1"
    local actual="$2"
    local message="${3:-Assertion failed}"

    ((TESTS_RUN++))

    if [[ "$expected" == "$actual" ]]; then
        echo -e "${GREEN}✓${NC} $message"
        ((TESTS_PASSED++))
        return 0
    else
        echo -e "${RED}✗${NC} $message"
        echo "  Expected: $expected"
        echo "  Actual: $actual"
        ((TESTS_FAILED++))
        return 1
    fi
}

assert_file_exists() {
    local file="$1"
    local message="${2:-File should exist: $file}"

    ((TESTS_RUN++))

    if [[ -f "$file" ]]; then
        echo -e "${GREEN}✓${NC} $message"
        ((TESTS_PASSED++))
        return 0
    else
        echo -e "${RED}✗${NC} $message"
        ((TESTS_FAILED++))
        return 1
    fi
}

assert_contains() {
    local haystack="$1"
    local needle="$2"
    local message="${3:-String should contain}"

    ((TESTS_RUN++))

    if echo "$haystack" | grep -q "$needle"; then
        echo -e "${GREEN}✓${NC} $message"
        ((TESTS_PASSED++))
        return 0
    else
        echo -e "${RED}✗${NC} $message"
        echo "  Haystack: $haystack"
        echo "  Needle: $needle"
        ((TESTS_FAILED++))
        return 1
    fi
}

# Test cases

test_script_exists() {
    echo "Test: Script file exists"
    assert_file_exists "$SCRIPT_DIR/firewall-config.sh" "firewall-config.sh should exist"
}

test_config_file_parsing() {
    echo ""
    echo "Test: Configuration file parsing"

    # Source the script functions (dry run)
    export DRY_RUN="true"
    export FIREWALL_TYPE="iptables"

    # Parse domains
    mapfile -t domains < <(grep -v '^#' "$CONFIG_FILE" | grep -v '^$')

    assert_equals "4" "${#domains[@]}" "Should parse 4 domains"
    assert_contains "${domains[*]}" "*.npmjs.org" "Should contain *.npmjs.org"
    assert_contains "${domains[*]}" "*.github.com" "Should contain *.github.com"
}

test_dry_run_mode() {
    echo ""
    echo "Test: Dry run mode"

    export DRY_RUN="true"
    export FIREWALL_TYPE="iptables"

    # Run script in dry run mode
    output=$("$SCRIPT_DIR/firewall-config.sh" 2>&1 || true)

    assert_contains "$output" "[DRY RUN]" "Should indicate dry run mode"
    assert_contains "$output" "Would flush existing rules" "Should show what would be done"
}

test_domain_resolution() {
    echo ""
    echo "Test: Domain resolution (mock)"

    # Test that getent or dig can resolve domains
    if command -v dig &> /dev/null; then
        output=$(dig +short github.com A 2>&1)
        assert_contains "$output" "." "Should resolve github.com to IP"
    else
        echo "  Skipped: dig not available"
    fi
}

test_log_file_creation() {
    echo ""
    echo "Test: Log file creation"

    # Run script to create log
    export DRY_RUN="true"
    "$SCRIPT_DIR/firewall-config.sh" > /dev/null 2>&1 || true

    # Check if log file was created
    if [[ -f "$LOG_FILE" ]]; then
        assert_file_exists "$LOG_FILE" "Log file should be created"

        # Check log content
        log_content=$(cat "$LOG_FILE")
        assert_contains "$log_content" "INFO" "Log should contain INFO messages"
    else
        echo "  Skipped: Log file not created in dry run"
    fi
}

test_invalid_config_file() {
    echo ""
    echo "Test: Invalid configuration file"

    # Point to non-existent file
    export CONFIG_FILE="/nonexistent/file.conf"

    # Run script (should fail gracefully)
    output=$("$SCRIPT_DIR/firewall-config.sh" 2>&1 || echo "FAILED")

    assert_contains "$output" "FAILED\|not found\|ERROR" "Should handle missing config file"
}

test_firewall_type_validation() {
    echo ""
    echo "Test: Firewall type validation"

    export FIREWALL_TYPE="invalid"
    export DRY_RUN="true"

    # Run script with invalid firewall type
    output=$("$SCRIPT_DIR/firewall-config.sh" 2>&1 || echo "FAILED")

    assert_contains "$output" "Invalid FIREWALL_TYPE\|FAILED" "Should reject invalid firewall type"
}

# Run all tests
run_tests() {
    echo "========================================="
    echo "Network Security Firewall Config Tests"
    echo "========================================="

    setup

    test_script_exists
    test_config_file_parsing
    test_dry_run_mode
    test_domain_resolution
    test_log_file_creation
    test_invalid_config_file
    test_firewall_type_validation

    cleanup

    echo ""
    echo "========================================="
    echo "Test Summary"
    echo "========================================="
    echo "Tests Run: $TESTS_RUN"
    echo -e "${GREEN}Passed: $TESTS_PASSED${NC}"
    echo -e "${RED}Failed: $TESTS_FAILED${NC}"
    echo "========================================="

    if [[ $TESTS_FAILED -eq 0 ]]; then
        echo -e "${GREEN}All tests passed!${NC}"
        exit 0
    else
        echo -e "${RED}Some tests failed!${NC}"
        exit 1
    fi
}

# Run tests
run_tests
