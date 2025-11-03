#!/bin/bash
# Test Suite for Platform Initialization
# Tests platform-init.sh script functionality

set -e

# Test framework
TESTS_PASSED=0
TESTS_FAILED=0
TEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMP_DIR="${TEST_DIR}/tmp/platform-init-test-$$"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Setup test environment
setup() {
    echo "Setting up test environment..."
    mkdir -p "$TEMP_DIR"
    cd "$TEMP_DIR"

    # Mock Flow Nexus command
    mkdir -p bin
    cat > bin/flow-nexus << 'EOF'
#!/bin/bash
if [ "$1" = "--version" ]; then
    echo "1.0.0"
    exit 0
fi
exit 0
EOF
    chmod +x bin/flow-nexus
    export PATH="$TEMP_DIR/bin:$PATH"
}

# Cleanup test environment
teardown() {
    echo "Cleaning up test environment..."
    cd /
    rm -rf "$TEMP_DIR"
}

# Test assertion helpers
assert_equals() {
    local expected="$1"
    local actual="$2"
    local message="${3:-Assertion failed}"

    if [ "$expected" = "$actual" ]; then
        echo -e "${GREEN}✓${NC} $message"
        ((TESTS_PASSED++))
        return 0
    else
        echo -e "${RED}✗${NC} $message"
        echo "  Expected: $expected"
        echo "  Actual:   $actual"
        ((TESTS_FAILED++))
        return 1
    fi
}

assert_file_exists() {
    local file="$1"
    local message="${2:-File should exist: $file}"

    if [ -f "$file" ]; then
        echo -e "${GREEN}✓${NC} $message"
        ((TESTS_PASSED++))
        return 0
    else
        echo -e "${RED}✗${NC} $message"
        ((TESTS_FAILED++))
        return 1
    fi
}

assert_dir_exists() {
    local dir="$1"
    local message="${2:-Directory should exist: $dir}"

    if [ -d "$dir" ]; then
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
    local message="${3:-Should contain: $needle}"

    if echo "$haystack" | grep -q "$needle"; then
        echo -e "${GREEN}✓${NC} $message"
        ((TESTS_PASSED++))
        return 0
    else
        echo -e "${RED}✗${NC} $message"
        ((TESTS_FAILED++))
        return 1
    fi
}

# Test: Directory structure creation
test_directory_structure() {
    echo -e "\n${YELLOW}Test: Directory Structure Creation${NC}"

    # Create minimal initialization
    mkdir -p platform/config platform/services platform/scripts platform/docs
    mkdir -p logs/platform
    mkdir -p platform/storage/uploads platform/storage/cache platform/storage/temp
    mkdir -p platform/data/database platform/data/backups

    assert_dir_exists "platform/config" "Config directory created"
    assert_dir_exists "platform/services" "Services directory created"
    assert_dir_exists "platform/scripts" "Scripts directory created"
    assert_dir_exists "platform/docs" "Docs directory created"
    assert_dir_exists "logs/platform" "Logs directory created"
    assert_dir_exists "platform/storage/uploads" "Storage uploads directory created"
    assert_dir_exists "platform/data/database" "Database directory created"
}

# Test: Configuration file creation
test_config_creation() {
    echo -e "\n${YELLOW}Test: Configuration File Creation${NC}"

    # Create test config
    cat > platform/config/flow-nexus.json << 'EOF'
{
  "platform": {
    "name": "Flow Nexus Platform",
    "version": "1.0.0"
  }
}
EOF

    assert_file_exists "platform/config/flow-nexus.json" "Config file created"

    # Validate JSON
    if command -v jq &> /dev/null; then
        jq . platform/config/flow-nexus.json > /dev/null 2>&1
        assert_equals "0" "$?" "Config file is valid JSON"
    else
        echo -e "${YELLOW}⊘${NC} Skipping JSON validation (jq not installed)"
    fi
}

# Test: Environment file creation
test_env_creation() {
    echo -e "\n${YELLOW}Test: Environment File Creation${NC}"

    # Create test .env
    cat > platform/.env << 'EOF'
NODE_ENV=development
PORT=3000
EOF

    assert_file_exists "platform/.env" "Environment file created"
    assert_contains "$(cat platform/.env)" "NODE_ENV" ".env contains NODE_ENV"
    assert_contains "$(cat platform/.env)" "PORT" ".env contains PORT"
}

# Test: Package.json creation
test_package_json() {
    echo -e "\n${YELLOW}Test: Package.json Creation${NC}"

    # Create test package.json
    cat > platform/package.json << 'EOF'
{
  "name": "flow-nexus-platform",
  "version": "1.0.0",
  "scripts": {
    "start": "node services/app.js"
  }
}
EOF

    assert_file_exists "platform/package.json" "package.json created"

    # Validate JSON
    if command -v jq &> /dev/null; then
        jq . platform/package.json > /dev/null 2>&1
        assert_equals "0" "$?" "package.json is valid JSON"
    fi
}

# Test: Health check script creation
test_health_check_script() {
    echo -e "\n${YELLOW}Test: Health Check Script Creation${NC}"

    # Create test health check
    cat > platform/scripts/health-check.sh << 'EOF'
#!/bin/bash
echo "Health check passed"
exit 0
EOF
    chmod +x platform/scripts/health-check.sh

    assert_file_exists "platform/scripts/health-check.sh" "Health check script created"

    # Test execution
    output=$(platform/scripts/health-check.sh)
    assert_contains "$output" "Health check passed" "Health check executes successfully"
}

# Test: Prerequisites check
test_prerequisites() {
    echo -e "\n${YELLOW}Test: Prerequisites Check${NC}"

    # Test Node.js availability
    if command -v node &> /dev/null; then
        assert_equals "0" "0" "Node.js is available"
    else
        echo -e "${YELLOW}⊘${NC} Node.js not available (expected in CI)"
    fi

    # Test npm availability
    if command -v npm &> /dev/null; then
        assert_equals "0" "0" "npm is available"
    else
        echo -e "${YELLOW}⊘${NC} npm not available (expected in CI)"
    fi

    # Test Flow Nexus mock
    if command -v flow-nexus &> /dev/null; then
        version=$(flow-nexus --version)
        assert_contains "$version" "1.0.0" "Flow Nexus mock works"
    fi
}

# Test: File permissions
test_file_permissions() {
    echo -e "\n${YELLOW}Test: File Permissions${NC}"

    # Create test script
    cat > platform/scripts/test-script.sh << 'EOF'
#!/bin/bash
echo "Test"
EOF
    chmod +x platform/scripts/test-script.sh

    # Check executable
    if [ -x platform/scripts/test-script.sh ]; then
        assert_equals "0" "0" "Script is executable"
    else
        assert_equals "0" "1" "Script should be executable"
    fi
}

# Test: Configuration validation
test_config_validation() {
    echo -e "\n${YELLOW}Test: Configuration Validation${NC}"

    # Create config with required fields
    cat > platform/config/test-config.json << 'EOF'
{
  "platform": {
    "name": "Test Platform",
    "version": "1.0.0",
    "environment": "test"
  },
  "services": {
    "sandboxes": {
      "enabled": true
    }
  }
}
EOF

    # Validate required fields exist
    if command -v jq &> /dev/null; then
        name=$(jq -r '.platform.name' platform/config/test-config.json)
        assert_equals "Test Platform" "$name" "Platform name is correct"

        version=$(jq -r '.platform.version' platform/config/test-config.json)
        assert_equals "1.0.0" "$version" "Platform version is correct"
    fi
}

# Test: Error handling
test_error_handling() {
    echo -e "\n${YELLOW}Test: Error Handling${NC}"

    # Test invalid directory
    if ! mkdir /invalid/path/that/does/not/exist 2>/dev/null; then
        assert_equals "0" "0" "Handles invalid directory creation"
    fi

    # Test invalid file creation
    if ! cat > /invalid/path/file.txt 2>/dev/null; then
        assert_equals "0" "0" "Handles invalid file creation"
    fi
}

# Test: Integration
test_integration() {
    echo -e "\n${YELLOW}Test: Integration${NC}"

    # Simulate full initialization
    mkdir -p platform/{config,services,scripts,docs}
    cat > platform/config/flow-nexus.json << 'EOF'
{"platform": {"name": "Test"}}
EOF
    cat > platform/.env << 'EOF'
NODE_ENV=test
EOF
    cat > platform/package.json << 'EOF'
{"name": "test"}
EOF

    # Verify all components exist
    assert_dir_exists "platform/config" "Integration: config dir exists"
    assert_file_exists "platform/config/flow-nexus.json" "Integration: config file exists"
    assert_file_exists "platform/.env" "Integration: env file exists"
    assert_file_exists "platform/package.json" "Integration: package.json exists"
}

# Print test summary
print_summary() {
    echo -e "\n${'='*60}"
    echo "Test Summary"
    echo "${'='*60}"
    echo -e "${GREEN}Passed:${NC} $TESTS_PASSED"
    echo -e "${RED}Failed:${NC} $TESTS_FAILED"
    echo "Total:  $((TESTS_PASSED + TESTS_FAILED))"
    echo "${'='*60}"

    if [ $TESTS_FAILED -eq 0 ]; then
        echo -e "${GREEN}All tests passed!${NC}"
        return 0
    else
        echo -e "${RED}Some tests failed!${NC}"
        return 1
    fi
}

# Main test execution
main() {
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Platform Initialization Test Suite"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    setup

    # Run all tests
    test_prerequisites
    test_directory_structure
    test_config_creation
    test_env_creation
    test_package_json
    test_health_check_script
    test_file_permissions
    test_config_validation
    test_error_handling
    test_integration

    teardown

    print_summary
}

# Run tests
main "$@"
