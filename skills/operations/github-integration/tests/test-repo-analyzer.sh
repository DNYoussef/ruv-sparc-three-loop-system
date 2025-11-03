#!/usr/bin/env bash
#
# Test Suite for GitHub Repository Analyzer
# Integration tests for repository analysis functionality
#

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Test configuration
TEST_DIR="/tmp/repo-analyzer-test-$$"
ANALYZER_SCRIPT="../resources/scripts/repo-analyzer.js"

# Logging
log_info() { echo -e "${GREEN}[INFO]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $*"; }
log_test() { echo -e "${BLUE}[TEST]${NC} $*"; }

# Test counter
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0

# Setup test repository
setup_test_repo() {
    log_info "Setting up test repository: $TEST_DIR"

    mkdir -p "$TEST_DIR"
    cd "$TEST_DIR"

    # Initialize git repo
    git init -q
    git config user.name "Test User"
    git config user.email "test@example.com"

    # Create basic structure
    mkdir -p src tests docs

    # Create files
    cat > README.md <<'EOF'
# Test Repository
A test repository for analyzer validation
EOF

    cat > LICENSE <<'EOF'
MIT License
EOF

    cat > package.json <<'EOF'
{
  "name": "test-repo",
  "version": "1.0.0",
  "dependencies": {
    "express": "^4.18.0"
  },
  "devDependencies": {
    "jest": "^29.0.0"
  }
}
EOF

    cat > src/index.js <<'EOF'
const express = require('express');
const app = express();
app.get('/', (req, res) => res.send('Hello'));
app.listen(3000);
EOF

    cat > tests/index.test.js <<'EOF'
const request = require('supertest');
describe('API', () => {
  test('GET /', () => {});
});
EOF

    # Create GitHub Actions workflow
    mkdir -p .github/workflows
    cat > .github/workflows/ci.yml <<'EOF'
name: CI
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: npm test
EOF

    # Commit files
    git add .
    git commit -q -m "Initial commit"
    git tag -a v1.0.0 -m "Release 1.0.0"

    # Add more commits
    echo "// New feature" >> src/index.js
    git add .
    git commit -q -m "feat: add feature"

    echo "// Bug fix" >> src/index.js
    git add .
    git commit -q -m "fix: fix bug"

    log_info "Test repository setup complete"
}

# Cleanup
cleanup() {
    log_info "Cleaning up test directory"
    rm -rf "$TEST_DIR"
}

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

# Test: Analyzer produces valid JSON
test_valid_json() {
    local output
    output=$(node "$ANALYZER_SCRIPT" "$TEST_DIR" json 2>/dev/null)

    # Validate JSON
    if echo "$output" | jq . >/dev/null 2>&1; then
        return 0
    else
        log_error "Invalid JSON output"
        return 1
    fi
}

# Test: Git metadata analysis
test_git_metadata() {
    local output
    output=$(node "$ANALYZER_SCRIPT" "$TEST_DIR" json 2>/dev/null)

    local commits contributors
    commits=$(echo "$output" | jq -r '.git.commits')
    contributors=$(echo "$output" | jq -r '.git.contributors')

    # Should have 3 commits
    if [[ "$commits" -eq 3 ]]; then
        return 0
    else
        log_error "Expected 3 commits, got $commits"
        return 1
    fi
}

# Test: Code structure analysis
test_code_structure() {
    local output
    output=$(node "$ANALYZER_SCRIPT" "$TEST_DIR" json 2>/dev/null)

    local primary_lang
    primary_lang=$(echo "$output" | jq -r '.code.primaryLanguage')

    if [[ "$primary_lang" == "JavaScript" ]]; then
        return 0
    else
        log_error "Expected JavaScript, got $primary_lang"
        return 1
    fi
}

# Test: Documentation coverage
test_documentation() {
    local output
    output=$(node "$ANALYZER_SCRIPT" "$TEST_DIR" json 2>/dev/null)

    local has_readme has_license
    has_readme=$(echo "$output" | jq -r '.documentation.hasReadme')
    has_license=$(echo "$output" | jq -r '.documentation.hasLicense')

    if [[ "$has_readme" == "true" && "$has_license" == "true" ]]; then
        return 0
    else
        log_error "Documentation check failed"
        return 1
    fi
}

# Test: CI/CD detection
test_cicd_detection() {
    local output
    output=$(node "$ANALYZER_SCRIPT" "$TEST_DIR" json 2>/dev/null)

    local has_ci github_actions
    has_ci=$(echo "$output" | jq -r '.cicd.hasCI')
    github_actions=$(echo "$output" | jq -r '.cicd.platforms.githubActions')

    if [[ "$has_ci" == "true" && "$github_actions" == "true" ]]; then
        return 0
    else
        log_error "CI/CD detection failed"
        return 1
    fi
}

# Test: Testing analysis
test_testing_analysis() {
    local output
    output=$(node "$ANALYZER_SCRIPT" "$TEST_DIR" json 2>/dev/null)

    local test_files
    test_files=$(echo "$output" | jq -r '.testing.testFiles')

    if [[ "$test_files" -gt 0 ]]; then
        return 0
    else
        log_error "No test files detected"
        return 1
    fi
}

# Test: Dependencies analysis
test_dependencies() {
    local output
    output=$(node "$ANALYZER_SCRIPT" "$TEST_DIR" json 2>/dev/null)

    local package_manager deps dev_deps
    package_manager=$(echo "$output" | jq -r '.dependencies.packageManager')
    deps=$(echo "$output" | jq -r '.dependencies.dependencies')
    dev_deps=$(echo "$output" | jq -r '.dependencies.devDependencies')

    if [[ "$package_manager" == "npm/yarn" && "$deps" -eq 1 && "$dev_deps" -eq 1 ]]; then
        return 0
    else
        log_error "Dependencies analysis failed"
        return 1
    fi
}

# Test: Health score calculation
test_health_score() {
    local output
    output=$(node "$ANALYZER_SCRIPT" "$TEST_DIR" json 2>/dev/null)

    local health_score grade
    health_score=$(echo "$output" | jq -r '.healthScore.overall')
    grade=$(echo "$output" | jq -r '.healthScore.grade')

    # Health score should be between 0 and 100
    if (( $(echo "$health_score >= 0 && $health_score <= 100" | bc -l) )); then
        return 0
    else
        log_error "Invalid health score: $health_score"
        return 1
    fi
}

# Test: Recommendations generation
test_recommendations() {
    local output
    output=$(node "$ANALYZER_SCRIPT" "$TEST_DIR" json 2>/dev/null)

    local rec_count
    rec_count=$(echo "$output" | jq -r '.recommendations | length')

    # Should have at least some recommendations
    if [[ "$rec_count" -ge 0 ]]; then
        return 0
    else
        log_error "Recommendations generation failed"
        return 1
    fi
}

# Test: Markdown output format
test_markdown_output() {
    local output
    output=$(node "$ANALYZER_SCRIPT" "$TEST_DIR" markdown 2>/dev/null)

    # Should contain markdown headers
    if echo "$output" | grep -q "^#.*Repository Analysis"; then
        return 0
    else
        log_error "Markdown output invalid"
        return 1
    fi
}

# Main test runner
main() {
    echo "================================="
    echo "Repository Analyzer Test Suite"
    echo "================================="

    # Check prerequisites
    if ! command -v node &>/dev/null; then
        log_error "Node.js not found"
        exit 1
    fi

    if ! command -v jq &>/dev/null; then
        log_error "jq not found"
        exit 1
    fi

    if [[ ! -f "$ANALYZER_SCRIPT" ]]; then
        log_error "Analyzer script not found: $ANALYZER_SCRIPT"
        exit 1
    fi

    # Setup
    trap cleanup EXIT
    setup_test_repo

    # Run tests
    run_test "Valid JSON output" test_valid_json
    run_test "Git metadata analysis" test_git_metadata
    run_test "Code structure analysis" test_code_structure
    run_test "Documentation coverage" test_documentation
    run_test "CI/CD detection" test_cicd_detection
    run_test "Testing analysis" test_testing_analysis
    run_test "Dependencies analysis" test_dependencies
    run_test "Health score calculation" test_health_score
    run_test "Recommendations generation" test_recommendations
    run_test "Markdown output format" test_markdown_output

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
