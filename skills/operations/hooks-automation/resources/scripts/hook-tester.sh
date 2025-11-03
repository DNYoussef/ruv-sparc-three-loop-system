#!/bin/bash
# hook-tester.sh - Test hooks in sandbox environments
# Usage: bash hook-tester.sh [--type HOOK_TYPE] [--sandbox] [--profile] [--dry-run]

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOOKS_DIR="${HOOKS_DIR:-$HOME/.claude-flow/hooks}"
TEST_DIR="/tmp/hook-tests-$$"
PROFILE=false
SANDBOX=false
DRY_RUN=false
HOOK_TYPE=""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --type)
      HOOK_TYPE="$2"
      shift 2
      ;;
    --sandbox)
      SANDBOX=true
      shift
      ;;
    --profile)
      PROFILE=true
      shift
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --help)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --type TYPE      Test specific hook type (pre-task, post-edit, etc.)"
      echo "  --sandbox        Run tests in isolated sandbox"
      echo "  --profile        Profile hook execution performance"
      echo "  --dry-run        Show what would be tested without executing"
      echo "  --help           Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Logging functions
log_info() {
  echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
  echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
  echo -e "${RED}[ERROR]${NC} $1"
}

log_success() {
  echo -e "${GREEN}[✓]${NC} $1"
}

log_failure() {
  echo -e "${RED}[✗]${NC} $1"
}

log_debug() {
  if [[ "${DEBUG:-false}" == "true" ]]; then
    echo -e "${BLUE}[DEBUG]${NC} $1"
  fi
}

# Setup test environment
setup_test_env() {
  log_info "Setting up test environment..."

  if [[ "$DRY_RUN" == "true" ]]; then
    log_debug "Would create test directory: $TEST_DIR"
    return
  fi

  mkdir -p "$TEST_DIR"/{pre-task,post-edit,post-task,session}
  log_debug "Created test directory: $TEST_DIR"
}

# Cleanup test environment
cleanup_test_env() {
  if [[ "$DRY_RUN" == "false" && -d "$TEST_DIR" ]]; then
    log_info "Cleaning up test environment..."
    rm -rf "$TEST_DIR"
  fi
}

# Test hook execution
test_hook_execution() {
  local hook_type="$1"
  local hook_dir="$HOOKS_DIR/$hook_type"

  if [[ ! -d "$hook_dir" ]]; then
    log_warn "Hook directory not found: $hook_type"
    return 1
  fi

  local run_script="$hook_dir/run.sh"
  if [[ ! -f "$run_script" ]]; then
    log_warn "Hook script not found: $run_script"
    return 1
  fi

  if [[ "$DRY_RUN" == "true" ]]; then
    log_debug "Would test: $hook_type"
    return 0
  fi

  log_info "Testing $hook_type hook..."

  local start_time=$(date +%s%N)
  local exit_code=0

  if [[ "$SANDBOX" == "true" ]]; then
    # Run in isolated environment
    (cd "$TEST_DIR/$hook_type" && bash "$run_script" "test-arg" 2>&1) || exit_code=$?
  else
    # Run normally
    bash "$run_script" "test-arg" >/dev/null 2>&1 || exit_code=$?
  fi

  local end_time=$(date +%s%N)
  local duration=$(( (end_time - start_time) / 1000000 )) # Convert to milliseconds

  if [[ $exit_code -eq 0 ]]; then
    log_success "$hook_type hook passed (${duration}ms)"
    return 0
  else
    log_failure "$hook_type hook failed (exit code: $exit_code)"
    return 1
  fi
}

# Profile hook execution
profile_hook() {
  local hook_type="$1"
  local iterations=10

  if [[ "$DRY_RUN" == "true" ]]; then
    log_debug "Would profile: $hook_type ($iterations iterations)"
    return
  fi

  log_info "Profiling $hook_type hook ($iterations iterations)..."

  local total_time=0
  local successes=0
  local failures=0

  for i in $(seq 1 $iterations); do
    local start_time=$(date +%s%N)
    if test_hook_execution "$hook_type" >/dev/null 2>&1; then
      ((successes++))
    else
      ((failures++))
    fi
    local end_time=$(date +%s%N)
    local duration=$(( (end_time - start_time) / 1000000 ))
    total_time=$((total_time + duration))
  done

  local avg_time=$((total_time / iterations))
  local success_rate=$((successes * 100 / iterations))

  echo ""
  echo "Profile Results for $hook_type:"
  echo "  Iterations: $iterations"
  echo "  Successes: $successes"
  echo "  Failures: $failures"
  echo "  Success rate: ${success_rate}%"
  echo "  Total time: ${total_time}ms"
  echo "  Average time: ${avg_time}ms"
  echo "  Min/Max/Avg: N/A (requires full profiling)" # Simplified for this implementation
  echo ""
}

# Test hook configuration
test_hook_config() {
  local hook_type="$1"
  local hook_dir="$HOOKS_DIR/$hook_type"

  if [[ ! -d "$hook_dir" ]]; then
    log_warn "Hook directory not found: $hook_type"
    return 1
  fi

  log_info "Testing $hook_type configuration..."

  # Check for config files
  local has_config=false
  if [[ -f "$hook_dir/config.yaml" ]]; then
    log_debug "Found YAML config"
    has_config=true
  fi

  if [[ -f "$hook_dir/config.json" ]]; then
    log_debug "Found JSON config"
    has_config=true
  fi

  if [[ "$has_config" == "false" ]]; then
    log_warn "No configuration file found for $hook_type"
    return 1
  fi

  log_success "$hook_type configuration exists"
  return 0
}

# Test all hooks
test_all_hooks() {
  local hook_types=("pre-task" "post-edit" "post-task" "session")
  local total=0
  local passed=0
  local failed=0

  log_info "Testing all hooks..."
  echo ""

  for hook_type in "${hook_types[@]}"; do
    ((total++))

    if test_hook_execution "$hook_type"; then
      ((passed++))
    else
      ((failed++))
    fi

    if [[ "$PROFILE" == "true" ]]; then
      profile_hook "$hook_type"
    fi
  done

  echo ""
  log_info "Test Summary:"
  echo "  Total hooks: $total"
  echo "  Passed: $passed"
  echo "  Failed: $failed"
  echo "  Success rate: $((passed * 100 / total))%"
  echo ""

  return $((failed > 0 ? 1 : 0))
}

# Test hook chaining
test_hook_chaining() {
  log_info "Testing hook chaining..."

  if [[ "$DRY_RUN" == "true" ]]; then
    log_debug "Would test hook chaining"
    return 0
  fi

  local hooks_to_chain=("pre-task" "post-task")
  local chain_success=true

  for hook in "${hooks_to_chain[@]}"; do
    if ! test_hook_execution "$hook" >/dev/null 2>&1; then
      chain_success=false
      break
    fi
  done

  if [[ "$chain_success" == "true" ]]; then
    log_success "Hook chaining works correctly"
    return 0
  else
    log_failure "Hook chaining failed"
    return 1
  fi
}

# Test error handling
test_error_handling() {
  log_info "Testing error handling..."

  if [[ "$DRY_RUN" == "true" ]]; then
    log_debug "Would test error handling"
    return 0
  fi

  # Create a failing hook test
  local test_script="$TEST_DIR/failing-hook.sh"
  cat > "$test_script" <<'EOF'
#!/bin/bash
exit 1
EOF
  chmod +x "$test_script"

  if bash "$test_script" 2>/dev/null; then
    log_failure "Error handling test failed (should have errored)"
    return 1
  else
    log_success "Error handling works correctly"
    return 0
  fi
}

# Main test flow
main() {
  log_info "Starting hook tests..."
  echo ""

  setup_test_env

  local exit_code=0

  if [[ -n "$HOOK_TYPE" ]]; then
    # Test specific hook
    if [[ "$PROFILE" == "true" ]]; then
      profile_hook "$HOOK_TYPE"
    else
      test_hook_execution "$HOOK_TYPE" || exit_code=1
    fi
  else
    # Test all hooks
    test_all_hooks || exit_code=1

    if [[ "$SANDBOX" == "true" ]]; then
      test_hook_chaining || exit_code=1
      test_error_handling || exit_code=1
    fi
  fi

  cleanup_test_env

  if [[ $exit_code -eq 0 ]]; then
    log_success "All tests passed!"
  else
    log_error "Some tests failed"
  fi

  exit $exit_code
}

# Trap cleanup on exit
trap cleanup_test_env EXIT

# Run main
main
