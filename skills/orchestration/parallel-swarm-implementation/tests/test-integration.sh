#!/bin/bash

###############################################################################
# Integration Test Suite for Parallel Swarm Implementation
# Part of Loop 2: Enhanced Tier
#
# Tests end-to-end workflow from Loop 1 package to Loop 3 delivery
###############################################################################

set -euo pipefail

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# Test configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESOURCES_DIR="$SCRIPT_DIR/../resources"
FIXTURES_DIR="$SCRIPT_DIR/fixtures"
TEST_OUTPUT_DIR="$SCRIPT_DIR/test-output"

# Test counters
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0

###############################################################################
# Helper Functions
###############################################################################

log_test() {
  echo -e "${BLUE}🧪 Test: $1${NC}"
  TESTS_RUN=$((TESTS_RUN + 1))
}

log_pass() {
  echo -e "${GREEN}   ✅ $1${NC}"
  TESTS_PASSED=$((TESTS_PASSED + 1))
}

log_fail() {
  echo -e "${RED}   ❌ $1${NC}"
  TESTS_FAILED=$((TESTS_FAILED + 1))
}

setup_test_env() {
  echo "Setting up test environment..."
  mkdir -p "$FIXTURES_DIR" "$TEST_OUTPUT_DIR"
}

cleanup_test_env() {
  echo "Cleaning up test environment..."
  rm -rf "$TEST_OUTPUT_DIR"
}

###############################################################################
# Test: Create Sample Loop 1 Package
###############################################################################

create_sample_loop1_package() {
  local output_file="$1"

  cat > "$output_file" <<'EOF'
{
  "project": "E-Commerce Authentication System",
  "planning": {
    "enhanced_plan": {
      "foundation": [
        "Design PostgreSQL schema for users, sessions, and refresh tokens",
        "Set up database migrations with Flyway"
      ],
      "implementation": [
        "Implement JWT authentication endpoints (login, refresh, logout)",
        "Create authentication middleware for Express",
        "Build React login and signup components",
        "Implement password reset flow with email"
      ],
      "quality": [
        "Create TDD unit tests with 90% coverage",
        "Run theater detection scan",
        "Validate in sandbox environment",
        "Run integration tests for auth flows"
      ],
      "documentation": [
        "Generate API documentation for auth endpoints",
        "Create authentication usage guide"
      ]
    }
  },
  "research": {
    "recommendations": "Use jsonwebtoken (10k+ stars) for JWT, bcrypt for password hashing, nodemailer for email",
    "confidence_score": 85
  },
  "risk_analysis": {
    "mitigations": "Defense-in-depth token validation: 1) Signature, 2) Expiry, 3) User exists, 4) Not revoked",
    "final_failure_confidence": 2.5
  }
}
EOF
}

###############################################################################
# Test 1: End-to-End Workflow
###############################################################################

test_end_to_end_workflow() {
  log_test "End-to-End Workflow (Loop 1 → Matrix → Execution → Loop 3)"

  # Step 1: Create Loop 1 package
  local loop1_package="$FIXTURES_DIR/loop1-planning-package.json"
  create_sample_loop1_package "$loop1_package"

  if [ -f "$loop1_package" ]; then
    log_pass "Loop 1 package created"
  else
    log_fail "Failed to create Loop 1 package"
    return 1
  fi

  # Step 2: Generate agent+skill matrix
  local matrix_file="$TEST_OUTPUT_DIR/agent-skill-assignments.json"

  echo "   Running swarm-coordinator.py..."
  python3 "$RESOURCES_DIR/swarm-coordinator.py" "$loop1_package" "$matrix_file"

  if [ -f "$matrix_file" ]; then
    log_pass "Agent+skill matrix generated"

    # Validate matrix structure
    if jq -e '.project' "$matrix_file" >/dev/null 2>&1 && \
       jq -e '.tasks' "$matrix_file" >/dev/null 2>&1 && \
       jq -e '.parallelGroups' "$matrix_file" >/dev/null 2>&1; then
      log_pass "Matrix structure valid"
    else
      log_fail "Matrix structure invalid"
      return 1
    fi
  else
    log_fail "Failed to generate matrix"
    return 1
  fi

  # Step 3: Run task distribution
  echo "   Running task-distributor.sh..."
  bash "$RESOURCES_DIR/task-distributor.sh" "$matrix_file" 11 2>&1 | head -20

  log_pass "Task distribution completed"

  # Step 4: Simulate parallel execution
  local execution_summary="$TEST_OUTPUT_DIR/execution-summary.json"

  echo "   Running parallel-executor.js..."
  node "$RESOURCES_DIR/parallel-executor.js" "$matrix_file" 2>&1 | tail -10

  if [ -f "$execution_summary" ]; then
    log_pass "Parallel execution completed"

    # Validate execution summary
    if jq -e '.execution.totalExecuted' "$execution_summary" >/dev/null 2>&1; then
      local total=$(jq -r '.execution.totalExecuted' "$execution_summary")
      log_pass "Executed $total tasks"
    fi
  else
    log_fail "Failed to generate execution summary"
    return 1
  fi

  # Step 5: Generate delivery package
  local delivery_package="$TEST_OUTPUT_DIR/loop2-delivery-package.json"

  echo "   Running result-aggregator.py..."
  python3 "$RESOURCES_DIR/result-aggregator.py" "$matrix_file" "$execution_summary" "$delivery_package"

  if [ -f "$delivery_package" ]; then
    log_pass "Loop 2 delivery package generated"

    # Validate delivery package
    if jq -e '.metadata.loop == 2' "$delivery_package" >/dev/null 2>&1 && \
       jq -e '.integrationPoints.feedsTo == "cicd-intelligent-recovery"' "$delivery_package" >/dev/null 2>&1; then
      log_pass "Delivery package ready for Loop 3"
    else
      log_fail "Delivery package invalid"
      return 1
    fi
  else
    log_fail "Failed to generate delivery package"
    return 1
  fi

  log_pass "End-to-end workflow completed successfully"
}

###############################################################################
# Test 2: Matrix Validation
###############################################################################

test_matrix_validation() {
  log_test "Matrix Validation (MECE, Dependencies, Skills)"

  local loop1_package="$FIXTURES_DIR/loop1-test.json"
  create_sample_loop1_package "$loop1_package"

  local matrix_file="$TEST_OUTPUT_DIR/matrix-validation.json"
  python3 "$RESOURCES_DIR/swarm-coordinator.py" "$loop1_package" "$matrix_file" >/dev/null 2>&1

  # Test 1: All tasks have IDs
  local tasks_without_ids=$(jq '[.tasks[] | select(.taskId == null)] | length' "$matrix_file")
  if [ "$tasks_without_ids" -eq 0 ]; then
    log_pass "All tasks have IDs"
  else
    log_fail "$tasks_without_ids tasks without IDs"
  fi

  # Test 2: All tasks have agents
  local tasks_without_agents=$(jq '[.tasks[] | select(.assignedAgent == null)] | length' "$matrix_file")
  if [ "$tasks_without_agents" -eq 0 ]; then
    log_pass "All tasks have assigned agents"
  else
    log_fail "$tasks_without_agents tasks without agents"
  fi

  # Test 3: Skills are valid (not null or valid skill name)
  local invalid_skills=$(jq '[.tasks[] | select(.useSkill != null and (.useSkill | length) < 3)] | length' "$matrix_file")
  if [ "$invalid_skills" -eq 0 ]; then
    log_pass "All skills valid"
  else
    log_fail "$invalid_skills tasks with invalid skills"
  fi

  # Test 4: Dependencies reference valid tasks
  local task_ids=$(jq -r '.tasks[].taskId' "$matrix_file")
  local invalid_deps=0

  for task_id in $(jq -r '.tasks[].taskId' "$matrix_file"); do
    local deps=$(jq -r ".tasks[] | select(.taskId==\"$task_id\") | .dependencies[]" "$matrix_file" 2>/dev/null || true)
    for dep in $deps; do
      if ! echo "$task_ids" | grep -q "^$dep$"; then
        invalid_deps=$((invalid_deps + 1))
      fi
    done
  done

  if [ "$invalid_deps" -eq 0 ]; then
    log_pass "All dependencies valid"
  else
    log_fail "$invalid_deps invalid dependencies"
  fi
}

###############################################################################
# Test 3: Parallel Group Optimization
###############################################################################

test_parallel_group_optimization() {
  log_test "Parallel Group Optimization"

  local loop1_package="$FIXTURES_DIR/loop1-groups.json"
  create_sample_loop1_package "$loop1_package"

  local matrix_file="$TEST_OUTPUT_DIR/groups-test.json"
  python3 "$RESOURCES_DIR/swarm-coordinator.py" "$loop1_package" "$matrix_file" >/dev/null 2>&1

  # Test 1: Multiple groups exist
  local num_groups=$(jq '.parallelGroups | length' "$matrix_file")
  if [ "$num_groups" -gt 1 ]; then
    log_pass "$num_groups parallel groups generated"
  else
    log_fail "Only $num_groups group(s) - expected multiple"
  fi

  # Test 2: Group 1 has no dependencies (foundation)
  local group1_tasks=$(jq -r '.parallelGroups[0].tasks[]' "$matrix_file")
  local group1_has_deps=false

  for task_id in $group1_tasks; do
    local deps=$(jq -r ".tasks[] | select(.taskId==\"$task_id\") | .dependencies | length" "$matrix_file")
    if [ "$deps" -gt 0 ]; then
      group1_has_deps=true
      break
    fi
  done

  if [ "$group1_has_deps" = false ]; then
    log_pass "Group 1 tasks have no dependencies (foundation)"
  else
    log_fail "Group 1 tasks have dependencies"
  fi

  # Test 3: Later groups have dependencies
  if [ "$num_groups" -gt 1 ]; then
    local group2_tasks=$(jq -r '.parallelGroups[1].tasks[]' "$matrix_file")
    local group2_has_deps=false

    for task_id in $group2_tasks; do
      local deps=$(jq -r ".tasks[] | select(.taskId==\"$task_id\") | .dependencies | length" "$matrix_file")
      if [ "$deps" -gt 0 ]; then
        group2_has_deps=true
        break
      fi
    done

    if [ "$group2_has_deps" = true ]; then
      log_pass "Group 2 tasks have dependencies"
    else
      log_fail "Group 2 tasks have no dependencies"
    fi
  fi
}

###############################################################################
# Main Test Runner
###############################################################################

main() {
  echo "═══════════════════════════════════════════════════════════"
  echo "Integration Test Suite - Parallel Swarm Implementation"
  echo "═══════════════════════════════════════════════════════════"
  echo ""

  setup_test_env

  # Run tests
  test_end_to_end_workflow
  test_matrix_validation
  test_parallel_group_optimization

  # Summary
  echo ""
  echo "═══════════════════════════════════════════════════════════"
  echo "Test Summary"
  echo "═══════════════════════════════════════════════════════════"
  echo "Tests Run:    $TESTS_RUN"
  echo "Tests Passed: $TESTS_PASSED"
  echo "Tests Failed: $TESTS_FAILED"

  if [ "$TESTS_FAILED" -eq 0 ]; then
    echo -e "\n${GREEN}✅ All tests passed!${NC}\n"
    cleanup_test_env
    exit 0
  else
    echo -e "\n${RED}❌ Some tests failed${NC}\n"
    exit 1
  fi
}

# Run main if executed directly
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
  main "$@"
fi
