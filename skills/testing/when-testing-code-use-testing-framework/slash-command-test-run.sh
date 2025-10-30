#!/bin/bash
# Slash Command: /test-run
# Description: Run test suite with various options (coverage, watch, specific types)
# Usage: /test-run [--type unit|integration|e2e|all] [--coverage] [--watch] [--parallel] [--file <path>]

set -e

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
TEST_TYPE="all"
COVERAGE=false
WATCH=false
PARALLEL=true
SPECIFIC_FILE=""
VERBOSE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --type)
      TEST_TYPE="$2"
      shift 2
      ;;
    --coverage)
      COVERAGE=true
      shift
      ;;
    --watch)
      WATCH=true
      shift
      ;;
    --no-parallel)
      PARALLEL=false
      shift
      ;;
    --file)
      SPECIFIC_FILE="$2"
      shift 2
      ;;
    --verbose)
      VERBOSE=true
      shift
      ;;
    --help)
      echo "Usage: /test-run [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --type <type>        Test type: unit, integration, e2e, or all (default: all)"
      echo "  --coverage           Generate coverage report"
      echo "  --watch              Run tests in watch mode"
      echo "  --no-parallel        Disable parallel execution"
      echo "  --file <path>        Run specific test file"
      echo "  --verbose            Show detailed output"
      echo "  --help               Show this help message"
      echo ""
      echo "Examples:"
      echo "  /test-run --type unit --coverage"
      echo "  /test-run --watch --file tests/unit/calculator.test.js"
      echo "  /test-run --type e2e --no-parallel"
      exit 0
      ;;
    *)
      echo -e "${RED}Error: Unknown option: $1${NC}"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Function to print section headers
print_header() {
  echo -e "\n${BLUE}═══════════════════════════════════════════════════════${NC}"
  echo -e "${BLUE}  $1${NC}"
  echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}\n"
}

# Function to execute hook
execute_hook() {
  local hook_type=$1
  local description=$2

  if command -v npx &> /dev/null && npx claude-flow@alpha --version &> /dev/null 2>&1; then
    npx claude-flow@alpha hooks "$hook_type" --description "$description" 2>/dev/null || true
  fi
}

# Start session
print_header "Testing Framework - Test Execution"
echo -e "${YELLOW}Configuration:${NC}"
echo "  Test Type: $TEST_TYPE"
echo "  Coverage: $COVERAGE"
echo "  Watch Mode: $WATCH"
echo "  Parallel: $PARALLEL"
[ -n "$SPECIFIC_FILE" ] && echo "  Specific File: $SPECIFIC_FILE"
echo ""

# Pre-task hook
execute_hook "pre-task" "Running test suite: $TEST_TYPE"

# Check if test scripts exist in package.json
if [ ! -f "package.json" ]; then
  echo -e "${RED}Error: package.json not found${NC}"
  exit 1
fi

# Verify test frameworks are installed
if ! grep -q "vitest\|jest" package.json; then
  echo -e "${RED}Error: No testing framework found in package.json${NC}"
  echo "Please install Vitest or Jest first:"
  echo "  npm install -D vitest @vitest/ui @vitest/coverage-v8"
  exit 1
fi

# Build test command based on options
build_test_command() {
  local base_cmd=""
  local test_args=""

  # Determine base command
  if grep -q "vitest" package.json; then
    base_cmd="npx vitest"
  elif grep -q "jest" package.json; then
    base_cmd="npx jest"
  else
    echo -e "${RED}Error: No supported testing framework found${NC}"
    exit 1
  fi

  # Add test type
  case $TEST_TYPE in
    unit)
      test_args="tests/unit"
      ;;
    integration)
      test_args="tests/integration"
      ;;
    e2e)
      if [ -f "playwright.config.js" ] || [ -f "playwright.config.ts" ]; then
        base_cmd="npx playwright test"
        test_args=""
      elif [ -f "cypress.config.js" ] || [ -f "cypress.config.ts" ]; then
        base_cmd="npx cypress run"
        test_args=""
      else
        test_args="tests/e2e"
      fi
      ;;
    all)
      test_args=""
      ;;
    *)
      echo -e "${RED}Error: Invalid test type: $TEST_TYPE${NC}"
      exit 1
      ;;
  esac

  # Add specific file if provided
  if [ -n "$SPECIFIC_FILE" ]; then
    test_args="$SPECIFIC_FILE"
  fi

  # Add watch mode
  if [ "$WATCH" = true ]; then
    if [[ $base_cmd == *"vitest"* ]]; then
      test_args="$test_args --watch"
    elif [[ $base_cmd == *"jest"* ]]; then
      test_args="$test_args --watch"
    fi
  else
    if [[ $base_cmd == *"vitest"* ]]; then
      test_args="$test_args run"
    fi
  fi

  # Add coverage
  if [ "$COVERAGE" = true ]; then
    if [[ $base_cmd == *"vitest"* ]]; then
      test_args="$test_args --coverage"
    elif [[ $base_cmd == *"jest"* ]]; then
      test_args="$test_args --coverage"
    fi
  fi

  # Add parallel/sequential
  if [ "$PARALLEL" = false ]; then
    if [[ $base_cmd == *"vitest"* ]]; then
      test_args="$test_args --no-threads"
    elif [[ $base_cmd == *"jest"* ]]; then
      test_args="$test_args --runInBand"
    fi
  fi

  # Add verbose
  if [ "$VERBOSE" = true ]; then
    if [[ $base_cmd == *"vitest"* ]]; then
      test_args="$test_args --reporter=verbose"
    elif [[ $base_cmd == *"jest"* ]]; then
      test_args="$test_args --verbose"
    fi
  fi

  echo "$base_cmd $test_args"
}

# Build and display command
TEST_COMMAND=$(build_test_command)
print_header "Executing Test Command"
echo -e "${YELLOW}Command:${NC} $TEST_COMMAND"
echo ""

# Execute tests
START_TIME=$(date +%s)

if eval "$TEST_COMMAND"; then
  END_TIME=$(date +%s)
  DURATION=$((END_TIME - START_TIME))

  echo ""
  print_header "Test Results"
  echo -e "${GREEN}✓ Tests passed successfully${NC}"
  echo -e "${YELLOW}Duration:${NC} ${DURATION}s"

  # Display coverage report if generated
  if [ "$COVERAGE" = true ]; then
    echo ""
    print_header "Coverage Report"

    if [ -f "coverage/coverage-summary.json" ]; then
      if command -v jq &> /dev/null; then
        echo "Coverage Summary:"
        jq -r '.total | "  Lines: \(.lines.pct)%\n  Statements: \(.statements.pct)%\n  Functions: \(.functions.pct)%\n  Branches: \(.branches.pct)%"' coverage/coverage-summary.json
      else
        cat coverage/coverage-summary.json
      fi

      echo ""
      echo -e "${BLUE}Full report available at:${NC} coverage/index.html"

      # Store coverage in memory
      execute_hook "post-task" "Test execution completed with coverage"

      if command -v npx &> /dev/null && npx claude-flow@alpha --version &> /dev/null 2>&1; then
        npx claude-flow@alpha memory store \
          --key "testing/coverage/$(date +%Y%m%d-%H%M%S)" \
          --value "$(cat coverage/coverage-summary.json)" \
          --tags "testing,coverage,metrics" 2>/dev/null || true
      fi
    else
      echo -e "${YELLOW}Coverage report not found${NC}"
    fi
  fi

  # Store test results in memory
  if command -v npx &> /dev/null && npx claude-flow@alpha --version &> /dev/null 2>&1; then
    npx claude-flow@alpha memory store \
      --key "testing/results/$(date +%Y%m%d-%H%M%S)" \
      --value "{\"type\":\"$TEST_TYPE\",\"status\":\"passed\",\"duration\":$DURATION}" \
      --tags "testing,results,success" 2>/dev/null || true
  fi

  echo ""
  echo -e "${GREEN}✓ Test execution completed successfully${NC}"

else
  END_TIME=$(date +%s)
  DURATION=$((END_TIME - START_TIME))

  echo ""
  print_header "Test Results"
  echo -e "${RED}✗ Tests failed${NC}"
  echo -e "${YELLOW}Duration:${NC} ${DURATION}s"

  # Store failure in memory
  execute_hook "post-task" "Test execution failed"

  if command -v npx &> /dev/null && npx claude-flow@alpha --version &> /dev/null 2>&1; then
    npx claude-flow@alpha memory store \
      --key "testing/results/$(date +%Y%m%d-%H%M%S)" \
      --value "{\"type\":\"$TEST_TYPE\",\"status\":\"failed\",\"duration\":$DURATION}" \
      --tags "testing,results,failure" 2>/dev/null || true
  fi

  echo ""
  echo -e "${RED}✗ Test execution failed${NC}"
  echo "Please review the test output above for details"

  exit 1
fi

# Open coverage report in browser if requested
if [ "$COVERAGE" = true ] && [ -f "coverage/index.html" ]; then
  echo ""
  read -p "Open coverage report in browser? (y/N) " -n 1 -r
  echo
  if [[ $REPLY =~ ^[Yy]$ ]]; then
    if command -v xdg-open &> /dev/null; then
      xdg-open coverage/index.html
    elif command -v open &> /dev/null; then
      open coverage/index.html
    else
      echo "Please open coverage/index.html manually"
    fi
  fi
fi
