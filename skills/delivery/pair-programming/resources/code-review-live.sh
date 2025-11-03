#!/bin/bash
# Live Code Review Script for Pair Programming
# Provides real-time code quality feedback during development

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
REVIEW_INTERVAL=30  # seconds
AUTO_FIX=false
STRICT_MODE=false
SECURITY_CHECK=true
PERFORMANCE_CHECK=true
STYLE_CHECK=true

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --interval)
      REVIEW_INTERVAL="$2"
      shift 2
      ;;
    --auto-fix)
      AUTO_FIX=true
      shift
      ;;
    --strict)
      STRICT_MODE=true
      shift
      ;;
    --no-security)
      SECURITY_CHECK=false
      shift
      ;;
    --no-performance)
      PERFORMANCE_CHECK=false
      shift
      ;;
    --no-style)
      STYLE_CHECK=false
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

print_header() {
  echo -e "${BLUE}╔════════════════════════════════════════════════╗${NC}"
  echo -e "${BLUE}║     Live Code Review - Pair Programming        ║${NC}"
  echo -e "${BLUE}╚════════════════════════════════════════════════╝${NC}"
  echo ""
}

print_section() {
  echo -e "\n${BLUE}═══ $1 ═══${NC}\n"
}

check_git_changes() {
  if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  Not a git repository. Change detection disabled.${NC}"
    return 1
  fi

  # Get modified and new files
  MODIFIED_FILES=$(git diff --name-only --diff-filter=M)
  NEW_FILES=$(git diff --name-only --cached --diff-filter=A)

  if [[ -z "$MODIFIED_FILES" && -z "$NEW_FILES" ]]; then
    return 1
  fi

  echo -e "${GREEN}📝 Files changed:${NC}"
  echo "$MODIFIED_FILES" | while read -r file; do
    [[ -n "$file" ]] && echo "  • $file (modified)"
  done
  echo "$NEW_FILES" | while read -r file; do
    [[ -n "$file" ]] && echo "  • $file (new)"
  done

  return 0
}

run_linter() {
  print_section "Style & Linting"

  local has_errors=false

  # Try ESLint for JavaScript/TypeScript
  if command -v eslint &> /dev/null; then
    echo "Running ESLint..."
    if $AUTO_FIX; then
      eslint . --fix --quiet || has_errors=true
    else
      eslint . --quiet || has_errors=true
    fi
  fi

  # Try Pylint for Python
  if command -v pylint &> /dev/null; then
    echo "Running Pylint..."
    pylint --errors-only **/*.py 2>/dev/null || has_errors=true
  fi

  # Try Rubocop for Ruby
  if command -v rubocop &> /dev/null; then
    echo "Running Rubocop..."
    if $AUTO_FIX; then
      rubocop -A --format quiet || has_errors=true
    else
      rubocop --format quiet || has_errors=true
    fi
  fi

  if $has_errors; then
    echo -e "${RED}❌ Linting issues found${NC}"
    return 1
  else
    echo -e "${GREEN}✅ No linting issues${NC}"
    return 0
  fi
}

run_security_check() {
  if ! $SECURITY_CHECK; then
    return 0
  fi

  print_section "Security Analysis"

  local has_issues=false

  # Check for common security issues
  echo "Scanning for security vulnerabilities..."

  # Check for hardcoded secrets
  if grep -r -E "(password|secret|api_key|token)\s*=\s*['\"][^'\"]+['\"]" --include="*.js" --include="*.py" --include="*.rb" . 2>/dev/null | grep -v node_modules | grep -v ".git"; then
    echo -e "${RED}⚠️  Potential hardcoded secrets found${NC}"
    has_issues=true
  fi

  # Check for SQL injection patterns
  if grep -r -E "execute\(.*\+.*\)" --include="*.js" --include="*.py" . 2>/dev/null | grep -v node_modules | grep -v ".git"; then
    echo -e "${YELLOW}⚠️  Potential SQL injection risk${NC}"
    has_issues=true
  fi

  # Try npm audit for Node.js
  if [[ -f "package.json" ]] && command -v npm &> /dev/null; then
    echo "Running npm audit..."
    npm audit --audit-level=high 2>/dev/null || has_issues=true
  fi

  # Try safety for Python
  if [[ -f "requirements.txt" ]] && command -v safety &> /dev/null; then
    echo "Running safety check..."
    safety check --json 2>/dev/null || has_issues=true
  fi

  if $has_issues; then
    echo -e "${RED}❌ Security issues found${NC}"
    return 1
  else
    echo -e "${GREEN}✅ No security issues detected${NC}"
    return 0
  fi
}

run_performance_check() {
  if ! $PERFORMANCE_CHECK; then
    return 0
  fi

  print_section "Performance Analysis"

  echo "Analyzing code complexity..."

  # Simple cyclomatic complexity check
  local complex_functions=$(grep -r -E "if.*if.*if.*if" --include="*.js" --include="*.py" . 2>/dev/null | grep -v node_modules | wc -l)

  if [[ $complex_functions -gt 5 ]]; then
    echo -e "${YELLOW}⚠️  Found $complex_functions potentially complex functions${NC}"
    echo "  Consider refactoring for better maintainability"
  else
    echo -e "${GREEN}✅ Code complexity looks good${NC}"
  fi

  # Check for large files
  find . -type f \( -name "*.js" -o -name "*.py" -o -name "*.rb" \) ! -path "*/node_modules/*" ! -path "*/.git/*" -exec wc -l {} + 2>/dev/null | sort -rn | head -5 | while read -r lines file; do
    if [[ $lines -gt 500 ]]; then
      echo -e "${YELLOW}⚠️  Large file: $file ($lines lines)${NC}"
    fi
  done

  return 0
}

run_tests() {
  print_section "Test Execution"

  # Try to detect and run tests
  if [[ -f "package.json" ]] && command -v npm &> /dev/null; then
    echo "Running npm tests..."
    npm test --silent 2>/dev/null && echo -e "${GREEN}✅ Tests passing${NC}" || echo -e "${RED}❌ Tests failing${NC}"
  elif [[ -f "pytest.ini" ]] || [[ -d "tests" ]] && command -v pytest &> /dev/null; then
    echo "Running pytest..."
    pytest -q 2>/dev/null && echo -e "${GREEN}✅ Tests passing${NC}" || echo -e "${RED}❌ Tests failing${NC}"
  elif [[ -f "Rakefile" ]] && command -v rake &> /dev/null; then
    echo "Running rake test..."
    rake test 2>/dev/null && echo -e "${GREEN}✅ Tests passing${NC}" || echo -e "${RED}❌ Tests failing${NC}"
  else
    echo -e "${YELLOW}ℹ️  No test framework detected${NC}"
  fi
}

calculate_truth_score() {
  print_section "Truth Score Calculation"

  local score=1.0
  local deductions=()

  # Run checks and calculate deductions
  if ! run_linter; then
    score=$(echo "$score - 0.02" | bc)
    deductions+=("Linting issues: -0.02")
  fi

  if ! run_security_check; then
    score=$(echo "$score - 0.05" | bc)
    deductions+=("Security issues: -0.05")
  fi

  run_performance_check

  # Check test status
  if [[ -f "package.json" ]] && command -v npm &> /dev/null; then
    if ! npm test --silent 2>/dev/null; then
      score=$(echo "$score - 0.10" | bc)
      deductions+=("Test failures: -0.10")
    fi
  fi

  # Display score
  echo ""
  if (( $(echo "$score >= 0.98" | bc -l) )); then
    echo -e "${GREEN}🌟 Truth Score: $score (Excellent)${NC}"
  elif (( $(echo "$score >= 0.95" | bc -l) )); then
    echo -e "${GREEN}✅ Truth Score: $score (Good)${NC}"
  elif (( $(echo "$score >= 0.90" | bc -l) )); then
    echo -e "${YELLOW}⚠️  Truth Score: $score (Warning)${NC}"
  else
    echo -e "${RED}❌ Truth Score: $score (Error)${NC}"
  fi

  if [[ ${#deductions[@]} -gt 0 ]]; then
    echo ""
    echo "Deductions:"
    for deduction in "${deductions[@]}"; do
      echo "  • $deduction"
    done
  fi
}

run_review_cycle() {
  clear
  print_header

  echo "Review interval: ${REVIEW_INTERVAL}s"
  echo "Auto-fix: $AUTO_FIX"
  echo "Strict mode: $STRICT_MODE"
  echo ""

  if check_git_changes; then
    echo ""
    calculate_truth_score
    run_tests
  else
    echo -e "${BLUE}ℹ️  No changes detected${NC}"
  fi

  echo ""
  echo -e "${BLUE}Next review in ${REVIEW_INTERVAL}s... (Ctrl+C to stop)${NC}"
}

# Main loop
print_header
echo "Starting live code review..."
echo "Configuration:"
echo "  • Review interval: ${REVIEW_INTERVAL}s"
echo "  • Auto-fix: $AUTO_FIX"
echo "  • Strict mode: $STRICT_MODE"
echo "  • Security check: $SECURITY_CHECK"
echo "  • Performance check: $PERFORMANCE_CHECK"
echo "  • Style check: $STYLE_CHECK"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Run initial review
run_review_cycle

# Start periodic reviews
while true; do
  sleep "$REVIEW_INTERVAL"
  run_review_cycle
done
