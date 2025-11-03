#!/bin/bash
# GitHub Actions CI/CD Optimizer
# Analyzes workflows and suggests optimizations for speed, cost, and reliability

set -euo pipefail

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
WORKFLOW_DIR=".github/workflows"
CACHE_THRESHOLD=5  # Minutes - suggest caching if setup takes longer
PARALLEL_THRESHOLD=3  # Suggest parallelization if more than N jobs

# Helper functions
log_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

log_success() {
    echo -e "${GREEN}✓${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

log_error() {
    echo -e "${RED}✗${NC} $1"
}

# Check if workflow directory exists
check_workflow_dir() {
    if [[ ! -d "$WORKFLOW_DIR" ]]; then
        log_error "No .github/workflows directory found"
        exit 1
    fi
    log_success "Found workflows directory"
}

# Analyze workflow files
analyze_workflows() {
    local workflow_count=0
    local optimization_suggestions=0

    log_info "Analyzing GitHub Actions workflows..."

    for workflow in "$WORKFLOW_DIR"/*.yml "$WORKFLOW_DIR"/*.yaml; do
        [[ -e "$workflow" ]] || continue

        workflow_count=$((workflow_count + 1))
        local filename=$(basename "$workflow")

        echo ""
        log_info "Analyzing: $filename"

        # Check for caching
        if ! grep -q "actions/cache" "$workflow"; then
            log_warning "  No caching detected - consider adding dependency caching"
            optimization_suggestions=$((optimization_suggestions + 1))

            # Suggest specific caching strategies
            if grep -q "npm\|yarn\|pnpm" "$workflow"; then
                echo "    Suggestion: Add Node.js dependency caching"
                echo "    uses: actions/cache@v3"
                echo "    with:"
                echo "      path: node_modules"
                echo "      key: \${{ runner.os }}-node-\${{ hashFiles('**/package-lock.json') }}"
            fi

            if grep -q "pip\|python" "$workflow"; then
                echo "    Suggestion: Add Python dependency caching"
                echo "    uses: actions/cache@v3"
                echo "    with:"
                echo "      path: ~/.cache/pip"
                echo "      key: \${{ runner.os }}-pip-\${{ hashFiles('**/requirements.txt') }}"
            fi
        else
            log_success "  Caching implemented"
        fi

        # Check for job parallelization
        local job_count=$(grep -c "^  [a-zA-Z-]*:" "$workflow" || true)
        if [[ $job_count -gt $PARALLEL_THRESHOLD ]]; then
            if ! grep -q "needs:" "$workflow"; then
                log_warning "  Multiple jobs without dependencies - ensure parallelization"
                optimization_suggestions=$((optimization_suggestions + 1))
            else
                log_success "  Job dependencies configured"
            fi
        fi

        # Check for timeouts
        if ! grep -q "timeout-minutes" "$workflow"; then
            log_warning "  No timeouts set - add timeout-minutes to prevent stuck workflows"
            optimization_suggestions=$((optimization_suggestions + 1))
            echo "    Suggestion: Add job-level timeouts"
            echo "    timeout-minutes: 30"
        else
            log_success "  Timeouts configured"
        fi

        # Check for matrix strategy optimization
        if grep -q "strategy:" "$workflow"; then
            if ! grep -q "fail-fast" "$workflow"; then
                log_warning "  Matrix strategy without fail-fast configuration"
                echo "    Suggestion: Configure fail-fast behavior"
                echo "    strategy:"
                echo "      fail-fast: false  # Continue other jobs on failure"
            fi
        fi

        # Check for conditional job execution
        if ! grep -q "if:" "$workflow"; then
            log_warning "  No conditional execution - consider adding conditions to skip unnecessary jobs"
            optimization_suggestions=$((optimization_suggestions + 1))
            echo "    Example: if: github.event_name == 'push' && github.ref == 'refs/heads/main'"
        fi

        # Check for secrets usage
        if grep -q "secrets\." "$workflow"; then
            log_success "  Using GitHub secrets for sensitive data"
        fi

        # Check for GITHUB_TOKEN permissions
        if ! grep -q "permissions:" "$workflow"; then
            log_warning "  No explicit permissions set - add least-privilege permissions"
            optimization_suggestions=$((optimization_suggestions + 1))
            echo "    Suggestion:"
            echo "    permissions:"
            echo "      contents: read"
            echo "      pull-requests: write"
        fi

        # Check for reusable workflows
        if [[ $workflow_count -gt 2 ]]; then
            if ! grep -q "workflow_call" "$workflow"; then
                log_warning "  Multiple workflows - consider creating reusable workflows"
                echo "    See: https://docs.github.com/en/actions/using-workflows/reusing-workflows"
            fi
        fi
    done

    echo ""
    log_info "Analysis Summary:"
    echo "  Total workflows: $workflow_count"
    echo "  Optimization suggestions: $optimization_suggestions"

    if [[ $optimization_suggestions -eq 0 ]]; then
        log_success "Workflows are well-optimized!"
    else
        log_warning "Found $optimization_suggestions potential optimizations"
    fi
}

# Analyze recent workflow runs
analyze_run_performance() {
    log_info "Analyzing recent workflow runs..."

    if ! command -v gh &> /dev/null; then
        log_warning "GitHub CLI (gh) not found - skipping run analysis"
        return
    fi

    # Get recent runs
    local runs=$(gh run list --limit 10 --json databaseId,conclusion,duration)

    if [[ -z "$runs" ]]; then
        log_warning "No recent runs found"
        return
    fi

    # Calculate average duration
    local total_duration=0
    local run_count=0
    local failed_count=0

    while IFS= read -r run; do
        local duration=$(echo "$run" | jq -r '.duration')
        local conclusion=$(echo "$run" | jq -r '.conclusion')

        if [[ -n "$duration" && "$duration" != "null" ]]; then
            total_duration=$((total_duration + duration))
            run_count=$((run_count + 1))
        fi

        if [[ "$conclusion" == "failure" ]]; then
            failed_count=$((failed_count + 1))
        fi
    done < <(echo "$runs" | jq -c '.[]')

    if [[ $run_count -gt 0 ]]; then
        local avg_duration=$((total_duration / run_count))
        local avg_minutes=$((avg_duration / 60))

        echo ""
        log_info "Performance Metrics (last 10 runs):"
        echo "  Average duration: ${avg_minutes}m"
        echo "  Failure rate: ${failed_count}/10"

        if [[ $avg_minutes -gt 10 ]]; then
            log_warning "  Average runtime over 10 minutes - consider optimization"
        fi

        if [[ $failed_count -gt 2 ]]; then
            log_warning "  High failure rate - investigate common failures"
        fi
    fi
}

# Generate optimization report
generate_report() {
    local report_file="workflow-optimization-report.md"

    log_info "Generating optimization report..."

    cat > "$report_file" << EOF
# GitHub Actions Workflow Optimization Report

Generated: $(date)

## Executive Summary

This report analyzes GitHub Actions workflows and provides recommendations for:
- **Performance**: Reduce runtime through caching and parallelization
- **Cost**: Minimize billable minutes
- **Reliability**: Improve success rates and stability

## Key Recommendations

### 1. Implement Dependency Caching

**Impact**: 30-50% runtime reduction

\`\`\`yaml
- name: Cache dependencies
  uses: actions/cache@v3
  with:
    path: node_modules
    key: \${{ runner.os }}-deps-\${{ hashFiles('**/package-lock.json') }}
\`\`\`

### 2. Optimize Job Parallelization

**Impact**: 40-60% total workflow time reduction

\`\`\`yaml
jobs:
  test:
    strategy:
      matrix:
        node-version: [16, 18, 20]
      fail-fast: false
\`\`\`

### 3. Add Workflow Timeouts

**Impact**: Prevent stuck workflows consuming resources

\`\`\`yaml
jobs:
  build:
    timeout-minutes: 30
\`\`\`

### 4. Implement Conditional Execution

**Impact**: Skip unnecessary jobs

\`\`\`yaml
- name: Deploy
  if: github.ref == 'refs/heads/main' && github.event_name == 'push'
\`\`\`

### 5. Use Least-Privilege Permissions

**Impact**: Improved security posture

\`\`\`yaml
permissions:
  contents: read
  pull-requests: write
  issues: write
\`\`\`

## Swarm Integration Opportunities

Consider integrating ruv-swarm for advanced coordination:

\`\`\`yaml
- name: Swarm Coordination
  run: |
    npx ruv-swarm init --topology mesh
    npx ruv-swarm actions coordinate --parallel
\`\`\`

## Next Steps

1. Review and implement caching strategies
2. Optimize job dependencies and parallelization
3. Add timeouts to all jobs
4. Configure conditional execution
5. Set explicit permissions

---

For more information, see:
- [GitHub Actions Best Practices](https://docs.github.com/en/actions/learn-github-actions/best-practices-for-github-actions)
- [Ruv-Swarm Documentation](https://github.com/ruvnet/ruv-swarm)
EOF

    log_success "Report generated: $report_file"
}

# Main execution
main() {
    echo "╔════════════════════════════════════════════════════╗"
    echo "║   GitHub Actions CI/CD Optimizer                  ║"
    echo "║   Analyze and optimize workflow performance       ║"
    echo "╚════════════════════════════════════════════════════╝"
    echo ""

    check_workflow_dir
    analyze_workflows
    analyze_run_performance
    generate_report

    echo ""
    log_success "Optimization analysis complete!"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --workflow-dir)
            WORKFLOW_DIR="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --workflow-dir DIR    Workflow directory (default: .github/workflows)"
            echo "  --help                Show this help message"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

main
