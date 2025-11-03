#!/bin/bash
###############################################################################
# Optimization Engine for ReasoningBank Intelligence
#
# Orchestrates continuous improvement workflows:
# - Pattern analysis pipeline
# - Strategy evaluation
# - Performance benchmarking
# - Automated recommendations
# - Model retraining triggers
#
# Usage:
#   ./optimization-engine.sh [command] [options]
#
# Commands:
#   analyze      - Run pattern analysis
#   evaluate     - Evaluate strategy performance
#   benchmark    - Run performance benchmarks
#   optimize     - Full optimization cycle
#   report       - Generate optimization report
###############################################################################

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="${PROJECT_ROOT}/data"
MODELS_DIR="${PROJECT_ROOT}/models"
REPORTS_DIR="${PROJECT_ROOT}/reports"
LOG_FILE="${PROJECT_ROOT}/optimization.log"

# Thresholds
MIN_CONFIDENCE=0.7
MIN_SUPPORT=3
PERFORMANCE_THRESHOLD=0.8

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

###############################################################################
# Logging
###############################################################################

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $*" | tee -a "$LOG_FILE"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING:${NC} $*" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR:${NC} $*" | tee -a "$LOG_FILE"
    exit 1
}

###############################################################################
# Setup
###############################################################################

setup_directories() {
    log "Setting up directories..."
    mkdir -p "$DATA_DIR" "$MODELS_DIR" "$REPORTS_DIR"
    mkdir -p "$DATA_DIR/experiences" "$DATA_DIR/patterns" "$DATA_DIR/strategies"
}

check_dependencies() {
    log "Checking dependencies..."

    local deps=("python3" "node" "jq")
    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            error "Required dependency not found: $dep"
        fi
    done

    # Check Python packages
    if ! python3 -c "import numpy, scipy, sklearn" 2>/dev/null; then
        warn "Missing Python packages. Install with: pip install numpy scipy scikit-learn"
    fi
}

###############################################################################
# Pattern Analysis
###############################################################################

analyze_patterns() {
    log "Running pattern analysis..."

    local experiences_file="${DATA_DIR}/experiences/latest.json"
    local patterns_output="${DATA_DIR}/patterns/patterns_$(date +%Y%m%d_%H%M%S).json"

    if [[ ! -f "$experiences_file" ]]; then
        warn "No experiences file found at $experiences_file"
        return 1
    fi

    # Run pattern recognizer
    python3 "${SCRIPT_DIR}/pattern-recognizer.py" \
        --input "$experiences_file" \
        --output "$patterns_output" \
        --min-support "$MIN_SUPPORT" \
        --confidence "$MIN_CONFIDENCE" \
        --verbose

    local pattern_count=$(jq 'length' "$patterns_output")
    log "Detected $pattern_count patterns"

    # Summary
    log "Pattern summary:"
    jq -r '.[] | "\(.pattern_type): \(.description) (confidence: \(.confidence))"' "$patterns_output" | head -10

    echo "$patterns_output"
}

###############################################################################
# Strategy Evaluation
###############################################################################

evaluate_strategies() {
    log "Evaluating strategies..."

    local strategies_file="${DATA_DIR}/strategies/strategies.json"
    local eval_output="${REPORTS_DIR}/strategy_eval_$(date +%Y%m%d_%H%M%S).json"

    if [[ ! -f "$strategies_file" ]]; then
        warn "No strategies file found. Running adaptive learner..."
        node "${SCRIPT_DIR}/adaptive-learner.js" export "$strategies_file"
    fi

    # Evaluate each strategy
    local strategies=$(jq -r '.strategies[].name' "$strategies_file")
    local results="{"

    while IFS= read -r strategy; do
        log "Evaluating: $strategy"

        # Get strategy stats
        local success_rate=$(jq -r ".strategies[] | select(.name==\"$strategy\") | .successRate" "$strategies_file")
        local attempts=$(jq -r ".strategies[] | select(.name==\"$strategy\") | .attempts" "$strategies_file")

        # Performance grade
        local grade="FAIL"
        if (( $(echo "$success_rate > $PERFORMANCE_THRESHOLD" | bc -l) )); then
            grade="EXCELLENT"
        elif (( $(echo "$success_rate > 0.6" | bc -l) )); then
            grade="GOOD"
        elif (( $(echo "$success_rate > 0.4" | bc -l) )); then
            grade="FAIR"
        fi

        results+="\"$strategy\": {\"successRate\": $success_rate, \"attempts\": $attempts, \"grade\": \"$grade\"},"
    done <<< "$strategies"

    results="${results%,}}"
    echo "$results" | jq '.' > "$eval_output"

    log "Strategy evaluation complete: $eval_output"
    echo "$eval_output"
}

###############################################################################
# Performance Benchmarking
###############################################################################

benchmark_performance() {
    log "Running performance benchmarks..."

    local benchmark_output="${REPORTS_DIR}/benchmark_$(date +%Y%m%d_%H%M%S).json"
    local start_time=$(date +%s)

    # Pattern recognition benchmark
    log "Benchmarking pattern recognition..."
    local pattern_start=$(date +%s%3N)
    analyze_patterns > /dev/null 2>&1 || true
    local pattern_end=$(date +%s%3N)
    local pattern_time=$((pattern_end - pattern_start))

    # Strategy selection benchmark
    log "Benchmarking strategy selection..."
    local strategy_start=$(date +%s%3N)
    node "${SCRIPT_DIR}/adaptive-learner.js" select 100 > /dev/null 2>&1 || true
    local strategy_end=$(date +%s%3N)
    local strategy_time=$((strategy_end - strategy_start))

    # Memory usage
    local memory_usage=$(ps aux | grep -E "(python3|node)" | awk '{sum+=$6} END {print sum/1024}')

    # Create benchmark report
    cat > "$benchmark_output" <<EOF
{
  "timestamp": "$(date -Iseconds)",
  "benchmarks": {
    "pattern_recognition_ms": $pattern_time,
    "strategy_selection_ms": $strategy_time,
    "memory_usage_mb": $memory_usage
  },
  "performance_grade": "$(grade_performance $pattern_time $strategy_time)"
}
EOF

    log "Benchmark complete: $benchmark_output"
    jq '.' "$benchmark_output"
}

grade_performance() {
    local pattern_time=$1
    local strategy_time=$2

    if (( pattern_time < 100 && strategy_time < 50 )); then
        echo "EXCELLENT"
    elif (( pattern_time < 500 && strategy_time < 200 )); then
        echo "GOOD"
    else
        echo "NEEDS_IMPROVEMENT"
    fi
}

###############################################################################
# Full Optimization Cycle
###############################################################################

optimize() {
    log "Starting full optimization cycle..."

    setup_directories
    check_dependencies

    # Step 1: Analyze patterns
    log "Step 1/4: Pattern Analysis"
    local patterns_file=$(analyze_patterns)

    # Step 2: Evaluate strategies
    log "Step 2/4: Strategy Evaluation"
    local eval_file=$(evaluate_strategies)

    # Step 3: Benchmark performance
    log "Step 3/4: Performance Benchmarking"
    benchmark_performance

    # Step 4: Generate recommendations
    log "Step 4/4: Generating Recommendations"
    generate_recommendations "$patterns_file" "$eval_file"

    log "Optimization cycle complete!"
}

###############################################################################
# Recommendations
###############################################################################

generate_recommendations() {
    local patterns_file=$1
    local eval_file=$2

    local rec_output="${REPORTS_DIR}/recommendations_$(date +%Y%m%d_%H%M%S).md"

    log "Generating optimization recommendations..."

    cat > "$rec_output" <<EOF
# ReasoningBank Intelligence Optimization Recommendations
Generated: $(date)

## Executive Summary
- Patterns Detected: $(jq 'length' "$patterns_file")
- Strategies Evaluated: $(jq '.strategies | length' "${DATA_DIR}/strategies/strategies.json" 2>/dev/null || echo 0)
- Overall Performance: $(jq -r '.performance_grade' "${REPORTS_DIR}"/benchmark_*.json | tail -1)

## Pattern Insights
$(jq -r '.[] | select(.confidence > 0.8) | "- **\(.pattern_type)**: \(.description) (confidence: \(.confidence))"' "$patterns_file" | head -5)

## Strategy Recommendations
$(jq -r 'to_entries | map(select(.value.grade == "EXCELLENT") | "- ✅ **\(.key)**: \(.value.successRate) success rate (\(.value.attempts) attempts)") | .[]' "$eval_file")

$(jq -r 'to_entries | map(select(.value.grade == "FAIL") | "- ❌ **\(.key)**: \(.value.successRate) success rate - Consider deprecating") | .[]' "$eval_file")

## Action Items
1. **High Priority**: Investigate low-performing strategies
2. **Medium Priority**: Expand successful pattern usage
3. **Low Priority**: Increase exploration for under-tested strategies

## Next Steps
- Re-run optimization after 100 more experiences
- Monitor strategy performance trends
- A/B test new pattern-based optimizations
EOF

    log "Recommendations saved: $rec_output"
    cat "$rec_output"
}

###############################################################################
# Reporting
###############################################################################

generate_report() {
    log "Generating comprehensive optimization report..."

    local report_output="${REPORTS_DIR}/optimization_report_$(date +%Y%m%d_%H%M%S).md"

    cat > "$report_output" <<EOF
# ReasoningBank Intelligence Optimization Report
Generated: $(date)

## System Status
- Data Directory: $DATA_DIR
- Models Directory: $MODELS_DIR
- Reports Directory: $REPORTS_DIR

## Recent Activity
$(tail -20 "$LOG_FILE")

## Performance Metrics
$(jq '.' "${REPORTS_DIR}"/benchmark_*.json 2>/dev/null | tail -30 || echo "No benchmarks available")

## Pattern Summary
$(jq -r '.[] | "- \(.pattern_type): \(.support) occurrences (confidence: \(.confidence))"' "${DATA_DIR}"/patterns/patterns_*.json 2>/dev/null | tail -10 || echo "No patterns available")

## Recommendations
See latest recommendations: $(ls -t "${REPORTS_DIR}"/recommendations_*.md | head -1)
EOF

    log "Report generated: $report_output"
    cat "$report_output"
}

###############################################################################
# Main
###############################################################################

main() {
    local command=${1:-help}

    case "$command" in
        analyze)
            setup_directories
            check_dependencies
            analyze_patterns
            ;;
        evaluate)
            setup_directories
            check_dependencies
            evaluate_strategies
            ;;
        benchmark)
            setup_directories
            check_dependencies
            benchmark_performance
            ;;
        optimize)
            optimize
            ;;
        report)
            generate_report
            ;;
        help|*)
            cat <<EOF
ReasoningBank Intelligence Optimization Engine

Usage: $0 [command]

Commands:
  analyze      - Run pattern analysis on experiences
  evaluate     - Evaluate strategy performance
  benchmark    - Run performance benchmarks
  optimize     - Run full optimization cycle
  report       - Generate comprehensive report
  help         - Show this help message

Examples:
  $0 analyze
  $0 optimize
  $0 report

Environment Variables:
  MIN_CONFIDENCE         - Minimum pattern confidence (default: 0.7)
  MIN_SUPPORT           - Minimum pattern support (default: 3)
  PERFORMANCE_THRESHOLD - Performance threshold (default: 0.8)
EOF
            ;;
    esac
}

main "$@"
