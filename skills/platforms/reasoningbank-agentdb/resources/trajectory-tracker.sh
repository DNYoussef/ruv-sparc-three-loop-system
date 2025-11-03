#!/bin/bash

###############################################################################
# Trajectory Tracker for ReasoningBank with AgentDB
# Tracks agent execution trajectories and stores them for learning.
###############################################################################

set -euo pipefail

# Configuration
AGENTDB_PATH="${AGENTDB_PATH:-.agentdb/reasoningbank.db}"
TRAJECTORY_LOG="${TRAJECTORY_LOG:-.agentdb/trajectories.jsonl}"
DOMAIN="${DOMAIN:-general}"
AGENT_NAME="${AGENT_NAME:-unknown}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

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

# Initialize trajectory tracking
init_trajectory() {
    local task_id="$1"
    local task_description="$2"

    TRAJECTORY_ID="traj_${task_id}_$(date +%s)"
    TRAJECTORY_START=$(date +%s%3N)

    # Create trajectory metadata
    cat > "/tmp/${TRAJECTORY_ID}.json" <<EOF
{
  "id": "${TRAJECTORY_ID}",
  "task": "${task_description}",
  "domain": "${DOMAIN}",
  "agent": "${AGENT_NAME}",
  "start_time": ${TRAJECTORY_START},
  "steps": [],
  "outcome": "in_progress",
  "metrics": {}
}
EOF

    log_info "Initialized trajectory: ${TRAJECTORY_ID}"
    echo "${TRAJECTORY_ID}"
}

# Record a trajectory step
record_step() {
    local trajectory_id="$1"
    local action="$2"
    local result="$3"
    local metadata="${4:-{}}"

    local step_timestamp=$(date +%s%3N)
    local step_file="/tmp/${trajectory_id}.json"

    if [[ ! -f "${step_file}" ]]; then
        log_error "Trajectory not found: ${trajectory_id}"
        return 1
    fi

    # Create step record
    local step_json=$(cat <<EOF
{
  "action": "${action}",
  "result": "${result}",
  "timestamp": ${step_timestamp},
  "metadata": ${metadata}
}
EOF
)

    # Append step to trajectory using jq
    if command -v jq &> /dev/null; then
        local updated=$(jq ".steps += [${step_json}]" "${step_file}")
        echo "${updated}" > "${step_file}"
    else
        log_warn "jq not installed, skipping step recording"
    fi

    log_info "Recorded step: ${action} -> ${result}"
}

# Complete trajectory with outcome
complete_trajectory() {
    local trajectory_id="$1"
    local outcome="$2"  # success, failure, partial
    local metrics_json="${3:-{}}"

    local end_time=$(date +%s%3N)
    local step_file="/tmp/${trajectory_id}.json"

    if [[ ! -f "${step_file}" ]]; then
        log_error "Trajectory not found: ${trajectory_id}"
        return 1
    fi

    # Update trajectory with outcome
    if command -v jq &> /dev/null; then
        local updated=$(jq \
            --arg outcome "${outcome}" \
            --argjson metrics "${metrics_json}" \
            --argjson end_time "${end_time}" \
            '.outcome = $outcome | .metrics = $metrics | .end_time = $end_time' \
            "${step_file}")
        echo "${updated}" > "${step_file}"
    fi

    # Append to trajectory log
    cat "${step_file}" >> "${TRAJECTORY_LOG}"
    echo "" >> "${TRAJECTORY_LOG}"  # Newline for JSONL format

    # Store in AgentDB using npx
    if command -v npx &> /dev/null; then
        log_info "Storing trajectory in AgentDB..."
        npx agentdb@latest import "${step_file}" --db "${AGENTDB_PATH}" 2>/dev/null || \
            log_warn "Failed to store in AgentDB (may need manual import)"
    fi

    # Cleanup
    rm -f "${step_file}"

    log_info "Completed trajectory: ${trajectory_id} with outcome: ${outcome}"
}

# Analyze trajectory patterns
analyze_patterns() {
    local domain="${1:-${DOMAIN}}"
    local min_success_rate="${2:-0.7}"

    log_info "Analyzing patterns for domain: ${domain}"

    if [[ ! -f "${TRAJECTORY_LOG}" ]]; then
        log_warn "No trajectory log found"
        return 0
    fi

    # Count successes and failures
    local total_count=$(grep -c "\"outcome\"" "${TRAJECTORY_LOG}" || echo "0")
    local success_count=$(grep -c "\"outcome\": \"success\"" "${TRAJECTORY_LOG}" || echo "0")

    if [[ ${total_count} -eq 0 ]]; then
        log_info "No trajectories recorded yet"
        return 0
    fi

    local success_rate=$(awk "BEGIN {printf \"%.2f\", ${success_count}/${total_count}}")

    echo ""
    echo "Trajectory Analysis for ${domain}:"
    echo "  Total Trajectories: ${total_count}"
    echo "  Successful: ${success_count}"
    echo "  Success Rate: ${success_rate}"
    echo ""

    # Identify common successful patterns
    if command -v jq &> /dev/null && [[ ${success_count} -gt 0 ]]; then
        log_info "Top successful action sequences:"
        grep "\"outcome\": \"success\"" "${TRAJECTORY_LOG}" | \
            jq -r '.steps[].action' | \
            sort | uniq -c | sort -rn | head -5
    fi
}

# Export trajectories for analysis
export_trajectories() {
    local output_file="${1:-trajectories_export_$(date +%Y%m%d_%H%M%S).json}"
    local domain="${2:-}"

    log_info "Exporting trajectories to: ${output_file}"

    if [[ ! -f "${TRAJECTORY_LOG}" ]]; then
        log_warn "No trajectory log found"
        return 0
    fi

    # Filter by domain if specified
    if [[ -n "${domain}" ]] && command -v jq &> /dev/null; then
        jq "select(.domain == \"${domain}\")" "${TRAJECTORY_LOG}" > "${output_file}"
    else
        cp "${TRAJECTORY_LOG}" "${output_file}"
    fi

    log_info "Exported trajectories to: ${output_file}"
}

# Clean old trajectories
cleanup_old_trajectories() {
    local days_old="${1:-30}"

    log_info "Cleaning trajectories older than ${days_old} days..."

    if [[ ! -f "${TRAJECTORY_LOG}" ]]; then
        log_info "No trajectory log to clean"
        return 0
    fi

    local cutoff_time=$(($(date +%s) - (${days_old} * 86400)))

    if command -v jq &> /dev/null; then
        # Filter trajectories newer than cutoff
        local temp_file=$(mktemp)
        jq "select(.start_time > ${cutoff_time}000)" "${TRAJECTORY_LOG}" > "${temp_file}"

        local old_count=$(grep -c "\"id\"" "${TRAJECTORY_LOG}" || echo "0")
        local new_count=$(grep -c "\"id\"" "${temp_file}" || echo "0")
        local removed=$((old_count - new_count))

        mv "${temp_file}" "${TRAJECTORY_LOG}"

        log_info "Removed ${removed} old trajectories"
    else
        log_warn "jq not installed, cannot clean trajectories"
    fi
}

# Main command dispatcher
main() {
    local command="${1:-help}"
    shift || true

    case "${command}" in
        init)
            init_trajectory "$@"
            ;;
        step)
            record_step "$@"
            ;;
        complete)
            complete_trajectory "$@"
            ;;
        analyze)
            analyze_patterns "$@"
            ;;
        export)
            export_trajectories "$@"
            ;;
        cleanup)
            cleanup_old_trajectories "$@"
            ;;
        help|*)
            cat <<EOF
Trajectory Tracker for ReasoningBank

Usage:
  trajectory-tracker.sh <command> [arguments]

Commands:
  init <task_id> <description>       Initialize new trajectory
  step <traj_id> <action> <result>   Record trajectory step
  complete <traj_id> <outcome>       Complete trajectory (success|failure|partial)
  analyze [domain] [min_success]     Analyze trajectory patterns
  export [output_file] [domain]      Export trajectories
  cleanup [days_old]                 Clean old trajectories (default: 30 days)
  help                               Show this help

Environment Variables:
  AGENTDB_PATH    Path to AgentDB database (default: .agentdb/reasoningbank.db)
  TRAJECTORY_LOG  Path to trajectory log (default: .agentdb/trajectories.jsonl)
  DOMAIN          Default domain for trajectories (default: general)
  AGENT_NAME      Name of agent executing tasks (default: unknown)

Examples:
  # Initialize trajectory
  TRAJ_ID=\$(./trajectory-tracker.sh init "task-123" "Optimize API endpoint")

  # Record steps
  ./trajectory-tracker.sh step "\${TRAJ_ID}" "analyze-bottleneck" "found N+1 queries"
  ./trajectory-tracker.sh step "\${TRAJ_ID}" "add-eager-loading" "reduced queries by 80%"

  # Complete trajectory
  ./trajectory-tracker.sh complete "\${TRAJ_ID}" "success" '{"latency_reduction": 0.85}'

  # Analyze patterns
  ./trajectory-tracker.sh analyze "api-optimization" 0.7
EOF
            ;;
    esac
}

# Run main if executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
