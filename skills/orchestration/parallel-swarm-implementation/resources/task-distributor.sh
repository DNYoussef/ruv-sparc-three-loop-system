#!/bin/bash

###############################################################################
# Task Distributor - Workload Balancing for Parallel Swarm Implementation
# Part of Loop 2: Parallel Swarm Implementation (Enhanced Tier)
#
# This script distributes tasks across agents based on:
# - Agent availability and workload
# - Task priority and dependencies
# - Resource constraints
# - Optimal parallelism
###############################################################################

set -euo pipefail

# Configuration
MATRIX_FILE="${1:-.claude/.artifacts/agent-skill-assignments.json}"
MAX_PARALLEL_AGENTS="${2:-11}"
MEMORY_NAMESPACE="swarm/coordination"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

###############################################################################
# Helper Functions
###############################################################################

log_info() {
  echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
  echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
  echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
  echo -e "${RED}❌ $1${NC}"
}

###############################################################################
# Agent Workload Tracking
###############################################################################

# Initialize agent workload map
declare -A AGENT_WORKLOAD
declare -A AGENT_TASKS

init_agent_tracking() {
  log_info "Initializing agent workload tracking..."

  # Get unique agents from matrix
  AGENTS=$(jq -r '.tasks[].assignedAgent' "$MATRIX_FILE" | sort -u)

  for agent in $AGENTS; do
    AGENT_WORKLOAD[$agent]=0
    AGENT_TASKS[$agent]=""
  done

  log_success "Tracking ${#AGENT_WORKLOAD[@]} unique agents"
}

###############################################################################
# Task Distribution Logic
###############################################################################

distribute_task() {
  local task_id="$1"
  local agent="$2"
  local priority="$3"

  # Check agent workload
  current_load=${AGENT_WORKLOAD[$agent]:-0}

  # Max tasks per agent based on priority
  local max_tasks
  case "$priority" in
    critical) max_tasks=1 ;;  # Critical tasks get dedicated agent
    high)     max_tasks=2 ;;  # High priority can share
    medium)   max_tasks=3 ;;  # Medium can share more
    low)      max_tasks=4 ;;  # Low priority can share heavily
    *)        max_tasks=3 ;;  # Default
  esac

  # Check if agent can take more tasks
  if [ "$current_load" -ge "$max_tasks" ]; then
    log_warning "Agent $agent at capacity ($current_load/$max_tasks) - task $task_id may be delayed"
    return 1
  fi

  # Assign task to agent
  AGENT_WORKLOAD[$agent]=$((current_load + 1))
  AGENT_TASKS[$agent]="${AGENT_TASKS[$agent]} $task_id"

  log_info "Assigned $task_id to $agent (workload: ${AGENT_WORKLOAD[$agent]}/$max_tasks)"

  return 0
}

###############################################################################
# Parallel Group Distribution
###############################################################################

distribute_parallel_group() {
  local group_num="$1"

  echo ""
  echo "═══════════════════════════════════════════════════════════"
  log_info "Distributing Parallel Group $group_num"
  echo "═══════════════════════════════════════════════════════════"

  # Get tasks in this group
  local tasks=$(jq -r ".parallelGroups[$((group_num - 1))].tasks[]" "$MATRIX_FILE")
  local task_count=$(echo "$tasks" | wc -w)

  log_info "Group $group_num has $task_count tasks"

  # Check if we exceed max parallel agents
  if [ "$task_count" -gt "$MAX_PARALLEL_AGENTS" ]; then
    log_warning "Group $group_num has $task_count tasks but max parallel agents is $MAX_PARALLEL_AGENTS"
    log_info "Tasks will be executed in batches"
  fi

  # Distribute tasks by priority
  local critical_tasks=()
  local high_tasks=()
  local medium_tasks=()
  local low_tasks=()

  for task_id in $tasks; do
    local priority=$(jq -r ".tasks[] | select(.taskId==\"$task_id\") | .priority" "$MATRIX_FILE")
    local agent=$(jq -r ".tasks[] | select(.taskId==\"$task_id\") | .assignedAgent" "$MATRIX_FILE")

    case "$priority" in
      critical) critical_tasks+=("$task_id:$agent:$priority") ;;
      high)     high_tasks+=("$task_id:$agent:$priority") ;;
      medium)   medium_tasks+=("$task_id:$agent:$priority") ;;
      low)      low_tasks+=("$task_id:$agent:$priority") ;;
    esac
  done

  # Distribute in priority order
  echo ""
  log_info "Distributing critical tasks..."
  for task_info in "${critical_tasks[@]}"; do
    IFS=':' read -r task_id agent priority <<< "$task_info"
    distribute_task "$task_id" "$agent" "$priority"
  done

  echo ""
  log_info "Distributing high priority tasks..."
  for task_info in "${high_tasks[@]}"; do
    IFS=':' read -r task_id agent priority <<< "$task_info"
    distribute_task "$task_id" "$agent" "$priority"
  done

  echo ""
  log_info "Distributing medium priority tasks..."
  for task_info in "${medium_tasks[@]}"; do
    IFS=':' read -r task_id agent priority <<< "$task_info"
    distribute_task "$task_id" "$agent" "$priority"
  done

  echo ""
  log_info "Distributing low priority tasks..."
  for task_info in "${low_tasks[@]}"; do
    IFS=':' read -r task_id agent priority <<< "$task_info"
    distribute_task "$task_id" "$agent" "$priority"
  done

  echo ""
  log_success "Group $group_num distribution complete"
}

###############################################################################
# Agent Workload Report
###############################################################################

generate_workload_report() {
  echo ""
  echo "═══════════════════════════════════════════════════════════"
  log_info "Agent Workload Report"
  echo "═══════════════════════════════════════════════════════════"
  echo ""

  printf "%-30s %10s %s\n" "Agent" "Workload" "Tasks"
  printf "%-30s %10s %s\n" "-----" "--------" "-----"

  for agent in "${!AGENT_WORKLOAD[@]}"; do
    local workload=${AGENT_WORKLOAD[$agent]}
    local tasks=${AGENT_TASKS[$agent]:-"none"}
    printf "%-30s %10d %s\n" "$agent" "$workload" "$tasks"
  done

  echo ""

  # Calculate statistics
  local total_agents=${#AGENT_WORKLOAD[@]}
  local active_agents=0
  local total_workload=0
  local max_workload=0

  for agent in "${!AGENT_WORKLOAD[@]}"; do
    local workload=${AGENT_WORKLOAD[$agent]}
    if [ "$workload" -gt 0 ]; then
      active_agents=$((active_agents + 1))
    fi
    total_workload=$((total_workload + workload))
    if [ "$workload" -gt "$max_workload" ]; then
      max_workload=$workload
    fi
  done

  local avg_workload=0
  if [ "$active_agents" -gt 0 ]; then
    avg_workload=$((total_workload / active_agents))
  fi

  echo "Statistics:"
  echo "  Total Agents: $total_agents"
  echo "  Active Agents: $active_agents"
  echo "  Total Tasks: $total_workload"
  echo "  Average Workload: $avg_workload tasks/agent"
  echo "  Max Workload: $max_workload tasks"

  # Workload balance score (0-100, higher is better)
  local balance_score=0
  if [ "$max_workload" -gt 0 ]; then
    balance_score=$(( (avg_workload * 100) / max_workload ))
  fi

  echo "  Balance Score: $balance_score/100"

  if [ "$balance_score" -lt 60 ]; then
    log_warning "Workload is unbalanced - consider redistributing tasks"
  elif [ "$balance_score" -lt 80 ]; then
    log_info "Workload balance is acceptable"
  else
    log_success "Workload is well-balanced"
  fi
}

###############################################################################
# Dependency Validation
###############################################################################

validate_dependencies() {
  log_info "Validating task dependencies..."

  local errors=0

  # Check each task's dependencies are in earlier groups
  local total_groups=$(jq '.parallelGroups | length' "$MATRIX_FILE")

  for group_num in $(seq 1 "$total_groups"); do
    local tasks=$(jq -r ".parallelGroups[$((group_num - 1))].tasks[]" "$MATRIX_FILE")

    for task_id in $tasks; do
      local deps=$(jq -r ".tasks[] | select(.taskId==\"$task_id\") | .dependencies[]" "$MATRIX_FILE" 2>/dev/null || true)

      if [ -n "$deps" ]; then
        for dep in $deps; do
          # Check if dependency is in an earlier group
          local dep_group=0
          for g in $(seq 1 "$total_groups"); do
            if jq -e ".parallelGroups[$((g - 1))].tasks | index(\"$dep\")" "$MATRIX_FILE" >/dev/null 2>&1; then
              dep_group=$g
              break
            fi
          done

          if [ "$dep_group" -ge "$group_num" ]; then
            log_error "Task $task_id in group $group_num depends on $dep in group $dep_group (circular or same-group dependency)"
            errors=$((errors + 1))
          fi
        done
      fi
    done
  done

  if [ "$errors" -eq 0 ]; then
    log_success "All dependencies validated - no circular dependencies"
  else
    log_error "Found $errors dependency errors"
    return 1
  fi

  return 0
}

###############################################################################
# Main Execution
###############################################################################

main() {
  echo "═══════════════════════════════════════════════════════════"
  echo "Task Distributor - Parallel Swarm Implementation"
  echo "═══════════════════════════════════════════════════════════"
  echo ""

  # Check matrix file exists
  if [ ! -f "$MATRIX_FILE" ]; then
    log_error "Matrix file not found: $MATRIX_FILE"
    exit 1
  fi

  log_info "Loading matrix: $MATRIX_FILE"
  log_info "Max parallel agents: $MAX_PARALLEL_AGENTS"

  # Initialize tracking
  init_agent_tracking

  # Validate dependencies first
  echo ""
  validate_dependencies || {
    log_error "Dependency validation failed - fix dependencies before distribution"
    exit 1
  }

  # Distribute each parallel group
  local total_groups=$(jq '.parallelGroups | length' "$MATRIX_FILE")

  for group_num in $(seq 1 "$total_groups"); do
    distribute_parallel_group "$group_num"
  done

  # Generate final report
  generate_workload_report

  # Save distribution to memory (simulated)
  echo ""
  log_info "Storing distribution in memory namespace: $MEMORY_NAMESPACE"
  # In production:
  # npx claude-flow@alpha memory store "task_distribution" "$(generate_distribution_json)" --namespace "$MEMORY_NAMESPACE"

  echo ""
  log_success "Task distribution complete - ready for parallel execution"
}

# Run main if executed directly
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
  main "$@"
fi
