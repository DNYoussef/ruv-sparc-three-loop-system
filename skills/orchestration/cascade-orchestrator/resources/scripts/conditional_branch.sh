#!/bin/bash
#
# Conditional Branch Executor
# Handles conditional workflow branching with dynamic path selection
#

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${LOG_FILE:-/tmp/conditional_branch.log}"
VERBOSE="${VERBOSE:-false}"

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $*" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*" | tee -a "$LOG_FILE"
}

log_debug() {
    if [[ "$VERBOSE" == "true" ]]; then
        echo -e "${BLUE}[DEBUG]${NC} $*" | tee -a "$LOG_FILE"
    fi
}

# Condition evaluation functions
evaluate_condition() {
    local condition="$1"
    local context="$2"

    log_debug "Evaluating condition: $condition"
    log_debug "Context: $context"

    # Extract condition type and parameters
    local condition_type
    condition_type=$(echo "$condition" | jq -r '.type // "comparison"')

    case "$condition_type" in
        comparison)
            evaluate_comparison "$condition" "$context"
            ;;
        exists)
            evaluate_exists "$condition" "$context"
            ;;
        regex)
            evaluate_regex "$condition" "$context"
            ;;
        threshold)
            evaluate_threshold "$condition" "$context"
            ;;
        custom)
            evaluate_custom "$condition" "$context"
            ;;
        *)
            log_error "Unknown condition type: $condition_type"
            return 1
            ;;
    esac
}

evaluate_comparison() {
    local condition="$1"
    local context="$2"

    local left
    local operator
    local right

    left=$(echo "$condition" | jq -r '.left')
    operator=$(echo "$condition" | jq -r '.operator')
    right=$(echo "$condition" | jq -r '.right')

    # Resolve variables
    left=$(resolve_variable "$left" "$context")
    right=$(resolve_variable "$right" "$context")

    log_debug "Comparison: $left $operator $right"

    case "$operator" in
        "==")
            [[ "$left" == "$right" ]]
            ;;
        "!=")
            [[ "$left" != "$right" ]]
            ;;
        ">")
            [[ "$left" -gt "$right" ]] 2>/dev/null
            ;;
        "<")
            [[ "$left" -lt "$right" ]] 2>/dev/null
            ;;
        ">=")
            [[ "$left" -ge "$right" ]] 2>/dev/null
            ;;
        "<=")
            [[ "$left" -le "$right" ]] 2>/dev/null
            ;;
        "contains")
            [[ "$left" == *"$right"* ]]
            ;;
        *)
            log_error "Unknown operator: $operator"
            return 1
            ;;
    esac
}

evaluate_exists() {
    local condition="$1"
    local context="$2"

    local path
    path=$(echo "$condition" | jq -r '.path')

    # Check if path exists in context
    echo "$context" | jq -e "$path" >/dev/null 2>&1
}

evaluate_regex() {
    local condition="$1"
    local context="$2"

    local value
    local pattern

    value=$(echo "$condition" | jq -r '.value')
    pattern=$(echo "$condition" | jq -r '.pattern')

    value=$(resolve_variable "$value" "$context")

    log_debug "Regex match: $value =~ $pattern"

    [[ "$value" =~ $pattern ]]
}

evaluate_threshold() {
    local condition="$1"
    local context="$2"

    local metric
    local threshold
    local comparison

    metric=$(echo "$condition" | jq -r '.metric')
    threshold=$(echo "$condition" | jq -r '.threshold')
    comparison=$(echo "$condition" | jq -r '.comparison // ">"')

    metric=$(resolve_variable "$metric" "$context")

    log_debug "Threshold check: $metric $comparison $threshold"

    case "$comparison" in
        ">")
            (( $(echo "$metric > $threshold" | bc -l) ))
            ;;
        "<")
            (( $(echo "$metric < $threshold" | bc -l) ))
            ;;
        ">=")
            (( $(echo "$metric >= $threshold" | bc -l) ))
            ;;
        "<=")
            (( $(echo "$metric <= $threshold" | bc -l) ))
            ;;
        *)
            log_error "Unknown comparison: $comparison"
            return 1
            ;;
    esac
}

evaluate_custom() {
    local condition="$1"
    local context="$2"

    local script
    script=$(echo "$condition" | jq -r '.script')

    log_debug "Running custom condition script"

    # Execute custom script with context
    echo "$context" | eval "$script"
}

resolve_variable() {
    local var="$1"
    local context="$2"

    # Check if variable reference (starts with $)
    if [[ "$var" == \$* ]]; then
        local path="${var:1}"  # Remove $
        echo "$context" | jq -r "$path // \"$var\""
    else
        echo "$var"
    fi
}

# Branch selection
select_branch() {
    local branches="$1"
    local context="$2"

    log_info "Selecting branch from $(echo "$branches" | jq '. | length') options"

    local selected_branch=""
    local branch_count
    branch_count=$(echo "$branches" | jq '. | length')

    for ((i=0; i<branch_count; i++)); do
        local branch
        branch=$(echo "$branches" | jq -c ".[$i]")

        local branch_name
        local condition

        branch_name=$(echo "$branch" | jq -r '.name')
        condition=$(echo "$branch" | jq -c '.condition')

        log_debug "Evaluating branch: $branch_name"

        if evaluate_condition "$condition" "$context"; then
            log_success "Branch selected: $branch_name"
            selected_branch="$branch"
            break
        fi
    done

    if [[ -z "$selected_branch" ]]; then
        # Check for default branch
        selected_branch=$(echo "$branches" | jq -c '.[] | select(.default == true)')

        if [[ -n "$selected_branch" ]]; then
            local default_name
            default_name=$(echo "$selected_branch" | jq -r '.name')
            log_info "No conditions met, using default branch: $default_name"
        else
            log_error "No branch selected and no default branch defined"
            return 1
        fi
    fi

    echo "$selected_branch"
}

# Branch execution
execute_branch() {
    local branch="$1"
    local context="$2"

    local branch_name
    local stages

    branch_name=$(echo "$branch" | jq -r '.name')
    stages=$(echo "$branch" | jq -c '.stages')

    log_info "Executing branch: $branch_name"

    local stage_count
    stage_count=$(echo "$stages" | jq '. | length')

    local outputs="[]"

    for ((i=0; i<stage_count; i++)); do
        local stage
        stage=$(echo "$stages" | jq -c ".[$i]")

        local stage_name
        local stage_type

        stage_name=$(echo "$stage" | jq -r '.name')
        stage_type=$(echo "$stage" | jq -r '.type // "skill"')

        log_info "Executing stage: $stage_name (type: $stage_type)"

        local output
        case "$stage_type" in
            skill)
                output=$(execute_skill "$stage" "$context")
                ;;
            cascade)
                output=$(execute_cascade "$stage" "$context")
                ;;
            script)
                output=$(execute_script "$stage" "$context")
                ;;
            *)
                log_error "Unknown stage type: $stage_type"
                return 1
                ;;
        esac

        # Append output
        outputs=$(echo "$outputs" | jq --argjson out "$output" '. + [$out]')

        # Update context with stage output
        context=$(echo "$context" | jq --argjson out "$output" --arg name "$stage_name" \
            '.stage_outputs[$name] = $out')
    done

    log_success "Branch execution completed: $branch_name"

    echo "$outputs"
}

execute_skill() {
    local stage="$1"
    local context="$2"

    local skill_name
    local inputs

    skill_name=$(echo "$stage" | jq -r '.skill')
    inputs=$(echo "$stage" | jq -c '.inputs // {}')

    log_debug "Executing skill: $skill_name"

    # Mock skill execution
    echo "{\"skill\": \"$skill_name\", \"status\": \"success\", \"output\": \"Result from $skill_name\"}"
}

execute_cascade() {
    local stage="$1"
    local context="$2"

    local cascade_name
    cascade_name=$(echo "$stage" | jq -r '.cascade')

    log_debug "Executing cascade: $cascade_name"

    # Mock cascade execution
    echo "{\"cascade\": \"$cascade_name\", \"status\": \"success\"}"
}

execute_script() {
    local stage="$1"
    local context="$2"

    local script
    script=$(echo "$stage" | jq -r '.script')

    log_debug "Executing script"

    # Execute script with context
    local output
    output=$(echo "$context" | eval "$script")

    echo "{\"script_output\": $output}"
}

# Main execution
main() {
    local config_file="${1:-}"

    if [[ -z "$config_file" ]]; then
        log_error "Usage: $0 <config_file>"
        exit 1
    fi

    if [[ ! -f "$config_file" ]]; then
        log_error "Config file not found: $config_file"
        exit 1
    fi

    log_info "Starting conditional branch execution"
    log_info "Config: $config_file"

    # Load configuration
    local config
    config=$(cat "$config_file")

    # Extract components
    local branches
    local context

    branches=$(echo "$config" | jq -c '.branches')
    context=$(echo "$config" | jq -c '.context // {}')

    # Select branch
    local selected_branch
    if ! selected_branch=$(select_branch "$branches" "$context"); then
        log_error "Branch selection failed"
        exit 1
    fi

    # Execute branch
    local results
    if ! results=$(execute_branch "$selected_branch" "$context"); then
        log_error "Branch execution failed"
        exit 1
    fi

    # Output results
    echo "$results" | jq '.'

    log_success "Conditional branch execution completed"
}

# Run if executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
