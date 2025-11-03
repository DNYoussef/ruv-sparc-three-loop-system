#!/bin/bash
#
# Sandbox Lifecycle Manager for functionality-audit
# Manages creation, startup, shutdown, and cleanup of E2B sandboxes
#
# Usage:
#   ./sandbox_manager.sh create --template python --timeout 600
#   ./sandbox_manager.sh start --sandbox-id abc123
#   ./sandbox_manager.sh stop --sandbox-id abc123
#   ./sandbox_manager.sh cleanup --sandbox-id abc123
#   ./sandbox_manager.sh install --sandbox-id abc123 --packages "pytest coverage"
#   ./sandbox_manager.sh monitor --sandbox-id abc123
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
TEMPLATE_DIR="$(dirname "$SCRIPT_DIR")/templates"
LOG_FILE="/tmp/sandbox-manager-$(date +%Y%m%d).log"
STATE_FILE="/tmp/sandbox-state.json"

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

# Create sandbox
create_sandbox() {
    local template="${1:-python}"
    local timeout="${2:-600}"
    local name="${3:-validator-$(date +%s)}"

    log_info "Creating sandbox with template: $template, timeout: $timeout"

    # Load sandbox config
    local config_file="$TEMPLATE_DIR/sandbox-config.json"
    if [[ ! -f "$config_file" ]]; then
        log_error "Config file not found: $config_file"
        return 1
    fi

    # Create sandbox using flow-nexus
    local result
    result=$(npx flow-nexus@latest sandbox create \
        --template "$template" \
        --timeout "$timeout" \
        --name "$name" \
        --output json 2>&1)

    if [[ $? -ne 0 ]]; then
        log_error "Failed to create sandbox: $result"
        return 1
    fi

    # Extract sandbox ID
    local sandbox_id
    sandbox_id=$(echo "$result" | jq -r '.sandbox_id')

    if [[ -z "$sandbox_id" || "$sandbox_id" == "null" ]]; then
        log_error "Failed to extract sandbox ID from response"
        return 1
    fi

    # Save state
    save_sandbox_state "$sandbox_id" "$template" "running"

    log_success "Sandbox created: $sandbox_id"
    echo "$sandbox_id"
    return 0
}

# Start existing sandbox
start_sandbox() {
    local sandbox_id="$1"

    log_info "Starting sandbox: $sandbox_id"

    # Check if sandbox exists
    local status
    status=$(get_sandbox_status "$sandbox_id")

    if [[ "$status" == "not_found" ]]; then
        log_error "Sandbox not found: $sandbox_id"
        return 1
    fi

    if [[ "$status" == "running" ]]; then
        log_warning "Sandbox already running: $sandbox_id"
        return 0
    fi

    # Start sandbox (implementation depends on sandbox type)
    # For E2B, sandboxes are typically auto-started
    update_sandbox_state "$sandbox_id" "running"

    log_success "Sandbox started: $sandbox_id"
    return 0
}

# Stop sandbox
stop_sandbox() {
    local sandbox_id="$1"

    log_info "Stopping sandbox: $sandbox_id"

    # For E2B, sandboxes stop automatically on timeout
    # For Docker, explicit stop needed
    if command -v docker &> /dev/null; then
        if docker ps -q -f name="$sandbox_id" | grep -q .; then
            docker stop "$sandbox_id" &>> "$LOG_FILE"
            log_success "Docker container stopped: $sandbox_id"
        fi
    fi

    update_sandbox_state "$sandbox_id" "stopped"

    log_success "Sandbox stopped: $sandbox_id"
    return 0
}

# Cleanup sandbox
cleanup_sandbox() {
    local sandbox_id="$1"
    local force="${2:-false}"

    log_info "Cleaning up sandbox: $sandbox_id"

    # Stop sandbox first
    stop_sandbox "$sandbox_id" || true

    # Delete sandbox
    if [[ "$force" == "true" ]]; then
        log_warning "Force cleanup enabled"
    fi

    # E2B cleanup
    if command -v npx &> /dev/null; then
        npx flow-nexus@latest sandbox delete --sandbox-id "$sandbox_id" &>> "$LOG_FILE" || true
    fi

    # Docker cleanup
    if command -v docker &> /dev/null; then
        docker rm -f "$sandbox_id" &>> "$LOG_FILE" 2>&1 || true
    fi

    # Remove from state
    remove_sandbox_state "$sandbox_id"

    log_success "Sandbox cleaned up: $sandbox_id"
    return 0
}

# Install packages in sandbox
install_packages() {
    local sandbox_id="$1"
    local packages="$2"
    local language="${3:-python}"

    log_info "Installing packages in sandbox $sandbox_id: $packages"

    local install_cmd
    case "$language" in
        python)
            install_cmd="pip install $packages"
            ;;
        node|javascript|typescript)
            install_cmd="npm install -g $packages"
            ;;
        *)
            log_error "Unsupported language: $language"
            return 1
            ;;
    esac

    # Execute install command
    if command -v npx &> /dev/null; then
        npx flow-nexus@latest sandbox execute \
            --sandbox-id "$sandbox_id" \
            --command "$install_cmd" &>> "$LOG_FILE"
    elif command -v docker &> /dev/null; then
        docker exec "$sandbox_id" bash -c "$install_cmd" &>> "$LOG_FILE"
    else
        log_error "No sandbox backend available"
        return 1
    fi

    log_success "Packages installed: $packages"
    return 0
}

# Monitor sandbox resources
monitor_sandbox() {
    local sandbox_id="$1"
    local interval="${2:-5}"
    local duration="${3:-60}"

    log_info "Monitoring sandbox $sandbox_id for ${duration}s (interval: ${interval}s)"

    local end_time=$(($(date +%s) + duration))

    while [[ $(date +%s) -lt $end_time ]]; do
        # Get resource usage
        if command -v docker &> /dev/null; then
            local stats
            stats=$(docker stats "$sandbox_id" --no-stream --format "CPU: {{.CPUPerc}} | Memory: {{.MemUsage}}" 2>/dev/null || echo "N/A")
            log_info "[$sandbox_id] $stats"
        else
            log_info "[$sandbox_id] Resource monitoring not available"
        fi

        sleep "$interval"
    done

    log_success "Monitoring complete"
    return 0
}

# Get sandbox status
get_sandbox_status() {
    local sandbox_id="$1"

    # Check state file first
    if [[ -f "$STATE_FILE" ]]; then
        local status
        status=$(jq -r ".sandboxes.\"$sandbox_id\".status // \"not_found\"" "$STATE_FILE" 2>/dev/null)
        echo "$status"
        return 0
    fi

    echo "not_found"
    return 0
}

# State management functions
save_sandbox_state() {
    local sandbox_id="$1"
    local template="$2"
    local status="$3"

    # Initialize state file if needed
    if [[ ! -f "$STATE_FILE" ]]; then
        echo '{"sandboxes":{}}' > "$STATE_FILE"
    fi

    # Update state
    local timestamp
    timestamp=$(date -Iseconds)

    jq ".sandboxes.\"$sandbox_id\" = {
        \"template\": \"$template\",
        \"status\": \"$status\",
        \"created_at\": \"$timestamp\",
        \"updated_at\": \"$timestamp\"
    }" "$STATE_FILE" > "$STATE_FILE.tmp"

    mv "$STATE_FILE.tmp" "$STATE_FILE"
}

update_sandbox_state() {
    local sandbox_id="$1"
    local status="$2"

    if [[ ! -f "$STATE_FILE" ]]; then
        log_warning "State file not found, cannot update"
        return 1
    fi

    local timestamp
    timestamp=$(date -Iseconds)

    jq ".sandboxes.\"$sandbox_id\".status = \"$status\" |
        .sandboxes.\"$sandbox_id\".updated_at = \"$timestamp\"" \
        "$STATE_FILE" > "$STATE_FILE.tmp"

    mv "$STATE_FILE.tmp" "$STATE_FILE"
}

remove_sandbox_state() {
    local sandbox_id="$1"

    if [[ ! -f "$STATE_FILE" ]]; then
        return 0
    fi

    jq "del(.sandboxes.\"$sandbox_id\")" "$STATE_FILE" > "$STATE_FILE.tmp"
    mv "$STATE_FILE.tmp" "$STATE_FILE"
}

# List all sandboxes
list_sandboxes() {
    log_info "Listing all sandboxes"

    if [[ ! -f "$STATE_FILE" ]]; then
        log_info "No sandboxes found"
        return 0
    fi

    echo ""
    echo "Sandbox ID                       | Template | Status  | Created"
    echo "--------------------------------|----------|---------|----------------------------"

    jq -r '.sandboxes | to_entries[] |
        "\(.key) | \(.value.template) | \(.value.status) | \(.value.created_at)"' \
        "$STATE_FILE" | while IFS='|' read -r id template status created; do
        printf "%-32s| %-8s | %-7s | %s\n" "$id" "$template" "$status" "$created"
    done

    echo ""
}

# Cleanup all sandboxes
cleanup_all() {
    local force="${1:-false}"

    log_warning "Cleaning up all sandboxes"

    if [[ ! -f "$STATE_FILE" ]]; then
        log_info "No sandboxes to clean up"
        return 0
    fi

    # Get all sandbox IDs
    local sandbox_ids
    sandbox_ids=$(jq -r '.sandboxes | keys[]' "$STATE_FILE")

    for sandbox_id in $sandbox_ids; do
        cleanup_sandbox "$sandbox_id" "$force"
    done

    log_success "All sandboxes cleaned up"
}

# Help message
show_help() {
    cat << EOF
Sandbox Lifecycle Manager for functionality-audit

Usage:
    $0 <command> [options]

Commands:
    create              Create new sandbox
    start               Start existing sandbox
    stop                Stop running sandbox
    cleanup             Cleanup sandbox resources
    install             Install packages in sandbox
    monitor             Monitor sandbox resources
    list                List all sandboxes
    cleanup-all         Cleanup all sandboxes
    help                Show this help message

Options:
    --sandbox-id ID     Sandbox identifier
    --template NAME     Sandbox template (python, node, etc.)
    --timeout SECONDS   Sandbox timeout
    --packages LIST     Space-separated package list
    --language LANG     Programming language
    --interval SECONDS  Monitoring interval
    --duration SECONDS  Monitoring duration
    --force             Force operation

Examples:
    # Create Python sandbox
    $0 create --template python --timeout 600

    # Install packages
    $0 install --sandbox-id abc123 --packages "pytest coverage"

    # Monitor for 2 minutes
    $0 monitor --sandbox-id abc123 --duration 120

    # Cleanup specific sandbox
    $0 cleanup --sandbox-id abc123

    # List all sandboxes
    $0 list

    # Force cleanup all
    $0 cleanup-all --force

EOF
}

# Parse command line arguments
main() {
    local command="${1:-help}"
    shift || true

    # Parse arguments
    local sandbox_id=""
    local template="python"
    local timeout="600"
    local packages=""
    local language="python"
    local interval="5"
    local duration="60"
    local force="false"

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --sandbox-id)
                sandbox_id="$2"
                shift 2
                ;;
            --template)
                template="$2"
                shift 2
                ;;
            --timeout)
                timeout="$2"
                shift 2
                ;;
            --packages)
                packages="$2"
                shift 2
                ;;
            --language)
                language="$2"
                shift 2
                ;;
            --interval)
                interval="$2"
                shift 2
                ;;
            --duration)
                duration="$2"
                shift 2
                ;;
            --force)
                force="true"
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done

    # Execute command
    case "$command" in
        create)
            create_sandbox "$template" "$timeout"
            ;;
        start)
            [[ -z "$sandbox_id" ]] && { log_error "--sandbox-id required"; exit 1; }
            start_sandbox "$sandbox_id"
            ;;
        stop)
            [[ -z "$sandbox_id" ]] && { log_error "--sandbox-id required"; exit 1; }
            stop_sandbox "$sandbox_id"
            ;;
        cleanup)
            [[ -z "$sandbox_id" ]] && { log_error "--sandbox-id required"; exit 1; }
            cleanup_sandbox "$sandbox_id" "$force"
            ;;
        install)
            [[ -z "$sandbox_id" ]] && { log_error "--sandbox-id required"; exit 1; }
            [[ -z "$packages" ]] && { log_error "--packages required"; exit 1; }
            install_packages "$sandbox_id" "$packages" "$language"
            ;;
        monitor)
            [[ -z "$sandbox_id" ]] && { log_error "--sandbox-id required"; exit 1; }
            monitor_sandbox "$sandbox_id" "$interval" "$duration"
            ;;
        list)
            list_sandboxes
            ;;
        cleanup-all)
            cleanup_all "$force"
            ;;
        help|--help|-h)
            show_help
            exit 0
            ;;
        *)
            log_error "Unknown command: $command"
            show_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
