#!/bin/bash
################################################################################
# Collective Intelligence Orchestration Script
# Coordinates hive mind operations through shell automation
################################################################################

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
HIVE_DIR="${HIVE_DIR:-.hive-mind}"
SESSION_FILE="$HIVE_DIR/session.json"
MEMORY_DB="$HIVE_DIR/collective-memory.db"
LOG_FILE="$HIVE_DIR/orchestration.log"

# Ensure hive directory exists
mkdir -p "$HIVE_DIR"

################################################################################
# Logging
################################################################################

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

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

################################################################################
# Hive Mind Initialization
################################################################################

hive_init() {
    log_info "Initializing hive mind..."

    # Initialize session
    cat > "$SESSION_FILE" <<EOF
{
  "swarmId": "hive-$(date +%s)",
  "created": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "status": "active",
  "workers": [],
  "tasks": [],
  "decisions": []
}
EOF

    # Initialize memory database (SQLite)
    if command -v sqlite3 &> /dev/null; then
        sqlite3 "$MEMORY_DB" <<EOF
CREATE TABLE IF NOT EXISTS memory (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    type TEXT NOT NULL,
    confidence REAL DEFAULT 1.0,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS associations (
    key1 TEXT NOT NULL,
    key2 TEXT NOT NULL,
    strength REAL NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (key1, key2)
);

CREATE INDEX IF NOT EXISTS idx_memory_type ON memory(type);
CREATE INDEX IF NOT EXISTS idx_memory_created ON memory(created_at);
EOF
        log_success "Collective memory database initialized"
    else
        log_warning "SQLite not found - memory persistence disabled"
    fi

    log_success "Hive mind initialized: $(jq -r '.swarmId' "$SESSION_FILE")"
}

################################################################################
# Worker Management
################################################################################

spawn_worker() {
    local worker_type="$1"
    local worker_id="worker-$(date +%s%N | cut -b1-13)"

    log_info "Spawning worker: $worker_id ($worker_type)"

    # Add worker to session
    jq --arg id "$worker_id" \
       --arg type "$worker_type" \
       --arg status "idle" \
       '.workers += [{
           id: $id,
           type: $type,
           status: $status,
           tasks_completed: 0,
           spawned_at: (now | todate)
       }]' "$SESSION_FILE" > "$SESSION_FILE.tmp" && mv "$SESSION_FILE.tmp" "$SESSION_FILE"

    log_success "Worker spawned: $worker_id"
    echo "$worker_id"
}

list_workers() {
    log_info "Active workers:"
    jq -r '.workers[] | "\(.id) (\(.type)) - \(.status)"' "$SESSION_FILE"
}

################################################################################
# Task Management
################################################################################

create_task() {
    local description="$1"
    local priority="${2:-5}"
    local task_id="task-$(date +%s%N | cut -b1-13)"

    log_info "Creating task: $task_id (priority: $priority)"

    jq --arg id "$task_id" \
       --arg desc "$description" \
       --arg prio "$priority" \
       '.tasks += [{
           id: $id,
           description: $desc,
           priority: ($prio | tonumber),
           status: "pending",
           created_at: (now | todate)
       }]' "$SESSION_FILE" > "$SESSION_FILE.tmp" && mv "$SESSION_FILE.tmp" "$SESSION_FILE"

    log_success "Task created: $task_id"
    echo "$task_id"
}

assign_task() {
    local task_id="$1"
    local worker_id="$2"

    log_info "Assigning task $task_id to $worker_id"

    # Update task assignment
    jq --arg tid "$task_id" \
       --arg wid "$worker_id" \
       '(.tasks[] | select(.id == $tid)).assigned_to = $wid |
        (.tasks[] | select(.id == $tid)).status = "in_progress" |
        (.workers[] | select(.id == $wid)).status = "busy"' \
       "$SESSION_FILE" > "$SESSION_FILE.tmp" && mv "$SESSION_FILE.tmp" "$SESSION_FILE"

    log_success "Task assigned: $task_id -> $worker_id"
}

complete_task() {
    local task_id="$1"
    local worker_id="$2"

    log_info "Completing task: $task_id"

    jq --arg tid "$task_id" \
       --arg wid "$worker_id" \
       '(.tasks[] | select(.id == $tid)).status = "completed" |
        (.tasks[] | select(.id == $tid)).completed_at = (now | todate) |
        (.workers[] | select(.id == $wid)).status = "idle" |
        (.workers[] | select(.id == $wid)).tasks_completed += 1' \
       "$SESSION_FILE" > "$SESSION_FILE.tmp" && mv "$SESSION_FILE.tmp" "$SESSION_FILE"

    log_success "Task completed: $task_id"
}

################################################################################
# Consensus Building
################################################################################

build_consensus() {
    local topic="$1"
    shift
    local options=("$@")

    log_info "Building consensus on: $topic"
    log_info "Options: ${options[*]}"

    # Simulate voting (in production, workers would vote)
    local votes=()
    local worker_count=$(jq '.workers | length' "$SESSION_FILE")

    for ((i=0; i<worker_count; i++)); do
        local vote=${options[$((RANDOM % ${#options[@]}))]}
        votes+=("$vote")
    done

    # Count votes
    declare -A vote_counts
    for vote in "${votes[@]}"; do
        ((vote_counts[$vote]++)) || vote_counts[$vote]=1
    done

    # Find winner
    local winner=""
    local max_votes=0
    for option in "${!vote_counts[@]}"; do
        if [ "${vote_counts[$option]}" -gt "$max_votes" ]; then
            max_votes="${vote_counts[$option]}"
            winner="$option"
        fi
    done

    local confidence=$(echo "scale=2; $max_votes / $worker_count" | bc)

    # Record decision
    jq --arg topic "$topic" \
       --arg decision "$winner" \
       --arg confidence "$confidence" \
       '.decisions += [{
           topic: $topic,
           decision: $decision,
           confidence: ($confidence | tonumber),
           timestamp: (now | todate)
       }]' "$SESSION_FILE" > "$SESSION_FILE.tmp" && mv "$SESSION_FILE.tmp" "$SESSION_FILE"

    log_success "Consensus reached: $winner ($confidence confidence)"
    echo "$winner"
}

################################################################################
# Memory Management
################################################################################

store_memory() {
    local key="$1"
    local value="$2"
    local type="${3:-knowledge}"
    local confidence="${4:-1.0}"

    if [ ! -f "$MEMORY_DB" ]; then
        log_error "Memory database not initialized"
        return 1
    fi

    sqlite3 "$MEMORY_DB" <<EOF
INSERT OR REPLACE INTO memory (key, value, type, confidence)
VALUES ('$key', '$value', '$type', $confidence);
EOF

    log_success "Stored in collective memory: $key"
}

retrieve_memory() {
    local key="$1"

    if [ ! -f "$MEMORY_DB" ]; then
        log_error "Memory database not initialized"
        return 1
    fi

    sqlite3 "$MEMORY_DB" "SELECT value FROM memory WHERE key = '$key';"
}

search_memory() {
    local pattern="$1"
    local type="${2:-*}"

    if [ ! -f "$MEMORY_DB" ]; then
        log_error "Memory database not initialized"
        return 1
    fi

    local where_clause="key LIKE '%$pattern%'"
    if [ "$type" != "*" ]; then
        where_clause="$where_clause AND type = '$type'"
    fi

    sqlite3 -header -column "$MEMORY_DB" \
        "SELECT key, type, confidence, created_at FROM memory WHERE $where_clause LIMIT 20;"
}

################################################################################
# Status and Monitoring
################################################################################

show_status() {
    log_info "Hive Mind Status:"
    echo ""
    echo "Swarm ID: $(jq -r '.swarmId' "$SESSION_FILE")"
    echo "Status: $(jq -r '.status' "$SESSION_FILE")"
    echo ""
    echo "Workers: $(jq '.workers | length' "$SESSION_FILE")"
    echo "  Idle: $(jq '[.workers[] | select(.status == "idle")] | length' "$SESSION_FILE")"
    echo "  Busy: $(jq '[.workers[] | select(.status == "busy")] | length' "$SESSION_FILE")"
    echo ""
    echo "Tasks:"
    echo "  Total: $(jq '.tasks | length' "$SESSION_FILE")"
    echo "  Pending: $(jq '[.tasks[] | select(.status == "pending")] | length' "$SESSION_FILE")"
    echo "  In Progress: $(jq '[.tasks[] | select(.status == "in_progress")] | length' "$SESSION_FILE")"
    echo "  Completed: $(jq '[.tasks[] | select(.status == "completed")] | length' "$SESSION_FILE")"
    echo ""
    echo "Consensus Decisions: $(jq '.decisions | length' "$SESSION_FILE")"
}

################################################################################
# Main CLI
################################################################################

usage() {
    cat <<EOF
Collective Intelligence Orchestration Script

Usage: $0 <command> [arguments]

Commands:
  init                           Initialize hive mind
  spawn <worker_type>            Spawn a worker agent
  workers                        List all workers
  task <description> [priority]  Create a new task
  assign <task_id> <worker_id>   Assign task to worker
  complete <task_id> <worker_id> Mark task as completed
  consensus <topic> <opt1> ...   Build consensus on options
  store <key> <value> [type]     Store in collective memory
  retrieve <key>                 Retrieve from memory
  search <pattern> [type]        Search memory
  status                         Show hive mind status
  help                           Show this help

Examples:
  $0 init
  $0 spawn researcher
  $0 task "Implement authentication" 9
  $0 consensus "API type" REST GraphQL gRPC
  $0 store api-pattern REST knowledge
  $0 status
EOF
}

main() {
    if [ $# -eq 0 ]; then
        usage
        exit 1
    fi

    local command="$1"
    shift

    case "$command" in
        init)
            hive_init
            ;;
        spawn)
            spawn_worker "$@"
            ;;
        workers)
            list_workers
            ;;
        task)
            create_task "$@"
            ;;
        assign)
            assign_task "$@"
            ;;
        complete)
            complete_task "$@"
            ;;
        consensus)
            build_consensus "$@"
            ;;
        store)
            store_memory "$@"
            ;;
        retrieve)
            retrieve_memory "$@"
            ;;
        search)
            search_memory "$@"
            ;;
        status)
            show_status
            ;;
        help|--help|-h)
            usage
            ;;
        *)
            log_error "Unknown command: $command"
            usage
            exit 1
            ;;
    esac
}

main "$@"
