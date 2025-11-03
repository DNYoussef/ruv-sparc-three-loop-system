#!/bin/bash
# Context Manager - Shell-based memory operations for AgentDB
# Provides CLI utilities for memory management and context operations

set -euo pipefail

# Configuration
DB_PATH="${AGENTDB_PATH:-.agentdb/memory.db}"
SESSION_ID="${AGENTDB_SESSION:-$(uuidgen 2>/dev/null || echo "session-$(date +%s)")}"
LOG_FILE="${AGENTDB_LOG:-/tmp/agentdb-context.log}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}ERROR: $*${NC}" >&2
    log "ERROR: $*"
}

info() {
    echo -e "${GREEN}INFO: $*${NC}"
    log "INFO: $*"
}

warn() {
    echo -e "${YELLOW}WARN: $*${NC}"
    log "WARN: $*"
}

# Initialize AgentDB
init_db() {
    local db_path="${1:-$DB_PATH}"
    local dimension="${2:-384}"

    info "Initializing AgentDB at $db_path (dimension: $dimension)"

    if npx agentdb@latest init "$db_path" --dimension "$dimension"; then
        info "AgentDB initialized successfully"
        return 0
    else
        error "Failed to initialize AgentDB"
        return 1
    fi
}

# Store memory with automatic layer assignment
store_memory() {
    local content="$1"
    local priority="${2:-0.5}"
    local metadata="${3:-{}}"

    info "Storing memory (priority: $priority)"

    # Determine layer based on priority
    local layer="short_term"
    if (( $(echo "$priority >= 0.8" | bc -l) )); then
        layer="long_term"
    elif (( $(echo "$priority >= 0.5" | bc -l) )); then
        layer="mid_term"
    fi

    # Calculate expiration
    local now=$(date +%s)
    local retention_hours
    case "$layer" in
        "short_term") retention_hours=24 ;;
        "mid_term") retention_hours=168 ;;  # 7 days
        "long_term") retention_hours=720 ;;  # 30 days
    esac
    local expires_at=$((now + retention_hours * 3600))

    # Insert into database using sqlite3
    sqlite3 "$DB_PATH" <<EOF
INSERT INTO memory_layers
(layer, session_id, content, priority, created_at, expires_at, metadata)
VALUES ('$layer', '$SESSION_ID', '$content', $priority, $now, $expires_at, '$metadata');
EOF

    info "Memory stored in layer: $layer"
}

# Retrieve memories from session
retrieve_memory() {
    local session_id="${1:-$SESSION_ID}"
    local layer="${2:-}"
    local limit="${3:-20}"

    info "Retrieving memories (session: $session_id, layer: ${layer:-all})"

    local query="SELECT layer, content, priority, datetime(created_at, 'unixepoch') as created
                 FROM memory_layers
                 WHERE session_id = '$session_id'
                 AND expires_at > $(date +%s)"

    if [ -n "$layer" ]; then
        query="$query AND layer = '$layer'"
    fi

    query="$query ORDER BY priority DESC, created_at DESC LIMIT $limit"

    sqlite3 -header -column "$DB_PATH" "$query"
}

# Get memory statistics
get_stats() {
    local session_id="${1:-$SESSION_ID}"

    info "Memory statistics for session: $session_id"

    sqlite3 -header -column "$DB_PATH" <<EOF
SELECT
    layer,
    COUNT(*) as count,
    ROUND(AVG(priority), 3) as avg_priority,
    SUM(access_count) as total_accesses
FROM memory_layers
WHERE session_id = '$session_id'
    AND expires_at > $(date +%s)
GROUP BY layer;
EOF
}

# Consolidate memories
consolidate_memory() {
    local session_id="${1:-$SESSION_ID}"
    local now=$(date +%s)

    info "Consolidating memories for session: $session_id"

    # Promote short-term -> mid-term
    local promoted_mid=$(sqlite3 "$DB_PATH" <<EOF
UPDATE memory_layers
SET layer = 'mid_term',
    expires_at = $now + (168 * 3600),
    priority = priority * 1.1
WHERE layer = 'short_term'
    AND session_id = '$session_id'
    AND access_count >= 5
    AND priority >= 0.5
    AND expires_at > $now;
SELECT changes();
EOF
    )

    # Promote mid-term -> long-term
    local promoted_long=$(sqlite3 "$DB_PATH" <<EOF
UPDATE memory_layers
SET layer = 'long_term',
    expires_at = $now + (720 * 3600),
    priority = priority * 1.2
WHERE layer = 'mid_term'
    AND session_id = '$session_id'
    AND access_count >= 10
    AND priority >= 0.7
    AND expires_at > $now;
SELECT changes();
EOF
    )

    # Remove expired
    local expired=$(sqlite3 "$DB_PATH" <<EOF
DELETE FROM memory_layers WHERE expires_at <= $now;
SELECT changes();
EOF
    )

    info "Consolidation complete:"
    info "  Promoted to mid-term: $promoted_mid"
    info "  Promoted to long-term: $promoted_long"
    info "  Expired entries: $expired"
}

# Export memories to JSON
export_memories() {
    local session_id="${1:-$SESSION_ID}"
    local output_file="${2:-memories-$session_id.json}"

    info "Exporting memories to $output_file"

    sqlite3 "$DB_PATH" <<EOF | python3 -m json.tool > "$output_file"
.mode json
SELECT * FROM memory_layers
WHERE session_id = '$session_id'
    AND expires_at > $(date +%s)
ORDER BY priority DESC, created_at DESC;
EOF

    info "Export complete: $output_file"
}

# Import memories from JSON
import_memories() {
    local input_file="$1"

    if [ ! -f "$input_file" ]; then
        error "File not found: $input_file"
        return 1
    fi

    info "Importing memories from $input_file"

    # Use Python to parse JSON and insert into database
    python3 <<EOF
import json
import sqlite3

with open('$input_file', 'r') as f:
    memories = json.load(f)

conn = sqlite3.connect('$DB_PATH')
cursor = conn.cursor()

for mem in memories:
    cursor.execute('''
        INSERT OR REPLACE INTO memory_layers
        (layer, session_id, content, priority, created_at, expires_at, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        mem['layer'],
        mem['session_id'],
        mem['content'],
        mem['priority'],
        mem['created_at'],
        mem['expires_at'],
        mem.get('metadata')
    ))

conn.commit()
conn.close()
print(f"Imported {len(memories)} memories")
EOF

    info "Import complete"
}

# Clean expired memories
clean_expired() {
    local now=$(date +%s)

    info "Cleaning expired memories"

    local deleted=$(sqlite3 "$DB_PATH" <<EOF
DELETE FROM memory_layers WHERE expires_at <= $now;
SELECT changes();
EOF
    )

    info "Cleaned $deleted expired entries"
}

# Search memories by content
search_memories() {
    local query="$1"
    local session_id="${2:-$SESSION_ID}"

    info "Searching memories for: $query"

    sqlite3 -header -column "$DB_PATH" <<EOF
SELECT layer, content, priority, datetime(created_at, 'unixepoch') as created
FROM memory_layers
WHERE session_id = '$session_id'
    AND content LIKE '%$query%'
    AND expires_at > $(date +%s)
ORDER BY priority DESC, created_at DESC
LIMIT 20;
EOF
}

# Show help
show_help() {
    cat <<EOF
AgentDB Context Manager - Memory operations for persistent agent context

USAGE:
    $0 <command> [options]

COMMANDS:
    init [db_path] [dimension]     Initialize AgentDB
    store <content> [priority]     Store memory (priority: 0.0-1.0)
    retrieve [session] [layer]     Retrieve memories
    stats [session]                Show memory statistics
    consolidate [session]          Consolidate and promote memories
    export [session] [file]        Export memories to JSON
    import <file>                  Import memories from JSON
    clean                          Clean expired memories
    search <query> [session]       Search memories by content
    help                           Show this help message

ENVIRONMENT:
    AGENTDB_PATH      Database path (default: .agentdb/memory.db)
    AGENTDB_SESSION   Session ID (default: auto-generated)
    AGENTDB_LOG       Log file path (default: /tmp/agentdb-context.log)

EXAMPLES:
    # Initialize database
    $0 init .agentdb/memory.db 384

    # Store high-priority memory
    $0 store "User prefers Python" 0.9

    # Retrieve all memories
    $0 retrieve

    # Get statistics
    $0 stats

    # Consolidate memories
    $0 consolidate

    # Export memories
    $0 export my-session memories.json

    # Search memories
    $0 search "Python"
EOF
}

# Main command dispatcher
main() {
    local command="${1:-help}"
    shift || true

    case "$command" in
        init)
            init_db "$@"
            ;;
        store)
            if [ $# -lt 1 ]; then
                error "Usage: $0 store <content> [priority]"
                exit 1
            fi
            store_memory "$@"
            ;;
        retrieve)
            retrieve_memory "$@"
            ;;
        stats)
            get_stats "$@"
            ;;
        consolidate)
            consolidate_memory "$@"
            ;;
        export)
            export_memories "$@"
            ;;
        import)
            if [ $# -lt 1 ]; then
                error "Usage: $0 import <file>"
                exit 1
            fi
            import_memories "$@"
            ;;
        clean)
            clean_expired
            ;;
        search)
            if [ $# -lt 1 ]; then
                error "Usage: $0 search <query> [session]"
                exit 1
            fi
            search_memories "$@"
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            error "Unknown command: $command"
            show_help
            exit 1
            ;;
    esac
}

# Run main
main "$@"
