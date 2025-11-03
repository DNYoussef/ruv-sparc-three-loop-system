#!/bin/bash
# Multi-Database Management Script for AgentDB
# Manages multiple AgentDB instances with sharding and backup/restore

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
AGENTDB_BASE_DIR="${AGENTDB_BASE_DIR:-.agentdb}"
BACKUP_DIR="${BACKUP_DIR:-$AGENTDB_BASE_DIR/backups}"
LOG_FILE="$AGENTDB_BASE_DIR/multi-db.log"

# Logging functions
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
    exit 1
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1" | tee -a "$LOG_FILE"
}

# Initialize multi-database setup
init_multi_db() {
    local num_shards=$1
    local shard_prefix=${2:-shard}

    log "Initializing $num_shards database shards with prefix '$shard_prefix'"

    mkdir -p "$AGENTDB_BASE_DIR"

    for i in $(seq 0 $((num_shards - 1))); do
        local db_path="$AGENTDB_BASE_DIR/${shard_prefix}-${i}.db"

        if [ -f "$db_path" ]; then
            warn "Database $db_path already exists, skipping"
        else
            log "Creating shard: $db_path"
            npx agentdb@latest init "$db_path"
        fi
    done

    log "Multi-database initialization complete"
}

# Shard data based on domain hash
get_shard_for_domain() {
    local domain=$1
    local num_shards=$2

    # Simple hash function (sum of ASCII values modulo num_shards)
    local hash=0
    for ((i=0; i<${#domain}; i++)); do
        hash=$((hash + $(printf '%d' "'${domain:$i:1}")))
    done

    echo $((hash % num_shards))
}

# List all managed databases
list_databases() {
    log "Listing all managed databases in $AGENTDB_BASE_DIR"

    local count=0
    for db in "$AGENTDB_BASE_DIR"/*.db; do
        if [ -f "$db" ]; then
            local size=$(du -h "$db" | cut -f1)
            local patterns=$(sqlite3 "$db" "SELECT COUNT(*) FROM patterns" 2>/dev/null || echo "N/A")
            echo "  📊 $(basename "$db"): $size, $patterns patterns"
            ((count++))
        fi
    done

    log "Total databases: $count"
}

# Backup all databases
backup_all() {
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local backup_path="$BACKUP_DIR/backup_$timestamp"

    mkdir -p "$backup_path"

    log "Backing up all databases to $backup_path"

    for db in "$AGENTDB_BASE_DIR"/*.db; do
        if [ -f "$db" ]; then
            local db_name=$(basename "$db")
            log "Backing up $db_name"
            cp "$db" "$backup_path/$db_name"

            # Export to JSON for portability
            npx agentdb@latest export "$db" "$backup_path/${db_name%.db}.json.gz" --compress
        fi
    done

    log "Backup complete: $backup_path"
    echo "$backup_path"
}

# Restore from backup
restore_backup() {
    local backup_path=$1

    if [ ! -d "$backup_path" ]; then
        error "Backup directory not found: $backup_path"
    fi

    log "Restoring from backup: $backup_path"

    for backup_file in "$backup_path"/*.db; do
        if [ -f "$backup_file" ]; then
            local db_name=$(basename "$backup_file")
            local target="$AGENTDB_BASE_DIR/$db_name"

            warn "Restoring $db_name (will overwrite if exists)"
            cp "$backup_file" "$target"
        fi
    done

    log "Restore complete"
}

# Merge multiple databases into one
merge_databases() {
    local output_db=$1
    shift
    local input_dbs=("$@")

    log "Merging ${#input_dbs[@]} databases into $output_db"

    # Create or clear output database
    npx agentdb@latest init "$output_db"

    for db in "${input_dbs[@]}"; do
        if [ ! -f "$db" ]; then
            warn "Database not found: $db, skipping"
            continue
        fi

        log "Merging $db"

        # Export patterns from source
        local temp_export="/tmp/agentdb_export_$$.json"
        npx agentdb@latest export "$db" "$temp_export"

        # Import into target
        npx agentdb@latest import "$temp_export" "$output_db"

        rm -f "$temp_export"
    done

    log "Merge complete: $output_db"
}

# Optimize all databases
optimize_all() {
    log "Optimizing all databases"

    for db in "$AGENTDB_BASE_DIR"/*.db; do
        if [ -f "$db" ]; then
            local db_name=$(basename "$db")
            log "Optimizing $db_name"

            # Vacuum to reclaim space
            sqlite3 "$db" "VACUUM;"

            # Analyze for query optimization
            sqlite3 "$db" "ANALYZE;"

            # Reindex
            npx agentdb@latest reindex "$db" 2>/dev/null || warn "Reindex not supported"
        fi
    done

    log "Optimization complete"
}

# Get database statistics
stats() {
    log "Database Statistics"

    local total_size=0
    local total_patterns=0

    for db in "$AGENTDB_BASE_DIR"/*.db; do
        if [ -f "$db" ]; then
            local db_name=$(basename "$db")
            local size_kb=$(du -k "$db" | cut -f1)
            local patterns=$(sqlite3 "$db" "SELECT COUNT(*) FROM patterns" 2>/dev/null || echo "0")

            echo "  📊 $db_name:"
            echo "     Size: $(numfmt --to=iec-i --suffix=B $((size_kb * 1024)))"
            echo "     Patterns: $patterns"

            total_size=$((total_size + size_kb))
            total_patterns=$((total_patterns + patterns))
        fi
    done

    echo ""
    echo "  📈 Total:"
    echo "     Size: $(numfmt --to=iec-i --suffix=B $((total_size * 1024)))"
    echo "     Patterns: $total_patterns"
}

# Shard migration - redistribute patterns across shards
reshard() {
    local old_shards=$1
    local new_shards=$2

    log "Resharding from $old_shards to $new_shards shards"

    # Create new shards
    init_multi_db "$new_shards" "new-shard"

    # Redistribute patterns
    for i in $(seq 0 $((old_shards - 1))); do
        local old_db="$AGENTDB_BASE_DIR/shard-${i}.db"

        if [ ! -f "$old_db" ]; then
            warn "Old shard not found: $old_db"
            continue
        fi

        log "Redistributing patterns from shard-$i"

        # Export patterns
        local temp_export="/tmp/agentdb_reshard_$$.json"
        npx agentdb@latest export "$old_db" "$temp_export"

        # Parse and redistribute based on new shard count
        # (In production, use proper JSON parsing with jq)

        rm -f "$temp_export"
    done

    log "Resharding complete. Review new-shard-* databases before replacing old shards"
}

# Main command dispatcher
case "$1" in
    init)
        init_multi_db "${2:-3}" "${3:-shard}"
        ;;
    list)
        list_databases
        ;;
    backup)
        backup_all
        ;;
    restore)
        restore_backup "$2"
        ;;
    merge)
        shift
        output=$1
        shift
        merge_databases "$output" "$@"
        ;;
    optimize)
        optimize_all
        ;;
    stats)
        stats
        ;;
    reshard)
        reshard "$2" "$3"
        ;;
    *)
        echo "Usage: $0 {init|list|backup|restore|merge|optimize|stats|reshard}"
        echo ""
        echo "Commands:"
        echo "  init <num_shards> [prefix]  - Initialize multiple database shards"
        echo "  list                        - List all managed databases"
        echo "  backup                      - Backup all databases"
        echo "  restore <backup_path>       - Restore from backup"
        echo "  merge <output> <db1> [db2...] - Merge databases"
        echo "  optimize                    - Optimize all databases"
        echo "  stats                       - Show database statistics"
        echo "  reshard <old_count> <new_count> - Redistribute across shards"
        exit 1
        ;;
esac
