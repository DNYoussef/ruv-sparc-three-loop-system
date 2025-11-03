#!/bin/bash
###############################################################################
# OpenTelemetry Log Aggregator
# Correlate logs with traces using trace_id and span_id
###############################################################################

set -euo pipefail

# Configuration
LOG_DIR="${LOG_DIR:-/var/log/app}"
OUTPUT_FILE="${OUTPUT_FILE:-aggregated-logs.jsonl}"
TRACE_ID_PATTERN="trace_id"
SPAN_ID_PATTERN="span_id"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

###############################################################################
# Functions
###############################################################################

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_success() {
    echo -e "${BLUE}[SUCCESS]${NC} $1"
}

###############################################################################
# Check if jq is installed (required for JSON processing)
###############################################################################
check_dependencies() {
    if ! command -v jq &> /dev/null; then
        log_error "jq is required but not installed. Install with: apt-get install jq"
        exit 1
    fi
    log_info "Dependencies verified ✓"
}

###############################################################################
# Aggregate logs with trace correlation
###############################################################################
aggregate_logs() {
    local log_files=("$@")

    log_info "Aggregating ${#log_files[@]} log file(s)..."

    > "$OUTPUT_FILE" # Clear output file

    local total_logs=0
    local correlated_logs=0

    for log_file in "${log_files[@]}"; do
        if [[ ! -f "$log_file" ]]; then
            log_warn "Log file not found: $log_file"
            continue
        fi

        log_info "Processing: $log_file"

        # Process each line
        while IFS= read -r line; do
            total_logs=$((total_logs + 1))

            # Check if line is valid JSON
            if echo "$line" | jq empty 2>/dev/null; then
                # Extract trace_id and span_id if present
                trace_id=$(echo "$line" | jq -r ".${TRACE_ID_PATTERN} // empty")
                span_id=$(echo "$line" | jq -r ".${SPAN_ID_PATTERN} // empty")

                # Add correlation metadata
                if [[ -n "$trace_id" ]]; then
                    correlated_logs=$((correlated_logs + 1))
                    enriched=$(echo "$line" | jq -c \
                        --arg source "$log_file" \
                        --arg correlated "true" \
                        '. + {source: $source, correlated: $correlated}')
                    echo "$enriched" >> "$OUTPUT_FILE"
                else
                    # Add uncorrelated log
                    enriched=$(echo "$line" | jq -c \
                        --arg source "$log_file" \
                        --arg correlated "false" \
                        '. + {source: $source, correlated: $correlated}')
                    echo "$enriched" >> "$OUTPUT_FILE"
                fi
            else
                # Handle non-JSON logs
                jq -n -c \
                    --arg message "$line" \
                    --arg source "$log_file" \
                    --arg correlated "false" \
                    '{message: $message, source: $source, correlated: $correlated}' >> "$OUTPUT_FILE"
            fi
        done < "$log_file"
    done

    log_success "Aggregation complete!"
    log_info "Total logs: $total_logs"
    log_info "Correlated logs: $correlated_logs ($(( correlated_logs * 100 / total_logs ))%)"
    log_info "Output written to: $OUTPUT_FILE"
}

###############################################################################
# Query logs by trace ID
###############################################################################
query_by_trace() {
    local trace_id="$1"

    log_info "Querying logs for trace_id: $trace_id"

    if [[ ! -f "$OUTPUT_FILE" ]]; then
        log_error "Aggregated log file not found: $OUTPUT_FILE"
        exit 1
    fi

    local results
    results=$(jq -c "select(.${TRACE_ID_PATTERN} == \"$trace_id\")" "$OUTPUT_FILE")

    if [[ -z "$results" ]]; then
        log_warn "No logs found for trace_id: $trace_id"
        return
    fi

    echo "$results" | jq -C '.'
}

###############################################################################
# Generate correlation report
###############################################################################
generate_report() {
    log_info "Generating correlation report..."

    if [[ ! -f "$OUTPUT_FILE" ]]; then
        log_error "Aggregated log file not found: $OUTPUT_FILE"
        exit 1
    fi

    local total=$(wc -l < "$OUTPUT_FILE")
    local correlated=$(jq -r 'select(.correlated == "true")' "$OUTPUT_FILE" | wc -l)
    local uncorrelated=$((total - correlated))

    echo ""
    echo "========================================="
    echo "  LOG CORRELATION REPORT"
    echo "========================================="
    echo "Total Logs:        $total"
    echo "Correlated:        $correlated ($(( correlated * 100 / total ))%)"
    echo "Uncorrelated:      $uncorrelated ($(( uncorrelated * 100 / total ))%)"
    echo ""

    # Top traces by log count
    echo "Top 5 Traces by Log Count:"
    jq -r ".${TRACE_ID_PATTERN} // empty" "$OUTPUT_FILE" | \
        sort | uniq -c | sort -rn | head -5 | \
        awk '{printf "  %s: %d logs\n", $2, $1}'

    echo "========================================="
}

###############################################################################
# Create sample logs for testing
###############################################################################
create_sample_logs() {
    local sample_file="sample-app.log"

    log_info "Creating sample logs with trace correlation..."

    cat > "$sample_file" <<EOF
{"timestamp":"2025-11-02T10:00:01Z","level":"info","message":"Request received","trace_id":"abc123","span_id":"span001","http.method":"POST","http.route":"/api/orders"}
{"timestamp":"2025-11-02T10:00:02Z","level":"info","message":"Order validated","trace_id":"abc123","span_id":"span002","order.id":"ORD-12345"}
{"timestamp":"2025-11-02T10:00:03Z","level":"error","message":"Payment failed","trace_id":"abc123","span_id":"span003","error":"Insufficient funds"}
{"timestamp":"2025-11-02T10:00:04Z","level":"info","message":"Request completed","trace_id":"abc123","span_id":"span001","http.status_code":402}
{"timestamp":"2025-11-02T10:01:01Z","level":"info","message":"Request received","trace_id":"def456","span_id":"span010","http.method":"GET","http.route":"/api/users"}
{"timestamp":"2025-11-02T10:01:02Z","level":"info","message":"Database query","trace_id":"def456","span_id":"span011","db.statement":"SELECT * FROM users"}
{"timestamp":"2025-11-02T10:01:03Z","level":"info","message":"Request completed","trace_id":"def456","span_id":"span010","http.status_code":200}
{"timestamp":"2025-11-02T10:02:01Z","level":"warn","message":"Uncorrelated log without trace context"}
EOF

    log_success "Sample logs created: $sample_file"
}

###############################################################################
# Main
###############################################################################
main() {
    echo "======================================"
    echo "  OpenTelemetry Log Aggregator"
    echo "======================================"
    echo ""

    check_dependencies

    # Parse command line arguments
    case "${1:-}" in
        aggregate)
            shift
            if [[ $# -eq 0 ]]; then
                log_error "Usage: $0 aggregate <log_file1> [log_file2 ...]"
                exit 1
            fi
            aggregate_logs "$@"
            generate_report
            ;;
        query)
            if [[ -z "${2:-}" ]]; then
                log_error "Usage: $0 query <trace_id>"
                exit 1
            fi
            query_by_trace "$2"
            ;;
        report)
            generate_report
            ;;
        sample)
            create_sample_logs
            aggregate_logs "sample-app.log"
            generate_report
            echo ""
            log_info "Try: $0 query abc123"
            ;;
        *)
            echo "Usage: $0 {aggregate|query|report|sample}"
            echo ""
            echo "Commands:"
            echo "  aggregate <files...>  Aggregate log files with trace correlation"
            echo "  query <trace_id>      Query logs by trace ID"
            echo "  report                Generate correlation report"
            echo "  sample                Create sample logs for testing"
            exit 1
            ;;
    esac
}

main "$@"
