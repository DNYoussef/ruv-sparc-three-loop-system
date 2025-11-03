#!/bin/bash
# Metrics Reporter
# Collect, aggregate, and report performance metrics for Claude Flow swarms

set -euo pipefail

# Configuration
METRICS_DIR="${METRICS_DIR:-./metrics}"
REPORT_DIR="${REPORT_DIR:-./reports}"
LOG_FILE="${LOG_FILE:-./metrics-reporter.log}"
RETENTION_DAYS="${RETENTION_DAYS:-30}"

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

# Error handling
error() {
    echo -e "${RED}ERROR: $*${NC}" >&2
    log "ERROR: $*"
    exit 1
}

# Warning
warn() {
    echo -e "${YELLOW}WARNING: $*${NC}" >&2
    log "WARNING: $*"
}

# Success message
success() {
    echo -e "${GREEN}$*${NC}"
    log "$*"
}

# Info message
info() {
    echo -e "${BLUE}$*${NC}"
    log "INFO: $*"
}

# Initialize directories
init_dirs() {
    mkdir -p "$METRICS_DIR" "$REPORT_DIR"
    log "Initialized directories: $METRICS_DIR, $REPORT_DIR"
}

# Collect system metrics
collect_system_metrics() {
    local output="$METRICS_DIR/system-$(date +'%Y%m%d-%H%M%S').json"

    info "Collecting system metrics..."

    # Get system info
    local cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | sed 's/%us,//')
    local mem_usage=$(free | grep Mem | awk '{printf "%.2f", $3/$2 * 100.0}')
    local disk_usage=$(df -h / | awk 'NR==2 {print $5}' | sed 's/%//')
    local load_avg=$(uptime | awk -F'load average:' '{print $2}' | sed 's/^ //')

    # Create JSON
    cat > "$output" <<EOF
{
  "timestamp": "$(date -Iseconds)",
  "system": {
    "cpu_usage": ${cpu_usage:-0},
    "memory_usage": ${mem_usage:-0},
    "disk_usage": ${disk_usage:-0},
    "load_average": "$load_avg"
  }
}
EOF

    log "System metrics collected: $output"
    echo "$output"
}

# Collect swarm metrics via MCP
collect_swarm_metrics() {
    local swarm_id="$1"
    local output="$METRICS_DIR/swarm-${swarm_id}-$(date +'%Y%m%d-%H%M%S').json"

    info "Collecting swarm metrics for: $swarm_id"

    # Call claude-flow MCP tools
    if command -v npx &> /dev/null; then
        npx claude-flow@alpha swarm status --swarm-id "$swarm_id" --json > "$output" 2>/dev/null || {
            warn "Failed to collect swarm metrics via MCP"
            return 1
        }
    else
        warn "npx not found, skipping swarm metrics collection"
        return 1
    fi

    log "Swarm metrics collected: $output"
    echo "$output"
}

# Aggregate metrics from multiple files
aggregate_metrics() {
    local time_range="${1:-1h}"
    local output="$REPORT_DIR/aggregated-$(date +'%Y%m%d-%H%M%S').json"

    info "Aggregating metrics for time range: $time_range"

    # Convert time range to seconds
    local seconds=3600 # default 1h
    case "$time_range" in
        *h) seconds=$((${time_range%h} * 3600)) ;;
        *d) seconds=$((${time_range%d} * 86400)) ;;
        *m) seconds=$((${time_range%m} * 60)) ;;
    esac

    # Find files within time range
    local cutoff=$(date -d "@$(($(date +%s) - seconds))" +'%Y%m%d-%H%M%S')
    local files=$(find "$METRICS_DIR" -name "*.json" -newer "$METRICS_DIR" 2>/dev/null | \
                  awk -v cutoff="$cutoff" -F'-' '{if ($NF >= cutoff) print}' || echo "")

    if [ -z "$files" ]; then
        warn "No metrics files found for aggregation"
        return 1
    fi

    # Aggregate using jq if available
    if command -v jq &> /dev/null; then
        jq -s '.' $files > "$output"
        log "Metrics aggregated: $output"
        echo "$output"
    else
        warn "jq not found, skipping aggregation"
        return 1
    fi
}

# Generate performance report
generate_report() {
    local format="${1:-text}"
    local time_range="${2:-24h}"
    local output="$REPORT_DIR/performance-report-$(date +'%Y%m%d-%H%M%S')"

    info "Generating performance report (format: $format, range: $time_range)"

    # Aggregate metrics first
    local aggregated=$(aggregate_metrics "$time_range")
    if [ -z "$aggregated" ]; then
        error "Failed to aggregate metrics"
    fi

    case "$format" in
        json)
            cp "$aggregated" "${output}.json"
            success "JSON report generated: ${output}.json"
            ;;

        html)
            generate_html_report "$aggregated" "${output}.html"
            success "HTML report generated: ${output}.html"
            ;;

        text|markdown)
            generate_text_report "$aggregated" "${output}.md"
            success "Markdown report generated: ${output}.md"
            ;;

        *)
            error "Unknown format: $format (use json, html, or text)"
            ;;
    esac

    echo "$output"
}

# Generate HTML report
generate_html_report() {
    local input="$1"
    local output="$2"

    cat > "$output" <<EOF
<!DOCTYPE html>
<html>
<head>
    <title>Performance Analysis Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        h1 { color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }
        h2 { color: #555; margin-top: 30px; }
        .metric { background: #f9f9f9; padding: 15px; margin: 10px 0; border-left: 4px solid #4CAF50; border-radius: 4px; }
        .metric-label { font-weight: bold; color: #666; }
        .metric-value { font-size: 1.2em; color: #333; }
        .critical { border-left-color: #f44336; }
        .warning { border-left-color: #ff9800; }
        .info { border-left-color: #2196F3; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th { background: #4CAF50; color: white; padding: 12px; text-align: left; }
        td { padding: 10px; border-bottom: 1px solid #ddd; }
        tr:hover { background: #f5f5f5; }
        .footer { margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #999; font-size: 0.9em; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Performance Analysis Report</h1>
        <p>Generated: $(date +'%Y-%m-%d %H:%M:%S')</p>

        <h2>Summary Metrics</h2>
        <div class="metric">
            <div class="metric-label">Data Source</div>
            <div class="metric-value">$input</div>
        </div>

        <h2>Detailed Analysis</h2>
        <p>See JSON data for complete metrics breakdown.</p>

        <div class="footer">
            Report generated by Claude Flow Metrics Reporter
        </div>
    </div>
</body>
</html>
EOF
}

# Generate text/markdown report
generate_text_report() {
    local input="$1"
    local output="$2"

    cat > "$output" <<EOF
# Performance Analysis Report

**Generated**: $(date +'%Y-%m-%d %H:%M:%S')

---

## Summary

- **Data Source**: $input
- **Time Range**: Last 24 hours
- **Report Format**: Markdown

---

## Metrics Overview

See JSON file for detailed metrics breakdown.

---

## Recommendations

1. Review aggregated metrics for bottlenecks
2. Analyze agent utilization patterns
3. Optimize slow performing components
4. Monitor trends over time

---

*Report generated by Claude Flow Metrics Reporter*
EOF
}

# Clean old metrics
cleanup_old_metrics() {
    info "Cleaning metrics older than $RETENTION_DAYS days..."

    find "$METRICS_DIR" -name "*.json" -mtime +$RETENTION_DAYS -delete
    find "$REPORT_DIR" -name "*" -mtime +$RETENTION_DAYS -delete

    success "Cleanup completed"
}

# Main function
main() {
    local command="${1:-help}"
    shift || true

    init_dirs

    case "$command" in
        collect-system)
            collect_system_metrics
            ;;

        collect-swarm)
            [ -z "${1:-}" ] && error "Usage: $0 collect-swarm <swarm-id>"
            collect_swarm_metrics "$1"
            ;;

        aggregate)
            aggregate_metrics "${1:-1h}"
            ;;

        report)
            generate_report "${1:-text}" "${2:-24h}"
            ;;

        cleanup)
            cleanup_old_metrics
            ;;

        help|*)
            cat <<EOF
Usage: $0 <command> [options]

Commands:
  collect-system              Collect system metrics
  collect-swarm <id>          Collect swarm metrics
  aggregate [time-range]      Aggregate metrics (default: 1h)
  report [format] [range]     Generate report (default: text, 24h)
  cleanup                     Clean old metrics
  help                        Show this help

Examples:
  $0 collect-system
  $0 collect-swarm swarm-123
  $0 aggregate 24h
  $0 report html 7d
  $0 cleanup

Environment Variables:
  METRICS_DIR       Metrics directory (default: ./metrics)
  REPORT_DIR        Reports directory (default: ./reports)
  RETENTION_DAYS    Metrics retention (default: 30)
EOF
            ;;
    esac
}

# Run main
main "$@"
