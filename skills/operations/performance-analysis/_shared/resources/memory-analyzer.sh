#!/bin/bash
##
## Memory Analyzer - Comprehensive memory profiling and leak detection
##
## Features:
## - System memory analysis
## - Process memory tracking
## - Heap dump analysis
## - Memory leak detection
## - Swap usage monitoring
## - OOM killer history
##

set -euo pipefail

# Colors for output
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
OUTPUT_DIR="${OUTPUT_DIR:-./memory-analysis}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-5}"
DURATION="${DURATION:-60}"
MEMORY_THRESHOLD="${MEMORY_THRESHOLD:-80}"
SWAP_THRESHOLD="${SWAP_THRESHOLD:-50}"

# Create output directory
mkdir -p "$OUTPUT_DIR"

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $*"
}

warn() {
    echo -e "${YELLOW}[WARNING]${NC} $*"
}

error() {
    echo -e "${RED}[ERROR]${NC} $*"
}

# Analyze system memory
analyze_system_memory() {
    log "Analyzing system memory..."

    local report="$OUTPUT_DIR/system_memory_$(date +%s).txt"

    {
        echo "=== System Memory Analysis ==="
        echo "Generated: $(date)"
        echo ""

        echo "--- Memory Usage ---"
        free -h
        echo ""

        echo "--- Memory Info ---"
        cat /proc/meminfo | head -20
        echo ""

        echo "--- Virtual Memory Stats ---"
        vmstat -s
        echo ""

        echo "--- Swap Usage ---"
        swapon --show
        echo ""

        echo "--- Memory Pressure ---"
        if [ -f /proc/pressure/memory ]; then
            cat /proc/pressure/memory
        else
            echo "Memory pressure info not available"
        fi
        echo ""

    } > "$report"

    log "System memory report saved to: $report"

    # Check for memory pressure
    local mem_usage=$(free | grep Mem | awk '{print int($3/$2 * 100)}')
    if [ "$mem_usage" -gt "$MEMORY_THRESHOLD" ]; then
        warn "High memory usage: ${mem_usage}% (threshold: ${MEMORY_THRESHOLD}%)"
    fi
}

# Track process memory over time
track_process_memory() {
    local pid="$1"
    local duration="${2:-$DURATION}"

    log "Tracking memory for PID $pid for ${duration}s..."

    local report="$OUTPUT_DIR/process_${pid}_$(date +%s).csv"

    # CSV header
    echo "timestamp,rss_mb,vms_mb,shared_mb,cpu_percent,threads" > "$report"

    local samples=$((duration / SAMPLE_INTERVAL))

    for ((i=0; i<samples; i++)); do
        if ! ps -p "$pid" > /dev/null 2>&1; then
            error "Process $pid no longer exists"
            break
        fi

        local timestamp=$(date +%s)
        local stats=$(ps -p "$pid" -o rss=,vsz=,sz=,%cpu=,nlwp= 2>/dev/null || echo "0 0 0 0 0")
        local rss=$(echo "$stats" | awk '{print $1/1024}')
        local vms=$(echo "$stats" | awk '{print $2/1024}')
        local shared=$(echo "$stats" | awk '{print $3/1024}')
        local cpu=$(echo "$stats" | awk '{print $4}')
        local threads=$(echo "$stats" | awk '{print $5}')

        echo "$timestamp,$rss,$vms,$shared,$cpu,$threads" >> "$report"

        sleep "$SAMPLE_INTERVAL"
    done

    log "Process memory tracking saved to: $report"

    # Analyze for memory growth
    analyze_memory_growth "$report"
}

# Analyze memory growth pattern
analyze_memory_growth() {
    local csv_file="$1"

    log "Analyzing memory growth pattern..."

    # Get first and last RSS values
    local first_rss=$(tail -n +2 "$csv_file" | head -1 | cut -d',' -f2)
    local last_rss=$(tail -1 "$csv_file" | cut -d',' -f2)

    # Calculate growth
    local growth=$(echo "$last_rss - $first_rss" | bc)
    local growth_percent=$(echo "scale=2; ($growth / $first_rss) * 100" | bc)

    echo ""
    echo "Memory Growth Analysis:"
    echo "  Initial RSS: ${first_rss} MB"
    echo "  Final RSS: ${last_rss} MB"
    echo "  Growth: ${growth} MB (${growth_percent}%)"

    # Check for potential leak
    if (( $(echo "$growth_percent > 10" | bc -l) )); then
        warn "Potential memory leak detected: ${growth_percent}% growth"

        # Generate gnuplot graph if available
        if command -v gnuplot &> /dev/null; then
            generate_memory_graph "$csv_file"
        fi
    fi
}

# Generate memory usage graph
generate_memory_graph() {
    local csv_file="$1"
    local graph_file="${csv_file%.csv}.png"

    log "Generating memory graph..."

    gnuplot <<EOF
set terminal png size 1200,800
set output '$graph_file'
set datafile separator ","
set xlabel "Time (samples)"
set ylabel "Memory (MB)"
set title "Memory Usage Over Time"
set grid
set key outside

plot '$csv_file' using 0:2 with lines title "RSS" lw 2, \
     '' using 0:3 with lines title "VMS" lw 2, \
     '' using 0:4 with lines title "Shared" lw 2
EOF

    log "Memory graph saved to: $graph_file"
}

# Find top memory consumers
find_top_consumers() {
    local count="${1:-10}"

    log "Finding top $count memory consumers..."

    echo ""
    echo "=== Top Memory Consumers ==="
    ps aux --sort=-%mem | head -n $((count + 1)) | \
        awk '{printf "%-8s %-6s %6s %6s %s\n", $1, $2, $3, $4, $11}'
    echo ""
}

# Check for OOM killer activity
check_oom_killer() {
    log "Checking OOM killer history..."

    local report="$OUTPUT_DIR/oom_history_$(date +%s).txt"

    {
        echo "=== OOM Killer History ==="
        echo "Generated: $(date)"
        echo ""

        # Check dmesg for OOM events
        if dmesg | grep -i "killed process" > /dev/null 2>&1; then
            echo "OOM killer events found:"
            dmesg | grep -i "killed process" | tail -20
        else
            echo "No OOM killer events found in recent history"
        fi

        echo ""
        echo "--- Recent OOM Logs ---"
        journalctl -k -g "Out of memory" --since "24 hours ago" 2>/dev/null || \
            echo "No journalctl access or no recent OOM events"

    } > "$report"

    log "OOM history saved to: $report"
}

# Analyze swap usage
analyze_swap() {
    log "Analyzing swap usage..."

    local swap_total=$(free | grep Swap | awk '{print $2}')

    if [ "$swap_total" -eq 0 ]; then
        warn "No swap space configured"
        return
    fi

    local swap_used=$(free | grep Swap | awk '{print $3}')
    local swap_percent=$(echo "scale=2; ($swap_used / $swap_total) * 100" | bc)

    echo ""
    echo "Swap Analysis:"
    echo "  Total: $(echo "scale=2; $swap_total / 1024 / 1024" | bc) GB"
    echo "  Used: $(echo "scale=2; $swap_used / 1024 / 1024" | bc) GB"
    echo "  Percent: ${swap_percent}%"

    if (( $(echo "$swap_percent > $SWAP_THRESHOLD" | bc -l) )); then
        warn "High swap usage: ${swap_percent}% (threshold: ${SWAP_THRESHOLD}%)"
    fi
}

# Detect memory leaks in running processes
detect_memory_leaks() {
    log "Detecting memory leaks in top processes..."

    # Get top 5 memory consumers
    local pids=$(ps aux --sort=-%mem | tail -n +2 | head -5 | awk '{print $2}')

    for pid in $pids; do
        local cmdline=$(ps -p "$pid" -o comm= 2>/dev/null || echo "unknown")
        log "Analyzing PID $pid ($cmdline)..."

        # Track for shorter duration
        track_process_memory "$pid" 30 &
    done

    wait
}

# Generate comprehensive report
generate_report() {
    log "Generating comprehensive memory analysis report..."

    local report="$OUTPUT_DIR/memory_report_$(date +%s).txt"

    {
        echo "=========================================="
        echo "   COMPREHENSIVE MEMORY ANALYSIS REPORT"
        echo "=========================================="
        echo "Generated: $(date)"
        echo ""

        analyze_system_memory
        find_top_consumers 10
        analyze_swap
        check_oom_killer

        echo ""
        echo "=========================================="
        echo "Report files saved in: $OUTPUT_DIR"
        echo "=========================================="

    } | tee "$report"

    log "Full report saved to: $report"
}

# Main execution
main() {
    log "Starting Memory Analyzer"
    log "Output directory: $OUTPUT_DIR"

    case "${1:-full}" in
        system)
            analyze_system_memory
            ;;
        track)
            if [ -z "${2:-}" ]; then
                error "Please provide PID to track"
                exit 1
            fi
            track_process_memory "$2" "${3:-$DURATION}"
            ;;
        top)
            find_top_consumers "${2:-10}"
            ;;
        oom)
            check_oom_killer
            ;;
        swap)
            analyze_swap
            ;;
        leaks)
            detect_memory_leaks
            ;;
        full|*)
            generate_report
            ;;
    esac

    log "Memory analysis complete"
}

# Run main function
main "$@"
