#!/bin/bash
# Resource Utilization Analysis Example
# Demonstrates comprehensive resource monitoring and optimization for swarms
#
# This example shows:
# - Multi-dimensional resource tracking (CPU, memory, disk, network)
# - Real-time utilization monitoring
# - Threshold-based alerting
# - Resource optimization recommendations

set -euo pipefail

# Configuration
EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="$EXAMPLE_DIR/output"
MONITOR_INTERVAL=2
DURATION=30

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Initialize
mkdir -p "$OUTPUT_DIR"

# Print header
print_header() {
    echo ""
    echo "======================================================================"
    echo "$1"
    echo "======================================================================"
    echo ""
}

# Simulate agent resource usage
simulate_agent_resources() {
    local agent_id="$1"
    local agent_type="$2"
    local iteration="$3"

    # Vary resource usage based on agent type and iteration
    local cpu_base
    local mem_base

    case "$agent_type" in
        coder)
            cpu_base=60
            mem_base=50
            ;;
        researcher)
            cpu_base=40
            mem_base=70
            ;;
        tester)
            cpu_base=70
            mem_base=45
            ;;
        *)
            cpu_base=50
            mem_base=50
            ;;
    esac

    # Add variance
    local variance=$((RANDOM % 20 - 10))
    local cpu_usage=$((cpu_base + variance))
    local mem_usage=$((mem_base + variance + iteration * 2))

    # Cap at 100
    [ $cpu_usage -gt 100 ] && cpu_usage=100
    [ $mem_usage -gt 100 ] && mem_usage=100
    [ $cpu_usage -lt 0 ] && cpu_usage=0
    [ $mem_usage -lt 0 ] && mem_usage=0

    # Simulate disk and network usage
    local disk_usage=$((30 + RANDOM % 40))
    local network_usage=$((20 + RANDOM % 60))

    echo "{
  \"agent_id\": \"$agent_id\",
  \"agent_type\": \"$agent_type\",
  \"timestamp\": \"$(date -Iseconds)\",
  \"resources\": {
    \"cpu_percent\": $cpu_usage,
    \"memory_percent\": $mem_usage,
    \"disk_percent\": $disk_usage,
    \"network_mbps\": $network_usage
  }
}"
}

# Collect system-wide metrics
collect_system_metrics() {
    local timestamp="$1"

    # Simulate system metrics
    local total_cpu=$((40 + RANDOM % 50))
    local total_mem=$((50 + RANDOM % 40))
    local total_disk=$((35 + RANDOM % 30))

    cat <<EOF
{
  "timestamp": "$timestamp",
  "system": {
    "cpu_usage": $total_cpu,
    "memory_usage": $total_mem,
    "disk_usage": $total_disk,
    "load_average": "$(echo "scale=2; $total_cpu / 100 * 4" | bc)"
  }
}
EOF
}

# Monitor resources
monitor_resources() {
    local duration="$1"
    local interval="$2"

    print_header "RESOURCE UTILIZATION MONITORING"

    echo "Monitoring for ${duration}s with ${interval}s intervals"
    echo ""

    # Agent configuration
    declare -A agents=(
        [agent-1]="coder"
        [agent-2]="researcher"
        [agent-3]="tester"
        [agent-4]="coder"
        [agent-5]="researcher"
    )

    local iterations=$((duration / interval))
    local all_metrics="$OUTPUT_DIR/all-metrics.json"

    echo "[" > "$all_metrics"

    for ((i=0; i<iterations; i++)); do
        local timestamp=$(date -Iseconds)

        echo -ne "\rIteration: $((i+1))/$iterations"

        # Collect metrics for each agent
        for agent_id in "${!agents[@]}"; do
            local agent_type="${agents[$agent_id]}"
            local metrics=$(simulate_agent_resources "$agent_id" "$agent_type" "$i")

            # Append to all metrics
            echo "  $metrics," >> "$all_metrics"

            # Parse and check thresholds
            local cpu=$(echo "$metrics" | grep -o '"cpu_percent": [0-9]*' | awk '{print $2}')
            local mem=$(echo "$metrics" | grep -o '"memory_percent": [0-9]*' | awk '{print $2}')

            # Alert on high usage
            if [ "$cpu" -gt 80 ]; then
                echo -e "\n${YELLOW}⚠️  HIGH CPU: $agent_id at ${cpu}%${NC}"
            fi

            if [ "$mem" -gt 80 ]; then
                echo -e "\n${RED}🔴 HIGH MEMORY: $agent_id at ${mem}%${NC}"
            fi
        done

        # Collect system metrics
        local sys_metrics=$(collect_system_metrics "$timestamp")
        echo "  $sys_metrics," >> "$all_metrics"

        sleep "$interval"
    done

    echo ""
    echo "]" >> "$all_metrics"

    # Fix JSON formatting (remove trailing comma)
    sed -i 's/,$//' "$all_metrics"

    echo ""
    echo -e "${GREEN}✓ Monitoring complete${NC}"
    echo "Metrics saved to: $all_metrics"
}

# Analyze resource utilization
analyze_resources() {
    local metrics_file="$1"

    print_header "RESOURCE UTILIZATION ANALYSIS"

    if ! command -v jq &> /dev/null; then
        echo "⚠️  jq not available, skipping detailed analysis"
        return
    fi

    # Calculate statistics
    echo "Calculating resource statistics..."
    echo ""

    # CPU statistics
    local cpu_values=$(jq -r '.[] | select(.resources) | .resources.cpu_percent' "$metrics_file")
    local cpu_avg=$(echo "$cpu_values" | awk '{sum+=$1; n++} END {print sum/n}')
    local cpu_max=$(echo "$cpu_values" | sort -n | tail -1)
    local cpu_min=$(echo "$cpu_values" | sort -n | head -1)

    echo "CPU Utilization:"
    echo "  Average: ${cpu_avg}%"
    echo "  Maximum: ${cpu_max}%"
    echo "  Minimum: ${cpu_min}%"
    echo ""

    # Memory statistics
    local mem_values=$(jq -r '.[] | select(.resources) | .resources.memory_percent' "$metrics_file")
    local mem_avg=$(echo "$mem_values" | awk '{sum+=$1; n++} END {print sum/n}')
    local mem_max=$(echo "$mem_values" | sort -n | tail -1)
    local mem_min=$(echo "$mem_values" | sort -n | head -1)

    echo "Memory Utilization:"
    echo "  Average: ${mem_avg}%"
    echo "  Maximum: ${mem_max}%"
    echo "  Minimum: ${mem_min}%"
    echo ""

    # Identify resource-intensive agents
    echo "Resource-Intensive Agents:"
    jq -r '.[] | select(.resources) | select(.resources.cpu_percent > 70 or .resources.memory_percent > 70) | "  - \(.agent_id) (\(.agent_type)): CPU=\(.resources.cpu_percent)%, MEM=\(.resources.memory_percent)%"' "$metrics_file" | sort -u
    echo ""

    # Generate recommendations
    generate_recommendations "$cpu_avg" "$mem_avg" "$cpu_max" "$mem_max"
}

# Generate optimization recommendations
generate_recommendations() {
    local cpu_avg="$1"
    local mem_avg="$2"
    local cpu_max="$3"
    local mem_max="$4"

    print_header "OPTIMIZATION RECOMMENDATIONS"

    local recommendations=()

    # High average CPU
    if (( $(echo "$cpu_avg > 70" | bc -l) )); then
        recommendations+=("${YELLOW}HIGH CPU AVERAGE${NC}: Consider scaling up agents or optimizing algorithms")
    fi

    # High average memory
    if (( $(echo "$mem_avg > 70" | bc -l) )); then
        recommendations+=("${YELLOW}HIGH MEMORY AVERAGE${NC}: Enable aggressive GC or add memory capacity")
    fi

    # CPU spikes
    if (( $(echo "$cpu_max > 90" | bc -l) )); then
        recommendations+=("${RED}CRITICAL CPU SPIKES${NC}: Implement CPU throttling or load balancing")
    fi

    # Memory spikes
    if (( $(echo "$mem_max > 90" | bc -l) )); then
        recommendations+=("${RED}CRITICAL MEMORY SPIKES${NC}: Review memory leaks and implement limits")
    fi

    # Balanced load
    if (( $(echo "$cpu_avg < 50" | bc -l) )) && (( $(echo "$mem_avg < 50" | bc -l) )); then
        recommendations+=("${GREEN}WELL BALANCED${NC}: Resources are well utilized")
    fi

    if [ ${#recommendations[@]} -eq 0 ]; then
        echo "No critical recommendations - system operating normally"
    else
        for rec in "${recommendations[@]}"; do
            echo -e "  • $rec"
        done
    fi

    echo ""
}

# Generate resource report
generate_report() {
    local metrics_file="$1"
    local report_file="$OUTPUT_DIR/resource-report.md"

    print_header "GENERATING COMPREHENSIVE REPORT"

    cat > "$report_file" <<EOF
# Resource Utilization Analysis Report

**Generated**: $(date '+%Y-%m-%d %H:%M:%S')

---

## Executive Summary

This report provides comprehensive resource utilization analysis for the Claude Flow swarm environment.

---

## Methodology

- **Monitoring Duration**: ${DURATION}s
- **Sample Interval**: ${MONITOR_INTERVAL}s
- **Agents Monitored**: 5
- **Metrics Collected**: CPU, Memory, Disk, Network

---

## Key Findings

EOF

    if command -v jq &> /dev/null; then
        # Add statistics to report
        local cpu_avg=$(jq -r '[.[] | select(.resources) | .resources.cpu_percent] | add / length' "$metrics_file")
        local mem_avg=$(jq -r '[.[] | select(.resources) | .resources.memory_percent] | add / length' "$metrics_file")

        cat >> "$report_file" <<EOF
### Resource Statistics

| Metric | Average | Maximum | Status |
|--------|---------|---------|--------|
| CPU Usage | ${cpu_avg}% | $(jq -r '[.[] | select(.resources) | .resources.cpu_percent] | max' "$metrics_file")% | $([ $(echo "$cpu_avg < 70" | bc -l) -eq 1 ] && echo "✓ Good" || echo "⚠️ High") |
| Memory Usage | ${mem_avg}% | $(jq -r '[.[] | select(.resources) | .resources.memory_percent] | max' "$metrics_file")% | $([ $(echo "$mem_avg < 70" | bc -l) -eq 1 ] && echo "✓ Good" || echo "⚠️ High") |

---

## Optimization Opportunities

1. **Resource Balancing**: Redistribute workload across agents for better utilization
2. **Caching Strategy**: Implement intelligent caching to reduce memory pressure
3. **Parallel Processing**: Increase parallelization to improve CPU efficiency
4. **Auto-Scaling**: Enable dynamic agent scaling based on resource usage

---

## Detailed Metrics

See attached JSON file for complete raw metrics: \`all-metrics.json\`

---

**Report Generated by Claude Flow Performance Analysis**
EOF
    else
        cat >> "$report_file" <<EOF
### Note

Detailed statistics require jq. Install jq for comprehensive analysis.

---

**Report Generated by Claude Flow Performance Analysis**
EOF
    fi

    echo "Report generated: $report_file"
    echo ""
    echo "--- Report Preview ---"
    head -30 "$report_file"
    echo "..."
    echo ""
}

# Main execution
main() {
    print_header "RESOURCE UTILIZATION ANALYSIS EXAMPLE"

    echo "This example demonstrates comprehensive resource monitoring"
    echo "and optimization for Claude Flow swarms."
    echo ""

    # Monitor resources
    monitor_resources "$DURATION" "$MONITOR_INTERVAL"

    # Analyze collected metrics
    analyze_resources "$OUTPUT_DIR/all-metrics.json"

    # Generate report
    generate_report "$OUTPUT_DIR/all-metrics.json"

    print_header "EXAMPLE COMPLETE"

    echo "Output files:"
    echo "  - Metrics: $OUTPUT_DIR/all-metrics.json"
    echo "  - Report:  $OUTPUT_DIR/resource-report.md"
    echo ""
}

# Run example
main "$@"
