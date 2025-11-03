#!/bin/bash
# Training Monitor
#
# Real-time monitoring of ML training progress with alert triggers.
# Monitors logs, metrics, and resource utilization.
#
# Usage:
#   ./training-monitor.sh --log-dir ./logs --alert-threshold 0.1
#   ./training-monitor.sh --log-file train.log --gpu-monitor --interval 5

set -euo pipefail

# Default configuration
LOG_DIR=""
LOG_FILE=""
ALERT_THRESHOLD=0.1
CHECK_INTERVAL=10
GPU_MONITOR=false
MEMORY_MONITOR=false
OUTPUT_FILE=""
ALERT_EMAIL=""

# Colors for output
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --log-dir)
      LOG_DIR="$2"
      shift 2
      ;;
    --log-file)
      LOG_FILE="$2"
      shift 2
      ;;
    --alert-threshold)
      ALERT_THRESHOLD="$2"
      shift 2
      ;;
    --interval)
      CHECK_INTERVAL="$2"
      shift 2
      ;;
    --gpu-monitor)
      GPU_MONITOR=true
      shift
      ;;
    --memory-monitor)
      MEMORY_MONITOR=true
      shift
      ;;
    --output)
      OUTPUT_FILE="$2"
      shift 2
      ;;
    --email)
      ALERT_EMAIL="$2"
      shift 2
      ;;
    --help)
      cat << EOF
Training Monitor - Real-time ML training monitoring with alerts

Usage:
  ./training-monitor.sh [OPTIONS]

Options:
  --log-dir <dir>          Directory containing training logs
  --log-file <file>        Specific log file to monitor
  --alert-threshold <num>  Loss increase threshold for alerts (default: 0.1)
  --interval <seconds>     Check interval in seconds (default: 10)
  --gpu-monitor            Enable GPU monitoring (requires nvidia-smi)
  --memory-monitor         Enable memory usage monitoring
  --output <file>          Save monitoring data to file
  --email <address>        Email address for critical alerts
  --help                   Show this message

Examples:
  # Monitor training logs in ./logs directory
  ./training-monitor.sh --log-dir ./logs --gpu-monitor

  # Monitor specific file with alerts
  ./training-monitor.sh --log-file train.log --alert-threshold 0.2 --interval 5

  # Full monitoring with output
  ./training-monitor.sh --log-dir ./logs --gpu-monitor --memory-monitor --output metrics.jsonl
EOF
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Validate inputs
if [[ -z "$LOG_DIR" && -z "$LOG_FILE" ]]; then
  echo -e "${RED}Error: Must provide --log-dir or --log-file${NC}"
  exit 1
fi

# Initialize monitoring state
PREV_LOSS=""
LOSS_HISTORY=()
ALERT_COUNT=0
START_TIME=$(date +%s)

# Function to extract latest loss from logs
extract_loss() {
  local log_source="$1"

  # Try common loss patterns
  local loss=$(tail -n 20 "$log_source" 2>/dev/null | \
    grep -oP 'loss[=:\s]+\K[\d.]+' | tail -n 1)

  if [[ -z "$loss" ]]; then
    loss=$(tail -n 20 "$log_source" 2>/dev/null | \
      grep -oP 'Loss[=:\s]+\K[\d.]+' | tail -n 1)
  fi

  echo "$loss"
}

# Function to get GPU stats
get_gpu_stats() {
  if ! command -v nvidia-smi &> /dev/null; then
    echo "N/A"
    return
  fi

  nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu \
    --format=csv,noheader,nounits 2>/dev/null || echo "N/A"
}

# Function to get memory stats
get_memory_stats() {
  if command -v free &> /dev/null; then
    free -m | awk 'NR==2{printf "%.1f%%", $3*100/$2}'
  else
    echo "N/A"
  fi
}

# Function to check for alerts
check_alerts() {
  local current_loss="$1"

  if [[ -z "$PREV_LOSS" || -z "$current_loss" ]]; then
    return
  fi

  # Calculate loss change
  local loss_change=$(awk -v curr="$current_loss" -v prev="$PREV_LOSS" \
    'BEGIN {printf "%.6f", (curr - prev) / prev}')

  # Check if loss increased significantly
  if (( $(awk -v change="$loss_change" -v thresh="$ALERT_THRESHOLD" \
    'BEGIN {print (change > thresh)}') )); then
    echo -e "${RED}⚠️  ALERT: Loss increased by ${loss_change}%${NC}"
    echo -e "${RED}   Previous: $PREV_LOSS → Current: $current_loss${NC}"

    ((ALERT_COUNT++))

    # Send email if configured
    if [[ -n "$ALERT_EMAIL" ]]; then
      echo "Loss spike detected: $PREV_LOSS → $current_loss" | \
        mail -s "ML Training Alert" "$ALERT_EMAIL" 2>/dev/null || true
    fi
  fi
}

# Function to log metrics
log_metrics() {
  local timestamp="$1"
  local loss="$2"
  local gpu_stats="$3"
  local mem_stats="$4"

  if [[ -n "$OUTPUT_FILE" ]]; then
    cat >> "$OUTPUT_FILE" << EOF
{"timestamp": $timestamp, "loss": $loss, "gpu": "$gpu_stats", "memory": "$mem_stats"}
EOF
  fi
}

# Function to display dashboard
display_dashboard() {
  local elapsed=$(($(date +%s) - START_TIME))
  local hours=$((elapsed / 3600))
  local minutes=$(((elapsed % 3600) / 60))
  local seconds=$((elapsed % 60))

  clear
  echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
  echo -e "${BLUE}║${NC}         ${GREEN}ML Training Monitor${NC} - Real-time Dashboard        ${BLUE}║${NC}"
  echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
  echo ""
  echo -e "${BLUE}Runtime:${NC} ${hours}h ${minutes}m ${seconds}s"
  echo -e "${BLUE}Alerts:${NC}  ${ALERT_COUNT}"
  echo ""
}

# Main monitoring loop
echo -e "${GREEN}🚀 Starting training monitor...${NC}"
echo -e "${BLUE}Log source:${NC} ${LOG_DIR:-$LOG_FILE}"
echo -e "${BLUE}Check interval:${NC} ${CHECK_INTERVAL}s"
echo -e "${BLUE}Alert threshold:${NC} ${ALERT_THRESHOLD}"
echo ""

while true; do
  # Determine log source
  if [[ -n "$LOG_DIR" ]]; then
    LOG_SOURCE=$(find "$LOG_DIR" -type f -name "*.log" -printf '%T@ %p\n' | \
      sort -rn | head -n1 | cut -d' ' -f2)
  else
    LOG_SOURCE="$LOG_FILE"
  fi

  if [[ ! -f "$LOG_SOURCE" ]]; then
    echo -e "${YELLOW}⏳ Waiting for log file: $LOG_SOURCE${NC}"
    sleep "$CHECK_INTERVAL"
    continue
  fi

  # Extract current metrics
  CURRENT_LOSS=$(extract_loss "$LOG_SOURCE")
  CURRENT_TIME=$(date +%s)

  # Get resource stats if enabled
  GPU_STATS=""
  MEM_STATS=""

  if [[ "$GPU_MONITOR" == true ]]; then
    GPU_STATS=$(get_gpu_stats)
  fi

  if [[ "$MEMORY_MONITOR" == true ]]; then
    MEM_STATS=$(get_memory_stats)
  fi

  # Display dashboard
  display_dashboard

  # Show current metrics
  if [[ -n "$CURRENT_LOSS" ]]; then
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}Current Loss:${NC} $CURRENT_LOSS"

    if [[ -n "$PREV_LOSS" ]]; then
      local change=$(awk -v curr="$CURRENT_LOSS" -v prev="$PREV_LOSS" \
        'BEGIN {printf "%.6f", curr - prev}')

      if (( $(awk -v c="$change" 'BEGIN {print (c < 0)}') )); then
        echo -e "${BLUE}Change:${NC} ${GREEN}$change ↓${NC}"
      else
        echo -e "${BLUE}Change:${NC} ${RED}+$change ↑${NC}"
      fi
    fi

    # Show resource stats
    if [[ -n "$GPU_STATS" && "$GPU_STATS" != "N/A" ]]; then
      IFS=',' read -r gpu_util gpu_mem_used gpu_mem_total gpu_temp <<< "$GPU_STATS"
      echo -e "${BLUE}GPU Util:${NC} ${gpu_util}%  ${BLUE}Mem:${NC} ${gpu_mem_used}MB/${gpu_mem_total}MB  ${BLUE}Temp:${NC} ${gpu_temp}°C"
    fi

    if [[ -n "$MEM_STATS" && "$MEM_STATS" != "N/A" ]]; then
      echo -e "${BLUE}System Memory:${NC} $MEM_STATS"
    fi

    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    # Check for alerts
    check_alerts "$CURRENT_LOSS"

    # Log metrics
    log_metrics "$CURRENT_TIME" "$CURRENT_LOSS" "$GPU_STATS" "$MEM_STATS"

    # Update history
    LOSS_HISTORY+=("$CURRENT_LOSS")
    PREV_LOSS="$CURRENT_LOSS"

    # Show recent trend
    if [[ ${#LOSS_HISTORY[@]} -ge 5 ]]; then
      echo ""
      echo -e "${BLUE}Recent Loss Trend (last 5):${NC}"
      echo -e "  ${LOSS_HISTORY[@]: -5}"
    fi
  else
    echo -e "${YELLOW}⏳ Waiting for loss metrics in log...${NC}"
  fi

  echo ""
  echo -e "${BLUE}Next check in ${CHECK_INTERVAL}s... (Ctrl+C to stop)${NC}"

  sleep "$CHECK_INTERVAL"
done
