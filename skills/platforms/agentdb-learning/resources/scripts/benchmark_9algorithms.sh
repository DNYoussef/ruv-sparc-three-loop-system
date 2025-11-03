#!/bin/bash
# Benchmark all 9 AgentDB RL algorithms
# Compares training time, convergence, and memory usage

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/../../benchmarks"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Algorithms to benchmark
ALGORITHMS=(
    "q-learning"
    "sarsa"
    "actor-critic"
    "decision-transformer"
    "active-learning"
    "adversarial"
    "curriculum"
    "federated"
    "multi-task"
)

# Training configurations
EPISODES=50
QUICK_MODE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--episodes)
            EPISODES="$2"
            shift 2
            ;;
        -q|--quick)
            QUICK_MODE=true
            EPISODES=10
            shift
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -e, --episodes N    Number of episodes (default: 50)"
            echo "  -q, --quick         Quick mode (10 episodes)"
            echo "  -o, --output DIR    Output directory for results"
            echo "  -h, --help          Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"
RESULTS_FILE="$OUTPUT_DIR/benchmark_${TIMESTAMP}.json"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}AgentDB RL Algorithm Benchmark${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Episodes: ${YELLOW}$EPISODES${NC}"
echo -e "Output: ${YELLOW}$RESULTS_FILE${NC}"
echo -e "Quick Mode: ${YELLOW}$QUICK_MODE${NC}"
echo ""

# Initialize results JSON
echo "{" > "$RESULTS_FILE"
echo "  \"timestamp\": \"$TIMESTAMP\"," >> "$RESULTS_FILE"
echo "  \"episodes\": $EPISODES," >> "$RESULTS_FILE"
echo "  \"results\": {" >> "$RESULTS_FILE"

FIRST=true

# Benchmark each algorithm
for algo in "${ALGORITHMS[@]}"; do
    echo -e "${GREEN}Benchmarking: $algo${NC}"
    echo "----------------------------------------"

    # Start timing
    START_TIME=$(date +%s.%N)

    # Run training
    MODEL_FILE="$OUTPUT_DIR/${algo}_model_${TIMESTAMP}.json"

    if python3 "$SCRIPT_DIR/train_rl_agent.py" \
        --algorithm "$algo" \
        --episodes "$EPISODES" \
        --save "$MODEL_FILE" > "$OUTPUT_DIR/${algo}_output_${TIMESTAMP}.log" 2>&1; then

        # Calculate training time
        END_TIME=$(date +%s.%N)
        DURATION=$(echo "$END_TIME - $START_TIME" | bc)

        # Extract metrics from output
        AVG_REWARD=$(grep "Average Reward:" "$OUTPUT_DIR/${algo}_output_${TIMESTAMP}.log" | awk '{print $3}')
        FINAL_LOSS=$(grep "Final Loss:" "$OUTPUT_DIR/${algo}_output_${TIMESTAMP}.log" | awk '{print $3}')

        # Get memory usage (if available)
        MEMORY_MB=$(cat "$OUTPUT_DIR/${algo}_output_${TIMESTAMP}.log" | grep -o "Memory: [0-9.]*" | awk '{print $2}' || echo "N/A")

        echo -e "${GREEN}✓ Completed in ${DURATION}s${NC}"
        echo -e "  Avg Reward: $AVG_REWARD"
        echo -e "  Final Loss: $FINAL_LOSS"
        echo ""

        # Add to results JSON
        if [ "$FIRST" = false ]; then
            echo "," >> "$RESULTS_FILE"
        fi
        FIRST=false

        cat >> "$RESULTS_FILE" << EOF
    "$algo": {
      "status": "success",
      "training_time_seconds": $DURATION,
      "avg_reward": ${AVG_REWARD:-0.0},
      "final_loss": ${FINAL_LOSS:-0.0},
      "memory_mb": "${MEMORY_MB}",
      "model_file": "$MODEL_FILE"
    }
EOF
    else
        echo -e "${RED}✗ Failed${NC}"
        echo ""

        if [ "$FIRST" = false ]; then
            echo "," >> "$RESULTS_FILE"
        fi
        FIRST=false

        cat >> "$RESULTS_FILE" << EOF
    "$algo": {
      "status": "failed",
      "error": "Training failed - check logs"
    }
EOF
    fi
done

# Close results JSON
echo "" >> "$RESULTS_FILE"
echo "  }" >> "$RESULTS_FILE"
echo "}" >> "$RESULTS_FILE"

# Generate summary
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Benchmark Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Display results table
echo -e "${YELLOW}Algorithm             Time(s)  Avg Reward  Final Loss  Status${NC}"
echo "----------------------------------------------------------------"

for algo in "${ALGORITHMS[@]}"; do
    STATUS=$(jq -r ".results[\"$algo\"].status" "$RESULTS_FILE")

    if [ "$STATUS" = "success" ]; then
        TIME=$(jq -r ".results[\"$algo\"].training_time_seconds" "$RESULTS_FILE")
        REWARD=$(jq -r ".results[\"$algo\"].avg_reward" "$RESULTS_FILE")
        LOSS=$(jq -r ".results[\"$algo\"].final_loss" "$RESULTS_FILE")

        printf "%-20s  %7.2f  %10.2f  %10.6f  ${GREEN}✓${NC}\n" "$algo" "$TIME" "$REWARD" "$LOSS"
    else
        printf "%-20s  ${RED}FAILED${NC}\n" "$algo"
    fi
done

echo ""
echo -e "Full results: ${YELLOW}$RESULTS_FILE${NC}"
echo -e "Log files: ${YELLOW}$OUTPUT_DIR/*_output_${TIMESTAMP}.log${NC}"
echo ""

# Find best algorithm by avg reward
BEST_ALGO=$(jq -r '.results | to_entries | max_by(.value.avg_reward // 0) | .key' "$RESULTS_FILE")
BEST_REWARD=$(jq -r ".results[\"$BEST_ALGO\"].avg_reward" "$RESULTS_FILE")

echo -e "${GREEN}Best Algorithm:${NC} $BEST_ALGO (Avg Reward: $BEST_REWARD)"
echo ""
