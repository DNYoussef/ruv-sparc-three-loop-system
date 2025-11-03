#!/bin/bash
set -e

#######################################################################
# Swarm Deployment Script
#
# Deploys multi-agent coordination swarm with specified topology.
# Handles initialization, agent spawning, and validation.
#
# Usage:
#   ./deploy_swarm.sh --topology mesh --agents 5 --strategy balanced
#   ./deploy_swarm.sh --config mesh-topology.yaml
#######################################################################

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
TOPOLOGY="mesh"
MAX_AGENTS=5
STRATEGY="balanced"
CONSENSUS=""
CONFIG_FILE=""
DRY_RUN=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --topology)
            TOPOLOGY="$2"
            shift 2
            ;;
        --agents)
            MAX_AGENTS="$2"
            shift 2
            ;;
        --strategy)
            STRATEGY="$2"
            shift 2
            ;;
        --consensus)
            CONSENSUS="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --topology <type>      Topology type (mesh|hierarchical|ring|star)"
            echo "  --agents <count>       Maximum number of agents"
            echo "  --strategy <type>      Distribution strategy (balanced|specialized|adaptive)"
            echo "  --consensus <type>     Consensus mechanism (byzantine|raft|gossip)"
            echo "  --config <file>        Load from YAML configuration file"
            echo "  --dry-run              Validate without deploying"
            echo "  --help                 Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Load from config file if specified
if [[ -n "$CONFIG_FILE" ]]; then
    echo -e "${YELLOW}Loading configuration from: $CONFIG_FILE${NC}"

    # Validate config file exists
    if [[ ! -f "$CONFIG_FILE" ]]; then
        echo -e "${RED}Error: Configuration file not found: $CONFIG_FILE${NC}"
        exit 1
    fi

    # Validate config file (requires Python script)
    if command -v python3 &> /dev/null; then
        SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        python3 "$SCRIPT_DIR/validate_topology.py" "$CONFIG_FILE" || {
            echo -e "${RED}Configuration validation failed${NC}"
            exit 1
        }
    fi

    # Extract values from YAML (simple grep approach)
    TOPOLOGY=$(grep "^topology:" "$CONFIG_FILE" | awk '{print $2}')
    MAX_AGENTS=$(grep "^maxAgents:" "$CONFIG_FILE" | awk '{print $2}')
    STRATEGY=$(grep "^strategy:" "$CONFIG_FILE" | awk '{print $2}')
    CONSENSUS=$(grep "^consensus:" "$CONFIG_FILE" | awk '{print $2}')
fi

# Validate inputs
validate_inputs() {
    local valid_topologies=("mesh" "hierarchical" "ring" "star")
    local valid_strategies=("balanced" "specialized" "adaptive")

    # Check topology
    if [[ ! " ${valid_topologies[@]} " =~ " ${TOPOLOGY} " ]]; then
        echo -e "${RED}Invalid topology: $TOPOLOGY${NC}"
        echo "Valid options: ${valid_topologies[*]}"
        exit 1
    fi

    # Check agent count
    if [[ $MAX_AGENTS -lt 2 ]]; then
        echo -e "${RED}maxAgents must be at least 2${NC}"
        exit 1
    fi

    # Check strategy
    if [[ ! " ${valid_strategies[@]} " =~ " ${STRATEGY} " ]]; then
        echo -e "${RED}Invalid strategy: $STRATEGY${NC}"
        echo "Valid options: ${valid_strategies[*]}"
        exit 1
    fi
}

# Print deployment summary
print_summary() {
    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║         Swarm Deployment Configuration                     ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
    echo "  Topology:       $TOPOLOGY"
    echo "  Max Agents:     $MAX_AGENTS"
    echo "  Strategy:       $STRATEGY"
    [[ -n "$CONSENSUS" ]] && echo "  Consensus:      $CONSENSUS"
    echo ""
}

# Initialize swarm
initialize_swarm() {
    echo -e "${YELLOW}Initializing swarm with $TOPOLOGY topology...${NC}"

    if [[ "$DRY_RUN" == true ]]; then
        echo "[DRY RUN] Would execute: npx claude-flow@alpha swarm init --topology $TOPOLOGY --max-agents $MAX_AGENTS --strategy $STRATEGY"
        return 0
    fi

    # Check if claude-flow is installed
    if ! command -v npx &> /dev/null; then
        echo -e "${RED}Error: npx not found. Please install Node.js${NC}"
        exit 1
    fi

    # Initialize swarm
    npx claude-flow@alpha swarm init \
        --topology "$TOPOLOGY" \
        --max-agents "$MAX_AGENTS" \
        --strategy "$STRATEGY" || {
        echo -e "${RED}Swarm initialization failed${NC}"
        exit 1
    }

    echo -e "${GREEN}✓ Swarm initialized successfully${NC}"
}

# Spawn initial agents
spawn_agents() {
    local agent_count=$((MAX_AGENTS < 5 ? MAX_AGENTS : 5))
    echo -e "${YELLOW}Spawning $agent_count initial agents...${NC}"

    if [[ "$DRY_RUN" == true ]]; then
        echo "[DRY RUN] Would spawn $agent_count agents"
        return 0
    fi

    local agent_types=("researcher" "coder" "analyst" "optimizer" "coordinator")

    for i in $(seq 1 "$agent_count"); do
        local agent_type="${agent_types[$((i-1))]}"
        echo "  Spawning $agent_type agent..."

        npx claude-flow@alpha agent spawn --type "$agent_type" || {
            echo -e "${RED}Failed to spawn $agent_type agent${NC}"
            exit 1
        }
    done

    echo -e "${GREEN}✓ Agents spawned successfully${NC}"
}

# Verify deployment
verify_deployment() {
    echo -e "${YELLOW}Verifying deployment...${NC}"

    if [[ "$DRY_RUN" == true ]]; then
        echo "[DRY RUN] Would verify swarm status"
        return 0
    fi

    npx claude-flow@alpha swarm status --verbose || {
        echo -e "${RED}Deployment verification failed${NC}"
        exit 1
    }

    echo -e "${GREEN}✓ Deployment verified${NC}"
}

# Main execution
main() {
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "          Advanced Coordination - Swarm Deployment          "
    echo "════════════════════════════════════════════════════════════"

    validate_inputs
    print_summary

    if [[ "$DRY_RUN" == true ]]; then
        echo -e "${YELLOW}Running in DRY RUN mode - no changes will be made${NC}"
        echo ""
    fi

    initialize_swarm
    spawn_agents
    verify_deployment

    echo ""
    echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║      Swarm Deployment Completed Successfully               ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Monitor swarm: npx claude-flow@alpha swarm monitor"
    echo "  2. Orchestrate task: npx claude-flow@alpha task orchestrate --task 'your-task'"
    echo "  3. Check status: npx claude-flow@alpha swarm status"
    echo ""
}

main
