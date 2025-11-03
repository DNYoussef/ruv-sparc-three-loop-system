#!/bin/bash
# Agent Generation Script
# Generates a complete agent specification from template
# Usage: ./generate_agent.sh <agent-name> <category> [--interactive]

set -euo pipefail

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATES_DIR="${SCRIPT_DIR}/../templates"
OUTPUT_DIR="${OUTPUT_DIR:-.}"

# Error handling
error() {
    echo -e "${RED}ERROR: $1${NC}" >&2
    exit 1
}

warning() {
    echo -e "${YELLOW}WARNING: $1${NC}" >&2
}

info() {
    echo -e "${BLUE}INFO: $1${NC}"
}

success() {
    echo -e "${GREEN}SUCCESS: $1${NC}"
}

# Usage information
usage() {
    cat <<EOF
Usage: $0 <agent-name> <category> [options]

Arguments:
    agent-name      Name of the agent (kebab-case, e.g., python-specialist)
    category        Agent category: specialist, coordinator, hybrid, research,
                    development, testing, documentation, security

Options:
    --interactive   Interactive mode with prompts
    --output DIR    Output directory (default: current directory)
    --force         Overwrite existing files
    --help          Show this help message

Examples:
    $0 python-specialist specialist
    $0 backend-coordinator coordinator --interactive
    $0 ml-researcher research --output ./agents
EOF
}

# Validate agent name format
validate_name() {
    local name="$1"
    if [[ ! "$name" =~ ^[a-z0-9-]+$ ]]; then
        error "Invalid agent name. Use kebab-case (lowercase, hyphens only)"
    fi
}

# Validate category
validate_category() {
    local category="$1"
    local valid_categories=("specialist" "coordinator" "hybrid" "research" "development" "testing" "documentation" "security")

    for valid in "${valid_categories[@]}"; do
        if [[ "$category" == "$valid" ]]; then
            return 0
        fi
    done

    error "Invalid category. Must be one of: ${valid_categories[*]}"
}

# Interactive prompts
interactive_mode() {
    local agent_name="$1"
    local category="$2"

    info "Interactive Agent Generation for: $agent_name ($category)"
    echo

    # Description
    echo -n "Agent Description (80-150 words): "
    read -r description

    # Expertise areas
    echo -n "Expertise Areas (comma-separated): "
    read -r expertise
    IFS=',' read -ra expertise_array <<< "$expertise"

    # Primary capabilities
    echo -n "Primary Capabilities (comma-separated): "
    read -r capabilities
    IFS=',' read -ra capabilities_array <<< "$capabilities"

    # Prompting techniques
    info "Available prompting techniques:"
    echo "  1. chain-of-thought"
    echo "  2. few-shot"
    echo "  3. role-based"
    echo "  4. plan-and-solve"
    echo "  5. self-consistency"
    echo -n "Select techniques (e.g., 1,2,3): "
    read -r techniques

    # Export for template substitution
    export AGENT_DESCRIPTION="$description"
    export AGENT_EXPERTISE="${expertise_array[*]}"
    export AGENT_CAPABILITIES="${capabilities_array[*]}"
    export AGENT_TECHNIQUES="$techniques"
}

# Generate agent specification
generate_agent() {
    local agent_name="$1"
    local category="$2"
    local interactive="${3:-false}"

    info "Generating agent: $agent_name (Category: $category)"

    # Check if template exists
    local template_file="${TEMPLATES_DIR}/agent-spec.yaml"
    if [[ ! -f "$template_file" ]]; then
        error "Template file not found: $template_file"
    fi

    # Create output directory
    local output_path="${OUTPUT_DIR}/${agent_name}"
    if [[ -d "$output_path" ]] && [[ "${FORCE:-false}" != "true" ]]; then
        error "Agent directory already exists: $output_path (use --force to overwrite)"
    fi

    mkdir -p "$output_path"

    # Interactive mode
    if [[ "$interactive" == "true" ]]; then
        interactive_mode "$agent_name" "$category"
    fi

    # Generate YAML specification
    local output_file="${output_path}/agent-spec.yaml"

    info "Generating specification: $output_file"

    # Template substitution
    sed -e "s/{{AGENT_NAME}}/${agent_name}/g" \
        -e "s/{{CATEGORY}}/${category}/g" \
        -e "s/{{DESCRIPTION}}/${AGENT_DESCRIPTION:-Specialist agent for ${agent_name} domain}/g" \
        "$template_file" > "$output_file"

    # Generate capabilities JSON
    local capabilities_file="${output_path}/capabilities.json"
    info "Generating capabilities: $capabilities_file"

    cp "${TEMPLATES_DIR}/capabilities.json" "$capabilities_file"

    # Validate generated specification
    if command -v python3 &> /dev/null; then
        info "Validating generated specification..."
        if python3 "${SCRIPT_DIR}/validate_agent.py" "$output_file" --json > /dev/null 2>&1; then
            success "Validation passed!"
        else
            warning "Validation warnings found. Review with: python3 ${SCRIPT_DIR}/validate_agent.py $output_file"
        fi
    else
        warning "Python3 not found. Skipping validation."
    fi

    # Create README
    cat > "${output_path}/README.md" <<EOF
# ${agent_name}

Agent specification for ${agent_name} (Category: ${category})

## Files

- \`agent-spec.yaml\` - Complete agent specification
- \`capabilities.json\` - Structured capabilities definition
- \`README.md\` - This file

## Usage

### Validate Specification
\`\`\`bash
python3 ../../resources/scripts/validate_agent.py agent-spec.yaml
\`\`\`

### Deploy Agent
\`\`\`bash
# Copy to Claude-Flow agents directory
cp agent-spec.yaml ~/.claude-flow/agents/${agent_name}.yaml
\`\`\`

## Generated
$(date)
EOF

    success "Agent generated successfully at: $output_path"
    info "Next steps:"
    echo "  1. Review and customize: $output_file"
    echo "  2. Validate: python3 ${SCRIPT_DIR}/validate_agent.py $output_file"
    echo "  3. Test with example inputs"
}

# Parse arguments
INTERACTIVE=false
FORCE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --interactive)
            INTERACTIVE=true
            shift
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --force)
            FORCE=true
            shift
            ;;
        --help)
            usage
            exit 0
            ;;
        -*)
            error "Unknown option: $1"
            ;;
        *)
            if [[ -z "${AGENT_NAME:-}" ]]; then
                AGENT_NAME="$1"
            elif [[ -z "${CATEGORY:-}" ]]; then
                CATEGORY="$1"
            else
                error "Unexpected argument: $1"
            fi
            shift
            ;;
    esac
done

# Validate required arguments
if [[ -z "${AGENT_NAME:-}" ]] || [[ -z "${CATEGORY:-}" ]]; then
    usage
    exit 1
fi

# Validate inputs
validate_name "$AGENT_NAME"
validate_category "$CATEGORY"

# Generate agent
generate_agent "$AGENT_NAME" "$CATEGORY" "$INTERACTIVE"
