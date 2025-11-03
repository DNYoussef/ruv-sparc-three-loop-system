#!/bin/bash

# Decision Matrix Generator for Interactive Planning
# Helps prioritize features and make trade-off decisions based on multiple criteria
# Usage: ./decision-matrix.sh --criteria criteria.yaml --options options.yaml --output matrix.md

set -euo pipefail

# Default values
CRITERIA_FILE=""
OPTIONS_FILE=""
OUTPUT_FILE="decision-matrix.md"
FORMAT="markdown"
WEIGHTS_ENABLED=false

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Usage information
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Generate decision matrix for comparing options against multiple criteria.

OPTIONS:
    -c, --criteria FILE     YAML file with evaluation criteria
    -o, --options FILE      YAML file with options to evaluate
    --output FILE           Output file path (default: decision-matrix.md)
    --format FORMAT         Output format: markdown, csv, json (default: markdown)
    --weights               Enable weighted scoring
    -h, --help              Show this help message

EXAMPLE:
    $0 --criteria criteria.yaml --options features.yaml --output matrix.md --weights

CRITERIA FILE FORMAT:
    criteria:
      - name: "Development Time"
        weight: 0.3
        scale: "1-5 (1=fastest, 5=slowest)"
      - name: "User Impact"
        weight: 0.4
        scale: "1-5 (1=low, 5=high)"

OPTIONS FILE FORMAT:
    options:
      - name: "Real-time Collaboration"
        scores:
          "Development Time": 4
          "User Impact": 5
      - name: "File Upload"
        scores:
          "Development Time": 2
          "User Impact": 3
EOF
    exit 1
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -c|--criteria)
                CRITERIA_FILE="$2"
                shift 2
                ;;
            -o|--options)
                OPTIONS_FILE="$2"
                shift 2
                ;;
            --output)
                OUTPUT_FILE="$2"
                shift 2
                ;;
            --format)
                FORMAT="$2"
                shift 2
                ;;
            --weights)
                WEIGHTS_ENABLED=true
                shift
                ;;
            -h|--help)
                usage
                ;;
            *)
                echo -e "${RED}Error: Unknown option $1${NC}"
                usage
                ;;
        esac
    done

    # Validate required arguments
    if [[ -z "$CRITERIA_FILE" ]] || [[ -z "$OPTIONS_FILE" ]]; then
        echo -e "${RED}Error: Both --criteria and --options are required${NC}"
        usage
    fi

    if [[ ! -f "$CRITERIA_FILE" ]]; then
        echo -e "${RED}Error: Criteria file not found: $CRITERIA_FILE${NC}"
        exit 1
    fi

    if [[ ! -f "$OPTIONS_FILE" ]]; then
        echo -e "${RED}Error: Options file not found: $OPTIONS_FILE${NC}"
        exit 1
    fi
}

# Check dependencies
check_dependencies() {
    local missing_deps=()

    # Check for yq (YAML processor)
    if ! command -v yq &> /dev/null; then
        missing_deps+=("yq")
    fi

    # Check for jq (JSON processor)
    if ! command -v jq &> /dev/null; then
        missing_deps+=("jq")
    fi

    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        echo -e "${RED}Error: Missing required dependencies: ${missing_deps[*]}${NC}"
        echo -e "${YELLOW}Install with: brew install yq jq (macOS) or apt-get install yq jq (Linux)${NC}"
        exit 1
    fi
}

# Parse YAML criteria file
parse_criteria() {
    echo -e "${BLUE}Parsing criteria from $CRITERIA_FILE...${NC}"

    # Extract criteria using yq
    yq eval '.criteria[] | .name' "$CRITERIA_FILE" > /tmp/criteria_names.txt

    if $WEIGHTS_ENABLED; then
        yq eval '.criteria[] | .weight' "$CRITERIA_FILE" > /tmp/criteria_weights.txt
    fi

    local count=$(wc -l < /tmp/criteria_names.txt)
    echo -e "${GREEN}Found $count criteria${NC}"
}

# Parse YAML options file
parse_options() {
    echo -e "${BLUE}Parsing options from $OPTIONS_FILE...${NC}"

    # Extract option names
    yq eval '.options[] | .name' "$OPTIONS_FILE" > /tmp/option_names.txt

    local count=$(wc -l < /tmp/option_names.txt)
    echo -e "${GREEN}Found $count options${NC}"
}

# Calculate weighted score for an option
calculate_weighted_score() {
    local option_name="$1"
    local total_score=0
    local total_weight=0

    # Read criteria
    while IFS= read -r criterion; do
        # Get score for this criterion
        local score=$(yq eval ".options[] | select(.name == \"$option_name\") | .scores.\"$criterion\"" "$OPTIONS_FILE")

        # Get weight for this criterion
        local weight=1
        if $WEIGHTS_ENABLED; then
            weight=$(yq eval ".criteria[] | select(.name == \"$criterion\") | .weight" "$CRITERIA_FILE")
        fi

        # Calculate weighted score
        if [[ -n "$score" ]] && [[ "$score" != "null" ]]; then
            total_score=$(echo "$total_score + ($score * $weight)" | bc -l)
            total_weight=$(echo "$total_weight + $weight" | bc -l)
        fi
    done < /tmp/criteria_names.txt

    # Calculate average
    if (( $(echo "$total_weight > 0" | bc -l) )); then
        echo "scale=2; $total_score / $total_weight" | bc -l
    else
        echo "0"
    fi
}

# Generate markdown decision matrix
generate_markdown() {
    echo -e "${BLUE}Generating markdown decision matrix...${NC}"

    {
        echo "# Decision Matrix"
        echo ""
        echo "**Generated**: $(date -u +"%Y-%m-%d %H:%M:%S UTC")"
        echo ""

        # Table header
        echo -n "| Option |"
        while IFS= read -r criterion; do
            echo -n " $criterion |"
        done < /tmp/criteria_names.txt

        if $WEIGHTS_ENABLED; then
            echo -n " Weighted Score |"
        fi
        echo ""

        # Table separator
        echo -n "|--------|"
        while IFS= read -r criterion; do
            echo -n "--------|"
        done < /tmp/criteria_names.txt

        if $WEIGHTS_ENABLED; then
            echo -n "----------------|"
        fi
        echo ""

        # Table rows (options)
        while IFS= read -r option; do
            echo -n "| **$option** |"

            # Scores for each criterion
            while IFS= read -r criterion; do
                score=$(yq eval ".options[] | select(.name == \"$option\") | .scores.\"$criterion\"" "$OPTIONS_FILE")
                if [[ "$score" == "null" ]] || [[ -z "$score" ]]; then
                    score="—"
                fi
                echo -n " $score |"
            done < /tmp/criteria_names.txt

            # Weighted score
            if $WEIGHTS_ENABLED; then
                weighted_score=$(calculate_weighted_score "$option")
                echo -n " **$weighted_score** |"
            fi
            echo ""
        done < /tmp/option_names.txt

        echo ""

        # Criteria weights (if enabled)
        if $WEIGHTS_ENABLED; then
            echo "## Criteria Weights"
            echo ""
            paste /tmp/criteria_names.txt /tmp/criteria_weights.txt | while IFS=$'\t' read -r name weight; do
                echo "- **$name**: $weight"
            done
            echo ""
        fi

        # Scoring scale
        echo "## Scoring Scale"
        echo ""
        while IFS= read -r criterion; do
            scale=$(yq eval ".criteria[] | select(.name == \"$criterion\") | .scale" "$CRITERIA_FILE")
            echo "- **$criterion**: $scale"
        done < /tmp/criteria_names.txt

    } > "$OUTPUT_FILE"

    echo -e "${GREEN}Decision matrix written to $OUTPUT_FILE${NC}"
}

# Generate CSV decision matrix
generate_csv() {
    echo -e "${BLUE}Generating CSV decision matrix...${NC}"

    {
        # Header
        echo -n "Option,"
        while IFS= read -r criterion; do
            echo -n "\"$criterion\","
        done < /tmp/criteria_names.txt

        if $WEIGHTS_ENABLED; then
            echo -n "Weighted Score"
        fi
        echo ""

        # Rows
        while IFS= read -r option; do
            echo -n "\"$option\","

            while IFS= read -r criterion; do
                score=$(yq eval ".options[] | select(.name == \"$option\") | .scores.\"$criterion\"" "$OPTIONS_FILE")
                if [[ "$score" == "null" ]] || [[ -z "$score" ]]; then
                    score=""
                fi
                echo -n "$score,"
            done < /tmp/criteria_names.txt

            if $WEIGHTS_ENABLED; then
                weighted_score=$(calculate_weighted_score "$option")
                echo -n "$weighted_score"
            fi
            echo ""
        done < /tmp/option_names.txt

    } > "$OUTPUT_FILE"

    echo -e "${GREEN}CSV decision matrix written to $OUTPUT_FILE${NC}"
}

# Generate JSON decision matrix
generate_json() {
    echo -e "${BLUE}Generating JSON decision matrix...${NC}"

    local json='{"criteria":[],"options":[],"generated":"'$(date -u +"%Y-%m-%dT%H:%M:%SZ")'"}'

    # Add criteria
    while IFS= read -r criterion; do
        local weight=1
        if $WEIGHTS_ENABLED; then
            weight=$(yq eval ".criteria[] | select(.name == \"$criterion\") | .weight" "$CRITERIA_FILE")
        fi
        local scale=$(yq eval ".criteria[] | select(.name == \"$criterion\") | .scale" "$CRITERIA_FILE")

        json=$(echo "$json" | jq --arg name "$criterion" --arg weight "$weight" --arg scale "$scale" \
            '.criteria += [{"name": $name, "weight": ($weight | tonumber), "scale": $scale}]')
    done < /tmp/criteria_names.txt

    # Add options
    while IFS= read -r option; do
        local option_json='{"name":"'$option'","scores":{}}'

        while IFS= read -r criterion; do
            score=$(yq eval ".options[] | select(.name == \"$option\") | .scores.\"$criterion\"" "$OPTIONS_FILE")
            if [[ "$score" != "null" ]] && [[ -n "$score" ]]; then
                option_json=$(echo "$option_json" | jq --arg criterion "$criterion" --arg score "$score" \
                    '.scores[$criterion] = ($score | tonumber)')
            fi
        done < /tmp/criteria_names.txt

        if $WEIGHTS_ENABLED; then
            weighted_score=$(calculate_weighted_score "$option")
            option_json=$(echo "$option_json" | jq --arg score "$weighted_score" \
                '.weightedScore = ($score | tonumber)')
        fi

        json=$(echo "$json" | jq --argjson opt "$option_json" '.options += [$opt]')
    done < /tmp/option_names.txt

    echo "$json" | jq '.' > "$OUTPUT_FILE"

    echo -e "${GREEN}JSON decision matrix written to $OUTPUT_FILE${NC}"
}

# Main execution
main() {
    parse_args "$@"
    check_dependencies
    parse_criteria
    parse_options

    case "$FORMAT" in
        markdown)
            generate_markdown
            ;;
        csv)
            generate_csv
            ;;
        json)
            generate_json
            ;;
        *)
            echo -e "${RED}Error: Unknown format $FORMAT${NC}"
            usage
            ;;
    esac

    # Cleanup temp files
    rm -f /tmp/criteria_names.txt /tmp/criteria_weights.txt /tmp/option_names.txt

    echo -e "${GREEN}Done!${NC}"
}

# Run main function
main "$@"
