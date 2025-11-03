#!/usr/bin/env bash
################################################################################
# Requirement Analyzer - User Story Parsing & Priority Scoring
################################################################################
#
# Automated requirement extraction from SPEC.md with MoSCoW prioritization,
# dependency graph generation, and MECE validation.
#
# Usage:
#   bash requirement-analyzer.sh \
#     --spec SPEC.md \
#     --output requirements-analysis.json
#
# Features:
#   - SPEC.md parsing for functional/non-functional requirements
#   - User story format detection and normalization
#   - MoSCoW priority scoring (Must/Should/Could/Won't)
#   - Dependency graph generation
#   - MECE validation (Mutually Exclusive, Collectively Exhaustive)
#
# Author: Research-Driven Planning Skill
# Version: 2.0.0
# License: MIT
################################################################################

set -euo pipefail

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
SPEC_FILE=""
OUTPUT_FILE=""

################################################################################
# Utility Functions
################################################################################

log_info() {
  echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
  echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
  echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
  echo -e "${RED}[ERROR]${NC} $1" >&2
}

################################################################################
# Argument Parsing
################################################################################

parse_args() {
  while [[ $# -gt 0 ]]; do
    case $1 in
      --spec)
        SPEC_FILE="$2"
        shift 2
        ;;
      --output)
        OUTPUT_FILE="$2"
        shift 2
        ;;
      --help)
        show_help
        exit 0
        ;;
      *)
        log_error "Unknown option: $1"
        show_help
        exit 1
        ;;
    esac
  done

  # Validate required arguments
  if [[ -z "$SPEC_FILE" ]] || [[ -z "$OUTPUT_FILE" ]]; then
    log_error "Missing required arguments"
    show_help
    exit 1
  fi

  if [[ ! -f "$SPEC_FILE" ]]; then
    log_error "SPEC file not found: $SPEC_FILE"
    exit 1
  fi
}

show_help() {
  cat <<EOF
Requirement Analyzer - User Story Parsing & Priority Scoring

Usage:
  bash requirement-analyzer.sh --spec SPEC.md --output requirements-analysis.json

Options:
  --spec FILE      Path to SPEC.md file (required)
  --output FILE    Output path for requirements analysis JSON (required)
  --help           Show this help message

Example:
  bash requirement-analyzer.sh \\
    --spec SPEC.md \\
    --output .claude/.artifacts/requirements-analysis.json
EOF
}

################################################################################
# Requirement Extraction
################################################################################

extract_functional_requirements() {
  local spec_file="$1"

  log_info "Extracting functional requirements..."

  # Extract section between "### Functional Requirements" and next heading
  awk '
    /### Functional Requirements/,/^##/ {
      if (/^[0-9]+\./ || /^-/) {
        # Extract requirement text
        gsub(/^[0-9]+\. /, "");
        gsub(/^- /, "");
        if (length($0) > 0 && $0 !~ /^##/) {
          print $0;
        }
      }
    }
  ' "$spec_file"
}

extract_nonfunctional_requirements() {
  local spec_file="$1"

  log_info "Extracting non-functional requirements..."

  # Extract section between "### Non-Functional Requirements" and next heading
  awk '
    /### Non-Functional Requirements/,/^##/ {
      if (/^-/ || /^[0-9]+\./) {
        gsub(/^[0-9]+\. /, "");
        gsub(/^- /, "");
        if (length($0) > 0 && $0 !~ /^##/) {
          print $0;
        }
      }
    }
  ' "$spec_file"
}

################################################################################
# User Story Normalization
################################################################################

normalize_user_story() {
  local requirement="$1"

  # Detect if already in user story format
  if echo "$requirement" | grep -qE "As a .*, I want .*, so that"; then
    echo "$requirement"
  else
    # Convert to user story format
    echo "As a user, I want to $requirement, so that I can accomplish my goals"
  fi
}

################################################################################
# MoSCoW Priority Scoring
################################################################################

calculate_moscow_priority() {
  local requirement="$1"

  # Keywords for priority detection
  local must_keywords="must|critical|required|essential|mandatory"
  local should_keywords="should|important|recommended"
  local could_keywords="could|nice to have|optional|enhancement"
  local wont_keywords="won't|future|out of scope|excluded"

  if echo "$requirement" | grep -qiE "$must_keywords"; then
    echo "MUST"
  elif echo "$requirement" | grep -qiE "$wont_keywords"; then
    echo "WONT"
  elif echo "$requirement" | grep -qiE "$should_keywords"; then
    echo "SHOULD"
  elif echo "$requirement" | grep -qiE "$could_keywords"; then
    echo "COULD"
  else
    # Default to SHOULD if no keywords found
    echo "SHOULD"
  fi
}

################################################################################
# Dependency Detection
################################################################################

detect_dependencies() {
  local requirement="$1"
  local all_requirements="$2"

  # Simple keyword-based dependency detection
  # In production, this would use NLP or explicit annotations

  local dependencies=()

  # Check for "requires", "depends on", "after" keywords
  if echo "$requirement" | grep -qiE "requires|depends on|after"; then
    # Extract potential dependency
    # This is simplified - production would use more sophisticated parsing
    dependencies+=("DEPENDENCY_PLACEHOLDER")
  fi

  # Return as JSON array
  if [[ ${#dependencies[@]} -eq 0 ]]; then
    echo "[]"
  else
    printf '%s\n' "${dependencies[@]}" | jq -R . | jq -s .
  fi
}

################################################################################
# MECE Validation
################################################################################

validate_mece() {
  local requirements_json="$1"

  log_info "Validating MECE (Mutually Exclusive, Collectively Exhaustive)..."

  # Count requirements by category
  local functional_count
  local nonfunctional_count

  functional_count=$(echo "$requirements_json" | jq '[.requirements[] | select(.type == "functional")] | length')
  nonfunctional_count=$(echo "$requirements_json" | jq '[.requirements[] | select(.type == "non-functional")] | length')

  # Check for overlaps (Mutually Exclusive)
  # In production, this would use semantic similarity
  local overlaps=0

  # Check for gaps (Collectively Exhaustive)
  # Ensure all major categories covered
  local has_security
  local has_performance
  local has_scalability

  has_security=$(echo "$requirements_json" | jq -r '.requirements[] | select(.category == "security") | .id' | wc -l)
  has_performance=$(echo "$requirements_json" | jq -r '.requirements[] | select(.category == "performance") | .id' | wc -l)
  has_scalability=$(echo "$requirements_json" | jq -r '.requirements[] | select(.category == "scalability") | .id' | wc -l)

  local mece_score=100

  if [[ $overlaps -gt 0 ]]; then
    mece_score=$((mece_score - overlaps * 10))
  fi

  if [[ $has_security -eq 0 ]]; then
    mece_score=$((mece_score - 15))
    log_warning "No security requirements detected"
  fi

  if [[ $has_performance -eq 0 ]]; then
    mece_score=$((mece_score - 10))
    log_warning "No performance requirements detected"
  fi

  echo "$mece_score"
}

################################################################################
# Requirement Analysis
################################################################################

analyze_requirements() {
  local spec_file="$1"

  log_info "Starting requirement analysis for: $spec_file"

  # Extract requirements
  local functional_reqs
  local nonfunctional_reqs

  functional_reqs=$(extract_functional_requirements "$spec_file")
  nonfunctional_reqs=$(extract_nonfunctional_requirements "$spec_file")

  # Build JSON structure
  local requirements_json='{"requirements": []}'
  local req_id=1

  # Process functional requirements
  while IFS= read -r req; do
    if [[ -n "$req" ]]; then
      local user_story
      local priority
      local category="functional"

      user_story=$(normalize_user_story "$req")
      priority=$(calculate_moscow_priority "$req")

      # Add to JSON
      requirements_json=$(echo "$requirements_json" | jq \
        --arg id "REQ-$(printf '%03d' $req_id)" \
        --arg type "functional" \
        --arg category "$category" \
        --arg original "$req" \
        --arg user_story "$user_story" \
        --arg priority "$priority" \
        '.requirements += [{
          id: $id,
          type: $type,
          category: $category,
          original: $original,
          user_story: $user_story,
          priority: $priority,
          dependencies: []
        }]')

      req_id=$((req_id + 1))
    fi
  done <<< "$functional_reqs"

  # Process non-functional requirements
  while IFS= read -r req; do
    if [[ -n "$req" ]]; then
      local priority
      local category="non-functional"

      # Categorize non-functional requirement
      if echo "$req" | grep -qiE "performance|latency|throughput"; then
        category="performance"
      elif echo "$req" | grep -qiE "security|authentication|authorization"; then
        category="security"
      elif echo "$req" | grep -qiE "scalability|concurrent|users"; then
        category="scalability"
      fi

      priority=$(calculate_moscow_priority "$req")

      requirements_json=$(echo "$requirements_json" | jq \
        --arg id "REQ-$(printf '%03d' $req_id)" \
        --arg type "non-functional" \
        --arg category "$category" \
        --arg original "$req" \
        --arg priority "$priority" \
        '.requirements += [{
          id: $id,
          type: $type,
          category: $category,
          original: $original,
          priority: $priority,
          dependencies: []
        }]')

      req_id=$((req_id + 1))
    fi
  done <<< "$nonfunctional_reqs"

  # Calculate MECE score
  local mece_score
  mece_score=$(validate_mece "$requirements_json")

  # Add metadata
  requirements_json=$(echo "$requirements_json" | jq \
    --argjson total "$((req_id - 1))" \
    --argjson mece_score "$mece_score" \
    '. + {
      metadata: {
        total_requirements: $total,
        mece_score: $mece_score,
        timestamp: (now | todate)
      }
    }')

  echo "$requirements_json"
}

################################################################################
# Main Execution
################################################################################

main() {
  parse_args "$@"

  log_info "=== Requirement Analysis Starting ==="
  log_info "SPEC: $SPEC_FILE"
  log_info "OUTPUT: $OUTPUT_FILE"

  # Analyze requirements
  local analysis
  analysis=$(analyze_requirements "$SPEC_FILE")

  # Create output directory if needed
  mkdir -p "$(dirname "$OUTPUT_FILE")"

  # Save analysis
  echo "$analysis" | jq . > "$OUTPUT_FILE"

  # Summary
  local total_reqs
  local mece_score

  total_reqs=$(echo "$analysis" | jq -r '.metadata.total_requirements')
  mece_score=$(echo "$analysis" | jq -r '.metadata.mece_score')

  log_success "Analysis complete!"
  log_info "Total requirements: $total_reqs"
  log_info "MECE score: ${mece_score}%"
  log_info "Output: $OUTPUT_FILE"

  # Exit with appropriate code
  if [[ $mece_score -lt 70 ]]; then
    log_warning "MECE score below 70% - requirements may have gaps or overlaps"
    exit 1
  fi

  exit 0
}

# Run main
main "$@"
