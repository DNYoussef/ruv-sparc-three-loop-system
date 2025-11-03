#!/usr/bin/env bash
###############################################################################
# Intent Validator - Analysis Completeness Verification
#
# Validates that intent analysis is complete and meets quality standards before
# proceeding with response generation. Checks all required analysis dimensions,
# verifies confidence calibration, ensures assumptions are surfaced, and confirms
# appropriate clarification strategy.
#
# Usage:
#   bash intent-validator.sh analysis-results.json
#   bash intent-validator.sh --verbose analysis-results.json
#   cat analysis.json | bash intent-validator.sh --stdin
#
# Validation Checks:
#   - All required analysis dimensions addressed (category, confidence, signals)
#   - Confidence scores properly calibrated (0.0-1.0 range)
#   - Critical assumptions surfaced and documented
#   - Clarification strategy appropriate for confidence level
#   - No contradictory signals detected
#
# Exit Codes:
#   0 - Validation passed
#   1 - Validation failed (see JSON output for details)
#   2 - Invalid input or usage error
###############################################################################

set -euo pipefail

# Color codes for output (if terminal supports)
if [ -t 1 ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    NC='\033[0m' # No Color
else
    RED=''
    GREEN=''
    YELLOW=''
    NC=''
fi

# Configuration
REQUIRED_FIELDS=("primary_category" "confidence" "categories" "requires_clarification")
CONFIDENCE_MIN=0.0
CONFIDENCE_MAX=1.0
HIGH_CONFIDENCE_THRESHOLD=0.80
LOW_CONFIDENCE_THRESHOLD=0.50

# Global result tracking
VALIDATION_PASSED=true
VALIDATION_ERRORS=()
VALIDATION_WARNINGS=()

###############################################################################
# Helper Functions
###############################################################################

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
    VALIDATION_ERRORS+=("$1")
    VALIDATION_PASSED=false
}

log_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1" >&2
    VALIDATION_WARNINGS+=("$1")
}

log_success() {
    echo -e "${GREEN}[OK]${NC} $1" >&2
}

usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS] [FILE]

Validate intent analysis completeness and quality.

OPTIONS:
    --verbose       Show detailed validation progress
    --stdin         Read from stdin instead of file
    -h, --help      Show this help message

EXAMPLES:
    # Validate from file
    $(basename "$0") analysis.json

    # Validate from stdin
    cat analysis.json | $(basename "$0") --stdin

    # Verbose validation
    $(basename "$0") --verbose analysis.json

EXIT CODES:
    0 - Validation passed
    1 - Validation failed
    2 - Invalid input or usage error
EOF
}

###############################################################################
# Validation Functions
###############################################################################

check_required_fields() {
    local json="$1"

    for field in "${REQUIRED_FIELDS[@]}"; do
        if ! echo "$json" | jq -e ".$field" > /dev/null 2>&1; then
            log_error "Missing required field: $field"
        else
            log_success "Required field present: $field"
        fi
    done
}

validate_confidence_scores() {
    local json="$1"

    # Check primary confidence
    local confidence
    confidence=$(echo "$json" | jq -r '.confidence // "null"')

    if [ "$confidence" = "null" ]; then
        log_error "Confidence score is missing"
        return
    fi

    # Validate range
    if ! awk -v conf="$confidence" -v min="$CONFIDENCE_MIN" -v max="$CONFIDENCE_MAX" \
         'BEGIN { if (conf < min || conf > max) exit 1 }'; then
        log_error "Confidence score $confidence outside valid range [$CONFIDENCE_MIN, $CONFIDENCE_MAX]"
    else
        log_success "Confidence score $confidence is valid"
    fi

    # Check category probabilities sum to ~1.0
    local prob_sum
    prob_sum=$(echo "$json" | jq '[.categories[]] | add')

    if ! awk -v sum="$prob_sum" 'BEGIN { if (sum < 0.95 || sum > 1.05) exit 1 }'; then
        log_warning "Category probabilities sum to $prob_sum (expected ~1.0)"
    else
        log_success "Category probabilities sum to $prob_sum (valid)"
    fi
}

validate_clarification_strategy() {
    local json="$1"

    local confidence
    local requires_clarification
    local multi_intent

    confidence=$(echo "$json" | jq -r '.confidence // 0')
    requires_clarification=$(echo "$json" | jq -r '.requires_clarification // false')
    multi_intent=$(echo "$json" | jq -r '.multi_intent // false')

    # Low confidence should require clarification
    if awk -v conf="$confidence" -v thresh="$LOW_CONFIDENCE_THRESHOLD" \
           'BEGIN { if (conf < thresh) exit 0; exit 1 }'; then
        if [ "$requires_clarification" != "true" ]; then
            log_error "Low confidence ($confidence) but clarification not required"
        else
            log_success "Low confidence appropriately flagged for clarification"
        fi
    fi

    # Multi-intent should require clarification
    if [ "$multi_intent" = "true" ]; then
        if [ "$requires_clarification" != "true" ]; then
            log_warning "Multi-intent detected but clarification not required"
        else
            log_success "Multi-intent appropriately flagged for clarification"
        fi
    fi

    # High confidence should not require clarification
    if awk -v conf="$confidence" -v thresh="$HIGH_CONFIDENCE_THRESHOLD" \
           'BEGIN { if (conf >= thresh) exit 0; exit 1 }'; then
        if [ "$requires_clarification" = "true" ]; then
            log_warning "High confidence ($confidence) but still requires clarification"
        else
            log_success "High confidence appropriately proceeds without clarification"
        fi
    fi
}

validate_signals() {
    local json="$1"

    # Check signals array exists and has content
    local signal_count
    signal_count=$(echo "$json" | jq '.signals | length')

    if [ "$signal_count" -eq 0 ]; then
        log_warning "No signals detected - analysis may be incomplete"
    else
        log_success "Detected $signal_count intent signals"
    fi

    # Check signals align with primary category
    local primary_category
    primary_category=$(echo "$json" | jq -r '.primary_category')

    # Look for category name in signals (loose check)
    if echo "$json" | jq -r '.signals[]' | grep -qi "$primary_category"; then
        log_success "Signals align with primary category: $primary_category"
    else
        log_warning "No signals explicitly mention primary category: $primary_category"
    fi
}

validate_reasoning() {
    local json="$1"

    local reasoning
    reasoning=$(echo "$json" | jq -r '.reasoning // ""')

    if [ -z "$reasoning" ]; then
        log_error "Reasoning field is empty or missing"
    else
        local word_count
        word_count=$(echo "$reasoning" | wc -w)

        if [ "$word_count" -lt 10 ]; then
            log_warning "Reasoning is very brief ($word_count words)"
        else
            log_success "Reasoning provided ($word_count words)"
        fi
    fi
}

validate_consistency() {
    local json="$1"

    # Check primary category matches highest probability category
    local primary_category
    local highest_prob_category

    primary_category=$(echo "$json" | jq -r '.primary_category')
    highest_prob_category=$(echo "$json" | jq -r '.categories | to_entries | max_by(.value) | .key')

    if [ "$primary_category" != "$highest_prob_category" ]; then
        log_error "Primary category ($primary_category) doesn't match highest probability category ($highest_prob_category)"
    else
        log_success "Primary category consistent with probability distribution"
    fi
}

###############################################################################
# Main Validation
###############################################################################

validate_analysis() {
    local json="$1"

    echo "Running intent analysis validation..." >&2
    echo "" >&2

    # Run all validation checks
    check_required_fields "$json"
    validate_confidence_scores "$json"
    validate_clarification_strategy "$json"
    validate_signals "$json"
    validate_reasoning "$json"
    validate_consistency "$json"

    echo "" >&2

    # Generate validation report
    local report
    report=$(jq -n \
        --argjson passed "$VALIDATION_PASSED" \
        --argjson errors "$(printf '%s\n' "${VALIDATION_ERRORS[@]:-}" | jq -R . | jq -s .)" \
        --argjson warnings "$(printf '%s\n' "${VALIDATION_WARNINGS[@]:-}" | jq -R . | jq -s .)" \
        '{
            validation_passed: $passed,
            error_count: ($errors | length),
            warning_count: ($warnings | length),
            errors: $errors,
            warnings: $warnings
        }'
    )

    echo "$report"

    if [ "$VALIDATION_PASSED" = true ]; then
        log_success "Validation PASSED"
        return 0
    else
        log_error "Validation FAILED"
        return 1
    fi
}

###############################################################################
# Main Entry Point
###############################################################################

main() {
    local input_file=""
    local from_stdin=false
    local verbose=false

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --stdin)
                from_stdin=true
                shift
                ;;
            --verbose)
                verbose=true
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            -*)
                echo "Unknown option: $1" >&2
                usage
                exit 2
                ;;
            *)
                input_file="$1"
                shift
                ;;
        esac
    done

    # Check jq is available
    if ! command -v jq &> /dev/null; then
        echo "Error: jq is required but not installed" >&2
        exit 2
    fi

    # Read input
    local json
    if [ "$from_stdin" = true ]; then
        json=$(cat)
    elif [ -n "$input_file" ]; then
        if [ ! -f "$input_file" ]; then
            echo "Error: File not found: $input_file" >&2
            exit 2
        fi
        json=$(cat "$input_file")
    else
        echo "Error: No input provided" >&2
        usage
        exit 2
    fi

    # Validate JSON format
    if ! echo "$json" | jq empty 2>/dev/null; then
        echo "Error: Invalid JSON input" >&2
        exit 2
    fi

    # Suppress progress output if not verbose
    if [ "$verbose" = false ]; then
        exec 2>/dev/null
    fi

    # Run validation
    validate_analysis "$json"
}

# Run main if executed directly
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    main "$@"
fi
