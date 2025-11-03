#!/bin/bash
# Fast Linter - Parallel linting with ESLint, Prettier, and custom rules
# Part of quick-quality-check Enhanced tier resources

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${CONFIG_FILE:-${SCRIPT_DIR}/check-config.yaml}"
OUTPUT_FORMAT="${OUTPUT_FORMAT:-json}"
PARALLEL_JOBS="${PARALLEL_JOBS:-4}"

# Color codes for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $*" >&2
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*" >&2
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
}

# Parse command line arguments
TARGET_PATH="${1:-.}"
OUTPUT_FILE="${2:-/dev/stdout}"

# Validate target path
if [[ ! -e "$TARGET_PATH" ]]; then
    log_error "Target path does not exist: $TARGET_PATH"
    exit 1
fi

# Function to run ESLint
run_eslint() {
    local target="$1"
    local output_file="$2"

    log_info "Running ESLint on $target..."

    if command -v eslint &> /dev/null; then
        eslint "$target" \
            --format json \
            --max-warnings 0 \
            --cache \
            --cache-location .eslintcache \
            --output-file "$output_file" \
            2>&1 || true
    else
        log_warn "ESLint not found, skipping..."
        echo '[]' > "$output_file"
    fi
}

# Function to run Prettier check
run_prettier() {
    local target="$1"
    local output_file="$2"

    log_info "Running Prettier check on $target..."

    if command -v prettier &> /dev/null; then
        prettier "$target" \
            --check \
            --list-different \
            --ignore-unknown \
            2>&1 | tee "$output_file" || true
    else
        log_warn "Prettier not found, skipping..."
        echo "" > "$output_file"
    fi
}

# Function to run custom style checks
run_custom_checks() {
    local target="$1"
    local output_file="$2"

    log_info "Running custom style checks on $target..."

    local issues=()

    # Check for console.log statements
    if grep -r "console\.log" "$target" 2>/dev/null | grep -v "node_modules" > /dev/null; then
        issues+=("Found console.log statements (should use logger)")
    fi

    # Check for TODO comments
    if grep -r "TODO\|FIXME\|XXX" "$target" 2>/dev/null | grep -v "node_modules" > /dev/null; then
        issues+=("Found TODO/FIXME comments")
    fi

    # Check for long lines (>120 chars)
    if find "$target" -type f -name "*.js" -o -name "*.ts" 2>/dev/null | \
       xargs awk 'length > 120 {print FILENAME":"NR; exit 1}' 2>/dev/null; then
        issues+=("Found lines longer than 120 characters")
    fi

    # Output issues as JSON
    if [[ ${#issues[@]} -gt 0 ]]; then
        printf '%s\n' "${issues[@]}" | jq -R -s 'split("\n") | map(select(length > 0))' > "$output_file"
    else
        echo '[]' > "$output_file"
    fi
}

# Function to aggregate results
aggregate_results() {
    local eslint_file="$1"
    local prettier_file="$2"
    local custom_file="$3"
    local output_file="$4"

    log_info "Aggregating lint results..."

    # Combine all results into single JSON
    jq -n \
        --argfile eslint "$eslint_file" \
        --arg prettier "$(cat "$prettier_file")" \
        --argfile custom "$custom_file" \
        '{
            eslint: $eslint,
            prettier: ($prettier | split("\n") | map(select(length > 0))),
            custom: $custom,
            summary: {
                total_issues: (
                    ($eslint | length) +
                    ($prettier | split("\n") | map(select(length > 0)) | length) +
                    ($custom | length)
                ),
                eslint_errors: ($eslint | map(select(.severity == 2)) | length),
                eslint_warnings: ($eslint | map(select(.severity == 1)) | length),
                prettier_violations: ($prettier | split("\n") | map(select(length > 0)) | length),
                custom_issues: ($custom | length)
            }
        }' > "$output_file"
}

# Main execution
main() {
    log_info "Starting fast linting for: $TARGET_PATH"

    # Create temp directory for intermediate results
    TEMP_DIR=$(mktemp -d)
    trap "rm -rf $TEMP_DIR" EXIT

    # Run checks in parallel
    run_eslint "$TARGET_PATH" "$TEMP_DIR/eslint.json" &
    PID_ESLINT=$!

    run_prettier "$TARGET_PATH" "$TEMP_DIR/prettier.txt" &
    PID_PRETTIER=$!

    run_custom_checks "$TARGET_PATH" "$TEMP_DIR/custom.json" &
    PID_CUSTOM=$!

    # Wait for all checks to complete
    wait $PID_ESLINT
    wait $PID_PRETTIER
    wait $PID_CUSTOM

    # Aggregate results
    aggregate_results \
        "$TEMP_DIR/eslint.json" \
        "$TEMP_DIR/prettier.txt" \
        "$TEMP_DIR/custom.json" \
        "$OUTPUT_FILE"

    # Calculate exit code based on severity
    local total_errors=$(jq '.summary.eslint_errors' "$OUTPUT_FILE")

    if [[ $total_errors -gt 0 ]]; then
        log_error "Linting failed with $total_errors errors"
        exit 1
    else
        log_info "Linting completed successfully"
        exit 0
    fi
}

# Run main function
main
