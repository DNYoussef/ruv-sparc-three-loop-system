#!/bin/bash
###############################################################################
# PPTX Builder - Orchestrate full presentation generation workflow
# Combines slide generation, chart creation, and html2pptx conversion
###############################################################################

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMP_DIR="${TEMP_DIR:-/tmp/pptx-builder-$$}"
OUTPUT_DIR="${OUTPUT_DIR:-./output}"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Cleanup on exit
cleanup() {
    if [[ -d "$TEMP_DIR" ]]; then
        log_info "Cleaning up temporary directory: $TEMP_DIR"
        rm -rf "$TEMP_DIR"
    fi
}
trap cleanup EXIT

# Check dependencies
check_dependencies() {
    log_info "Checking dependencies..."

    local missing_deps=()

    # Python 3
    if ! command -v python3 &> /dev/null; then
        missing_deps+=("python3")
    fi

    # Node.js
    if ! command -v node &> /dev/null; then
        missing_deps+=("node")
    fi

    # html2pptx (Python package)
    if ! python3 -c "import html2pptx" 2> /dev/null; then
        missing_deps+=("html2pptx (pip install html2pptx)")
    fi

    # matplotlib (Python package)
    if ! python3 -c "import matplotlib" 2> /dev/null; then
        missing_deps+=("matplotlib (pip install matplotlib)")
    fi

    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        log_error "Missing dependencies: ${missing_deps[*]}"
        log_error "Install missing dependencies and try again"
        exit 1
    fi

    log_info "All dependencies satisfied"
}

# Initialize workspace
init_workspace() {
    log_info "Initializing workspace..."
    mkdir -p "$TEMP_DIR"
    mkdir -p "$OUTPUT_DIR"
    mkdir -p "$TEMP_DIR/charts"
    mkdir -p "$TEMP_DIR/slides"
}

# Generate presentation from specification
generate_presentation() {
    local spec_file="$1"
    local output_name="${2:-presentation}"

    log_info "Generating presentation from: $spec_file"

    # Validate spec file exists
    if [[ ! -f "$spec_file" ]]; then
        log_error "Specification file not found: $spec_file"
        exit 1
    fi

    # Parse specification (JSON or YAML)
    local spec_type="${spec_file##*.}"
    log_info "Specification format: $spec_type"

    # Step 1: Load template configuration
    log_info "Step 1/5: Loading template configuration..."
    local template_config="$TEMP_DIR/template.yaml"

    if [[ -f "${SCRIPT_DIR}/templates/slide-layout.yaml" ]]; then
        cp "${SCRIPT_DIR}/templates/slide-layout.yaml" "$template_config"
    else
        log_warn "Default template not found, using minimal config"
        cat > "$template_config" <<EOF
name: "Default Template"
colorScheme:
  primary: "#1E3A8A"
  secondary: "#3B82F6"
  accent: "#F59E0B"
  background: "#FFFFFF"
  text: "#1F2937"
typography:
  fontSize:
    h1: 36
    h2: 28
    body: 18
EOF
    fi

    # Validate template
    node "${SCRIPT_DIR}/template-engine.js" validate "$template_config"

    # Step 2: Generate charts (if data provided)
    log_info "Step 2/5: Generating charts..."
    local chart_count=0

    # Parse spec for chart data
    # TODO: Implement chart generation from spec
    # For now, skip if no charts

    log_info "Generated $chart_count charts"

    # Step 3: Generate slide HTML
    log_info "Step 3/5: Generating slide HTML..."
    local html_file="$TEMP_DIR/presentation.html"

    python3 "${SCRIPT_DIR}/slide-generator.py" \
        --spec "$spec_file" \
        --template "$template_config" \
        --output "$html_file"

    if [[ ! -f "$html_file" ]]; then
        log_error "Slide generation failed - no HTML output"
        exit 1
    fi

    # Step 4: Validate HTML
    log_info "Step 4/5: Validating HTML structure..."

    # Check for prohibited elements
    if grep -q "border:" "$html_file"; then
        log_warn "Detected border elements in HTML - review design constraints"
    fi

    if grep -q "border-radius:" "$html_file"; then
        log_warn "Detected rounded corners - violates design guidelines"
    fi

    # Validate font sizes
    local min_font_sizes=$(grep -oP 'font-size:\s*\K\d+(?=pt)' "$html_file" | sort -n | head -1)
    if [[ -n "$min_font_sizes" && "$min_font_sizes" -lt 18 ]]; then
        log_warn "Found font size below 18pt: ${min_font_sizes}pt"
    fi

    # Step 5: Convert to PPTX
    log_info "Step 5/5: Converting HTML to PPTX..."
    local pptx_file="$OUTPUT_DIR/${output_name}.pptx"

    python3 -m html2pptx "$html_file" "$pptx_file"

    if [[ ! -f "$pptx_file" ]]; then
        log_error "PPTX conversion failed"
        exit 1
    fi

    # Success
    log_info "Presentation generated successfully: $pptx_file"

    # Calculate file size
    local file_size=$(du -h "$pptx_file" | cut -f1)
    log_info "File size: $file_size"

    # Count slides (approximate from HTML)
    local slide_count=$(grep -c 'class="slide' "$html_file" || echo "0")
    log_info "Slide count: $slide_count"

    echo "$pptx_file"
}

# Batch generate multiple presentations
batch_generate() {
    local spec_dir="$1"

    log_info "Batch generating presentations from: $spec_dir"

    if [[ ! -d "$spec_dir" ]]; then
        log_error "Specification directory not found: $spec_dir"
        exit 1
    fi

    local generated=0
    local failed=0

    for spec_file in "$spec_dir"/*.{json,yaml,yml}; do
        if [[ -f "$spec_file" ]]; then
            local base_name=$(basename "$spec_file" | sed 's/\.[^.]*$//')
            log_info "Processing: $base_name"

            if generate_presentation "$spec_file" "$base_name"; then
                ((generated++))
            else
                ((failed++))
                log_error "Failed to generate: $base_name"
            fi
        fi
    done

    log_info "Batch complete - Generated: $generated, Failed: $failed"
}

# Watch mode for iterative development
watch_mode() {
    local spec_file="$1"
    local output_name="${2:-presentation}"

    log_info "Starting watch mode for: $spec_file"
    log_info "Press Ctrl+C to stop"

    # Initial generation
    generate_presentation "$spec_file" "$output_name"

    # Watch for changes (requires inotify-tools on Linux, fswatch on macOS)
    if command -v inotifywait &> /dev/null; then
        while inotifywait -e modify "$spec_file"; do
            log_info "Change detected, regenerating..."
            generate_presentation "$spec_file" "$output_name"
        done
    elif command -v fswatch &> /dev/null; then
        fswatch -o "$spec_file" | while read; do
            log_info "Change detected, regenerating..."
            generate_presentation "$spec_file" "$output_name"
        done
    else
        log_warn "Watch mode requires inotify-tools (Linux) or fswatch (macOS)"
        log_warn "Falling back to manual regeneration"
    fi
}

# Usage information
usage() {
    cat <<EOF
PPTX Builder v1.0 - Professional PowerPoint Generation

Usage:
    $0 generate <spec-file> [output-name]    Generate single presentation
    $0 batch <spec-dir>                      Generate multiple presentations
    $0 watch <spec-file> [output-name]       Watch mode for development
    $0 check                                 Check dependencies
    $0 help                                  Show this help

Examples:
    $0 generate quarterly-report.json
    $0 batch ./presentations/
    $0 watch board-deck.yaml board-deck

Environment Variables:
    TEMP_DIR      Temporary directory (default: /tmp/pptx-builder-$$)
    OUTPUT_DIR    Output directory (default: ./output)

Specification Format (JSON/YAML):
    {
      "title": "Presentation Title",
      "subtitle": "Subtitle",
      "slides": [
        {
          "type": "content",
          "title": "Slide Title",
          "bullets": ["Point 1", "Point 2", "Point 3"]
        }
      ],
      "template": "corporate"
    }

EOF
}

# Main command dispatcher
main() {
    local command="${1:-help}"

    case "$command" in
        generate)
            check_dependencies
            init_workspace
            generate_presentation "${2:-}" "${3:-presentation}"
            ;;

        batch)
            check_dependencies
            init_workspace
            batch_generate "${2:-.}"
            ;;

        watch)
            check_dependencies
            init_workspace
            watch_mode "${2:-}" "${3:-presentation}"
            ;;

        check)
            check_dependencies
            ;;

        help|--help|-h)
            usage
            ;;

        *)
            log_error "Unknown command: $command"
            usage
            exit 1
            ;;
    esac
}

main "$@"
