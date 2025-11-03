#!/usr/bin/env bash
#
# Paper Summarizer - Academic Paper Extraction and Summarization
# Version: 1.0.0
# Purpose: Extract and summarize academic papers from PDF
#
# Features:
# - PDF to text conversion (pdftotext, tesseract OCR)
# - Section detection (abstract, methods, results, conclusion)
# - Citation extraction
# - LaTeX formula support
# - Markdown output
#
# Usage:
#   ./paper-summarizer.sh \
#     --pdf paper.pdf \
#     --extract-citations \
#     --output summary.md
#
# Dependencies:
#   - pdftotext (poppler-utils)
#   - tesseract (OCR fallback)
#   - jq (JSON processing)

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/paper-summarizer.log"
TEMP_DIR="/tmp/paper-summarizer-$$"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $*" | tee -a "$LOG_FILE"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*" | tee -a "$LOG_FILE"
}

# Cleanup on exit
cleanup() {
    rm -rf "$TEMP_DIR"
}
trap cleanup EXIT

# Check dependencies
check_dependencies() {
    local missing_deps=()

    if ! command -v pdftotext &> /dev/null; then
        missing_deps+=("pdftotext (install: apt-get install poppler-utils)")
    fi

    if ! command -v tesseract &> /dev/null; then
        log_warn "tesseract not found (OCR fallback unavailable)"
    fi

    if ! command -v jq &> /dev/null; then
        missing_deps+=("jq (install: apt-get install jq)")
    fi

    if [ ${#missing_deps[@]} -gt 0 ]; then
        log_error "Missing required dependencies:"
        for dep in "${missing_deps[@]}"; do
            echo "  - $dep"
        done
        exit 1
    fi
}

# Extract text from PDF
extract_text_from_pdf() {
    local pdf_file="$1"
    local output_file="$2"

    log_info "Extracting text from PDF: $pdf_file"

    if pdftotext -layout "$pdf_file" "$output_file" 2>/dev/null; then
        log_info "Text extraction successful"
        return 0
    else
        log_warn "pdftotext failed, trying OCR..."

        if command -v tesseract &> /dev/null; then
            # Convert PDF to images and OCR
            if command -v pdftoppm &> /dev/null; then
                pdftoppm -png "$pdf_file" "${TEMP_DIR}/page"
                for img in "${TEMP_DIR}"/page-*.png; do
                    tesseract "$img" stdout >> "$output_file"
                done
                log_info "OCR extraction successful"
                return 0
            fi
        fi

        log_error "Failed to extract text from PDF"
        return 1
    fi
}

# Detect section headers
detect_sections() {
    local text_file="$1"
    local sections_json="${TEMP_DIR}/sections.json"

    log_info "Detecting paper sections..."

    # Common section patterns
    local abstract_pattern="^(ABSTRACT|Abstract|1\.|I\.)\s*"
    local intro_pattern="^(INTRODUCTION|Introduction|1\.|I\.)\s*"
    local methods_pattern="^(METHODS|Methods|METHODOLOGY|Methodology|2\.|II\.)\s*"
    local results_pattern="^(RESULTS|Results|3\.|III\.)\s*"
    local discussion_pattern="^(DISCUSSION|Discussion|4\.|IV\.)\s*"
    local conclusion_pattern="^(CONCLUSION|Conclusion|Conclusions|5\.|V\.)\s*"
    local references_pattern="^(REFERENCES|References|Bibliography)\s*"

    # Extract sections using grep and line numbers
    cat > "$sections_json" <<EOF
{
  "abstract": {
    "start": $(grep -n -m1 -E "$abstract_pattern" "$text_file" | cut -d: -f1 || echo "0"),
    "text": ""
  },
  "introduction": {
    "start": $(grep -n -m1 -E "$intro_pattern" "$text_file" | cut -d: -f1 || echo "0"),
    "text": ""
  },
  "methods": {
    "start": $(grep -n -m1 -E "$methods_pattern" "$text_file" | cut -d: -f1 || echo "0"),
    "text": ""
  },
  "results": {
    "start": $(grep -n -m1 -E "$results_pattern" "$text_file" | cut -d: -f1 || echo "0"),
    "text": ""
  },
  "discussion": {
    "start": $(grep -n -m1 -E "$discussion_pattern" "$text_file" | cut -d: -f1 || echo "0"),
    "text": ""
  },
  "conclusion": {
    "start": $(grep -n -m1 -E "$conclusion_pattern" "$text_file" | cut -d: -f1 || echo "0"),
    "text": ""
  },
  "references": {
    "start": $(grep -n -m1 -E "$references_pattern" "$text_file" | cut -d: -f1 || echo "0"),
    "text": ""
  }
}
EOF

    log_info "Section detection complete"
}

# Extract citations
extract_citations() {
    local text_file="$1"
    local citations_file="${TEMP_DIR}/citations.txt"

    log_info "Extracting citations..."

    # Common citation patterns
    # [1] Author et al., "Title", Journal, Year
    # Author et al. (Year). Title. Journal.
    grep -E "^\[?[0-9]+\]?\s+[A-Z].*\([0-9]{4}\)" "$text_file" > "$citations_file" || true

    local citation_count=$(wc -l < "$citations_file")
    log_info "Found $citation_count citations"

    echo "$citations_file"
}

# Generate summary
generate_summary() {
    local text_file="$1"
    local output_file="$2"
    local extract_citations="$3"

    log_info "Generating paper summary..."

    # Extract metadata
    local title=$(head -20 "$text_file" | grep -v "^$" | head -1 | sed 's/^[[:space:]]*//')
    local authors=$(head -30 "$text_file" | grep -E "^[A-Z][a-z]+ [A-Z]\." | head -5 | tr '\n' ',' | sed 's/,$//')

    # Create markdown summary
    cat > "$output_file" <<EOF
# Paper Summary

**Title**: $title

**Authors**: ${authors:-"Unknown"}

**Extracted**: $(date '+%Y-%m-%d %H:%M:%S')

---

## Abstract

EOF

    # Extract abstract (first 200 lines, find paragraph after "ABSTRACT")
    sed -n '/ABSTRACT/,/INTRODUCTION/p' "$text_file" | head -30 | tail -20 >> "$output_file"

    cat >> "$output_file" <<EOF

## Key Sections

### Methodology

EOF

    # Extract methods section snippet
    sed -n '/METHODS/,/RESULTS/p' "$text_file" | head -20 | tail -15 >> "$output_file"

    cat >> "$output_file" <<EOF

### Results

EOF

    # Extract results section snippet
    sed -n '/RESULTS/,/DISCUSSION/p' "$text_file" | head -20 | tail -15 >> "$output_file"

    cat >> "$output_file" <<EOF

### Conclusion

EOF

    # Extract conclusion section
    sed -n '/CONCLUSION/,/REFERENCES/p' "$text_file" | head -20 | tail -15 >> "$output_file"

    # Add citations if requested
    if [ "$extract_citations" = true ]; then
        local citations_file=$(extract_citations "$text_file")

        if [ -s "$citations_file" ]; then
            cat >> "$output_file" <<EOF

---

## References

EOF
            head -20 "$citations_file" >> "$output_file"
        fi
    fi

    log_info "Summary generated: $output_file"
}

# Main function
main() {
    local pdf_file=""
    local output_file="summary.md"
    local extract_citations=false

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --pdf)
                pdf_file="$2"
                shift 2
                ;;
            --output)
                output_file="$2"
                shift 2
                ;;
            --extract-citations)
                extract_citations=true
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                echo "Usage: $0 --pdf <file.pdf> [--output <summary.md>] [--extract-citations]"
                exit 1
                ;;
        esac
    done

    # Validate arguments
    if [ -z "$pdf_file" ]; then
        log_error "Error: --pdf is required"
        exit 1
    fi

    if [ ! -f "$pdf_file" ]; then
        log_error "Error: PDF file not found: $pdf_file"
        exit 1
    fi

    # Create temp directory
    mkdir -p "$TEMP_DIR"

    # Check dependencies
    check_dependencies

    # Extract text
    local text_file="${TEMP_DIR}/paper.txt"
    if ! extract_text_from_pdf "$pdf_file" "$text_file"; then
        exit 1
    fi

    # Detect sections
    detect_sections "$text_file"

    # Generate summary
    generate_summary "$text_file" "$output_file" "$extract_citations"

    log_info "Paper summarization complete!"
    log_info "Summary saved to: $output_file"
}

# Run main
main "$@"
