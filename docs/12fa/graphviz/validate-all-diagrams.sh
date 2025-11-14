#!/bin/bash

# Graphviz Validation and Rendering Script
# Phase 3: Validation & Integration
# Date: 2025-11-01

set -e  # Exit on error

echo "============================================"
echo "Phase 3: Graphviz Validation & Rendering"
echo "============================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Counters
TOTAL=0
SUCCESS=0
FAILED=0
SKIPPED=0

# Base directory
BASE_DIR="C:/Users/17175/docs/12fa/graphviz"
cd "$BASE_DIR"

# Check if graphviz is installed
if ! command -v dot &> /dev/null; then
    echo -e "${RED}ERROR: Graphviz 'dot' command not found${NC}"
    echo "Please install Graphviz:"
    echo "  Ubuntu/Debian: sudo apt-get install graphviz"
    echo "  macOS: brew install graphviz"
    echo "  Windows: choco install graphviz"
    echo ""
    echo "Skipping SVG generation, but will validate syntax..."
    GRAPHVIZ_INSTALLED=false
else
    echo -e "${GREEN}✓ Graphviz found: $(dot -V 2>&1)${NC}"
    GRAPHVIZ_INSTALLED=true
fi

echo ""

# Validation function
validate_and_render() {
    local category=$1
    local dir=$2

    echo "----------------------------------------"
    echo "Processing: $category"
    echo "----------------------------------------"

    if [ ! -d "$dir" ]; then
        echo -e "${YELLOW}⚠ Directory not found: $dir${NC}"
        return
    fi

    cd "$dir"
    local count=$(ls -1 *-process.dot 2>/dev/null | wc -l)

    if [ $count -eq 0 ]; then
        echo -e "${YELLOW}⚠ No .dot files found in $dir${NC}"
        cd "$BASE_DIR"
        return
    fi

    echo "Found: $count diagram(s)"

    for dotfile in *-process.dot; do
        TOTAL=$((TOTAL + 1))
        local basename="${dotfile%.dot}"

        # Syntax validation (always run)
        if dot -Tsvg "$dotfile" -o /dev/null 2>/dev/null; then
            echo -e "${GREEN}✓${NC} $dotfile - Valid syntax"

            # Render to SVG if Graphviz installed
            if [ "$GRAPHVIZ_INSTALLED" = true ]; then
                if dot -Tsvg "$dotfile" -o "$basename.svg" 2>/dev/null; then
                    SUCCESS=$((SUCCESS + 1))
                    # Also generate PNG
                    dot -Tpng "$dotfile" -o "$basename.png" 2>/dev/null || true
                else
                    echo -e "${RED}✗${NC} $dotfile - SVG generation failed"
                    FAILED=$((FAILED + 1))
                fi
            else
                SKIPPED=$((SKIPPED + 1))
            fi
        else
            echo -e "${RED}✗${NC} $dotfile - Invalid syntax"
            FAILED=$((FAILED + 1))
        fi
    done

    echo ""
    cd "$BASE_DIR"
}

# Process all categories
validate_and_render "Skills" "skills"
validate_and_render "Agents" "agents"
validate_and_render "Commands" "commands"

# Summary
echo "============================================"
echo "Validation Summary"
echo "============================================"
echo "Total diagrams: $TOTAL"
echo -e "${GREEN}✓ Valid: $SUCCESS${NC}"
echo -e "${RED}✗ Failed: $FAILED${NC}"
if [ "$GRAPHVIZ_INSTALLED" = false ]; then
    echo -e "${YELLOW}⚠ Skipped (no Graphviz): $SKIPPED${NC}"
fi
echo ""

# Calculate success rate
if [ $TOTAL -gt 0 ]; then
    SUCCESS_RATE=$(echo "scale=1; $SUCCESS * 100 / $TOTAL" | bc)
    echo "Success Rate: $SUCCESS_RATE%"
fi

echo ""
echo "============================================"
echo "Output Files"
echo "============================================"
if [ "$GRAPHVIZ_INSTALLED" = true ]; then
    echo "SVG files: $(find . -name "*.svg" | wc -l)"
    echo "PNG files: $(find . -name "*.png" | wc -l)"
else
    echo "Graphviz not installed - no renders generated"
fi

echo ""
echo "Done!"
