#!/bin/bash
# Code Formatter - Auto-format Python and TypeScript code.
#
# Runs black for Python, Prettier for TypeScript, and organizes imports.
# Generates formatting report with file counts and changes.
#
# Usage:
#   bash code-formatter.sh --all
#   bash code-formatter.sh --python
#   bash code-formatter.sh --typescript

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Counters
PYTHON_FILES=0
TS_FILES=0
FORMATTED_FILES=0

# Format Python files with black and isort
format_python() {
    echo -e "${YELLOW}🐍 Formatting Python files...${NC}"

    # Find all Python files
    PYTHON_FILES=$(find . -name "*.py" -not -path "*/\.*" -not -path "*/node_modules/*" | wc -l)

    if [ "$PYTHON_FILES" -eq 0 ]; then
        echo -e "${YELLOW}⚠️  No Python files found${NC}"
        return
    fi

    echo "Found $PYTHON_FILES Python files"

    # Run black formatter
    if command -v black &> /dev/null; then
        black . --exclude="(node_modules|\.venv|\.git)" 2>&1 | tee /tmp/black_output.txt
        FORMATTED=$(grep -c "reformatted" /tmp/black_output.txt || echo "0")
        FORMATTED_FILES=$((FORMATTED_FILES + FORMATTED))
        echo -e "${GREEN}✅ Black formatting complete${NC}"
    else
        echo -e "${RED}❌ black not found, skipping${NC}"
    fi

    # Run isort for import organization
    if command -v isort &> /dev/null; then
        isort . --skip-gitignore 2>&1
        echo -e "${GREEN}✅ Import organization complete${NC}"
    else
        echo -e "${RED}❌ isort not found, skipping${NC}"
    fi

    echo ""
}

# Format TypeScript files with Prettier
format_typescript() {
    echo -e "${YELLOW}🔷 Formatting TypeScript files...${NC}"

    # Find all TypeScript files
    TS_FILES=$(find . -name "*.ts" -o -name "*.tsx" -not -path "*/node_modules/*" -not -path "*/\.*" | wc -l)

    if [ "$TS_FILES" -eq 0 ]; then
        echo -e "${YELLOW}⚠️  No TypeScript files found${NC}"
        return
    fi

    echo "Found $TS_FILES TypeScript files"

    # Check if Prettier is available
    if command -v npx &> /dev/null; then
        npx prettier --write "**/*.{ts,tsx,json}" --ignore-path .gitignore 2>&1 | tee /tmp/prettier_output.txt
        echo -e "${GREEN}✅ Prettier formatting complete${NC}"
    else
        echo -e "${RED}❌ npx not found, skipping TypeScript formatting${NC}"
    fi

    echo ""
}

# Generate formatting report
generate_report() {
    echo -e "${GREEN}📊 Formatting Report${NC}"
    echo "===================="
    echo "Python files: $PYTHON_FILES"
    echo "TypeScript files: $TS_FILES"
    echo "Files formatted: $FORMATTED_FILES"
    echo "===================="

    if [ "$FORMATTED_FILES" -gt 0 ]; then
        echo -e "${YELLOW}⚠️  $FORMATTED_FILES file(s) were reformatted${NC}"
        echo "💡 Review changes before committing"
    else
        echo -e "${GREEN}✅ All files already properly formatted${NC}"
    fi
}

# Main script
main() {
    MODE="${1:---all}"

    echo "🚀 Code Formatter"
    echo ""

    case "$MODE" in
        --all)
            format_python
            format_typescript
            ;;
        --python)
            format_python
            ;;
        --typescript)
            format_typescript
            ;;
        *)
            echo "Usage: $0 [--all|--python|--typescript]"
            exit 1
            ;;
    esac

    generate_report
}

main "$@"
