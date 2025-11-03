#!/bin/bash
# Validate Agent System Prompt Quality
# Checks for evidence-based prompting patterns, structure quality, and completeness

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
PROMPT_FILE=""
VERBOSE=0
MIN_SCORE=70

# Usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS] <prompt-file>

Validate agent system prompt quality against evidence-based prompting standards.

OPTIONS:
    -v, --verbose       Verbose output with detailed analysis
    -s, --min-score N   Minimum passing score (default: 70)
    -h, --help          Show this help message

EXAMPLE:
    $0 marketing-specialist-base-prompt-v1.md
    $0 -v -s 80 backend-dev-enhanced-prompt-v2.md

VALIDATION CHECKS:
    1. Core Identity Section (required)
    2. Universal Commands (required)
    3. Specialist Commands (recommended)
    4. MCP Server Tools (required)
    5. Cognitive Framework (required)
    6. Guardrails (required)
    7. Success Criteria (required)
    8. Workflow Examples (required)
    9. Evidence-based techniques (self-consistency, plan-and-solve, etc.)
    10. Structural quality (clear sections, formatting, examples)

EXIT CODES:
    0 - Prompt passes validation
    1 - Prompt fails validation
    2 - Invalid arguments or file not found
EOF
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            ;;
        -v|--verbose)
            VERBOSE=1
            shift
            ;;
        -s|--min-score)
            MIN_SCORE="$2"
            shift 2
            ;;
        -*)
            echo "Unknown option: $1"
            usage
            ;;
        *)
            PROMPT_FILE="$1"
            shift
            ;;
    esac
done

# Validate arguments
if [[ -z "$PROMPT_FILE" ]]; then
    echo -e "${RED}Error: No prompt file specified${NC}"
    usage
fi

if [[ ! -f "$PROMPT_FILE" ]]; then
    echo -e "${RED}Error: File not found: $PROMPT_FILE${NC}"
    exit 2
fi

echo "==============================================================================="
echo "AGENT SYSTEM PROMPT VALIDATION"
echo "File: $PROMPT_FILE"
echo "Min Score: $MIN_SCORE"
echo "==============================================================================="
echo

# Initialize scoring
TOTAL_SCORE=0
MAX_SCORE=100
CHECKS_PASSED=0
CHECKS_FAILED=0

# Helper functions
check_section() {
    local section_name="$1"
    local pattern="$2"
    local points="$3"
    local required="$4"

    if grep -q "$pattern" "$PROMPT_FILE"; then
        echo -e "${GREEN}✓${NC} $section_name"
        TOTAL_SCORE=$((TOTAL_SCORE + points))
        CHECKS_PASSED=$((CHECKS_PASSED + 1))
        return 0
    else
        if [[ "$required" == "true" ]]; then
            echo -e "${RED}✗${NC} $section_name (REQUIRED)"
            CHECKS_FAILED=$((CHECKS_FAILED + 1))
        else
            echo -e "${YELLOW}⚠${NC} $section_name (recommended)"
            CHECKS_FAILED=$((CHECKS_FAILED + 1))
        fi
        return 1
    fi
}

check_pattern() {
    local description="$1"
    local pattern="$2"
    local points="$3"

    if grep -iq "$pattern" "$PROMPT_FILE"; then
        [[ $VERBOSE -eq 1 ]] && echo -e "  ${GREEN}✓${NC} $description"
        TOTAL_SCORE=$((TOTAL_SCORE + points))
        return 0
    else
        [[ $VERBOSE -eq 1 ]] && echo -e "  ${YELLOW}⚠${NC} $description (missing)"
        return 1
    fi
}

count_examples() {
    local pattern="$1"
    grep -c "$pattern" "$PROMPT_FILE" || echo "0"
}

# 1. CORE IDENTITY SECTION (15 points)
echo "1. Core Identity Section"
check_section "Core Identity Header" "## 🎭 CORE IDENTITY" 5 true
check_pattern "Role description" "I am a \*\*.*\*\*" 5
check_pattern "Domain expertise list" "- \*\*.*\*\* -" 5
echo

# 2. UNIVERSAL COMMANDS (10 points)
echo "2. Universal Commands"
check_section "Universal Commands Section" "## 📋 UNIVERSAL COMMANDS" 3 true
check_pattern "File operations" "/file-read\|/file-write" 2
check_pattern "Git operations" "/git-status\|/git-commit" 2
check_pattern "Memory operations" "/memory-store\|/memory-retrieve" 3
echo

# 3. SPECIALIST COMMANDS (10 points)
echo "3. Specialist Commands"
check_section "Specialist Commands Section" "## 🎯 MY SPECIALIST COMMANDS" 5 false
specialist_count=$(count_examples "^- /[a-z-]*:")
if [[ $specialist_count -ge 3 ]]; then
    echo -e "  ${GREEN}✓${NC} Found $specialist_count specialist commands"
    TOTAL_SCORE=$((TOTAL_SCORE + 5))
else
    echo -e "  ${YELLOW}⚠${NC} Only $specialist_count specialist commands (recommend 3+)"
fi
echo

# 4. MCP SERVER TOOLS (15 points)
echo "4. MCP Server Tools"
check_section "MCP Tools Section" "## 🔧 MCP SERVER TOOLS" 5 true
check_pattern "Claude Flow MCP" "mcp__claude-flow__" 5
check_pattern "MCP usage patterns (WHEN/HOW)" "WHEN:.*HOW:" 5
echo

# 5. COGNITIVE FRAMEWORK (15 points)
echo "5. Cognitive Framework"
check_section "Cognitive Framework Section" "## 🧠 COGNITIVE FRAMEWORK" 5 true
check_pattern "Self-Consistency" "Self-Consistency" 3
check_pattern "Program-of-Thought" "Program-of-Thought\|Decomposition" 3
check_pattern "Plan-and-Solve" "Plan-and-Solve" 4
echo

# 6. GUARDRAILS (10 points)
echo "6. Guardrails"
check_section "Guardrails Section" "## 🚧 GUARDRAILS" 5 true
guardrails_count=$(count_examples "❌ NEVER:")
if [[ $guardrails_count -ge 3 ]]; then
    echo -e "  ${GREEN}✓${NC} Found $guardrails_count guardrails"
    TOTAL_SCORE=$((TOTAL_SCORE + 5))
else
    echo -e "  ${YELLOW}⚠${NC} Only $guardrails_count guardrails (recommend 3+)"
fi
echo

# 7. SUCCESS CRITERIA (10 points)
echo "7. Success Criteria"
check_section "Success Criteria Section" "## ✅ SUCCESS CRITERIA" 5 true
success_count=$(count_examples "- \[ \]")
if [[ $success_count -ge 5 ]]; then
    echo -e "  ${GREEN}✓${NC} Found $success_count success checkpoints"
    TOTAL_SCORE=$((TOTAL_SCORE + 5))
else
    echo -e "  ${YELLOW}⚠${NC} Only $success_count checkpoints (recommend 5+)"
fi
echo

# 8. WORKFLOW EXAMPLES (15 points)
echo "8. Workflow Examples"
check_section "Workflow Examples Section" "## 📖 WORKFLOW EXAMPLES" 5 true
workflow_count=$(count_examples "### Workflow [0-9]:")
if [[ $workflow_count -ge 2 ]]; then
    echo -e "  ${GREEN}✓${NC} Found $workflow_count workflow examples"
    TOTAL_SCORE=$((TOTAL_SCORE + 10))
else
    echo -e "  ${YELLOW}⚠${NC} Only $workflow_count workflows (recommend 2+)"
fi
echo

# Calculate final score percentage
SCORE_PERCENT=$((TOTAL_SCORE * 100 / MAX_SCORE))

echo "==============================================================================="
echo "VALIDATION SUMMARY"
echo "==============================================================================="
echo "Total Score: $TOTAL_SCORE / $MAX_SCORE ($SCORE_PERCENT%)"
echo "Checks Passed: $CHECKS_PASSED"
echo "Checks Failed: $CHECKS_FAILED"
echo

# Determine pass/fail
if [[ $SCORE_PERCENT -ge $MIN_SCORE ]]; then
    echo -e "${GREEN}✓ VALIDATION PASSED${NC} (Score: $SCORE_PERCENT% >= Minimum: $MIN_SCORE%)"

    # Provide tier classification
    if [[ $SCORE_PERCENT -ge 90 ]]; then
        echo -e "${GREEN}Tier: GOLD${NC} - Production-ready with excellent evidence-based patterns"
    elif [[ $SCORE_PERCENT -ge 75 ]]; then
        echo -e "${GREEN}Tier: SILVER${NC} - Well-structured, recommended minor enhancements"
    else
        echo -e "${YELLOW}Tier: BRONZE${NC} - Functional, consider adding more patterns"
    fi

    exit 0
else
    echo -e "${RED}✗ VALIDATION FAILED${NC} (Score: $SCORE_PERCENT% < Minimum: $MIN_SCORE%)"
    echo
    echo "Recommendations:"
    echo "1. Ensure all REQUIRED sections are present"
    echo "2. Add evidence-based prompting patterns (self-consistency, plan-and-solve)"
    echo "3. Include 2+ workflow examples with specific commands"
    echo "4. Define 3+ guardrails with examples"
    echo "5. Add domain-specific specialist commands"

    exit 1
fi
