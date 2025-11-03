#!/bin/bash
# Micro-Skill Validator - Validate micro-skill structure, contracts, and agent design
# Version: 2.0.0

set -e

SKILL_DIR="${1:-.}"
ERRORS=0
WARNINGS=0

# Colors for output
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔍 Micro-Skill Validator v2.0.0${NC}"
echo "Validating: $SKILL_DIR"
echo ""

# Validation functions
validate_structure() {
    echo -e "${BLUE}[1/7] Validating directory structure...${NC}"

    if [[ ! -f "$SKILL_DIR/SKILL.md" ]]; then
        echo -e "${RED}✗ Missing SKILL.md${NC}"
        ((ERRORS++))
    else
        echo -e "${GREEN}✓ SKILL.md exists${NC}"
    fi

    # Optional but recommended
    if [[ -d "$SKILL_DIR/tests" ]]; then
        echo -e "${GREEN}✓ tests/ directory found${NC}"
    else
        echo -e "${YELLOW}⚠ tests/ directory missing (recommended)${NC}"
        ((WARNINGS++))
    fi

    if [[ -d "$SKILL_DIR/examples" ]]; then
        echo -e "${GREEN}✓ examples/ directory found${NC}"
    else
        echo -e "${YELLOW}⚠ examples/ directory missing (recommended)${NC}"
        ((WARNINGS++))
    fi
}

validate_frontmatter() {
    echo -e "\n${BLUE}[2/7] Validating YAML frontmatter...${NC}"

    if ! grep -q "^---$" "$SKILL_DIR/SKILL.md"; then
        echo -e "${RED}✗ Missing YAML frontmatter delimiters${NC}"
        ((ERRORS++))
        return
    fi

    # Extract frontmatter
    FRONTMATTER=$(sed -n '/^---$/,/^---$/p' "$SKILL_DIR/SKILL.md" | sed '1d;$d')

    # Check required fields
    for field in name description version; do
        if echo "$FRONTMATTER" | grep -q "^$field:"; then
            echo -e "${GREEN}✓ Field '$field' present${NC}"
        else
            echo -e "${RED}✗ Missing required field: $field${NC}"
            ((ERRORS++))
        fi
    done

    # Check tags
    if echo "$FRONTMATTER" | grep -q "^tags:"; then
        echo -e "${GREEN}✓ Tags defined${NC}"
    else
        echo -e "${YELLOW}⚠ No tags defined (recommended)${NC}"
        ((WARNINGS++))
    fi
}

validate_agent_design() {
    echo -e "\n${BLUE}[3/7] Validating specialist agent design...${NC}"

    CONTENT=$(cat "$SKILL_DIR/SKILL.md")

    # Check for agent section
    if echo "$CONTENT" | grep -qi "## Specialist Agent\|## Agent"; then
        echo -e "${GREEN}✓ Specialist agent section found${NC}"
    else
        echo -e "${RED}✗ Missing specialist agent section${NC}"
        ((ERRORS++))
    fi

    # Check for evidence-based pattern
    PATTERNS=("self-consistency" "program-of-thought" "plan-and-solve")
    PATTERN_FOUND=false

    for pattern in "${PATTERNS[@]}"; do
        if echo "$CONTENT" | grep -qi "$pattern"; then
            echo -e "${GREEN}✓ Evidence-based pattern detected: $pattern${NC}"
            PATTERN_FOUND=true
            break
        fi
    done

    if [[ "$PATTERN_FOUND" == false ]]; then
        echo -e "${YELLOW}⚠ No evidence-based pattern detected${NC}"
        ((WARNINGS++))
    fi

    # Check for methodology
    if echo "$CONTENT" | grep -qi "methodology"; then
        echo -e "${GREEN}✓ Methodology section present${NC}"
    else
        echo -e "${YELLOW}⚠ Missing methodology description${NC}"
        ((WARNINGS++))
    fi
}

validate_contracts() {
    echo -e "\n${BLUE}[4/7] Validating input/output contracts...${NC}"

    CONTENT=$(cat "$SKILL_DIR/SKILL.md")

    # Check for input contract
    if echo "$CONTENT" | grep -qi "## Input Contract\|## Input"; then
        echo -e "${GREEN}✓ Input contract defined${NC}"
    else
        echo -e "${RED}✗ Missing input contract${NC}"
        ((ERRORS++))
    fi

    # Check for output contract
    if echo "$CONTENT" | grep -qi "## Output Contract\|## Output"; then
        echo -e "${GREEN}✓ Output contract defined${NC}"
    else
        echo -e "${RED}✗ Missing output contract${NC}"
        ((ERRORS++))
    fi

    # Check for YAML/JSON schema definition
    if echo "$CONTENT" | grep -q '```yaml\|```json'; then
        echo -e "${GREEN}✓ Structured contract format (YAML/JSON)${NC}"
    else
        echo -e "${YELLOW}⚠ Consider using YAML/JSON for contract clarity${NC}"
        ((WARNINGS++))
    fi
}

validate_failure_modes() {
    echo -e "\n${BLUE}[5/7] Validating failure mode awareness...${NC}"

    CONTENT=$(cat "$SKILL_DIR/SKILL.md")

    if echo "$CONTENT" | grep -qi "failure mode\|error handling\|edge case"; then
        echo -e "${GREEN}✓ Failure modes documented${NC}"
    else
        echo -e "${YELLOW}⚠ No failure mode documentation (recommended)${NC}"
        ((WARNINGS++))
    fi
}

validate_atomicity() {
    echo -e "\n${BLUE}[6/7] Validating atomicity principle...${NC}"

    CONTENT=$(cat "$SKILL_DIR/SKILL.md")
    WORD_COUNT=$(echo "$CONTENT" | wc -w)

    echo "  Skill content: $WORD_COUNT words"

    if [[ $WORD_COUNT -gt 2000 ]]; then
        echo -e "${YELLOW}⚠ Skill may be too complex (>2000 words) - consider splitting${NC}"
        ((WARNINGS++))
    elif [[ $WORD_COUNT -lt 200 ]]; then
        echo -e "${YELLOW}⚠ Skill may be too minimal (<200 words)${NC}"
        ((WARNINGS++))
    else
        echo -e "${GREEN}✓ Skill complexity appropriate (200-2000 words)${NC}"
    fi

    # Check for single responsibility indicators
    SECTION_COUNT=$(grep -c "^## " "$SKILL_DIR/SKILL.md" || true)
    echo "  Sections: $SECTION_COUNT"

    if [[ $SECTION_COUNT -gt 15 ]]; then
        echo -e "${YELLOW}⚠ High section count ($SECTION_COUNT) - may indicate multiple responsibilities${NC}"
        ((WARNINGS++))
    else
        echo -e "${GREEN}✓ Section count reasonable${NC}"
    fi
}

validate_integration() {
    echo -e "\n${BLUE}[7/7] Validating integration points...${NC}"

    CONTENT=$(cat "$SKILL_DIR/SKILL.md")

    # Check for integration documentation
    if echo "$CONTENT" | grep -qi "integration\|cascade\|composition"; then
        echo -e "${GREEN}✓ Integration points documented${NC}"
    else
        echo -e "${YELLOW}⚠ No integration/composition documentation${NC}"
        ((WARNINGS++))
    fi

    # Check for neural training mention
    if echo "$CONTENT" | grep -qi "neural\|training\|learning"; then
        echo -e "${GREEN}✓ Neural training integration mentioned${NC}"
    else
        echo -e "${YELLOW}⚠ Neural training integration not mentioned${NC}"
        ((WARNINGS++))
    fi
}

# Run all validations
validate_structure
validate_frontmatter
validate_agent_design
validate_contracts
validate_failure_modes
validate_atomicity
validate_integration

# Summary
echo ""
echo -e "${BLUE}═══════════════════════════════════════${NC}"
echo -e "${BLUE}Validation Summary${NC}"
echo -e "${BLUE}═══════════════════════════════════════${NC}"

if [[ $ERRORS -eq 0 && $WARNINGS -eq 0 ]]; then
    echo -e "${GREEN}✓ Perfect! No issues found.${NC}"
    exit 0
elif [[ $ERRORS -eq 0 ]]; then
    echo -e "${YELLOW}⚠ $WARNINGS warning(s) - consider addressing${NC}"
    exit 0
else
    echo -e "${RED}✗ $ERRORS error(s) found${NC}"
    echo -e "${YELLOW}⚠ $WARNINGS warning(s)${NC}"
    echo ""
    echo "Please fix errors before packaging/deploying this micro-skill."
    exit 1
fi
