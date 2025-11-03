#!/bin/bash
###############################################################################
# Integration Test Suite for PPTX Generation Workflow
# Tests end-to-end presentation generation with validation
###############################################################################

set -euo pipefail

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test counters
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0

# Test directories
TEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESOURCES_DIR="$(dirname "$TEST_DIR")/resources"
TEMP_DIR="/tmp/pptx-test-$$"

# Logging
log_test() {
    echo -e "${BLUE}[TEST]${NC} $1"
}

log_pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
    ((TESTS_PASSED++))
}

log_fail() {
    echo -e "${RED}[FAIL]${NC} $1"
    ((TESTS_FAILED++))
}

log_info() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

# Setup test environment
setup() {
    log_info "Setting up test environment..."
    mkdir -p "$TEMP_DIR"
    ((TESTS_RUN++))
}

# Cleanup test environment
cleanup() {
    if [[ -d "$TEMP_DIR" ]]; then
        log_info "Cleaning up test directory: $TEMP_DIR"
        rm -rf "$TEMP_DIR"
    fi
}
trap cleanup EXIT

# Test 1: Slide generator module loads successfully
test_slide_generator_import() {
    log_test "Slide generator module import"

    if python3 -c "import sys; sys.path.insert(0, '$RESOURCES_DIR'); from slide_generator import SlideGenerator" 2>/dev/null; then
        log_pass "Slide generator module imported successfully"
        return 0
    else
        log_fail "Failed to import slide generator module"
        return 1
    fi
}

# Test 2: Template engine module loads successfully
test_template_engine_import() {
    log_test "Template engine module import"

    if node -e "require('$RESOURCES_DIR/template-engine.js')" 2>/dev/null; then
        log_pass "Template engine module loaded successfully"
        return 0
    else
        log_fail "Failed to load template engine module"
        return 1
    fi
}

# Test 3: Chart creator module loads successfully
test_chart_creator_import() {
    log_test "Chart creator module import"

    if python3 -c "import sys; sys.path.insert(0, '$RESOURCES_DIR'); from chart_creator import ChartCreator" 2>/dev/null; then
        log_pass "Chart creator module imported successfully"
        return 0
    else
        log_fail "Failed to import chart creator module"
        return 1
    fi
}

# Test 4: Template validation with valid config
test_template_validation_valid() {
    log_test "Template validation with valid configuration"

    local template_file="$TEMP_DIR/valid-template.yaml"

    cat > "$template_file" <<EOF
name: "Test Template"
colorScheme:
  primary: "#1E3A8A"
  secondary: "#3B82F6"
  accent: "#F59E0B"
  background: "#FFFFFF"
  text: "#1F2937"
  textLight: "#6B7280"
typography:
  fontSize:
    h1: 36
    h2: 28
    body: 18
    caption: 14
EOF

    if node "$RESOURCES_DIR/template-engine.js" validate "$template_file" 2>/dev/null; then
        log_pass "Valid template passed validation"
        return 0
    else
        log_fail "Valid template failed validation"
        return 1
    fi
}

# Test 5: Template validation with invalid config (low contrast)
test_template_validation_invalid() {
    log_test "Template validation with invalid configuration"

    local template_file="$TEMP_DIR/invalid-template.yaml"

    cat > "$template_file" <<EOF
name: "Invalid Template"
colorScheme:
  background: "#FFFFFF"
  text: "#CCCCCC"
  textLight: "#DDDDDD"
typography:
  fontSize:
    h1: 12
    body: 10
EOF

    if node "$RESOURCES_DIR/template-engine.js" validate "$template_file" 2>/dev/null; then
        log_fail "Invalid template passed validation (should have failed)"
        return 1
    else
        log_pass "Invalid template correctly rejected"
        return 0
    fi
}

# Test 6: Slide generation creates HTML output
test_slide_html_generation() {
    log_test "Slide HTML generation"

    local output_file="$TEMP_DIR/test-slides.html"

    python3 <<EOF
import sys
sys.path.insert(0, '$RESOURCES_DIR')
from slide_generator import SlideGenerator, ColorPalette, LayoutConstraints

palette = ColorPalette(
    primary="#1E3A8A",
    secondary="#3B82F6",
    accent="#F59E0B",
    background="#FFFFFF",
    text="#1F2937"
)

constraints = LayoutConstraints()
generator = SlideGenerator(palette, constraints)

generator.create_title_slide("Test Presentation", "Integration Test")
generator.create_content_slide("Test Slide", ["Point 1", "Point 2", "Point 3"])

generator.export_for_html2pptx("$output_file")
EOF

    if [[ -f "$output_file" ]]; then
        log_pass "HTML slides generated successfully"

        # Verify HTML structure
        if grep -q "<!DOCTYPE html>" "$output_file"; then
            log_info "  ✓ Valid HTML structure"
        else
            log_fail "  ✗ Invalid HTML structure"
            return 1
        fi

        # Check for content
        if grep -q "Test Presentation" "$output_file"; then
            log_info "  ✓ Title slide content present"
        else
            log_fail "  ✗ Title slide content missing"
            return 1
        fi

        return 0
    else
        log_fail "HTML generation failed - no output file"
        return 1
    fi
}

# Test 7: Chart generation creates image output
test_chart_image_generation() {
    log_test "Chart image generation"

    local output_file="$TEMP_DIR/test-chart.png"

    python3 <<EOF
import sys
sys.path.insert(0, '$RESOURCES_DIR')
from chart_creator import ChartCreator, ChartConfig

config = ChartConfig(
    color_palette=['#1E3A8A', '#3B82F6', '#60A5FA'],
    title_font_size=20,
    label_font_size=14
)

creator = ChartCreator(config)

data = {
    'Q1': 2.5,
    'Q2': 3.2,
    'Q3': 3.8,
    'Q4': 4.1
}

fig = creator.create_bar_chart(
    data,
    'Quarterly Revenue',
    x_label='Quarter',
    y_label='Revenue (\$M)'
)

creator.save_chart(fig, "$output_file")
EOF

    if [[ -f "$output_file" ]]; then
        # Check file size (should be >10KB for a valid chart)
        local file_size=$(stat -c%s "$output_file" 2>/dev/null || stat -f%z "$output_file" 2>/dev/null || echo "0")

        if [[ "$file_size" -gt 10240 ]]; then
            log_pass "Chart image generated successfully (${file_size} bytes)"
            return 0
        else
            log_fail "Chart image too small (${file_size} bytes) - likely invalid"
            return 1
        fi
    else
        log_fail "Chart generation failed - no output file"
        return 1
    fi
}

# Test 8: Prohibited elements detection
test_prohibited_elements_detection() {
    log_test "Prohibited elements detection"

    python3 <<EOF
import sys
sys.path.insert(0, '$RESOURCES_DIR')
from slide_generator import SlideGenerator, ColorPalette, LayoutConstraints

palette = ColorPalette(
    primary="#1E3A8A",
    secondary="#3B82F6",
    accent="#F59E0B",
    background="#FFFFFF",
    text="#1F2937"
)

generator = SlideGenerator(palette, LayoutConstraints())

# Test prohibited elements
html_with_border = '<div style="border: 1px solid black;">Content</div>'
violations = generator._check_prohibited_elements(html_with_border)

if len(violations) > 0 and any('border' in v.lower() for v in violations):
    print("PASS: Border detected")
    sys.exit(0)
else:
    print("FAIL: Border not detected")
    sys.exit(1)
EOF

    if [[ $? -eq 0 ]]; then
        log_pass "Prohibited elements correctly detected"
        return 0
    else
        log_fail "Prohibited elements detection failed"
        return 1
    fi
}

# Test 9: Accessibility validation (contrast ratio)
test_accessibility_contrast() {
    log_test "Accessibility contrast validation"

    python3 <<EOF
import sys
sys.path.insert(0, '$RESOURCES_DIR')
from slide_generator import ColorPalette

palette = ColorPalette(
    primary="#1E3A8A",
    secondary="#3B82F6",
    accent="#F59E0B",
    background="#FFFFFF",
    text="#1F2937"
)

# Test high contrast
passes_high, ratio_high = palette.validate_contrast("#1F2937", "#FFFFFF")

# Test low contrast
passes_low, ratio_low = palette.validate_contrast("#CCCCCC", "#FFFFFF")

if passes_high and not passes_low:
    print(f"PASS: Contrast validation working (high: {ratio_high:.2f}, low: {ratio_low:.2f})")
    sys.exit(0)
else:
    print(f"FAIL: Contrast validation not working correctly")
    sys.exit(1)
EOF

    if [[ $? -eq 0 ]]; then
        log_pass "Accessibility contrast validation working"
        return 0
    else
        log_fail "Accessibility contrast validation failed"
        return 1
    fi
}

# Test 10: Bullet limit enforcement
test_bullet_limit_enforcement() {
    log_test "Bullet limit enforcement"

    python3 <<EOF
import sys
sys.path.insert(0, '$RESOURCES_DIR')
from slide_generator import SlideGenerator, ColorPalette, LayoutConstraints

palette = ColorPalette(
    primary="#1E3A8A",
    secondary="#3B82F6",
    accent="#F59E0B",
    background="#FFFFFF",
    text="#1F2937"
)

generator = SlideGenerator(palette, LayoutConstraints())

try:
    # Try to create slide with too many bullets
    generator.create_content_slide("Test", ["1", "2", "3", "4", "5"])
    print("FAIL: Should have rejected too many bullets")
    sys.exit(1)
except ValueError as e:
    if "exceeds maximum" in str(e).lower():
        print("PASS: Bullet limit correctly enforced")
        sys.exit(0)
    else:
        print(f"FAIL: Wrong error: {e}")
        sys.exit(1)
EOF

    if [[ $? -eq 0 ]]; then
        log_pass "Bullet limit enforcement working"
        return 0
    else
        log_fail "Bullet limit enforcement failed"
        return 1
    fi
}

# Run all tests
main() {
    echo "=========================================="
    echo "PPTX Generation Integration Test Suite"
    echo "=========================================="
    echo ""

    setup

    # Module loading tests
    test_slide_generator_import
    ((TESTS_RUN++))

    test_template_engine_import
    ((TESTS_RUN++))

    test_chart_creator_import
    ((TESTS_RUN++))

    # Template validation tests
    test_template_validation_valid
    ((TESTS_RUN++))

    test_template_validation_invalid
    ((TESTS_RUN++))

    # Generation tests
    test_slide_html_generation
    ((TESTS_RUN++))

    test_chart_image_generation
    ((TESTS_RUN++))

    # Validation tests
    test_prohibited_elements_detection
    ((TESTS_RUN++))

    test_accessibility_contrast
    ((TESTS_RUN++))

    test_bullet_limit_enforcement
    ((TESTS_RUN++))

    # Summary
    echo ""
    echo "=========================================="
    echo "Test Summary"
    echo "=========================================="
    echo "Tests run:    $TESTS_RUN"
    echo "Tests passed: $TESTS_PASSED"
    echo "Tests failed: $TESTS_FAILED"
    echo ""

    if [[ $TESTS_FAILED -eq 0 ]]; then
        echo -e "${GREEN}✓ All tests passed!${NC}"
        exit 0
    else
        echo -e "${RED}✗ Some tests failed${NC}"
        exit 1
    fi
}

main "$@"
