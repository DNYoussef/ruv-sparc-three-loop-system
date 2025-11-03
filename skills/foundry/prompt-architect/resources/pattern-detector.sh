#!/bin/bash
# Pattern Detector - Identify prompting patterns and techniques in text
#
# This script uses regex pattern matching to detect evidence-based
# prompting techniques and structural patterns in prompt files.
#
# Usage:
#   ./pattern-detector.sh <file>
#   ./pattern-detector.sh --batch *.txt
#   ./pattern-detector.sh --help

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Pattern definitions
declare -A PATTERNS=(
    # Evidence-based techniques
    ["chain_of_thought"]="step\s+by\s+step|explain.*reasoning|think\s+through|show.*thinking"
    ["self_consistency"]="validate|verify|cross-check|alternative\s+(perspective|interpretation)"
    ["program_of_thought"]="calculate|compute|solve.*step|show.*(intermediate|calculations)"
    ["plan_and_solve"]="first.*plan|then\s+execute|finally.*verify"
    ["few_shot"]="(here\s+(are|is)\s+)?examples?|input:.*output:|example\s+\d+"

    # Structural patterns
    ["hierarchical"]="^#{1,6}\s+|\d+\.\s+|[-*]\s+"
    ["delimiters"]='```|<[^>]+>|---+|\*\*\*+'
    ["sections"]="(context|background|objective|requirements|constraints|output)"

    # Quality indicators
    ["constraints"]="(must|should|cannot|require|ensure|constraint)"
    ["success_criteria"]="success|criteria|verify|validate|check"
    ["edge_cases"]="edge\s+case|if.*then|boundary|exception"

    # Anti-patterns
    ["vague_modifiers"]="(quickly|fast|brief|simply)"
    ["assumptions"]="(obviously|clearly)"
    ["contradictions"]="(comprehensive|detailed).*(brief|concise|short)"
)

# Function to print colored output
print_color() {
    local color=$1
    shift
    echo -e "${color}$*${NC}"
}

# Function to detect patterns in a file
detect_patterns() {
    local file=$1
    local content
    content=$(cat "$file")

    print_color "$BLUE" "\n📄 Analyzing: $file"
    print_color "$BLUE" "$(printf '=%.0s' {1..60})"

    # Detect evidence-based techniques
    print_color "$GREEN" "\n✅ Evidence-Based Techniques:"
    local found_techniques=0

    for technique in chain_of_thought self_consistency program_of_thought plan_and_solve few_shot; do
        local pattern="${PATTERNS[$technique]}"
        if echo "$content" | grep -iP "$pattern" >/dev/null 2>&1; then
            local count
            count=$(echo "$content" | grep -ioP "$pattern" | wc -l)
            print_color "$GREEN" "  ✓ $technique (found $count times)"
            ((found_techniques++))
        fi
    done

    if [ $found_techniques -eq 0 ]; then
        print_color "$YELLOW" "  ⚠️  No evidence-based techniques detected"
    fi

    # Detect structural patterns
    print_color "$BLUE" "\n🏗️  Structural Patterns:"
    local found_structure=0

    for pattern_type in hierarchical delimiters sections; do
        local pattern="${PATTERNS[$pattern_type]}"
        if echo "$content" | grep -iP "$pattern" >/dev/null 2>&1; then
            local count
            count=$(echo "$content" | grep -ioP "$pattern" | wc -l)
            print_color "$BLUE" "  ✓ $pattern_type (found $count times)"
            ((found_structure++))
        fi
    done

    if [ $found_structure -eq 0 ]; then
        print_color "$YELLOW" "  ⚠️  Limited structural organization"
    fi

    # Detect quality indicators
    print_color "$GREEN" "\n💎 Quality Indicators:"
    local found_quality=0

    for indicator in constraints success_criteria edge_cases; do
        local pattern="${PATTERNS[$indicator]}"
        if echo "$content" | grep -iP "$pattern" >/dev/null 2>&1; then
            local count
            count=$(echo "$content" | grep -ioP "$pattern" | wc -l)
            print_color "$GREEN" "  ✓ $indicator (found $count times)"
            ((found_quality++))
        fi
    done

    if [ $found_quality -eq 0 ]; then
        print_color "$YELLOW" "  ⚠️  Few explicit quality indicators"
    fi

    # Detect anti-patterns
    print_color "$RED" "\n⚠️  Anti-Patterns:"
    local found_antipatterns=0

    for antipattern in vague_modifiers assumptions contradictions; do
        local pattern="${PATTERNS[$antipattern]}"
        if echo "$content" | grep -iP "$pattern" >/dev/null 2>&1; then
            local count
            count=$(echo "$content" | grep -ioP "$pattern" | wc -l)
            print_color "$RED" "  ✗ $antipattern (found $count times)"
            ((found_antipatterns++))
        fi
    done

    if [ $found_antipatterns -eq 0 ]; then
        print_color "$GREEN" "  ✓ No common anti-patterns detected"
    fi

    # Calculate metrics
    print_color "$BLUE" "\n📊 Metrics:"
    local word_count
    word_count=$(echo "$content" | wc -w)
    local line_count
    line_count=$(echo "$content" | wc -l)
    local sentence_count
    sentence_count=$(echo "$content" | grep -o '[.!?]' | wc -l)

    echo "  Word count:      $word_count"
    echo "  Line count:      $line_count"
    echo "  Sentence count:  $sentence_count"

    # Complexity assessment
    local complexity="Simple"
    if [ "$word_count" -gt 800 ]; then
        complexity="Very Complex"
    elif [ "$word_count" -gt 500 ]; then
        complexity="Complex"
    elif [ "$word_count" -gt 200 ]; then
        complexity="Medium"
    fi
    echo "  Complexity:      $complexity"

    # Overall score
    local score=$((found_techniques * 20 + found_structure * 15 + found_quality * 15 - found_antipatterns * 10))
    score=$((score > 100 ? 100 : score))
    score=$((score < 0 ? 0 : score))

    print_color "$BLUE" "\n⭐ Overall Pattern Score: $score/100"

    echo ""
}

# Function to analyze patterns across multiple files
batch_analyze() {
    local files=("$@")
    local total_files=${#files[@]}

    print_color "$BLUE" "🔍 Batch Analysis Mode: Analyzing $total_files files"

    for file in "${files[@]}"; do
        if [ -f "$file" ]; then
            detect_patterns "$file"
        else
            print_color "$RED" "❌ File not found: $file"
        fi
    done

    print_color "$GREEN" "\n✅ Batch analysis complete!"
}

# Function to show usage
show_usage() {
    cat << EOF
Pattern Detector - Identify prompting patterns and techniques

Usage:
  ./pattern-detector.sh <file>              Analyze single file
  ./pattern-detector.sh --batch <files>     Analyze multiple files
  ./pattern-detector.sh --help              Show this help

Options:
  --batch                 Batch mode for multiple files
  --json                  Output in JSON format (future)
  --help                  Show this help message

Examples:
  ./pattern-detector.sh prompt.txt
  ./pattern-detector.sh --batch prompts/*.txt
  find . -name "*.txt" -exec ./pattern-detector.sh {} \;

Detected Patterns:
  Evidence-Based Techniques:
    - Chain-of-Thought      Step-by-step reasoning
    - Self-Consistency      Validation and verification
    - Program-of-Thought    Computational problem solving
    - Plan-and-Solve        Structured workflow
    - Few-Shot              Example-based learning

  Structural Patterns:
    - Hierarchical          Headers, lists, organization
    - Delimiters            Code blocks, XML tags, separators
    - Sections              Named sections (context, objectives, etc.)

  Quality Indicators:
    - Constraints           Explicit requirements and limitations
    - Success Criteria      Clear success definitions
    - Edge Cases            Boundary condition handling

  Anti-Patterns:
    - Vague Modifiers       Unclear qualifiers (quickly, simply)
    - Assumptions           Assumed knowledge (obviously, clearly)
    - Contradictions        Conflicting requirements

EOF
}

# Main execution
main() {
    if [ $# -eq 0 ]; then
        show_usage
        exit 1
    fi

    case "$1" in
        --help|-h)
            show_usage
            exit 0
            ;;
        --batch)
            shift
            if [ $# -eq 0 ]; then
                print_color "$RED" "❌ Error: --batch requires file arguments"
                exit 1
            fi
            batch_analyze "$@"
            ;;
        *)
            if [ ! -f "$1" ]; then
                print_color "$RED" "❌ Error: File not found: $1"
                exit 1
            fi
            detect_patterns "$1"
            ;;
    esac
}

# Run main function
main "$@"
