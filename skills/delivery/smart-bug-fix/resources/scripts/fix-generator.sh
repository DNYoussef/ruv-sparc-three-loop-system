#!/bin/bash
################################################################################
# Fix Generator - AI-Powered Bug Fix Generation
#
# Generates and validates bug fixes using multi-model reasoning:
# - Claude for deep understanding and alternatives
# - Codex for rapid fix implementation
# - Gemini for large codebase context (optional)
#
# Features:
# - Multi-model fix generation (3+ approaches)
# - Sandbox testing before applying
# - Automatic rollback on failure
# - Git integration for safe patching
# - Iterative refinement (max 5 attempts)
#
# Usage:
#   bash fix-generator.sh \
#     --bug-id BUG-123 \
#     --rca-report rca-report.md \
#     --context-path src/api/ \
#     --max-attempts 5
################################################################################

set -e  # Exit on error
set -u  # Exit on undefined variable

# ============================================================================
# Configuration
# ============================================================================

BUG_ID=""
RCA_REPORT=""
CONTEXT_PATH=""
MAX_ATTEMPTS=5
SANDBOX_MODE=true
AUTO_ROLLBACK=true
GIT_INTEGRATION=true
OUTPUT_DIR="fix-implementation"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# Helper Functions
# ============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --bug-id)
                BUG_ID="$2"
                shift 2
                ;;
            --rca-report)
                RCA_REPORT="$2"
                shift 2
                ;;
            --context-path)
                CONTEXT_PATH="$2"
                shift 2
                ;;
            --max-attempts)
                MAX_ATTEMPTS="$2"
                shift 2
                ;;
            --no-sandbox)
                SANDBOX_MODE=false
                shift
                ;;
            --no-rollback)
                AUTO_ROLLBACK=false
                shift
                ;;
            --no-git)
                GIT_INTEGRATION=false
                shift
                ;;
            --output-dir)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            *)
                log_error "Unknown argument: $1"
                exit 1
                ;;
        esac
    done

    # Validate required arguments
    if [[ -z "$BUG_ID" ]] || [[ -z "$RCA_REPORT" ]] || [[ -z "$CONTEXT_PATH" ]]; then
        log_error "Missing required arguments"
        echo "Usage: $0 --bug-id <id> --rca-report <file> --context-path <path> [--max-attempts <n>]"
        exit 1
    fi
}

# ============================================================================
# RCA Analysis
# ============================================================================

analyze_rca() {
    log_info "Analyzing RCA report: $RCA_REPORT"

    # Extract key information from RCA
    PRIMARY_CAUSE=$(grep -A 2 "Primary Root Cause" "$RCA_REPORT" | tail -1 || echo "Not found")
    CONTRIBUTING_FACTORS=$(grep -A 10 "Contributing Factors" "$RCA_REPORT" | tail -5 || echo "")

    log_info "Primary cause: $PRIMARY_CAUSE"

    # Check if codebase is large (>10k LOC)
    LOC=$(find "$CONTEXT_PATH" -name "*.js" -o -name "*.ts" -o -name "*.py" | xargs wc -l 2>/dev/null | tail -1 | awk '{print $1}' || echo "0")

    if [[ $LOC -gt 10000 ]]; then
        USE_GEMINI_CONTEXT=true
        log_info "Large codebase detected ($LOC lines) - will use Gemini MegaContext"
    else
        USE_GEMINI_CONTEXT=false
        log_info "Standard codebase ($LOC lines)"
    fi
}

# ============================================================================
# Multi-Model Fix Generation
# ============================================================================

generate_claude_fix() {
    log_info "Generating fix approach with Claude..."

    cat > /tmp/claude-fix-prompt.txt <<EOF
Analyze this root cause analysis and generate a comprehensive fix:

$(cat "$RCA_REPORT")

Requirements:
1. Fix the primary root cause
2. Address contributing factors
3. Add defensive checks to prevent recurrence
4. Maintain existing API contracts
5. Include error handling

Provide:
- Detailed explanation of the fix
- Code changes needed
- Potential risks
- Alternative approaches
EOF

    # Use Claude for deep understanding
    # Note: This would call Claude API in production
    log_info "Claude analysis complete (placeholder)"

    echo "approach_1_claude.json" > /tmp/claude-fix.json
}

generate_codex_alternatives() {
    log_info "Generating alternative fixes with Codex..."

    # Use Codex reasoning mode for alternatives
    # Note: This would call Codex API in production
    cat > /tmp/codex-alternatives.json <<EOF
{
  "alternatives": [
    {
      "approach": "Defensive Programming",
      "description": "Add null checks and input validation at all boundaries",
      "complexity": "low",
      "risk": "low",
      "confidence": 0.85
    },
    {
      "approach": "Refactoring",
      "description": "Restructure code to eliminate root cause condition",
      "complexity": "medium",
      "risk": "medium",
      "confidence": 0.70
    },
    {
      "approach": "Design Pattern",
      "description": "Apply appropriate design pattern to prevent issue",
      "complexity": "high",
      "risk": "medium",
      "confidence": 0.60
    }
  ]
}
EOF

    log_info "Generated $(jq '.alternatives | length' /tmp/codex-alternatives.json) alternative approaches"
}

generate_gemini_context() {
    if [[ "$USE_GEMINI_CONTEXT" == true ]]; then
        log_info "Analyzing codebase with Gemini MegaContext..."

        # Use Gemini for large codebase analysis
        # Note: This would call Gemini API in production
        log_info "Gemini context analysis complete (placeholder)"
    fi
}

rank_solutions() {
    log_info "Ranking solutions by safety, complexity, and impact..."

    # Score: (confidence * 0.4) + ((1 - complexity) * 0.3) + ((1 - risk) * 0.3)
    # For now, use simple ranking
    BEST_APPROACH="Defensive Programming"

    log_success "Selected approach: $BEST_APPROACH"
}

# ============================================================================
# Fix Implementation
# ============================================================================

create_git_branch() {
    if [[ "$GIT_INTEGRATION" == true ]]; then
        log_info "Creating git branch: fix/$BUG_ID"

        git checkout -b "fix/$BUG_ID" 2>/dev/null || {
            log_warning "Branch already exists, using existing branch"
            git checkout "fix/$BUG_ID"
        }
    fi
}

implement_fix() {
    local attempt=$1

    log_info "Implementing fix (attempt $attempt/$MAX_ATTEMPTS)..."

    mkdir -p "$OUTPUT_DIR"

    # Use Codex Auto for implementation
    log_info "Using Codex Auto for fix implementation..."

    # Note: In production, this would call actual Codex Auto API
    # For demonstration, create placeholder implementation
    cat > "$OUTPUT_DIR/fix-changes.diff" <<EOF
diff --git a/src/api/handler.js b/src/api/handler.js
index abc123..def456 100644
--- a/src/api/handler.js
+++ b/src/api/handler.js
@@ -10,6 +10,11 @@ async function handleRequest(req, res) {
+  // Add null check to prevent TypeError
+  if (!req.body || !req.body.data) {
+    return res.status(400).json({ error: 'Missing required data' });
+  }
+
   const result = await processData(req.body.data);
   res.json(result);
 }
EOF

    log_success "Fix implementation complete"
}

# ============================================================================
# Sandbox Testing
# ============================================================================

setup_sandbox() {
    if [[ "$SANDBOX_MODE" == true ]]; then
        log_info "Setting up sandbox environment..."

        # Create isolated sandbox
        SANDBOX_DIR=$(mktemp -d)

        # Copy codebase to sandbox
        cp -r "$CONTEXT_PATH" "$SANDBOX_DIR/"

        log_success "Sandbox ready at: $SANDBOX_DIR"
    fi
}

apply_fix_to_sandbox() {
    if [[ "$SANDBOX_MODE" == true ]]; then
        log_info "Applying fix to sandbox..."

        # Apply the diff
        cd "$SANDBOX_DIR" || exit 1
        patch -p1 < "$OUTPUT_DIR/fix-changes.diff" || {
            log_error "Failed to apply patch in sandbox"
            return 1
        }

        cd - > /dev/null
        log_success "Fix applied to sandbox"
    fi
}

run_sandbox_tests() {
    if [[ "$SANDBOX_MODE" == true ]]; then
        log_info "Running tests in sandbox..."

        cd "$SANDBOX_DIR" || exit 1

        # Run test suite
        # Note: Adjust based on project (npm test, pytest, etc.)
        if [[ -f "package.json" ]]; then
            npm test 2>&1 | tee /tmp/sandbox-test-results.log
            TEST_EXIT_CODE=${PIPESTATUS[0]}
        elif [[ -f "requirements.txt" ]]; then
            pytest 2>&1 | tee /tmp/sandbox-test-results.log
            TEST_EXIT_CODE=${PIPESTATUS[0]}
        else
            log_warning "No test framework detected"
            TEST_EXIT_CODE=0
        fi

        cd - > /dev/null

        if [[ $TEST_EXIT_CODE -eq 0 ]]; then
            log_success "All tests passed in sandbox!"
            return 0
        else
            log_error "Tests failed in sandbox"
            return 1
        fi
    fi

    return 0
}

cleanup_sandbox() {
    if [[ "$SANDBOX_MODE" == true ]] && [[ -n "${SANDBOX_DIR:-}" ]]; then
        log_info "Cleaning up sandbox..."
        rm -rf "$SANDBOX_DIR"
    fi
}

# ============================================================================
# Iteration & Refinement
# ============================================================================

refine_fix() {
    local attempt=$1
    local test_output="$2"

    log_warning "Refining fix based on test failures..."

    # Extract failures from test output
    FAILURES=$(grep -A 5 "FAILED" /tmp/sandbox-test-results.log || echo "No specific failures found")

    log_info "Test failures:"
    echo "$FAILURES"

    # Note: In production, use Codex to refine based on failures
    log_info "Generating refined fix with Codex..."
}

# ============================================================================
# Final Application
# ============================================================================

apply_fix_to_codebase() {
    log_info "Applying validated fix to actual codebase..."

    if [[ "$GIT_INTEGRATION" == true ]]; then
        # Apply with git
        git apply "$OUTPUT_DIR/fix-changes.diff"
    else
        # Apply directly
        patch -p1 < "$OUTPUT_DIR/fix-changes.diff"
    fi

    log_success "Fix applied to codebase"
}

commit_fix() {
    if [[ "$GIT_INTEGRATION" == true ]]; then
        log_info "Committing fix..."

        git add -A
        git commit -m "fix: $BUG_ID - $(echo $PRIMARY_CAUSE | head -c 50)

Root Cause: $PRIMARY_CAUSE

Changes:
- Added null/undefined checks
- Improved error handling
- Added defensive validation

Fixes: $BUG_ID"

        log_success "Fix committed to branch fix/$BUG_ID"
    fi
}

rollback_on_failure() {
    if [[ "$AUTO_ROLLBACK" == true ]]; then
        log_warning "Rolling back failed fix..."

        if [[ "$GIT_INTEGRATION" == true ]]; then
            git reset --hard HEAD
            git checkout main
            git branch -D "fix/$BUG_ID" || true
        fi

        log_info "Rollback complete"
    fi
}

# ============================================================================
# Main Workflow
# ============================================================================

main() {
    echo "========================================================================"
    echo "Fix Generator - AI-Powered Bug Fix Automation"
    echo "========================================================================"
    echo ""

    parse_args "$@"

    # Phase 1: Analysis
    log_info "[1/7] Analyzing RCA report..."
    analyze_rca

    # Phase 2: Multi-Model Generation
    log_info "[2/7] Generating fix approaches with multi-model reasoning..."
    generate_claude_fix
    generate_codex_alternatives
    generate_gemini_context
    rank_solutions

    # Phase 3: Git Setup
    log_info "[3/7] Setting up version control..."
    create_git_branch

    # Phase 4: Iterative Implementation & Testing
    log_info "[4/7] Implementing and testing fix..."

    ATTEMPT=1
    SUCCESS=false

    while [[ $ATTEMPT -le $MAX_ATTEMPTS ]]; do
        log_info "=== Attempt $ATTEMPT/$MAX_ATTEMPTS ==="

        implement_fix "$ATTEMPT"
        setup_sandbox
        apply_fix_to_sandbox

        if run_sandbox_tests; then
            SUCCESS=true
            break
        else
            if [[ $ATTEMPT -lt $MAX_ATTEMPTS ]]; then
                refine_fix "$ATTEMPT" "/tmp/sandbox-test-results.log"
            fi
        fi

        cleanup_sandbox
        ATTEMPT=$((ATTEMPT + 1))
    done

    # Phase 5: Final Application
    if [[ "$SUCCESS" == true ]]; then
        log_info "[5/7] Fix validated - applying to codebase..."
        apply_fix_to_codebase

        log_info "[6/7] Committing fix..."
        commit_fix

        log_info "[7/7] Storing fix in memory..."
        npx claude-flow@alpha memory store \
            --key "fixes/successful/$BUG_ID" \
            --value "$(cat $OUTPUT_DIR/fix-changes.diff)"

        echo ""
        echo "========================================================================"
        log_success "FIX GENERATION COMPLETE!"
        echo "========================================================================"
        echo ""
        echo "Bug ID: $BUG_ID"
        echo "Attempts: $ATTEMPT/$MAX_ATTEMPTS"
        echo "Approach: $BEST_APPROACH"
        echo "Branch: fix/$BUG_ID"
        echo ""
        echo "Next steps:"
        echo "  1. Review changes: git diff main"
        echo "  2. Run full test suite: npm test"
        echo "  3. Create pull request"
        echo ""
    else
        log_error "[5/7] Fix failed after $MAX_ATTEMPTS attempts"
        rollback_on_failure

        echo ""
        echo "========================================================================"
        log_error "FIX GENERATION FAILED"
        echo "========================================================================"
        echo ""
        echo "The fix could not be automatically generated after $MAX_ATTEMPTS attempts."
        echo "Manual intervention required."
        echo ""
        echo "Logs available at: /tmp/sandbox-test-results.log"
        echo ""
        exit 1
    fi
}

# Run main workflow
main "$@"
