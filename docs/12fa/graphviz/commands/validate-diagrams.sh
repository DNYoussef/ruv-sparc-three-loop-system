#!/bin/bash
# Validation script for command Graphviz diagrams

echo "=== Command Diagram Validation Report ==="
echo "Date: $(date)"
echo ""

DIAGRAM_DIR="C:/Users/17175/docs/12fa/graphviz/commands"
TOTAL=28
SUCCESS=0
FAILED=0

COMMANDS=(
  "claude-flow-swarm"
  "claude-flow-memory"
  "sparc"
  "sparc-code"
  "sparc-integration"
  "sparc-security-review"
  "sparc-spec-pseudocode"
  "sparc-debug"
  "sparc-sparc"
  "sparc-refinement-optimization-mode"
  "audit-pipeline"
  "sparc-devops"
  "theater-detect"
  "functionality-audit"
  "sparc-ask"
  "gemini-megacontext"
  "codex-auto"
  "sparc-mcp"
  "agent-rca"
  "sparc-docs-writer"
  "create-micro-skill"
  "sparc-post-deployment-monitoring-mode"
  "create-cascade"
  "fix-bug"
  "gemini-search"
  "quick-check"
  "build-feature"
  "gemini-media"
)

echo "Validating $TOTAL command diagrams..."
echo ""

for cmd in "${COMMANDS[@]}"; do
  FILE="${DIAGRAM_DIR}/${cmd}-process.dot"

  if [ -f "$FILE" ]; then
    # Check if file has content
    if [ -s "$FILE" ]; then
      # Try to validate with dot (if available)
      if command -v dot &> /dev/null; then
        if dot -Tsvg "$FILE" -o /dev/null 2>&1; then
          echo "✓ ${cmd}-process.dot - Valid"
          ((SUCCESS++))
        else
          echo "✗ ${cmd}-process.dot - Invalid syntax"
          ((FAILED++))
        fi
      else
        # Just check for basic structure
        if grep -q "digraph" "$FILE" && grep -q "}" "$FILE"; then
          echo "✓ ${cmd}-process.dot - Exists (syntax not validated)"
          ((SUCCESS++))
        else
          echo "✗ ${cmd}-process.dot - Missing required structure"
          ((FAILED++))
        fi
      fi
    else
      echo "✗ ${cmd}-process.dot - Empty file"
      ((FAILED++))
    fi
  else
    echo "✗ ${cmd}-process.dot - Not found"
    ((FAILED++))
  fi
done

echo ""
echo "=== Summary ==="
echo "Total expected: $TOTAL"
echo "Success: $SUCCESS"
echo "Failed: $FAILED"
echo ""

if [ $SUCCESS -eq $TOTAL ]; then
  echo "✓ All command diagrams validated successfully!"
  exit 0
else
  echo "✗ Some diagrams failed validation"
  exit 1
fi
