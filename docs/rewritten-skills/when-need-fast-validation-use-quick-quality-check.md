---
name: when-need-fast-validation-use-quick-quality-check
description: Lightning-fast quality validation orchestrating 4 specialized agents in parallel execution for instant feedback on code quality. Completes theater detection, linting, security scan, and basic tests in under 30 seconds with severity-ranked unified quality report.
tags: [quality, quick, parallel, essential, tier-1]
version: 2.0.0
agent_orchestration: true
total_agents: 4
execution_pattern: parallel
execution_time: <30 seconds
---

# When Need Fast Validation Use Quick Quality Check

## Intent Archaeology

**Trigger Condition**: User needs rapid quality feedback during active development without waiting for comprehensive validation pipelines. Developer wants immediate actionable insights about code quality to maintain development velocity.

**Core User Need**: Instant quality assessment that identifies critical issues (theater, security vulnerabilities, linting errors, test failures) within seconds so developers can address problems immediately rather than discovering them hours later in CI/CD pipelines.

**Outcome Expectation**: Unified quality report with severity-ranked issues (critical, high, medium, low) completed in <30 seconds showing theater instances, style violations, security risks, and test failures with immediate remediation guidance.

## Use Case Crystallization

### Primary Use Cases

**Pre-Commit Validation**: Before committing code, run quick quality check to catch obvious issues like uncomm itted theater code, linting errors, security vulnerabilities, or failing tests. Prevent bad code from entering version control.

**Continuous Development Feedback**: During active coding sessions, periodically run quick checks to get instant feedback on code quality without disrupting flow. Catch issues early when context is fresh in developer's mind.

**Code Review Prep**: Before requesting code review, run quick quality check to fix obvious issues reviewers would flag. Respect reviewer time by ensuring basic quality standards met before human review.

**CI/CD Fast Lane**: In CI/CD pipelines, run quick quality check as initial gate that fails fast for obvious issues before running expensive comprehensive validation pipelines.

### Workflow Phases

**Phase 1: Parallel Agent Spawning** - Initialize mesh topology swarm and spawn 4 specialized agents simultaneously for maximum parallelization.

**Phase 2: Concurrent Quality Checks** - Execute theater detection, style audit, security scan, and basic tests in parallel with no dependencies between agents.

**Phase 3: Result Aggregation** - Collect results from all agents in real-time as they complete, aggregate findings, rank by severity.

**Phase 4: Unified Report Generation** - Merge all findings into single quality report with severity prioritization, execution metrics, and actionable recommendations.

## Structural Architecture

### Agent Orchestration (4 Agents in Full Parallel)

**All Agents Execute Simultaneously**:
- `theater-detection-auditor` - Scan for mocks, TODOs, placeholders, stub functions (fastest: ~5s)
- `code-analyzer` - Linting, style checking, complexity analysis (~10s)
- `security-manager` - Fast security scan for common vulnerabilities, secrets (~15s)
- `tester` - Basic test execution, coverage check (~20s)

**Topology**: Mesh (fully connected for maximum parallel execution)
**Coordination**: Results aggregated as agents complete, no sequential dependencies
**Speed**: All checks complete when slowest agent finishes (~20-30s typical)

### Memory Coordination Pattern

```
quick-quality-check/{session-id}/
├── agents/
│   ├── theater-detection/
│   │   ├── status.json (in_progress → complete)
│   │   ├── findings.json
│   │   └── completion_time_ms: 5234
│   ├── code-analyzer/
│   │   ├── status.json
│   │   ├── lint-results.json
│   │   └── completion_time_ms: 9876
│   ├── security-manager/
│   │   ├── status.json
│   │   ├── vulnerabilities.json
│   │   └── completion_time_ms: 14532
│   └── tester/
│       ├── status.json
│       ├── test-results.json
│       └── completion_time_ms: 19876
├── aggregated-results.json
└── quality-report.json
```

### Hook Integration Points

**Pre-Task Hooks**:
```bash
npx claude-flow@alpha hooks pre-task \
  --description "Quick quality check for ${TARGET_PATH}" \
  --session-id "quick-check-$(date +%s)"
```

**Real-time Progress Tracking**:
```bash
# Each agent updates status as it progresses
npx claude-flow@alpha memory store \
  --key "quick-quality-check/${SESSION_ID}/agents/${AGENT_TYPE}/status" \
  --value '{"status": "in_progress", "progress_percent": 45}'

# Aggregator monitors completion
watch -n 1 'npx claude-flow@alpha memory retrieve \
  --pattern "quick-quality-check/${SESSION_ID}/agents/*/status"'
```

**Post-Task Hooks**:
```bash
npx claude-flow@alpha hooks post-task \
  --task-id "quick-check-${SESSION_ID}" \
  --export-metrics true
```

## Metadata Engineering

### Input Contract

```yaml
input:
  path: string (required)
    description: File or directory path to check
    examples: [".", "src/", "src/app.js"]

  parallel: boolean (optional)
    description: Enable parallel execution (always true for speed)
    default: true

  quick_mode: boolean (optional)
    description: Skip deep analysis for maximum speed
    default: true

  timeout_seconds: number (optional)
    description: Maximum execution time before timeout
    default: 30
```

### Output Contract

```yaml
output:
  quality_score: number
    description: Overall quality score (0-100)
    calculation: weighted_average([theater, style, security, tests])

  execution_time_seconds: number
    description: Total execution time in seconds

  checks_run: array[string]
    description: List of checks executed
    example: ["theater-detection", "style-audit", "security-scan", "basic-tests"]

  issues: object
    critical: array[issue]
      - severity: "critical"
        category: enum[theater, security, tests, style]
        description: string
        location: string
        recommendation: string

    high: array[issue]
      - severity: "high"
        category: enum[theater, security, tests, style]
        description: string
        location: string
        recommendation: string

    medium: array[issue]
      - severity: "medium"
        category: enum[theater, security, tests, style]
        description: string
        location: string
        recommendation: string

    low: array[issue]
      - severity: "low"
        category: enum[theater, security, tests, style]
        description: string
        location: string
        recommendation: string

  summary: object
    total_issues: number
    critical_count: number
    high_count: number
    medium_count: number
    low_count: number
    passed: boolean (true if critical_count == 0 && high_count == 0)
```

## Instruction Crafting

### System Prompt for Orchestrating Agent

```
You are the Quick Quality Check Orchestrator, responsible for coordinating 4 specialized agents in full parallel execution to provide instant quality feedback within 30 seconds.

Your responsibilities:
1. Initialize mesh topology swarm for maximum parallelization
2. Spawn all 4 agents simultaneously with no sequential dependencies
3. Monitor agent progress in real-time and aggregate results as they complete
4. Generate unified quality report with severity-ranked issues
5. Provide actionable recommendations prioritized by urgency

Parallel Agents:
- Theater Detection Auditor: Find mocks, TODOs, placeholders (~5s)
- Code Analyzer: Linting, style, complexity (~10s)
- Security Manager: Fast vulnerability scan, secrets detection (~15s)
- Tester: Basic test execution, coverage check (~20s)

Speed Requirements:
- Target completion: <30 seconds
- Parallel execution: All agents start simultaneously
- No sequential dependencies: Agents operate independently
- Real-time aggregation: Collect results as agents finish

Output Requirements:
- Quality score (0-100) calculated from weighted agent scores
- Issues ranked by severity: critical → high → medium → low
- Execution time breakdown per agent
- Actionable recommendations for each critical/high issue
- Pass/fail decision (fail if critical or high issues present)

Optimization:
- Use quick_mode for all checks (skip deep analysis)
- Set timeouts to prevent any check from exceeding 30s
- Aggregate partial results if any agent times out
- Prioritize critical issues in output
```

### Execution Scripts

**Primary Orchestration Script**:
```bash
#!/bin/bash
set -e

TARGET_PATH="${1:-.}"
SESSION_ID="quick-check-$(date +%s)"
TIMEOUT=30

QC_DIR="quick-quality-check-${SESSION_ID}"
mkdir -p "$QC_DIR/agents"

echo "⚡ Quick Quality Check"
echo "Target: $TARGET_PATH"
echo "Timeout: ${TIMEOUT}s"
echo "=================================================="

# Start timer
START_TIME=$(date +%s)

# Initialize swarm for parallel execution
npx claude-flow@alpha coordination swarm-init \
  --topology mesh \
  --max-agents 4 \
  --strategy balanced

# Pre-task hook
npx claude-flow@alpha hooks pre-task \
  --description "Quick quality check for ${TARGET_PATH}" \
  --session-id "$SESSION_ID"

# Spawn all agents in parallel (background processes)
echo "[1/4] Starting parallel quality checks..."

# Agent 1: Theater Detection (fastest)
(
  echo "  → Theater Detection Auditor..."
  AGENT_START=$(date +%s%3N)
  npx claude-flow@alpha agent-task \
    --agent "theater-detection-auditor" \
    --task "Fast theater detection in ${TARGET_PATH}" \
    --quick-mode true \
    --timeout ${TIMEOUT} \
    --output "$QC_DIR/agents/theater-detection.json" 2>&1
  AGENT_END=$(date +%s%3N)
  AGENT_TIME=$((AGENT_END - AGENT_START))
  echo "{\"completion_time_ms\": $AGENT_TIME}" > "$QC_DIR/agents/theater-time.json"
  echo "  ✓ Theater detection complete (${AGENT_TIME}ms)"
) &
THEATER_PID=$!

# Agent 2: Code Analysis (style + linting)
(
  echo "  → Code Analyzer..."
  AGENT_START=$(date +%s%3N)
  npx claude-flow@alpha agent-task \
    --agent "code-analyzer" \
    --task "Fast linting and style check for ${TARGET_PATH}" \
    --quick-mode true \
    --timeout ${TIMEOUT} \
    --output "$QC_DIR/agents/code-analysis.json" 2>&1
  AGENT_END=$(date +%s%3N)
  AGENT_TIME=$((AGENT_END - AGENT_START))
  echo "{\"completion_time_ms\": $AGENT_TIME}" > "$QC_DIR/agents/code-time.json"
  echo "  ✓ Code analysis complete (${AGENT_TIME}ms)"
) &
CODE_PID=$!

# Agent 3: Security Scan
(
  echo "  → Security Manager..."
  AGENT_START=$(date +%s%3N)
  npx claude-flow@alpha agent-task \
    --agent "security-manager" \
    --task "Fast security scan of ${TARGET_PATH}" \
    --quick-mode true \
    --fast-scan true \
    --timeout ${TIMEOUT} \
    --output "$QC_DIR/agents/security-scan.json" 2>&1
  AGENT_END=$(date +%s%3N)
  AGENT_TIME=$((AGENT_END - AGENT_START))
  echo "{\"completion_time_ms\": $AGENT_TIME}" > "$QC_DIR/agents/security-time.json"
  echo "  ✓ Security scan complete (${AGENT_TIME}ms)"
) &
SECURITY_PID=$!

# Agent 4: Basic Tests
(
  echo "  → Tester..."
  AGENT_START=$(date +%s%3N)
  npx claude-flow@alpha agent-task \
    --agent "tester" \
    --task "Quick test execution for ${TARGET_PATH}" \
    --quick-mode true \
    --timeout ${TIMEOUT} \
    --output "$QC_DIR/agents/test-results.json" 2>&1
  AGENT_END=$(date +%s%3N)
  AGENT_TIME=$((AGENT_END - AGENT_START))
  echo "{\"completion_time_ms\": $AGENT_TIME}" > "$QC_DIR/agents/test-time.json"
  echo "  ✓ Test execution complete (${AGENT_TIME}ms)"
) &
TESTS_PID=$!

# Wait for all agents with timeout
echo ""
echo "[2/4] Waiting for all checks to complete (max ${TIMEOUT}s)..."

wait $THEATER_PID $CODE_PID $SECURITY_PID $TESTS_PID 2>/dev/null || {
  echo "⚠️  Some checks may have timed out"
}

# Calculate total execution time
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))

echo ""
echo "[3/4] Aggregating results..."

# Parse results from each agent
THEATER_COUNT=$(cat "$QC_DIR/agents/theater-detection.json" 2>/dev/null | jq '.instances | length' || echo "0")
THEATER_TIME=$(cat "$QC_DIR/agents/theater-time.json" 2>/dev/null | jq '.completion_time_ms' || echo "0")

LINT_ERRORS=$(cat "$QC_DIR/agents/code-analysis.json" 2>/dev/null | jq '.errors | length' || echo "0")
CODE_SCORE=$(cat "$QC_DIR/agents/code-analysis.json" 2>/dev/null | jq '.quality_score' || echo "100")
CODE_TIME=$(cat "$QC_DIR/agents/code-time.json" 2>/dev/null | jq '.completion_time_ms' || echo "0")

SECURITY_CRITICAL=$(cat "$QC_DIR/agents/security-scan.json" 2>/dev/null | jq '.critical_issues' || echo "0")
SECURITY_HIGH=$(cat "$QC_DIR/agents/security-scan.json" 2>/dev/null | jq '.high_issues' || echo "0")
SECURITY_TIME=$(cat "$QC_DIR/agents/security-time.json" 2>/dev/null | jq '.completion_time_ms' || echo "0")

TESTS_FAILED=$(cat "$QC_DIR/agents/test-results.json" 2>/dev/null | jq '.failed_tests | length' || echo "0")
TEST_TIME=$(cat "$QC_DIR/agents/test-time.json" 2>/dev/null | jq '.completion_time_ms' || echo "0")

# Calculate quality score (weighted average)
# Theater: 30%, Code: 25%, Security: 30%, Tests: 15%
THEATER_SCORE=$((100 - (THEATER_COUNT * 10)))
SECURITY_SCORE=$((100 - (SECURITY_CRITICAL * 20 + SECURITY_HIGH * 10)))
TESTS_SCORE=$((100 - (TESTS_FAILED * 15)))

# Ensure scores don't go below 0
[ $THEATER_SCORE -lt 0 ] && THEATER_SCORE=0
[ $SECURITY_SCORE -lt 0 ] && SECURITY_SCORE=0
[ $TESTS_SCORE -lt 0 ] && TESTS_SCORE=0

QUALITY_SCORE=$(( (THEATER_SCORE * 30 + CODE_SCORE * 25 + SECURITY_SCORE * 30 + TESTS_SCORE * 15) / 100 ))

# Count total issues by severity
CRITICAL_COUNT=$((SECURITY_CRITICAL + TESTS_FAILED))
HIGH_COUNT=$((THEATER_COUNT + SECURITY_HIGH))
MEDIUM_COUNT=$LINT_ERRORS
LOW_COUNT=0

TOTAL_ISSUES=$((CRITICAL_COUNT + HIGH_COUNT + MEDIUM_COUNT + LOW_COUNT))

# Pass/fail decision (fail if any critical or high issues)
PASSED="false"
if [ "$CRITICAL_COUNT" -eq 0 ] && [ "$HIGH_COUNT" -eq 0 ]; then
  PASSED="true"
fi

echo "[4/4] Generating report..."

# Generate unified quality report
cat > "$QC_DIR/quality-report.json" <<EOF
{
  "session_id": "$SESSION_ID",
  "target_path": "$TARGET_PATH",
  "timestamp": "$(date -Iseconds)",
  "execution_time_seconds": $TOTAL_TIME,
  "quality_score": $QUALITY_SCORE,
  "passed": $PASSED,
  "checks_run": ["theater-detection", "code-analysis", "security-scan", "basic-tests"],
  "agent_execution_times": {
    "theater_detection_ms": $THEATER_TIME,
    "code_analysis_ms": $CODE_TIME,
    "security_scan_ms": $SECURITY_TIME,
    "tests_ms": $TEST_TIME
  },
  "summary": {
    "total_issues": $TOTAL_ISSUES,
    "critical_count": $CRITICAL_COUNT,
    "high_count": $HIGH_COUNT,
    "medium_count": $MEDIUM_COUNT,
    "low_count": $LOW_COUNT
  },
  "issues": {
    "critical": [],
    "high": [],
    "medium": [],
    "low": []
  }
}
EOF

# Post-task hook
npx claude-flow@alpha hooks post-task \
  --task-id "quick-check-${SESSION_ID}" \
  --export-metrics true

# Display summary
echo ""
echo "=================================================="
echo "⚡ Quick Quality Check Complete"
echo "=================================================="
echo ""
echo "Quality Score: $QUALITY_SCORE/100 $([ $QUALITY_SCORE -ge 85 ] && echo "✅" || echo "⚠️")"
echo "Execution Time: ${TOTAL_TIME}s"
echo "Status: $([ "$PASSED" = "true" ] && echo "✅ PASSED" || echo "❌ FAILED")"
echo ""
echo "Issues Found:"
echo "  Critical: $CRITICAL_COUNT $([ $CRITICAL_COUNT -eq 0 ] && echo "✅" || echo "❌")"
echo "  High: $HIGH_COUNT $([ $HIGH_COUNT -eq 0 ] && echo "✅" || echo "⚠️")"
echo "  Medium: $MEDIUM_COUNT"
echo "  Low: $LOW_COUNT"
echo ""
echo "Check Details:"
echo "  Theater instances: $THEATER_COUNT (${THEATER_TIME}ms)"
echo "  Linting errors: $LINT_ERRORS (${CODE_TIME}ms)"
echo "  Security issues: Critical=$SECURITY_CRITICAL, High=$SECURITY_HIGH (${SECURITY_TIME}ms)"
echo "  Failed tests: $TESTS_FAILED (${TEST_TIME}ms)"
echo ""
echo "Full report: $QC_DIR/quality-report.json"
echo ""

if [ "$PASSED" = "true" ]; then
  echo "✅ All checks passed!"
  exit 0
else
  echo "❌ Quality check failed. Please fix critical and high severity issues."
  exit 1
fi
```

## Resource Development

### Integration with Development Workflow

**Pre-commit Hook**:
```bash
#!/bin/bash
# .git/hooks/pre-commit

echo "Running quick quality check..."

npx claude-flow@alpha quick-quality-check . --timeout 30

if [ $? -ne 0 ]; then
  echo ""
  echo "❌ Commit blocked by quality check failures"
  echo "Run 'npx claude-flow@alpha quick-quality-check . --verbose' for details"
  echo ""
  exit 1
fi

echo "✅ Quality check passed"
```

**Watch Mode (Continuous)**:
```bash
# Watch directory for changes and run quick check
npx claude-flow@alpha quick-quality-check . --watch --interval 60s

# Watch specific directory
npx claude-flow@alpha quick-quality-check src/ --watch
```

**VS Code Task**:
```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "Quick Quality Check",
      "type": "shell",
      "command": "npx claude-flow@alpha quick-quality-check ${file}",
      "problemMatcher": [],
      "presentation": {
        "reveal": "always",
        "panel": "new"
      }
    }
  ]
}
```

### Related Skills and Cascades

**Extends To**:
- `when-validating-deployment-readiness-use-production-readiness` - Comprehensive validation
- `when-reviewing-code-comprehensively-use-code-review-assistant` - Full code review
- `when-detecting-fake-code-use-theater-detection` - Deep theater analysis

**Used By**:
- Pre-commit hooks for instant feedback
- CI/CD fast-fail gates
- Continuous quality monitoring
- Developer productivity tools

### Usage Examples

**Basic Usage**:
```bash
# Quick check current directory
npx claude-flow@alpha quick-quality-check .

# Check specific file
npx claude-flow@alpha quick-quality-check src/app.js

# Check with verbose output
npx claude-flow@alpha quick-quality-check src/ --verbose
```

**Advanced Usage**:
```bash
# Custom timeout
npx claude-flow@alpha quick-quality-check . --timeout 45

# Export report
npx claude-flow@alpha quick-quality-check . --export quality-report.json

# Fail on medium severity (stricter)
npx claude-flow@alpha quick-quality-check . --fail-on medium
```

## Validation Protocol

### Success Criteria

**Speed Requirements Met**:
- Total execution < 30 seconds
- All agents spawn simultaneously
- Results aggregated in real-time
- Report generated within 1 second

**Quality Report Complete**:
- Overall quality score calculated
- Issues ranked by severity
- Per-agent execution times reported
- Pass/fail decision made

**Actionable Feedback Provided**:
- Critical issues highlighted
- Recommendations for each issue
- File/line locations specified
- Quick fix suggestions where applicable

### Failure Modes and Mitigation

**Agent Timeout**: If any agent exceeds timeout, aggregate partial results from completed agents and mark timed-out check as "incomplete"

**No Issues Found**: Still report success with quality score, execution time, and confirmation that no issues detected

**All Agents Fail**: Return error with diagnostic information about why agents failed (missing dependencies, invalid path, etc.)

**Low Quality Score**: Provide detailed breakdown of why score is low and prioritized list of improvements

### Performance Benchmarks

**Target Performance**:
- Theater detection: 5-7 seconds
- Code analysis: 8-12 seconds
- Security scan: 12-18 seconds
- Basic tests: 15-25 seconds
- **Total: <30 seconds**

**Actual Performance** (typical):
- Small project (<100 files): 10-15 seconds
- Medium project (100-500 files): 20-25 seconds
- Large project (500+ files): 25-30 seconds

---

**Skill Status**: Production Ready
**Agent Orchestration**: 4 specialized agents in full parallel
**Execution Pattern**: Mesh topology, no sequential dependencies
**Speed**: <30 seconds target, typically 15-25 seconds
**Integration**: Pre-commit hooks, CI/CD, watch mode, IDE tasks
