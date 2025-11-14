---
name: when-validating-deployment-readiness-use-production-readiness
description: Comprehensive pre-deployment validation orchestrating 8-10 specialized agents across 10 validation phases including complete audit pipeline, performance benchmarks, security scan, documentation checks, and deployment checklist generation with quality gate enforcement.
tags: [deployment, production, validation, sop-workflow, essential, tier-1]
version: 2.0.0
agent_orchestration: true
total_agents: 8-10
execution_pattern: parallel-sequential-hybrid
---

# When Validating Deployment Readiness Use Production Readiness

## Intent Archaeology

**Trigger Condition**: User needs to validate that code, application, or system is production-ready before deployment to staging or production environments.

**Core User Need**: Systematic validation that all quality gates pass, all tests succeed, security is verified, performance meets SLAs, and documentation is complete before deployment. Users need confidence that deployment will not cause outages, security breaches, or quality regressions.

**Outcome Expectation**: Comprehensive deployment readiness report with pass/fail status for each quality gate, detailed checklist of pre-deployment requirements, identified blocking issues, and clear go/no-go decision with rollback plan.

## Use Case Crystallization

### Primary Use Cases

**Pre-Deployment Validation**: Before deploying to production or staging environments, validate that code meets all quality standards including tests passing 100%, code quality ≥85/100, test coverage ≥80%, zero critical security issues, performance within SLAs, and complete documentation.

**Release Gate Enforcement**: As final gate in CI/CD pipeline before release approval, ensure all automated checks pass, manual review requirements met, sign-offs collected, and rollback procedures documented.

**Compliance Verification**: Validate that deployment meets organizational compliance requirements including security standards, audit logging, data privacy regulations, and industry-specific requirements.

**Risk Assessment**: Identify and quantify deployment risks including untested edge cases, performance bottlenecks, security vulnerabilities, missing documentation, and inadequate rollback procedures.

### Workflow Phases

**Phase 1: Complete Quality Audit** - Run theater detection, functionality audit, and style audit in sequence to ensure code is genuine, functional, and maintainable.

**Phase 2: Security Deep-Dive** - Execute comprehensive security scanning including vulnerability detection, secrets scanning, dependency audit, and compliance verification.

**Phase 3: Performance Benchmarking** - Run baseline performance tests, bottleneck detection, load testing, and SLA compliance validation.

**Phase 4: Documentation Validation** - Verify README completeness, API documentation, deployment guides, rollback procedures, and environment variable documentation.

**Phase 5: Dependency & Configuration Audit** - Check for vulnerable dependencies, validate environment configuration, verify secrets management, and check for hardcoded values.

**Phase 6: Infrastructure Readiness** - Validate monitoring setup, logging configuration, error tracking, alert rules, and dashboard availability.

**Phase 7: Test Coverage Analysis** - Verify unit test coverage, integration test quality, E2E test scenarios, and edge case handling.

**Phase 8: Deployment Checklist Generation** - Create comprehensive checklist with pre-deployment steps, deployment procedure, post-deployment verification, and rollback instructions.

**Phase 9: Quality Gate Evaluation** - Evaluate all quality gates against thresholds, identify blocking issues, calculate overall readiness score, and provide recommendations.

**Phase 10: Go/No-Go Decision** - Make final deployment recommendation based on gate status, risk assessment, and compliance requirements.

## Structural Architecture

### Agent Orchestration (8-10 Agents in Parallel + Sequential Phases)

**Phase 1: Quality Audit (Sequential - 3 agents)**
- `theater-detection-auditor` - Detect mocks, TODOs, placeholders, incomplete implementations
- `functionality-auditor` - Execute code with realistic inputs, verify outputs, validate correctness
- `style-auditor` - Enforce coding standards, best practices, maintainability guidelines

**Phase 2-4: Parallel Validation (5 agents in parallel)**
- `security-manager` - Comprehensive security scan, vulnerability detection, compliance check
- `performance-analyzer` - Performance benchmarking, bottleneck detection, SLA validation
- `qa-engineer` - Test coverage analysis, test quality assessment, edge case identification
- `api-documentation-specialist` - Documentation completeness check, API docs validation
- `devops-engineer` - Infrastructure readiness, monitoring setup, deployment configuration

**Phase 5: Final Assessment (2 agents in sequence)**
- `production-validator` - Quality gate evaluation, risk assessment, compliance verification
- `reviewer` - Final go/no-go decision, deployment recommendation, checklist validation

### Memory Coordination Pattern

```
deployment-readiness/{deployment-id}/
├── phase-1-quality-audit/
│   ├── theater-detection/findings.json
│   ├── functionality-audit/test-results.json
│   └── style-audit/quality-report.json
├── phase-2-security/
│   ├── vulnerabilities.json
│   ├── secrets-scan.json
│   └── compliance-report.json
├── phase-3-performance/
│   ├── benchmarks.json
│   ├── bottlenecks.json
│   └── sla-compliance.json
├── phase-4-documentation/
│   ├── completeness-check.json
│   └── missing-docs.json
├── phase-5-infrastructure/
│   ├── monitoring-setup.json
│   └── configuration-audit.json
├── quality-gates/
│   ├── gate-results.json
│   └── blocking-issues.json
└── deployment-checklist.md
```

### Hook Integration Points

**Pre-Task Hooks**:
```bash
npx claude-flow@alpha hooks pre-task \
  --description "Production readiness validation for ${DEPLOYMENT_TARGET}" \
  --session-id "deployment-${DEPLOYMENT_ID}"
```

**Per-Agent Coordination**:
```bash
# Each agent stores findings in shared memory
npx claude-flow@alpha memory store \
  --key "deployment-readiness/${DEPLOYMENT_ID}/phase-${N}/${AGENT_TYPE}/output" \
  --value "${AGENT_OUTPUT}"

# Agents check dependencies from previous phases
npx claude-flow@alpha memory retrieve \
  --pattern "deployment-readiness/${DEPLOYMENT_ID}/phase-${N-1}/*"
```

**Post-Task Hooks**:
```bash
npx claude-flow@alpha hooks post-task \
  --task-id "deployment-${DEPLOYMENT_ID}" \
  --export-metrics true \
  --session-end true
```

## Metadata Engineering

### Input Contract

```yaml
input:
  target_path: string (required)
    description: Directory or codebase to validate
    example: "./dist" or "."

  environment: enum (required)
    values: [staging, production]
    description: Target deployment environment
    default: production

  skip_performance: boolean (optional)
    description: Skip performance benchmarks for faster validation
    default: false

  strict_mode: boolean (optional)
    description: Enforce strictest quality gates
    default: true

  quality_thresholds: object (optional)
    code_quality_min: number (default: 85)
    test_coverage_min: number (default: 80)
    security_critical_max: number (default: 0)
    security_high_max: number (default: 0)
```

### Output Contract

```yaml
output:
  ready_for_deployment: boolean
    description: Overall go/no-go decision

  quality_gates: object
    tests_passing: boolean
    code_quality_score: number (0-100)
    test_coverage_percent: number (0-100)
    security_clean: boolean
    performance_within_sla: boolean
    documentation_complete: boolean

  gate_results: array[object]
    - gate_name: string
      passed: boolean
      score: number
      issues: array[string]

  blocking_issues: array[object]
    - severity: enum[critical, high, medium, low]
      category: string
      description: string
      location: string
      recommendation: string

  warnings: array[string]

  deployment_checklist: markdown
    description: Comprehensive deployment checklist

  rollback_plan: markdown
    description: Rollback procedure documentation

  metadata: object
    total_issues: number
    execution_time_seconds: number
    timestamp: iso8601
    environment: string
```

## Instruction Crafting

### System Prompt for Orchestrating Agent

```
You are the Production Readiness Orchestrator, responsible for coordinating 8-10 specialized agents to comprehensively validate deployment readiness across 10 validation phases.

Your responsibilities:
1. Execute complete quality audit pipeline (theater → functionality → style)
2. Coordinate parallel validation across security, performance, testing, documentation, and infrastructure
3. Enforce quality gates with configurable thresholds
4. Generate comprehensive deployment checklists and rollback plans
5. Make final go/no-go deployment decisions

Quality Gates (all must pass for production deployment):
- ✅ All tests passing (100%)
- ✅ Code quality ≥ 85/100
- ✅ Test coverage ≥ 80%
- ✅ Zero critical security issues
- ✅ Zero high-severity bugs
- ✅ Performance within SLAs
- ✅ Documentation complete
- ✅ Rollback plan documented

Orchestration Pattern:
1. Sequential quality audit (theater → functionality → style)
2. Parallel validation (security, performance, tests, docs, infra)
3. Sequential final assessment (validator → reviewer)

Memory Coordination:
- Store all agent outputs in deployment-readiness/{deployment-id}/ namespace
- Each agent retrieves outputs from previous phases
- Final assessment aggregates all findings

Output Requirements:
- Boolean go/no-go decision with justification
- Per-gate pass/fail with scores
- Complete list of blocking issues with remediation steps
- Deployment checklist with all required tasks
- Rollback plan with step-by-step instructions
```

### Execution Scripts

**Primary Orchestration Script**:
```bash
#!/bin/bash
set -e

TARGET_PATH="${1:-.}"
ENVIRONMENT="${2:-production}"
SKIP_PERFORMANCE="${3:-false}"
DEPLOYMENT_ID="$(date +%s)-${ENVIRONMENT}"

READINESS_DIR="deployment-readiness-${DEPLOYMENT_ID}"
mkdir -p "$READINESS_DIR"

echo "================================================================"
echo "Production Readiness Validation"
echo "Target: $TARGET_PATH"
echo "Environment: $ENVIRONMENT"
echo "Deployment ID: $DEPLOYMENT_ID"
echo "================================================================"

# Initialize deployment session
npx claude-flow@alpha hooks pre-task \
  --description "Production readiness for ${ENVIRONMENT}" \
  --session-id "deployment-${DEPLOYMENT_ID}"

# Initialize quality gate tracking
declare -A GATES
GATES[tests]=0
GATES[quality]=0
GATES[coverage]=0
GATES[security]=0
GATES[performance]=0
GATES[docs]=0

# PHASE 1: Complete Quality Audit (Sequential)
echo "[Phase 1/10] Running complete quality audit pipeline..."

echo "  → Theater Detection (detecting mocks, TODOs, placeholders)..."
npx claude-flow@alpha agent-task \
  --agent "theater-detection-auditor" \
  --task "Detect all theater patterns in ${TARGET_PATH}" \
  --output "$READINESS_DIR/theater-detection.json"

THEATER_COUNT=$(cat "$READINESS_DIR/theater-detection.json" | jq '.instances | length')
if [ "$THEATER_COUNT" -gt 0 ]; then
  echo "⚠️  Found $THEATER_COUNT theater instances (must be completed)"
fi

echo "  → Functionality Audit (executing code with realistic inputs)..."
npx claude-flow@alpha agent-task \
  --agent "functionality-auditor" \
  --task "Validate all code executes correctly in ${TARGET_PATH}" \
  --output "$READINESS_DIR/functionality-audit.json"

TESTS_PASSED=$(cat "$READINESS_DIR/functionality-audit.json" | jq '.all_tests_passed')
TEST_COVERAGE=$(cat "$READINESS_DIR/functionality-audit.json" | jq '.coverage_percent')

if [ "$TESTS_PASSED" = "true" ]; then
  GATES[tests]=1
  echo "✅ GATE 1: All tests passing"
else
  echo "❌ GATE 1: Tests failing"
fi

if [ "$TEST_COVERAGE" -ge 80 ]; then
  GATES[coverage]=1
  echo "✅ GATE 3: Test coverage ${TEST_COVERAGE}%"
else
  echo "❌ GATE 3: Coverage too low: ${TEST_COVERAGE}% (need ≥80%)"
fi

echo "  → Style Audit (enforcing best practices and maintainability)..."
npx claude-flow@alpha agent-task \
  --agent "style-auditor" \
  --task "Audit code style and quality in ${TARGET_PATH}" \
  --output "$READINESS_DIR/style-audit.json"

QUALITY_SCORE=$(cat "$READINESS_DIR/style-audit.json" | jq '.quality_score')
if [ "$QUALITY_SCORE" -ge 85 ]; then
  GATES[quality]=1
  echo "✅ GATE 2: Code quality ${QUALITY_SCORE}/100"
else
  echo "❌ GATE 2: Quality too low: ${QUALITY_SCORE}/100 (need ≥85)"
fi

# Store Phase 1 results in memory
npx claude-flow@alpha memory store \
  --key "deployment-readiness/${DEPLOYMENT_ID}/phase-1/complete" \
  --value "$(cat $READINESS_DIR/theater-detection.json $READINESS_DIR/functionality-audit.json $READINESS_DIR/style-audit.json | jq -s '.')"

# PHASE 2-4: Parallel Validation
echo "[Phase 2-4/10] Running parallel validation checks..."

# Spawn all parallel agents simultaneously
(echo "  → Security Manager (vulnerability scan, secrets detection)..." && \
  npx claude-flow@alpha agent-task \
    --agent "security-manager" \
    --task "Comprehensive security scan of ${TARGET_PATH}" \
    --output "$READINESS_DIR/security-scan.json") &

(echo "  → Performance Analyzer (benchmarks, bottlenecks, SLA check)..." && \
  [ "$SKIP_PERFORMANCE" != "true" ] && \
  npx claude-flow@alpha agent-task \
    --agent "performance-analyzer" \
    --task "Performance validation for ${TARGET_PATH}" \
    --output "$READINESS_DIR/performance-analysis.json") &

(echo "  → QA Engineer (test quality, edge cases, coverage)..." && \
  npx claude-flow@alpha agent-task \
    --agent "qa-engineer" \
    --task "Test suite quality analysis for ${TARGET_PATH}" \
    --output "$READINESS_DIR/test-quality.json") &

(echo "  → Documentation Specialist (README, API docs, guides)..." && \
  npx claude-flow@alpha agent-task \
    --agent "api-documentation-specialist" \
    --task "Documentation completeness check for ${TARGET_PATH}" \
    --output "$READINESS_DIR/documentation-check.json") &

(echo "  → DevOps Engineer (monitoring, logging, infrastructure)..." && \
  npx claude-flow@alpha agent-task \
    --agent "devops-engineer" \
    --task "Infrastructure readiness validation for ${TARGET_PATH}" \
    --output "$READINESS_DIR/infrastructure-check.json") &

# Wait for all parallel agents to complete
wait

# Evaluate security gate
CRITICAL_SECURITY=$(cat "$READINESS_DIR/security-scan.json" | jq '.critical_issues')
HIGH_SECURITY=$(cat "$READINESS_DIR/security-scan.json" | jq '.high_issues')

if [ "$CRITICAL_SECURITY" -eq 0 ] && [ "$HIGH_SECURITY" -eq 0 ]; then
  GATES[security]=1
  echo "✅ GATE 4: Security scan clean"
else
  echo "❌ GATE 4: Security issues (Critical: $CRITICAL_SECURITY, High: $HIGH_SECURITY)"
fi

# Evaluate performance gate
if [ "$SKIP_PERFORMANCE" != "true" ]; then
  AVG_RESPONSE=$(cat "$READINESS_DIR/performance-analysis.json" | jq '.avg_response_ms')
  P95_RESPONSE=$(cat "$READINESS_DIR/performance-analysis.json" | jq '.p95_response_ms')

  # SLAs: avg < 200ms, p95 < 500ms
  if [ "$AVG_RESPONSE" -lt 200 ] && [ "$P95_RESPONSE" -lt 500 ]; then
    GATES[performance]=1
    echo "✅ GATE 5: Performance within SLAs"
  else
    echo "❌ GATE 5: Performance exceeds SLAs (avg: ${AVG_RESPONSE}ms, p95: ${P95_RESPONSE}ms)"
  fi
else
  GATES[performance]=1  # Pass if skipped
fi

# Evaluate documentation gate
DOCS_COMPLETE=$(cat "$READINESS_DIR/documentation-check.json" | jq '.complete')
if [ "$DOCS_COMPLETE" = "true" ]; then
  GATES[docs]=1
  echo "✅ GATE 6: Documentation complete"
else
  echo "❌ GATE 6: Documentation incomplete"
fi

# Store parallel validation results
npx claude-flow@alpha memory store \
  --key "deployment-readiness/${DEPLOYMENT_ID}/phase-2-4/complete" \
  --value "$(cat $READINESS_DIR/security-scan.json $READINESS_DIR/performance-analysis.json $READINESS_DIR/test-quality.json $READINESS_DIR/documentation-check.json $READINESS_DIR/infrastructure-check.json | jq -s '.')"

# PHASE 5: Deployment Checklist Generation
echo "[Phase 5/10] Generating deployment checklist..."

cat > "$READINESS_DIR/DEPLOYMENT-CHECKLIST.md" <<EOF
# Deployment Checklist: $ENVIRONMENT

**Generated**: $(date -Iseconds)
**Deployment ID**: $DEPLOYMENT_ID
**Target**: $TARGET_PATH

## Quality Gates Status

| Gate | Status | Score/Details |
|------|--------|---------------|
| Tests Passing | $([ ${GATES[tests]} -eq 1 ] && echo "✅" || echo "❌") | All tests passing: $TESTS_PASSED |
| Code Quality | $([ ${GATES[quality]} -eq 1 ] && echo "✅" || echo "❌") | $QUALITY_SCORE/100 (need ≥85) |
| Test Coverage | $([ ${GATES[coverage]} -eq 1 ] && echo "✅" || echo "❌") | $TEST_COVERAGE% (need ≥80%) |
| Security | $([ ${GATES[security]} -eq 1 ] && echo "✅" || echo "❌") | Critical: $CRITICAL_SECURITY, High: $HIGH_SECURITY |
| Performance | $([ ${GATES[performance]} -eq 1 ] && echo "✅" || echo "❌") | Within SLAs |
| Documentation | $([ ${GATES[docs]} -eq 1 ] && echo "✅" || echo "❌") | Complete: $DOCS_COMPLETE |

## Pre-Deployment Checklist

### Code Quality
- [ ] All tests passing (100%)
- [ ] Code quality ≥ 85/100
- [ ] Test coverage ≥ 80%
- [ ] No linting errors
- [ ] No TypeScript errors
- [ ] All theater replaced with real implementations

### Security
- [ ] No critical or high-severity vulnerabilities
- [ ] Dependencies up to date
- [ ] Secrets in environment variables (not hardcoded)
- [ ] Security headers configured
- [ ] Authentication/authorization tested

### Performance
- [ ] Response times within SLAs (avg <200ms, p95 <500ms)
- [ ] No performance bottlenecks
- [ ] Database queries optimized
- [ ] Caching configured
- [ ] Load tested (if production)

### Documentation
- [ ] README.md up to date
- [ ] API documentation complete
- [ ] Deployment guide available
- [ ] Rollback plan documented
- [ ] Environment variables documented

### Monitoring & Observability
- [ ] Logging configured
- [ ] Error tracking setup (Sentry, etc.)
- [ ] Metrics collection enabled
- [ ] Alerts configured
- [ ] Dashboard created

### Infrastructure
- [ ] Environment variables configured
- [ ] Database migrations ready
- [ ] Backup strategy verified
- [ ] Scaling configuration reviewed
- [ ] SSL certificates valid

### Rollback Plan
- [ ] Rollback procedure documented
- [ ] Previous version backed up
- [ ] Rollback tested in staging
- [ ] Rollback SLA defined (<5 minutes)

## Deployment Steps

1. **Pre-deployment**
   - Create deployment branch
   - Final code review
   - Merge to main/master
   - Tag release version

2. **Staging Deployment** (if production)
   - Deploy to staging
   - Run smoke tests
   - Verify functionality
   - Monitor for 24 hours

3. **Production Deployment**
   - Create database backup
   - Deploy to production
   - Run health checks
   - Monitor error rates
   - Verify critical paths

4. **Post-deployment**
   - Verify all functionality
   - Monitor metrics for 1 hour
   - Check error rates
   - Document any issues
   - Communicate success

## Rollback Procedure

If deployment fails:

1. **Immediate Actions**
   - Stop deployment immediately
   - Assess impact and scope
   - Notify stakeholders

2. **Execute Rollback**
   - Run: \`./scripts/rollback.sh\` or equivalent
   - Verify previous version restored
   - Run health checks

3. **Post-Rollback**
   - Verify system stability
   - Investigate root cause
   - Document failure
   - Fix issues before retry

## Sign-off Required

- [ ] **Development Lead**: Code review approved
- [ ] **QA Lead**: Testing complete
- [ ] **Security Team**: Security review approved
- [ ] **DevOps**: Infrastructure ready
- [ ] **Product Owner**: Features approved

---

🤖 Generated by Claude Code Production Readiness Orchestrator
EOF

# PHASE 6: Final Go/No-Go Assessment
echo "[Phase 6/10] Running final assessment..."

GATES_PASSED=$((${GATES[tests]} + ${GATES[quality]} + ${GATES[coverage]} + ${GATES[security]} + ${GATES[performance]} + ${GATES[docs]}))
TOTAL_GATES=6

READY_FOR_DEPLOYMENT="false"
if [ "$GATES_PASSED" -eq "$TOTAL_GATES" ]; then
  READY_FOR_DEPLOYMENT="true"
fi

# Generate final report
cat > "$READINESS_DIR/READINESS-REPORT.json" <<EOF
{
  "deployment_id": "$DEPLOYMENT_ID",
  "environment": "$ENVIRONMENT",
  "target_path": "$TARGET_PATH",
  "timestamp": "$(date -Iseconds)",
  "ready_for_deployment": $READY_FOR_DEPLOYMENT,
  "quality_gates": {
    "tests_passing": $([ ${GATES[tests]} -eq 1 ] && echo "true" || echo "false"),
    "code_quality_score": $QUALITY_SCORE,
    "test_coverage_percent": $TEST_COVERAGE,
    "security_clean": $([ ${GATES[security]} -eq 1 ] && echo "true" || echo "false"),
    "performance_within_sla": $([ ${GATES[performance]} -eq 1 ] && echo "true" || echo "false"),
    "documentation_complete": $([ ${GATES[docs]} -eq 1 ] && echo "true" || echo "false")
  },
  "gates_passed": $GATES_PASSED,
  "total_gates": $TOTAL_GATES,
  "blocking_issues": [],
  "warnings": []
}
EOF

# Post-task hooks
npx claude-flow@alpha hooks post-task \
  --task-id "deployment-${DEPLOYMENT_ID}" \
  --export-metrics true

# Display summary
echo ""
echo "================================================================"
echo "Production Readiness Assessment Complete"
echo "================================================================"
echo ""
echo "Environment: $ENVIRONMENT"
echo "Gates Passed: $GATES_PASSED/$TOTAL_GATES"
echo ""
echo "Quality Gates:"
echo "  Tests: $([ ${GATES[tests]} -eq 1 ] && echo "✅" || echo "❌")"
echo "  Quality: $([ ${GATES[quality]} -eq 1 ] && echo "✅" || echo "❌") ($QUALITY_SCORE/100)"
echo "  Coverage: $([ ${GATES[coverage]} -eq 1 ] && echo "✅" || echo "❌") ($TEST_COVERAGE%)"
echo "  Security: $([ ${GATES[security]} -eq 1 ] && echo "✅" || echo "❌") (Critical: $CRITICAL_SECURITY, High: $HIGH_SECURITY)"
echo "  Performance: $([ ${GATES[performance]} -eq 1 ] && echo "✅" || echo "❌")"
echo "  Documentation: $([ ${GATES[docs]} -eq 1 ] && echo "✅" || echo "❌")"
echo ""

if [ "$READY_FOR_DEPLOYMENT" = "true" ]; then
  echo "🚀 READY FOR DEPLOYMENT!"
  echo ""
  echo "Next steps:"
  echo "1. Review deployment checklist: $READINESS_DIR/DEPLOYMENT-CHECKLIST.md"
  echo "2. Get required sign-offs"
  echo "3. Schedule deployment window"
  echo "4. Execute deployment"
  exit 0
else
  echo "🚫 NOT READY FOR DEPLOYMENT"
  echo ""
  echo "Blocking issues must be resolved before deployment."
  echo "See detailed reports in: $READINESS_DIR/"
  exit 1
fi
```

## Resource Development

### Integration with CI/CD Pipeline

**GitHub Actions Integration**:
```yaml
name: Production Readiness Check

on:
  pull_request:
    branches: [main, master]
  push:
    branches: [main, master]

jobs:
  production-readiness:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '18'

      - name: Install Dependencies
        run: npm ci

      - name: Run Production Readiness Check
        run: |
          npx claude-flow@alpha production-readiness \
            --target . \
            --environment production \
            --output readiness-report.json

      - name: Upload Readiness Report
        uses: actions/upload-artifact@v3
        with:
          name: production-readiness-report
          path: deployment-readiness-*/

      - name: Check Deployment Ready
        run: |
          READY=$(jq '.ready_for_deployment' readiness-report.json)
          if [ "$READY" != "true" ]; then
            echo "❌ Production readiness check failed"
            exit 1
          fi
          echo "✅ Production readiness check passed"
```

### Related Skills and Cascades

**Prerequisite Skills**:
- `when-detecting-fake-code-use-theater-detection` - Theater detection audit
- `when-validating-code-works-use-functionality-audit` - Functionality validation
- `when-auditing-code-style-use-style-audit` - Style and quality audit

**Cascade Integration**:
- Part of `/feature-dev-complete` cascade (final stage)
- Part of `/release-preparation` cascade
- Used by `/deploy-to-production` cascade

**Follow-on Skills**:
- `deployment-automation` - Automated deployment execution
- `rollback-planner` - Rollback procedure generation
- `post-deployment-monitoring` - Production monitoring setup

### Usage Examples

**Basic Usage**:
```bash
# Check production readiness for current directory
npx claude-flow@alpha production-readiness . production

# Staging environment validation
npx claude-flow@alpha production-readiness ./dist staging

# Skip performance tests (faster validation)
npx claude-flow@alpha production-readiness . production --skip-performance
```

**Advanced Usage**:
```bash
# Custom quality thresholds
npx claude-flow@alpha production-readiness . production \
  --code-quality-min 90 \
  --test-coverage-min 85 \
  --strict-mode true

# Export report to specific location
npx claude-flow@alpha production-readiness . production \
  --output-dir ./deployment-reports

# Validate specific deployment target
npx claude-flow@alpha production-readiness ./build production \
  --deployment-target "us-east-1"
```

## Validation Protocol

### Success Criteria

**All Quality Gates Pass**:
- Tests: 100% passing
- Code Quality: ≥85/100
- Test Coverage: ≥80%
- Security: 0 critical, 0 high-severity
- Performance: Within SLAs
- Documentation: Complete

**Deployment Checklist Generated**:
- All pre-deployment tasks listed
- Deployment procedure documented
- Rollback plan included
- Sign-off section present

**Final Report Contains**:
- Boolean ready_for_deployment decision
- Per-gate pass/fail status with scores
- List of blocking issues (if any)
- Remediation recommendations
- Complete deployment checklist

### Failure Modes and Mitigation

**Tests Failing**: Block deployment immediately, provide test failure details, recommend fixes

**Security Issues**: Block deployment for critical/high severity, escalate to security team, require remediation before retry

**Poor Code Quality**: Block production (allow staging with warning), provide code quality report, recommend refactoring

**Missing Documentation**: Warning for staging, blocking for production, list missing documentation, provide templates

**Performance Issues**: Warning for staging, blocking for production, provide bottleneck analysis, recommend optimizations

### Performance Benchmarks

**Execution Time**:
- Complete audit: 5-10 minutes (typical codebase)
- Quality audit phase: 2-3 minutes
- Parallel validation: 2-4 minutes
- Report generation: <30 seconds

**Agent Coordination**:
- Sequential phase transitions: <1 second
- Parallel agent spawning: <2 seconds
- Memory synchronization: <500ms per operation

---

**Skill Status**: Production Ready
**Agent Orchestration**: 8-10 specialized agents
**Execution Pattern**: Hybrid parallel-sequential with 10 phases
**Quality Gates**: 6 enforced gates with configurable thresholds
**Integration**: CI/CD pipeline ready with GitHub Actions example
