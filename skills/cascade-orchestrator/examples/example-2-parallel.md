# Example 2: Parallel Cascade - Code Quality Gate

## Overview

This example demonstrates a **parallel execution cascade** where multiple independent tasks run simultaneously. This pattern maximizes throughput by leveraging concurrent execution and multi-agent coordination.

## Use Case

Run comprehensive code quality checks (linting, security scanning, test coverage, complexity analysis) in parallel, then aggregate results into a single quality report before allowing code to merge.

## Cascade Definition

```yaml
cascade:
  name: code-quality-gate
  description: Parallel execution of code quality checks with aggregated reporting
  version: 1.0.0

  inputs:
    - name: repository
      type: string
      description: Git repository path or URL
      required: true

    - name: branch
      type: string
      description: Branch to analyze
      default: "main"

    - name: quality_threshold
      type: float
      description: Minimum quality score to pass (0-100)
      default: 80.0

  stages:
    - stage_id: checkout
      name: Checkout Repository
      model: auto-select
      skill: git-checkout
      inputs:
        repository: ${inputs.repository}
        branch: ${inputs.branch}
      outputs:
        - workspace_path
        - commit_sha

    - stage_id: parallel-quality-checks
      name: Run Quality Checks in Parallel
      type: parallel
      swarm_config:
        topology: mesh
        max_agents: 4
        strategy: balanced
        memory_shared: true
      skills:
        - skill_id: lint
          name: Code Linting
          model: claude
          skill: lint-code
          inputs:
            path: ${checkout.workspace_path}
            rules: "eslint:recommended"
            auto_fix: false
          outputs:
            - lint_errors
            - lint_warnings
            - lint_score

        - skill_id: security
          name: Security Scanning
          model: claude
          skill: security-scan
          inputs:
            path: ${checkout.workspace_path}
            scanners: ["snyk", "trivy", "semgrep"]
            severity_threshold: "medium"
          outputs:
            - vulnerabilities
            - security_score

        - skill_id: tests
          name: Test Coverage
          model: codex-auto
          skill: run-tests
          inputs:
            path: ${checkout.workspace_path}
            framework: "jest"
            coverage: true
          outputs:
            - test_results
            - coverage_percentage
            - test_score

        - skill_id: complexity
          name: Complexity Analysis
          model: claude
          skill: analyze-complexity
          inputs:
            path: ${checkout.workspace_path}
            metrics: ["cyclomatic", "cognitive", "maintainability"]
          outputs:
            - complexity_metrics
            - complexity_score

    - stage_id: aggregate
      name: Aggregate Quality Metrics
      model: claude
      skill: aggregate-metrics
      inputs:
        lint: ${parallel-quality-checks.lint}
        security: ${parallel-quality-checks.security}
        tests: ${parallel-quality-checks.tests}
        complexity: ${parallel-quality-checks.complexity}
        weights:
          lint: 0.25
          security: 0.35
          tests: 0.25
          complexity: 0.15
      outputs:
        - overall_score
        - detailed_report
        - pass_fail

    - stage_id: visualize
      name: Generate Quality Dashboard
      model: gemini-media
      skill: generate-dashboard
      inputs:
        metrics: ${aggregate.detailed_report}
        score: ${aggregate.overall_score}
        threshold: ${inputs.quality_threshold}
        commit: ${checkout.commit_sha}
      outputs:
        - dashboard_html
        - charts

    - stage_id: gate-decision
      name: Quality Gate Decision
      type: conditional
      condition: ${aggregate.overall_score} >= ${inputs.quality_threshold}
      branches:
        pass:
          - stage: notify-success
            skill: send-notification
            inputs:
              channel: "slack:#dev-team"
              message: "✅ Quality gate PASSED: ${aggregate.overall_score}/100"
              dashboard: ${visualize.dashboard_html}
        fail:
          - stage: block-merge
            skill: github-status-check
            inputs:
              status: "failure"
              context: "code-quality-gate"
              description: "Quality score ${aggregate.overall_score} below threshold ${inputs.quality_threshold}"

          - stage: create-issue
            skill: github-create-issue
            inputs:
              title: "Code Quality Below Threshold"
              body: ${visualize.dashboard_html}
              labels: ["quality", "needs-improvement"]

  memory:
    persistence: enabled
    scope: global
    keys:
      - quality_history
      - trend_analysis

  github_integration:
    create_status_check: true
    context: "code-quality-gate"
    report_as_comment: true
```

## Execution Flow

```
┌──────────────┐
│   Checkout   │
│   (Stage 1)  │
└──────┬───────┘
       │ workspace_path
       ▼
┌──────────────────────────────────────────────┐
│         Parallel Quality Checks              │
│              (Stage 2)                       │
│                                              │
│  ┌─────────┐  ┌──────────┐  ┌────────┐  ┌──────────┐
│  │  Lint   │  │ Security │  │ Tests  │  │Complexity│
│  │ Agent 1 │  │ Agent 2  │  │Agent 3 │  │ Agent 4  │
│  └────┬────┘  └─────┬────┘  └───┬────┘  └────┬─────┘
│       │             │            │            │
└───────┼─────────────┼────────────┼────────────┼──────┘
        │             │            │            │
        └─────────────┴────────────┴────────────┘
                      │
                      ▼
              ┌──────────────┐
              │  Aggregate   │
              │   (Stage 3)  │
              └──────┬───────┘
                     │ overall_score
                     ▼
              ┌──────────────┐
              │  Visualize   │
              │   (Stage 4)  │
              └──────┬───────┘
                     │ dashboard
                     ▼
              ┌──────────────┐
              │Gate Decision │
              │   (Stage 5)  │
              └──────┬───────┘
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
    ┌─────────┐            ┌──────────┐
    │  PASS   │            │   FAIL   │
    │ Notify  │            │Block+Issue
    └─────────┘            └──────────┘
```

## Invocation

```bash
# Via Claude Code
"Run the code quality gate cascade on repository https://github.com/company/api-service with threshold 85"

# Or simple form
"Check code quality in parallel for the current repository"
```

## Parallel Execution Details

### Swarm Configuration

```yaml
swarm_config:
  topology: mesh           # Peer-to-peer coordination
  max_agents: 4           # 4 concurrent quality checks
  strategy: balanced      # Even load distribution
  memory_shared: true     # Shared context across agents
```

### Agent Assignment

```javascript
// Via Claude Code Task tool
[Single Message - Parallel Agent Execution]:
  Task("Lint Agent", "Run ESLint on codebase. Report errors/warnings.", "coder")
  Task("Security Agent", "Scan for vulnerabilities with Snyk/Trivy/Semgrep.", "reviewer")
  Task("Test Agent", "Run Jest test suite with coverage analysis.", "tester")
  Task("Complexity Agent", "Analyze code complexity metrics.", "code-analyzer")
```

### Coordination via Hooks

Each agent runs:
```bash
# Before starting
npx claude-flow@alpha hooks pre-task --description "quality-check-[type]"

# During execution
npx claude-flow@alpha hooks post-edit --file "[report]" --memory-key "quality/[type]"

# After completion
npx claude-flow@alpha hooks post-task --task-id "quality-check-[type]"
```

## Parallel Results

### Individual Agent Outputs

```yaml
# Lint Agent Output
lint:
  errors: 12
  warnings: 45
  score: 82.5
  details:
    - file: "src/api/users.js"
      line: 42
      rule: "no-unused-vars"
      severity: "error"
    - file: "src/utils/helpers.js"
      line: 15
      rule: "complexity"
      severity: "warning"

# Security Agent Output
security:
  vulnerabilities:
    critical: 0
    high: 2
    medium: 5
    low: 12
  score: 75.0
  details:
    - package: "express@4.17.1"
      cve: "CVE-2022-24999"
      severity: "high"
      fix_available: true

# Test Agent Output
tests:
  total: 234
  passed: 228
  failed: 6
  coverage: 87.3
  score: 89.0
  details:
    - suite: "API Endpoints"
      passed: 45
      failed: 2
      coverage: 92.1

# Complexity Agent Output
complexity:
  cyclomatic_avg: 8.2
  cognitive_avg: 12.5
  maintainability: "A"
  score: 88.0
  hotspots:
    - file: "src/core/processor.js"
      function: "processData"
      cyclomatic: 23
      cognitive: 45
```

### Aggregated Result

```yaml
aggregate:
  overall_score: 83.6
  pass_fail: "pass"
  detailed_report:
    lint:
      score: 82.5
      weight: 0.25
      contribution: 20.6
    security:
      score: 75.0
      weight: 0.35
      contribution: 26.3
    tests:
      score: 89.0
      weight: 0.25
      contribution: 22.3
    complexity:
      score: 88.0
      weight: 0.15
      contribution: 13.2

  # overall_score = sum(contributions) = 82.4

  recommendations:
    - "Fix 2 high-severity security vulnerabilities"
    - "Reduce cyclomatic complexity in src/core/processor.js"
    - "Improve test coverage in src/utils/ directory"
```

## Performance Characteristics

### Timing Comparison

**Sequential Execution** (traditional):
```
Lint:       45s
Security:   60s
Tests:      90s
Complexity: 30s
────────────────
Total:      225s (3m 45s)
```

**Parallel Execution** (cascade):
```
All 4 tasks: 90s (limited by slowest task)
────────────────
Total:      90s (1m 30s)
Speedup:    2.5x faster
```

### Resource Usage

```yaml
cpu_usage:
  sequential: 25% (single core)
  parallel: 85% (4 cores)

memory_usage:
  sequential: 500MB
  parallel: 1.2GB (4 agents × 300MB)

token_usage:
  sequential: ~8000 tokens
  parallel: ~6000 tokens (32% reduction via shared context)
```

## Error Scenarios

### Scenario 1: One Agent Fails

```yaml
Problem: Security scan times out
Action: Other agents continue execution
Result:
  - Lint: ✅ Complete
  - Security: ❌ Timeout
  - Tests: ✅ Complete
  - Complexity: ✅ Complete

Fallback:
  - Use cached security results from previous run
  - Mark security score as "pending"
  - Retry security scan in background
```

### Scenario 2: Multiple Agent Failures

```yaml
Problem: Tests fail AND security scan fails
Action: Swarm recovery redistributes tasks
Result:
  - Spawn new test agent → Retry
  - Spawn new security agent → Retry
  - Continue with available results

Escalation:
  - If retries fail → Mark as "partial results"
  - Lower overall score confidence
  - Notify team of incomplete analysis
```

## When to Use This Pattern

**Best for:**
- Independent quality checks
- Code analysis tasks
- CI/CD pipelines
- Resource-intensive operations
- Time-sensitive workflows

**Not ideal for:**
- Tasks with dependencies (use sequential)
- Limited compute resources
- Shared state modifications
- Order-dependent operations

## Variations

### Variation 1: Dynamic Agent Scaling

```yaml
swarm_config:
  topology: mesh
  max_agents: auto  # Scale based on available resources
  strategy: adaptive
  scale_rules:
    - if: cpu_usage < 50% → spawn more agents
    - if: memory > 80% → reduce agents
```

### Variation 2: Weighted Prioritization

```yaml
parallel-quality-checks:
  skills:
    - skill: security-scan
      priority: critical  # Run first with more resources

    - skill: lint-code
      priority: high

    - skill: run-tests
      priority: medium

    - skill: analyze-complexity
      priority: low  # Run last if resources available
```

### Variation 3: Progressive Results

```yaml
aggregate:
  mode: progressive  # Report results as they complete
  streaming: true

  # User sees:
  # [10s] Lint: ✅ 82.5
  # [30s] Complexity: ✅ 88.0
  # [60s] Security: ✅ 75.0
  # [90s] Tests: ✅ 89.0
  # [95s] Overall: ✅ 83.6
```

## Integration with GitHub

```yaml
# Automatic status check creation
github:
  status_check:
    context: "code-quality-gate"
    description: "Score: 83.6/100 ✅"
    target_url: "https://dashboard.company.com/quality/run-123"

  # PR comment with results
  comment:
    header: "## Code Quality Gate Results"
    body: |
      **Overall Score**: 83.6/100 ✅

      | Check | Score | Status |
      |-------|-------|--------|
      | Lint | 82.5 | ✅ |
      | Security | 75.0 | ⚠️ |
      | Tests | 89.0 | ✅ |
      | Complexity | 88.0 | ✅ |

      [View detailed report](dashboard_link)
```

## Related Examples

- **example-1-sequential.md**: Sequential execution pattern
- **example-3-conditional.md**: Conditional branching pattern
- **references/orchestration-patterns.md**: Advanced parallel patterns

---

**Key Takeaway**: Parallel cascades maximize throughput by running independent tasks concurrently with multi-agent coordination.
