# Code Review Assistant - Multi-Agent Swarm Review System

Comprehensive automated code review using specialized multi-agent swarm architecture for pull requests. Provides security analysis, performance optimization, style consistency, test coverage validation, and documentation review with automated fix suggestions.

## Quick Start

```bash
# Review PR with all checks (security, performance, style, tests, docs)
code-review-assistant 123

# Review focusing on security only
code-review-assistant 123 security

# Review with auto-merge if passing
code-review-assistant 123 "security,tests" true --auto-merge true

# Review specific files
code-review-assistant --files "src/auth.js,src/api.js" --focus-areas security,performance
```

## Multi-Agent Architecture

The Code Review Assistant coordinates **5 specialized review agents** working in parallel:

### Review Agent Specialists

1. **Security Reviewer**
   - Vulnerability detection (SQL injection, XSS, CSRF)
   - Unsafe pattern identification
   - Secret/credential scanning
   - Dependency security analysis
   - OWASP Top 10 compliance

2. **Performance Analyst**
   - Bottleneck detection
   - Algorithm complexity analysis
   - Memory leak identification
   - Database query optimization
   - Bundle size analysis

3. **Style Reviewer**
   - Code style consistency
   - Best practices enforcement
   - Maintainability assessment
   - Design pattern validation
   - Clean code principles

4. **Test Specialist**
   - Test coverage analysis
   - Edge case identification
   - Test quality assessment
   - Mock/stub validation
   - Integration test gaps

5. **Documentation Reviewer**
   - API documentation completeness
   - Code comment quality
   - README updates needed
   - JSDoc/TypeDoc validation
   - Changelog verification

## Features

### Parallel Execution
All 5 review agents execute simultaneously using mesh topology for maximum speed:
- **2.8-4.4x faster** than sequential reviews
- Real-time coordination via Claude Flow hooks
- Shared memory for cross-agent insights

### Quality Audit Pipeline
Integrated complete audit pipeline with:
- **Functionality Audit**: Sandbox execution testing
- **Theater Detection**: Byzantine consensus verification
- **Production Readiness**: Deployment checklist validation

### Intelligent Fix Suggestions
Powered by Codex reasoning engine:
- Context-aware code fixes
- Multiple solution alternatives
- Risk assessment for each fix
- One-click application ready

### Merge Readiness Assessment
Automated GO/NO-GO decision based on:
- Zero critical security issues
- All tests passing
- Overall score ≥80/100
- Configurable quality gates

## Review Process

```
PR #123 Submission
       ↓
[1] PR Information Gathering
       ↓
[2] Initialize Review Swarm (mesh topology, 5 agents)
       ↓
[3] Parallel Specialized Reviews ←─┐
    ├─ Security Scan            │
    ├─ Performance Analysis     │ (Parallel)
    ├─ Style Audit              │
    ├─ Test Coverage            │
    └─ Documentation Check   ───┘
       ↓
[4] Complete Quality Audit
       ↓
[5] Aggregate Findings + Calculate Scores
       ↓
[6] Generate Fix Suggestions (Codex)
       ↓
[7] Assess Merge Readiness
       ↓
[8] Post Review Comment + PR Decision
```

## Input Contract

```yaml
input:
  pr_number: number (required) or
  changed_files: array[string] (file paths)
  focus_areas: array[enum] (default: all)
    - security
    - performance
    - style
    - tests
    - documentation
  suggest_fixes: boolean (default: true)
  auto_merge_if_passing: boolean (default: false)
```

## Output Contract

```yaml
output:
  review_summary:
    overall_score: number (0-100)
    merge_ready: boolean
    blocking_issues: number
    warnings: number
    suggestions: number
  detailed_reviews:
    security: { score: number, issues: array, summary: string }
    performance: { score: number, bottlenecks: array, summary: string }
    style: { score: number, violations: array, summary: string }
    tests: { coverage_percent: number, gaps: array, summary: string }
    documentation: { score: number, missing: array, summary: string }
  fix_suggestions: array[{
    issue: string,
    file: string,
    line: number,
    severity: enum[critical, high, medium, low],
    suggested_fix: string,
    reasoning: string
  }]
  merge_decision: enum[approve, request_changes, needs_work]
```

## Examples

See the `/examples` directory for detailed review scenarios:

1. **Security Review** - SQL injection detection and fix
2. **Performance Review** - O(n²) algorithm optimization
3. **Style Review** - Code consistency enforcement

## Integration

### GitHub Workflows
```yaml
# .github/workflows/code-review.yml
name: AI Code Review
on: [pull_request]
jobs:
  review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run Code Review Assistant
        run: npx code-review-assistant ${{ github.event.pull_request.number }}
```

### Git Hooks
```bash
# .git/hooks/pre-commit
#!/bin/bash
code-review-assistant --files "$(git diff --name-only HEAD)" --suggest-fixes true
```

### Claude Flow Cascades
```bash
# Part of /github-automation-workflow cascade
npx claude-flow cascade run github-automation-workflow
```

## Configuration

Create `.code-review-config.json`:

```json
{
  "quality_gates": {
    "min_overall_score": 80,
    "critical_security_issues": 0,
    "required_test_coverage": 80,
    "all_tests_passing": true
  },
  "focus_areas": ["security", "performance", "style", "tests", "documentation"],
  "suggest_fixes": true,
  "auto_merge_if_passing": false,
  "review_agents": {
    "security": { "enabled": true, "severity_threshold": "medium" },
    "performance": { "enabled": true, "max_complexity": 10 },
    "style": { "enabled": true, "enforce_strict": true },
    "tests": { "enabled": true, "min_coverage": 80 },
    "documentation": { "enabled": true, "require_jsdoc": true }
  }
}
```

## Quality Metrics

### Scoring System
Each review category scored 0-100:
- **100-90**: Excellent (✅)
- **89-80**: Good (✅)
- **79-70**: Acceptable (⚠️)
- **69-60**: Needs Improvement (⚠️)
- **<60**: Critical Issues (❌)

**Overall Score** = Average of all category scores

### Merge Decision Logic
```
IF critical_security_issues = 0
   AND all_tests_passing = true
   AND overall_score >= 80
THEN
   IF overall_score >= 90
      DECISION = "approve"
   ELSE
      DECISION = "approve_with_suggestions"
   END IF
ELSE
   DECISION = "request_changes"
END IF
```

## Failure Modes & Recovery

| Failure | Detection | Recovery |
|---------|-----------|----------|
| PR not found | GitHub API error | Verify PR number and repo access |
| Critical security issues | Security agent score <60 | Block merge, escalate to security team |
| Tests failing | Quality audit failure | Request changes, provide fix suggestions |
| GitHub CLI not authenticated | `gh` command error | Guide user to `gh auth login` |
| Agent spawn failure | Swarm init timeout | Retry with reduced agent count |
| Fix suggestion generation failure | Codex timeout | Post review without suggestions |

## Advanced Usage

### Custom Review Focus
```bash
# Security + Performance only
code-review-assistant 123 "security,performance"

# Tests + Documentation only
code-review-assistant 123 "tests,documentation"
```

### Bypass Auto-Merge
```bash
# Review but never auto-merge
code-review-assistant 123 --auto-merge false
```

### Batch Review Multiple PRs
```bash
# Review all open PRs
gh pr list --json number --jq '.[].number' | xargs -I {} code-review-assistant {}
```

## Performance Benchmarks

- **Sequential Review**: ~8-12 minutes for 500 LOC change
- **Parallel Review (mesh)**: ~2-3 minutes for 500 LOC change
- **Speedup**: 2.8-4.4x faster
- **Token Reduction**: 32.3% via shared context
- **Accuracy**: 84.8% issue detection rate

## Related Skills

- `quick-quality-check` - Fast parallel lint/security/tests
- `smart-bug-fix` - Intelligent debugging for identified issues
- `functionality-audit` - Sandbox execution testing
- `theater-detection-audit` - Byzantine consensus verification
- `github-project-management` - Issue tracking integration

## References

- `/references/review-categories.md` - Comprehensive review category guide
- `/references/best-practices.md` - Effective code review techniques
- `/graphviz/workflow.dot` - Visual workflow diagram

## Support

- GitHub: https://github.com/ruvnet/claude-flow/issues
- Documentation: https://claude-flow.ruv.io/skills/code-review-assistant
- Examples: See `/examples` directory
