# Feature Development Automation Scripts

This directory contains 4 production-ready automation scripts for the complete 12-stage feature development lifecycle.

---

## Scripts Overview

### 1. feature-workflow.js
**Complete 12-Stage Feature Development Automation**

Orchestrates the entire feature development lifecycle from research to deployment.

**Usage:**
```bash
node feature-workflow.js "feature description" [target-directory] [options]
```

**Options:**
- `--no-pr` - Skip automatic pull request creation
- `--deploy` - Deploy to production after validation

**Examples:**
```bash
# Standard feature development
node feature-workflow.js "User authentication with JWT" src/auth/

# Without PR creation
node feature-workflow.js "Payment integration" src/payments/ --no-pr

# With deployment
node feature-workflow.js "Dark mode toggle" src/ui/ --deploy
```

**Workflow Stages:**
1. Research best practices (Gemini Search)
2. Analyze existing codebase (Gemini MegaContext)
3. Initialize development swarm (6 agents)
4. Design architecture
5. Generate diagrams (Gemini Media)
6. Rapid prototyping (Codex Auto)
7. Theater detection
8. Comprehensive testing (auto-fix iterations)
9. Style polish
10. Security review
11. Documentation generation
12. Production readiness check

**Output Files:**
- `research.md` - Best practices research
- `codebase-analysis.md` - Existing code patterns
- `architecture-design.md` - System architecture
- `architecture-diagram.png` - Visual architecture
- `data-flow.png` - Data flow diagram
- `implementation/` - Source code
- `test-results.json` - Test metrics
- `style-report.json` - Code quality metrics
- `security-report.json` - Security scan
- `theater-report.json` - Placeholder detection
- `FEATURE-DOCUMENTATION.md` - Complete docs

---

### 2. quality-validator.js
**Quality Threshold Validation**

Validates feature implementation against configurable quality thresholds.

**Usage:**
```bash
node quality-validator.js <output-directory> [--strict]
```

**Validation Checks:**

**Standard Mode:**
- Test coverage ≥80%
- Code quality score ≥85/100
- Zero critical security issues
- ≤3 high security issues
- All tests passing

**Strict Mode:**
- Test coverage ≥90%
- Code quality score ≥90/100
- Zero critical security issues
- Zero high security issues
- All tests passing

**Examples:**
```bash
# Standard validation
node quality-validator.js output/

# Strict validation (for production)
node quality-validator.js output/ --strict
```

**Exit Codes:**
- `0` - All quality checks passed
- `1` - Critical failures (tests failing, critical security issues)
- `1` - Failures in strict mode

**Use Cases:**
- CI/CD pipeline quality gates
- Pre-deployment validation
- Code review automation
- Production readiness verification

---

### 3. stage-executor.js
**Individual Stage Execution**

Execute individual workflow stages independently for debugging or iterative development.

**Usage:**
```bash
node stage-executor.js <stage-name> <output-directory> [stage-args...]
```

**Available Stages:**
- `research` - Research best practices
- `analyze` - Analyze codebase patterns
- `swarm-init` - Initialize development swarm
- `architecture` - Design architecture
- `diagrams` - Generate visual diagrams
- `prototype` - Rapid prototype with Codex
- `theater-detect` - Detect placeholder code
- `testing` - Run comprehensive tests
- `style-polish` - Analyze and fix code quality
- `security` - Security vulnerability scan
- `documentation` - Generate feature docs
- `deploy-check` - Verify production readiness

**Examples:**
```bash
# Research best practices only
node stage-executor.js research output/ "authentication best practices"

# Run security scan on existing implementation
node stage-executor.js security output/ implementation/

# Generate documentation
node stage-executor.js documentation output/ "User Authentication Feature"

# Check production readiness
node stage-executor.js deploy-check output/
```

**Use Cases:**
- Debug specific workflow stages
- Re-run failed stages
- Incremental development
- Testing individual components

---

### 4. metrics-collector.js
**Metrics Aggregation and Reporting**

Aggregates quality metrics from all workflow stages and generates comprehensive reports.

**Usage:**
```bash
node metrics-collector.js <output-directory> [--format json|markdown|both]
```

**Metrics Collected:**
- **Testing**: Pass/fail status, coverage %, execution time
- **Code Quality**: Quality score, violations, complexity
- **Security**: Issues by severity level
- **Implementation**: Theater issues, placeholder count

**Scoring System:**
- Testing: 35% weight
- Code Quality: 30% weight
- Security: 25% weight
- Implementation: 10% weight

**Examples:**
```bash
# Generate JSON report
node metrics-collector.js output/ --format json

# Generate Markdown report
node metrics-collector.js output/ --format markdown

# Generate both formats
node metrics-collector.js output/ --format both
```

**Output Files:**
- `metrics-summary.json` - Structured metrics data
- `METRICS-REPORT.md` - Human-readable report

**Exit Codes:**
- `0` - Production ready (all checks passed)
- `1` - Not production ready

**Use Cases:**
- CI/CD reporting
- Quality dashboards
- Management reporting
- Trend analysis

---

## Common Workflows

### Complete Feature Development
```bash
# 1. Run full 12-stage workflow
node feature-workflow.js "User profile management" src/profiles/

# 2. Validate quality
node quality-validator.js feature-*/

# 3. Collect metrics
node metrics-collector.js feature-*/ --format both

# 4. Review artifacts
cat feature-*/FEATURE-DOCUMENTATION.md
cat feature-*/METRICS-REPORT.md
```

### Iterative Development
```bash
# 1. Research and design
node stage-executor.js research output/ "payment integration"
node stage-executor.js architecture output/ "Stripe payment integration"

# 2. Implement
node stage-executor.js prototype output/

# 3. Test and validate
node stage-executor.js testing output/
node stage-executor.js security output/

# 4. Polish and document
node stage-executor.js style-polish output/
node stage-executor.js documentation output/

# 5. Final check
node stage-executor.js deploy-check output/
```

### CI/CD Pipeline Integration
```yaml
# GitHub Actions example
- name: Feature Development
  run: |
    node resources/scripts/feature-workflow.js "${{ env.FEATURE_NAME }}" src/

- name: Quality Validation
  run: |
    node resources/scripts/quality-validator.js feature-*/ --strict

- name: Metrics Collection
  run: |
    node resources/scripts/metrics-collector.js feature-*/ --format both

- name: Upload Reports
  uses: actions/upload-artifact@v3
  with:
    name: feature-reports
    path: |
      feature-*/FEATURE-DOCUMENTATION.md
      feature-*/METRICS-REPORT.md
      feature-*/metrics-summary.json
```

---

## Environment Requirements

### Required Tools
- Node.js ≥16.x
- npm or yarn
- Git
- GitHub CLI (gh) for PR creation

### Optional Tools (for enhanced features)
- Gemini API access (for search and media generation)
- Codex access (for auto-implementation)
- Claude Flow (for swarm coordination)
- Docker (for sandboxed execution)

### Environment Variables
```bash
# Required for full functionality
export GEMINI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
export GITHUB_TOKEN="your-token"

# Optional
export CODEX_API_KEY="your-key"
```

---

## Troubleshooting

### Script Fails to Find Output Directory
**Problem:** Error: Output directory required
**Solution:** Always provide absolute or relative path to output directory
```bash
node feature-workflow.js "feature" src/  # ✅ Correct
node feature-workflow.js "feature"       # ❌ Missing directory
```

### Quality Validation Fails
**Problem:** Exit code 1 from quality-validator.js
**Solution:** Review validation report for specific failures
```bash
node quality-validator.js output/ 2>&1 | tee validation.log
# Review validation.log for specific threshold failures
```

### Stage Executor Unknown Stage
**Problem:** Unknown stage: xyz
**Solution:** Use one of the 12 valid stage names
```bash
node stage-executor.js --help  # List valid stages
```

### Metrics Collector Missing Files
**Problem:** Could not read test-results.json
**Solution:** Ensure workflow or individual stages completed successfully
```bash
# Re-run failing stage
node stage-executor.js testing output/

# Then collect metrics
node metrics-collector.js output/
```

### GitHub PR Creation Fails
**Problem:** gh pr create failed
**Solution:** Ensure GitHub CLI authenticated and repository configured
```bash
gh auth login
gh repo set-default
```

---

## Performance Optimization

### For Large Codebases (>5000 LOC)
```bash
# Use Gemini MegaContext for analysis
# Automatically triggered in feature-workflow.js
# Manual:
node stage-executor.js analyze output/ "large-codebase-directory/"
```

### For Fast Iteration
```bash
# Skip heavy stages during development
node stage-executor.js prototype output/
node stage-executor.js testing output/
node stage-executor.js deploy-check output/

# Run full workflow only before deployment
node feature-workflow.js "feature" src/
```

### For CI/CD Optimization
```bash
# Cache dependencies
npm ci --prefer-offline

# Run validation in parallel
node quality-validator.js output/ &
node metrics-collector.js output/ &
wait

# Fail fast on critical issues
node quality-validator.js output/ --strict || exit 1
```

---

## Script Internals

### feature-workflow.js Architecture
```
main()
  ├─ stage1_research()           # Gemini Search
  ├─ stage2_analyze()            # Gemini MegaContext
  ├─ stage3_swarmInit()          # Claude Flow
  ├─ stage4_architecture()       # Design
  ├─ stage5_diagrams()           # Gemini Media
  ├─ stage6_prototype()          # Codex Auto
  ├─ stage7_theaterDetection()   # Theater detect
  ├─ stage8_testing()            # Auto-fix testing
  ├─ stage9_stylePolish()        # Quality analysis
  ├─ stage10_security()          # Security scan
  ├─ stage11_documentation()     # Doc generation
  ├─ stage12_productionReadiness() # Final validation
  └─ createPullRequest()         # Optional PR
```

### quality-validator.js Validation Logic
```javascript
checks = {
  testsPassing: testResults.all_passed === true,
  testCoverage: coverage >= threshold,
  qualityScore: score >= threshold,
  securityCritical: critical === 0,
  securityHigh: high <= threshold
}

if (hasCriticalFailures) exit(1)
if (hasFailures && strictMode) exit(1)
exit(0)
```

### metrics-collector.js Scoring Algorithm
```javascript
overallScore =
  (testingScore * 0.35) +
  (qualityScore * 0.30) +
  (securityScore * 0.25) +
  (implementationScore * 0.10)

grade =
  overallScore >= 90 ? 'A' :
  overallScore >= 80 ? 'B' :
  overallScore >= 70 ? 'C' :
  overallScore >= 60 ? 'D' : 'F'
```

---

## Contributing

When modifying scripts, ensure:
1. Backward compatibility maintained
2. Exit codes preserved (0 = success, 1 = failure)
3. Error handling comprehensive
4. Documentation updated
5. Tests updated in `../tests/`

---

## Support

For issues or questions:
- Check `../GOLD-TIER-ENHANCEMENT.md` for detailed documentation
- Review test files in `../tests/` for usage examples
- Consult main skill file `../SKILL.md` for integration points

---

**Last Updated**: 2025-11-02
**Version**: 1.0.0 (Gold Tier)
**Maintainer**: Claude Code Feature Development Team
