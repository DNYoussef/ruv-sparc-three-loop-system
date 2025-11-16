# Audit Pipeline - Complete Guide

## Overview

The Audit Pipeline is a comprehensive 3-phase code quality orchestrator that systematically transforms prototype code into production-ready software through:

1. **Theater Detection** - Identifies and eliminates mock/fake/placeholder code
2. **Functionality Audit** - Validates code works through sandbox testing with Codex iteration
3. **Style & Quality Audit** - Polishes code to professional production standards

## Why This Pipeline Exists

Code goes through stages:
- **Prototype**: "Make it work" - lots of shortcuts, mocks, TODOs
- **Functional**: "Make it right" - real implementations, tests passing
- **Production**: "Make it excellent" - clean, documented, maintainable

The Audit Pipeline automates this transformation.

## The 3-Phase System

### Phase 1: Theater Detection Audit

**Purpose**: Find all code that *looks* like it works but is actually fake.

**What It Finds**:
- Hardcoded mock data (`return {"id": 123, "name": "Test User"}`)
- TODO/FIXME markers indicating incomplete work
- Stub functions that exist but don't do anything
- Commented-out production code
- Simplified error handling that always succeeds
- Test mode conditionals that bypass real logic

**Process**:
1. Pattern-based scanning for theater indicators
2. Contextual analysis to understand intended behavior
3. Dependency mapping between theater instances
4. Risk assessment (critical vs. minor)
5. Completion or documentation

**Output**: Theater audit report listing all instances with locations, severity, and completion status.

**Skill Used**: `theater-detection-audit`

---

### Phase 2: Functionality Audit with Codex Sandbox

**Purpose**: Prove code actually works through execution testing.

**What It Does**:
- Creates isolated sandbox environment
- Generates comprehensive test cases
- Executes code with realistic inputs
- Verifies outputs match expectations
- **Uses Codex to fix failures automatically**

**The Codex Integration Loop**:

```
For each failing test:
┌─────────────────────────────────────┐
│ 1. Capture test failure details    │
│    - Error message                  │
│    - File and line                  │
│    - Code context                   │
└──────────┬──────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│ 2. Spawn Codex in sandbox           │
│    codex --full-auto "Fix test..."  │
│    - Network disabled (secure)      │
│    - CWD only (isolated)            │
│    - Autonomous execution           │
└──────────┬──────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│ 3. Codex analyzes and fixes         │
│    - Reads code                     │
│    - Identifies root cause          │
│    - Implements fix                 │
│    - Runs tests                     │
└──────────┬──────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│ 4. Re-test in sandbox               │
│    - Run the specific test          │
│    - Check if passing               │
└──────────┬──────────────────────────┘
           │
      ┌────┴────┐
      │ Passing? │
      └────┬────┘
      YES  │  NO (iteration < 5)
           │     │
           │     └──► Repeat with more context
           │
           ▼
┌─────────────────────────────────────┐
│ 5. Validate no regressions          │
│    - Run full test suite            │
│    - Ensure other tests still pass  │
└──────────┬──────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│ 6. Apply fix to main codebase       │
│    - Copy changes from sandbox      │
│    - Document what was fixed        │
│    - Track in audit report          │
└─────────────────────────────────────┘
```

**Safety Features**:
- **Sandboxed**: Network disabled, can't access parent directories
- **Isolated**: Changes tested before applying to main code
- **Validated**: Regression tests ensure no breakage
- **Limited**: Max 5 iterations per test (escalates if stuck)
- **Documented**: Every fix and iteration logged

**Output**: Functionality report with test results, Codex iteration logs, and verified fixes.

**Skills Used**: `functionality-audit` + `codex-auto`

---

### Phase 3: Style & Quality Audit

**Purpose**: Transform working code into production-grade code.

**What It Improves**:
- Code formatting and consistency
- Naming conventions
- Documentation (docstrings, comments, type hints)
- Security (input validation, error handling)
- Performance (algorithmic efficiency)
- Maintainability (decomposition, organization)

**Process**:
1. Automated linting (pylint, eslint, etc.)
2. Manual style review
3. Security and performance analysis
4. Documentation review
5. Refactoring for clarity
6. Error handling improvements
7. Re-test to ensure refactoring didn't break anything

**Output**: Style audit report with metrics showing before/after quality improvements.

**Skill Used**: `style-audit`

---

## Complete Pipeline Workflow

### Initialization
```bash
$ /audit-pipeline

Audit Pipeline Initializing...

Configuration:
- Target: src/
- Phases: [1, 2, 3]
- Codex Mode: auto
- Strictness: normal

Estimated Duration: 20-25 minutes
```

### Phase 1 Execution
```
=== Phase 1: Theater Detection ===

Scanning for theater patterns...
✓ Scanned 47 files

Theater found:
- 8 hardcoded mock responses
- 12 TODO markers
- 3 stub functions
- 2 commented-out sections

Risk Assessment:
- Critical: 2 (auth bypass, payment mock)
- High: 5
- Medium: 11
- Low: 7

Completing critical theater...
[Details of completions]

Phase 1 Complete: 15 minutes
```

### Phase 2 Execution
```
=== Phase 2: Functionality Audit ===

Creating sandbox environment...
✓ Sandbox created

Generating test cases...
✓ 247 tests generated

Executing tests...
- Passed: 239
- Failed: 8

Codex Iteration Loop:
Test 1/8: test_user_authentication
  Iteration 1: Fixed token generation → PASSED ✓

Test 2/8: test_payment_validation
  Iteration 1: Added amount validation → FAILED
  Iteration 2: Fixed currency handling → PASSED ✓

Test 3/8: test_email_sending
  Iteration 1: Configured SMTP → FAILED
  Iteration 2: Added retry logic → FAILED
  Iteration 3: Fixed template path → PASSED ✓

[...]

Codex Statistics:
- Total iterations: 12
- Success rate: 87.5% (7/8 fixed)
- Average iterations: 1.5 per fix
- Manual intervention: 1 test

Phase 2 Complete: 18 minutes
```

### Phase 3 Execution
```
=== Phase 3: Style & Quality Audit ===

Running linters...
✓ pylint: 89 issues
✓ mypy: 23 type issues

Applying automated fixes...
✓ Black formatting
✓ Import organization
✓ Auto-fixable lints

Manual refactoring...
- Decomposed large functions: 5
- Improved naming: 34 variables
- Added docstrings: 41 functions
- Enhanced error handling: 18 functions

Security review...
✓ No vulnerabilities found

Performance analysis...
- Optimized queries: 3
- Removed unnecessary loops: 2

Final validation...
✓ All tests still passing

Phase 3 Complete: 12 minutes
```

### Final Report
```
=== Audit Pipeline Complete ===

Executive Summary:
✅ Theater: 23 instances found, 23 completed
✅ Functionality: 247 tests, 247 passing
   - Codex fixed 7 test failures automatically
   - 1 test required manual fix
✅ Style: 89 issues fixed

Quality Metrics:
- Code Quality: 45% → 94%
- Test Coverage: 68% → 91%
- Maintainability: C → A
- Technical Debt: 10 weeks → 4 days

Production Readiness: ✅ APPROVED

Total Duration: 45 minutes

Next Steps:
1. Review detailed reports
2. Run manual smoke tests
3. Commit changes
4. Deploy to staging
```

---

## Configuration Options

### Phase Selection

Run all phases (default):
```bash
/audit-pipeline
```

Skip phases already done:
```bash
/audit-pipeline --phases=2,3          # Skip theater detection
/audit-pipeline --skip-theater         # Same as above
/audit-pipeline --phases=1,2          # Skip style (do later)
```

### Codex Integration Level

**Full Auto** (default):
```bash
/audit-pipeline --codex=auto
```
Codex fixes failing tests autonomously in sandbox. Fastest execution.

**Assisted**:
```bash
/audit-pipeline --codex=assisted
```
Codex suggests fixes, you approve before applying. More control.

**Off**:
```bash
/audit-pipeline --codex=off
```
Manual fixes only. Maximum control, slower.

### Strictness Levels

**Normal** (default):
```bash
/audit-pipeline
```
Blocks on critical issues, warns on moderate issues.

**Strict**:
```bash
/audit-pipeline --strict
```
Blocks on any issues. Zero tolerance for production deployment.

**Lenient**:
```bash
/audit-pipeline --lenient
```
Warnings only, no blocks. For exploratory audits.

### Target Specification

Entire codebase:
```bash
/audit-pipeline
```

Specific directory:
```bash
/audit-pipeline "Audit src/api directory"
```

Specific files:
```bash
/audit-pipeline --files="src/auth.py,src/payment.py"
```

---

## Real-World Examples

### Example 1: Pre-Production Release

```bash
$ /audit-pipeline "Pre-production audit for v2.0 release"

[Pipeline runs all 3 phases]

Result:
- 45 theater instances completed
- 312 tests passing (Codex fixed 23)
- 156 style issues resolved
- Quality: 52% → 96%
- Production Ready: ✅ APPROVED
```

### Example 2: Legacy Code Modernization

```bash
$ /audit-pipeline "Modernize legacy auth module"

[Pipeline runs]

Result:
- Theater: Removed SQL injection vulnerability (hardcoded query)
- Functionality: Fixed 8 authentication edge cases
- Style: Added type hints, improved naming, documented API
- Quality: 38% → 89%
```

### Example 3: Post-Prototype Hardening

```bash
$ /audit-pipeline "Harden rapid prototype for production"

[Pipeline runs]

Result:
- Theater: 67 mocks replaced with real implementations
- Functionality: All 189 tests passing
- Style: Refactored for maintainability
- Ready for deployment
```

---

## Codex Sandbox Iteration Details

### What Happens in the Sandbox

1. **Environment Creation**:
   ```bash
   # Create isolated directory
   mkdir /tmp/audit-sandbox-12345
   cd /tmp/audit-sandbox-12345

   # Copy project files
   cp -r /original/project/* .

   # Install dependencies in isolation
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Test Execution**:
   ```bash
   # Run failing test
   pytest tests/test_auth.py::test_user_login -v

   # Capture failure details
   ERROR: AssertionError: Expected 200, got 401
   File: tests/test_auth.py:45
   ```

3. **Codex Invocation**:
   ```bash
   codex --full-auto "Fix test failure in test_user_login:

   Error: AssertionError: Expected 200, got 401
   File: tests/test_auth.py:45

   Context:
   The test expects successful login but receives unauthorized.

   Requirements:
   1. Analyze why login is failing
   2. Fix the authentication logic
   3. Ensure test passes
   4. Preserve all other functionality"
   ```

4. **Codex Execution**:
   - Codex reads test file and implementation
   - Identifies issue (e.g., wrong password hash comparison)
   - Edits source file to fix
   - Runs test to verify
   - Reports success

5. **Validation**:
   ```bash
   # Re-run the specific test
   pytest tests/test_auth.py::test_user_login -v
   # ✓ PASSED

   # Run full test suite (regression check)
   pytest
   # ✓ 247/247 passed
   ```

6. **Application**:
   ```bash
   # Copy fixed file from sandbox to main
   cp /tmp/audit-sandbox-12345/src/auth.py /original/project/src/auth.py

   # Document in report
   echo "Codex fixed test_user_login by correcting password hash comparison"
   ```

### Iteration Example

```
Test: test_payment_processing

Iteration 1:
  Issue: Payment always fails
  Codex fix: Added Stripe API initialization
  Result: Still failing

Iteration 2:
  Issue: API key not found
  Codex fix: Added environment variable loading
  Result: Still failing

Iteration 3:
  Issue: Amount validation incorrect
  Codex fix: Fixed amount conversion (dollars → cents)
  Result: PASSED ✓

Applied fix after 3 iterations
```

---

## Output Reports

### Theater Detection Report
```markdown
# Theater Detection Audit Report

## Summary
- Files scanned: 47
- Theater instances: 23
- Critical: 2
- High: 5
- Medium: 11
- Low: 5

## Detailed Findings

### Critical Theater
1. **Authentication Bypass** (src/auth.py:34)
   - Pattern: Hardcoded admin check
   - Code: `if user == "admin": return True`
   - Risk: Security vulnerability
   - Status: ✅ Completed
   - Fix: Implemented proper role-based auth

2. **Payment Mock** (src/payment.py:78)
   - Pattern: Always succeeds
   - Code: `return {"status": "success"}`
   - Risk: Financial loss
   - Status: ✅ Completed
   - Fix: Integrated Stripe API

[... more findings ...]
```

### Functionality Report
```markdown
# Functionality Audit Report

## Test Execution Summary
- Total tests: 247
- Initially passing: 239
- Initially failing: 8
- Final passing: 247
- Final failing: 0

## Codex Iteration Log

### Test: test_user_authentication
- **Initial Error**: Token generation returns None
- **Iterations**: 1
- **Fix Applied**: Added JWT token generation logic
- **Result**: ✅ PASSED

### Test: test_payment_validation
- **Initial Error**: Amount validation fails
- **Iterations**: 2
- **Fix Applied**: Corrected amount conversion and added currency validation
- **Result**: ✅ PASSED after 2 iterations

[... more tests ...]

## Codex Statistics
- Total iterations: 12
- Average per fix: 1.5
- Success rate: 87.5%
- Manual fixes needed: 1
```

### Style Audit Report
```markdown
# Style & Quality Audit Report

## Summary
- Linting issues: 89 → 0
- Type issues: 23 → 0
- Missing docs: 41 → 0
- Complexity issues: 7 → 1

## Improvements by Category

### Formatting (34 fixes)
- Applied Black formatting
- Organized imports
- Fixed line lengths

### Naming (34 improvements)
- Renamed cryptic variables
- Improved function names
- Consistent naming conventions

### Documentation (41 additions)
- Added module docstrings: 5
- Added function docstrings: 36
- Added type hints: 67 functions

### Security (3 fixes)
- Added input validation: 3 functions
- Improved error messages (no sensitive data)

### Performance (5 optimizations)
- Optimized database queries: 3
- Removed unnecessary loops: 2

## Metrics Improvement
| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Linting errors | 89 | 0 | -100% |
| Complexity | 28 avg | 12 avg | -57% |
| Test coverage | 68% | 91% | +34% |
| Maintainability | C | A | +2 grades |
```

---

## Best Practices

### Before Running Pipeline

1. **Commit Current State**
   ```bash
   git add .
   git commit -m "Pre-audit snapshot"
   ```
   Allows rollback if needed.

2. **Ensure Tests Exist**
   - Pipeline needs tests for Phase 2
   - Will create basic tests if none exist
   - Better results with existing comprehensive tests

3. **Define Quality Standards**
   - Configure linters if custom rules needed
   - Document team style preferences
   - Set acceptable quality thresholds

4. **Estimate Time**
   - Small projects: 5-10 min
   - Medium: 15-30 min
   - Large: 30-60 min
   - Plan accordingly

### During Pipeline

1. **Monitor Progress**
   - Each phase reports status
   - Watch for escalations
   - Respond to user prompts if needed

2. **Trust Codex for Routine Fixes**
   - Codex is excellent at fixing common test failures
   - Don't micromanage - let it iterate
   - Intervene only for complex issues

3. **Review Escalations Promptly**
   - If Codex can't fix after 5 iterations, it escalates
   - Provide additional context or manual fix
   - Pipeline waits for your input

### After Pipeline

1. **Review Comprehensive Report**
   - Understand what changed
   - Validate critical fixes
   - Check metrics improvement

2. **Run Manual Tests**
   - Smoke test critical paths
   - Verify UI/UX not broken
   - Test integrations

3. **Commit with Detailed Message**
   ```bash
   git add .
   git commit -m "Audit pipeline: theater eliminated, all tests passing, style improved

   - Completed 23 theater instances
   - Codex fixed 7 test failures
   - Resolved 89 style issues
   - Quality improved from 45% to 94%
   - Production ready"
   ```

4. **Update Documentation**
   - Document new implementations
   - Update API docs
   - Note architectural changes

---

## Integration with Development Workflow

### Pre-Commit Hook
```bash
# .git/hooks/pre-commit
if [[ $(git diff --cached --name-status | grep -E '\.py$' | wc -l) -gt 5 ]]; then
  echo "Many Python files changed, running audit pipeline..."
  /audit-pipeline --phases=3 --quick
fi
```

### CI/CD Pipeline
```yaml
# .github/workflows/quality.yml
name: Quality Audit
on: [pull_request]
jobs:
  audit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run Audit Pipeline
        run: /audit-pipeline --strict
      - name: Upload Report
        uses: actions/upload-artifact@v2
        with:
          name: audit-report
          path: audit-report.md
```

### Release Process
```bash
# Before every release
/audit-pipeline --strict "Pre-release quality audit for v$(cat VERSION)"

# Review report
# If approved → tag and deploy
# If issues → fix and re-run
```

---

## Troubleshooting

### Pipeline Takes Too Long
**Cause**: Large codebase
**Solution**:
- Run phases individually
- Target specific directories
- Use `--quick` mode for faster (less thorough) audit

### Codex Can't Fix a Test
**Cause**: Complex issue requiring human judgment
**Solution**:
- Codex escalates after 5 iterations
- Review error details
- Provide manual fix or additional context
- Pipeline continues with other tests

### Style Changes Break Tests
**Cause**: Refactoring introduced bug
**Solution**:
- Pipeline auto-detects (runs tests after refactoring)
- Automatically rolls back problematic change
- Logs issue for manual review
- Continues with other improvements

### Too Many Issues Found
**Cause**: Low initial code quality
**Solution**:
- Use `--lenient` for first pass
- Prioritize critical issues
- Run phases separately
- Create phased improvement plan

### Sandbox Creation Fails
**Cause**: Dependency issues, permissions
**Solution**:
- Falls back to local testing (without sandbox isolation)
- Check logs for specific error
- Ensure dependencies installable
- May need manual environment setup

---

## FAQ

**Q: Will the pipeline change my code?**
A: Yes, it completes theater, fixes bugs, and improves style. But only in safe, validated ways. All changes are tested.

**Q: What if I don't trust Codex fixes?**
A: Use `--codex=assisted` mode. Codex suggests, you approve before applying.

**Q: Can I run just one phase?**
A: Yes! Use `/theater-detection-audit`, `/functionality-audit`, or `/style-audit` directly.

**Q: How much does Codex cost?**
A: Uses your ChatGPT Plus subscription ($20/month). No additional cost.

**Q: Is it safe?**
A: Yes. Sandboxed execution, regression testing, and you approve final changes.

**Q: What if tests don't exist?**
A: Pipeline creates basic tests for Phase 2. Better results with existing comprehensive tests.

**Q: Can it break my code?**
A: Extremely unlikely. All changes tested before applying. Regression checks prevent breakage. You can rollback if needed.

**Q: How often should I run it?**
A: Before every production deployment. Can also run regularly (weekly) for continuous quality.

---

## Summary

The Audit Pipeline is your automated path from prototype to production:

1. **Theater Detection** → Finds all fake code
2. **Functionality** → Tests everything, Codex fixes failures
3. **Style** → Polishes to professional standards

**Result**: Code that's genuine, functional, and production-ready.

**Time**: 15-30 minutes for medium projects

**Command**: `/audit-pipeline`

**When**: Before every production deployment

Transform your code quality automatically! 🚀
