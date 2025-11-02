# Dogfooding System Safety Rules

**Version**: 1.0
**Created**: 2025-11-02
**Purpose**: Prevent breaking functionality when dogfooding improvements

---

## 🛡️ CRITICAL SAFETY RULES

### Rule 1: Sandbox Testing REQUIRED

**BEFORE applying ANY fix to production code:**

```bash
# Create isolated test environment
mkdir -p /tmp/dogfood-sandbox
cp -r <target-project> /tmp/dogfood-sandbox/

# Apply fix in sandbox
cd /tmp/dogfood-sandbox/<target-project>
<apply-fix-commands>

# Run full test suite
npm test  # or pytest, or appropriate test command
npm run build  # Verify build succeeds

# If all tests pass, apply to production
# If ANY test fails, reject fix and store failure in Memory-MCP
```

### Rule 2: Automated Rollback

**Every fix MUST be reversible:**

```bash
# Before applying fix
git stash push -u -m "Pre-dogfooding-fix backup"

# Apply fix
<fix-commands>

# Test
npm test

# If tests fail
git stash pop
echo "Fix rolled back - tests failed"
```

### Rule 3: Progressive Application

**Apply fixes ONE AT A TIME, not in batches:**

```
BAD:  Fix 10 violations simultaneously
GOOD: Fix 1 violation → Test → Commit → Fix next violation
```

### Rule 4: Test Coverage Requirement

**ONLY apply fixes to code with ≥70% test coverage:**

```bash
# Check coverage before applying fix
npm run test:coverage  # or pytest --cov

# If coverage < 70%, add tests FIRST
# Then apply fix
```

### Rule 5: Continuous Integration Gate

**ALL fixes must pass CI/CD before merge:**

```yaml
# .github/workflows/dogfooding-safety.yml
name: Dogfooding Safety Check

on: [push, pull_request]

jobs:
  safety-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Run Tests
        run: npm test

      - name: Check Coverage
        run: npm run test:coverage

      - name: Verify Build
        run: npm run build

      - name: Run Connascence Analysis
        run: |
          python -m mcp.cli analyze-workspace . --profile strict
          # Fail if NEW violations introduced
```

---

## 📋 Safety Workflow

### Phase 1: Pre-Fix Validation (2 minutes)

```bash
# 1. Check test coverage
npm run test:coverage

# 2. Backup current state
git stash push -u -m "dogfooding-backup-$(date +%s)"

# 3. Document baseline
npm test > /tmp/baseline-tests.txt
```

### Phase 2: Sandbox Application (5 minutes)

```bash
# 1. Create sandbox
mkdir -p /tmp/dogfood-sandbox
cp -r . /tmp/dogfood-sandbox/current-project

# 2. Apply fix in sandbox
cd /tmp/dogfood-sandbox/current-project
<apply-fix>

# 3. Run tests
npm test
npm run build

# 4. Compare results
diff /tmp/baseline-tests.txt <(npm test)
```

### Phase 3: Production Application (3 minutes)

```bash
# ONLY if Phase 2 passed all tests

# 1. Apply fix to production
cd <original-project>
<apply-fix>

# 2. Verify
npm test
npm run build

# 3. Commit with safety metadata
git add .
git commit -m "dogfooding: <fix-description>

Safety checks:
- Sandbox tests: PASSED
- Production tests: PASSED
- Coverage change: <before>% -> <after>%
- Connascence impact: <violations-before> -> <violations-after>

Applied via: dogfooding-continuous-improvement.bat
"
```

### Phase 4: Monitoring (24 hours)

```bash
# Track for regressions
# Monitor CI/CD pipeline
# Check production metrics
# Roll back if issues detected
```

---

## 🚨 Failure Handling

### If Sandbox Tests Fail:

```bash
# 1. DO NOT apply to production
# 2. Store failure in Memory-MCP
cat << EOF | python -c "
from src.indexing.vector_indexer import VectorIndexer
from src.indexing.embedding_pipeline import EmbeddingPipeline
vi = VectorIndexer()
ep = EmbeddingPipeline()
text = '''
Fix FAILED in sandbox testing
Fix: <description>
Test failures: <test-output>
Reason: <analysis>
DO NOT APPLY THIS FIX
'''
emb = ep.encode_single(text)
vi.collection.add(
    ids=['fix-failure-$(date +%s)'],
    embeddings=[emb.tolist()],
    documents=[text],
    metadatas=[{
        'agent': 'dogfooding-system',
        'project': '<project-name>',
        'intent': 'fix-failure-warning',
        'severity': 'critical'
    }]
)
print('Failure stored in Memory-MCP')
"
EOF

# 3. Analyze why fix failed
# 4. Refine fix approach
# 5. Re-test in sandbox
```

### If Production Tests Fail:

```bash
# 1. IMMEDIATE ROLLBACK
git reset --hard HEAD~1

# 2. Verify rollback worked
npm test

# 3. Alert team
echo "PRODUCTION FIX FAILED - ROLLED BACK" | mail -s "Dogfooding Alert" team@example.com

# 4. Post-mortem analysis
# Why did sandbox pass but production fail?
# Update sandbox testing to catch this case
```

---

## 📊 Safety Metrics

Track these metrics to ensure safety:

1. **Fix Success Rate**: `(successful_fixes / total_attempted) * 100`
   - Target: ≥95%

2. **Rollback Rate**: `(rollbacks / total_fixes) * 100`
   - Target: ≤5%

3. **Test Coverage Change**: `coverage_after - coverage_before`
   - Target: ≥0% (never decrease)

4. **Build Success Rate**: `(successful_builds / total_builds) * 100`
   - Target: 100%

5. **Time to Detection** (for failures): `time_failure_detected - time_fix_applied`
   - Target: ≤10 minutes

---

## 🔒 Security Rules

### Rule S1: Never Auto-Apply to Security-Critical Code

**Manual review REQUIRED for:**
- Authentication/authorization code
- Cryptography functions
- Input validation/sanitization
- Database access layers
- API endpoint handlers

### Rule S2: Secret Scanning

```bash
# Before committing any fix
git diff | grep -E '(password|secret|key|token|api_key)' && echo "POTENTIAL SECRET DETECTED"
```

### Rule S3: Dependency Validation

```bash
# If fix modifies dependencies
npm audit  # or pip check
# Fail if HIGH or CRITICAL vulnerabilities
```

---

## 🎯 Integration with Scripts

### Update `dogfood-continuous-improvement.bat`:

```batch
@echo off
echo [SAFETY] Creating sandbox environment...
mkdir C:\Users\17175\tmp\dogfood-sandbox
xcopy /E /I /Q . C:\Users\17175\tmp\dogfood-sandbox\

echo [SAFETY] Applying fix in sandbox...
cd C:\Users\17175\tmp\dogfood-sandbox
<apply-fix-commands>

echo [SAFETY] Running tests in sandbox...
npm test
IF %ERRORLEVEL% NEQ 0 (
    echo [FAIL] Sandbox tests failed - FIX REJECTED
    cd ..
    rmdir /S /Q C:\Users\17175\tmp\dogfood-sandbox
    exit /b 1
)

echo [SAFETY] Sandbox tests passed - applying to production...
cd <original-dir>
<apply-fix-commands>

echo [SAFETY] Running production tests...
npm test
IF %ERRORLEVEL% NEQ 0 (
    echo [FAIL] Production tests failed - ROLLING BACK
    git reset --hard HEAD~1
    exit /b 1
)

echo [SUCCESS] Fix applied safely
```

---

## 📖 Best Practices

1. **Always test in sandbox first** - no exceptions
2. **One fix at a time** - easier to debug failures
3. **Monitor for 24h** - catch delayed regressions
4. **Document all failures** - learn from mistakes
5. **Update safety rules** - evolve based on failures
6. **Automate safety checks** - reduce human error
7. **Maintain rollback capability** - always reversible

---

## 🚀 Quick Reference

```bash
# Safe fix workflow (copy-paste)
git stash push -u -m "backup-$(date +%s)"
mkdir -p /tmp/sandbox && cp -r . /tmp/sandbox/
cd /tmp/sandbox && <apply-fix> && npm test
if [ $? -eq 0 ]; then
    cd - && <apply-fix> && npm test && git commit -m "dogfooding: fix"
else
    echo "FAILED - fix rejected"
fi
```

**Remember**: Better to reject a safe fix than apply an unsafe one!

---

**Status**: ✅ MANDATORY FOR ALL DOGFOODING OPERATIONS
**Enforcement**: Automated via CI/CD + manual review for security-critical code
