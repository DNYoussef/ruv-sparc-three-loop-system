# Test 2: Automated Repair with Connascence-Aware Bundling

**Skill**: cicd-intelligent-recovery
**Phase**: Step 4 (Intelligent Fixes)
**Version**: 2.0.0

## Test Objective

Validate that the auto-repair system correctly:
1. Generates fix plans using program-of-thought structure
2. Identifies connascence coupling for bundled fixes
3. Executes fixes with appropriate strategy (isolated/bundled/architectural)
4. Validates fixes through dual validation (sandbox + theater)
5. Makes consensus-based approval decisions

## Prerequisites

```bash
# Create test directory
mkdir -p .claude/.artifacts/test-repair
cd .claude/.artifacts/test-repair

# Create mock root causes consensus
cat > root-causes-consensus.json << 'EOF'
{
  "roots": [
    {
      "failure": {
        "testName": "should validate password length",
        "file": "src/auth/validators.js",
        "line": 15,
        "errorMessage": "TypeError: Cannot read property 'length' of null"
      },
      "rootCause": "Missing null check before accessing password.length property",
      "cascadedFailures": [
        "should handle empty password",
        "should reject short passwords"
      ],
      "connascenceContext": {
        "name": ["src/auth/login.js", "src/auth/register.js"],
        "type": [],
        "algorithm": ["src/utils/validation.js"]
      },
      "fixStrategy": "bundled"
    },
    {
      "failure": {
        "testName": "should connect to database",
        "file": "src/database/pool.js",
        "line": 42,
        "errorMessage": "Error: Connection timeout after 5000ms"
      },
      "rootCause": "Fixed timeout value doesn't handle slow networks",
      "cascadedFailures": [],
      "connascenceContext": {
        "name": [],
        "type": [],
        "algorithm": []
      },
      "fixStrategy": "isolated"
    }
  ],
  "stats": {
    "totalFailures": 5,
    "rootFailures": 2,
    "cascadedFailures": 3
  }
}
EOF
```

## Test Execution

### Step 1: Generate Fix Plans

```bash
# Run auto-repair in plan-only mode
python3 ../../../resources/scripts/auto_repair.py .

# Check generated plans
ls -la fix-plan-*.json
```

**Expected Files**:
- `fix-plan-should validate password length.json`
- `fix-plan-should connect to database.json`

### Step 2: Validate Bundled Fix Plan

```bash
# Check bundled fix plan structure
cat "fix-plan-should validate password length.json" | jq '.'
```

**Expected Structure**:
```json
{
  "rootCause": "Missing null check before accessing password.length property",
  "fixStrategy": "bundled",
  "files": [
    {
      "path": "src/auth/validators.js",
      "reason": "primary failure location",
      "changes": "Fix root cause: Missing null check..."
    },
    {
      "path": "src/auth/login.js",
      "reason": "connascence of name (shared symbols)",
      "changes": "Update symbol references to match fix"
    },
    {
      "path": "src/auth/register.js",
      "reason": "connascence of name (shared symbols)",
      "changes": "Update symbol references to match fix"
    },
    {
      "path": "src/utils/validation.js",
      "reason": "connascence of algorithm (shared logic)",
      "changes": "Update algorithm implementation"
    }
  ],
  "minimalChanges": "Bundled fix across 4 files - atomic changes to preserve consistency",
  "predictedSideEffects": [
    "Auto-resolves 3 cascaded failures"
  ],
  "validationPlan": {
    "mustPass": [
      "should validate password length",
      "should handle empty password",
      "should reject short passwords"
    ]
  },
  "reasoning": [
    "Root cause: Missing null check before accessing password.length property",
    "Strategy: bundled (4 files)",
    "Connascence coupling: 3 files",
    "Expected cascade resolution: 3 tests"
  ]
}
```

**Validation**:
- ✅ 4 files identified (1 primary + 3 connascence)
- ✅ Strategy is "bundled" (not isolated)
- ✅ Cascade failures are tracked
- ✅ Validation plan includes all affected tests

### Step 3: Validate Isolated Fix Plan

```bash
# Check isolated fix plan
cat "fix-plan-should connect to database.json" | jq '.'
```

**Expected Structure**:
```json
{
  "rootCause": "Fixed timeout value doesn't handle slow networks",
  "fixStrategy": "isolated",
  "files": [
    {
      "path": "src/database/pool.js",
      "reason": "primary failure location",
      "changes": "Fix root cause: Fixed timeout value..."
    }
  ],
  "minimalChanges": "Single file fix - isolated change with no cascading updates",
  "predictedSideEffects": [],
  "validationPlan": {
    "mustPass": ["should connect to database"]
  },
  "reasoning": [
    "Root cause: Fixed timeout value doesn't handle slow networks",
    "Strategy: isolated (1 files)",
    "Connascence coupling: 0 files",
    "Expected cascade resolution: 0 tests"
  ]
}
```

**Validation**:
- ✅ Single file (isolated)
- ✅ Strategy is "isolated"
- ✅ No connascence coupling
- ✅ No cascade failures

### Step 4: Validate Fix Implementation

```bash
# Check fix implementations
cat "fix-impl-should validate password length.json" | jq '.filesChanged'
```

**Expected**:
```json
[
  "src/auth/validators.js",
  "src/auth/login.js",
  "src/auth/register.js",
  "src/utils/validation.js"
]
```

### Step 5: Validate Commit Messages

```bash
# Check commit message for bundled fix
cat "fix-impl-should validate password length.json" | jq -r '.commitMessage'
```

**Expected Pattern**:
```
fix: should validate password length

Root Cause: Missing null check before accessing password.length property

Fix Strategy: bundled
Files Changed: 4

Changes:
- src/auth/validators.js: Fix root cause: Missing null check...
- src/auth/login.js: Update symbol references to match fix
- src/auth/register.js: Update symbol references to match fix
- src/utils/validation.js: Update algorithm implementation

Resolves 3 cascaded failures

Connascence Context: 3 coupled files

Program-of-Thought: Plan → Execute → Validate → Approve
```

### Step 6: Validate Generated Patches

```bash
# Check patch files exist
ls -la fixes/

# Check bundled patch structure
cat "fixes/should validate password length.patch"
```

**Expected**:
```
# Git patch would be generated here
# Files: src/auth/validators.js, src/auth/login.js, src/auth/register.js, src/utils/validation.js
# Strategy: bundled
```

### Step 7: Validate Approval Summary

```bash
# Check auto-repair summary
cat auto-repair-summary.json | jq '.'
```

**Expected**:
```json
{
  "total": 2,
  "approved": 2,
  "rejected": 0,
  "fixes": [
    {
      "failureId": "should validate password length",
      "decision": "APPROVED",
      "strategy": "bundled",
      "filesChanged": 4
    },
    {
      "failureId": "should connect to database",
      "decision": "APPROVED",
      "strategy": "isolated",
      "filesChanged": 1
    }
  ]
}
```

**Validation**:
- ✅ Both fixes approved
- ✅ Correct strategies assigned
- ✅ File counts match plans

## Advanced Tests

### Test 8: High Coupling (Architectural Strategy)

```bash
# Add high-coupling root cause
cat >> root-causes-consensus.json << 'EOF'
{
  "roots": [
    {
      "failure": {
        "testName": "authentication flow",
        "file": "src/core/auth.js",
        "line": 100,
        "errorMessage": "Auth system failure"
      },
      "rootCause": "Tight coupling across authentication system",
      "cascadedFailures": ["test1", "test2", "test3", "test4", "test5"],
      "connascenceContext": {
        "name": ["file1.js", "file2.js", "file3.js"],
        "type": ["file4.js", "file5.js"],
        "algorithm": ["file6.js", "file7.js"]
      },
      "fixStrategy": "architectural"
    }
  ]
}
EOF

# Re-run auto-repair
python3 ../../../resources/scripts/auto_repair.py .

# Check architectural fix plan
cat "fix-plan-authentication flow.json" | jq '.fixStrategy'
```

**Expected**: `"architectural"`

### Test 9: Validation Failure Handling

```bash
# Mock validation failure
cat > "fix-validation-sandbox-authentication flow.json" << 'EOF'
{
  "fixApplied": true,
  "originalTestPassed": false,
  "allTestsResult": {
    "total": 100,
    "passed": 95,
    "failed": 5
  },
  "verdict": "FAIL",
  "reasoning": "Sandbox tests failed - fix doesn't resolve root cause"
}
EOF

# Check approval decision
cat "fix-approval-authentication flow.json" | jq '.decision'
```

**Expected**: `"REJECTED"`

## Validation Criteria

✅ **PASS** if:
- Fix plans are generated for all root causes
- Bundled strategy is correctly applied for connascence coupling
- Isolated strategy is correctly applied for single-file fixes
- Architectural strategy is triggered for high coupling (6+ files)
- Commit messages include reasoning and connascence context
- Approval decisions respect validation results
- Rejected fixes include actionable feedback

❌ **FAIL** if:
- Fix strategies don't match connascence coupling
- Plans don't identify all affected files
- Bundled fixes don't include all coupled files
- Validation failures aren't properly handled
- Approval logic is incorrect

## Cleanup

```bash
cd ../../..
rm -rf .claude/.artifacts/test-repair
```

## Performance Expectations

- Plan generation: < 1 second per root cause
- Implementation generation: < 2 seconds per fix
- Validation simulation: < 500ms per fix
- Total time for 2 fixes: < 10 seconds

## Notes

- This test uses mock data to isolate repair logic
- Real-world usage would integrate with actual code files and test execution
- Connascence analysis determines fix strategy automatically
- Program-of-thought structure ensures systematic fix generation
- Dual validation (sandbox + theater) happens in Step 5 & 6 of full pipeline

## Next Steps

After passing this test:
1. Run **test-3-root-cause-analysis.md** to validate cascade detection
2. Integrate with full pipeline via **recovery_pipeline.sh**
