# Functionality-Audit Skill Test Suite

## Overview
This directory contains **meta-tests** for the `functionality-audit` skill itself. These tests validate that the audit skill correctly identifies bugs, validates working code, detects integration issues, and generates comprehensive edge case tests.

## Test Files

### 1. [`test-basic-validation.md`](./test-basic-validation.md)
**Purpose**: Validates the "happy path" - audit correctly reports on bug-free code

**What it tests**:
- Sandbox creation and execution
- Correct function validation
- Test execution (all tests pass)
- Edge case generation for correct code
- Report generation (PASSED status)

**Expected outcome**: ✅ Audit reports "PASSED" with no critical issues

**Key metric**: Establishes baseline - if this fails, audit infrastructure is broken

---

### 2. [`test-bug-detection.md`](./test-bug-detection.md)
**Purpose**: Validates bug detection capability - audit finds subtle off-by-one error

**What it tests**:
- Bug detection in logic (not just syntax)
- Line number precision (pinpoints exact bug location)
- Root cause analysis (explains WHY bug occurs)
- Fix suggestion generation
- Fix verification (re-runs tests after fix)

**Expected outcome**: ❌ Audit reports "FAILED" with off-by-one error identified at line 11

**Key metric**: Bug detection accuracy - must identify bug at exact line

---

### 3. [`test-integration-failure.md`](./test-integration-failure.md)
**Purpose**: Validates integration testing - audit finds bugs at component boundaries

**What it tests**:
- Component isolation testing (both components work individually)
- Integration execution (fails when combined)
- Data flow tracing (List from UserAuth → String expected in PermissionChecker)
- Type mismatch detection
- Multiple fix strategies (adapter pattern, type union, source fix)

**Expected outcome**: ⚠️ Audit reports "INTEGRATION FAILURE" with type mismatch at line 57

**Key metric**: Cross-component analysis - must trace bug across function boundaries

---

### 4. [`test-edge-case-coverage.md`](./test-edge-case-coverage.md)
**Purpose**: Validates edge case generation - audit creates comprehensive test suite

**What it tests**:
- Edge case identification (boundary, null, type, exception, combo, precision)
- Test generation (24+ tests across 7 categories)
- Coverage improvement (42.8% → 100%)
- Bug discovery via edge cases (finds 2 previously unknown bugs)
- Test quality (syntactically correct, executable, meaningful)

**Expected outcome**: ✅ Audit generates 24 tests, achieves 100% coverage, finds 2 bugs

**Key metric**: Test generation intelligence - must create diverse, meaningful tests

---

## Test Execution Workflow

### For Test Creators (Reviewers)
1. Read test file (e.g., `test-basic-validation.md`)
2. Understand the test subject code (buggy or correct)
3. Run the audit skill on the test subject
4. Verify audit output matches expected results
5. Check pass/fail criteria

### For Audit Skill Developers
1. Use these tests as acceptance criteria
2. Run all 4 tests after any audit skill changes
3. All tests must pass before merging changes
4. Update tests if audit behavior changes intentionally

---

## Success Criteria (All 4 Tests)

| Test                        | Key Validation                          | Must Pass? |
|-----------------------------|-----------------------------------------|------------|
| test-basic-validation       | Correct code → PASSED                   | ✅ YES     |
| test-bug-detection          | Off-by-one bug → FAILED at line 11      | ✅ YES     |
| test-integration-failure    | Type mismatch → INTEGRATION FAILURE     | ✅ YES     |
| test-edge-case-coverage     | Generates 24+ tests, 100% coverage      | ✅ YES     |

**Overall Pass**: All 4 tests must pass (audit correctly handles all scenarios)

---

## Quick Reference

### Test 1: Basic Validation
- **Input**: Correct `add_numbers(a, b)` function
- **Expected**: ✅ PASSED, no issues
- **Tests**: Audit infrastructure works

### Test 2: Bug Detection
- **Input**: `get_last_n_elements()` with off-by-one bug
- **Expected**: ❌ FAILED, bug at line 11
- **Tests**: Bug detection + fix generation

### Test 3: Integration Failure
- **Input**: `UserAuth` + `PermissionChecker` with type mismatch
- **Expected**: ⚠️ INTEGRATION FAILURE, List vs. String at line 57
- **Tests**: Cross-component analysis

### Test 4: Edge Case Coverage
- **Input**: `calculate_discount()` with minimal tests
- **Expected**: ✅ 24 tests generated, 100% coverage, 2 bugs found
- **Tests**: Test generation intelligence

---

## File Organization

```
tests/
├── README.md                        # This file
├── test-basic-validation.md         # Test 1: Happy path
├── test-bug-detection.md            # Test 2: Off-by-one bug
├── test-integration-failure.md      # Test 3: Component boundary
└── test-edge-case-coverage.md       # Test 4: Test generation
```

---

## Running Tests Manually

### Step 1: Create test subject file
```bash
# Example: test-bug-detection.md
cat > test_subject_bug.py << 'EOF'
def get_last_n_elements(arr: list, n: int) -> list:
    if n == 0:
        return []
    return arr[-(n-1):]  # BUG: Off-by-one error

if __name__ == "__main__":
    result = get_last_n_elements([1, 2, 3, 4, 5], 3)
    assert result == [3, 4, 5], f"Expected [3, 4, 5], got {result}"
EOF
```

### Step 2: Run functionality-audit skill
```bash
npx claude-code skill invoke functionality-audit \
  --file "test_subject_bug.py" \
  --mode "comprehensive"
```

### Step 3: Verify results
- Check if bug was detected
- Verify line number precision
- Validate fix suggestions
- Confirm report completeness

---

## Notes for Maintainers

### When to Update Tests
- Audit skill behavior changes intentionally
- New audit features added (e.g., new bug types)
- Coverage targets change (e.g., 90% → 100%)
- Test subjects need modernization (e.g., Python 3.11 features)

### Test Maintenance Checklist
- [ ] Test files are self-contained (include all context)
- [ ] Expected results match current audit version
- [ ] Pass/fail criteria are clear and objective
- [ ] Tests cover both positive and negative cases
- [ ] Tests are independent (can run in any order)

---

## Future Test Ideas

### Additional Test Coverage (Not Yet Implemented)
- **Test 5**: Security vulnerability detection (SQL injection, XSS)
- **Test 6**: Performance regression testing (audit detects slow code)
- **Test 7**: Concurrency bugs (race conditions, deadlocks)
- **Test 8**: Memory leaks (resource management issues)
- **Test 9**: Large codebase handling (multi-file audit)
- **Test 10**: False positive rate (audit doesn't over-report)

---

**Last Updated**: 2025-11-02
**Test Suite Version**: 1.0.0
**Compatible with**: functionality-audit skill v1.0.0+
