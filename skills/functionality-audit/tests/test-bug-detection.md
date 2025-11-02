# Test: Bug Detection - Off-by-One Error

## Purpose
Validates that the functionality-audit skill can detect subtle bugs, specifically off-by-one errors, and accurately pinpoint the bug location with line number precision. This tests the audit's ability to identify logic errors that pass syntax checks but fail runtime tests.

## Setup

### Test Subject: Array Index Function with Off-by-One Bug

**File**: `test_subject_bug.py`

```python
def get_last_n_elements(arr: list, n: int) -> list:
    """
    Get the last n elements from an array.

    Args:
        arr: Input array
        n: Number of elements to retrieve from end

    Returns:
        Last n elements of the array

    Example:
        >>> get_last_n_elements([1, 2, 3, 4, 5], 3)
        [3, 4, 5]
    """
    # BUG: Off-by-one error - should be arr[-n:] not arr[-(n-1):]
    if n == 0:
        return []
    return arr[-(n-1):]  # ⚠️ INCORRECT: loses one element


# Self-contained test
if __name__ == "__main__":
    # Test case that will FAIL due to off-by-one error
    result = get_last_n_elements([1, 2, 3, 4, 5], 3)
    expected = [3, 4, 5]
    assert result == expected, f"Expected {expected}, got {result}"
    print("✅ All tests passed!")
```

### Environment Requirements
- Python 3.8+
- E2B sandbox
- No external dependencies

## Execution

### Step 1: Create Buggy Test Subject
```bash
# Create file with deliberate off-by-one bug
cat > test_subject_bug.py << 'EOF'
def get_last_n_elements(arr: list, n: int) -> list:
    """Get the last n elements from an array."""
    if n == 0:
        return []
    return arr[-(n-1):]  # BUG: Off-by-one error

if __name__ == "__main__":
    result = get_last_n_elements([1, 2, 3, 4, 5], 3)
    expected = [3, 4, 5]
    assert result == expected, f"Expected {expected}, got {result}"
    print("✅ All tests passed!")
EOF
```

### Step 2: Run Functionality Audit
```bash
# Invoke functionality-audit skill
npx claude-code skill invoke functionality-audit \
  --file "test_subject_bug.py" \
  --mode "comprehensive" \
  --enable-bug-detection true
```

### Step 3: Monitor Bug Detection Process
```bash
# Expected audit phases:
# 1. Code Analysis (identify get_last_n_elements function)
# 2. Sandbox Creation
# 3. Test Execution (assertion will FAIL)
# 4. Bug Localization (pinpoint line 11: arr[-(n-1):])
# 5. Root Cause Analysis (off-by-one in slice indexing)
# 6. Fix Suggestion (change to arr[-n:])
```

## Expected Results

### Phase 1: Code Analysis
```json
{
  "file": "test_subject_bug.py",
  "functions_found": ["get_last_n_elements"],
  "complexity": {
    "cyclomatic": 2,
    "lines_of_code": 7,
    "parameters": 2
  },
  "type_hints": "complete",
  "suspicious_patterns": [
    {
      "line": 11,
      "pattern": "arr[-(n-1):]",
      "warning": "Potential off-by-one in slice indexing"
    }
  ]
}
```

### Phase 2: Sandbox Execution (FAILURE)
```bash
$ python test_subject_bug.py
Traceback (most recent call last):
  File "test_subject_bug.py", line 16, in <module>
    assert result == expected, f"Expected {expected}, got {result}"
AssertionError: Expected [3, 4, 5], got [4, 5]

Exit code: 1
```

### Phase 3: Bug Localization
```
🔍 Bug Detection Analysis

Location: test_subject_bug.py:11
Function: get_last_n_elements
Line: return arr[-(n-1):]

❌ Assertion Failed:
  Expected: [3, 4, 5]
  Got: [4, 5]
  Difference: Missing element at index 0

Root Cause: Off-by-one error in slice indexing
  - Current: arr[-(n-1):] → retrieves (n-1) elements, not n
  - Should be: arr[-n:] → correctly retrieves last n elements
```

### Phase 4: Fix Suggestion
```python
# BEFORE (buggy):
def get_last_n_elements(arr: list, n: int) -> list:
    if n == 0:
        return []
    return arr[-(n-1):]  # ❌ Returns (n-1) elements

# AFTER (fixed):
def get_last_n_elements(arr: list, n: int) -> list:
    if n == 0:
        return []
    return arr[-n:]  # ✅ Returns n elements
```

### Phase 5: Validation of Fix
```bash
# Audit should auto-apply fix and re-test
$ python test_subject_bug_FIXED.py
✅ All tests passed!

Exit code: 0
```

### Phase 6: Final Report
```markdown
# Functionality Audit Report: test_subject_bug.py

## Summary
❌ **FAILED** - Critical bug detected and fixed

## Bug Analysis

### Bug #1: Off-by-One Error in Array Slicing
**Severity**: HIGH
**Location**: Line 11, `get_last_n_elements` function
**Type**: Logic Error (Off-by-One)

**Description**:
The function uses `arr[-(n-1):]` instead of `arr[-n:]`, causing it to return (n-1) elements instead of n elements.

**Evidence**:
- Input: `get_last_n_elements([1, 2, 3, 4, 5], 3)`
- Expected: `[3, 4, 5]` (3 elements)
- Actual: `[4, 5]` (2 elements)
- Missing: First element of expected range

**Root Cause**:
Incorrect slice indexing arithmetic. Python negative indices count from the end:
- `arr[-3:]` = last 3 elements ✅
- `arr[-2:]` = last 2 elements (what the bug does) ❌

**Fix Applied**:
```diff
- return arr[-(n-1):]
+ return arr[-n:]
```

**Verification**:
- Original test: FAILED (AssertionError)
- After fix: PASSED ✅
- Additional edge cases tested: 5/5 PASSED

## Execution Results
- Built-in tests (original): 0/1 passed (FAILED)
- Built-in tests (after fix): 1/1 passed ✅
- Generated edge cases: 5/5 passed ✅

## Code Quality Metrics
- Cyclomatic Complexity: 2 (Simple)
- Bug Density: 1 bug per 7 LoC (HIGH - needs attention)

## Recommendations
1. **Add boundary tests**: Test with n=0, n=1, n=len(arr), n>len(arr)
2. **Property-based testing**: Use hypothesis to test random array slicing
3. **Code review**: Review all slice operations for similar off-by-one errors

## Verdict
❌ Original code has critical bug (off-by-one error)
✅ Bug identified, fixed, and verified successfully
```

## Validation Criteria

### Must-Have Validations
- [x] **Bug Detection**: Audit identifies the off-by-one error
- [x] **Line Number Precision**: Bug location pinpointed to line 11
- [x] **Root Cause Analysis**: Report explains WHY the bug occurs
- [x] **Fix Generation**: Audit suggests correct fix (`arr[-n:]`)
- [x] **Fix Verification**: Audit validates fix resolves the issue
- [x] **Edge Case Testing**: Audit tests n=0, n=1, n=len(arr), n>len(arr)
- [x] **Report Completeness**: All bug analysis sections present

### Bug Detection Checks
- [x] Test execution fails with AssertionError
- [x] Error message captured: "Expected [3, 4, 5], got [4, 5]"
- [x] Exit code: 1 (failure)
- [x] Bug type classified as "Off-by-One Error"
- [x] Severity assigned: HIGH

### Fix Validation
- [x] Suggested fix is syntactically correct
- [x] Fixed code passes original test
- [x] Fixed code passes additional edge cases
- [x] No new bugs introduced by fix
- [x] Fix is minimal (1 line change)

### Reporting Validation
- [x] Report status: "FAILED" (for original code)
- [x] Bug section includes: severity, location, type, description
- [x] Evidence section shows expected vs. actual values
- [x] Root cause explanation is technically accurate
- [x] Fix is clearly presented (diff format)
- [x] Recommendations are actionable

## Pass/Fail Criteria

### ✅ PASS Conditions (ALL must be true)
1. **Bug is detected** - Audit identifies off-by-one error
2. **Location is precise** - Line 11 identified correctly
3. **Root cause is correct** - Explanation of `-(n-1)` vs `-n` is accurate
4. **Fix is valid** - Suggested change resolves the bug
5. **Fix is verified** - Re-running tests with fix shows success
6. **No false negatives** - Audit does not miss the bug
7. **Report is actionable** - Developer can understand and apply fix

### ❌ FAIL Conditions (ANY triggers failure)
1. Audit reports "PASSED" (false negative - misses the bug)
2. Bug location is incorrect or vague
3. Root cause analysis is missing or inaccurate
4. Suggested fix does not resolve the issue
5. Suggested fix introduces new bugs
6. Report lacks evidence (expected vs. actual values)
7. Audit times out without completing bug analysis

## Success Metrics

### Quantitative
- **Bug Detection Rate**: 100% (1/1 bug found)
- **Location Precision**: Exact line number (11) identified
- **Fix Accuracy**: 100% (fix resolves bug without side effects)
- **False Negative Rate**: 0% (bug not missed)
- **Analysis Time**: <30 seconds (detection + root cause + fix)

### Qualitative
- Root cause explanation demonstrates understanding of Python slice indexing
- Fix is minimal and idiomatic (Pythonic)
- Edge cases are comprehensive (boundary value testing)
- Report is clear enough for junior developers to understand

## Edge Cases to Test After Fix

### Auto-Generated Test Suite
```python
# Edge case 1: Empty array
assert get_last_n_elements([], 0) == []
assert get_last_n_elements([], 3) == []

# Edge case 2: Single element
assert get_last_n_elements([1], 1) == [1]
assert get_last_n_elements([1], 0) == []

# Edge case 3: n equals array length
assert get_last_n_elements([1, 2, 3], 3) == [1, 2, 3]

# Edge case 4: n exceeds array length
assert get_last_n_elements([1, 2], 5) == [1, 2]

# Edge case 5: Large array
assert len(get_last_n_elements(list(range(1000)), 100)) == 100
```

## Notes
- This test validates the core bug detection capability of functionality-audit
- Off-by-one errors are common and subtle - good test of detection sophistication
- Successful detection proves the audit can perform static analysis + dynamic testing
- The fix suggestion demonstrates the audit's ability to provide actionable remediation
- This test should ALWAYS result in bug detection and successful fix
