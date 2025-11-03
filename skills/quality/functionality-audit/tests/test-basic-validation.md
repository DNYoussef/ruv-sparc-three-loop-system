# Test: Basic Validation - Correct Function

## Purpose
Validates that the functionality-audit skill correctly identifies and reports on a simple, correctly implemented function with no bugs or issues. This tests the "happy path" where code works as intended.

## Setup

### Test Subject: Simple Addition Function

**File**: `test_subject_basic.py`

```python
def add_numbers(a: int, b: int) -> int:
    """
    Add two numbers together.

    Args:
        a: First integer
        b: Second integer

    Returns:
        Sum of a and b
    """
    return a + b


# Self-contained test
if __name__ == "__main__":
    # Test cases
    assert add_numbers(2, 3) == 5, "Basic addition failed"
    assert add_numbers(-1, 1) == 0, "Negative number handling failed"
    assert add_numbers(0, 0) == 0, "Zero handling failed"
    assert add_numbers(100, 200) == 300, "Large number addition failed"
    print("✅ All tests passed!")
```

### Environment Requirements
- Python 3.8+
- E2B sandbox (via Claude Code or Flow-Nexus)
- No external dependencies

## Execution

### Step 1: Create Test Subject
```bash
# Create test file in sandbox
cat > test_subject_basic.py << 'EOF'
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

if __name__ == "__main__":
    assert add_numbers(2, 3) == 5
    assert add_numbers(-1, 1) == 0
    assert add_numbers(0, 0) == 0
    assert add_numbers(100, 200) == 300
    print("✅ All tests passed!")
EOF
```

### Step 2: Run Functionality Audit
```bash
# Invoke the functionality-audit skill
# Via Claude Code:
npx claude-code skill invoke functionality-audit \
  --file "test_subject_basic.py" \
  --mode "validation"

# Or via direct skill execution:
Skill("functionality-audit") with context:
  file_path: "test_subject_basic.py"
  validation_mode: "comprehensive"
```

### Step 3: Monitor Audit Process
```bash
# Expected audit phases:
# 1. Code Analysis (AST parsing, complexity metrics)
# 2. Sandbox Creation (E2B Python environment)
# 3. Test Execution (run built-in tests)
# 4. Edge Case Generation (boundary value tests)
# 5. Results Analysis (success/failure reporting)
```

## Expected Results

### Phase 1: Code Analysis
```json
{
  "file": "test_subject_basic.py",
  "functions_found": ["add_numbers"],
  "complexity": {
    "cyclomatic": 1,
    "lines_of_code": 3,
    "parameters": 2
  },
  "type_hints": "complete",
  "documentation": "present"
}
```

### Phase 2: Sandbox Execution
```
🔧 Creating E2B sandbox (Python 3.11)...
✅ Sandbox created: sandbox-abc123
📦 Uploading test_subject_basic.py...
✅ File uploaded successfully
```

### Phase 3: Test Execution
```bash
$ python test_subject_basic.py
✅ All tests passed!

Exit code: 0
```

### Phase 4: Edge Case Generation
```python
# Audit should auto-generate additional tests:
assert add_numbers(sys.maxsize, 0) == sys.maxsize  # Boundary value
assert add_numbers(-sys.maxsize, sys.maxsize) == 0  # Extreme negatives
# Type validation (if type hints enforced)
```

### Phase 5: Final Report
```markdown
# Functionality Audit Report: test_subject_basic.py

## Summary
✅ **PASSED** - All validations successful

## Execution Results
- Built-in tests: 4/4 passed (100%)
- Generated edge cases: 2/2 passed (100%)
- Total test execution time: 0.143s

## Code Quality Metrics
- Cyclomatic Complexity: 1 (Excellent)
- Type Safety: Full type hints present
- Documentation: Complete with docstring

## Issues Found
None

## Recommendations
- Consider adding explicit type validation for non-integer inputs
- Add performance benchmarks for large number handling

## Verdict
✅ Code is production-ready with no critical issues
```

## Validation Criteria

### Must-Have Validations
- [x] **Sandbox Creation**: E2B sandbox created without errors
- [x] **File Upload**: Test subject uploaded to sandbox successfully
- [x] **Test Execution**: All built-in tests pass (4/4)
- [x] **Edge Case Generation**: Audit generates at least 2 additional test cases
- [x] **Results Analysis**: Report correctly identifies 0 bugs/issues
- [x] **Report Generation**: Markdown report created with all sections
- [x] **Cleanup**: Sandbox terminated properly after execution

### Code Analysis Checks
- [x] AST parsed successfully
- [x] Function signature extracted: `add_numbers(a: int, b: int) -> int`
- [x] Docstring recognized and parsed
- [x] Type hints validated
- [x] Complexity metrics calculated (cyclomatic=1, LoC=3)

### Execution Verification
- [x] Exit code: 0 (success)
- [x] Stdout contains "✅ All tests passed!"
- [x] Stderr is empty (no warnings/errors)
- [x] No exceptions raised during execution

### Reporting Validation
- [x] Report status: "PASSED"
- [x] Test coverage: 100% (all tests passed)
- [x] Issues list: Empty or "None"
- [x] Recommendations: Present (even if no critical issues)
- [x] Execution time: Recorded and reasonable (<5s for simple test)

## Pass/Fail Criteria

### ✅ PASS Conditions (ALL must be true)
1. **Sandbox execution completes without errors** (exit code 0)
2. **All built-in tests pass** (4/4 assertions succeed)
3. **Report status is "PASSED"** with no critical issues flagged
4. **Edge cases are generated** (at least 2 additional test scenarios)
5. **Report structure is complete** (all required sections present)
6. **Audit completes within timeout** (<60 seconds total)
7. **No false positives** (audit does not report bugs that don't exist)

### ❌ FAIL Conditions (ANY triggers failure)
1. Sandbox fails to create or times out
2. File upload fails
3. Test execution throws unexpected exceptions
4. Audit reports false positives (bugs in correct code)
5. Report is malformed or missing critical sections
6. Edge case generation fails or is skipped
7. Audit times out (>60 seconds for basic validation)

## Success Metrics

### Quantitative
- **Test Pass Rate**: 100% (6/6 tests including generated edge cases)
- **Execution Time**: <5 seconds (sandbox creation + execution + analysis)
- **Report Completeness**: 100% (all 5 sections present)
- **False Positive Rate**: 0% (no incorrect bug reports)

### Qualitative
- Report is human-readable and actionable
- Recommendations are relevant (even if non-critical)
- Edge cases are meaningful and test boundary conditions
- Audit provides confidence in code correctness

## Notes
- This test establishes the baseline "happy path" for functionality-audit
- Successful completion proves the audit can validate correct code
- Failure here indicates fundamental issues with audit infrastructure (sandbox, execution, or reporting)
- This test should ALWAYS pass if the audit skill is working correctly
