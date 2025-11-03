# Test 2: Skill Validation

## Test Objective
Verify that skill-validator.sh correctly identifies structural issues, missing components, and quality problems in micro-skills.

## Test Scenarios

### Scenario 2.1: Perfect Skill (Control Test)

**Setup**:
Create a complete, valid micro-skill with all required components.

**Expected Result**:
```
✓ Perfect! No issues found.
Exit code: 0
```

**Validation**:
- [ ] 0 errors
- [ ] 0 warnings
- [ ] All checks pass
- [ ] Exit code 0

### Scenario 2.2: Missing Frontmatter

**Setup**:
Create SKILL.md without YAML frontmatter delimiters.

**Expected Result**:
```
✗ Missing YAML frontmatter delimiters
Exit code: 1
```

**Validation**:
- [ ] Error detected
- [ ] Appropriate error message
- [ ] Exit code 1 (failure)

### Scenario 2.3: Incomplete Frontmatter

**Setup**:
Create SKILL.md with frontmatter missing required fields (name, description, version).

**Expected Result**:
```
✗ Missing required field: name
✗ Missing required field: description
✗ Missing required field: version
Exit code: 1
```

**Validation**:
- [ ] All missing fields detected
- [ ] Specific field names identified
- [ ] Exit code 1

### Scenario 2.4: Missing Agent Section

**Setup**:
Create SKILL.md without specialist agent section.

**Expected Result**:
```
✗ Missing specialist agent section
Exit code: 1
```

**Validation**:
- [ ] Missing agent section detected
- [ ] Clear error message
- [ ] Exit code 1

### Scenario 2.5: No Evidence-Based Pattern

**Setup**:
Create SKILL.md without any evidence-based prompting pattern mention.

**Expected Result**:
```
⚠ No evidence-based pattern detected
Exit code: 0 (warning only)
```

**Validation**:
- [ ] Warning issued (not error)
- [ ] Exit code 0 (allows packaging with warning)

### Scenario 2.6: Missing Contracts

**Setup**:
Create SKILL.md without input and output contract sections.

**Expected Result**:
```
✗ Missing input contract
✗ Missing output contract
Exit code: 1
```

**Validation**:
- [ ] Both contracts flagged
- [ ] Exit code 1
- [ ] Clear error messages

### Scenario 2.7: Skill Too Complex

**Setup**:
Create SKILL.md with >2000 words and >15 sections.

**Expected Result**:
```
⚠ Skill may be too complex (>2000 words) - consider splitting
⚠ High section count (16) - may indicate multiple responsibilities
Exit code: 0
```

**Validation**:
- [ ] Complexity warnings issued
- [ ] Word count reported
- [ ] Section count reported
- [ ] Exit code 0 (warnings only)

### Scenario 2.8: Missing Integration Documentation

**Setup**:
Create SKILL.md without integration/cascade/composition sections.

**Expected Result**:
```
⚠ No integration/composition documentation
⚠ Neural training integration not mentioned
Exit code: 0
```

**Validation**:
- [ ] Integration warnings issued
- [ ] Neural training warning issued
- [ ] Exit code 0

## Test Execution

### Manual Test Execution

```bash
# Create test skills directory
mkdir -p ./test-validation

# Scenario 2.1: Perfect skill (use generated skill from Test 1)
cp -r ./test-output/invoice-data-extractor ./test-validation/perfect-skill
bash resources/scripts/skill-validator.sh ./test-validation/perfect-skill

# Scenario 2.2: Missing frontmatter
mkdir ./test-validation/no-frontmatter
echo "# My Skill" > ./test-validation/no-frontmatter/SKILL.md
bash resources/scripts/skill-validator.sh ./test-validation/no-frontmatter

# Scenario 2.3: Incomplete frontmatter
mkdir ./test-validation/incomplete-frontmatter
cat > ./test-validation/incomplete-frontmatter/SKILL.md << 'EOF'
---
name: test-skill
---

# Test Skill
EOF
bash resources/scripts/skill-validator.sh ./test-validation/incomplete-frontmatter

# Scenario 2.4: Missing agent section
mkdir ./test-validation/no-agent
cat > ./test-validation/no-agent/SKILL.md << 'EOF'
---
name: test-skill
description: Test skill
version: 1.0.0
---

# Test Skill

## Purpose
This is a test skill without an agent section.
EOF
bash resources/scripts/skill-validator.sh ./test-validation/no-agent

# Continue for other scenarios...
```

### Automated Test Script

Create `run-validation-tests.sh`:

```bash
#!/bin/bash

echo "Running Micro-Skill Validator Tests"
echo "===================================="

PASSED=0
FAILED=0

# Test 2.1: Perfect skill
echo "Test 2.1: Perfect skill"
if bash resources/scripts/skill-validator.sh ./test-validation/perfect-skill; then
  echo "✓ PASS"
  ((PASSED++))
else
  echo "✗ FAIL"
  ((FAILED++))
fi

# Test 2.2: Missing frontmatter
echo "Test 2.2: Missing frontmatter"
if ! bash resources/scripts/skill-validator.sh ./test-validation/no-frontmatter 2>/dev/null; then
  echo "✓ PASS (correctly failed)"
  ((PASSED++))
else
  echo "✗ FAIL (should have failed)"
  ((FAILED++))
fi

# ... Add all other tests

echo ""
echo "===================================="
echo "Results: $PASSED passed, $FAILED failed"
if [[ $FAILED -eq 0 ]]; then
  echo "✓ All validation tests passed!"
  exit 0
else
  echo "✗ Some validation tests failed"
  exit 1
fi
```

## Validation Checks

### Required Validations

1. **Structure Validation**
   - [ ] SKILL.md existence
   - [ ] Directory structure
   - [ ] Recommended directories (tests/, examples/)

2. **Frontmatter Validation**
   - [ ] YAML delimiters (---)
   - [ ] Required fields (name, description, version)
   - [ ] Optional fields (tags)
   - [ ] YAML syntax correctness

3. **Agent Design Validation**
   - [ ] Specialist agent section exists
   - [ ] Evidence-based pattern detected
   - [ ] Methodology documented

4. **Contract Validation**
   - [ ] Input contract section
   - [ ] Output contract section
   - [ ] Structured format (YAML/JSON)

5. **Failure Mode Validation**
   - [ ] Failure modes documented
   - [ ] Error handling mentioned
   - [ ] Edge cases covered

6. **Atomicity Validation**
   - [ ] Word count reasonable (200-2000)
   - [ ] Section count appropriate (<15)
   - [ ] Single responsibility indicators

7. **Integration Validation**
   - [ ] Integration points documented
   - [ ] Cascade compatibility mentioned
   - [ ] Neural training integration

## Success Criteria

**Test passes if**:
1. Perfect skill (2.1) passes validation with 0 errors/warnings
2. All error scenarios (2.2-2.4, 2.6) correctly fail with exit code 1
3. All warning scenarios (2.5, 2.7, 2.8) pass with exit code 0 but issue warnings
4. Error messages are clear and actionable
5. All validation checks execute correctly

**Test fails if**:
1. False positives (errors on valid content)
2. False negatives (missing actual errors)
3. Incorrect exit codes
4. Unclear or missing error messages
5. Script crashes or hangs

## Test Results Template

```
Test Date: _________________
Tester: ___________________

Scenario 2.1 (Perfect): [ PASS / FAIL ]
  Exit code: ___
  Errors: ___
  Warnings: ___

Scenario 2.2 (No frontmatter): [ PASS / FAIL ]
  Correctly failed: [ YES / NO ]
  Error message clear: [ YES / NO ]

Scenario 2.3 (Incomplete frontmatter): [ PASS / FAIL ]
  All missing fields detected: [ YES / NO ]

Scenario 2.4 (No agent): [ PASS / FAIL ]
  Error detected: [ YES / NO ]

Scenario 2.5 (No pattern): [ PASS / FAIL ]
  Warning issued: [ YES / NO ]
  Exit code 0: [ YES / NO ]

Scenario 2.6 (No contracts): [ PASS / FAIL ]
  Both contracts flagged: [ YES / NO ]

Scenario 2.7 (Too complex): [ PASS / FAIL ]
  Warnings issued: [ YES / NO ]

Scenario 2.8 (No integration): [ PASS / FAIL ]
  Warnings issued: [ YES / NO ]

Overall: [ PASS / FAIL ]
```

## Edge Cases

- Skills with unusual formatting
- Very long skills (>3000 words)
- Skills with nested YAML in code blocks
- Skills with special characters in frontmatter
- Skills with multiple agent sections
- Skills mixing multiple evidence patterns

## Performance Testing

- Validate 100 skills in batch
- Measure validation time per skill (should be <1 second)
- Check memory usage
- Verify no resource leaks

## Related Tests

- Test 1: Generation accuracy (generates valid skills)
- Test 3: Optimization effectiveness
- Example 2: Real-world validation scenarios
