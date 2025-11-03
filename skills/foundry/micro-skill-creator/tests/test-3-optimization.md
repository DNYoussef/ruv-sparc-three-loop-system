# Test 3: Skill Optimization

## Test Objective
Verify that skill-optimizer.js correctly analyzes and identifies optimization opportunities in micro-skills for clarity, performance, and quality.

## Test Scenarios

### Scenario 3.1: Optimal Skill (Baseline)

**Setup**:
Use a well-crafted skill from Test 1 as baseline.

**Expected Result**:
```
Quality Score: 95-100/100
✅ Excellent! No optimization needed.
```

**Validation**:
- [ ] Quality score ≥ 95
- [ ] 0 issues
- [ ] 0 suggestions
- [ ] All 5 analysis phases pass

### Scenario 3.2: High Token Count

**Setup**:
Create SKILL.md with >2000 estimated tokens (>1540 words).

**Expected Result**:
```
⚠ High token count (2100 > 2000) - consider moving content to resources/
💡 Suggestions:
  - Move detailed content to resources/
Quality Score: 75-85/100
```

**Validation**:
- [ ] Token count warning issued
- [ ] Suggestion to use resources/ directory
- [ ] Quality score reduced appropriately
- [ ] Word and token counts displayed

### Scenario 3.3: Vague Language

**Setup**:
Create SKILL.md with vague terms (thing, stuff, various, etc.).

**Expected Result**:
```
⚠ Vague language detected: thing, stuff, various
💡 Suggestions:
  - Replace vague terms with specific language
Quality Score: 80-90/100
```

**Validation**:
- [ ] Vague terms detected and listed
- [ ] Suggestion to improve specificity
- [ ] Quality score reduced

### Scenario 3.4: Excessive Passive Voice

**Setup**:
Create SKILL.md with >5 passive voice constructions.

**Expected Result**:
```
💡 Consider reducing passive voice (8 instances)
Quality Score: 85-92/100
```

**Validation**:
- [ ] Passive voice instances counted
- [ ] Suggestion to use active voice
- [ ] Count displayed

### Scenario 3.5: Long Sentences

**Setup**:
Create SKILL.md with average sentence length >25 words.

**Expected Result**:
```
💡 Average sentence length high (27.3 words) - aim for 15-20
Quality Score: 85-92/100
```

**Validation**:
- [ ] Average sentence length calculated
- [ ] Warning for readability
- [ ] Target range specified (15-20)

### Scenario 3.6: Missing Evidence Pattern

**Setup**:
Create SKILL.md without evidence-based pattern keywords.

**Expected Result**:
```
⚠ No evidence-based prompting pattern detected
⚠ Missing Methodology section
Quality Score: 70-80/100
```

**Validation**:
- [ ] Pattern absence detected
- [ ] Methodology section check
- [ ] Quality score significantly reduced

### Scenario 3.7: Redundant Content

**Setup**:
Create SKILL.md with duplicate lines or repeated content.

**Expected Result**:
```
💡 3 potentially duplicate lines detected
Quality Score: 85-90/100
```

**Validation**:
- [ ] Duplicate lines identified
- [ ] Count displayed
- [ ] Suggestion to remove redundancy

### Scenario 3.8: Large Code Blocks

**Setup**:
Create SKILL.md with code blocks >500 characters.

**Expected Result**:
```
💡 2 long code block(s) - consider moving to resources/scripts/
Quality Score: 85-92/100
```

**Validation**:
- [ ] Long code blocks detected
- [ ] Suggestion to externalize
- [ ] Count displayed

### Scenario 3.9: Missing Contracts

**Setup**:
Create SKILL.md without YAML/JSON contract formats.

**Expected Result**:
```
💡 Consider using YAML/JSON for contract clarity
Quality Score: 80-88/100
```

**Validation**:
- [ ] Absence of structured formats detected
- [ ] Suggestion for YAML/JSON

### Scenario 3.10: No Integration Documentation

**Setup**:
Create SKILL.md without cascade/integration/command mentions.

**Expected Result**:
```
💡 Consider documenting integration points
Quality Score: 82-90/100
```

**Validation**:
- [ ] Missing integration docs detected
- [ ] Suggestion to add integration info

## Test Execution

### Individual Tests

```bash
# Test 3.1: Optimal skill
node resources/scripts/skill-optimizer.js ./test-validation/perfect-skill

# Test 3.2: High token count
node resources/scripts/skill-optimizer.js ./test-validation/high-tokens

# Test 3.3: Vague language
node resources/scripts/skill-optimizer.js ./test-validation/vague-language

# Test 3.4: Passive voice
node resources/scripts/skill-optimizer.js ./test-validation/passive-voice

# Test 3.5: Long sentences
node resources/scripts/skill-optimizer.js ./test-validation/long-sentences

# Test 3.6: No pattern
node resources/scripts/skill-optimizer.js ./test-validation/no-pattern

# Test 3.7: Redundant
node resources/scripts/skill-optimizer.js ./test-validation/redundant

# Test 3.8: Large code blocks
node resources/scripts/skill-optimizer.js ./test-validation/large-code

# Test 3.9: Missing contracts
node resources/scripts/skill-optimizer.js ./test-validation/no-contracts

# Test 3.10: No integration
node resources/scripts/skill-optimizer.js ./test-validation/no-integration
```

### Automated Test Suite

Create `run-optimization-tests.sh`:

```bash
#!/bin/bash

echo "Running Micro-Skill Optimizer Tests"
echo "====================================="

PASSED=0
FAILED=0

test_optimization() {
  local scenario=$1
  local skill_dir=$2
  local expected_min_score=$3
  local expected_max_score=$4

  echo "Testing: $scenario"

  result=$(node resources/scripts/skill-optimizer.js "$skill_dir" 2>&1)
  score=$(echo "$result" | grep "Quality Score:" | sed 's/.*: \([0-9]*\).*/\1/')

  if [[ $score -ge $expected_min_score && $score -le $expected_max_score ]]; then
    echo "✓ PASS (Score: $score, Expected: $expected_min_score-$expected_max_score)"
    ((PASSED++))
  else
    echo "✗ FAIL (Score: $score, Expected: $expected_min_score-$expected_max_score)"
    ((FAILED++))
  fi
  echo ""
}

# Run all tests
test_optimization "3.1: Optimal skill" "./test-validation/perfect-skill" 95 100
test_optimization "3.2: High tokens" "./test-validation/high-tokens" 75 85
test_optimization "3.3: Vague language" "./test-validation/vague-language" 80 90
test_optimization "3.4: Passive voice" "./test-validation/passive-voice" 85 92
test_optimization "3.5: Long sentences" "./test-validation/long-sentences" 85 92
test_optimization "3.6: No pattern" "./test-validation/no-pattern" 70 80
test_optimization "3.7: Redundant" "./test-validation/redundant" 85 90
test_optimization "3.8: Large code" "./test-validation/large-code" 85 92
test_optimization "3.9: No contracts" "./test-validation/no-contracts" 80 88
test_optimization "3.10: No integration" "./test-validation/no-integration" 82 90

echo "====================================="
echo "Results: $PASSED passed, $FAILED failed"
if [[ $FAILED -eq 0 ]]; then
  echo "✓ All optimization tests passed!"
  exit 0
else
  echo "✗ Some optimization tests failed"
  exit 1
fi
```

## Analysis Phases

### Phase 1: Structure Analysis
- [ ] Frontmatter presence
- [ ] Section count (min 5, max 15)
- [ ] Required sections present

### Phase 2: Clarity Analysis
- [ ] Vague language detection
- [ ] Passive voice analysis
- [ ] Sentence length calculation

### Phase 3: Performance Analysis
- [ ] Token estimation
- [ ] Redundancy detection
- [ ] Code block size analysis

### Phase 4: Evidence Pattern Analysis
- [ ] Pattern keyword detection
- [ ] Methodology section check
- [ ] Failure mode documentation

### Phase 5: Contract Analysis
- [ ] Structured format (YAML/JSON)
- [ ] Validation mention
- [ ] Integration documentation

## Quality Scoring

### Score Calculation
```
Base Score: 100
- Errors: -20 points each
- Warnings: -5 points each
- Suggestions: -2 points each
Minimum: 0
```

### Score Ranges
- **95-100**: Excellent, no optimization needed
- **85-94**: Good, minor improvements suggested
- **70-84**: Fair, several improvements recommended
- **50-69**: Poor, significant issues to address
- **<50**: Critical, major rework required

## Success Criteria

**Test passes if**:
1. Optimal skill (3.1) scores ≥95
2. All problem scenarios correctly identify issues
3. Quality scores fall within expected ranges
4. All 5 analysis phases execute correctly
5. Suggestions are actionable and clear

**Test fails if**:
1. False positives (issues flagged in optimal skill)
2. False negatives (issues missed in problem scenarios)
3. Quality scores outside expected ranges
4. Analyzer crashes or hangs
5. Unclear or unhelpful suggestions

## Test Results Template

```
Test Date: _________________
Tester: ___________________
Node Version: ______________

Scenario 3.1 (Optimal): [ PASS / FAIL ]
  Quality Score: ___/100
  Issues: ___
  Suggestions: ___

Scenario 3.2 (High tokens): [ PASS / FAIL ]
  Quality Score: ___/100
  Token warning: [ YES / NO ]

Scenario 3.3 (Vague): [ PASS / FAIL ]
  Quality Score: ___/100
  Terms detected: _______________

Scenario 3.4 (Passive): [ PASS / FAIL ]
  Quality Score: ___/100
  Instances counted: ___

Scenario 3.5 (Long sentences): [ PASS / FAIL ]
  Quality Score: ___/100
  Avg length: ___

Scenario 3.6 (No pattern): [ PASS / FAIL ]
  Quality Score: ___/100
  Pattern warning: [ YES / NO ]

Scenario 3.7 (Redundant): [ PASS / FAIL ]
  Quality Score: ___/100
  Duplicates found: ___

Scenario 3.8 (Large code): [ PASS / FAIL ]
  Quality Score: ___/100
  Long blocks: ___

Scenario 3.9 (No contracts): [ PASS / FAIL ]
  Quality Score: ___/100
  Suggestion issued: [ YES / NO ]

Scenario 3.10 (No integration): [ PASS / FAIL ]
  Quality Score: ___/100
  Suggestion issued: [ YES / NO ]

Overall: [ PASS / FAIL ]
```

## Performance Benchmarks

- Analysis time: <500ms per skill
- Memory usage: <50MB
- Support for skills up to 5000 words
- Batch processing: 100 skills in <1 minute

## Related Tests

- Test 1: Generation (produces optimizable skills)
- Test 2: Validation (structural correctness)
- Example 3: Optimization workflow in practice
