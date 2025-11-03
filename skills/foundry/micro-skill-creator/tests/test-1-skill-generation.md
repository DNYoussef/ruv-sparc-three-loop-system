# Test 1: Skill Generation

## Test Objective
Verify that the micro-skill-creator can generate complete, valid atomic micro-skills with evidence-based agent design.

## Test Scenarios

### Scenario 1.1: Generate Data Extraction Micro-Skill

**Input**:
```yaml
name: invoice-data-extractor
pattern: self-consistency
domain: invoice processing
purpose: Extract line items from PDF invoices
```

**Expected Output**:
- SKILL.md with valid YAML frontmatter
- Specialist agent using self-consistency pattern
- Explicit input contract (PDF file path, target schema)
- Explicit output contract (extracted data, confidence scores, ambiguities)
- Failure mode documentation (unclear text, missing fields)
- Test cases (normal, boundary, error, edge)

**Validation Criteria**:
- [ ] SKILL.md exists and is valid
- [ ] Frontmatter contains name, description, version
- [ ] Agent section includes self-consistency methodology
- [ ] Input/output contracts use YAML format
- [ ] At least 3 failure modes documented
- [ ] At least 4 test cases included

### Scenario 1.2: Generate API Validation Micro-Skill

**Input**:
```yaml
name: openapi-response-validator
pattern: program-of-thought
domain: API testing
purpose: Validate API responses against OpenAPI 3.0 schemas
```

**Expected Output**:
- SKILL.md with program-of-thought methodology
- Agent with systematic decomposition approach
- Input contract (response object, schema file)
- Output contract (validation results, violations, suggested fixes)
- Step-by-step reasoning documentation

**Validation Criteria**:
- [ ] Program-of-thought pattern clearly stated
- [ ] Methodology includes decomposition steps
- [ ] Validation rules explicitly defined
- [ ] Output includes violations array with severity levels
- [ ] Suggested fixes provided for common errors

### Scenario 1.3: Generate Code Generation Micro-Skill

**Input**:
```yaml
name: rest-endpoint-generator
pattern: plan-and-solve
domain: backend development
purpose: Generate REST API endpoints from OpenAPI specifications
```

**Expected Output**:
- SKILL.md with plan-and-solve framework
- Agent with planning and execution phases
- Input contract (OpenAPI spec, framework preference)
- Output contract (generated code, metadata with decisions made)
- Completeness validation steps

**Validation Criteria**:
- [ ] Plan-and-solve pattern with distinct planning phase
- [ ] Methodology includes plan creation and systematic execution
- [ ] Generation metadata includes decision rationale
- [ ] Validation checkpoints at each step
- [ ] Quality standards defined (code style, completeness)

### Scenario 1.4: Generate Analysis Micro-Skill

**Input**:
```yaml
name: security-vulnerability-analyzer
pattern: program-of-thought
domain: security analysis
purpose: Analyze code for security vulnerabilities using OWASP Top 10
```

**Expected Output**:
- SKILL.md combining program-of-thought and self-consistency
- Agent with analytical framework and cross-validation
- Input contract (code files, analysis depth)
- Output contract (findings with severity, recommendations, confidence levels)
- OWASP Top 10 reference integration

**Validation Criteria**:
- [ ] Hybrid pattern approach documented
- [ ] Analysis framework explicitly defined
- [ ] Findings categorized by severity
- [ ] Confidence levels provided for each finding
- [ ] Recommendations prioritized by risk

## Test Execution

### Using skill-generator.py

```bash
# Test Scenario 1.1
python resources/scripts/skill-generator.py \
  --name invoice-data-extractor \
  --pattern self-consistency \
  --domain "invoice processing" \
  --output ./test-output

# Test Scenario 1.2
python resources/scripts/skill-generator.py \
  --name openapi-response-validator \
  --pattern program-of-thought \
  --domain "API testing" \
  --output ./test-output

# Test Scenario 1.3
python resources/scripts/skill-generator.py \
  --name rest-endpoint-generator \
  --pattern plan-and-solve \
  --domain "backend development" \
  --output ./test-output

# Test Scenario 1.4
python resources/scripts/skill-generator.py \
  --name security-vulnerability-analyzer \
  --pattern program-of-thought \
  --domain "security analysis" \
  --output ./test-output
```

### Using Interactive Mode

```bash
python resources/scripts/skill-generator.py --interactive
```

Follow prompts to create each test skill.

## Validation Steps

### Step 1: Structure Validation
```bash
for skill in ./test-output/*; do
  bash resources/scripts/skill-validator.sh "$skill"
done
```

Expected: All validations pass with 0 errors

### Step 2: Content Quality
```bash
for skill in ./test-output/*; do
  node resources/scripts/skill-optimizer.js "$skill"
done
```

Expected: Quality score ≥ 80/100 for all generated skills

### Step 3: Manual Review

Check each generated SKILL.md for:
- Clarity and readability
- Evidence-based pattern correctly applied
- Agent methodology matches pattern
- Contracts are complete and unambiguous
- Failure modes are realistic and actionable

## Success Criteria

**Test passes if**:
1. All 4 skills generate without errors
2. All structure validations pass
3. Quality scores ≥ 80/100
4. Manual review confirms pattern correctness
5. All validation criteria checkboxes checked

**Test fails if**:
1. Any skill fails to generate
2. Validation errors occur
3. Quality scores < 80/100
4. Evidence patterns incorrectly applied
5. Contracts are ambiguous or incomplete

## Test Results Template

```
Test Date: _________________
Tester: ___________________

Scenario 1.1: [ PASS / FAIL ]
  - Validation: [ PASS / FAIL ]
  - Quality Score: ____/100
  - Notes: ________________________________

Scenario 1.2: [ PASS / FAIL ]
  - Validation: [ PASS / FAIL ]
  - Quality Score: ____/100
  - Notes: ________________________________

Scenario 1.3: [ PASS / FAIL ]
  - Validation: [ PASS / FAIL ]
  - Quality Score: ____/100
  - Notes: ________________________________

Scenario 1.4: [ PASS / FAIL ]
  - Validation: [ PASS / FAIL ]
  - Quality Score: ____/100
  - Notes: ________________________________

Overall: [ PASS / FAIL ]
```

## Notes

- Test with different Python versions (3.8, 3.9, 3.10+)
- Verify generated skills work in Claude Code environment
- Check that neural training integration is properly configured
- Ensure cascade compatibility is correctly set

## Related Tests

- Test 2: Validation accuracy
- Test 3: Optimization effectiveness
- Example 1: Real-world usage scenarios
