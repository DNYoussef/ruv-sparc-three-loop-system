# Test 1: Basic Agent Creation

**Objective**: Validate basic agent creation workflow and specification structure

**Test Type**: Functional validation
**Difficulty**: Basic
**Duration**: ~10 minutes

---

## Test Scenario

Create a basic specialist agent for Python development with minimal configuration.

### Prerequisites

- Agent creation skill loaded
- Python 3.8+ installed
- Access to templates and scripts

### Test Steps

#### 1. Generate Agent Specification

```bash
cd resources/scripts
./generate_agent.sh python-specialist specialist --output ../../tests/output
```

**Expected Output**:
- Directory created: `tests/output/python-specialist/`
- Files generated:
  - `agent-spec.yaml`
  - `capabilities.json`
  - `README.md`
- Success message displayed

#### 2. Validate Generated Specification

```bash
python3 validate_agent.py ../../tests/output/python-specialist/agent-spec.yaml
```

**Expected Results**:
```
======================================================================
AGENT SPECIFICATION VALIDATION REPORT
======================================================================

METADATA: ✓ PASS
ROLE: ✓ PASS
CAPABILITIES: ✓ PASS
PROMPTING: ✓ PASS
QUALITY: ✓ PASS
INTEGRATION: ✓ PASS

======================================================================
✓ All validations passed - Agent specification is ready!
```

#### 3. Verify YAML Syntax

```bash
python3 -c "import yaml; yaml.safe_load(open('../../tests/output/python-specialist/agent-spec.yaml'))"
```

**Expected**: No errors, clean exit

#### 4. Verify JSON Syntax

```bash
python3 -c "import json; json.load(open('../../tests/output/python-specialist/capabilities.json'))"
```

**Expected**: No errors, clean exit

#### 5. Check File Structure

```bash
ls -la ../../tests/output/python-specialist/
```

**Expected Files**:
- `agent-spec.yaml` (5-10 KB)
- `capabilities.json` (3-5 KB)
- `README.md` (500-1000 bytes)

---

## Validation Checklist

### Metadata Validation
- [ ] Agent name is in kebab-case
- [ ] Version follows semver (1.0.0)
- [ ] Category is valid ("specialist")
- [ ] Description is 80-150 words
- [ ] All required fields present

### Role Validation
- [ ] Identity clearly defined
- [ ] Expertise list has 3-7 items
- [ ] Responsibilities list is present
- [ ] Role description is specific

### Capabilities Validation
- [ ] Primary capabilities defined (at least 1)
- [ ] Secondary capabilities present
- [ ] Tools list included
- [ ] Integration points specified

### Prompting Validation
- [ ] At least 2 prompting techniques specified
- [ ] Few-shot examples present (2-3 examples)
- [ ] Each example has input and output
- [ ] Reasoning steps defined

### Quality Validation
- [ ] Success criteria defined
- [ ] Failure modes identified
- [ ] Metrics specified
- [ ] Quality thresholds present

### Integration Validation
- [ ] Claude Code task template present
- [ ] Memory MCP configuration included
- [ ] Hooks automation configured
- [ ] Coordination protocol defined

---

## Expected Behavior

### Success Criteria
1. All files generated without errors
2. Validation script passes all checks
3. YAML and JSON syntax valid
4. File structure matches template
5. Content is semantically correct

### Common Issues

**Issue**: `generate_agent.sh: Permission denied`
**Solution**: Make script executable
```bash
chmod +x generate_agent.sh
```

**Issue**: `Template file not found`
**Solution**: Ensure you're running from correct directory
```bash
cd resources/scripts
```

**Issue**: `Validation fails on description length`
**Solution**: Edit `agent-spec.yaml` and expand description to 80+ words

---

## Manual Review

After automated validation, manually review:

1. **Semantic Correctness**
   - Does the role definition make sense?
   - Are capabilities appropriate for a Python specialist?
   - Are examples relevant?

2. **Completeness**
   - Are all sections filled in?
   - Are placeholder values replaced?
   - Is documentation complete?

3. **Consistency**
   - Do capabilities match the role?
   - Do examples align with expertise?
   - Are integration points appropriate?

---

## Test Results

**Date**: _______________
**Tester**: _______________
**Status**: ☐ PASS ☐ FAIL

### Notes

_Record any observations, issues, or improvements_

---

---

## Cleanup

```bash
# Remove test output
rm -rf ../../tests/output/python-specialist/
```

---

## Next Steps

- Proceed to **Test 2: Specialist Agent** for advanced configuration
- Review generated agent for customization opportunities
- Test agent deployment with Claude Code Task tool
