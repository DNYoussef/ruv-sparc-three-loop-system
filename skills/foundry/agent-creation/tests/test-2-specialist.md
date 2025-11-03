# Test 2: Specialist Agent with Advanced Configuration

**Objective**: Create and validate a specialist agent with advanced prompting techniques and integration

**Test Type**: Integration validation
**Difficulty**: Intermediate
**Duration**: ~20 minutes

---

## Test Scenario

Create a fully-configured TypeScript specialist agent with chain-of-thought reasoning, few-shot learning, and Memory MCP integration.

### Prerequisites

- Test 1 completed successfully
- Understanding of TypeScript domain
- Claude-Flow and Memory MCP configured

### Test Steps

#### 1. Interactive Agent Generation

```bash
cd resources/scripts
./generate_agent.sh typescript-specialist specialist --interactive
```

**Interactive Prompts**:

```
Agent Description (80-150 words):
> Expert TypeScript specialist focused on type-safe development, advanced type
> manipulation, and modern JavaScript/TypeScript patterns. Provides guidance on
> TypeScript configuration, type inference, generics, decorators, and integration
> with frameworks like React, Node.js, and Nest.js. Emphasizes code quality through
> static analysis, comprehensive testing with Jest, and adherence to TypeScript
> best practices. Specializes in monorepo setups, build tooling with tsc/esbuild,
> and performance optimization.

Expertise Areas (comma-separated):
> TypeScript,Type Systems,React,Node.js,Generics,Decorators,Build Tools

Primary Capabilities (comma-separated):
> Type-safe development,Advanced type manipulation,Framework integration,Testing with Jest,Build optimization

Select techniques (e.g., 1,2,3):
> 1,2,3,4
```

**Expected Output**: Agent created with customized configuration

#### 2. Validate Advanced Configuration

```bash
python3 validate_agent.py ../../tests/output/typescript-specialist/agent-spec.yaml
```

**Expected**: All checks pass

#### 3. Verify Prompting Techniques

```bash
grep -A 10 "prompting:" ../../tests/output/typescript-specialist/agent-spec.yaml
```

**Expected Output**:
```yaml
prompting:
  techniques:
    - chain-of-thought
    - few-shot
    - role-based
    - plan-and-solve
```

#### 4. Test Few-Shot Examples

Verify examples are present and well-structured:

```bash
python3 -c "
import yaml
with open('../../tests/output/typescript-specialist/agent-spec.yaml') as f:
    spec = yaml.safe_load(f)
    examples = spec['prompting']['examples']
    print(f'Examples found: {len(examples)}')
    for i, ex in enumerate(examples):
        print(f'  Example {i+1}: {\"input\" in ex} (input), {\"output\" in ex} (output)')
"
```

**Expected**: At least 2 examples with input/output/reasoning

#### 5. Verify Integration Configuration

```bash
python3 -c "
import json
with open('../../tests/output/typescript-specialist/capabilities.json') as f:
    caps = json.load(f)
    integrations = caps['capabilities']['integrations']
    print(f'Integrations configured: {len(integrations)}')
    for integ in integrations:
        print(f'  - {integ[\"name\"]}: {integ[\"enabled\"]}')
"
```

**Expected Output**:
```
Integrations configured: 3
  - memory_mcp: True
  - claude_flow: True
  - connascence_analyzer: True
```

---

## Advanced Validation Checklist

### Prompting Techniques
- [ ] Chain-of-thought enabled
- [ ] Few-shot examples (2-3 minimum)
- [ ] Role-based identity clear
- [ ] Plan-and-solve for complex workflows
- [ ] Reasoning steps defined

### Few-Shot Examples
- [ ] Examples are domain-specific (TypeScript)
- [ ] Each example has input, reasoning, output
- [ ] Examples demonstrate key capabilities
- [ ] Reasoning shows step-by-step thinking
- [ ] Outputs are realistic and complete

### Integration Points
- [ ] Memory MCP configuration complete
- [ ] Tagging protocol (WHO/WHEN/PROJECT/WHY)
- [ ] Claude-Flow hooks defined
- [ ] Connascence Analyzer thresholds set
- [ ] Task template customized

### Quality Assurance
- [ ] Success criteria specific to TypeScript
- [ ] Failure modes identified (type errors, etc.)
- [ ] Metrics measurable (coverage, complexity)
- [ ] Quality thresholds appropriate

---

## Custom Configuration Test

Edit `agent-spec.yaml` to add TypeScript-specific configuration:

```yaml
# Add to capabilities.integrations
- name: "typescript-compiler"
  type: "build_tool"
  enabled: true
  config:
    strict: true
    target: "ES2022"
    module: "ESNext"
    incremental: true
```

**Revalidate**:
```bash
python3 validate_agent.py ../../tests/output/typescript-specialist/agent-spec.yaml
```

---

## Integration Testing

### Test 1: Memory MCP Integration

Create a simple test to verify Memory MCP tagging:

```python
# test_memory_integration.py
import yaml

with open('agent-spec.yaml') as f:
    spec = yaml.safe_load(f)

memory_config = spec['integration']['memory_mcp']
print("Memory MCP Enabled:", memory_config['enabled'])
print("Tagging Protocol:", memory_config['tagging_protocol'])

# Expected output:
# Memory MCP Enabled: True
# Tagging Protocol: {'WHO': 'typescript-specialist', 'PROJECT': '{{PROJECT_NAME}}', 'WHY': '{{INTENT}}'}
```

### Test 2: Claude Code Task Template

Verify task template is properly formatted:

```bash
grep -A 5 "task_template:" ../../tests/output/typescript-specialist/agent-spec.yaml
```

**Expected**:
```yaml
task_template: |
  Task("typescript-specialist", "{{TASK_DESCRIPTION}}", "specialist")
```

### Test 3: Hooks Automation

Verify pre/post task hooks:

```bash
python3 -c "
import yaml
with open('../../tests/output/typescript-specialist/agent-spec.yaml') as f:
    spec = yaml.safe_load(f)
    hooks = spec['integration']['hooks']
    print('Pre-task hooks:', len(hooks['pre_task']))
    print('Post-task hooks:', len(hooks['post_task']))
    print('Post-edit hooks:', len(hooks['post_edit']))
"
```

**Expected**: At least 1 hook in each category

---

## Semantic Validation

### Domain Expertise Check

Verify TypeScript-specific content:

```bash
grep -i "typescript\|type\|generic\|decorator" ../../tests/output/typescript-specialist/agent-spec.yaml | wc -l
```

**Expected**: 10+ mentions of TypeScript concepts

### Example Quality Check

Manually review few-shot examples:

1. **Example 1**: Should demonstrate TypeScript type manipulation
2. **Example 2**: Should show framework integration (React/Node.js)
3. **Reasoning**: Should show step-by-step type-level thinking

**Quality Criteria**:
- Examples are realistic and practical
- Reasoning is clear and educational
- Outputs are production-ready

---

## Performance Testing

### Validation Speed

```bash
time python3 validate_agent.py ../../tests/output/typescript-specialist/agent-spec.yaml --json > /dev/null
```

**Expected**: < 1 second

### File Size

```bash
du -h ../../tests/output/typescript-specialist/agent-spec.yaml
```

**Expected**: 8-15 KB (comprehensive but not bloated)

---

## Test Results

**Date**: _______________
**Tester**: _______________
**Status**: ☐ PASS ☐ FAIL

### Validation Results

| Check | Status | Notes |
|-------|--------|-------|
| Metadata | ☐ PASS ☐ FAIL | |
| Role Definition | ☐ PASS ☐ FAIL | |
| Capabilities | ☐ PASS ☐ FAIL | |
| Prompting Techniques | ☐ PASS ☐ FAIL | |
| Few-Shot Examples | ☐ PASS ☐ FAIL | |
| Integration Config | ☐ PASS ☐ FAIL | |
| Quality Criteria | ☐ PASS ☐ FAIL | |

### Notes

_Record observations on TypeScript-specific configuration quality_

---

## Cleanup

```bash
rm -rf ../../tests/output/typescript-specialist/
```

---

## Next Steps

- Proceed to **Test 3: Integration Testing** for multi-agent scenarios
- Deploy agent to Claude-Flow for real-world testing
- Create additional specialist agents using learned patterns
