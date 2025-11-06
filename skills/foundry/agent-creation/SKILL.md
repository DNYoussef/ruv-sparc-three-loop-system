---
name: agent-creation
description: Systematic agent creation using evidence-based prompting principles and 4-phase SOP methodology. Use when creating new specialist agents, refining existing agent prompts, or designing multi-agent systems. Applies Chain-of-Thought, few-shot learning, and role-based prompting. Includes validation scripts, templates, comprehensive tests, and production-ready examples.
---

# Agent Creation - Systematic Agent Design

Evidence-based agent creation following best practices for prompt engineering and agent specialization.

---

## When to Use This Skill

Use when creating new specialist agents for specific domains, refining existing agent capabilities, designing multi-agent coordination systems, or implementing role-based agent hierarchies.

---

## 4-Phase Agent Creation SOP

### Phase 1: Specification
- Define agent purpose and domain
- Identify core capabilities needed
- Determine input/output formats
- Specify quality criteria

**Tools**: Use `resources/scripts/generate_agent.sh` for automated generation

### Phase 2: Prompt Engineering
- Apply evidence-based prompting principles
- Use Chain-of-Thought for reasoning tasks
- Implement few-shot learning with examples (2-5 examples)
- Define role and persona clearly

**Reference**: See `references/prompting-principles.md` for detailed techniques

### Phase 3: Testing & Validation
- Test with diverse inputs
- Validate output quality using `resources/scripts/validate_agent.py`
- Measure performance metrics
- Iterate based on results

**Tests**: Run tests from `tests/` directory (basic, specialist, integration)

### Phase 4: Integration
- Define agent coordination protocols
- Establish communication patterns via Memory MCP
- Configure memory and state management
- Deploy with monitoring using Claude-Flow hooks

**Examples**: See `examples/example-1-python-specialist.md` for complete walkthrough

---

## Quick Start

### Generate Agent

```bash
cd resources/scripts
./generate_agent.sh <agent-name> <category> --interactive
```

**Categories**: specialist, coordinator, hybrid, research, development, testing, documentation, security

### Validate Agent

```bash
python3 validate_agent.py <path-to-agent-spec.yaml>
```

**Expected Output**: All validation checks pass (metadata, role, capabilities, prompting, quality, integration)

### Deploy Agent

```bash
# Copy to Claude-Flow agents directory
cp agent-spec.yaml ~/.claude-flow/agents/<agent-name>.yaml

# Test with Claude Code
Task("<Agent Name>", "<Task Description>", "<category>")
```

---

## Evidence-Based Prompting Principles

### 1. Role Definition
Clear agent identity and expertise improves performance by 15-30%

```yaml
role:
  identity: "You are a [Specific Role] with expertise in [Domain]"
  expertise: ["skill-1", "skill-2", "skill-3"]
  responsibilities: ["task-1", "task-2"]
```

### 2. Chain-of-Thought Reasoning
Explicit reasoning steps improve accuracy by 20-40% on complex tasks

```yaml
reasoning_steps:
  - "Step 1: Analyze requirements"
  - "Step 2: Identify solutions"
  - "Step 3: Evaluate trade-offs"
  - "Step 4: Select optimal approach"
```

### 3. Few-Shot Learning
2-5 examples improve performance by 30-50% compared to zero-shot

```yaml
examples:
  - input: "Concrete example input"
    reasoning: "Step-by-step thinking"
    output: "Expected output with explanation"
```

### 4. Plan-and-Solve
Planning before execution reduces errors by 25-35% on complex workflows

```yaml
workflow:
  - name: "Planning Phase"
    steps: ["Understand requirements", "Outline approach"]
  - name: "Execution Phase"
    steps: ["Implement solution", "Handle edge cases"]
```

**Reference**: Complete guide in `references/prompting-principles.md`

---

## Agent Types

### Specialist Agents
- Domain-specific expertise (Python, React, SQL)
- Single responsibility principle
- 5-7 core competencies
- Deep technical knowledge

**Example**: Python Backend Specialist with FastAPI, SQLAlchemy, pytest

### Coordinator Agents
- Multi-agent orchestration
- Task delegation and routing
- Progress monitoring
- Dependency management

**Example**: Backend Coordinator managing API, database, and testing agents

### Hybrid Agents
- Multi-domain capabilities
- Adaptive role switching
- 2-3 related domains
- Context-aware mode switching

**Example**: Full-Stack Developer switching between backend and frontend

**Reference**: Complete patterns in `references/agent-patterns.md`

---

## Integration

### Claude Code Task Tool

```javascript
// Spawn agent for parallel execution
Task("Agent Name", "Task description", "category")
```

### Memory MCP

```yaml
integration:
  memory_mcp:
    enabled: true
    tagging_protocol:
      WHO: "agent-name"
      WHEN: "timestamp"
      PROJECT: "project-name"
      WHY: "intent"
```

### Claude-Flow Hooks

```yaml
integration:
  hooks:
    pre_task: ["npx claude-flow@alpha hooks pre-task"]
    post_task: ["npx claude-flow@alpha hooks post-task"]
```

---

## Resources

### Scripts
- **validate_agent.py**: Comprehensive agent specification validator
- **generate_agent.sh**: Interactive agent generation with templates

### Templates
- **agent-spec.yaml**: Complete agent specification (250+ lines)
- **capabilities.json**: Structured capabilities configuration (150+ lines)

### Tests
- **test-1-basic.md**: Basic agent creation (~10 min)
- **test-2-specialist.md**: Specialist agent with advanced config (~20 min)
- **test-3-integration.md**: Multi-agent coordination (~30 min)

### Examples
- **example-1-python-specialist.md**: Complete Python specialist walkthrough

### Documentation
- **prompting-principles.md**: Evidence-based prompting techniques
- **agent-patterns.md**: Agent design patterns and anti-patterns
- **agent-creation-process.dot**: GraphViz process flow diagram

---

## Quality Assurance

### Validation Checks
- YAML syntax and structure
- Metadata completeness (name, version, category, description)
- Role definition (identity, expertise, responsibilities)
- Capability specifications (primary, secondary, tools)
- Prompting techniques (chain-of-thought, few-shot, role-based)
- Few-shot examples (2+ with input/reasoning/output)
- Quality criteria (success criteria, failure modes, metrics)
- Integration configuration (Memory MCP, Claude-Flow hooks)

### Success Criteria
- Functional correctness > 95%
- Output completeness > 90%
- Test coverage > 80%
- Response time < 30 seconds

---

## Process Flow

See `graphviz/agent-creation-process.dot` for visual workflow:

1. **Specification Phase**: Define purpose, capabilities, quality criteria
2. **Prompting Phase**: Apply evidence-based techniques, create examples
3. **Testing Phase**: Validate with diverse inputs, measure metrics
4. **Integration Phase**: Configure coordination, deploy with monitoring

**Decision Points**: Quality checks at each phase with iteration loops

---

## Next Steps

1. **Read Documentation**: Start with `references/prompting-principles.md`
2. **Run Example**: Follow `examples/example-1-python-specialist.md`
3. **Generate Agent**: Use `resources/scripts/generate_agent.sh`
4. **Validate**: Run `resources/scripts/validate_agent.py`
5. **Test**: Execute tests from `tests/` directory
6. **Deploy**: Copy to Claude-Flow and test with Claude Code
7. **Monitor**: Track performance and iterate

---

**Version**: 2.0.0 (Gold Tier)
**Status**: Production Ready
**Files**: 14+ (scripts, templates, tests, examples, documentation)
