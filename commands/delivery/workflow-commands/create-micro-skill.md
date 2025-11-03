---
name: create-micro-skill
binding: skill:micro-skill-creator
category: workflow
version: 1.0.0
---

# /create-micro-skill

Creates atomic, focused micro-skills with evidence-based specialist agents.

## Usage
```bash
/create-micro-skill "<task>" [options]
```

## Parameters
- `task` - Single responsibility description (required)
- `--technique` - Evidence technique: self-consistency|program-of-thought|plan-and-solve (default: auto)
- `--output` - Output directory for skill (default: .claude/skills/<skill-name>)

## Examples
```bash
# Create data extraction skill
/create-micro-skill "Extract structured data from unstructured documents" \
  --technique self-consistency

# Create validation skill
/create-micro-skill "Validate API responses against OpenAPI schemas" \
  --technique program-of-thought

# Create generation skill
/create-micro-skill "Generate API documentation from code annotations" \
  --technique plan-and-solve
```

## Evidence-Based Techniques

### Self-Consistency (Factual Tasks)
For extraction, parsing, data collection
```
1. Extract from multiple perspectives
2. Cross-reference findings
3. Flag inconsistencies
4. Provide confidence scores
```

### Program-of-Thought (Analytical Tasks)
For validation, analysis, reasoning
```
1. Decompose into logical components
2. Work through systematically
3. Show intermediate reasoning
4. Validate consistency
```

### Plan-and-Solve (Complex Tasks)
For generation, design, multi-step tasks
```
1. Create comprehensive plan
2. Break into executable steps
3. Execute systematically
4. Validate at each step
```

## Created Structure
```
.claude/skills/<skill-name>/
├── SKILL.md           # Skill definition
├── templates/         # Template files
└── examples/          # Usage examples
```

## Integration
After creation, skill is automatically:
- ✅ Registered with Claude Code
- ✅ Available for cascade composition
- ✅ Can be bound to slash command

## Chains with
```bash
# Create skill → create command → use in cascade
/create-micro-skill "Validate JSON schemas" | \
/create-command validate-json --binding <skill> | \
/create-cascade quality-pipeline
```

## See also
- `/create-cascade` - Create workflow orchestration
- `/create-command` - Create slash command binding
