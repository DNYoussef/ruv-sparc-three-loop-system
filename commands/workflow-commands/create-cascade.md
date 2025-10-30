---
name: create-cascade
binding: skill:cascade-orchestrator
category: workflow
version: 1.0.0
---

# /create-cascade

Creates workflow cascades that coordinate micro-skills with multi-model routing.

## Usage
```bash
/create-cascade "<workflow>" [options]
```

## Parameters
- `workflow` - Workflow goal description (required)
- `--stages` - Comma-separated list of stages or commands (optional)
- `--parallel` - Enable parallel execution where possible (default: false)
- `--multi-model` - Enable intelligent AI routing (default: true)
- `--output` - Output file for cascade definition (default: .claude/cascades/<name>.yaml)

## Examples
```bash
# Simple sequential cascade
/create-cascade "Complete code quality audit" \
  --stages "/theater-detect,/functionality-audit,/style-audit"

# With multi-model routing
/create-cascade "Research and implement feature" \
  --stages "/gemini-search,/codex-auto,/functionality-audit" \
  --multi-model true

# Parallel execution
/create-cascade "Comprehensive quality checks" \
  --stages "/lint-code,/security-scan,/test-coverage" \
  --parallel true
```

## Cascade Patterns

### Pattern 1: Sequential Pipeline
```yaml
# Just call slash commands in order!
/extract-data input.json
/validate-data --strict
/transform-data --format csv
/generate-report
```

### Pattern 2: Parallel Fan-Out
```yaml
parallel:
  - /lint-code src/
  - /security-scan src/
  - /test-coverage src/
then:
  - /merge-reports
```

### Pattern 3: Conditional Branching
```bash
/validate-quality src/
if pass: /deploy-prod
if fail: /generate-error-report
```

### Pattern 4: Multi-Model Cascade
```bash
/gemini-search "Latest best practices"
/codex-auto "Implement with best practices"
/functionality-audit --model codex-auto
/style-audit
```

### Pattern 5: Codex Iteration Loop
```bash
/functionality-audit src/ --model codex-auto
# Automatically:
# 1. Run tests
# 2. For failures: /codex-auto fix
# 3. Re-test (iterate 5x)
# 4. Apply if passing
```

## Key Insight
Cascades are just sequences of slash commands! No need for complex YAML - just call commands in order.

## Created Structure
```
.claude/cascades/<cascade-name>/
├── cascade.yaml       # Cascade definition
├── README.md          # Usage documentation
└── examples/          # Usage examples
```

## Integration
- ✅ Uses existing slash commands
- ✅ Can be bound to new slash command
- ✅ Supports multi-model routing
- ✅ Enables parallel execution

## Chains with
```bash
# Create cascade → bind to command
/create-cascade "Quality pipeline" --stages "..." | \
/create-command quality-check --binding cascade:quality-pipeline
```

## See also
- `/create-micro-skill` - Create atomic skills first
- `/create-command` - Bind cascade to command
- `/audit-pipeline` - Example cascade
