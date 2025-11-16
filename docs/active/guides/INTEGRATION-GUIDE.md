# Integration Guide: Three-Tier Modular Architecture

## Overview

This guide shows how all 14 skills integrate into the three-tier modular architecture with slash commands.

## Architecture Summary

```
Tier 3: Slash Commands (/command)
         ↓
Tier 2: Cascades (workflows)
         ↓
Tier 1: Micro-Skills (atomic units)
```

**Key Insight**: Cascades are just sequences of slash commands!

## All Available Commands (14 Skills)

### Audit Commands (4)
```bash
/theater-detect <path>              # Find mocks/TODOs/placeholders
/functionality-audit <path>         # Test with Codex auto-fix
/style-audit <path>                 # Lint and polish
/audit-pipeline <path>              # All 3 phases
```

### Multi-Model Commands (7)
```bash
/gemini-megacontext "<task>"        # 1M token context
/gemini-search "<query>"            # Real-time web info
/gemini-media "<description>"       # Generate images/videos
/gemini-extensions "<task>"         # Figma, Stripe, etc.
/codex-auto "<task>"                # Rapid prototyping
/codex-reasoning "<problem>"        # Alternative reasoning
/multi-model "<task>"               # Intelligent orchestrator
```

### Agent Commands (1)
```bash
/agent-rca "<problem>"              # Root cause analysis
```

### Workflow Commands (2)
```bash
/create-micro-skill "<task>"        # Create new micro-skill
/create-cascade "<workflow>"        # Create new cascade
```

## Integration Patterns

### Pattern 1: Simple Sequential Cascade

**Just call commands in order!**

```bash
# Audit pipeline
/theater-detect src/
/functionality-audit src/ --model codex-auto
/style-audit src/
```

No complex YAML needed - cascades are just command sequences.

### Pattern 2: Multi-Model Development Flow

```bash
# Research → design → implement → test → polish
/gemini-search "Best practices for user auth"
/agent-architect "Design auth system with best practices"
/codex-auto "Implement designed auth system"
/functionality-audit --model codex-auto
/style-audit
```

Each command uses the optimal AI:
- Gemini Search: Real-time info
- Claude (agent-architect): Best reasoning
- Codex Auto: Rapid prototyping
- Codex iteration: Auto-fixing tests
- Claude (style): Quality polish

### Pattern 3: Bug Fix with RCA

```bash
# Systematic debugging
/agent-rca "API timeout under load" --context src/api/
/codex-auto "Fix identified root cause"
/functionality-audit --model codex-auto
/test-suite --type regression
```

### Pattern 4: Complete Feature Development

```bash
# Full feature lifecycle
/gemini-search "Latest React 19 patterns"
/gemini-megacontext "Analyze existing React patterns" --context src/
/agent-architect "Design feature using latest patterns"
/gemini-media "Generate UI mockup" --output mockup.png
/codex-auto "Implement feature from design and mockup"
/functionality-audit --model codex-auto
/style-audit
/generate-docs --feature "new feature"
```

### Pattern 5: Legacy Code Modernization

```bash
# Comprehensive modernization
/gemini-megacontext "Analyze entire legacy codebase" --context src/
/theater-detect src/ --fix  # Replace mocks with real code
/codex-auto "Refactor to modern patterns"
/functionality-audit --model codex-auto  # Test everything
/style-audit  # Polish to standards
/gemini-media "Generate new architecture diagram"
```

## Codex Iteration Pattern (Core Innovation)

The `/functionality-audit` command with `--model codex-auto` implements an automatic fix loop:

```bash
/functionality-audit src/ --model codex-auto
```

**What happens internally:**
```
1. Run test suite
2. For each failing test:
   a. Spawn: /codex-auto "Fix test failure" --sandbox true
   b. Re-run test
   c. Iterate (max 5 times)
   d. If passing: validate no regressions → apply fix
   e. If still failing: escalate to user
3. Repeat until all tests pass or max iterations reached
```

This is the secret sauce - automated test-fix-retest loops with Codex in sandbox.

## Command Composition

### Chaining with Pipes
```bash
# Output of one command feeds next
/extract-data input.json | /validate-api --schema schema.yaml | /transform --format csv
```

### Conditional Execution
```bash
# Run second command only if first succeeds
/validate-quality src/ && /deploy-prod || /notify-team "Quality check failed"
```

### Parallel Execution
```bash
# Run multiple commands simultaneously
parallel ::: "/lint-code src/" "/security-scan src/" "/test-coverage src/"
# Then merge results
/merge-reports lint.json security.json coverage.json
```

### Loops
```bash
# Iterative refinement
while [[ $(quality-score src/) -lt 85 ]]; do
  /refactor-code src/
  /style-audit src/
done
```

## Creating New Skills and Commands

### Step 1: Create Micro-Skill
```bash
/create-micro-skill "Validate JSON against schemas" \
  --technique program-of-thought
```

Creates: `.claude/skills/validate-json/SKILL.md`

### Step 2: Create Slash Command (Auto-Generated)
Command is automatically generated: `/validate-json`

Or manually create:
```bash
/create-command validate-json --binding skill:validate-json
```

### Step 3: Use in Cascades
```bash
# Now use in any cascade
/extract-json data.json
/validate-json --schema schema.yaml
/transform-json --format csv
```

## Best Practices

### 1. Prefer Commands Over Skills Directly
✅ Good: `/theater-detect src/`
❌ Avoid: Invoking theater-detection-audit skill directly

### 2. Compose Cascades from Commands
✅ Good:
```bash
/theater-detect src/
/functionality-audit src/
/style-audit src/
```

❌ Avoid: Complex YAML cascade definitions

### 3. Use Appropriate AI for Each Task
- Large context (30K+ lines) → `/gemini-megacontext`
- Current information → `/gemini-search`
- Visual output → `/gemini-media`
- Rapid prototyping → `/codex-auto`
- Auto-fixing tests → `/functionality-audit --model codex-auto`
- Best reasoning → Claude (default)

### 4. Leverage Codex Iteration for Testing
Always use `--model codex-auto` with functionality-audit for automatic test fixing:
```bash
/functionality-audit src/ --model codex-auto --max-iterations 5
```

### 5. Build Reusable Cascade Scripts
Create executable cascade scripts:
```bash
#!/bin/bash
/audit-pipeline "$1"
```

Save as: `.claude/cascades/audit.sh`
Run as: `./audit.sh src/`

## Example Workflow Scripts

All example cascades are in `examples/cascades/`:

1. **simple-audit-cascade.sh** - Basic 3-phase audit
2. **multi-model-cascade.sh** - Research → design → implement → test
3. **bug-fix-cascade.sh** - RCA → fix → test → validate

Usage:
```bash
# Run simple audit
./examples/cascades/simple-audit-cascade.sh src/

# Develop new feature
./examples/cascades/multi-model-cascade.sh "User authentication with JWT"

# Fix bug
./examples/cascades/bug-fix-cascade.sh "API timeout under load" src/api/
```

## Command Reference

### Audit Pipeline
```bash
# Complete audit (all phases)
/audit-pipeline src/

# Specific phase
/audit-pipeline src/ --phase functionality --model codex-auto

# With report
/audit-pipeline src/ --output comprehensive-audit.json
```

### Multi-Model Routing
```bash
# Explicit model selection
/analyze src/ --model gemini-megacontext

# Auto-select based on task
/analyze src/  # Auto-routes to appropriate model
```

### Codex Iteration
```bash
# Standard (5 iterations)
/functionality-audit src/ --model codex-auto

# Custom iteration limit
/functionality-audit src/ --model codex-auto --max-iterations 10

# Sandbox disabled (not recommended)
/functionality-audit src/ --model codex-auto --sandbox false
```

## Troubleshooting

### Command Not Found
```bash
# List all available commands
/help

# Check if skill is installed
ls ~/.claude/skills/
ls .claude/skills/
```

### Cascade Failures
```bash
# Run phases individually to isolate issue
/theater-detect src/       # Phase 1
/functionality-audit src/  # Phase 2
/style-audit src/          # Phase 3
```

### Codex Iteration Not Fixing
```bash
# Increase iteration limit
/functionality-audit src/ --model codex-auto --max-iterations 10

# Check sandbox logs
/functionality-audit src/ --model codex-auto --verbose
```

### Model Selection Issues
```bash
# Explicitly specify model
/analyze src/ --model gemini-megacontext

# Check model availability
/multi-model status
```

## Next Steps

1. **Start Simple**: Use existing commands like `/audit-pipeline src/`
2. **Create Micro-Skills**: Build atomic skills with `/create-micro-skill`
3. **Compose Cascades**: Chain commands into workflows
4. **Save Scripts**: Create reusable cascade scripts
5. **Iterate**: Refine skills and cascades based on real use

## Summary

The three-tier architecture is now fully integrated with 14 skills and commands:

- **Tier 1**: 14 micro-skills (atomic capabilities)
- **Tier 2**: Cascades = command sequences (no complex YAML)
- **Tier 3**: 14+ slash commands (ergonomic interfaces)

**Key Innovation**: Cascades are just sequences of slash commands, making them simple, readable, and scriptable.

**Secret Sauce**: Codex iteration loop in `/functionality-audit` automatically fixes failing tests.

Start building with these commands and create new micro-skills as needed!
