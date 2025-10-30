---
name: audit-pipeline
binding: cascade:audit-pipeline
category: workflow
version: 1.0.0
---

# /audit-pipeline

Complete 3-phase quality audit: theater detection → functionality testing → style polish.

## Usage
```bash
/audit-pipeline <path> [options]
```

## Parameters
- `path` - File or directory to audit (required)
- `--phase` - Run specific phase: theater|functionality|style|all (default: all)
- `--model` - AI model for functionality phase: codex-auto|claude (default: codex-auto)
- `--output` - Output file for comprehensive report (default: stdout)
- `--parallel` - Run independent phases in parallel (default: false)

## Examples
```bash
# Complete pipeline
/audit-pipeline src/

# Specific phase only
/audit-pipeline src/ --phase functionality --model codex-auto

# Generate comprehensive report
/audit-pipeline src/ --output comprehensive-audit.json
```

## Pipeline Stages

### Phase 1: Theater Detection
Finds all mocks, TODOs, placeholders, and incomplete implementations.
**Command**: `/theater-detect src/`

### Phase 2: Functionality Audit (with Codex Iteration)
Tests everything in sandbox, auto-fixes failures with Codex iteration loop.
**Command**: `/functionality-audit src/ --model codex-auto`

**Codex Iteration Pattern**:
- For each failing test
- Spawn Codex in sandbox
- Auto-fix failure
- Re-test (max 5 iterations)
- Apply fix if passing
- Escalate if still failing

### Phase 3: Style Audit
Lint, refactor, document to production standards.
**Command**: `/style-audit src/`

## As Cascade Script
```bash
# Simple cascade - just call commands in order!
/theater-detect src/
/functionality-audit src/ --model codex-auto
/style-audit src/
```

## Integration
- Uses `/theater-detect` from theater-detection-audit skill
- Uses `/functionality-audit` from functionality-audit skill
- Uses `/style-audit` from style-audit skill
- Uses `/codex-auto` for auto-fixing in Phase 2

## See also
- `/theater-detect` - Phase 1 only
- `/functionality-audit` - Phase 2 only
- `/style-audit` - Phase 3 only
- `/quality-check` - Alternative quality workflow
