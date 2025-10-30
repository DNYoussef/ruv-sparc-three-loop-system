---
name: agent-rca
binding: agent:root-cause-analyzer
category: agent
version: 1.0.0
---

# /agent-rca

Spawns Root Cause Analyzer agent for systematic debugging and finding real underlying problems.

## Usage
```bash
/agent-rca "<problem>" [options]
```

## Parameters
- `problem` - Problem description (required)
- `--context` - Relevant files or directory (optional)
- `--depth` - Investigation depth: shallow|normal|deep (default: normal)
- `--output` - Output file for RCA report (default: stdout)

## Examples
```bash
# Debug intermittent issue
/agent-rca "API times out intermittently under load" --context src/api/

# Production incident
/agent-rca "Database connection pool exhausted in production" --depth deep

# Performance issue
/agent-rca "Dashboard load time increased from 200ms to 3s" --context src/dashboard/

# Generate report
/agent-rca "Memory leak in background worker" --output rca-memory-leak.md
```

## 5-Phase Methodology

### Phase 1: Symptom Collection
- Gather observable symptoms
- Collect logs, metrics, traces
- Document reproduction steps
- Apply 5 Whys technique

### Phase 2: Hypothesis Generation
- Generate potential root causes
- Work backwards from failure points
- Apply inverse reasoning
- Consider system interactions

### Phase 3: Forensic Investigation
- Test each hypothesis
- Gather evidence
- Use eliminative logic
- Isolate variables

### Phase 4: Root Cause Identification
- Distinguish symptoms from root causes
- Validate the real underlying problem
- Verify causality

### Phase 5: Validation & Solution Design
- Confirm root cause
- Design fix
- Verify fix resolves issue
- Check for regressions

## Chains with
```bash
# RCA → fix with Codex
/agent-rca "Bug in auth flow" --context src/auth/ | \
/codex-auto "Fix identified root cause"

# RCA → comprehensive test
/agent-rca "Intermittent failure" | \
/functionality-audit --model codex-auto
```

## See also
- `/codex-auto` - Quick prototyping/fixing
- `/functionality-audit` - Systematic testing
- `/agent-reviewer` - Code review
