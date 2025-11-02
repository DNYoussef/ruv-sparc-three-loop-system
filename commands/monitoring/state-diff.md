---
name: state-diff
description: Compare state snapshots to identify changes and drift
version: 2.0.0
category: state-management
complexity: medium
tags: [state, diff, comparison, analysis, change-tracking]
author: ruv-SPARC State Team
created: 2025-11-01
last_updated: 2025-11-01
dependencies: [state-checkpoint, memory-retrieve]
chains_with: [state-checkpoint, state-restore, memory-stats]
evidence_based_techniques: [self-consistency, program-of-thought]
---

# /state-diff - State Comparison & Analysis

## Overview

The `/state-diff` command compares two state snapshots to identify changes, additions, deletions, and drift between states.

## Usage

```bash
# Compare two checkpoints
npx claude-flow@alpha state diff \
  --from "checkpoint-20251001" \
  --to "checkpoint-20251101"

# Compare checkpoint with current state
npx claude-flow@alpha state diff --from "baseline" --to current

# Show only specific components
npx claude-flow@alpha state diff \
  --from "old" --to "new" \
  --components "memory,agents"

# Export diff report
npx claude-flow@alpha state diff \
  --from "before" --to "after" \
  --export "diff-report.json"
```

## Parameters

- `--from <checkpoint>` - Source checkpoint (required)
- `--to <checkpoint>` - Target checkpoint (required, or "current")
- `--components <list>` - Components to compare
- `--detailed` - Show detailed diff with values
- `--summary` - Summary only
- `--export <path>` - Export diff report

## Output

Shows:
- Added keys/entries
- Deleted keys/entries
- Modified values
- Unchanged entries
- Statistics summary

## See Also

- `/state-checkpoint` - Create checkpoints
- `/state-restore` - Restore states
- `/memory-stats` - Memory statistics

---

**Version**: 2.0.0
