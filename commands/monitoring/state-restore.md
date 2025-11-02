---
name: state-restore
description: Restore system state from checkpoints with validation and rollback
version: 2.0.0
category: state-management
complexity: high
tags: [state, restore, recovery, rollback, disaster-recovery]
author: ruv-SPARC State Team
created: 2025-11-01
last_updated: 2025-11-01
dependencies: [state-checkpoint, memory-import, agent-spawn]
chains_with: [state-checkpoint, state-diff, memory-import]
evidence_based_techniques: [self-consistency, plan-and-solve]
---

# /state-restore - System State Recovery

## Overview

The `/state-restore` command restores the system to a previous state from checkpoints with validation, component selection, and automatic rollback on failure.

## Usage

```bash
# Restore from checkpoint
npx claude-flow@alpha state restore --checkpoint "pre-deploy-v2.1"

# Restore with validation
npx claude-flow@alpha state restore --checkpoint "safe-point" --validate

# Partial restore (memory only)
npx claude-flow@alpha state restore --checkpoint "backup" --components "memory"

# Dry run to preview
npx claude-flow@alpha state restore --checkpoint "old-state" --dry-run

# Force restore without backup
npx claude-flow@alpha state restore --checkpoint "emergency" --force --no-backup
```

## Parameters

- `--checkpoint <name>` - Checkpoint name to restore (required)
- `--components <list>` - Specific components: `memory`, `agents`, `tasks`, `config`
- `--validate` - Validate before restoring
- `--dry-run` - Preview restore
- `--force` - Skip confirmation
- `--no-backup` - Skip pre-restore backup
- `--rollback-on-error` - Auto-rollback if restore fails

## See Also

- `/state-checkpoint` - Create checkpoints
- `/state-diff` - Compare states
- `/memory-import` - Import memory

---

**Version**: 2.0.0
