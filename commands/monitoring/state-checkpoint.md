---
name: state-checkpoint
description: Create point-in-time state snapshots with full system context
version: 2.0.0
category: state-management
complexity: medium
tags: [state, checkpoint, snapshot, backup, recovery, versioning]
author: ruv-SPARC State Team
created: 2025-11-01
last_updated: 2025-11-01
dependencies: [memory-export, agent-list, task-status]
chains_with: [state-restore, state-diff, memory-export]
evidence_based_techniques: [self-consistency, plan-and-solve]
---

# /state-checkpoint - System State Snapshots

## Overview

The `/state-checkpoint` command creates comprehensive point-in-time snapshots of the entire system state, including memory, agent states, task states, and configuration. It enables safe rollback, state comparison, and disaster recovery.

## Purpose

- **Rollback Points**: Create safe restore points before major changes
- **State Versioning**: Track system evolution over time
- **Disaster Recovery**: Full system state backup
- **Testing**: Save/restore states for testing scenarios
- **Debugging**: Capture state for troubleshooting
- **Auditing**: Track state changes for compliance

## Usage

```bash
# Create checkpoint with auto-generated name
npx claude-flow@alpha state checkpoint

# Create named checkpoint
npx claude-flow@alpha state checkpoint --name "before-deployment"

# Create checkpoint with description
npx claude-flow@alpha state checkpoint \
  --name "feature-auth-complete" \
  --description "Auth feature completed, all tests passing"

# Create lightweight checkpoint (memory only)
npx claude-flow@alpha state checkpoint --lightweight

# Create full checkpoint with all context
npx claude-flow@alpha state checkpoint --full

# List existing checkpoints
npx claude-flow@alpha state checkpoint --list

# Show checkpoint details
npx claude-flow@alpha state checkpoint --info "before-deployment"

# Delete old checkpoints
npx claude-flow@alpha state checkpoint --delete "old-checkpoint"

# Auto-checkpoint before dangerous operations
npx claude-flow@alpha state checkpoint --auto --trigger "memory-clear"
```

## Parameters

### Creation Options

- `--name <name>` - Checkpoint name
  - Auto-generated if not provided
  - Format: `checkpoint-YYYYMMDD-HHMMSS`

- `--description <text>` - Checkpoint description
  - Helps identify checkpoint purpose

- `--tags <tags>` - Metadata tags (comma-separated)
  - Example: `--tags "deployment,production,v2.0"`

### Scope Options

- `--full` - Full checkpoint (default)
  - Includes: memory, agents, tasks, config, metrics

- `--lightweight` - Memory only
  - Faster, smaller snapshots

- `--include <components>` - Specific components
  - Options: `memory`, `agents`, `tasks`, `config`, `metrics`
  - Example: `--include "memory,agents"`

- `--exclude <components>` - Exclude components
  - Example: `--exclude "metrics"`

### Management

- `--list` - List all checkpoints
  - Shows name, date, size, tags

- `--info <name>` - Show checkpoint details
  - Full metadata and contents summary

- `--delete <name>` - Delete checkpoint
  - Requires confirmation

- `--delete-old <days>` - Delete checkpoints older than N days
  - Example: `--delete-old 30`

### Automation

- `--auto` - Enable auto-checkpointing
  - Creates checkpoint before specified operations

- `--trigger <operation>` - Operations that trigger auto-checkpoint
  - Examples: `memory-clear`, `memory-import`, `state-restore`

- `--schedule <schedule>` - Scheduled checkpoints
  - Examples: `daily`, `weekly`, `hourly`

## Examples

### Example 1: Basic Checkpoint

```bash
npx claude-flow@alpha state checkpoint --name "safe-point"

# Output:
# [State Checkpoint] Creating checkpoint: safe-point
# [State Checkpoint] Capturing state...
#
# Components:
#   ✓ Memory (1,247 keys, 12.4 MB)
#   ✓ Agents (8 active)
#   ✓ Tasks (23 total, 5 running)
#   ✓ Configuration (45 settings)
#   ✓ Metrics (234 data points)
#
# [State Checkpoint] Writing checkpoint...
# [State Checkpoint] Checkpoint created!
#
# Checkpoint: safe-point
# Location: .checkpoints/safe-point.json
# Size: 13.2 MB
# Created: 2025-11-01 10:30:45
# Components: 5
# Restorable: Yes
```

### Example 2: Pre-Deployment Checkpoint

```bash
npx claude-flow@alpha state checkpoint \
  --name "pre-deploy-v2.1" \
  --description "Before deploying version 2.1 to production" \
  --tags "deployment,production,v2.1" \
  --full

# Output:
# [State Checkpoint] Creating full checkpoint: pre-deploy-v2.1
# [State Checkpoint] Description: Before deploying version 2.1 to production
# [State Checkpoint] Tags: deployment, production, v2.1
#
# Capturing full system state...
# ────────────────────────────────────────────────────────────
# [1/5] Memory snapshot
#   ✓ Exported 1,247 keys (12.4 MB)
#   ✓ Included metadata and tags
#
# [2/5] Agent states
#   ✓ Captured 8 active agents
#   ✓ Saved agent configurations
#   ✓ Exported agent memory
#
# [3/5] Task states
#   ✓ Captured 23 tasks (5 running, 12 pending, 6 complete)
#   ✓ Saved task dependencies
#   ✓ Exported task results
#
# [4/5] Configuration
#   ✓ Exported 45 configuration settings
#   ✓ Included environment variables
#   ✓ Saved feature flags
#
# [5/5] Metrics
#   ✓ Exported 234 metric data points
#   ✓ Included performance history
#
# [State Checkpoint] Checkpoint created!
# Location: .checkpoints/pre-deploy-v2.1.json
# Size: 15.7 MB
# Restore command:
#   npx claude-flow@alpha state restore --checkpoint "pre-deploy-v2.1"
```

### Example 3: List Checkpoints

```bash
npx claude-flow@alpha state checkpoint --list

# Output:
# ════════════════════════════════════════════════════════════
#                    STATE CHECKPOINTS
# ════════════════════════════════════════════════════════════
#
# Name                    Created              Size    Tags
# ────────────────────────────────────────────────────────────
# pre-deploy-v2.1         2025-11-01 10:30     15.7MB  deployment, production, v2.1
# feature-auth-complete   2025-10-30 14:22     13.2MB  feature, auth
# before-major-refactor   2025-10-28 09:15     14.1MB  refactor, backup
# daily-backup-20251027   2025-10-27 03:00     12.8MB  auto, daily
# safe-point              2025-10-25 16:45     13.0MB  manual
#
# Total: 5 checkpoints (68.8 MB)
#
# Commands:
#   Info:    npx claude-flow@alpha state checkpoint --info <name>
#   Restore: npx claude-flow@alpha state restore --checkpoint <name>
#   Delete:  npx claude-flow@alpha state checkpoint --delete <name>
```

### Example 4: Checkpoint Details

```bash
npx claude-flow@alpha state checkpoint --info "pre-deploy-v2.1"

# Output:
# ════════════════════════════════════════════════════════════
#          CHECKPOINT: pre-deploy-v2.1
# ════════════════════════════════════════════════════════════
#
# Metadata
# ────────────────────────────────────────────────────────────
# Created:      2025-11-01 10:30:45
# Description:  Before deploying version 2.1 to production
# Tags:         deployment, production, v2.1
# Size:         15.7 MB
# Location:     .checkpoints/pre-deploy-v2.1.json
# Restorable:   Yes
#
# Components
# ────────────────────────────────────────────────────────────
# Memory:
#   Keys:       1,247
#   Size:       12.4 MB
#   Namespaces: 15
#
# Agents:
#   Active:     8
#   Types:      coder (3), tester (2), reviewer (1), planner (2)
#   Memory:     2.1 MB
#
# Tasks:
#   Total:      23
#   Running:    5
#   Pending:    12
#   Complete:   6
#
# Configuration:
#   Settings:   45
#   Env Vars:   23
#   Flags:      12
#
# Metrics:
#   Data Points: 234
#   Time Range:  2025-10-25 to 2025-11-01
#
# Restore Preview
# ────────────────────────────────────────────────────────────
# This checkpoint will restore:
#   • All memory keys (1,247 keys)
#   • 8 agent states
#   • 23 task states
#   • 45 configuration settings
#   • 234 metric data points
#
# Warning: Current state will be backed up automatically
#
# To restore:
#   npx claude-flow@alpha state restore --checkpoint "pre-deploy-v2.1"
```

### Example 5: Lightweight Checkpoint

```bash
npx claude-flow@alpha state checkpoint --lightweight --name "quick-save"

# Output:
# [State Checkpoint] Creating lightweight checkpoint: quick-save
# [State Checkpoint] Memory only (faster, smaller)
#
# Capturing memory...
#   ✓ Exported 1,247 keys (12.4 MB)
#
# [State Checkpoint] Checkpoint created!
# Location: .checkpoints/quick-save.json
# Size: 12.4 MB (vs 15.7 MB for full)
# Restore: Memory only (agents/tasks not included)
```

### Example 6: Auto-Checkpoint

```bash
# Enable auto-checkpoint before dangerous operations
npx claude-flow@alpha state checkpoint --auto \
  --trigger "memory-clear,memory-import,state-restore"

# Output:
# [State Checkpoint] Auto-checkpoint enabled
# [State Checkpoint] Triggers: memory-clear, memory-import, state-restore
#
# Auto-checkpoint will create snapshots before:
#   • npx claude-flow@alpha memory clear
#   • npx claude-flow@alpha memory import
#   • npx claude-flow@alpha state restore
#
# Configuration saved.
#
# Example trigger:
#   User runs: npx claude-flow@alpha memory clear --namespace "dev/*"
#   System auto-creates: checkpoint-pre-clear-20251101-103045
#   Then executes: memory clear operation
```

## Checkpoint Structure

### Full Checkpoint Format

```json
{
  "version": "2.0.0",
  "checkpoint_name": "pre-deploy-v2.1",
  "created_at": "2025-11-01T10:30:45Z",
  "description": "Before deploying version 2.1 to production",
  "tags": ["deployment", "production", "v2.1"],
  "components": {
    "memory": {
      "keys": 1247,
      "size": 12400000,
      "data": { /* memory export */ }
    },
    "agents": {
      "count": 8,
      "states": [ /* agent states */ ]
    },
    "tasks": {
      "total": 23,
      "states": [ /* task states */ ]
    },
    "config": {
      "settings": { /* configuration */ }
    },
    "metrics": {
      "datapoints": 234,
      "data": [ /* metrics */ ]
    }
  },
  "metadata": {
    "system_version": "2.0.0",
    "can_restore": true,
    "restore_requires": ["memory", "agents"]
  }
}
```

## Implementation Details

### Checkpoint Creation

```typescript
interface CheckpointOptions {
  name?: string;
  description?: string;
  tags?: string[];
  full?: boolean;
  lightweight?: boolean;
  include?: string[];
  exclude?: string[];
}

async function createCheckpoint(options: CheckpointOptions): Promise<Checkpoint> {
  const name = options.name || generateCheckpointName();

  // Determine components to include
  const components = determineComponents(options);

  const checkpoint: Checkpoint = {
    version: '2.0.0',
    checkpoint_name: name,
    created_at: new Date().toISOString(),
    description: options.description,
    tags: options.tags,
    components: {}
  };

  // Capture each component
  if (components.includes('memory')) {
    checkpoint.components.memory = await captureMemory();
  }

  if (components.includes('agents')) {
    checkpoint.components.agents = await captureAgents();
  }

  if (components.includes('tasks')) {
    checkpoint.components.tasks = await captureTasks();
  }

  if (components.includes('config')) {
    checkpoint.components.config = await captureConfig();
  }

  if (components.includes('metrics')) {
    checkpoint.components.metrics = await captureMetrics();
  }

  // Write checkpoint
  await writeCheckpoint(checkpoint);

  return checkpoint;
}
```

## Integration with Other Commands

### Chains With

**Before Checkpoint**:
- `/memory-stats` - Analyze current state
- `/agent-list` - Review active agents
- `/task-status` - Check task states

**After Checkpoint**:
- `/state-restore` - Restore from checkpoint
- `/state-diff` - Compare checkpoints
- `/memory-export` - Additional memory backup

**Workflow**:
```bash
# Safe deployment workflow
npx claude-flow@alpha state checkpoint --name "pre-deploy" --full
npx claude-flow@alpha memory import --file "new-config.json"
# ... deployment operations ...
# If issues:
npx claude-flow@alpha state restore --checkpoint "pre-deploy"
```

## Best Practices

### 1. Checkpoint Before Major Changes
```bash
npx claude-flow@alpha state checkpoint --name "before-refactor" --full
```

### 2. Use Descriptive Names and Tags
```bash
npx claude-flow@alpha state checkpoint \
  --name "feature-auth-v1.2" \
  --description "Auth feature complete" \
  --tags "feature,auth,milestone"
```

### 3. Regular Scheduled Checkpoints
```bash
npx claude-flow@alpha state checkpoint --schedule "daily" --auto
```

## See Also

- `/state-restore` - Restore from checkpoints
- `/state-diff` - Compare checkpoints
- `/memory-export` - Export memory
- `/agent-list` - List agents
- `/task-status` - Task status

---

**Version**: 2.0.0
**Last Updated**: 2025-11-01
**Maintained By**: ruv-SPARC State Team
