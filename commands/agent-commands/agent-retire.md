---
name: agent-retire
description: Gracefully retire agents with memory preservation and task handoff
version: 2.0.0
category: agent-management
complexity: medium
tags: [agents, retirement, cleanup, handoff, graceful-shutdown]
author: ruv-SPARC Agent Team
created: 2025-11-01
last_updated: 2025-11-01
dependencies: [agent-list, memory-export, task-orchestrate]
chains_with: [agent-spawn, agent-clone, memory-export]
evidence_based_techniques: [plan-and-solve, self-consistency]
---

# /agent-retire - Graceful Agent Retirement

## Overview

The `/agent-retire` command gracefully retires agents with proper cleanup, memory preservation, task handoff, and resource deallocation.

## Usage

```bash
# Retire specific agent
npx claude-flow@alpha agent retire --agent-id "coder-123"

# Retire agent with memory preservation
npx claude-flow@alpha agent retire --agent-id "coder-123" --preserve-memory

# Retire and handoff tasks
npx claude-flow@alpha agent retire \
  --agent-id "tester-456" \
  --handoff-to "tester-789"

# Retire all idle agents
npx claude-flow@alpha agent retire --idle --threshold 1h

# Force retire (immediate shutdown)
npx claude-flow@alpha agent retire --agent-id "stuck-agent" --force
```

## Parameters

- `--agent-id <id>` - Agent ID to retire
- `--preserve-memory` - Export agent memory before retirement
- `--handoff-to <agent-id>` - Agent to receive tasks
- `--idle` - Retire all idle agents
- `--threshold <duration>` - Idle threshold (e.g., `1h`, `30m`)
- `--force` - Immediate shutdown without cleanup
- `--reason <text>` - Retirement reason (for audit)

## Retirement Process

1. **Pre-Retirement**:
   - Pause new task assignments
   - Complete in-progress tasks or handoff
   - Export agent memory

2. **Handoff**:
   - Transfer tasks to designated agent
   - Share learned patterns
   - Update coordination topology

3. **Cleanup**:
   - Deallocate resources
   - Clear temporary data
   - Update agent registry

4. **Finalization**:
   - Log retirement event
   - Archive agent data
   - Update metrics

## See Also

- `/agent-spawn` - Spawn new agents
- `/agent-clone` - Clone agents
- `/agent-list` - List active agents
- `/memory-export` - Export memory

---

**Version**: 2.0.0
