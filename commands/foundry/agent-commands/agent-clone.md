---
name: agent-clone
description: Clone high-performing agents with learned patterns and memory
version: 2.0.0
category: agent-management
complexity: medium
tags: [agents, cloning, replication, scaling, performance]
author: ruv-SPARC Agent Team
created: 2025-11-01
last_updated: 2025-11-01
dependencies: [agent-list, agent-metrics, memory-export]
chains_with: [agent-spawn, agent-benchmark, neural-train]
evidence_based_techniques: [self-consistency, program-of-thought]
---

# /agent-clone - High-Performing Agent Replication

## Overview

The `/agent-clone` command creates clones of high-performing agents with their learned patterns, memory, and optimized configurations.

## Usage

```bash
# Clone specific agent
npx claude-flow@alpha agent clone --agent-id "coder-123"

# Clone with custom name
npx claude-flow@alpha agent clone \
  --agent-id "coder-123" \
  --name "coder-clone-01"

# Clone with memory
npx claude-flow@alpha agent clone \
  --agent-id "top-performer" \
  --clone-memory

# Clone without learned patterns (fresh agent)
npx claude-flow@alpha agent clone \
  --agent-id "coder-123" \
  --no-patterns

# Auto-clone top performers
npx claude-flow@alpha agent clone \
  --auto \
  --top 3 \
  --metric "success-rate"
```

## Parameters

- `--agent-id <id>` - Agent to clone (required)
- `--name <name>` - Clone name (auto-generated if not provided)
- `--clone-memory` - Copy agent's memory
- `--clone-patterns` - Copy learned patterns (default: true)
- `--no-patterns` - Fresh clone without patterns
- `--auto` - Auto-clone top performers
- `--top <n>` - Number of top agents to clone
- `--metric <metric>` - Performance metric for selection

## Cloning Process

1. **Analysis**:
   - Analyze source agent performance
   - Identify successful patterns
   - Extract learned behaviors

2. **Replication**:
   - Create new agent instance
   - Copy configuration
   - Transfer learned patterns

3. **Memory Transfer** (if enabled):
   - Export source agent memory
   - Filter relevant memories
   - Import to clone

4. **Initialization**:
   - Register clone in swarm
   - Update topology
   - Start clone agent

## Clone Types

### Full Clone
Copies everything: config, memory, patterns, metrics baseline

### Pattern Clone
Copies learned patterns and config only (no memory)

### Fresh Clone
Uses same config but starts with blank memory and patterns

## See Also

- `/agent-spawn` - Spawn new agents
- `/agent-benchmark` - Benchmark performance
- `/agent-retire` - Retire agents
- `/neural-train` - Train neural patterns

---

**Version**: 2.0.0
