---
name: coordination-visualize
description: Visualize swarm coordination topology and agent interactions
version: 2.0.0
category: monitoring
complexity: medium
tags: [visualization, coordination, topology, agents, monitoring, analysis]
author: ruv-SPARC Monitoring Team
created: 2025-11-01
last_updated: 2025-11-01
dependencies: [swarm-status, agent-list, task-status]
chains_with: [swarm-monitor, agent-metrics, performance-analysis]
evidence_based_techniques: [self-consistency]
---

# /coordination-visualize - Swarm Topology Visualization

## Overview

The `/coordination-visualize` command generates visual representations of swarm coordination topologies, agent interactions, and communication patterns.

## Usage

```bash
# Visualize current swarm topology
npx claude-flow@alpha coordination visualize

# Generate topology diagram
npx claude-flow@alpha coordination visualize --output "topology.svg"

# Real-time visualization
npx claude-flow@alpha coordination visualize --live

# Visualize specific swarm
npx claude-flow@alpha coordination visualize --swarm-id "swarm-123"

# Show agent interactions
npx claude-flow@alpha coordination visualize --show-interactions

# Export to different formats
npx claude-flow@alpha coordination visualize \
  --output "topology.png" \
  --format png
```

## Parameters

- `--swarm-id <id>` - Specific swarm to visualize
- `--output <path>` - Output file path
- `--format <format>` - Output format: `svg`, `png`, `pdf`, `ascii`, `json`
- `--live` - Real-time visualization (web interface)
- `--show-interactions` - Show agent communication patterns
- `--show-tasks` - Show task assignments
- `--show-metrics` - Include performance metrics
- `--layout <layout>` - Layout algorithm: `hierarchical`, `force`, `circular`

## Visualization Types

### Topology Diagram
Shows:
- Agents as nodes
- Coordination links as edges
- Hierarchy levels (for hierarchical topology)
- Communication patterns

### Interaction Graph
Shows:
- Message flow between agents
- Frequency of interactions
- Memory sharing patterns
- Task handoffs

### Performance Heatmap
Shows:
- Agent load (color-coded)
- Bottlenecks (highlighted)
- Idle agents (grayed)
- Active tasks (animated)

### Timeline View
Shows:
- Agent activity over time
- Task execution timeline
- Coordination events
- Performance trends

## Example Output (ASCII)

```bash
npx claude-flow@alpha coordination visualize --format ascii

# Output:
#                    SWARM TOPOLOGY: mesh
#
#          ┌─────────────────────────────────┐
#          │    Coordinator (Queen)          │
#          │    Tasks: 5 | Load: 67%         │
#          └─────────────┬───────────────────┘
#                        │
#        ┌───────────────┼───────────────┐
#        │               │               │
#   ┌────▼────┐     ┌────▼────┐     ┌───▼─────┐
#   │ Coder-1 │◄───►│ Coder-2 │◄───►│ Coder-3 │
#   │ Load: 82%│     │ Load: 45%│     │ Load: 91%│
#   └────┬────┘     └────┬────┘     └───┬─────┘
#        │               │               │
#        └───────┬───────┴───────┬───────┘
#                │               │
#           ┌────▼────┐     ┌────▼────┐
#           │ Tester-1│     │ Tester-2│
#           │ Load: 56%│     │ Load: 73%│
#           └─────────┘     └─────────┘
#
# Legend:
#   ◄──► : Bidirectional communication
#   ──►  : Unidirectional task flow
#   Load : Current task load %
```

## Web Interface (Live Mode)

```bash
npx claude-flow@alpha coordination visualize --live

# Output:
# [Coordination Visualize] Starting live visualization server...
# [Coordination Visualize] Server running at http://localhost:3000
# [Coordination Visualize] Open in browser to view real-time topology
#
# Features:
#   • Drag and drop nodes
#   • Click nodes for details
#   • Filter by agent type
#   • Timeline scrubber
#   • Export snapshots
```

## See Also

- `/swarm-status` - Swarm status
- `/agent-list` - List agents
- `/swarm-monitor` - Monitor activity
- `/performance-analysis` - Performance metrics

---

**Version**: 2.0.0
