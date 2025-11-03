# Coordination Skill - Gold Tier

**Version:** 2.0.0
**Tier:** Gold
**Files:** 13+
**Category:** Multi-Agent Orchestration

## Overview

Advanced multi-agent coordination system implementing sophisticated topologies (mesh, hierarchical, adaptive), consensus mechanisms (Byzantine, Raft), and distributed task execution patterns.

## Skill Tier: Gold

This skill has been enhanced to **Gold tier** with:
- ✅ Comprehensive skill documentation (`skill.md`)
- ✅ 5 automation scripts in `resources/scripts/`
- ✅ 4 YAML templates in `resources/templates/`
- ✅ 3 comprehensive test suites in `tests/`
- ✅ 2+ specialized sub-skills (cascade-orchestrator, hive-mind)

## Directory Structure

```
coordination/
├── skill.md                                    # Main skill documentation
├── README.md                                   # This file
│
├── when-chaining-workflows-use-cascade-orchestrator/
│   ├── SKILL.md                               # Cascade workflow orchestration
│   ├── PROCESS.md
│   ├── README.md
│   └── process-diagram.gv
│
├── when-coordinating-collective-intelligence-use-hive-mind/
│   ├── SKILL.md                               # Hive mind coordination
│   ├── PROCESS.md
│   ├── README.md
│   └── process-diagram.gv
│
├── resources/
│   ├── scripts/                               # Automation scripts
│   │   ├── topology-selector.js              # Recommend optimal topology
│   │   ├── init-mesh-topology.js             # Initialize mesh coordination
│   │   ├── init-hierarchical-topology.js     # Initialize hierarchical coordination
│   │   ├── spawn-coordinated-agents.js       # Spawn agents with roles
│   │   └── orchestrate-distributed-task.js   # Distribute tasks across agents
│   │
│   └── templates/                             # Configuration templates
│       ├── mesh-topology.yaml                # Mesh coordination config
│       ├── hierarchical-topology.yaml        # Hierarchical coordination config
│       ├── adaptive-topology.yaml            # Adaptive coordination config
│       └── agent-roles.yaml                  # Agent role definitions
│
└── tests/                                     # Comprehensive test suites
    ├── test-mesh-topology.js                 # Mesh topology tests (9 tests)
    ├── test-hierarchical-topology.js         # Hierarchical topology tests (10 tests)
    └── test-consensus-mechanisms.js          # Consensus protocol tests (10 tests)
```

## Quick Start

### 1. Select Optimal Topology

```bash
node resources/scripts/topology-selector.js \
  --agents 6 \
  --task-type distributed \
  --fault-tolerance high \
  --output topology-recommendation.json
```

### 2. Initialize Topology

**Mesh (3-8 agents, consensus-heavy):**
```bash
node resources/scripts/init-mesh-topology.js \
  --max-agents 6 \
  --strategy balanced \
  --config topology-recommendation.json
```

**Hierarchical (8+ agents, delegation-heavy):**
```bash
node resources/scripts/init-hierarchical-topology.js \
  --max-agents 15 \
  --levels 3 \
  --config topology-recommendation.json
```

### 3. Spawn Coordinated Agents

```bash
node resources/scripts/spawn-coordinated-agents.js \
  --topology mesh \
  --roles researcher,coder,analyst,optimizer \
  --config resources/templates/agent-roles.yaml
```

### 4. Orchestrate Distributed Task

```bash
node resources/scripts/orchestrate-distributed-task.js \
  --task "Build full-stack feature with authentication" \
  --strategy adaptive \
  --priority high
```

### 5. Monitor Execution

```bash
npx claude-flow@alpha swarm_monitor --interval 10
npx claude-flow@alpha task_status --detailed
```

## Coordination Patterns

### Mesh Topology
**Use when:** 3-8 agents, consensus-heavy, high fault tolerance
**Characteristics:** All-to-all communication, Byzantine consensus
**Template:** `resources/templates/mesh-topology.yaml`

### Hierarchical Topology
**Use when:** 8+ agents, delegation-heavy, clear command structure
**Characteristics:** Tree-like, parent-child delegation
**Template:** `resources/templates/hierarchical-topology.yaml`

### Adaptive Topology
**Use when:** Variable workloads, unpredictable requirements
**Characteristics:** Dynamic reconfiguration, self-organizing
**Template:** `resources/templates/adaptive-topology.yaml`

## Consensus Mechanisms

### Byzantine Consensus
**Use when:** Critical decisions, security-sensitive, potential malicious agents
**Quorum:** 2f+1 (where f = max faults)
**Latency:** Higher (~30ms) due to cryptographic verification
**Script:** `resources/scripts/byzantine-consensus.js`

### Raft Consensus
**Use when:** State machine replication, distributed logs
**Quorum:** Majority (n/2 + 1)
**Latency:** Lower (~15ms) with leader-based approach
**Script:** `resources/scripts/raft-consensus.js`

## Agent Roles

Defined in `resources/templates/agent-roles.yaml`:

| Role | Type | Category | Capabilities |
|------|------|----------|-------------|
| **researcher** | researcher | analysis | Requirements gathering, pattern research, best practices |
| **coder** | coder | implementation | Feature implementation, bug fixing, refactoring |
| **analyst** | analyst | validation | Code review, testing, quality assurance |
| **tester** | analyst | validation | Unit/integration/e2e testing, coverage analysis |
| **optimizer** | optimizer | optimization | Performance tuning, bottleneck detection, benchmarking |
| **coordinator** | coordinator | orchestration | Task delegation, progress monitoring, conflict resolution |
| **architect** | researcher | design | System design, architecture patterns, API design |
| **reviewer** | analyst | validation | Code review, documentation review, quality gates |

## Testing

### Run All Tests

```bash
# Mesh topology tests (9 tests)
node tests/test-mesh-topology.js

# Hierarchical topology tests (10 tests)
node tests/test-hierarchical-topology.js

# Consensus mechanism tests (10 tests)
node tests/test-consensus-mechanisms.js
```

### Test Coverage

- **Mesh Topology**: Initialization, agent spawning, all-to-all communication, task orchestration, Byzantine consensus, state sync, fault tolerance, metrics, cleanup
- **Hierarchical Topology**: Initialization, coordinator/team-lead/worker spawning, hierarchy structure, delegation, reporting, failover, failure recovery, metrics
- **Consensus Mechanisms**: Byzantine quorum/voting/fault-tolerance/signatures, Raft leader-election/log-replication/commit/terms, latency comparison, use-case selection

## Performance Targets

| Metric | Target | Topology |
|--------|--------|----------|
| Agent Spawning | <5s per agent | All |
| Task Assignment | <2s | All |
| Consensus Latency (Byzantine) | <10s | Mesh |
| Consensus Latency (Raft) | <5s | Hierarchical |
| State Sync Frequency | 5-10s | All |
| Agent Utilization | 70-85% | All |
| Task Success Rate | >95% | All |

## Sub-Skills

### 1. Cascade Orchestrator
**Location:** `when-chaining-workflows-use-cascade-orchestrator/`
**Purpose:** Chain multiple workflows with sequential/parallel/conditional execution
**Use when:** Complex workflow pipelines, multi-stage processes

### 2. Hive Mind Coordination
**Location:** `when-coordinating-collective-intelligence-use-hive-mind/`
**Purpose:** Queen-led hierarchical coordination with collective intelligence
**Use when:** Complex decision-making, emergent behavior, swarm intelligence

## Integration Points

- **Claude Flow MCP**: Core orchestration (`swarm_init`, `agent_spawn`, `task_orchestrate`)
- **Memory MCP**: State synchronization, persistent coordination state
- **Connascence Analyzer**: Code quality checks for coordination implementations
- **Performance Analysis**: Bottleneck detection, optimization recommendations

## Best Practices

1. **Topology Selection**: Match to task characteristics (agent count, fault tolerance, communication overhead)
2. **Consensus Usage**: Reserve for critical decisions (overhead cost)
3. **State Management**: Use shared memory for coordination state
4. **Error Handling**: Implement retry logic and graceful degradation
5. **Monitoring**: Continuous performance tracking via metrics
6. **Scaling**: Add agents incrementally, test at each scale
7. **Testing**: Validate with chaos engineering (agent failures, network partitions)

## Troubleshooting

### Consensus Timeout
**Symptoms:** Consensus fails to reach agreement
**Solutions:** Increase timeout, reduce agent count, check network connectivity, verify agent health

### State Desynchronization
**Symptoms:** Agents have conflicting state
**Solutions:** Increase sync frequency, use stronger consistency (CRDT), implement conflict resolution, reset shared state

### Performance Degradation
**Symptoms:** Slow task completion, high latency
**Solutions:** Analyze bottlenecks, optimize topology, reduce consensus overhead, scale horizontally

## Contributing

To enhance this Gold tier skill:
1. Add new automation scripts to `resources/scripts/`
2. Create additional templates in `resources/templates/`
3. Expand test coverage in `tests/`
4. Document new patterns in `skill.md`

## References

- Distributed Systems: Concepts and Design (Coulouris et al.)
- Byzantine Fault Tolerance in Practice (Castro & Liskov)
- Raft Consensus Algorithm (Ongaro & Ousterhout)
- Multi-Agent Systems: A Modern Approach (Wooldridge)

---

**Skill Tier Achieved:** 🥇 Gold (13+ files, scripts, templates, comprehensive tests)
