---
name: coordination
description: Advanced multi-agent coordination patterns including mesh, hierarchical, and adaptive topologies with state synchronization, consensus mechanisms, and distributed task execution. Use when orchestrating complex multi-agent workflows requiring sophisticated coordination, consensus building, or distributed decision-making.
version: 2.0.0
tier: gold
tags:
  - coordination
  - multi-agent
  - orchestration
  - topology
  - consensus
  - distributed
category: orchestration
agents:
  - hierarchical-coordinator
  - mesh-coordinator
  - adaptive-coordinator
  - collective-intelligence-coordinator
  - consensus-validator
complexity: advanced
estimated_duration: 30-120 minutes
prerequisites:
  - Claude Flow MCP installed
  - Understanding of distributed systems
  - Multi-agent architecture knowledge
outputs:
  - Coordination topologies
  - Agent network configurations
  - Consensus protocols
  - Distributed execution plans
---

# Agent Coordination Skill

## Overview

Advanced multi-agent coordination system implementing sophisticated topologies (mesh, hierarchical, ring, star), consensus mechanisms (Byzantine, Raft, gossip), and distributed task execution patterns. This skill enables orchestration of complex workflows requiring multiple agents to collaborate, synchronize state, reach consensus, and execute tasks in a coordinated manner.

## When to Use This Skill

Activate this skill when:
- **Complex Multi-Agent Workflows**: Need to coordinate 3+ agents with interdependencies
- **Distributed Decision-Making**: Require consensus among multiple agents
- **Sophisticated Topologies**: Need mesh, hierarchical, or adaptive network structures
- **State Synchronization**: Agents must share and synchronize state
- **Fault-Tolerant Execution**: Require Byzantine fault tolerance or graceful degradation
- **Dynamic Scaling**: Agent count or topology changes during execution
- **Performance Optimization**: Need optimal agent allocation and load balancing

## Core Coordination Patterns

### 1. Topology Patterns

#### Mesh Topology
**When:** All agents need to communicate directly with each other
**Characteristics:**
- Peer-to-peer communication
- No single point of failure
- High redundancy
- Complex coordination overhead
- Best for: Small agent counts (3-8), consensus building

**Script:** `resources/scripts/init-mesh-topology.js`
**Template:** `resources/templates/mesh-topology.yaml`

#### Hierarchical Topology
**When:** Clear command structure with parent-child relationships
**Characteristics:**
- Tree-like structure
- Clear authority hierarchy
- Efficient delegation
- Centralized coordination
- Best for: Large agent counts (8+), task delegation

**Script:** `resources/scripts/init-hierarchical-topology.js`
**Template:** `resources/templates/hierarchical-topology.yaml`

#### Adaptive Topology
**When:** Topology needs to evolve based on task requirements
**Characteristics:**
- Dynamic reconfiguration
- Self-organizing structure
- Performance-driven optimization
- Complexity management
- Best for: Variable workloads, unpredictable requirements

**Script:** `resources/scripts/init-adaptive-topology.js`
**Template:** `resources/templates/adaptive-topology.yaml`

### 2. Consensus Mechanisms

#### Byzantine Consensus
**When:** Need fault tolerance against malicious or faulty agents
**Characteristics:**
- 2f+1 agents required for f faults
- Cryptographic verification
- High security guarantees
- Moderate performance overhead
- Best for: Critical decisions, security-sensitive operations

**Script:** `resources/scripts/byzantine-consensus.js`

#### Raft Consensus
**When:** Need leader-based consensus with strong consistency
**Characteristics:**
- Leader election
- Log replication
- Strong consistency
- Simple mental model
- Best for: State machine replication, distributed logs

**Script:** `resources/scripts/raft-consensus.js`

### 3. Execution Patterns

#### Parallel Execution
**When:** Tasks are independent and can run simultaneously
**Benefits:** Maximum throughput, reduced latency
**Trade-offs:** Higher resource usage, complex coordination

#### Sequential Execution
**When:** Tasks have strict dependencies
**Benefits:** Simple coordination, deterministic order
**Trade-offs:** Lower throughput, longer total time

#### Adaptive Execution
**When:** Mix of dependent and independent tasks
**Benefits:** Balanced throughput and coordination
**Trade-offs:** Complex orchestration logic

## Implementation Workflow

### Phase 1: Topology Selection & Initialization

**1.1 Analyze Requirements**
```bash
# Use topology selector script
node resources/scripts/topology-selector.js \
  --agents 6 \
  --task-type distributed \
  --fault-tolerance high \
  --output topology-recommendation.json
```

**1.2 Initialize Topology**
```bash
# Initialize selected topology (example: mesh)
node resources/scripts/init-mesh-topology.js \
  --max-agents 6 \
  --strategy balanced \
  --config topology-config.yaml
```

**Key Considerations:**
- Agent count: <8 → mesh, >8 → hierarchical
- Fault tolerance: High → Byzantine, Medium → Raft, Low → simple consensus
- Communication overhead: Minimize for large swarms
- Task dependencies: Strong → hierarchical, Weak → mesh

### Phase 2: Agent Spawning & Network Formation

**2.1 Spawn Specialized Agents**
```bash
# Spawn agents with role assignments
node resources/scripts/spawn-coordinated-agents.js \
  --topology mesh \
  --roles researcher,coder,analyst,optimizer \
  --config agent-roles.yaml
```

**2.2 Establish Communication Channels**
- Mesh: All-to-all connections
- Hierarchical: Parent-child links
- Ring: Circular connections
- Star: Hub-and-spoke

**2.3 Initialize Shared State**
```bash
# Initialize distributed state store
npx claude-flow@alpha memory store \
  --key "coordination/shared-state" \
  --value '{
    "topology": "mesh",
    "agent_count": 6,
    "consensus": "byzantine",
    "session_id": "coord-2024-001"
  }'
```

### Phase 3: Task Orchestration

**3.1 Task Decomposition**
Break complex tasks into agent-specific subtasks:
- Analyze dependencies between subtasks
- Identify parallel vs. sequential execution opportunities
- Assign subtasks to specialized agents
- Define success criteria for each subtask

**3.2 Distributed Task Assignment**
```bash
# Orchestrate task across agents
node resources/scripts/orchestrate-distributed-task.js \
  --task "Full-stack feature development" \
  --strategy adaptive \
  --priority high \
  --agents researcher,coder,tester,reviewer
```

**3.3 Execution Monitoring**
```bash
# Monitor distributed execution
npx claude-flow@alpha swarm_monitor \
  --duration 60 \
  --interval 5 \
  --output execution-metrics.json
```

### Phase 4: Consensus & Decision Making

**4.1 Initiate Consensus Protocol**
For critical decisions requiring agreement among agents:

```bash
# Run Byzantine consensus
node resources/scripts/byzantine-consensus.js \
  --proposal "Deploy to production" \
  --agents agent-1,agent-2,agent-3,agent-4,agent-5 \
  --threshold 0.67 \
  --timeout 30
```

**4.2 Validate Consensus**
- Verify quorum reached (e.g., 2f+1 for Byzantine)
- Check signature validity
- Confirm majority agreement
- Handle dissenting opinions

**4.3 Execute Consensus Decision**
Once consensus achieved, coordinate execution across agents.

### Phase 5: State Synchronization

**5.1 Continuous State Sync**
```bash
# Synchronize state across agents
node resources/scripts/sync-agent-state.js \
  --topology mesh \
  --interval 10 \
  --conflict-resolution last-write-wins
```

**5.2 Conflict Resolution**
- **Last-Write-Wins**: Simple, eventual consistency
- **Vector Clocks**: Causal consistency
- **CRDT**: Conflict-free replicated data types
- **Consensus-Based**: Strong consistency, higher latency

### Phase 6: Performance Optimization

**6.1 Analyze Bottlenecks**
```bash
# Identify coordination bottlenecks
npx claude-flow@alpha agent_metrics \
  --metric all \
  --output performance-analysis.json
```

**6.2 Optimize Topology**
- Rebalance agent load
- Adjust communication patterns
- Scale agent count
- Switch topology if needed

**6.3 Measure Improvements**
Track key metrics:
- Task completion time
- Agent utilization
- Communication overhead
- Consensus latency
- Error rates

## Advanced Features

### Dynamic Reconfiguration

Adapt topology during execution:
```bash
# Reconfigure topology on-the-fly
node resources/scripts/reconfigure-topology.js \
  --from mesh \
  --to hierarchical \
  --preserve-state true \
  --migration-strategy gradual
```

### Fault Detection & Recovery

Implement automatic fault handling:
```bash
# Enable fault detection
node resources/scripts/enable-fault-detection.js \
  --heartbeat-interval 5 \
  --failure-threshold 3 \
  --recovery-strategy spawn-replacement
```

### Load Balancing

Distribute work evenly:
```bash
# Enable dynamic load balancing
node resources/scripts/load-balancer.js \
  --strategy adaptive \
  --rebalance-interval 30 \
  --target-utilization 0.75
```

## Success Criteria

- [ ] Appropriate topology selected and initialized
- [ ] Agents spawned with correct role assignments
- [ ] Communication channels established
- [ ] Tasks distributed and executed
- [ ] Consensus reached on critical decisions
- [ ] State synchronized across agents
- [ ] Performance metrics within targets
- [ ] Fault tolerance validated

### Performance Targets

- **Agent Spawning**: <5 seconds per agent
- **Task Assignment**: <2 seconds
- **Consensus Latency**: <10 seconds (Byzantine), <5 seconds (Raft)
- **State Sync Frequency**: Every 5-10 seconds
- **Agent Utilization**: 70-85%
- **Task Success Rate**: >95%

## Best Practices

1. **Topology Selection**: Match topology to task characteristics
2. **Consensus Usage**: Only for critical decisions (overhead cost)
3. **State Management**: Use shared memory for coordination state
4. **Error Handling**: Implement retry logic and graceful degradation
5. **Monitoring**: Continuous performance tracking
6. **Scaling**: Add agents incrementally, test at each scale
7. **Testing**: Validate with chaos engineering (agent failures, network partitions)
8. **Documentation**: Record topology decisions and rationale

## Common Patterns & Solutions

### Pattern: Research → Implementation → Validation
**Topology**: Hierarchical (coordinator → specialist agents)
**Execution**: Sequential with parallel validation
**Consensus**: Simple majority vote on validation

### Pattern: Distributed Code Review
**Topology**: Mesh (peer review among 4-6 agents)
**Execution**: Parallel analysis, consensus on approval
**Consensus**: Byzantine (2f+1 security checks)

### Pattern: Multi-Model Inference
**Topology**: Star (hub routes requests to model agents)
**Execution**: Parallel inference, ensemble aggregation
**Consensus**: Weighted voting based on confidence

## Troubleshooting

### Issue: Consensus Timeout
**Symptoms**: Consensus protocol fails to reach agreement
**Solutions:**
- Increase timeout threshold
- Reduce agent count
- Check network connectivity
- Verify agent health

### Issue: State Desynchronization
**Symptoms**: Agents have conflicting state
**Solutions:**
- Increase sync frequency
- Use stronger consistency (CRDT, consensus)
- Implement conflict resolution
- Reset shared state

### Issue: Performance Degradation
**Symptoms**: Slow task completion, high latency
**Solutions:**
- Analyze bottlenecks (use metrics)
- Optimize topology (fewer connections)
- Reduce consensus overhead
- Scale horizontally (more agents)

## Integration Points

- **when-chaining-workflows-use-cascade-orchestrator**: For workflow cascades
- **when-coordinating-collective-intelligence-use-hive-mind**: For hierarchical queen-led coordination
- **swarm-orchestration**: For basic swarm operations
- **performance-analysis**: For bottleneck detection

## Resources

### Scripts (resources/scripts/)
- `topology-selector.js` - Recommend optimal topology
- `init-mesh-topology.js` - Initialize mesh coordination
- `init-hierarchical-topology.js` - Initialize hierarchical coordination
- `init-adaptive-topology.js` - Initialize adaptive coordination
- `spawn-coordinated-agents.js` - Spawn agents with roles
- `orchestrate-distributed-task.js` - Distribute tasks
- `byzantine-consensus.js` - Run Byzantine consensus
- `raft-consensus.js` - Run Raft consensus
- `sync-agent-state.js` - Synchronize agent state
- `reconfigure-topology.js` - Dynamic reconfiguration
- `enable-fault-detection.js` - Fault detection & recovery
- `load-balancer.js` - Dynamic load balancing

### Templates (resources/templates/)
- `mesh-topology.yaml` - Mesh coordination config
- `hierarchical-topology.yaml` - Hierarchical coordination config
- `adaptive-topology.yaml` - Adaptive coordination config
- `agent-roles.yaml` - Agent role definitions

### Tests (tests/)
- `test-mesh-topology.js` - Mesh coordination tests
- `test-hierarchical-topology.js` - Hierarchical coordination tests
- `test-consensus-mechanisms.js` - Consensus protocol tests

## References

- Distributed Systems: Concepts and Design (Coulouris et al.)
- Byzantine Fault Tolerance in Practice (Castro & Liskov)
- Raft Consensus Algorithm (Ongaro & Ousterhout)
- Multi-Agent Systems: A Modern Approach (Wooldridge)
