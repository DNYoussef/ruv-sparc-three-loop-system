# Test Case 3: Multi-Topology Integration with Consensus Protocols

## Test Metadata
- **Test ID**: ADV-COORD-003
- **Category**: Integration & Protocol Verification
- **Topologies**: Mesh → Hierarchical → Ring (progressive)
- **Duration**: ~10 minutes

## Scenario
Test integration between multiple coordination topologies and consensus protocols in a single workflow.
Start with mesh for research, transition to hierarchical for implementation, then ring for testing.

## Prerequisites
- Claude Flow installed
- All MCP servers configured
- Memory MCP for cross-topology context persistence
- No active swarms

## Input Configuration

**Multi-Phase Workflow**:

### Phase 1: Research Phase (Mesh + Gossip)
```yaml
topology: mesh
maxAgents: 7
strategy: balanced
consensus: gossip

task: |
  Research distributed consensus algorithms.
  Focus on: Raft, Paxos, Byzantine, Gossip
```

### Phase 2: Implementation Phase (Hierarchical + Raft)
```yaml
topology: hierarchical
maxAgents: 15
strategy: specialized
consensus: raft

task: |
  Implement consensus protocol library based on research findings.
  Coordinator: Architecture design
  Managers: Protocol implementations (Raft, Byzantine, Gossip)
  Workers: Unit tests, integration tests, benchmarks
```

### Phase 3: Testing Phase (Ring + Byzantine)
```yaml
topology: ring
maxAgents: 9
strategy: adaptive
consensus: byzantine

task: |
  Run comprehensive testing of consensus library.
  Each agent tests one protocol in sequence, passing results to next.
```

## Expected Behavior

### Phase 1: Mesh Research (0-3 minutes)
1. 7 agents spawn in mesh topology
2. Gossip protocol shares research findings across all agents
3. Eventually consistent knowledge graph built
4. Results stored in Memory MCP with `layer: short-term`

**Key Metrics**:
- Gossip convergence time: <30 seconds
- Knowledge coverage: 100% (all agents aware of all findings)
- Consensus: eventual (no strict agreement required)

### Phase 2: Hierarchical Implementation (3-8 minutes)
1. Mesh swarm destroyed, memory persisted
2. 15 agents spawn in hierarchical topology (1+3+11)
3. Coordinator retrieves research from Memory MCP
4. Task decomposed into 3 sub-tasks (one per consensus protocol)
5. Managers coordinate implementation with workers
6. Raft consensus used for code review approvals (majority vote)

**Key Metrics**:
- Task decomposition time: <10 seconds
- Memory retrieval latency: <200ms
- Raft leader elections: 3 (one per manager layer)
- Code approval consensus: ≥50% (Raft majority)

### Phase 3: Ring Testing (8-10 minutes)
1. Hierarchical swarm destroyed, artifacts preserved
2. 9 agents spawn in ring topology
3. Byzantine consensus validates test results
4. Sequential execution around ring (agent N → agent N+1)
5. Byzantine fault injection to verify library robustness

**Key Metrics**:
- Ring traversal time: <60 seconds per cycle
- Byzantine fault detection: <10 seconds
- Test coverage: ≥90%
- Integration test pass rate: 100%

## Expected Output

### Phase 1: Research Results
```json
{
  "phase": "research",
  "topology": "mesh",
  "consensus": "gossip",
  "agents": 7,
  "findings": {
    "raft": {
      "complexity": "O(log n)",
      "faultTolerance": "crash failures",
      "sources": ["agent-1", "agent-3", "agent-5"]
    },
    "byzantine": {
      "complexity": "O(n²)",
      "faultTolerance": "malicious agents",
      "sources": ["agent-2", "agent-4", "agent-6"]
    },
    "gossip": {
      "complexity": "O(log log n)",
      "faultTolerance": "eventual consistency",
      "sources": ["agent-7", "agent-1"]
    }
  },
  "gossipConvergence": {
    "rounds": 4,
    "timeSeconds": 23.5,
    "finalCoverage": 1.0
  },
  "memoryStored": {
    "key": "swarm/research/consensus-algorithms",
    "layer": "short-term",
    "retention": "24h"
  }
}
```

### Phase 2: Implementation Results
```json
{
  "phase": "implementation",
  "topology": "hierarchical",
  "consensus": "raft",
  "agents": 15,
  "contextRestored": {
    "key": "swarm/research/consensus-algorithms",
    "retrievalTimeMs": 187
  },
  "taskDecomposition": {
    "coordinator": "Architecture design",
    "managers": [
      {
        "id": "manager-1",
        "task": "Raft implementation",
        "workers": ["worker-1", "worker-2", "worker-3", "worker-4"]
      },
      {
        "id": "manager-2",
        "task": "Byzantine implementation",
        "workers": ["worker-5", "worker-6", "worker-7", "worker-8"]
      },
      {
        "id": "manager-3",
        "task": "Gossip implementation",
        "workers": ["worker-9", "worker-10", "worker-11"]
      }
    ]
  },
  "raftConsensus": {
    "leaderElections": 3,
    "codeApprovals": 12,
    "approvalRate": 0.92,
    "majorityThreshold": 0.5
  },
  "artifacts": {
    "filesCreated": 27,
    "linesOfCode": 3542,
    "testCoverage": 0.94
  }
}
```

### Phase 3: Testing Results
```json
{
  "phase": "testing",
  "topology": "ring",
  "consensus": "byzantine",
  "agents": 9,
  "ringSequence": [
    "agent-1 → agent-2 → agent-3 → agent-4 → agent-5 → agent-6 → agent-7 → agent-8 → agent-9 → agent-1"
  ],
  "testExecution": {
    "totalTests": 147,
    "passed": 147,
    "failed": 0,
    "coverage": 0.93
  },
  "byzantineValidation": {
    "faultsInjected": 2,
    "faultsDetected": 2,
    "detectionTimeAvg": 8.3,
    "consensusThreshold": 0.67,
    "consensusAchieved": true
  },
  "ringTraversal": {
    "cycles": 3,
    "avgCycleTime": 52.4,
    "allAgentsParticipated": true
  }
}
```

### Cross-Topology Integration Summary
```json
{
  "workflow": "multi-topology-consensus-library",
  "totalDuration": 587,
  "phases": 3,
  "topologies": ["mesh", "hierarchical", "ring"],
  "consensusProtocols": ["gossip", "raft", "byzantine"],
  "totalAgents": 31,
  "contextTransitions": 2,
  "memoryPersistence": {
    "writes": 3,
    "reads": 2,
    "latencyAvgMs": 193
  },
  "finalArtifacts": {
    "consensusLibrary": "consensus-lib-v1.0.0.tar.gz",
    "documentation": "consensus-algorithms.md",
    "benchmarks": "performance-results.json"
  },
  "integrationSuccess": true
}
```

## Validation Steps

### 1. Phase 1 Validation (Mesh Research)
```bash
# Deploy mesh swarm
bash resources/scripts/deploy_swarm.sh --config test-configs/phase1-mesh.yaml

# Monitor gossip convergence
npx claude-flow@alpha swarm monitor --metric gossip-coverage --interval 5

# Verify memory persistence
npx claude-flow@alpha memory retrieve --key "swarm/research/consensus-algorithms"

# Expected: Knowledge graph with 100% coverage across 7 agents
```

### 2. Phase 1 → Phase 2 Transition
```bash
# Destroy mesh swarm (preserves memory)
npx claude-flow@alpha swarm destroy --preserve-memory

# Deploy hierarchical swarm
bash resources/scripts/deploy_swarm.sh --config test-configs/phase2-hierarchical.yaml

# Verify context restoration
npx claude-flow@alpha memory retrieve --key "swarm/research/consensus-algorithms"

# Expected: Research findings loaded, <200ms latency
```

### 3. Phase 2 Validation (Hierarchical Implementation)
```bash
# Monitor Raft leader elections
npx claude-flow@alpha swarm monitor --metric raft-elections

# Track code approval consensus
npx claude-flow@alpha task status --include-approvals

# Expected: 3 leader elections, ≥92% approval rate
```

### 4. Phase 2 → Phase 3 Transition
```bash
# Destroy hierarchical swarm (preserves artifacts)
npx claude-flow@alpha swarm destroy --preserve-artifacts

# Deploy ring swarm
bash resources/scripts/deploy_swarm.sh --config test-configs/phase3-ring.yaml
```

### 5. Phase 3 Validation (Ring Testing)
```bash
# Monitor ring traversal
npx claude-flow@alpha swarm monitor --metric ring-progress --interval 10

# Inject Byzantine faults
npx claude-flow@alpha test inject-fault --type byzantine --count 2

# Verify Byzantine detection
npx claude-flow@alpha metrics analyze --focus byzantine-detection

# Expected: All faults detected <10 seconds, consensus maintained
```

### 6. End-to-End Integration Validation
```bash
# Generate integration report
npx claude-flow@alpha workflow report --workflow consensus-library-development

# Verify all phases completed
npx claude-flow@alpha memory retrieve --key "swarm/workflow/integration-summary"

# Expected: 3 phases completed, 2 successful transitions, 100% test pass rate
```

## Acceptance Criteria

- [ ] **Phase 1 (Mesh)**: Gossip converges <30s, 100% knowledge coverage
- [ ] **Phase 1→2 Transition**: Memory persisted and restored <200ms
- [ ] **Phase 2 (Hierarchical)**: Raft consensus with 3 leader elections, ≥90% approval rate
- [ ] **Phase 2 Artifacts**: 25+ files created, ≥90% test coverage
- [ ] **Phase 2→3 Transition**: Artifacts preserved and loaded
- [ ] **Phase 3 (Ring)**: Byzantine consensus validates test results, faults detected <10s
- [ ] **Phase 3 Testing**: 100% test pass rate, ≥90% coverage
- [ ] **End-to-End**: All 3 phases complete, 2 successful transitions, consensus library functional

## Known Issues & Limitations

- **Context size growth**: Each phase adds to memory context (monitor token usage)
- **Transition latency**: Swarm destroy → create takes 5-10 seconds
- **Topology overhead**: Each topology has different connection complexity (mesh O(n²), ring O(n))
- **Consensus switching**: Different protocols have different guarantees (gossip eventual, Raft/Byzantine strong)

## Advanced Integration Scenarios

### Scenario A: Fault During Transition
- Inject fault during Phase 1→2 transition
- Verify memory rollback and re-initialization
- Expected: Transition retries successfully, no data loss

### Scenario B: Consensus Protocol Conflict
- Use Byzantine consensus in Phase 2 instead of Raft
- Verify hierarchical topology adapts to stronger consensus
- Expected: Performance degradation but correctness maintained

### Scenario C: Memory Corruption
- Corrupt memory between Phase 2→3
- Verify fallback to artifact-based recovery
- Expected: Phase 3 loads from artifacts instead of memory

## Cleanup
```bash
# Destroy all swarms and clean up
npx claude-flow@alpha swarm destroy --all
npx claude-flow@alpha memory clear --pattern "swarm/*/consensus-*"
npx claude-flow@alpha artifacts clean --older-than 1h
```
