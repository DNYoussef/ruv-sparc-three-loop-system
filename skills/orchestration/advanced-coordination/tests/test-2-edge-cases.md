# Test Case 2: Hierarchical Topology with Byzantine Fault Tolerance

## Test Metadata
- **Test ID**: ADV-COORD-002
- **Category**: Edge Cases & Fault Tolerance
- **Topology**: Hierarchical
- **Duration**: ~5 minutes

## Scenario
Deploy a 3-level hierarchical swarm with 13 agents (1 coordinator + 3 managers + 9 workers).
Simulate Byzantine failures (malicious/faulty agents) and verify fault tolerance mechanisms.

## Prerequisites
- Claude Flow installed with fault injection support
- MCP servers configured
- Advanced coordination skill loaded

## Input Configuration

**Topology File**: `resources/templates/hierarchical-topology.yaml`

```yaml
topology: hierarchical
maxAgents: 13
strategy: specialized
consensus: raft

hierarchy:
  levels: 3
  branchingFactor: 3
  coordinator:
    type: hierarchical-coordinator
  managers:
    count: 3
  workers:
    types: [researcher, coder, analyst]
```

**Task**:
```
Implement a distributed logging system with the following requirements:
1. Central log aggregator
2. Log rotation and archival
3. Real-time log streaming
4. Query interface
Each layer should handle specific responsibilities.
```

## Expected Behavior

### Phase 1: Normal Operation
1. Hierarchical structure established (1 → 3 → 9 agents)
2. Task decomposed by coordinator into 3 sub-tasks
3. Managers delegate to 3 workers each
4. Results bubble up through hierarchy

### Phase 2: Fault Injection
1. **Worker failure**: Worker-5 becomes unresponsive
   - Manager-2 detects failure via health check
   - Manager-2 redistributes work to Worker-4 and Worker-6
   - Task completion continues without coordinator intervention

2. **Manager failure**: Manager-3 crashes
   - Coordinator detects failure (timeout after 300s)
   - Coordinator reassigns Worker-7, Worker-8, Worker-9 to Manager-1 and Manager-2
   - Raft leader election ensures consistency

3. **Byzantine agent**: Worker-3 sends conflicting results
   - Manager-1 detects inconsistency
   - Manager-1 requests re-execution from Worker-3
   - If inconsistency persists, escalates to coordinator
   - Coordinator marks Worker-3 as faulty, spawns replacement

### Phase 3: Recovery
1. System self-heals by spawning replacement agents
2. Task completes successfully despite 3 failures
3. Final results validated by coordinator

## Expected Output

### Initial Topology
```
Coordinator (hierarchical-coordinator)
├── Manager-1 (mesh-coordinator)
│   ├── Worker-1 (researcher)
│   ├── Worker-2 (coder)
│   └── Worker-3 (analyst)
├── Manager-2 (mesh-coordinator)
│   ├── Worker-4 (researcher)
│   ├── Worker-5 (coder)
│   └── Worker-6 (analyst)
└── Manager-3 (mesh-coordinator)
    ├── Worker-7 (researcher)
    ├── Worker-8 (coder)
    └── Worker-9 (analyst)
```

### Fault Injection Results
```json
{
  "injectedFaults": [
    {
      "type": "unresponsive",
      "target": "Worker-5",
      "timestamp": "2025-11-02T14:32:15Z",
      "detected": true,
      "detectionTime": 12,
      "recoveryAction": "workload-redistribution",
      "recoverySuccess": true
    },
    {
      "type": "crash",
      "target": "Manager-3",
      "timestamp": "2025-11-02T14:33:45Z",
      "detected": true,
      "detectionTime": 305,
      "recoveryAction": "agent-reassignment",
      "recoverySuccess": true
    },
    {
      "type": "byzantine",
      "target": "Worker-3",
      "timestamp": "2025-11-02T14:34:20Z",
      "detected": true,
      "detectionTime": 8,
      "recoveryAction": "agent-replacement",
      "recoverySuccess": true
    }
  ],
  "taskCompletion": "success",
  "totalTime": 487,
  "faultTolerance": "verified"
}
```

### Recovery Metrics
```json
{
  "recoveryActions": 3,
  "agentsReplaced": 1,
  "workloadRedistributions": 2,
  "escalationsToCoordinator": 1,
  "meanTimeToDetection": 108.3,
  "meanTimeToRecovery": 45.7,
  "taskSuccessRate": 1.0
}
```

## Validation Steps

### 1. Pre-Fault Validation
```bash
# Verify hierarchical structure
npx claude-flow@alpha swarm status --verbose

# Expected: 13 agents, 3 levels, all active
```

### 2. Fault Injection
```bash
# Inject worker failure
npx claude-flow@alpha test inject-fault --agent Worker-5 --type unresponsive

# Inject manager crash
npx claude-flow@alpha test inject-fault --agent Manager-3 --type crash --delay 60

# Inject Byzantine behavior
npx claude-flow@alpha test inject-fault --agent Worker-3 --type byzantine --pattern conflicting-results
```

### 3. Monitor Recovery
```bash
# Watch swarm adapt to failures
npx claude-flow@alpha swarm monitor --interval 5 --duration 300
```

### 4. Validate Task Completion
```bash
# Check if task completed despite failures
npx claude-flow@alpha task status

# Expected: status=completed, agentsParticipated=13, faultsRecovered=3
```

### 5. Analyze Fault Tolerance Metrics
```bash
npx claude-flow@alpha metrics analyze --focus fault-tolerance
```

## Acceptance Criteria

- [ ] Hierarchical topology with 13 agents deployed successfully
- [ ] Worker failure detected and work redistributed within 15 seconds
- [ ] Manager failure detected and agents reassigned within 310 seconds
- [ ] Byzantine agent detected and replaced within 30 seconds
- [ ] Task completed successfully despite 3 injected failures
- [ ] No data loss or corruption during recovery
- [ ] Raft consensus maintained throughout failures
- [ ] All recovery actions logged in memory with metadata

## Known Issues & Limitations

- **Detection latency**: Health checks run every 15 seconds - failures not instant
- **Coordinator SPOF**: If coordinator fails, entire swarm requires manual recovery
- **Byzantine detection**: Requires multiple samples to confirm (8-12 seconds)
- **Raft overhead**: Leader election adds 100-200ms latency during manager failures

## Edge Cases Tested

1. **Cascading failures**: Multiple agents failing simultaneously
2. **Byzantine at different levels**: Worker vs. Manager Byzantine behavior
3. **Network partition**: Simulated by making agents unresponsive
4. **Consensus split-brain**: Prevented by Raft's majority requirement

## Cleanup
```bash
npx claude-flow@alpha swarm destroy --force
npx claude-flow@alpha memory clear --key "swarm/hierarchy-fault-test"
```
