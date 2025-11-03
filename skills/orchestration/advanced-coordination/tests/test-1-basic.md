# Test Case 1: Basic Mesh Topology Deployment

## Test Metadata
- **Test ID**: ADV-COORD-001
- **Category**: Basic Functionality
- **Topology**: Mesh
- **Duration**: ~2 minutes

## Scenario
Initialize a 5-agent mesh topology swarm for collaborative code analysis task.
Verify all agents connect peer-to-peer and achieve consensus on identified patterns.

## Prerequisites
- Claude Flow installed (`npm install -g claude-flow@alpha`)
- MCP servers configured
- No existing active swarms

## Input Configuration

**Topology File**: `resources/templates/mesh-topology.yaml`

```yaml
topology: mesh
maxAgents: 5
strategy: balanced
consensus: byzantine
```

**Task**:
```
Analyze the codebase for common design patterns and anti-patterns.
Focus on: singleton usage, factory patterns, observer patterns, god objects.
```

## Expected Behavior

### Phase 1: Initialization
1. Swarm initializes with mesh topology
2. 5 agents spawn with default capabilities
3. Peer-to-peer connections established (10 edges for 5 nodes)
4. Byzantine consensus protocol activated

### Phase 2: Task Execution
1. All 5 agents receive task in parallel
2. Agents independently analyze codebase
3. Agents exchange findings via mesh network
4. Consensus mechanism validates patterns (≥67% agreement)

### Phase 3: Result Aggregation
1. Agreed-upon patterns collected
2. Conflicting findings flagged
3. Final report generated with consensus metadata

## Expected Output

### Topology Validation
```
✓ Topology: mesh
✓ Max Agents: 5
✓ Connections: 10 (n*(n-1)/2 = 5*4/2)
✓ Strategy: balanced
✓ Consensus: byzantine (threshold: 0.67)
```

### Agent List
```
Agent 1: researcher (status: active)
Agent 2: coder (status: active)
Agent 3: analyst (status: active)
Agent 4: optimizer (status: active)
Agent 5: coordinator (status: active)
```

### Consensus Results
```json
{
  "patternsIdentified": {
    "singleton": {
      "count": 7,
      "consensus": 1.0,
      "locations": ["auth.js", "db.js", "logger.js"]
    },
    "factory": {
      "count": 3,
      "consensus": 0.8,
      "locations": ["userFactory.js"]
    },
    "godObject": {
      "count": 2,
      "consensus": 1.0,
      "locations": ["AppController.js"],
      "severity": "high"
    }
  },
  "consensusRate": 0.92,
  "disagreements": [
    {
      "pattern": "observer",
      "agentsFor": ["agent-1", "agent-3"],
      "agentsAgainst": ["agent-2", "agent-4", "agent-5"],
      "resolution": "rejected (40% < 67% threshold)"
    }
  ]
}
```

## Validation Steps

### 1. Topology Structure
```bash
npx claude-flow@alpha swarm status --verbose
```
**Expected**:
- Topology type: mesh
- Active agents: 5
- Connection count: 10
- All agents status: active

### 2. Consensus Verification
```bash
npx claude-flow@alpha swarm metrics --metric consensusRate
```
**Expected**:
- Consensus rate: ≥0.67
- Byzantine fault tolerance: active
- Disagreements resolved: yes

### 3. Task Completion
```bash
npx claude-flow@alpha task status
```
**Expected**:
- Task status: completed
- Agents participated: 5/5
- Results aggregated: yes

### 4. Memory Persistence
```bash
npx claude-flow@alpha memory retrieve --key "swarm/mesh-analysis"
```
**Expected**:
- Patterns stored with metadata
- Consensus metadata included
- Cross-agent references present

## Acceptance Criteria

- [ ] Mesh topology initialized successfully
- [ ] All 5 agents active and connected (10 edges)
- [ ] Byzantine consensus achieved (≥67% threshold)
- [ ] All agents participated in task
- [ ] Patterns identified with consensus scores
- [ ] Disagreements properly resolved
- [ ] Results stored in persistent memory
- [ ] No agent failures or timeouts

## Known Issues & Limitations

- **Consensus overhead**: Byzantine consensus adds 15-20% overhead vs. simple voting
- **Small swarm size**: 5 agents is minimum recommended for Byzantine (tolerates 1 fault)
- **Network complexity**: Mesh creates O(n²) connections - not suitable for >15 agents

## Cleanup
```bash
npx claude-flow@alpha swarm destroy
```
