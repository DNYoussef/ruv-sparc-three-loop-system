# Example 1: RAFT Consensus - Leader Election for 5+ Agents

## Overview

This example demonstrates how to implement RAFT consensus for a distributed task processing system with 5 agents. RAFT provides strong consistency guarantees through leader-based coordination, making it ideal for workflows requiring ordered operations and state machine replication.

## Use Case: Distributed Task Queue with Exactly-Once Processing

**Scenario**: Process 1,000 API requests across 5 agents with guaranteed exactly-once execution, even during agent failures or network partitions.

**Requirements**:
- Strong consistency (no duplicate task processing)
- Automatic leader election (failover within 300ms)
- Replicated log (all agents maintain consistent task history)
- Fault tolerance (survive 2 agent failures with 5 total agents)

## Architecture

```
┌─────────────┐
│   Leader    │ ← Elected via RAFT
│  (Agent 1)  │
└──────┬──────┘
       │
       │ Heartbeats + Log Replication
       │
   ┌───┴────┬─────────┬─────────┐
   │        │         │         │
┌──▼───┐ ┌─▼────┐ ┌──▼───┐ ┌───▼──┐
│Follower│ │Follower│ │Follower│ │Follower│
│(Agent 2)│ │(Agent 3)│ │(Agent 4)│ │(Agent 5)│
└────────┘ └───────┘ └────────┘ └──────┘
```

## Step-by-Step Implementation

### Step 1: Initialize RAFT Cluster

```bash
# Initialize mesh topology with RAFT consensus
npx claude-flow@alpha swarm init \
  --topology mesh \
  --consensus raft \
  --max-agents 5

# Verify swarm creation
npx claude-flow@alpha swarm status --verbose
```

**Expected Output**:
```json
{
  "swarm_id": "swarm-raft-001",
  "topology": "mesh",
  "consensus": "raft",
  "status": "initializing",
  "agents": [],
  "created_at": "2025-11-02T10:00:00Z"
}
```

### Step 2: Configure RAFT Parameters

```bash
# Set election timeout (150-300ms randomized)
npx claude-flow@alpha consensus config \
  --protocol raft \
  --election-timeout-min 150 \
  --election-timeout-max 300 \
  --heartbeat-interval 50 \
  --log-compaction-threshold 1000

# Configure quorum size (majority = 3 for 5 agents)
npx claude-flow@alpha consensus config \
  --quorum-size 3 \
  --max-log-entries 10000
```

### Step 3: Spawn RAFT Agents

```bash
# Spawn 5 agents for RAFT cluster
for i in {1..5}; do
  npx claude-flow@alpha agent spawn \
    --type coordinator \
    --name "raft-agent-$i" \
    --capabilities "task-processing,log-replication,voting"
done

# Verify all agents registered
npx claude-flow@alpha agent list --filter active
```

**Expected Output**:
```
Agent ID          | Type        | Status | Capabilities
raft-agent-1      | coordinator | active | task-processing,log-replication,voting
raft-agent-2      | coordinator | active | task-processing,log-replication,voting
raft-agent-3      | coordinator | active | task-processing,log-replication,voting
raft-agent-4      | coordinator | active | task-processing,log-replication,voting
raft-agent-5      | coordinator | active | task-processing,log-replication,voting
```

### Step 4: Trigger Leader Election

```bash
# Leader election happens automatically
# Monitor election progress
npx claude-flow@alpha consensus status --watch

# Wait for leader election (typically 150-300ms)
sleep 1

# Verify leader elected
npx claude-flow@alpha consensus leader
```

**Expected Output**:
```json
{
  "leader_id": "raft-agent-1",
  "term": 1,
  "voted_by": ["raft-agent-1", "raft-agent-2", "raft-agent-3"],
  "election_duration_ms": 187,
  "heartbeat_status": "healthy"
}
```

### Step 5: Submit Distributed Tasks

```bash
# Submit 1,000 tasks to RAFT leader
npx claude-flow@alpha task orchestrate \
  --task "Process API requests with RAFT coordination" \
  --strategy sequential \
  --max-agents 5 \
  --priority high \
  --count 1000

# Monitor task progress
npx claude-flow@alpha task status --watch
```

### Step 6: Monitor Log Replication

```bash
# Check log replication across all agents
npx claude-flow@alpha consensus log-status

# Expected: All agents have identical commit index
```

**Expected Output**:
```json
{
  "leader": "raft-agent-1",
  "followers": [
    {
      "agent_id": "raft-agent-2",
      "match_index": 500,
      "next_index": 501,
      "replication_lag_ms": 12
    },
    {
      "agent_id": "raft-agent-3",
      "match_index": 500,
      "next_index": 501,
      "replication_lag_ms": 15
    },
    {
      "agent_id": "raft-agent-4",
      "match_index": 500,
      "next_index": 501,
      "replication_lag_ms": 10
    },
    {
      "agent_id": "raft-agent-5",
      "match_index": 500,
      "next_index": 501,
      "replication_lag_ms": 18
    }
  ],
  "committed_entries": 500,
  "uncommitted_entries": 0
}
```

### Step 7: Simulate Leader Failure

```bash
# Manually terminate leader to test failover
npx claude-flow@alpha agent terminate --agent-id raft-agent-1

# Monitor new leader election
npx claude-flow@alpha consensus status --watch

# Verify new leader elected within 300ms
```

**Expected Output**:
```json
{
  "event": "leader_election",
  "old_leader": "raft-agent-1",
  "new_leader": "raft-agent-2",
  "term": 2,
  "election_duration_ms": 243,
  "voted_by": ["raft-agent-2", "raft-agent-3", "raft-agent-5"],
  "tasks_recovered": 500
}
```

### Step 8: Verify Exactly-Once Execution

```bash
# Check task execution counts
npx claude-flow@alpha task results --format summary

# Expected: All 1,000 tasks executed exactly once
```

**Expected Output**:
```json
{
  "total_tasks": 1000,
  "completed": 1000,
  "failed": 0,
  "duplicates": 0,
  "leader_failures": 1,
  "elections": 2,
  "execution_guarantee": "exactly-once"
}
```

## Integration with Memory MCP

Store RAFT state persistently across sessions:

```bash
# Store leader election history
npx claude-flow@alpha memory store \
  --key "coordination/raft/election-history" \
  --value '{
    "elections": [
      {"term": 1, "leader": "raft-agent-1", "duration_ms": 187},
      {"term": 2, "leader": "raft-agent-2", "duration_ms": 243}
    ]
  }'

# Store committed log entries
npx claude-flow@alpha memory store \
  --key "coordination/raft/commit-index" \
  --value '{"commit_index": 1000, "term": 2, "timestamp": "2025-11-02T10:30:00Z"}'

# Retrieve RAFT state for recovery
npx claude-flow@alpha memory retrieve \
  --key "coordination/raft/commit-index"
```

## Hooks Integration

Automate RAFT coordination with lifecycle hooks:

```javascript
// File: hooks/raft-coordination.js
module.exports = {
  preTask: async (context) => {
    // Initialize RAFT before task execution
    await context.exec('npx claude-flow@alpha swarm init --consensus raft');

    // Wait for leader election
    await context.waitFor('leader-elected', { timeout: 5000 });
  },

  postEdit: async (context, file) => {
    // Replicate file changes to all RAFT followers
    await context.exec(`npx claude-flow@alpha consensus replicate --file ${file}`);
  },

  onFailure: async (context, agent) => {
    // Trigger leader election if leader fails
    if (agent.role === 'leader') {
      await context.exec('npx claude-flow@alpha consensus trigger-election');
    }
  }
};
```

## Performance Metrics

### Baseline Performance (No Failures)

- **Leader election time**: 187ms (within 150-300ms target)
- **Task throughput**: 2,500 tasks/sec (leader bottleneck)
- **Replication lag**: 10-18ms (network latency)
- **Commit latency**: 25ms average (leader → followers → commit)

### Failure Scenario (Leader Crash)

- **Failure detection time**: 150ms (heartbeat timeout)
- **Election time**: 243ms (randomized election timeout)
- **Total failover time**: 393ms (detection + election)
- **Tasks lost**: 0 (committed log preserved)
- **Tasks in-flight**: 50 (uncommitted, replayed by new leader)

## Troubleshooting

### Problem: Split-Brain (Multiple Leaders)

**Symptom**: Two agents both claiming to be leader

**Root Cause**: Network partition preventing majority quorum

**Solution**:
```bash
# Check network connectivity
npx claude-flow@alpha network diagnose

# Verify quorum size (must be majority)
npx claude-flow@alpha consensus verify-quorum

# Force step-down of invalid leader
npx claude-flow@alpha consensus step-down --agent raft-agent-4
```

### Problem: Slow Commit Latency

**Symptom**: Tasks taking >100ms to commit

**Root Cause**: High replication lag or slow disk I/O

**Solution**:
```bash
# Increase heartbeat frequency
npx claude-flow@alpha consensus config --heartbeat-interval 25

# Enable log batching
npx claude-flow@alpha consensus config --batch-size 100

# Monitor disk I/O
npx claude-flow@alpha agent metrics --metric disk-io
```

### Problem: Leader Election Loop

**Symptom**: Continuous leader elections (terms incrementing rapidly)

**Root Cause**: Network instability or faulty agent

**Solution**:
```bash
# Increase election timeout to reduce false elections
npx claude-flow@alpha consensus config --election-timeout-min 300

# Identify problematic agent
npx claude-flow@alpha consensus election-history --limit 10

# Remove faulty agent from cluster
npx claude-flow@alpha agent terminate --agent-id faulty-agent
```

## Advanced: Multi-Group RAFT

For larger deployments, partition agents into multiple RAFT groups:

```bash
# Group 1: Critical tasks (strong consistency)
npx claude-flow@alpha swarm init \
  --name critical-group \
  --consensus raft \
  --max-agents 5

# Group 2: Background tasks (best-effort)
npx claude-flow@alpha swarm init \
  --name background-group \
  --consensus raft \
  --max-agents 3

# Route tasks to appropriate group
npx claude-flow@alpha task orchestrate \
  --task "Critical payment processing" \
  --swarm critical-group
```

## Key Takeaways

1. **Leader-based coordination**: Single leader eliminates split-brain scenarios
2. **Majority quorum**: N/2+1 agents must agree for commits (3/5 in this example)
3. **Automatic failover**: New leader elected within 300ms of failure
4. **Replicated log**: All agents maintain identical task history
5. **Exactly-once execution**: Committed tasks never duplicated
6. **Bounded recovery**: Failed agents catch up from committed log
7. **Tuneable parameters**: Balance responsiveness vs stability

## Next Steps

- **Example 2**: [Gossip Protocol](./example-2-gossip-protocol.md) for high scalability (100+ agents)
- **Example 3**: [Byzantine Fault Tolerance](./example-3-byzantine-fault-tolerance.md) for malicious agent detection
- **Reference**: [Coordination Strategies](../references/coordination-strategies.md) for in-depth RAFT analysis
