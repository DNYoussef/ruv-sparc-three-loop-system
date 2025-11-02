# Example 2: Gossip Protocol - Peer-to-Peer State Sharing for 100+ Agents

## Overview

This example demonstrates how to implement Gossip-based coordination for a large-scale distributed caching system with 100+ agents. Gossip protocols provide eventual consistency and high scalability through epidemic-style state propagation, making them ideal for scenarios requiring high availability over strong consistency.

## Use Case: Distributed Configuration Management with Eventual Consistency

**Scenario**: Propagate configuration updates across 150 agents in 3 data centers with sub-second convergence time and resilience to network partitions.

**Requirements**:
- High scalability (100+ agents without coordinator bottleneck)
- Eventual consistency (all agents converge to same state)
- Network partition resilience (auto-healing after reconnection)
- Low latency (O(log N) convergence rounds)
- Decentralized (no single point of failure)

## Architecture

```
┌─────────┐     ┌─────────┐     ┌─────────┐
│ Agent 1 │────▶│ Agent 2 │────▶│ Agent 3 │
└────┬────┘     └────┬────┘     └────┬────┘
     │               │               │
     │  Fanout=3     │  Fanout=3     │  Fanout=3
     │               │               │
┌────▼────┐     ┌───▼─────┐     ┌───▼─────┐
│ Agent 4 │     │ Agent 5 │     │ Agent 6 │
└────┬────┘     └────┬────┘     └────┬────┘
     │               │               │
     └───────────┬───┴───────────────┘
                 │
          ┌──────▼───────┐
          │   ... 144    │
          │ More Agents  │
          └──────────────┘

Epidemic Spread: Each agent gossips to N random peers per round
Convergence: O(log N) rounds = ~7 rounds for 150 agents
```

## Step-by-Step Implementation

### Step 1: Initialize Gossip Cluster

```bash
# Initialize mesh topology with Gossip protocol
npx claude-flow@alpha swarm init \
  --topology mesh \
  --protocol gossip \
  --max-agents 150 \
  --regions us-east,us-west,eu-central

# Verify swarm creation
npx claude-flow@alpha swarm status --verbose
```

**Expected Output**:
```json
{
  "swarm_id": "swarm-gossip-001",
  "topology": "mesh",
  "protocol": "gossip",
  "status": "initializing",
  "agents": [],
  "regions": ["us-east", "us-west", "eu-central"],
  "created_at": "2025-11-02T11:00:00Z"
}
```

### Step 2: Configure Gossip Parameters

```bash
# Configure epidemic parameters
npx claude-flow@alpha gossip config \
  --fanout 3 \
  --interval 100 \
  --convergence-threshold 0.95 \
  --anti-entropy true \
  --anti-entropy-interval 10000

# Configure network partition handling
npx claude-flow@alpha gossip config \
  --partition-detection true \
  --healing-strategy merge-last-write-wins \
  --conflict-resolution vector-clock
```

**Parameter Explanation**:
- `fanout=3`: Each agent gossips to 3 random peers per round
- `interval=100`: Gossip rounds every 100ms
- `convergence-threshold=0.95`: 95% agents must agree
- `anti-entropy=true`: Periodic full state sync (backup)
- `partition-detection=true`: Detect network splits

### Step 3: Spawn Gossip Agents Across Regions

```bash
# Spawn 50 agents in us-east
for i in {1..50}; do
  npx claude-flow@alpha agent spawn \
    --type worker \
    --name "gossip-us-east-$i" \
    --region us-east \
    --capabilities "config-caching,state-replication,peer-gossip" &
done
wait

# Spawn 50 agents in us-west
for i in {1..50}; do
  npx claude-flow@alpha agent spawn \
    --type worker \
    --name "gossip-us-west-$i" \
    --region us-west \
    --capabilities "config-caching,state-replication,peer-gossip" &
done
wait

# Spawn 50 agents in eu-central
for i in {1..50}; do
  npx claude-flow@alpha agent spawn \
    --type worker \
    --name "gossip-eu-central-$i" \
    --region eu-central \
    --capabilities "config-caching,state-replication,peer-gossip" &
done
wait

# Verify all 150 agents registered
npx claude-flow@alpha agent list --filter active | wc -l
```

**Expected Output**: `150` (all agents active)

### Step 4: Initialize Distributed State

```bash
# Inject initial configuration state into random agent
npx claude-flow@alpha gossip state-update \
  --key "app.config" \
  --value '{
    "database_url": "postgres://db.example.com/prod",
    "cache_ttl": 3600,
    "feature_flags": {
      "new_ui": true,
      "beta_api": false
    }
  }' \
  --agent gossip-us-east-1

# Monitor gossip propagation
npx claude-flow@alpha gossip convergence --watch
```

### Step 5: Monitor Epidemic Spread

```bash
# Track convergence progress (expect O(log N) rounds)
npx claude-flow@alpha gossip metrics --metric convergence-rate

# Expected: ~7 rounds for 150 agents
```

**Expected Output (Per Round)**:
```
Round | Infected Agents | Convergence % | Duration
------|-----------------|---------------|----------
1     | 1               | 0.67%         | 100ms
2     | 4               | 2.67%         | 100ms
3     | 13              | 8.67%         | 100ms
4     | 40              | 26.67%        | 100ms
5     | 85              | 56.67%        | 100ms
6     | 132             | 88.00%        | 100ms
7     | 148             | 98.67%        | 100ms

Total convergence time: 700ms (7 rounds × 100ms)
```

### Step 6: Verify Eventual Consistency

```bash
# Query configuration from multiple random agents
for region in us-east us-west eu-central; do
  echo "Checking $region..."
  npx claude-flow@alpha gossip query-state \
    --key "app.config" \
    --agent "gossip-$region-10"
done

# Expected: All agents return identical configuration
```

**Expected Output**:
```json
{
  "region": "us-east",
  "agent": "gossip-us-east-10",
  "key": "app.config",
  "value": {
    "database_url": "postgres://db.example.com/prod",
    "cache_ttl": 3600,
    "feature_flags": {
      "new_ui": true,
      "beta_api": false
    }
  },
  "version": 1,
  "last_updated": "2025-11-02T11:01:00.700Z"
}
```

### Step 7: Simulate Network Partition

```bash
# Partition network between us-east and us-west/eu-central
npx claude-flow@alpha network partition \
  --group1 us-east \
  --group2 us-west,eu-central \
  --duration 30000

# Update configuration in both partitions (conflict!)
# Group 1: Update in us-east
npx claude-flow@alpha gossip state-update \
  --key "app.config.cache_ttl" \
  --value 7200 \
  --agent gossip-us-east-1

# Group 2: Update in us-west (different value)
npx claude-flow@alpha gossip state-update \
  --key "app.config.cache_ttl" \
  --value 1800 \
  --agent gossip-us-west-1

# Monitor partition status
npx claude-flow@alpha network partition-status
```

### Step 8: Test Partition Healing

```bash
# Heal network partition after 30 seconds
npx claude-flow@alpha network heal-partition

# Monitor conflict resolution (last-write-wins)
npx claude-flow@alpha gossip conflicts --watch

# Wait for re-convergence
sleep 5

# Verify conflict resolved
npx claude-flow@alpha gossip query-state \
  --key "app.config.cache_ttl" \
  --agent gossip-us-east-1
```

**Expected Output (Conflict Resolution)**:
```json
{
  "conflict_detected": true,
  "resolution_strategy": "last-write-wins",
  "winning_value": 1800,
  "winning_timestamp": "2025-11-02T11:01:25.000Z",
  "losing_value": 7200,
  "losing_timestamp": "2025-11-02T11:01:20.000Z",
  "affected_agents": 150
}
```

### Step 9: Performance Benchmarking

```bash
# Benchmark gossip throughput
npx claude-flow@alpha gossip benchmark \
  --updates 10000 \
  --concurrency 50

# Expected: 10,000 updates/sec (highly parallel)
```

**Expected Output**:
```json
{
  "total_updates": 10000,
  "duration_sec": 1.2,
  "throughput_ops_sec": 8333,
  "avg_convergence_ms": 700,
  "p95_convergence_ms": 850,
  "p99_convergence_ms": 1100,
  "partition_events": 0,
  "conflicts_resolved": 0
}
```

## Integration with Memory MCP

Store gossip state persistently across sessions:

```bash
# Store gossip convergence metrics
npx claude-flow@alpha memory store \
  --key "coordination/gossip/metrics" \
  --value '{
    "convergence_rounds": 7,
    "convergence_time_ms": 700,
    "agents_count": 150,
    "fanout": 3,
    "timestamp": "2025-11-02T11:01:00.700Z"
  }'

# Store partition healing events
npx claude-flow@alpha memory store \
  --key "coordination/gossip/partition-history" \
  --value '[
    {
      "partition_start": "2025-11-02T11:01:10.000Z",
      "partition_end": "2025-11-02T11:01:40.000Z",
      "affected_regions": ["us-east", "us-west"],
      "conflicts_resolved": 1,
      "healing_time_ms": 5000
    }
  ]'

# Retrieve gossip metrics for analysis
npx claude-flow@alpha memory retrieve \
  --key "coordination/gossip/metrics"
```

## Hooks Integration

Automate gossip coordination with lifecycle hooks:

```javascript
// File: hooks/gossip-coordination.js
module.exports = {
  preTask: async (context) => {
    // Initialize gossip before task execution
    await context.exec('npx claude-flow@alpha swarm init --protocol gossip');

    // Wait for 95% convergence threshold
    await context.waitFor('convergence-reached', { threshold: 0.95 });
  },

  postEdit: async (context, file) => {
    // Propagate file metadata via gossip
    await context.exec(`npx claude-flow@alpha gossip state-update \
      --key "file-metadata/${file}" \
      --value '{"modified": "${new Date().toISOString()}"}'`);
  },

  onPartitionDetected: async (context, groups) => {
    // Log partition event
    console.warn(`Network partition detected: ${groups.join(' | ')}`);

    // Enable degraded mode (read-only)
    await context.exec('npx claude-flow@alpha gossip degraded-mode --enable');
  },

  onPartitionHealed: async (context) => {
    // Resume normal operations
    await context.exec('npx claude-flow@alpha gossip degraded-mode --disable');

    // Force anti-entropy for rapid re-convergence
    await context.exec('npx claude-flow@alpha gossip anti-entropy --immediate');
  }
};
```

## Performance Metrics

### Baseline Performance (No Partitions)

- **Convergence time**: 700ms (7 rounds × 100ms)
- **Throughput**: 8,333 updates/sec (fully distributed)
- **Message complexity**: O(N log N) = ~1,050 messages (150 agents × 7 rounds)
- **Network bandwidth**: ~5 MB/sec (assuming 500-byte state)

### Partition Scenario (30-second split)

- **Partition detection time**: 500ms (3-5 missed gossip rounds)
- **Healing time**: 5,000ms (partition heal + re-convergence)
- **Conflicts resolved**: 1 (last-write-wins)
- **Data loss**: 0 (eventual consistency preserved)

### Scalability Analysis

| Agents | Convergence Rounds | Convergence Time (100ms/round) |
|--------|-------------------|-------------------------------|
| 10     | 4                 | 400ms                         |
| 50     | 6                 | 600ms                         |
| 150    | 7                 | 700ms                         |
| 500    | 9                 | 900ms                         |
| 1000   | 10                | 1,000ms                       |

**Observation**: Logarithmic scaling (O(log N)) maintains sub-second convergence even for 1,000+ agents.

## Troubleshooting

### Problem: Slow Convergence

**Symptom**: Taking >2 seconds for 150 agents to converge

**Root Cause**: Low fanout or high interval

**Solution**:
```bash
# Increase fanout (more peers per round)
npx claude-flow@alpha gossip config --fanout 5

# Decrease interval (faster rounds)
npx claude-flow@alpha gossip config --interval 50

# Enable push-pull (bidirectional gossip)
npx claude-flow@alpha gossip config --mode push-pull
```

### Problem: High Network Bandwidth

**Symptom**: Network utilization >100 MB/sec

**Root Cause**: Large state size or high fanout

**Solution**:
```bash
# Reduce fanout
npx claude-flow@alpha gossip config --fanout 2

# Enable delta compression
npx claude-flow@alpha gossip config --compression delta

# Increase interval (fewer rounds per second)
npx claude-flow@alpha gossip config --interval 200
```

### Problem: Partition Not Healing

**Symptom**: Two groups remain diverged after network restoration

**Root Cause**: Partition detection disabled or anti-entropy not running

**Solution**:
```bash
# Force partition detection
npx claude-flow@alpha network verify-connectivity

# Trigger manual anti-entropy
npx claude-flow@alpha gossip anti-entropy --force --all-agents

# Check conflict resolution logs
npx claude-flow@alpha gossip conflicts --history
```

## Advanced: Multi-Datacenter Gossip

Optimize gossip for cross-region deployments:

```bash
# Configure region-aware gossip (prefer local peers)
npx claude-flow@alpha gossip config \
  --region-affinity 0.7 \
  --cross-region-fanout 1

# Region-affinity=0.7: 70% gossip to same region, 30% cross-region
# This reduces WAN bandwidth while maintaining global convergence
```

## Key Takeaways

1. **Decentralized coordination**: No single point of failure (peer-to-peer)
2. **Epidemic spread**: O(log N) convergence rounds scale logarithmically
3. **Eventual consistency**: All agents converge to same state eventually
4. **Partition resilience**: Automatic healing after network restoration
5. **High throughput**: 8,000+ updates/sec (no coordinator bottleneck)
6. **Tuneable parameters**: Balance convergence speed vs network bandwidth
7. **Conflict resolution**: Last-write-wins or vector clocks for diverged state

## Comparison with RAFT (Example 1)

| Metric                | RAFT (Example 1) | Gossip (Example 2) |
|-----------------------|------------------|-------------------|
| **Agents**            | 5                | 150               |
| **Consistency**       | Strong           | Eventual          |
| **Coordinator**       | Leader-based     | Decentralized     |
| **Throughput**        | 2,500 ops/sec    | 8,333 ops/sec     |
| **Fault Tolerance**   | N/2+1 failures   | Very high         |
| **Convergence Time**  | 25ms (commit)    | 700ms (epidemic)  |
| **Use Case**          | Critical tasks   | Scalable caching  |

## Next Steps

- **Example 3**: [Byzantine Fault Tolerance](./example-3-byzantine-fault-tolerance.md) for malicious agent detection
- **Reference**: [Coordination Strategies](../references/coordination-strategies.md) for Gossip deep dive
- **Reference**: [Fault Tolerance](../references/fault-tolerance.md) for partition handling patterns
