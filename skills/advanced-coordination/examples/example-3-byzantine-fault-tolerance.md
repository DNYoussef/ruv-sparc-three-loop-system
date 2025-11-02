# Example 3: Byzantine Fault Tolerance - Malicious Agent Detection with 3F+1 Redundancy

## Overview

This example demonstrates how to implement Byzantine Fault Tolerance (BFT) for a security-critical transaction processing system with 7 agents. BFT provides protection against malicious or faulty agents through cryptographic verification and quorum-based consensus, making it essential for untrusted distributed environments.

## Use Case: Financial Transaction Validation with Malicious Agent Protection

**Scenario**: Process 1,000 financial transactions with cryptographic verification across 7 agents, tolerating up to 2 Byzantine (malicious/faulty) agents without compromising system integrity.

**Requirements**:
- Byzantine fault tolerance (survive 2 malicious agents with 7 total)
- Cryptographic verification (Ed25519 signatures)
- Strong consistency (no double-spending or conflicting transactions)
- Quorum-based consensus (2F+1 agreement required)
- Malicious agent detection and replacement

## Architecture

```
┌────────────────────────────────────────┐
│          Byzantine Network             │
│                                        │
│  ┌────────┐                            │
│  │Primary │ ← Proposes transactions    │
│  │(Agent 1)│                            │
│  └───┬────┘                            │
│      │                                 │
│      │ Broadcast + Digital Signatures  │
│      │                                 │
│  ┌───▼────┬────────┬────────┬────────┐│
│  │        │        │        │        ││
│┌─▼───┐ ┌─▼───┐ ┌──▼───┐ ┌──▼───┐ ┌──▼┐│
││Rep 2│ │Rep 3│ │Rep 4 │ │Rep 5 │ │Rep││
││✓    │ │✓    │ │✓     │ │⚠Mal  │ │⚠Ma││
│└─────┘ └─────┘ └──────┘ └──────┘ └───┘│
│                                        │
│ Quorum = 2F+1 = 5 (for F=2 failures)   │
└────────────────────────────────────────┘

3F+1 = 7 total agents
F = 2 max Byzantine failures
Quorum = 5 correct signatures required
```

## Step-by-Step Implementation

### Step 1: Initialize Byzantine Cluster

```bash
# Initialize star topology with Byzantine consensus
# 3F+1 rule: 7 agents tolerate 2 Byzantine failures (F=2)
npx claude-flow@alpha swarm init \
  --topology star \
  --consensus byzantine \
  --max-agents 7 \
  --max-failures 2

# Verify swarm creation
npx claude-flow@alpha swarm status --verbose
```

**Expected Output**:
```json
{
  "swarm_id": "swarm-bft-001",
  "topology": "star",
  "consensus": "byzantine",
  "status": "initializing",
  "max_failures": 2,
  "required_agents": 7,
  "quorum_size": 5,
  "created_at": "2025-11-02T12:00:00Z"
}
```

### Step 2: Configure Byzantine Parameters

```bash
# Configure Byzantine Fault Tolerance parameters
npx claude-flow@alpha byzantine config \
  --signature-algorithm ed25519 \
  --verification-threshold 0.71 \
  --view-change-timeout 5000 \
  --checkpoint-interval 100 \
  --max-concurrent-requests 50

# Enable malicious agent detection
npx claude-flow@alpha byzantine config \
  --malicious-detection true \
  --detection-strategy signature-mismatch \
  --quarantine-duration 3600000
```

**Parameter Explanation**:
- `signature-algorithm=ed25519`: Cryptographic signing (32-byte keys)
- `verification-threshold=0.71`: 71% = 5/7 agents (2F+1 quorum)
- `view-change-timeout=5000`: Replace suspected malicious primary
- `checkpoint-interval=100`: Garbage collect verified transactions
- `quarantine-duration=3600000`: Ban malicious agents for 1 hour

### Step 3: Generate Cryptographic Identities

```bash
# Generate Ed25519 keypairs for all agents
for i in {1..7}; do
  npx claude-flow@alpha byzantine generate-keypair \
    --agent-name "bft-agent-$i" \
    --output "keys/bft-agent-$i-key.pem"
done

# Distribute public keys to all agents (PKI)
npx claude-flow@alpha byzantine distribute-pubkeys \
  --key-directory "keys/" \
  --all-agents
```

**Expected Output**:
```
Generated keypairs:
  bft-agent-1: Public=0x1a2b3c4d... Private=keys/bft-agent-1-key.pem
  bft-agent-2: Public=0x2b3c4d5e... Private=keys/bft-agent-2-key.pem
  bft-agent-3: Public=0x3c4d5e6f... Private=keys/bft-agent-3-key.pem
  bft-agent-4: Public=0x4d5e6f7a... Private=keys/bft-agent-4-key.pem
  bft-agent-5: Public=0x5e6f7a8b... Private=keys/bft-agent-5-key.pem
  bft-agent-6: Public=0x6f7a8b9c... Private=keys/bft-agent-6-key.pem
  bft-agent-7: Public=0x7a8b9cad... Private=keys/bft-agent-7-key.pem
```

### Step 4: Spawn Byzantine Agents

```bash
# Spawn 7 agents with Byzantine capabilities
for i in {1..7}; do
  npx claude-flow@alpha agent spawn \
    --type validator \
    --name "bft-agent-$i" \
    --capabilities "transaction-validation,signature-verification,consensus" \
    --keypair "keys/bft-agent-$i-key.pem" &
done
wait

# Verify all agents registered
npx claude-flow@alpha agent list --filter active
```

**Expected Output**:
```
Agent ID     | Type      | Status | Public Key           | Role
bft-agent-1  | validator | active | 0x1a2b3c4d...       | primary
bft-agent-2  | validator | active | 0x2b3c4d5e...       | replica
bft-agent-3  | validator | active | 0x3c4d5e6f...       | replica
bft-agent-4  | validator | active | 0x4d5e6f7a...       | replica
bft-agent-5  | validator | active | 0x5e6f7a8b...       | replica
bft-agent-6  | validator | active | 0x6f7a8b9c...       | replica
bft-agent-7  | validator | active | 0x7a8b9cad...       | replica
```

### Step 5: Submit Byzantine Transactions

```bash
# Submit 1,000 financial transactions to primary
npx claude-flow@alpha task orchestrate \
  --task "Process financial transactions with BFT verification" \
  --strategy sequential \
  --max-agents 7 \
  --priority critical \
  --count 1000 \
  --consensus byzantine

# Monitor consensus progress
npx claude-flow@alpha byzantine status --watch
```

### Step 6: Inject Malicious Agent (Simulation)

```bash
# Simulate malicious agent by corrupting signatures
# This represents a compromised agent attempting fraud

# Manually corrupt agent-5's behavior
npx claude-flow@alpha byzantine inject-fault \
  --agent bft-agent-5 \
  --fault-type signature-corruption \
  --probability 0.5

# Manually corrupt agent-6's behavior
npx claude-flow@alpha byzantine inject-fault \
  --agent bft-agent-6 \
  --fault-type double-spend \
  --probability 0.3

# Monitor malicious detection
npx claude-flow@alpha byzantine detect-malicious --watch
```

### Step 7: Observe Byzantine Detection

```bash
# Check malicious agent detection logs
npx claude-flow@alpha byzantine detection-log --tail 20
```

**Expected Output**:
```json
[
  {
    "timestamp": "2025-11-02T12:05:32.450Z",
    "detected_agent": "bft-agent-5",
    "fault_type": "signature-mismatch",
    "detection_method": "quorum-verification",
    "evidence": {
      "transaction_id": "txn-12345",
      "expected_signature": "0xabcd1234...",
      "received_signature": "0x0000dead...",
      "votes_against": ["bft-agent-1", "bft-agent-2", "bft-agent-3", "bft-agent-4", "bft-agent-7"]
    },
    "action": "quarantined",
    "quarantine_expires": "2025-11-02T13:05:32.450Z"
  },
  {
    "timestamp": "2025-11-02T12:06:15.200Z",
    "detected_agent": "bft-agent-6",
    "fault_type": "double-spend-attempt",
    "detection_method": "transaction-log-verification",
    "evidence": {
      "transaction_ids": ["txn-45678", "txn-45679"],
      "conflicting_outputs": ["account-A: -$100", "account-A: -$100"],
      "double_spend_detected": true
    },
    "action": "quarantined",
    "quarantine_expires": "2025-11-02T13:06:15.200Z"
  }
]
```

### Step 8: Verify Quorum Consensus Despite Malicious Agents

```bash
# Check consensus status (should still be healthy)
npx claude-flow@alpha byzantine consensus-health
```

**Expected Output**:
```json
{
  "status": "healthy",
  "total_agents": 7,
  "malicious_detected": 2,
  "healthy_agents": 5,
  "quorum_met": true,
  "quorum_threshold": 5,
  "transactions_committed": 1000,
  "transactions_rejected": 23,
  "false_positive_rate": 0.0,
  "system_integrity": "intact"
}
```

**Analysis**: System remains functional with 5/7 healthy agents (exceeds 2F+1=5 quorum), rejecting 23 malicious transactions.

### Step 9: Replace Malicious Agents

```bash
# Spawn replacement agents for quarantined ones
npx claude-flow@alpha agent spawn \
  --type validator \
  --name "bft-agent-8" \
  --capabilities "transaction-validation,signature-verification,consensus" \
  --keypair "keys/bft-agent-8-key.pem"

npx claude-flow@alpha agent spawn \
  --type validator \
  --name "bft-agent-9" \
  --capabilities "transaction-validation,signature-verification,consensus" \
  --keypair "keys/bft-agent-9-key.pem"

# Update quorum membership
npx claude-flow@alpha byzantine update-quorum \
  --remove bft-agent-5,bft-agent-6 \
  --add bft-agent-8,bft-agent-9
```

### Step 10: Trigger View Change (Primary Replacement)

```bash
# Simulate malicious primary by corrupting agent-1
npx claude-flow@alpha byzantine inject-fault \
  --agent bft-agent-1 \
  --fault-type censorship \
  --probability 1.0

# Trigger view change (replace primary)
npx claude-flow@alpha byzantine view-change --force

# Monitor new primary election
npx claude-flow@alpha byzantine status --watch
```

**Expected Output**:
```json
{
  "event": "view-change",
  "old_primary": "bft-agent-1",
  "new_primary": "bft-agent-2",
  "view_number": 2,
  "view_change_duration_ms": 4823,
  "triggered_by": ["bft-agent-3", "bft-agent-4", "bft-agent-7", "bft-agent-8", "bft-agent-9"],
  "quorum_met": true,
  "transactions_recovered": 0
}
```

## Integration with Memory MCP

Store Byzantine fault events persistently:

```bash
# Store malicious detection events
npx claude-flow@alpha memory store \
  --key "coordination/byzantine/malicious-detections" \
  --value '[
    {
      "agent": "bft-agent-5",
      "fault": "signature-corruption",
      "timestamp": "2025-11-02T12:05:32.450Z",
      "quarantine_duration": 3600000
    },
    {
      "agent": "bft-agent-6",
      "fault": "double-spend-attempt",
      "timestamp": "2025-11-02T12:06:15.200Z",
      "quarantine_duration": 3600000
    }
  ]'

# Store view change history
npx claude-flow@alpha memory store \
  --key "coordination/byzantine/view-changes" \
  --value '[
    {
      "view": 1,
      "primary": "bft-agent-1",
      "start": "2025-11-02T12:00:00.000Z",
      "end": "2025-11-02T12:10:00.000Z"
    },
    {
      "view": 2,
      "primary": "bft-agent-2",
      "start": "2025-11-02T12:10:00.000Z",
      "end": null
    }
  ]'

# Retrieve Byzantine metrics
npx claude-flow@alpha memory retrieve \
  --key "coordination/byzantine/malicious-detections"
```

## Hooks Integration

Automate Byzantine coordination with lifecycle hooks:

```javascript
// File: hooks/byzantine-coordination.js
const crypto = require('crypto');

module.exports = {
  preTask: async (context) => {
    // Initialize Byzantine cluster before task execution
    await context.exec('npx claude-flow@alpha swarm init --consensus byzantine');

    // Generate keypairs for all agents
    for (let i = 1; i <= 7; i++) {
      await context.exec(`npx claude-flow@alpha byzantine generate-keypair \
        --agent-name bft-agent-${i}`);
    }

    // Wait for quorum formation
    await context.waitFor('quorum-formed', { timeout: 10000 });
  },

  onTransactionProposed: async (context, transaction) => {
    // Sign transaction with agent's private key
    const signature = await context.sign(transaction, context.agent.privateKey);

    // Broadcast signed transaction to all replicas
    await context.broadcast({
      type: 'PRE-PREPARE',
      transaction,
      signature,
      view: context.currentView,
      sequence: context.nextSequence++
    });
  },

  onMaliciousDetected: async (context, agent) => {
    // Log malicious agent detection
    console.error(`Malicious agent detected: ${agent.id} (${agent.faultType})`);

    // Quarantine agent
    await context.exec(`npx claude-flow@alpha byzantine quarantine \
      --agent ${agent.id} \
      --duration 3600000`);

    // Spawn replacement agent
    await context.exec(`npx claude-flow@alpha agent spawn \
      --type validator \
      --name replacement-${Date.now()}`);

    // Store event in Memory MCP
    await context.memory.store(`byzantine/malicious/${agent.id}`, {
      timestamp: new Date().toISOString(),
      faultType: agent.faultType,
      evidence: agent.evidence
    });
  },

  onViewChange: async (context, newPrimary) => {
    // Update primary reference
    context.currentPrimary = newPrimary;
    context.currentView++;

    // Persist view change
    await context.memory.store('byzantine/current-view', {
      view: context.currentView,
      primary: newPrimary,
      timestamp: new Date().toISOString()
    });
  }
};
```

## Performance Metrics

### Baseline Performance (No Byzantine Agents)

- **Consensus latency**: 85ms average (3 message rounds: PRE-PREPARE → PREPARE → COMMIT)
- **Throughput**: 470 transactions/sec (cryptographic overhead)
- **Signature verification**: 2ms per signature × 7 agents = 14ms
- **Message complexity**: O(N²) = 49 messages per transaction (7 agents × 7 broadcasts)

### Byzantine Scenario (2 Malicious Agents)

- **Detection latency**: 120ms average (1-2 consensus rounds)
- **False positive rate**: 0.0% (cryptographic verification)
- **Transactions rejected**: 23/1000 (2.3%) from malicious agents
- **System availability**: 100% (5/7 healthy agents exceed quorum)
- **View change time**: 4.8 seconds (timeout + election + catch-up)

### Scalability Analysis

| Agents | F (max failures) | Quorum (2F+1) | Throughput | Latency |
|--------|------------------|---------------|-----------|---------|
| 4      | 1                | 3             | 1,200/sec | 40ms    |
| 7      | 2                | 5             | 470/sec   | 85ms    |
| 10     | 3                | 7             | 180/sec   | 150ms   |
| 13     | 4                | 9             | 90/sec    | 250ms   |

**Observation**: O(N²) message complexity limits scalability (Byzantine for <20 agents).

## Troubleshooting

### Problem: Quorum Not Met

**Symptom**: Transactions not committing (stuck in PRE-PREPARE phase)

**Root Cause**: Too many agents offline or malicious (≥F+1)

**Solution**:
```bash
# Check agent health
npx claude-flow@alpha agent list --filter active

# Count healthy agents
HEALTHY=$(npx claude-flow@alpha byzantine health --json | jq '.healthy_agents')

# Ensure HEALTHY >= 2F+1
# If not, spawn replacement agents
if [ "$HEALTHY" -lt 5 ]; then
  npx claude-flow@alpha agent spawn --count $((5 - HEALTHY)) --type validator
fi
```

### Problem: High False Positive Rate

**Symptom**: Legitimate agents quarantined incorrectly

**Root Cause**: Clock skew or network delays causing signature mismatches

**Solution**:
```bash
# Synchronize clocks across all agents
npx claude-flow@alpha byzantine sync-clocks --ntp-server pool.ntp.org

# Increase verification threshold (allow minor discrepancies)
npx claude-flow@alpha byzantine config --verification-threshold 0.65

# Enable signature timestamp tolerance
npx claude-flow@alpha byzantine config --signature-tolerance 5000
```

### Problem: Slow View Change

**Symptom**: View change taking >10 seconds

**Root Cause**: High view-change-timeout or slow network

**Solution**:
```bash
# Decrease view-change-timeout
npx claude-flow@alpha byzantine config --view-change-timeout 3000

# Enable fast view change (optimistic)
npx claude-flow@alpha byzantine config --fast-view-change true

# Monitor network latency
npx claude-flow@alpha network latency --all-agents
```

## Key Takeaways

1. **3F+1 redundancy**: Tolerate F Byzantine failures (7 agents for F=2)
2. **Cryptographic verification**: Ed25519 signatures prevent forgery
3. **Quorum consensus**: 2F+1 agreement required (5/7 for F=2)
4. **Malicious detection**: Automatic quarantine of faulty agents
5. **View changes**: Replace malicious primary within 5 seconds
6. **Strong consistency**: No double-spending or conflicting transactions
7. **Limited scalability**: O(N²) messages restrict to <20 agents

## Comparison with RAFT and Gossip

| Metric                | RAFT (Example 1) | Gossip (Example 2) | Byzantine (Example 3) |
|-----------------------|------------------|-------------------|---------------------|
| **Agents**            | 5                | 150               | 7                   |
| **Consistency**       | Strong           | Eventual          | Strong              |
| **Fault Tolerance**   | Crash failures   | Crash failures    | Malicious failures  |
| **Throughput**        | 2,500 ops/sec    | 8,333 ops/sec     | 470 ops/sec         |
| **Latency**           | 25ms             | 700ms             | 85ms                |
| **Security**          | Basic            | None              | Cryptographic       |
| **Scalability**       | <20 agents       | 1,000+ agents     | <20 agents          |
| **Use Case**          | Critical tasks   | Scalable caching  | Financial systems   |

## Next Steps

- **Reference**: [Coordination Strategies](../references/coordination-strategies.md) for Byzantine deep dive
- **Reference**: [Fault Tolerance](../references/fault-tolerance.md) for malicious detection patterns
- **Reference**: [Workflow Diagram](../graphviz/workflow.dot) for Byzantine message flow visualization
