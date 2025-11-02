# Coordination Strategies - In-Depth Comparison

## Overview

This reference provides comprehensive analysis of three primary distributed coordination strategies: RAFT consensus, Gossip protocols, and Byzantine Fault Tolerance (BFT). Each strategy offers different trade-offs between consistency, fault tolerance, scalability, and security.

## RAFT Consensus Protocol

### Design Philosophy

RAFT is designed for **understandability and strong consistency** in crash-fault-tolerant systems. It uses leader-based coordination to ensure all agents maintain identical replicated logs.

### Core Components

#### 1. Leader Election

**Mechanism**: Randomized election timeouts prevent split votes

```
Election Timeout: 150-300ms (randomized)
Heartbeat Interval: 50ms (leader → followers)

State Machine:
  Follower → (timeout) → Candidate → (majority votes) → Leader
  Leader → (higher term detected) → Follower
  Candidate → (split vote) → Candidate (new election)
```

**Vote Request**:
```json
{
  "term": 5,
  "candidate_id": "agent-3",
  "last_log_index": 1000,
  "last_log_term": 4
}
```

**Vote Response**:
```json
{
  "term": 5,
  "vote_granted": true,
  "voter_id": "agent-2"
}
```

**Election Guarantees**:
- **Safety**: At most one leader per term
- **Liveness**: Eventually elects a leader (if majority available)
- **Log matching**: Leader has all committed entries

#### 2. Log Replication

**Mechanism**: Leader appends entries to followers, commits when majority replicated

```
Leader:
  1. Receive client request
  2. Append entry to local log
  3. Broadcast AppendEntries RPC to followers
  4. Wait for majority acknowledgment
  5. Commit entry (apply to state machine)
  6. Notify followers of commit index

Follower:
  1. Receive AppendEntries RPC
  2. Check term and log consistency
  3. Append entries to local log
  4. Acknowledge to leader
  5. Apply committed entries to state machine
```

**AppendEntries RPC**:
```json
{
  "term": 5,
  "leader_id": "agent-1",
  "prev_log_index": 999,
  "prev_log_term": 4,
  "entries": [
    {"term": 5, "command": "SET key=value"}
  ],
  "leader_commit": 999
}
```

**Consistency Guarantees**:
- **Log Matching Property**: If two logs have same index/term, they're identical up to that point
- **Leader Append-Only**: Leader never overwrites or deletes log entries
- **Leader Completeness**: All committed entries present in future leaders

#### 3. Safety Properties

**Theorem 1 (Election Safety)**: At most one leader per term

**Proof**: Majority quorum ensures two candidates cannot both receive N/2+1 votes

**Theorem 2 (Leader Append-Only)**: Leader never overwrites log entries

**Proof**: Leader only appends entries, never modifies past entries

**Theorem 3 (Log Matching)**: If two entries have same index/term, logs are identical up to that index

**Proof**: Induction on log index with consistency check in AppendEntries

**Theorem 4 (Leader Completeness)**: If entry committed in term T, present in all leaders for term >T

**Proof**: Voting restriction ensures candidates with incomplete logs cannot win

**Theorem 5 (State Machine Safety)**: If agent applies log entry at index i, no other agent applies different entry at i

**Proof**: Follows from Log Matching and Leader Completeness

### Performance Characteristics

#### Commit Latency

```
Best case (local cluster):
  - Leader → Followers: 1ms (RPC)
  - Followers → Leader: 1ms (ACK)
  - Total: 2ms (1 RTT)

Typical case (datacenter):
  - Leader → Followers: 10ms
  - Followers → Leader: 10ms
  - Total: 20ms (1 RTT)

Worst case (cross-region):
  - Leader → Followers: 50ms
  - Followers → Leader: 50ms
  - Total: 100ms (1 RTT)
```

#### Throughput Analysis

```
Single leader bottleneck:
  - CPU-bound: ~10,000 ops/sec (signature verification)
  - Network-bound: ~5,000 ops/sec (1 Gbps network)
  - Disk-bound: ~1,000 ops/sec (HDDs, fsync)

Parallelization:
  - Batching: Group N entries in single AppendEntries (10x throughput)
  - Pipelining: Send next batch before previous commits (2-3x throughput)
  - Read replicas: Route reads to followers (10x read throughput)
```

#### Failure Recovery

```
Leader failure:
  - Detection time: 150-300ms (election timeout)
  - Election time: 150-300ms (randomized timeout)
  - Catch-up time: O(uncommitted entries)
  - Total: 300-600ms + catch-up

Follower failure:
  - Detection time: 50ms (heartbeat timeout)
  - Impact: None (majority still reachable)
  - Recovery: Replay log from leader (O(log length))

Network partition (minority):
  - Detection time: 150-300ms
  - Mode: Read-only (cannot commit)
  - Recovery: Rejoin majority, revert uncommitted entries
```

### Use Cases

**Ideal for**:
- State machine replication (e.g., replicated key-value stores)
- Configuration management (e.g., Kubernetes control plane)
- Coordination services (e.g., leader election, distributed locks)
- Critical workflows requiring strong consistency

**Not suitable for**:
- Large-scale systems (>20 nodes, leader bottleneck)
- High-throughput write workloads (single leader limit)
- Geo-distributed systems (cross-region latency)
- Systems requiring partition tolerance over consistency

### Implementation Considerations

**Clock Skew**:
- RAFT does not depend on synchronized clocks
- Terms provide logical clock (monotonically increasing)
- Election timeouts randomized to prevent split votes

**Log Compaction**:
- Snapshots: Compact committed entries into snapshot
- Snapshot transfer: Send snapshot to slow followers
- Garbage collection: Delete log entries before snapshot

**Membership Changes**:
- Joint consensus: Transition through intermediate configuration
- Safety: Majority of both old and new configurations must agree
- Single-server changes: Add/remove one server at a time

---

## Gossip Protocol

### Design Philosophy

Gossip (epidemic) protocols are designed for **high scalability and partition tolerance** through peer-to-peer state propagation with eventual consistency.

### Core Components

#### 1. Epidemic Spread

**Mechanism**: Each agent periodically gossips to N random peers

```
Gossip Round:
  1. Select N random peers (fanout)
  2. Send current state digest to peers
  3. Receive state digests from peers
  4. Exchange missing state (anti-entropy)
  5. Update local state with new information

Convergence: O(log N) rounds
Message Complexity: O(N log N) total messages
```

**Gossip Message**:
```json
{
  "agent_id": "agent-42",
  "state_digest": {
    "key1": {"value": "v1", "version": 5, "timestamp": "2025-11-02T12:00:00Z"},
    "key2": {"value": "v2", "version": 3, "timestamp": "2025-11-02T11:59:00Z"}
  },
  "round": 7
}
```

#### 2. State Reconciliation

**Push**: Send state to peers (active dissemination)
```javascript
push(peer, state) {
  peer.receive(state);
  peer.merge(state, local_state);
}
```

**Pull**: Request state from peers (passive dissemination)
```javascript
pull(peer) {
  remote_state = peer.get_state();
  merge(remote_state, local_state);
}
```

**Push-Pull**: Bidirectional exchange (fastest convergence)
```javascript
push_pull(peer) {
  peer.receive(local_state);
  remote_state = peer.get_state();
  merge(remote_state, local_state);
}
```

#### 3. Conflict Resolution

**Last-Write-Wins (LWW)**:
```javascript
merge(remote, local) {
  for (key in remote) {
    if (!local.has(key) || remote[key].timestamp > local[key].timestamp) {
      local[key] = remote[key]; // LWW
    }
  }
}
```

**Vector Clocks**:
```javascript
merge(remote, local) {
  for (key in remote) {
    if (vector_clock_greater(remote[key].vc, local[key].vc)) {
      local[key] = remote[key]; // Causally later
    } else if (concurrent(remote[key].vc, local[key].vc)) {
      local[key] = application_merge(remote[key], local[key]); // Conflict
    }
  }
}
```

**CRDTs (Conflict-free Replicated Data Types)**:
```javascript
// G-Counter (grow-only counter)
merge(remote, local) {
  for (agent_id in remote.counters) {
    local.counters[agent_id] = max(
      local.counters[agent_id],
      remote.counters[agent_id]
    );
  }
}

value() {
  return sum(local.counters.values());
}
```

### Performance Characteristics

#### Convergence Analysis

**Theorem (Epidemic Convergence)**: With fanout F, convergence occurs in O(log_F N) rounds

**Proof Sketch**:
```
Round 0: 1 agent knows update
Round 1: F agents know update (fanout)
Round 2: F² agents know update
...
Round k: F^k agents know update

Convergence when F^k ≥ N
k = log_F(N) = log(N) / log(F)
```

**Example (150 agents, fanout=3)**:
```
Round 1: 1 → 3 agents (3¹ = 3)
Round 2: 3 → 9 agents (3² = 9)
Round 3: 9 → 27 agents (3³ = 27)
Round 4: 27 → 81 agents (3⁴ = 81)
Round 5: 81 → 150 agents (3⁵ = 243, capped at 150)

Total: 5 rounds = log₃(150) ≈ 4.6
```

#### Throughput Analysis

```
Fully distributed:
  - No coordinator bottleneck
  - Aggregate throughput: N × local_throughput
  - Example: 150 agents × 100 ops/sec = 15,000 ops/sec

Gossip overhead:
  - Fanout × round_frequency × state_size
  - Example: 3 peers × 10 rounds/sec × 1 KB = 30 KB/sec per agent
  - Total network: 150 agents × 30 KB/sec = 4.5 MB/sec
```

#### Partition Tolerance

```
Network partition:
  - Detection: Automatic (agents in partition only gossip to each other)
  - Mode: Continue operating (eventual consistency allows divergence)
  - Conflicts: Resolved via LWW/vector clocks/CRDTs

Partition healing:
  - Detection: Agents reconnect, resume gossiping
  - Convergence: O(log N) rounds (same as initial convergence)
  - Time: ~1-10 seconds (depends on fanout/interval)
```

### Use Cases

**Ideal for**:
- Distributed caching (e.g., Redis Cluster, Memcached)
- Service discovery (e.g., Consul, Serf)
- Monitoring/alerting (e.g., Prometheus federation)
- Large-scale metadata propagation (e.g., Cassandra hints)

**Not suitable for**:
- Strong consistency requirements (e.g., financial transactions)
- Low-latency critical operations (eventual consistency delay)
- Small clusters (<10 nodes, RAFT more efficient)
- Workloads intolerant to temporary inconsistencies

### Implementation Considerations

**Fanout Selection**:
- Low fanout (F=2): Slower convergence, lower overhead
- High fanout (F=5): Faster convergence, higher overhead
- Optimal: F=3 balances speed and bandwidth

**Gossip Interval**:
- Frequent (50ms): Fast convergence, high CPU/network
- Infrequent (500ms): Slow convergence, low CPU/network
- Optimal: 100-200ms balances latency and resources

**State Size**:
- Large state: Use delta updates (only send changes)
- Very large state: Use Merkle trees (efficient reconciliation)
- Unbounded state: Use bloom filters (probabilistic membership)

---

## Byzantine Fault Tolerance

### Design Philosophy

BFT is designed for **security-critical systems** that must tolerate malicious or arbitrarily faulty agents through cryptographic verification and quorum-based consensus.

### Core Components

#### 1. PBFT (Practical Byzantine Fault Tolerance)

**Phases**: Three-phase commit protocol

```
Phase 1 (PRE-PREPARE):
  Primary broadcasts <PRE-PREPARE, v, n, m>
  v = view number
  n = sequence number
  m = message (client request)

Phase 2 (PREPARE):
  Replicas broadcast <PREPARE, v, n, d, i>
  d = digest of m
  i = replica id
  Wait for 2F PREPARE messages (quorum)

Phase 3 (COMMIT):
  Replicas broadcast <COMMIT, v, n, d, i>
  Wait for 2F+1 COMMIT messages (quorum)
  Execute request, reply to client
```

**Message Flow**:
```
Client → Primary:   REQUEST
Primary → Replicas: PRE-PREPARE (1 message)
Replicas → All:     PREPARE (N messages)
Replicas → All:     COMMIT (N messages)
Replicas → Client:  REPLY (N messages)

Total messages: 1 + N + N + N = 3N+1 = O(N)
```

#### 2. Cryptographic Verification

**Digital Signatures (Ed25519)**:
```javascript
// Signing
const message = {view: 1, sequence: 100, digest: "0xabcd..."};
const signature = ed25519.sign(message, privateKey);

// Verification
const valid = ed25519.verify(message, signature, publicKey);
if (!valid) {
  quarantine(agent); // Malicious agent detected
}
```

**Message Authentication Codes (MAC)**:
```javascript
// Faster than signatures (10x), but requires pairwise keys
const mac = hmac_sha256(message, shared_key_with_replica_i);

// Verification
const valid = hmac_sha256(message, shared_key) === mac;
```

#### 3. View Changes (Leader Replacement)

**Trigger**: Replicas timeout waiting for primary (censorship attack)

```
View Change Protocol:
  1. Replica i suspects primary (timeout)
  2. Broadcast VIEW-CHANGE message
  3. Wait for 2F+1 VIEW-CHANGE messages
  4. New primary (i mod N) broadcasts NEW-VIEW
  5. Resume normal operation in new view
```

**VIEW-CHANGE Message**:
```json
{
  "new_view": 2,
  "replica_id": "agent-3",
  "checkpoint": 100,
  "prepared_requests": [
    {"sequence": 101, "digest": "0x1234..."},
    {"sequence": 102, "digest": "0x5678..."}
  ],
  "signature": "0xabcd..."
}
```

#### 4. Malicious Detection

**Detection Methods**:

1. **Signature Mismatch**:
```javascript
if (!verify_signature(message, agent.public_key)) {
  report_malicious(agent, "signature-mismatch");
}
```

2. **Conflicting Messages**:
```javascript
if (message1.sequence === message2.sequence && message1.digest !== message2.digest) {
  report_malicious(agent, "equivocation");
}
```

3. **Invalid State Transitions**:
```javascript
if (!validate_state_transition(old_state, new_state)) {
  report_malicious(agent, "invalid-transition");
}
```

4. **Censorship (Liveness Attack)**:
```javascript
if (time_since_last_progress() > view_change_timeout) {
  trigger_view_change(); // Suspected malicious primary
}
```

### Performance Characteristics

#### Latency Analysis

```
Best case (no Byzantine agents):
  - PRE-PREPARE: 1ms
  - PREPARE: 1ms (parallel broadcasts)
  - COMMIT: 1ms (parallel broadcasts)
  - Total: 3ms (3 RTT)

Typical case (cryptographic overhead):
  - PRE-PREPARE: 5ms
  - PREPARE: 10ms (signature verification × N)
  - COMMIT: 10ms (signature verification × N)
  - Total: 25ms

Worst case (view change):
  - Primary timeout: 5,000ms
  - View change protocol: 3 RTT ≈ 15ms
  - Total: 5,015ms
```

#### Throughput Analysis

```
Cryptographic bottleneck:
  - Signature generation: 50 µs
  - Signature verification: 100 µs
  - Per transaction: N verifications = 700 µs (7 agents)
  - Throughput: 1 / 700 µs ≈ 1,400 ops/sec (single core)

Message complexity:
  - O(N²) messages per request (N replicas broadcast to N replicas)
  - Example: 7 agents → 49 messages per request
  - Network bottleneck: 1 Gbps / (49 × 1 KB) ≈ 2,500 ops/sec
```

#### Scalability Limits

```
Practical limits:
  - <10 agents: Good throughput (1,000+ ops/sec)
  - 10-20 agents: Moderate throughput (500 ops/sec)
  - >20 agents: Poor throughput (<100 ops/sec)

Reason: O(N²) message complexity dominates
```

### Use Cases

**Ideal for**:
- Financial systems (blockchain, payment networks)
- Critical infrastructure (power grids, air traffic control)
- Untrusted environments (multi-organization coordination)
- Security-critical applications (authentication, authorization)

**Not suitable for**:
- Large-scale systems (>20 nodes, O(N²) overhead)
- High-throughput workloads (cryptographic bottleneck)
- Low-latency applications (3-phase commit delay)
- Trusted environments (RAFT more efficient)

### Implementation Considerations

**Cryptographic Trade-offs**:
- **Signatures (Ed25519)**: Universal verification, slow (100 µs)
- **MACs (HMAC)**: Pairwise keys, fast (10 µs), O(N²) keys
- **Threshold Signatures**: Best of both worlds, complex

**Quorum Size**:
- **3F+1 total agents**: Tolerate F Byzantine failures
- **2F+1 quorum**: Majority agreement required
- **Example**: 7 agents (F=2), 5 quorum

**View Change Optimization**:
- **Fast view change**: Skip checkpoint verification (optimistic)
- **Pipelining**: Process next request while committing previous
- **Batching**: Group multiple requests in single consensus instance

---

## Comparison Matrix

| Dimension | RAFT | Gossip | Byzantine |
|-----------|------|--------|-----------|
| **Consistency** | Strong | Eventual | Strong |
| **Fault Model** | Crash | Crash | Byzantine (malicious) |
| **Fault Tolerance** | N/2+1 failures | Very high | F failures (3F+1 total) |
| **Coordination** | Leader-based | Peer-to-peer | Primary-based |
| **Message Complexity** | O(N) | O(N log N) | O(N²) |
| **Latency** | Low (1-2 RTT) | High (O(log N)) | Medium (3-5 RTT) |
| **Throughput** | Moderate (1,000-10,000) | High (10,000-100,000) | Low (100-1,000) |
| **Scalability** | 5-20 agents | 100-1,000+ agents | 4-20 agents |
| **Security** | Basic | None | Cryptographic |
| **Partition Tolerance** | Majority partition only | High (auto-healing) | Majority partition only |
| **Implementation Complexity** | Moderate | Simple | High |

## Decision Tree

```
Do you need protection against malicious agents?
├── YES → Byzantine Fault Tolerance
└── NO → Do you need strong consistency?
    ├── YES → RAFT Consensus
    └── NO → Do you need high scalability (>50 agents)?
        ├── YES → Gossip Protocol
        └── NO → RAFT Consensus
```

## Hybrid Approaches

### RAFT + Gossip

**Use Case**: Strong consistency for critical data, eventual consistency for metadata

```
Critical path (RAFT):
  - User transactions
  - Configuration changes
  - Leader elections

Metadata path (Gossip):
  - Agent health status
  - Performance metrics
  - Cache invalidation hints
```

### Byzantine + Gossip

**Use Case**: Byzantine consensus for validation, gossip for dissemination

```
Validation layer (Byzantine):
  - Validate transactions (BFT consensus)
  - Commit to replicated log

Dissemination layer (Gossip):
  - Propagate validated transactions to all agents
  - Eventual consistency for reads
```

## References

- **RAFT**: [In Search of an Understandable Consensus Algorithm](https://raft.github.io/raft.pdf)
- **Gossip**: [Epidemic Algorithms for Replicated Database Maintenance](https://dl.acm.org/doi/10.1145/41840.41841)
- **Byzantine**: [Practical Byzantine Fault Tolerance](http://pmg.csail.mit.edu/papers/osdi99.pdf)
- **CRDTs**: [A comprehensive study of CRDTs](https://hal.inria.fr/inria-00555588/document)

---

**Last Updated**: 2025-11-02
**Author**: Claude Code Advanced Coordination Skill
