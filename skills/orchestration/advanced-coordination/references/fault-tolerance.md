# Fault Tolerance Patterns - Failure Detection and Recovery

## Overview

This reference provides comprehensive patterns for implementing fault tolerance in distributed multi-agent systems, including failure detection mechanisms, recovery strategies, and network partition healing protocols.

## Table of Contents

1. [Failure Detection](#failure-detection)
2. [Recovery Mechanisms](#recovery-mechanisms)
3. [Network Partition Handling](#network-partition-handling)
4. [Split-Brain Prevention](#split-brain-prevention)
5. [Health Monitoring](#health-monitoring)
6. [Self-Healing Patterns](#self-healing-patterns)

---

## Failure Detection

### Heartbeat-Based Detection

**Mechanism**: Periodic liveness messages between agents

```javascript
// Leader sends heartbeats
setInterval(() => {
  followers.forEach(follower => {
    follower.send({type: 'HEARTBEAT', leader_id, term, timestamp: Date.now()});
  });
}, HEARTBEAT_INTERVAL); // 50ms

// Follower detects leader failure
if (Date.now() - last_heartbeat > ELECTION_TIMEOUT) {
  transition_to_candidate(); // 150-300ms
  start_election();
}
```

**Parameters**:
- **Heartbeat interval**: 50ms (typical), 10-100ms (range)
- **Timeout threshold**: 150-300ms (3-6 missed heartbeats)
- **False positive rate**: <1% (randomized timeouts prevent split votes)

**Trade-offs**:
- ✅ Simple to implement
- ✅ Low overhead (small messages)
- ⚠️ Assumes synchronized clocks (or relative timeouts)
- ❌ Cannot detect Byzantine failures (malicious agents can send heartbeats)

### Phi Accrual Failure Detector

**Mechanism**: Probabilistic failure detection based on heartbeat arrival distribution

```javascript
class PhiAccrualDetector {
  constructor() {
    this.arrival_intervals = []; // Sliding window of heartbeat intervals
    this.threshold = 8.0; // Phi threshold for suspicion
  }

  heartbeat_received(timestamp) {
    const interval = timestamp - this.last_heartbeat;
    this.arrival_intervals.push(interval);

    if (this.arrival_intervals.length > 100) {
      this.arrival_intervals.shift(); // Keep last 100 samples
    }

    this.last_heartbeat = timestamp;
  }

  phi_value() {
    const now = Date.now();
    const elapsed = now - this.last_heartbeat;

    const mean = this.mean(this.arrival_intervals);
    const stddev = this.stddev(this.arrival_intervals);

    // Cumulative distribution function (normal distribution)
    const phi = -Math.log10(1 - this.cdf(elapsed, mean, stddev));
    return phi;
  }

  is_failed() {
    return this.phi_value() > this.threshold; // Phi > 8 → 99.9% confidence
  }
}
```

**Advantages**:
- ✅ Adaptive to network conditions (adjusts based on observed intervals)
- ✅ Probabilistic confidence (Phi value = -log10(P(failure)))
- ✅ No fixed timeout (continuous suspicion level)

**Phi Value Interpretation**:
```
Phi < 1:  Low suspicion (0-90% confidence)
Phi 1-5:  Moderate suspicion (90-99.999% confidence)
Phi 5-8:  High suspicion (99.999-99.999999% confidence)
Phi > 8:  Very high suspicion (>99.999999% confidence)
```

### Gossip-Based Failure Detection

**Mechanism**: Agents periodically gossip membership information

```javascript
class GossipFailureDetector {
  constructor() {
    this.membership = new Map(); // agent_id → {heartbeat_counter, timestamp}
    this.gossip_interval = 100; // ms
  }

  increment_heartbeat() {
    this.membership.get(this.agent_id).heartbeat_counter++;
  }

  gossip() {
    const peer = this.select_random_peer();
    peer.send({type: 'GOSSIP', membership: this.membership});
  }

  receive_gossip(remote_membership) {
    remote_membership.forEach((info, agent_id) => {
      const local_info = this.membership.get(agent_id);

      if (!local_info || info.heartbeat_counter > local_info.heartbeat_counter) {
        // Update with newer information
        this.membership.set(agent_id, {
          heartbeat_counter: info.heartbeat_counter,
          timestamp: Date.now()
        });
      }
    });

    // Detect failures (stale heartbeats)
    this.membership.forEach((info, agent_id) => {
      if (Date.now() - info.timestamp > FAILURE_TIMEOUT) {
        this.mark_failed(agent_id);
      }
    });
  }
}
```

**Advantages**:
- ✅ Decentralized (no single point of failure)
- ✅ Scalable (O(log N) convergence)
- ✅ Partition-tolerant (eventually consistent)

**Disadvantages**:
- ❌ Eventual detection (not immediate)
- ❌ False positives during partitions

---

## Recovery Mechanisms

### Automatic Failover (RAFT)

**Mechanism**: Elect new leader when current leader fails

```javascript
class RaftAgent {
  on_leader_failure() {
    // 1. Transition to candidate state
    this.state = 'candidate';
    this.current_term++;
    this.voted_for = this.agent_id;

    // 2. Request votes from all agents
    this.send_vote_requests();

    // 3. Set election timeout (randomized)
    this.election_timeout = 150 + Math.random() * 150; // 150-300ms

    setTimeout(() => {
      if (this.state === 'candidate') {
        this.start_election(); // Retry if no winner
      }
    }, this.election_timeout);
  }

  receive_vote(vote) {
    this.votes_received++;

    if (this.votes_received >= this.quorum_size) {
      // 4. Transition to leader
      this.state = 'leader';
      this.send_heartbeats();

      // 5. Replay uncommitted log entries
      this.replay_log();
    }
  }

  replay_log() {
    const uncommitted = this.log.filter(e => e.index > this.commit_index);
    uncommitted.forEach(entry => {
      this.replicate_entry(entry); // Re-propose to followers
    });
  }
}
```

**Recovery Time**:
```
Detection:   150-300ms (election timeout)
Election:    150-300ms (randomized timeout)
Catch-up:    O(uncommitted_entries)
Total:       300-600ms + catch-up
```

### State Transfer (Catch-up)

**Mechanism**: Recover failed agent by replaying committed log

```javascript
class RaftLeader {
  recover_follower(follower) {
    const follower_index = follower.match_index; // Last known index
    const leader_index = this.log.length - 1;

    if (leader_index - follower_index > SNAPSHOT_THRESHOLD) {
      // Use snapshot for large gaps
      this.send_snapshot(follower);
    } else {
      // Replay missing log entries
      const missing = this.log.slice(follower_index + 1);
      this.send_append_entries(follower, missing);
    }
  }

  send_snapshot(follower) {
    const snapshot = {
      last_included_index: this.last_snapshot_index,
      last_included_term: this.last_snapshot_term,
      data: this.state_machine.serialize()
    };

    follower.send({type: 'INSTALL_SNAPSHOT', snapshot});
  }
}
```

**Optimization**:
- **Incremental snapshots**: Transfer only delta since last snapshot
- **Parallel recovery**: Recover multiple followers concurrently
- **Rate limiting**: Avoid overwhelming recovering agent

### Anti-Entropy (Gossip)

**Mechanism**: Periodically synchronize full state between peers

```javascript
class GossipAgent {
  anti_entropy() {
    setInterval(() => {
      const peer = this.select_random_peer();

      // 1. Send full state to peer
      peer.send({type: 'ANTI_ENTROPY', state: this.full_state()});

      // 2. Receive full state from peer
      const peer_state = peer.request_state();

      // 3. Merge states (conflict resolution)
      this.merge_state(peer_state);
    }, ANTI_ENTROPY_INTERVAL); // 10 seconds
  }

  merge_state(peer_state) {
    Object.keys(peer_state).forEach(key => {
      const local_value = this.state.get(key);
      const peer_value = peer_state[key];

      if (!local_value || peer_value.timestamp > local_value.timestamp) {
        // Last-write-wins
        this.state.set(key, peer_value);
      }
    });
  }
}
```

**Recovery Time**:
```
Detection:   O(gossip_rounds) = O(log N) × interval
Recovery:    1 anti-entropy round (immediate)
Total:       1-10 seconds (depends on interval)
```

### Checkpointing (Byzantine)

**Mechanism**: Periodically create stable checkpoints of committed state

```javascript
class ByzantineAgent {
  create_checkpoint() {
    const checkpoint = {
      sequence_number: this.last_executed,
      state_digest: this.hash(this.state),
      proof: [] // 2F+1 checkpoint signatures
    };

    // Broadcast checkpoint to all replicas
    this.broadcast({type: 'CHECKPOINT', checkpoint});
  }

  receive_checkpoint_proof(proof) {
    if (proof.signatures.length >= this.quorum_size) {
      // Valid checkpoint: garbage collect log before checkpoint
      this.log = this.log.filter(e => e.sequence > proof.sequence_number);
      this.stable_checkpoint = proof;
    }
  }

  recover_from_checkpoint() {
    // Request checkpoint from 2F+1 replicas
    const checkpoints = this.request_checkpoints();

    // Verify checkpoint authenticity (2F+1 matching signatures)
    const valid_checkpoint = this.verify_checkpoint_quorum(checkpoints);

    // Restore state from checkpoint
    this.state = valid_checkpoint.state;
    this.last_executed = valid_checkpoint.sequence_number;

    // Replay log entries after checkpoint
    this.replay_log(valid_checkpoint.sequence_number);
  }
}
```

---

## Network Partition Handling

### Partition Detection

#### Method 1: Quorum Loss (RAFT/Byzantine)

```javascript
class RaftAgent {
  detect_partition() {
    const reachable = this.followers.filter(f => f.is_reachable());

    if (reachable.length < this.quorum_size - 1) {
      // Lost majority: enter degraded mode
      this.enter_degraded_mode();
      return true;
    }

    return false;
  }

  enter_degraded_mode() {
    this.state = 'follower'; // Step down if leader
    this.read_only = true; // Reject writes

    console.warn('Partition detected: operating in read-only mode');
  }
}
```

#### Method 2: Gossip Convergence (Gossip)

```javascript
class GossipAgent {
  detect_partition() {
    const expected_agents = this.membership.size;
    const reachable_agents = this.count_reachable();

    const reachability_ratio = reachable_agents / expected_agents;

    if (reachability_ratio < 0.5) {
      // Likely partition: 50% agents unreachable
      console.warn(`Partition suspected: ${reachable_agents}/${expected_agents} reachable`);
      return true;
    }

    return false;
  }
}
```

### Partition Strategies

#### 1. Majority Partition (RAFT/Byzantine)

**Strategy**: Only majority partition continues operating

```javascript
class RaftAgent {
  handle_partition() {
    const reachable = this.count_reachable();

    if (reachable >= this.quorum_size) {
      // Majority partition: continue normal operation
      this.continue_operation();
    } else {
      // Minority partition: enter read-only mode
      this.enter_degraded_mode();

      // Wait for partition healing
      this.wait_for_healing();
    }
  }

  wait_for_healing() {
    setInterval(() => {
      if (this.count_reachable() >= this.quorum_size) {
        // Partition healed: rejoin cluster
        this.exit_degraded_mode();
        this.sync_state_from_leader();
      }
    }, 1000);
  }
}
```

#### 2. Eventual Consistency (Gossip)

**Strategy**: Both partitions continue operating, resolve conflicts later

```javascript
class GossipAgent {
  handle_partition() {
    // Continue accepting writes in both partitions
    this.continue_operation();

    // Mark partition timestamp for conflict resolution
    this.partition_start = Date.now();
  }

  heal_partition() {
    // Resume gossiping across partition boundary
    const other_partition = this.discover_other_partition();

    // Exchange states and resolve conflicts
    other_partition.forEach(agent => {
      const remote_state = agent.get_state();
      this.merge_with_conflict_resolution(remote_state);
    });
  }

  merge_with_conflict_resolution(remote_state) {
    Object.keys(remote_state).forEach(key => {
      const local = this.state.get(key);
      const remote = remote_state[key];

      if (!local) {
        // No conflict: accept remote value
        this.state.set(key, remote);
      } else if (local.timestamp === remote.timestamp && local.value !== remote.value) {
        // Concurrent updates during partition: resolve conflict
        const resolved = this.application_merge(local, remote);
        this.state.set(key, resolved);
      } else {
        // Last-write-wins
        const winner = (remote.timestamp > local.timestamp) ? remote : local;
        this.state.set(key, winner);
      }
    });
  }
}
```

### Partition Healing Protocols

#### RAFT Log Reconciliation

```javascript
class RaftLeader {
  heal_partition() {
    // Agents from minority partition rejoin
    const rejoining = this.detect_rejoining_agents();

    rejoining.forEach(agent => {
      // 1. Check agent's last log index
      const agent_index = agent.last_log_index;
      const leader_index = this.log.length - 1;

      // 2. Find divergence point
      let divergence = agent_index;
      while (divergence >= 0 && this.log[divergence].term !== agent.log[divergence].term) {
        divergence--;
      }

      // 3. Revert agent's uncommitted entries after divergence
      agent.truncate_log(divergence);

      // 4. Replay leader's committed entries
      const missing = this.log.slice(divergence + 1);
      this.send_append_entries(agent, missing);
    });
  }
}
```

#### Gossip Anti-Entropy

```javascript
class GossipAgent {
  heal_partition() {
    // Force immediate anti-entropy with all agents
    this.membership.forEach(agent => {
      if (agent.is_reachable()) {
        const remote_state = agent.get_full_state();
        this.merge_state(remote_state);
      }
    });

    // Increase gossip frequency temporarily (faster convergence)
    this.gossip_interval = 50; // 50ms for 5 seconds
    setTimeout(() => {
      this.gossip_interval = 100; // Restore normal interval
    }, 5000);
  }
}
```

---

## Split-Brain Prevention

### Quorum-Based Prevention

**Mechanism**: Require majority agreement for all operations

```javascript
class RaftLeader {
  commit_entry(entry) {
    // 1. Replicate to all followers
    const acks = this.replicate_to_followers(entry);

    // 2. Wait for quorum acknowledgments
    if (acks.length >= this.quorum_size - 1) {
      // Quorum met: commit entry
      this.commit_index = entry.index;
      this.apply_to_state_machine(entry);
      return true;
    } else {
      // Quorum not met: reject commit
      console.warn('Quorum not met: entry not committed');
      return false;
    }
  }
}
```

**Guarantees**:
- ✅ At most one partition has majority
- ✅ Minority partition cannot commit changes
- ✅ Prevents divergent state

### Fencing Tokens

**Mechanism**: Monotonically increasing tokens prevent stale operations

```javascript
class FencedLeader {
  constructor() {
    this.fencing_token = 0; // Incremented on each leader election
  }

  on_elected() {
    this.fencing_token = this.current_term; // Use RAFT term as fencing token
  }

  execute_operation(op) {
    return {
      type: 'OPERATION',
      fencing_token: this.fencing_token,
      operation: op
    };
  }
}

class StateMachine {
  apply(message) {
    if (message.fencing_token < this.last_token) {
      // Reject stale operation from old leader
      console.warn('Rejecting stale operation from old leader');
      return false;
    }

    this.last_token = message.fencing_token;
    this.execute(message.operation);
    return true;
  }
}
```

**Advantages**:
- ✅ Prevents stale leader from corrupting state
- ✅ No coordination required (stateless check)

### Witness Agents

**Mechanism**: Lightweight agents break ties in partitions

```javascript
class WitnessAgent {
  // Witness participates in voting but does not replicate data
  vote(candidate) {
    // Witness votes for first candidate that requests
    if (!this.voted_in_term[candidate.term]) {
      this.voted_in_term[candidate.term] = candidate.id;
      return {vote_granted: true};
    }
    return {vote_granted: false};
  }
}

class RaftCluster {
  // Cluster with 4 data agents + 1 witness = 5 total
  // Quorum = 3, tolerate 1 failure
  constructor() {
    this.data_agents = 4;
    this.witness_agents = 1;
    this.quorum_size = Math.floor((this.data_agents + this.witness_agents) / 2) + 1; // 3
  }
}
```

**Use Case**: Even number of data centers (2 DCs + 1 witness in cloud)

---

## Health Monitoring

### Agent-Level Metrics

```javascript
class HealthMonitor {
  collect_metrics() {
    return {
      // Liveness
      last_heartbeat: Date.now() - this.last_heartbeat_time,
      heartbeat_timeout: this.heartbeat_timeout,
      is_alive: Date.now() - this.last_heartbeat_time < this.heartbeat_timeout,

      // Performance
      cpu_usage: process.cpuUsage(),
      memory_usage: process.memoryUsage(),
      task_queue_length: this.task_queue.length,

      // Coordination
      state: this.state, // leader, follower, candidate
      term: this.current_term,
      commit_index: this.commit_index,
      last_applied: this.last_applied,

      // Network
      reachable_agents: this.count_reachable(),
      network_latency_p50: this.latency_percentile(0.5),
      network_latency_p99: this.latency_percentile(0.99)
    };
  }
}
```

### Cluster-Level Health

```javascript
class ClusterHealthMonitor {
  assess_cluster_health() {
    const agents = this.get_all_agents();
    const alive = agents.filter(a => a.is_alive);

    return {
      // Availability
      total_agents: agents.length,
      alive_agents: alive.length,
      availability: alive.length / agents.length,

      // Quorum
      quorum_size: this.quorum_size,
      quorum_available: alive.length >= this.quorum_size,

      // Consensus
      has_leader: this.current_leader !== null,
      leader_stable: this.leader_stable_duration(),
      elections_per_hour: this.elections_count / this.uptime_hours,

      // Performance
      commit_latency_p50: this.commit_latency_percentile(0.5),
      commit_latency_p99: this.commit_latency_percentile(0.99),
      throughput_ops_per_sec: this.committed_ops / this.uptime_seconds,

      // Failures
      failed_agents: agents.length - alive.length,
      partitions_detected: this.partition_count,
      split_brain_events: this.split_brain_count
    };
  }
}
```

---

## Self-Healing Patterns

### Automatic Agent Replacement

```javascript
class SelfHealingCluster {
  monitor_agent_health() {
    setInterval(() => {
      this.agents.forEach(agent => {
        if (this.is_permanently_failed(agent)) {
          // Agent permanently failed: replace it
          this.replace_agent(agent);
        }
      });
    }, 10000); // Check every 10 seconds
  }

  is_permanently_failed(agent) {
    // Consider agent permanently failed if unresponsive for 60 seconds
    return Date.now() - agent.last_heartbeat > 60000;
  }

  async replace_agent(failed_agent) {
    console.log(`Replacing failed agent: ${failed_agent.id}`);

    // 1. Spawn new agent
    const new_agent = await this.spawn_agent({
      type: failed_agent.type,
      capabilities: failed_agent.capabilities
    });

    // 2. Transfer state from stable checkpoint
    await this.transfer_state(failed_agent, new_agent);

    // 3. Update cluster membership
    this.remove_agent(failed_agent.id);
    this.add_agent(new_agent);

    // 4. Notify all agents of membership change
    this.broadcast_membership_update();
  }
}
```

### Circuit Breaker

```javascript
class CircuitBreaker {
  constructor() {
    this.state = 'CLOSED'; // CLOSED, OPEN, HALF_OPEN
    this.failure_count = 0;
    this.failure_threshold = 5;
    this.timeout = 60000; // 60 seconds
  }

  async call(agent, operation) {
    if (this.state === 'OPEN') {
      // Circuit open: fail fast
      if (Date.now() - this.open_timestamp > this.timeout) {
        this.state = 'HALF_OPEN'; // Try again after timeout
      } else {
        throw new Error('Circuit breaker open');
      }
    }

    try {
      const result = await agent.execute(operation);

      if (this.state === 'HALF_OPEN') {
        this.state = 'CLOSED'; // Success: close circuit
        this.failure_count = 0;
      }

      return result;
    } catch (error) {
      this.failure_count++;

      if (this.failure_count >= this.failure_threshold) {
        this.state = 'OPEN'; // Too many failures: open circuit
        this.open_timestamp = Date.now();
      }

      throw error;
    }
  }
}
```

### Exponential Backoff

```javascript
class RetryPolicy {
  async retry(operation, max_attempts = 5) {
    let attempt = 0;

    while (attempt < max_attempts) {
      try {
        return await operation();
      } catch (error) {
        attempt++;

        if (attempt >= max_attempts) {
          throw error; // Give up after max attempts
        }

        // Exponential backoff with jitter
        const delay = Math.min(1000 * Math.pow(2, attempt), 30000); // Cap at 30s
        const jitter = Math.random() * delay * 0.1; // ±10% jitter

        await this.sleep(delay + jitter);
      }
    }
  }
}
```

---

## Best Practices

1. **Detection**: Use adaptive failure detectors (Phi Accrual) for dynamic environments
2. **Recovery**: Implement automatic failover with bounded recovery time (<1 second)
3. **Partitions**: Design for partition tolerance (graceful degradation)
4. **Split-Brain**: Require quorum for all critical operations
5. **Monitoring**: Track cluster health metrics (availability, latency, throughput)
6. **Self-Healing**: Automate agent replacement and state transfer
7. **Testing**: Regularly test failure scenarios (chaos engineering)

---

**Last Updated**: 2025-11-02
**Author**: Claude Code Advanced Coordination Skill
