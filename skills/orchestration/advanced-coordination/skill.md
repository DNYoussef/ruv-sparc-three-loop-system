---
name: advanced-coordination
description: Advanced multi-agent coordination patterns for complex distributed systems. Use when coordinating 5+ agents with dynamic task dependencies, implementing Byzantine fault tolerance, or managing distributed consensus protocols. Provides RAFT, Gossip, and Byzantine coordination strategies.
---

# Advanced Coordination - Distributed Agent Management

Sophisticated coordination protocols for large-scale multi-agent systems with fault tolerance and consensus requirements.

## When to Use This Skill

Use when coordinating complex multi-agent workflows with 5+ concurrent agents, implementing fault-tolerant distributed systems, managing dynamic task dependencies across agents, or requiring consensus protocols (RAFT, Byzantine, Gossip).

## Coordination Strategies

### RAFT Consensus
- Leader election among agents
- Replicated log consistency
- Fault tolerance for N/2+1 failures
- Strong consistency guarantees

### Gossip Protocol
- Peer-to-peer agent communication
- Eventually consistent state sharing
- Highly scalable (100+ agents)
- Network partition resilient

### Byzantine Fault Tolerance
- Malicious or faulty agent detection
- 3F+1 redundancy for F failures
- Cryptographic verification
- Critical system integrity

## Process

1. **Analyze coordination requirements**
   - Determine number of agents needed
   - Identify fault tolerance needs
   - Assess consistency requirements
   - Define communication patterns

2. **Select coordination strategy**
   - RAFT for strong consistency
   - Gossip for high scalability
   - Byzantine for security-critical systems
   - Hybrid for complex requirements

3. **Initialize coordination topology**
   - Set up agent communication channels
   - Establish leader election if needed
   - Configure heartbeat and timeouts
   - Define state replication rules

4. **Monitor coordination health**
   - Track agent liveness
   - Detect network partitions
   - Monitor consensus progress
   - Log coordination events

5. **Handle failures gracefully**
   - Automatic failover
   - State recovery mechanisms
   - Network partition healing
   - Agent replacement strategies

## Integration

- **Claude-Flow**: `npx claude-flow@alpha swarm init --topology mesh`
- **Monitoring**: Real-time coordination metrics
- **Memory-MCP**: Cross-agent state persistence
