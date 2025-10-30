---
name: when-coordinating-collective-intelligence-use-hive-mind
description: Advanced Hive Mind collective intelligence for queen-led multi-agent coordination with consensus and memory
version: 1.0.0
tags:
  - hive-mind
  - collective-intelligence
  - consensus
  - queen-coordinator
  - distributed-memory
category: coordination
agents:
  - collective-intelligence-coordinator
  - queen-coordinator
  - swarm-memory-manager
complexity: advanced
estimated_duration: 60-120 minutes
prerequisites:
  - Claude Flow installed
  - Understanding of collective intelligence
  - Multi-agent coordination experience
outputs:
  - Hive Mind infrastructure
  - Collective decision outputs
  - Consensus results
  - Shared memory state
---

# Hive Mind Collective Intelligence SOP

## Overview

Implement advanced Hive Mind collective intelligence system with queen-led coordination, consensus mechanisms, persistent memory, and distributed decision-making.

## Agents & Responsibilities

### collective-intelligence-coordinator
**Role:** Coordinate collective intelligence processing
**Responsibilities:**
- Aggregate agent insights
- Synthesize collective knowledge
- Identify patterns across agents
- Facilitate group learning

### queen-coordinator
**Role:** Lead and direct hive activities
**Responsibilities:**
- Set strategic direction
- Prioritize tasks
- Resolve conflicts
- Make final decisions

### swarm-memory-manager
**Role:** Manage shared memory and knowledge base
**Responsibilities:**
- Store collective memory
- Synchronize agent states
- Maintain knowledge graph
- Ensure data consistency

## Phase 1: Initialize Hive Mind

### Objective
Establish Hive Mind infrastructure with queen and collective intelligence systems.

### Scripts

```bash
# Initialize Hive Mind
npx claude-flow@alpha hive init \
  --queen-enabled \
  --collective-intelligence \
  --consensus-mechanism "proof-of-intelligence" \
  --max-agents 20

# Spawn queen coordinator
npx claude-flow@alpha agent spawn \
  --type coordinator \
  --role "queen-coordinator" \
  --capabilities "strategic-direction,conflict-resolution,final-decisions"

# Spawn collective intelligence coordinator
npx claude-flow@alpha agent spawn \
  --type coordinator \
  --role "collective-intelligence-coordinator" \
  --capabilities "insight-aggregation,pattern-recognition,group-learning"

# Spawn memory manager
npx claude-flow@alpha agent spawn \
  --type coordinator \
  --role "swarm-memory-manager" \
  --capabilities "memory-storage,state-sync,knowledge-graph"

# Initialize shared memory
npx claude-flow@alpha memory init \
  --type "distributed" \
  --replication 3 \
  --consistency "strong"

# Verify Hive Mind status
npx claude-flow@alpha hive status --show-queen --show-collective
```

### Hive Mind Architecture

**Queen Layer:**
```
Queen Coordinator
    ↓
Strategic Direction
    ↓
Task Prioritization
    ↓
Final Decisions
```

**Collective Intelligence Layer:**
```
Agent 1 → Insights →┐
Agent 2 → Insights →├─ Collective Intelligence → Synthesis
Agent 3 → Insights →│
Agent N → Insights →┘
```

**Memory Layer:**
```
Local Memory ←→ Swarm Memory Manager ←→ Distributed Memory Store
```

### Memory Patterns

```bash
# Store hive configuration
npx claude-flow@alpha memory store \
  --key "hive/config" \
  --value '{
    "queenEnabled": true,
    "consensusMechanism": "proof-of-intelligence",
    "maxAgents": 20,
    "initialized": "'$(date -Iseconds)'"
  }'

# Initialize collective memory
npx claude-flow@alpha memory store \
  --key "hive/collective/insights" \
  --value '[]'

npx claude-flow@alpha memory store \
  --key "hive/collective/patterns" \
  --value '{}'
```

## Phase 2: Coordinate Agents

### Objective
Queen-led coordination of agent activities and task assignments.

### Scripts

```bash
# Spawn worker agents
for i in {1..5}; do
  npx claude-flow@alpha agent spawn \
    --type researcher \
    --hive-member \
    --report-to queen
done

for i in {1..5}; do
  npx claude-flow@alpha agent spawn \
    --type coder \
    --hive-member \
    --report-to queen
done

# Queen assigns tasks
npx claude-flow@alpha hive assign \
  --task "Analyze codebase" \
  --agents "researcher-*" \
  --priority high

npx claude-flow@alpha hive assign \
  --task "Implement features" \
  --agents "coder-*" \
  --priority high \
  --depends-on "Analyze codebase"

# Monitor coordination
npx claude-flow@alpha hive monitor \
  --show-assignments \
  --show-progress \
  --interval 10

# Queen reviews progress
npx claude-flow@alpha hive review \
  --by queen \
  --output review-report.json
```

### Queen Decision Process

```bash
#!/bin/bash
# queen-decision-process.sh

# Collect agent insights
INSIGHTS=$(npx claude-flow@alpha agent collect-insights --all --format json)

# Queen analyzes
ANALYSIS=$(npx claude-flow@alpha queen analyze \
  --insights "$INSIGHTS" \
  --format json)

# Queen makes decision
DECISION=$(npx claude-flow@alpha queen decide \
  --analysis "$ANALYSIS" \
  --strategy "consensus-weighted")

# Broadcast decision to hive
npx claude-flow@alpha hive broadcast \
  --from queen \
  --message "$DECISION"

# Store decision in memory
npx claude-flow@alpha memory store \
  --key "hive/decisions/$(date +%s)" \
  --value "$DECISION"
```

## Phase 3: Synchronize Memory

### Objective
Maintain synchronized shared memory across all hive members.

### Scripts

```bash
# Setup memory synchronization
npx claude-flow@alpha memory sync init \
  --interval 5s \
  --consistency strong \
  --conflict-resolution "queen-decides"

# Monitor memory sync
npx claude-flow@alpha memory sync status \
  --show-lag \
  --show-conflicts

# Store collective insights
npx claude-flow@alpha memory store \
  --key "hive/collective/insights/$(date +%s)" \
  --value '{
    "agent": "researcher-1",
    "insight": "Pattern X detected in codebase",
    "confidence": 0.85,
    "timestamp": "'$(date -Iseconds)'"
  }'

# Retrieve collective knowledge
KNOWLEDGE=$(npx claude-flow@alpha memory retrieve \
  --key "hive/collective/*" \
  --format json)

# Build knowledge graph
npx claude-flow@alpha hive knowledge-graph \
  --build-from "$KNOWLEDGE" \
  --output knowledge-graph.json

# Visualize knowledge graph
npx claude-flow@alpha hive visualize \
  --type knowledge-graph \
  --input knowledge-graph.json \
  --output knowledge-graph.png
```

### Memory Synchronization Patterns

**Immediate Sync:**
```bash
# Critical data - sync immediately
npx claude-flow@alpha memory store \
  --key "critical/data" \
  --value "..." \
  --sync immediate
```

**Eventual Consistency:**
```bash
# Non-critical data - eventual sync
npx claude-flow@alpha memory store \
  --key "insights/data" \
  --value "..." \
  --sync eventual
```

**Conflict Resolution:**
```bash
# Queen resolves conflicts
npx claude-flow@alpha memory resolve-conflicts \
  --strategy "queen-decides" \
  --auto-apply
```

## Phase 4: Reach Consensus

### Objective
Collective decision-making through consensus mechanisms.

### Scripts

```bash
# Initiate consensus process
npx claude-flow@alpha hive consensus init \
  --proposal "Should we refactor module X?" \
  --mechanism "proof-of-intelligence" \
  --threshold 0.67

# Agents vote
npx claude-flow@alpha agent vote \
  --agent-id "researcher-1" \
  --proposal-id "proposal-001" \
  --vote yes \
  --reasoning "Complexity metrics indicate need"

# Collect votes
VOTES=$(npx claude-flow@alpha hive consensus votes \
  --proposal-id "proposal-001" \
  --format json)

# Calculate consensus
RESULT=$(npx claude-flow@alpha hive consensus calculate \
  --votes "$VOTES" \
  --mechanism "proof-of-intelligence")

# Queen validates consensus
npx claude-flow@alpha queen validate-consensus \
  --result "$RESULT" \
  --apply-decision

# Store consensus result
npx claude-flow@alpha memory store \
  --key "hive/consensus/proposal-001" \
  --value "$RESULT"
```

### Consensus Mechanisms

**Proof of Intelligence:**
- Agents with higher intelligence scores have more weight
- Based on past performance and accuracy

**Byzantine Fault Tolerant:**
- Tolerates up to 33% malicious agents
- Requires 67% agreement

**Weighted Voting:**
- Votes weighted by agent specialization
- Domain experts have more influence

**Queen Override:**
- Queen can override consensus in critical situations
- Logged and justified

## Phase 5: Execute Collectively

### Objective
Coordinated execution of collective decisions with synchronized actions.

### Scripts

```bash
# Plan collective execution
npx claude-flow@alpha hive plan-execution \
  --decision "$RESULT" \
  --output execution-plan.json

# Assign execution tasks
npx claude-flow@alpha hive execute-plan \
  --plan execution-plan.json \
  --strategy "parallel"

# Monitor collective execution
npx claude-flow@alpha hive monitor-execution \
  --plan-id "plan-001" \
  --interval 5 \
  --output execution-log.json

# Collect execution results
RESULTS=$(npx claude-flow@alpha hive collect-results \
  --plan-id "plan-001" \
  --format json)

# Queen evaluates results
EVALUATION=$(npx claude-flow@alpha queen evaluate \
  --results "$RESULTS")

# Update collective memory
npx claude-flow@alpha memory store \
  --key "hive/executions/plan-001" \
  --value '{
    "plan": "'$PLAN_ID'",
    "results": '$RESULTS',
    "evaluation": '$EVALUATION',
    "timestamp": "'$(date -Iseconds)'"
  }'

# Generate hive report
npx claude-flow@alpha hive report \
  --include-consensus \
  --include-execution \
  --include-learnings \
  --output hive-report.md
```

### Collective Learning

```bash
# Extract learnings from execution
LEARNINGS=$(npx claude-flow@alpha hive extract-learnings \
  --execution-id "plan-001" \
  --format json)

# Update collective intelligence
npx claude-flow@alpha hive update-intelligence \
  --learnings "$LEARNINGS"

# Train collective patterns
npx claude-flow@alpha neural train \
  --agent-id "collective-intelligence-coordinator" \
  --pattern convergent \
  --data "$LEARNINGS"

# Verify learning
npx claude-flow@alpha hive verify-learning \
  --test-cases test-cases.json
```

## Success Criteria

- [ ] Hive Mind initialized
- [ ] Queen coordinating effectively
- [ ] Memory synchronized
- [ ] Consensus reached on decisions
- [ ] Collective execution successful

### Performance Targets
- Consensus time: <2 minutes
- Memory sync latency: <100ms
- Queen decision time: <30 seconds
- Collective accuracy: >90%
- Coordination overhead: <15%

## Best Practices

1. **Clear Hierarchy:** Queen has final authority
2. **Shared Memory:** All agents access common knowledge
3. **Consensus Building:** Seek agreement before major decisions
4. **Continuous Learning:** Update collective intelligence
5. **Conflict Resolution:** Queen resolves conflicts quickly
6. **Pattern Recognition:** Identify and share patterns
7. **Knowledge Sharing:** Propagate insights rapidly
8. **Performance Tracking:** Monitor collective performance

## Common Issues & Solutions

### Issue: Consensus Deadlock
**Symptoms:** Agents can't reach agreement
**Solution:** Queen intervenes and makes final decision

### Issue: Memory Sync Lag
**Symptoms:** Agents have inconsistent state
**Solution:** Increase sync frequency, reduce data volume

### Issue: Queen Bottleneck
**Symptoms:** Queen overwhelmed with decisions
**Solution:** Delegate routine decisions to collective intelligence

## Integration Points

- **advanced-swarm:** For topology optimization
- **swarm-orchestration:** For task coordination
- **performance-analysis:** For collective performance metrics

## References

- Collective Intelligence Theory
- Consensus Algorithms
- Distributed Memory Systems
- Queen-Worker Patterns
