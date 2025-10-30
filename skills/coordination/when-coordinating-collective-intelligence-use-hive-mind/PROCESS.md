# Hive Mind - Workflow

## Complete Script

```bash
#!/bin/bash

# Phase 1: Initialize Hive Mind
npx claude-flow@alpha hive init --queen-enabled --collective-intelligence
npx claude-flow@alpha agent spawn --role "queen-coordinator"
npx claude-flow@alpha agent spawn --role "collective-intelligence-coordinator"
npx claude-flow@alpha memory init --type "distributed"

# Phase 2: Coordinate Agents
npx claude-flow@alpha agent spawn --type researcher --hive-member --count 5
npx claude-flow@alpha hive assign --task "Research" --agents "researcher-*"
npx claude-flow@alpha hive monitor --interval 10

# Phase 3: Synchronize Memory
npx claude-flow@alpha memory sync init --interval 5s
npx claude-flow@alpha hive knowledge-graph --build-from "$KNOWLEDGE"

# Phase 4: Reach Consensus
npx claude-flow@alpha hive consensus init --proposal "Decision X"
npx claude-flow@alpha hive consensus calculate --mechanism "proof-of-intelligence"

# Phase 5: Execute Collectively
npx claude-flow@alpha hive execute-plan --plan execution-plan.json
npx claude-flow@alpha hive report --output hive-report.md
```

## Success Criteria
- [ ] Hive Mind operational
- [ ] Consensus mechanisms functional
- [ ] Memory synchronized
- [ ] Collective execution successful
