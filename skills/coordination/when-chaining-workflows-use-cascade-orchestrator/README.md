# Cascade Orchestrator - Quick Start

Chain workflows with sequential, parallel, and conditional execution.

## Quick Start

```bash
# 1. Design cascade
cat > cascade.yaml <<EOF
workflows:
  - {id: design, type: sequential}
  - {id: impl, type: parallel, depends_on: [design]}
  - {id: test, type: sequential, depends_on: [impl]}
EOF

# 2. Execute cascade
npx claude-flow@alpha cascade execute --definition cascade.yaml

# 3. Monitor
npx claude-flow@alpha cascade monitor --interval 10

# 4. Optimize
npx claude-flow@alpha cascade optimize --enable-parallelism
```

## Agents
- **task-orchestrator:** Workflow coordination
- **hierarchical-coordinator:** Workflow hierarchy
- **memory-coordinator:** State management

## Success Metrics
- Completion time: Within ±15%
- Resource utilization: 70-85%
- Success rate: >95%
