# Cascade Orchestrator - Workflow

## Complete Script

```bash
#!/bin/bash

# Phase 1: Design Cascade
cat > cascade.yaml <<EOF
name: development-cascade
workflows:
  - {id: design, type: sequential}
  - {id: backend, type: parallel, depends_on: [design]}
  - {id: frontend, type: sequential, depends_on: [design]}
  - {id: testing, type: parallel, depends_on: [backend, frontend]}
  - {id: deploy, type: conditional, condition: "testing.success > 0.95", depends_on: [testing]}
EOF

# Phase 2: Chain Workflows
npx claude-flow@alpha cascade init --definition cascade.yaml
npx claude-flow@alpha cascade connect --from design --to backend
npx claude-flow@alpha cascade connect --from design --to frontend

# Phase 3: Execute Cascade
npx claude-flow@alpha cascade execute --definition cascade.yaml --strategy adaptive

# Phase 4: Monitor Progress
npx claude-flow@alpha cascade monitor --interval 10 &
npx claude-flow@alpha cascade progress --format json > progress.json

# Phase 5: Optimize Flow
npx claude-flow@alpha cascade analyze --detect-bottlenecks
npx claude-flow@alpha cascade optimize --enable-parallelism
npx claude-flow@alpha cascade report --output cascade-report.md
```

## Success Criteria
- [ ] Cascade designed
- [ ] Workflows chained
- [ ] Execution successful
- [ ] Flow optimized
