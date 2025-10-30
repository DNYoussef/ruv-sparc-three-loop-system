---
name: when-chaining-workflows-use-cascade-orchestrator
description: Create sophisticated workflow cascades with sequential pipelines, parallel execution, and conditional branching
version: 1.0.0
tags:
  - cascade
  - workflows
  - orchestration
  - conditional
  - branching
category: coordination
agents:
  - task-orchestrator
  - hierarchical-coordinator
  - memory-coordinator
complexity: advanced
estimated_duration: 45-90 minutes
prerequisites:
  - Claude Flow installed
  - Workflow design experience
  - Conditional logic knowledge
outputs:
  - Cascade workflow definitions
  - Execution results
  - Performance metrics
  - Workflow reports
---

# Cascade Orchestrator SOP

## Overview

Create sophisticated workflow cascades that chain multiple workflows together with sequential pipelines, parallel execution, conditional branching, and error handling.

## Agents & Responsibilities

### task-orchestrator
**Role:** Orchestrate workflow execution
**Responsibilities:**
- Coordinate workflow execution
- Handle workflow transitions
- Manage workflow state
- Monitor progress

### hierarchical-coordinator
**Role:** Manage workflow hierarchy
**Responsibilities:**
- Organize workflow levels
- Handle parent-child workflows
- Manage dependencies
- Ensure proper ordering

### memory-coordinator
**Role:** Manage workflow state and data flow
**Responsibilities:**
- Store workflow state
- Pass data between workflows
- Maintain execution history
- Ensure data consistency

## Phase 1: Design Cascade

### Objective
Design cascade structure with workflows, dependencies, and branching logic.

### Scripts

```bash
# Create cascade definition
cat > cascade-definition.yaml <<EOF
name: full-stack-development
workflows:
  - id: design
    type: sequential
    steps: [requirements, architecture, database-design]
  - id: backend
    type: parallel
    steps: [api-impl, auth-impl, db-impl]
    depends_on: [design]
  - id: frontend
    type: sequential
    steps: [ui-impl, integration]
    depends_on: [backend]
  - id: testing
    type: parallel
    steps: [unit-tests, integration-tests, e2e-tests]
    depends_on: [backend, frontend]
  - id: deployment
    type: conditional
    condition: "testing.success_rate > 0.95"
    steps: [build, deploy, verify]
    depends_on: [testing]
EOF

# Validate cascade
npx claude-flow@alpha cascade validate --definition cascade-definition.yaml

# Visualize cascade
npx claude-flow@alpha cascade visualize \
  --definition cascade-definition.yaml \
  --output cascade-diagram.png

# Store cascade definition
npx claude-flow@alpha memory store \
  --key "cascade/definition" \
  --file cascade-definition.yaml
```

### Cascade Patterns

**Sequential Cascade:**
```
Workflow A → Workflow B → Workflow C
```

**Parallel Cascade:**
```
         ┌─ Workflow B ─┐
Workflow A ├─ Workflow C ─┤ Workflow E
         └─ Workflow D ─┘
```

**Conditional Cascade:**
```
Workflow A → Decision → [if true] Workflow B
                      → [if false] Workflow C
```

**Hybrid Cascade:**
```
Design → ┬─ Backend ─┐
         └─ Frontend─┴─ [if tests pass] → Deploy
```

## Phase 2: Chain Workflows

### Objective
Connect workflows with proper data flow and dependency management.

### Scripts

```bash
# Initialize cascade
npx claude-flow@alpha cascade init \
  --definition cascade-definition.yaml

# Connect workflow stages
npx claude-flow@alpha cascade connect \
  --from design \
  --to backend \
  --data-flow "architecture-docs"

npx claude-flow@alpha cascade connect \
  --from design \
  --to frontend \
  --data-flow "ui-specs"

npx claude-flow@alpha cascade connect \
  --from backend \
  --to testing \
  --data-flow "api-endpoints"

# Setup conditional branching
npx claude-flow@alpha cascade branch \
  --workflow testing \
  --condition "success_rate > 0.95" \
  --true-path deployment \
  --false-path debugging

# Verify connections
npx claude-flow@alpha cascade status --show-connections
```

### Data Flow Configuration

```bash
# Configure data passing
npx claude-flow@alpha cascade data-flow \
  --from "design.architecture" \
  --to "backend.api-spec" \
  --transform "extract-api-endpoints"

# Setup shared state
npx claude-flow@alpha memory store \
  --key "cascade/shared-state" \
  --value '{
    "project": "full-stack-app",
    "version": "1.0.0",
    "environment": "production"
  }'
```

## Phase 3: Execute Cascade

### Objective
Execute cascading workflows with proper sequencing and error handling.

### Scripts

```bash
# Execute cascade
npx claude-flow@alpha cascade execute \
  --definition cascade-definition.yaml \
  --input initial-requirements.json \
  --strategy adaptive

# Monitor execution
npx claude-flow@alpha cascade monitor \
  --interval 10 \
  --show-current-workflow \
  --show-progress

# Handle workflow transitions
npx claude-flow@alpha cascade transitions \
  --show-pending \
  --show-completed

# Track workflow status
npx claude-flow@alpha cascade workflow-status \
  --workflow backend \
  --detailed
```

### Execution Strategies

**Adaptive Execution:**
```bash
# Automatically adjusts based on performance
npx claude-flow@alpha cascade execute \
  --strategy adaptive \
  --optimize-for throughput
```

**Fail-Fast Execution:**
```bash
# Stops immediately on first error
npx claude-flow@alpha cascade execute \
  --strategy fail-fast \
  --error-action stop
```

**Resilient Execution:**
```bash
# Continues despite failures, retries failed workflows
npx claude-flow@alpha cascade execute \
  --strategy resilient \
  --retry-attempts 3 \
  --error-action continue
```

## Phase 4: Monitor Progress

### Objective
Track cascade execution with real-time monitoring and progress reporting.

### Scripts

```bash
# Real-time monitoring
npx claude-flow@alpha cascade monitor \
  --interval 5 \
  --output cascade-monitor.log &

# Get progress summary
npx claude-flow@alpha cascade progress \
  --format json > progress.json

# Track workflow metrics
npx claude-flow@alpha cascade metrics \
  --per-workflow \
  --include-timing

# Detect bottlenecks
npx claude-flow@alpha cascade analyze \
  --detect-bottlenecks \
  --output analysis.json

# Generate progress report
npx claude-flow@alpha cascade report \
  --type progress \
  --include-timeline \
  --output progress-report.md
```

### Progress Visualization

```bash
# Generate Gantt chart
npx claude-flow@alpha cascade gantt \
  --output cascade-gantt.png

# Generate flow diagram
npx claude-flow@alpha cascade flow-diagram \
  --show-current-state \
  --output cascade-flow.png
```

## Phase 5: Optimize Flow

### Objective
Optimize cascade performance through parallel execution and resource allocation.

### Scripts

```bash
# Analyze cascade performance
npx claude-flow@alpha cascade analyze \
  --identify-parallelism \
  --output optimization-recommendations.json

# Apply optimizations
npx claude-flow@alpha cascade optimize \
  --enable-parallelism \
  --balance-load \
  --cache-results

# Rebalance workflows
npx claude-flow@alpha cascade rebalance \
  --strategy adaptive \
  --target-utilization 0.8

# Measure improvements
npx claude-flow@alpha cascade compare \
  --baseline baseline-metrics.json \
  --current current-metrics.json \
  --output improvements.json

# Generate optimization report
npx claude-flow@alpha cascade report \
  --type optimization \
  --show-improvements \
  --output optimization-report.md
```

### Optimization Techniques

**Parallel Execution:**
- Identify independent workflows
- Execute simultaneously
- Reduce total execution time

**Workflow Caching:**
- Cache workflow results
- Reuse for subsequent executions
- Skip unnecessary recomputation

**Resource Pooling:**
- Share resources across workflows
- Reduce overhead
- Improve efficiency

**Smart Scheduling:**
- Schedule workflows based on dependencies
- Optimize resource utilization
- Minimize idle time

## Success Criteria

- [ ] Cascade designed with clear structure
- [ ] Workflows chained correctly
- [ ] Execution proceeding smoothly
- [ ] Progress tracked continuously
- [ ] Flow optimized for performance

### Performance Targets
- Cascade completion time: Within estimates ±15%
- Workflow transition time: <5 seconds
- Resource utilization: 70-85%
- Error recovery time: <30 seconds
- Overall success rate: >95%

## Best Practices

1. **Clear Dependencies:** Explicitly define workflow dependencies
2. **Error Handling:** Implement retry and fallback mechanisms
3. **Data Validation:** Validate data at workflow boundaries
4. **Progress Tracking:** Monitor execution continuously
5. **Resource Management:** Allocate resources efficiently
6. **State Management:** Use shared memory for state
7. **Conditional Logic:** Test all branches
8. **Performance Monitoring:** Track metrics throughout

## Common Issues & Solutions

### Issue: Workflow Deadlock
**Symptoms:** Workflows waiting indefinitely
**Solution:** Check for circular dependencies, implement timeout

### Issue: Data Loss Between Workflows
**Symptoms:** Missing data in downstream workflows
**Solution:** Verify data flow configuration, use persistent storage

### Issue: Performance Degradation
**Symptoms:** Slow execution, increasing latency
**Solution:** Analyze bottlenecks, optimize critical workflows

## Integration Points

- **swarm-orchestration:** For agent coordination
- **pipeline-chaining:** For data flow
- **performance-analysis:** For optimization insights

## References

- Workflow Orchestration Patterns
- Cascade Design Principles
- Conditional Execution Strategies
- Performance Optimization Techniques
