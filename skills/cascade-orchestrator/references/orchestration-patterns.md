# Orchestration Patterns Reference

## Overview

This document catalogs advanced orchestration patterns for building sophisticated cascades. Each pattern addresses specific workflow requirements and coordination challenges.

## Core Patterns

### 1. Sequential Pipeline

**Description**: Execute stages one after another, with each stage consuming outputs from the previous stage.

**When to Use**:
- Stages have dependencies on previous results
- Order of execution matters
- Data transformation flows
- ETL/ELT pipelines

**Structure**:
```yaml
cascade:
  stages:
    - stage_1 → stage_2 → stage_3 → stage_4
```

**Data Flow**:
```
Input → [Stage 1] → Output_1 → [Stage 2] → Output_2 → [Stage 3] → Final Output
```

**Advantages**:
- Simple to understand and debug
- Predictable execution order
- Clear data lineage
- Easy error isolation

**Disadvantages**:
- No parallelism
- Total time = sum of all stages
- Single point of failure blocks entire pipeline
- Resource underutilization

**Example**:
```yaml
stages:
  - extract-data:
      outputs: [raw_data]

  - validate-data:
      inputs: [${extract-data.raw_data}]
      outputs: [clean_data]

  - transform-data:
      inputs: [${validate-data.clean_data}]
      outputs: [transformed_data]

  - load-data:
      inputs: [${transform-data.transformed_data}]
```

---

### 2. Parallel Fan-Out

**Description**: Execute multiple independent tasks simultaneously, then aggregate results.

**When to Use**:
- Independent operations
- Resource-intensive tasks
- Code quality checks
- Multi-source data collection
- Distributed computations

**Structure**:
```yaml
cascade:
  stages:
    - setup
    - [parallel_1, parallel_2, parallel_3, parallel_4]
    - aggregate
```

**Data Flow**:
```
         ┌─→ [Task 1] ─┐
         │              │
Input ──→├─→ [Task 2] ─┼──→ Aggregate → Output
         │              │
         └─→ [Task 3] ─┘
```

**Advantages**:
- Maximum throughput
- Efficient resource utilization
- Reduced total execution time
- Fault isolation (one failure doesn't block others)

**Disadvantages**:
- Higher resource consumption
- Coordination overhead
- Result aggregation complexity
- Potential race conditions

**Example**:
```yaml
stages:
  - parallel-checks:
      type: parallel
      swarm_config:
        topology: mesh
        max_agents: 4
      skills:
        - lint-code
        - security-scan
        - test-coverage
        - complexity-analysis

  - aggregate-results:
      inputs: [${parallel-checks.*}]
```

---

### 3. Conditional Branching

**Description**: Choose execution paths based on runtime conditions or data characteristics.

**When to Use**:
- Adaptive workflows
- Risk-based routing
- Complexity-aware processing
- Multi-strategy approaches
- Quality gates

**Structure**:
```yaml
cascade:
  stages:
    - analysis
    - conditional-routing:
        branches:
          path_a: [conditions]
          path_b: [conditions]
          path_c: [conditions]
    - finalization
```

**Data Flow**:
```
            ┌─→ [Path A] ─┐
            │             │
Analyze ──→ ├─→ [Path B] ─┼──→ Finalize
            │             │
            └─→ [Path C] ─┘
            (choose one)
```

**Advantages**:
- Intelligent resource allocation
- Optimized for specific scenarios
- Flexible workflow adaptation
- Cost optimization

**Disadvantages**:
- Complex condition management
- Harder to test all paths
- Potential for dead code
- Maintenance overhead

**Example**:
```yaml
stages:
  - analyze-complexity:
      outputs: [complexity_score]

  - route-by-complexity:
      type: conditional
      condition: ${complexity_score}
      branches:
        simple:
          conditions: ${complexity_score} < 30
          stages: [quick-fix]

        medium:
          conditions: ${complexity_score} >= 30 AND ${complexity_score} < 70
          stages: [analysis, implementation, testing]

        complex:
          conditions: ${complexity_score} >= 70
          stages: [multi-model-analysis, collaborative-fix, comprehensive-testing]
```

---

### 4. Iterative Refinement

**Description**: Repeatedly execute a stage until quality criteria are met or max iterations reached.

**When to Use**:
- Quality improvement loops
- Test-fix cycles
- Optimization tasks
- Convergence algorithms
- Auto-correction workflows

**Structure**:
```yaml
cascade:
  stages:
    - initial-attempt
    - iterative-refinement:
        repeat_until: ${quality_score} > threshold
        max_iterations: 10
    - finalization
```

**Data Flow**:
```
        ┌──────────────────┐
        │                  │
Input → │ [Process] → Test │ → Quality Check
        │    ↑         ↓   │       │
        │    └─ Refine ─┘  │       │
        └──────────────────┘       │
                │                  │
                └─── Pass ─────────┴──→ Output
```

**Advantages**:
- Automatic quality improvement
- No manual intervention
- Learns from failures
- Converges to optimal solution

**Disadvantages**:
- Potentially long execution time
- May not converge
- Resource intensive
- Difficult to predict duration

**Example**:
```yaml
stages:
  - initial-implementation:
      outputs: [code, tests]

  - test-and-refine:
      type: iterative
      max_iterations: 5
      repeat_while: ${tests.failed} > 0
      stages:
        - run-tests:
            inputs: [${code}]
            outputs: [test_results]

        - codex-fix:
            condition: ${test_results.failed} > 0
            inputs: [${test_results.failures}]
            outputs: [fixed_code]

        - update-code:
            inputs: [${fixed_code}]
            outputs: [code]
```

---

### 5. Codex Sandbox Iteration

**Description**: Execute code in isolated sandbox, auto-fix failures through iterative testing.

**When to Use**:
- Functionality testing
- Auto-fixing test failures
- Safe experimentation
- Regression prevention
- Quality assurance

**Structure**:
```yaml
cascade:
  stages:
    - implementation
    - codex-sandbox-testing:
        type: codex-sandbox
        max_iterations: 5
    - integration
```

**Data Flow**:
```
              ┌─────────────────────────┐
              │   Sandbox Environment   │
              │                         │
Code → Test ──┤  Failed? → Codex Fix   ├──→ All Pass → Deploy
              │              ↓          │
              │              Test ──────┘
              └─────────────────────────┘
```

**Advantages**:
- Safe failure environment
- Automatic bug fixing
- No manual debugging
- Fast iteration cycles

**Disadvantages**:
- Limited to test-detectable issues
- May not fix root cause
- Resource intensive
- Requires comprehensive test suite

**Example**:
```yaml
stages:
  - functionality-testing:
      type: codex-sandbox
      codex_config:
        mode: full-auto
        sandbox: true
        max_iterations: 5
        network_disabled: true
      stages:
        - run-tests:
            outputs: [test_results]

        - auto-fix:
            condition: ${test_results.failed} > 0
            iteration_loop:
              - analyze-failure
              - generate-fix
              - apply-patch
              - rerun-test
```

---

## Advanced Patterns

### 6. Hybrid Sequential-Parallel

**Description**: Combine sequential and parallel execution for optimal workflow.

**Structure**:
```yaml
stages:
  - sequential-stage-1
  - [parallel-stage-2a, parallel-stage-2b, parallel-stage-2c]
  - sequential-stage-3
  - [parallel-stage-4a, parallel-stage-4b]
  - sequential-stage-5
```

**Use Case**: CI/CD pipeline with parallel builds and sequential deployments.

**Example**:
```yaml
stages:
  - checkout-code  # Sequential

  - parallel-build:  # Parallel
      - build-frontend
      - build-backend
      - build-workers

  - integration-tests  # Sequential (needs all builds)

  - parallel-deploy:  # Parallel
      - deploy-staging-us
      - deploy-staging-eu

  - smoke-tests  # Sequential (needs all deployments)
```

---

### 7. Multi-Model Collaboration

**Description**: Route stages to different AI models based on capabilities.

**Structure**:
```yaml
stages:
  - stage_1:
      model: gemini-megacontext  # Large context
  - stage_2:
      model: codex-auto          # Fast iteration
  - stage_3:
      model: claude              # Best reasoning
```

**Use Case**: Leverage model-specific strengths for optimal results.

**Example**:
```yaml
stages:
  - analyze-codebase:
      model: gemini-megacontext
      inputs: [entire_repository]

  - prototype-solution:
      model: codex-auto
      inputs: [${analyze-codebase.insights}]

  - refine-implementation:
      model: claude
      inputs: [${prototype-solution.code}]

  - generate-diagrams:
      model: gemini-media
      inputs: [${refine-implementation.final_code}]
```

---

### 8. Error Recovery Cascade

**Description**: Graceful degradation with multiple fallback strategies.

**Structure**:
```yaml
stages:
  - primary-strategy:
      error_handling:
        retry: 3
        fallback: secondary-strategy

  - secondary-strategy:
      error_handling:
        retry: 2
        fallback: tertiary-strategy

  - tertiary-strategy:
      error_handling:
        fallback: manual-escalation
```

**Use Case**: Mission-critical workflows requiring high availability.

**Example**:
```yaml
stages:
  - primary-deployment:
      skill: deploy-to-production
      error_handling:
        strategy: retry
        max_retries: 3
        backoff: exponential
        fallback: canary-deployment

  - canary-deployment:
      skill: deploy-canary
      error_handling:
        strategy: retry
        max_retries: 2
        fallback: blue-green-deployment

  - blue-green-deployment:
      skill: deploy-blue-green
      error_handling:
        fallback: rollback-and-alert
```

---

### 9. Progressive Streaming

**Description**: Stream results as they become available, not waiting for completion.

**Structure**:
```yaml
stages:
  - parallel-tasks:
      mode: progressive
      streaming: true
      skills:
        - task_1  # Results available at t=10s
        - task_2  # Results available at t=30s
        - task_3  # Results available at t=60s
```

**Use Case**: Real-time dashboards, incremental reporting.

**Example**:
```yaml
stages:
  - progressive-analysis:
      type: parallel
      streaming: true
      report_as_available: true
      skills:
        - quick-lint:
            estimated_time: 10s
            stream_output: true

        - security-scan:
            estimated_time: 60s
            stream_output: true

        - deep-analysis:
            estimated_time: 300s
            stream_output: true
```

---

### 10. Nested Cascades

**Description**: Invoke sub-cascades from parent cascade stages.

**Structure**:
```yaml
parent_cascade:
  stages:
    - stage_1
    - nested_cascade:
        cascade: child_cascade_definition
    - stage_3
```

**Use Case**: Reusable workflow components, modular orchestration.

**Example**:
```yaml
# Parent cascade
stages:
  - prepare-data

  - process-items:
      type: nested
      cascade: item-processing-cascade
      for_each: ${prepare-data.items}

  - aggregate-results

# Child cascade (item-processing-cascade)
cascade:
  name: item-processing-cascade
  stages:
    - validate-item
    - transform-item
    - store-item
```

---

## Pattern Selection Guide

| Pattern | Complexity | Performance | Use Case |
|---------|-----------|-------------|----------|
| Sequential | Low | Moderate | Simple pipelines |
| Parallel | Medium | High | Independent tasks |
| Conditional | High | Variable | Adaptive workflows |
| Iterative | Medium | Low | Quality loops |
| Codex Sandbox | High | Low | Auto-fixing |
| Hybrid | High | High | Complex workflows |
| Multi-Model | Medium | High | AI optimization |
| Error Recovery | High | Moderate | Mission-critical |
| Progressive | Medium | High | Real-time reporting |
| Nested | High | Variable | Modular reuse |

## Best Practices

1. **Start Simple**: Begin with sequential, add complexity as needed
2. **Measure First**: Profile workflows before optimizing
3. **Document Decisions**: Explain why patterns were chosen
4. **Test All Paths**: Especially conditionals and error handling
5. **Monitor Performance**: Track execution time and resource usage
6. **Use Memory Wisely**: Share context across stages
7. **Handle Failures**: Always have fallback strategies
8. **Version Cascades**: Track changes to workflow definitions

## Related Documentation

- **micro-skills.md**: Atomic skill composition principles
- **../examples/**: Concrete pattern implementations
- **../graphviz/**: Visual workflow diagrams

---

**Remember**: The best pattern is the simplest one that meets your requirements.
