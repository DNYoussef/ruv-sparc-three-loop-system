# Cascade Orchestrator - Silver Tier Documentation

## Overview

The Cascade Orchestrator is a sophisticated workflow engine that coordinates multiple micro-skills into cohesive, production-ready processes. It combines sequential pipelines, parallel execution, conditional branching, and intelligent error recovery to create powerful automation workflows.

**Version**: 2.0.0
**Tier**: Silver (7+ files)
**Tags**: orchestration, workflows, cascades, multi-model, codex-integration

## What is a Cascade?

A **cascade** is a declarative workflow definition that:
- Coordinates multiple micro-skills into a cohesive process
- Supports sequential, parallel, and conditional execution patterns
- Integrates multi-model AI routing (Claude, Gemini, Codex)
- Provides intelligent error handling and auto-recovery
- Maintains state and memory across workflow stages
- Enables GitHub CI/CD integration

Think of it as a "recipe" that combines simple ingredients (micro-skills) into complex dishes (complete workflows).

## Quick Start

### Basic Sequential Cascade

```yaml
cascade:
  name: simple-data-pipeline
  stages:
    - stage: extract
      skill: extract-data
      inputs: { source: "database" }

    - stage: validate
      skill: validate-data
      inputs: { strict: true }

    - stage: transform
      skill: transform-data
      inputs: { format: "json" }

    - stage: load
      skill: load-data
      inputs: { target: "warehouse" }
```

**Usage**:
```bash
# Invoke via Claude Code
"Create a cascade that extracts, validates, transforms, and loads data"
```

### Parallel Execution Cascade

```yaml
cascade:
  name: code-quality-checks
  stages:
    - stage: parallel-checks
      type: parallel
      skills:
        - skill: lint-code
        - skill: security-scan
        - skill: test-coverage
        - skill: complexity-analysis

    - stage: aggregate
      skill: merge-reports
```

**Usage**:
```bash
# Automatic parallelization
"Run all code quality checks in parallel"
```

## Core Concepts

### 1. Composable Excellence

Complex capabilities emerge from composing simple, well-defined components:
- **Micro-skills**: Atomic, single-purpose operations
- **Cascades**: Coordinated sequences of micro-skills
- **Workflows**: Production-ready processes with error handling

### 2. Separation of Concerns

- **Micro-skills execute** - They perform specific tasks
- **Cascades coordinate** - They manage workflow orchestration
- **Execution engine runs** - It handles scheduling and state

### 3. Intelligent Model Routing

Automatically select the best AI model for each stage:
- **Claude**: Best overall reasoning and code generation
- **Gemini**: Large context (1M+ tokens), web search, media generation
- **Codex**: Rapid prototyping, auto-fixing, sandbox iteration

### 4. Error Recovery

Multiple strategies for handling failures:
- **Retry with backoff**: Automatic retries with exponential delays
- **Codex auto-fix**: Sandbox iteration to fix test failures
- **Model switching**: Try different AI models on failure
- **Swarm recovery**: Redistribute tasks to healthy agents

## Orchestration Patterns

### Sequential Pipeline
Execute stages one after another, passing data forward:
```yaml
stages:
  - extract → validate → transform → load
```

### Parallel Fan-Out
Execute multiple tasks simultaneously:
```yaml
stages:
  - [lint, security, tests, complexity] → aggregate
```

### Conditional Branching
Choose paths based on runtime conditions:
```yaml
stages:
  - analyze → if (quality > 80) ? fast-deploy : deep-audit
```

### Codex Sandbox Iteration
Auto-fix test failures in isolated environment:
```yaml
stages:
  - test → (if failed) → codex-fix → retest → (repeat until pass)
```

### Hybrid Workflows
Combine multiple patterns:
```yaml
stages:
  - extract
  - [validate, enrich] (parallel)
  - if (valid) → transform → load
  - else → repair → retry
```

## Key Features

### Multi-Model Integration

**Automatic Model Selection**:
```python
def select_optimal_model(task):
    if task.requires_large_context:
        return "gemini-megacontext"  # 1M tokens
    elif task.needs_current_info:
        return "gemini-search"       # Web grounding
    elif task.needs_visual_output:
        return "gemini-media"         # Images/diagrams
    elif task.needs_rapid_prototype:
        return "codex-auto"           # Fast iteration
    else:
        return "claude"               # Best reasoning
```

### Codex Sandbox Iteration

**From audit-pipeline Phase 2 pattern**:
```python
for test in failed_tests:
    iteration = 0
    while test.failed and iteration < max_iterations:
        # Spawn Codex in sandbox
        fix = spawn_codex_auto(
            task=f"Fix test failure: {test.error}",
            sandbox=True
        )

        # Re-test
        test.result = rerun_test(test)
        iteration += 1

        if test.passed:
            apply_fix_to_main(fix)
            break
```

### Swarm Coordination

**Via ruv-swarm MCP for parallel execution**:
```yaml
swarm_config:
  topology: mesh           # peer-to-peer
  max_agents: 4           # concurrent tasks
  strategy: balanced      # load distribution
  memory_shared: true     # shared context
```

### Memory Persistence

**Maintain state across stages**:
```yaml
memory:
  persistence: enabled
  scope: cascade          # cascade | global
  storage: mcp__ruv-swarm__memory
  keys:
    - analysis_results
    - intermediate_outputs
    - learned_patterns
```

## Real-World Examples

### Example 1: Complete Development Workflow
```yaml
cascade: complete-dev-workflow
stages:
  1. gemini-search: "Research latest framework best practices"
  2. codex-auto: "Rapid prototype with best practices"
  3. codex-sandbox: "Test everything, auto-fix failures"
  4. gemini-media: "Generate architecture diagrams"
  5. style-audit: "Polish code to production standards"
  6. github-pr: "Create PR with comprehensive report"
```

**Use Case**: Building a new feature from scratch with best practices, testing, and documentation.

### Example 2: Legacy Modernization
```yaml
cascade: modernize-legacy-code
stages:
  1. gemini-megacontext: "Analyze entire 50K line codebase"
  2. theater-detection: "Find all mocks and placeholders"
  3. [swarm-parallel]:
     - codex-auto: "Complete implementations"
     - gemini-media: "Document architecture"
  4. codex-sandbox: "Test all changes with auto-fix"
  5. style-audit: "Final polish"
  6. generate-pr: "Create PR with before/after comparison"
```

**Use Case**: Modernizing legacy code with comprehensive analysis and parallel execution.

### Example 3: Intelligent Bug Fix
```yaml
cascade: intelligent-bug-fix
stages:
  1. root-cause-analyzer: "Deep RCA analysis"
  2. multi-model-decision:
     condition: ${rca.complexity}
     simple: codex-auto (quick fix)
     complex: [
       gemini-megacontext (understand context),
       codex-reasoning (explore approaches),
       claude (implement best approach)
     ]
  3. codex-sandbox: "Test fix thoroughly"
  4. regression-suite: "Ensure no breakage"
  5. github-issue-update: "Document fix with RCA report"
```

**Use Case**: Fixing bugs with adaptive complexity handling and thorough validation.

## Creating Your Own Cascades

### Step 1: Identify Requirements

Ask yourself:
- What micro-skills do I need?
- Should stages run sequentially or in parallel?
- Do I need conditional branching?
- Which AI model is best for each stage?
- What error recovery strategies do I need?

### Step 2: Design the Workflow

```yaml
cascade:
  name: my-workflow
  description: What this accomplishes
  version: 1.0.0

  inputs:
    - name: source
      type: string
      description: Data source location

  stages:
    - stage_id: stage-1
      name: Extract Data
      skill: extract-data
      inputs:
        source: ${inputs.source}

    - stage_id: stage-2
      name: Process
      skill: process-data
      inputs:
        data: ${stage-1.output}
```

### Step 3: Add AI Model Selection

```yaml
stages:
  - stage: analyze
    model: gemini-megacontext  # Large context analysis
    skill: analyze-codebase

  - stage: implement
    model: codex-auto          # Fast prototyping
    skill: generate-code

  - stage: refine
    model: claude              # Best reasoning
    skill: optimize-code
```

### Step 4: Configure Error Handling

```yaml
stages:
  - stage: validate
    skill: validate-data
    error_handling:
      strategy: codex-auto-fix  # Auto-fix with Codex
      max_retries: 3
      fallback: manual-review
```

### Step 5: Enable Parallelism (Optional)

```yaml
stages:
  - stage: quality-checks
    type: parallel
    skills:
      - lint-code
      - security-scan
      - test-coverage
    swarm_config:
      topology: mesh
      max_agents: 3
```

## Integration Points

### With Micro-Skills
- Cascades execute micro-skills in defined stages
- Data flows between skills via outputs/inputs
- Error handling wraps skill execution

### With Multi-Model System
- Intelligent routing to Claude/Gemini/Codex
- Model-specific capabilities (search, media, iteration)
- Automatic fallback on model failures

### With Audit Pipeline
- Incorporates theater → functionality → style pattern
- Uses Codex sandbox iteration from Phase 2
- Applies quality gates throughout workflow

### With Claude Code Task Tool
- Spawn agents concurrently for parallel stages
- Each agent executes assigned micro-skills
- Coordination via hooks and memory

### With Ruv-Swarm MCP
- Parallel stage coordination
- Memory persistence across sessions
- Neural training from workflow patterns
- Performance tracking and optimization

## Best Practices

1. **Keep Micro-Skills Atomic**: Each skill should do one thing well
2. **Use Parallel Execution**: Run independent tasks simultaneously
3. **Choose the Right Model**: Match AI capabilities to stage requirements
4. **Handle Errors Gracefully**: Plan for failures with retries and fallbacks
5. **Share Memory**: Maintain context across stages
6. **Monitor Performance**: Track execution time and resource usage
7. **Document Workflows**: Explain why each stage exists

## Performance Benefits

With proper cascade design:
- **32.3% token reduction** through efficient coordination
- **2.8-4.4x speed improvement** via parallel execution
- **84.8% SWE-Bench solve rate** for complex tasks
- **Auto-recovery** from test failures reduces manual intervention

## Advanced Topics

### Custom Execution Strategies
See `references/orchestration-patterns.md` for advanced patterns.

### Micro-Skill Composition
See `references/micro-skills.md` for skill design principles.

### Workflow Visualization
See `graphviz/workflow.dot` for cascade flow diagrams.

### Complete Examples
See `examples/` directory for full cascade implementations.

## Getting Help

**Invoke the Orchestrator**:
```bash
"Create a cascade that [end goal] using [micro-skills] with [Codex/Gemini/swarm] capabilities"
```

**The orchestrator will**:
1. Design workflow with optimal AI model selection
2. Configure Codex sandbox for testing stages
3. Set up swarm coordination for parallel stages
4. Enable memory persistence across stages
5. Integrate with GitHub for CI/CD
6. Generate executable cascade definition

## Version History

- **2.0.0**: Added multi-model routing, Codex sandbox iteration, swarm coordination, memory persistence
- **1.0.0**: Initial release with sequential and parallel patterns

## Related Skills

- `micro-skill-creator`: Create new atomic skills
- `skill-builder`: Build complex skills
- `parallel-swarm-implementation`: Multi-agent development
- `feature-dev-complete`: End-to-end feature development

---

**Remember**: Cascades coordinate, micro-skills execute!
