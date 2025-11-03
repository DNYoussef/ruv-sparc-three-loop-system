# Cascade Templates

Production-ready workflow templates for common orchestration patterns.

## Available Templates

### 1. cascade-pipeline.yaml
**Complete development pipeline** with quality gates, multi-model routing, and GitHub integration.

**Use Case:** End-to-end feature development from research to deployment

**Features:**
- Gemini Search for research
- Claude for architecture design
- Swarm coordination for parallel implementation
- Theater detection quality gate
- Codex sandbox iteration for testing
- Conditional quality routing
- Gemini Media for documentation
- GitHub PR automation

**Stages:**
1. Research & Planning (Gemini Search)
2. Architecture Design (Claude)
3. Parallel Implementation (Swarm)
4. Theater Detection (Quality Gate)
5. Functionality Testing (Codex Sandbox)
6. Conditional Quality Check
7. Documentation (Gemini Media)
8. GitHub PR Creation

**Configuration:**
```yaml
inputs:
  project_path: "path/to/project"
  requirements: "Feature description"
  quality_threshold: 80

config:
  multi_model: enabled
  swarm_coordination: enabled
  memory_persistence: enabled
  github_integration: enabled
  quality_gates: enabled
```

**Expected Duration:** 5-15 minutes depending on complexity

**Success Rate:** 85-95% for well-defined requirements

### 2. sequential-workflow.json
**Linear data processing pipeline** with validation and transformation.

**Use Case:** ETL (Extract, Transform, Load) workflows, data pipelines

**Features:**
- Sequential stage execution
- Data validation with auto-fix
- Codex for rapid transformation
- Quality metrics
- Gemini Media for reporting
- Memory persistence

**Stages:**
1. Data Extraction
2. Data Validation (with Codex auto-fix)
3. Data Transformation (Codex)
4. Quality Assurance
5. Data Export
6. Report Generation (Gemini Media)

**Configuration:**
```json
{
  "inputs": {
    "data_source": "path/or/url",
    "output_format": "json",
    "validation_strict": true
  }
}
```

**Expected Duration:** 2-5 minutes for typical datasets

**Throughput:** 1000-10000 records/second depending on transformations

### 3. conditional-dag.yaml
**Adaptive quality workflow** with dynamic branching based on code quality scores.

**Use Case:** Code quality improvement, technical debt reduction, CI/CD quality gates

**Features:**
- Initial quality analysis (Gemini Megacontext)
- Three quality-based paths:
  - High (≥90): Quick polish
  - Medium (≥70): Moderate improvements with auto-fix
  - Low (<70): Comprehensive audit and phased refactoring
- Codex sandbox iteration
- Swarm-based parallel audit
- Convergence to documentation
- Final quality gate
- Conditional GitHub action (PR vs Issues)

**Stages:**
1. Codebase Analysis (Gemini Megacontext)
2. Quality-Based Routing (Conditional)
   - **High Quality Path:**
     - Quick Polish
   - **Medium Quality Path:**
     - Identify Issues
     - Auto-Fix (Codex Sandbox)
     - Verification
   - **Low Quality Path:**
     - Comprehensive Audit (Swarm)
     - Refactoring Plan
     - Phased Refactoring (Critical → High → Verification)
3. Documentation (Convergence)
4. Final Quality Assessment
5. GitHub Action (Conditional: PR or Issues)

**Configuration:**
```yaml
inputs:
  codebase_path: "path/to/code"
  quality_threshold_high: 90
  quality_threshold_medium: 70
  auto_fix_enabled: true
```

**Quality Routing:**
- Score ≥90: ~1-2 minutes (quick path)
- Score 70-89: ~3-5 minutes (moderate path)
- Score <70: ~8-15 minutes (comprehensive path)

## Template Selection Guide

### Choose cascade-pipeline.yaml when:
- ✓ Building new features from scratch
- ✓ Need comprehensive quality gates
- ✓ Want automated GitHub integration
- ✓ Require multi-model capabilities
- ✓ Working on production code

### Choose sequential-workflow.json when:
- ✓ Processing data in linear steps
- ✓ Need strict validation and transformation
- ✓ Want automatic error correction
- ✓ Generating reports
- ✓ ETL workflows

### Choose conditional-dag.yaml when:
- ✓ Code quality varies widely
- ✓ Want adaptive workflows
- ✓ Need different strategies per quality level
- ✓ Optimizing existing codebases
- ✓ Technical debt management

## Customizing Templates

### Modify Stage Types

```yaml
# Change from sequential to parallel
stages:
  - stage_id: processing
    type: parallel  # was: sequential
    skills:
      - process-1
      - process-2
      - process-3
```

### Add Custom Stages

```yaml
stages:
  # ... existing stages ...

  - stage_id: custom_analysis
    name: Custom Analysis Stage
    type: sequential
    model: claude
    dependencies:
      - previous_stage

    skills:
      - skill: my-custom-skill
        inputs:
          data: "${memory.previous_output}"

    memory:
      write_keys:
        - custom_results
```

### Modify Model Selection

```yaml
# Use specific model instead of auto-select
stages:
  - stage_id: research
    model: gemini-search  # was: auto-select

  - stage_id: implementation
    model: codex-auto  # was: auto-select
```

### Add Quality Gates

```yaml
quality_gates:
  - gate: custom_check
    required: true
    fail_on_error: true
    min_score: 85

  - gate: optional_check
    required: false
    warn_on_error: true
```

### Configure Error Handling

```yaml
stages:
  - stage_id: critical_stage
    error_handling:
      strategy: codex-auto-fix  # was: retry
      max_iterations: 10  # was: 3
      escalate_after: 10
      escalation:
        type: slack
        channel: "#alerts"
```

## Advanced Patterns

### Pattern 1: Fan-Out / Fan-In

```yaml
stages:
  # Setup (1 stage)
  - stage_id: setup
    type: sequential

  # Fan-out (parallel processing)
  - stage_id: parallel_processing
    type: swarm-parallel
    dependencies: [setup]
    skills:
      - process-a
      - process-b
      - process-c
      - process-d

  # Fan-in (aggregate results)
  - stage_id: aggregate
    type: sequential
    dependencies: [parallel_processing]
    skills:
      - merge-results
```

### Pattern 2: Iterative Refinement

```yaml
stages:
  - stage_id: refine
    type: sequential
    skills:
      - improve-code

  - stage_id: check
    type: sequential
    dependencies: [refine]
    skills:
      - quality-check

  - stage_id: repeat
    type: conditional
    dependencies: [check]
    condition: "${check.quality_score} < 90"

    branches:
      - name: continue_refining
        condition:
          type: threshold
          metric: "${check.iteration}"
          threshold: 5
          comparison: "<"
        stages:
          - stage_id: refine  # Loop back
            repeat: true

      - name: done
        default: true
```

### Pattern 3: Multi-Stage Quality Gates

```yaml
stages:
  - stage_id: implementation
    type: sequential

  # Gate 1: Theater Detection
  - stage_id: gate_theater
    type: sequential
    dependencies: [implementation]
    skills:
      - theater-detection-audit

  - stage_id: stop_if_theater
    type: conditional
    condition: "${gate_theater.has_theater} == false"

  # Gate 2: Functionality
  - stage_id: gate_functionality
    type: codex-sandbox
    dependencies: [stop_if_theater]

  # Gate 3: Style
  - stage_id: gate_style
    type: sequential
    dependencies: [gate_functionality]
    skills:
      - style-audit
```

## Template Variables

### Built-in Variables

- `${inputs.*}` - Input parameters
- `${memory.*}` - Memory store values
- `${stage_outputs.*}` - Previous stage outputs
- `${timestamp}` - Current timestamp
- `${cascade.name}` - Cascade name
- `${cascade.version}` - Cascade version

### Custom Variables

```yaml
variables:
  max_retries: 3
  timeout_seconds: 300
  quality_threshold: 80

stages:
  - stage_id: process
    error_handling:
      max_retries: "${variables.max_retries}"
    timeout: "${variables.timeout_seconds}"
```

## Memory Management

### Scope Options

```yaml
memory:
  persistence: enabled
  scope: cascade  # cascade | global | stage

  # Cascade scope: Available to all stages in this cascade
  # Global scope: Available across cascades
  # Stage scope: Only within current stage
```

### Memory Keys

```yaml
stages:
  - stage_id: producer
    memory:
      write_keys:
        - output_data
        - metadata

  - stage_id: consumer
    memory:
      read_keys:
        - output_data
      write_keys:
        - processed_data
```

### Retention Policies

```yaml
memory:
  retention: 7d  # 7 days
  cleanup_on_exit: true
  max_size_mb: 100

  checkpoints:
    - stage_1
    - stage_5
    - final_stage
```

## GitHub Integration

### Auto-Create PR

```yaml
github_integration:
  enabled: true
  create_pr: on_success
  auto_merge: false
  require_review: true

  pr_template:
    title: "Automated: ${cascade.name}"
    labels:
      - automated
      - quality-checked
    reviewers:
      - team-lead
    assignees:
      - ${author}
```

### Create Issues on Failure

```yaml
github_integration:
  report_issues: on_failure

  issue_template:
    title: "Cascade Failed: ${cascade.name}"
    labels:
      - bug
      - automated
    body: |
      # Failure Report

      **Stage:** ${failed_stage}
      **Error:** ${error_message}
      **Timestamp:** ${timestamp}

      ## Logs
      ${error_logs}
```

## Notifications

### Slack Integration

```yaml
notifications:
  on_success:
    - type: slack
      channel: "#builds"
      message: "✓ ${cascade.name} completed successfully"

  on_failure:
    - type: slack
      channel: "#alerts"
      message: "✗ ${cascade.name} failed at ${failed_stage}"
      mentions:
        - "@oncall"
```

### Email Notifications

```yaml
notifications:
  on_success:
    - type: email
      to: ["team@example.com"]
      subject: "Success: ${cascade.name}"
      body: "${execution_summary}"

  on_failure:
    - type: email
      to: ["team@example.com", "manager@example.com"]
      subject: "FAILURE: ${cascade.name}"
      body: "${error_report}"
      priority: high
```

## Performance Optimization

### Caching

```yaml
cache:
  enabled: true
  strategy: memory  # memory | disk | distributed

  keys:
    - analysis_results
    - dependency_graph

  ttl: 3600  # 1 hour
```

### Parallel Optimization

```yaml
stages:
  - stage_id: parallel_ops
    type: swarm-parallel
    swarm_config:
      topology: mesh  # mesh | hierarchical | star
      max_agents: 8  # Increase for more parallelism
      strategy: balanced  # balanced | specialized | adaptive
      load_balancing: true
      memory_shared: true
```

### Resource Limits

```yaml
resources:
  max_memory_mb: 2048
  max_cpu_percent: 80
  max_duration_seconds: 3600

  per_stage:
    timeout: 300
    memory_limit_mb: 512
```

## Validation

All templates include validation schemas:

```bash
# Validate template
python3 -c "
import yaml
import json
from jsonschema import validate

with open('cascade-pipeline.yaml') as f:
    cascade = yaml.safe_load(f)

with open('cascade-schema.json') as f:
    schema = json.load(f)

validate(cascade, schema)
print('✓ Template valid')
"
```

## Best Practices

1. **Start Simple:** Use sequential-workflow.json as baseline
2. **Add Parallelism:** Upgrade to swarm-parallel for independent tasks
3. **Add Conditionals:** Use conditional-dag.yaml for adaptive behavior
4. **Enable Quality Gates:** Add theater detection, functionality audit
5. **Integrate GitHub:** Automate PR creation and issue reporting
6. **Monitor Performance:** Track duration, success rate, resource usage
7. **Iterate:** Refine based on actual usage patterns

## Support

For template questions or custom requirements:
1. Review main skill documentation (`SKILL.md`)
2. Check script README (`resources/scripts/README.md`)
3. Run tests for examples (`tests/`)
4. Enable verbose mode for debugging
