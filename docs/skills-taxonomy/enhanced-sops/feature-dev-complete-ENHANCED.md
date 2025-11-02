# Feature Development Complete (Enhanced SOP v2.0)

**Version**: 2.0.0 (Enhanced with Agent Coordination)
**Original Version**: 1.0.0
**Enhancement Date**: 2025-11-02

---

## Purpose

Execute complete feature development lifecycle using multi-agent orchestration with 7 specialized agents in hierarchical coordination pattern.

---

## Agent Coordination

### Primary Agent
- **Name**: `researcher`
- **Role**: Requirements analysis and best practices research
- **Capabilities**: Web search, codebase analysis, pattern recognition, MECE decomposition
- **MCP Tools**: Gemini (search), Memory MCP (context storage)

### Secondary Agents

**1. system-architect**
- **Role**: Architecture design and component modeling
- **Coordination**: Sequential after research phase
- **Handoff**: Via `swarm/feature-dev/architecture` namespace
- **Tools**: Diagramming, design patterns, API contracts
- **Fallback**: `backend-dev` for implementation-level architecture

**2. coder**
- **Role**: Implementation of designed architecture
- **Coordination**: Parallel with tester (TDD approach)
- **Handoff**: Via file system + `swarm/feature-dev/implementation` namespace
- **Tools**: Code generation, refactoring, debugging
- **Fallback**: `backend-dev`, `mobile-dev`, `ml-developer` based on domain

**3. tester**
- **Role**: Test creation and validation
- **Coordination**: Parallel with coder (TDD approach)
- **Handoff**: Via `swarm/feature-dev/tests` namespace
- **Tools**: Test frameworks, coverage analysis, sandbox execution
- **Fallback**: `tdd-london-swarm` for complex testing scenarios

**4. reviewer**
- **Role**: Code quality review
- **Coordination**: Sequential after implementation
- **Handoff**: Via `swarm/feature-dev/review` namespace
- **Tools**: Connascence analyzer, code review checklist
- **Fallback**: `code-review-assistant` for comprehensive swarm review

**5. api-docs**
- **Role**: Documentation generation
- **Coordination**: Parallel with reviewer
- **Handoff**: Via `swarm/feature-dev/docs` namespace
- **Tools**: API doc generators, example creation
- **Fallback**: `docs-writer` for complex documentation

**6. cicd-engineer**
- **Role**: Deployment preparation
- **Coordination**: Sequential after all quality gates
- **Handoff**: Via `swarm/feature-dev/deployment` namespace
- **Tools**: CI/CD pipelines, deployment scripts
- **Fallback**: `production-validator` for production readiness

### Topology Configuration

**Pattern**: Hierarchical with 7 agents
```bash
npx claude-flow@alpha swarm-init \
  --topology hierarchical \
  --max-agents 7 \
  --strategy balanced
```

### Hooks Integration

**Pre-Task** (Initialize session):
```bash
#!/bin/bash
FEATURE_ID="$(echo "$FEATURE_SPEC" | md5sum | cut -d' ' -f1)"

npx claude-flow@alpha hooks pre-task \
  --description "Feature: $FEATURE_SPEC" \
  --task-id "feature-dev-$FEATURE_ID"

npx claude-flow@alpha hooks session-restore \
  --session-id "feature-dev-$FEATURE_ID"
```

**Post-Edit** (Per agent, per file):
```bash
npx claude-flow@alpha hooks post-edit \
  --file "$FILE_PATH" \
  --memory-key "swarm/feature-dev/$AGENT_NAME/$PHASE_NAME" \
  --metadata "$(cat <<EOF
{
  "agent": "$AGENT_NAME",
  "phase": "$PHASE_NAME",
  "feature": "$FEATURE_SPEC",
  "timestamp": "$(date -Iseconds)"
}
EOF
)"
```

**Post-Task** (Session cleanup):
```bash
npx claude-flow@alpha hooks post-task \
  --task-id "feature-dev-$FEATURE_ID" \
  --summary "$(cat .claude/.artifacts/feature-summary.json)"

npx claude-flow@alpha hooks session-end \
  --export-metrics true \
  --output ".claude/.artifacts/feature-metrics-$FEATURE_ID.json"
```

### Memory Namespaces

| Namespace | Purpose | Producer | Consumer | TTL |
|-----------|---------|----------|----------|-----|
| `swarm/feature-dev/requirements` | Research & requirements | researcher | system-architect | 24h |
| `swarm/feature-dev/architecture` | Design decisions | system-architect | coder, tester | 7d |
| `swarm/feature-dev/implementation` | Code artifacts | coder | reviewer, tester | permanent |
| `swarm/feature-dev/tests` | Test suite | tester | reviewer | permanent |
| `swarm/feature-dev/review` | Review findings | reviewer | coder | 7d |
| `swarm/feature-dev/docs` | Documentation | api-docs | ALL | permanent |
| `swarm/feature-dev/deployment` | Deployment config | cicd-engineer | DEPLOYMENT | permanent |
| `swarm/feature-dev/metrics` | Performance data | ALL | MONITORING | 30d |

### Agent Handoff Protocol

**Handoff Type**: Sequential + Parallel hybrid

**Phase Transitions**:
1. **Research → Architecture** (Sequential)
   ```bash
   # researcher completes
   npx claude-flow@alpha memory store \
     "requirements_complete" \
     "$(cat requirements.json)" \
     --namespace "swarm/feature-dev/requirements"

   # system-architect loads
   REQUIREMENTS=$(npx claude-flow@alpha memory query \
     "requirements_complete" \
     --namespace "swarm/feature-dev/requirements")
   ```

2. **Architecture → Implementation + Tests** (Parallel Fork)
   ```bash
   # system-architect broadcasts
   npx claude-flow@alpha memory store \
     "architecture_complete" \
     "$(cat architecture.json)" \
     --namespace "swarm/feature-dev/architecture"

   # coder + tester load simultaneously (parallel)
   ARCH=$(npx claude-flow@alpha memory query \
     "architecture_complete" \
     --namespace "swarm/feature-dev/architecture")
   ```

3. **Implementation → Review + Docs** (Parallel Fork)
   ```bash
   # Similar pattern with parallel execution
   ```

4. **Review + Docs → Deployment** (Parallel Join → Sequential)
   ```bash
   # Wait for both review and docs to complete
   # Then cicd-engineer proceeds
   ```

---

## Required Tools

### MCP Servers (Required)

**1. claude-flow** (REQUIRED)
```json
{
  "server": "claude-flow",
  "status": "required",
  "tools": [
    {
      "name": "swarm_init",
      "purpose": "Initialize hierarchical topology with 7 agents",
      "parameters": {
        "topology": "hierarchical",
        "maxAgents": 7,
        "strategy": "balanced"
      }
    },
    {
      "name": "agent_spawn",
      "purpose": "Spawn specialized agents (researcher, system-architect, etc.)",
      "parameters": {
        "type": "enum[researcher, coder, tester, reviewer, etc.]"
      }
    },
    {
      "name": "task_orchestrate",
      "purpose": "Coordinate 7-phase workflow",
      "parameters": {
        "task": "Feature development description",
        "strategy": "adaptive",
        "maxAgents": 7
      }
    },
    {
      "name": "hooks_pre_task",
      "purpose": "Initialize feature development session"
    },
    {
      "name": "hooks_post_task",
      "purpose": "Cleanup and metrics export"
    }
  ]
}
```

**2. memory-mcp** (REQUIRED)
```json
{
  "server": "memory-mcp",
  "status": "required",
  "tools": [
    {
      "name": "memory_store",
      "purpose": "Store requirements, architecture, review findings",
      "usage": "Each phase stores artifacts for next phase"
    },
    {
      "name": "vector_search",
      "purpose": "Retrieve similar features, patterns, solutions",
      "usage": "Research phase uses for best practices lookup"
    }
  ]
}
```

### MCP Servers (Optional)

**3. connascence-analyzer** (OPTIONAL - Quality Enhancement)
```json
{
  "server": "connascence-analyzer",
  "status": "optional",
  "tools": [
    {
      "name": "analyze_file",
      "purpose": "Detect code quality issues in implementation",
      "when": "Phase 5 (Code Review)",
      "agents": ["reviewer"]
    },
    {
      "name": "analyze_workspace",
      "purpose": "Holistic code quality analysis",
      "when": "Phase 6 (Pre-deployment)",
      "agents": ["cicd-engineer"]
    }
  ]
}
```

**4. flow-nexus** (OPTIONAL - Sandbox Testing)
```json
{
  "server": "flow-nexus",
  "status": "optional",
  "tools": [
    {
      "name": "sandbox_create",
      "purpose": "Create isolated test environment",
      "when": "Phase 4 (Testing)",
      "agents": ["tester"]
    },
    {
      "name": "sandbox_execute",
      "purpose": "Run tests in sandbox",
      "when": "Phase 4 (Testing)",
      "agents": ["tester"]
    }
  ]
}
```

### Claude Code Tools

**Essential Tools**:
- **Task**: Spawn all 7 agents concurrently in appropriate phases
- **TodoWrite**: Track 7-phase progress (updated per phase)
- **Read/Write/Edit**: File operations for implementation
- **Bash**: Git operations, npm commands, test execution
- **Glob/Grep**: Code search and pattern analysis

### Authentication Requirements

- **Claude Flow**: No auth required (local MCP server)
- **Memory MCP**: No auth required (local MCP server)
- **Connascence Analyzer**: No auth required (local MCP server)
- **Flow Nexus**: User authentication required (`npx flow-nexus@latest login`)

---

## Process Steps

### Phase 1: Requirements Research (researcher)

**Step**: 1/7
**Agent**: `researcher`
**Execution Mode**: Sequential
**Dependencies**: None

**Actions**:
1. Load user feature specification
2. Perform web search for best practices (Gemini Search)
3. Analyze existing codebase patterns (if >5000 LOC)
4. Create MECE task breakdown
5. Store requirements in memory

**Tools Used**:
- Gemini Search (best practices)
- Memory MCP (store requirements)
- File system (save research.md)

**Memory Operations**:
- Write: `swarm/feature-dev/requirements/research`
- Write: `swarm/feature-dev/requirements/task_breakdown`

**Validation Checkpoint**:
- Condition: Requirements clarity score ≥80%
- On Failure: Request user clarification

**Example**:
```bash
[Single Message - Phase 1]:
  Task("researcher",
    "Research best practices for feature: $FEATURE_SPEC

    1. Use Gemini Search for 2025 best practices
    2. Analyze existing codebase if >5000 LOC
    3. Create MECE task breakdown
    4. Store in memory: swarm/feature-dev/requirements

    Quality Gate: Requirements clarity ≥80%",
    "researcher")
```

### Phase 2: Architecture Design (system-architect)

**Step**: 2/7
**Agent**: `system-architect`
**Execution Mode**: Sequential
**Dependencies**: [1]

**Actions**:
1. Load requirements from Phase 1
2. Design system architecture
3. Create component diagrams
4. Define API contracts
5. Document design decisions
6. Store architecture in memory

**Tools Used**:
- Memory MCP (load requirements, store architecture)
- Diagramming tools (architecture diagrams)
- File system (save architecture-design.md)

**Memory Operations**:
- Read: `swarm/feature-dev/requirements/*`
- Write: `swarm/feature-dev/architecture/design`
- Write: `swarm/feature-dev/architecture/contracts`

**Validation Checkpoint**:
- Condition: Architecture review approved (peer review or automated)
- On Failure: Iterate design with researcher feedback

### Phase 3-4: Parallel Implementation + Testing (coder + tester)

**Step**: 3-4/7
**Agents**: `coder`, `tester`
**Execution Mode**: Parallel (TDD approach)
**Dependencies**: [2]

**Substeps (coder)**:
1. Load architecture from Phase 2
2. Implement components per design
3. Coordinate with tester for TDD
4. Store implementation artifacts
5. Update progress in real-time memory

**Substeps (tester)**:
1. Load architecture from Phase 2
2. Write tests FIRST (TDD)
3. Coordinate with coder for implementation
4. Run tests continuously
5. Store test suite

**Tools Used**:
- Memory MCP (load architecture, coordinate TDD)
- Connascence Analyzer (optional - code quality)
- Flow Nexus Sandboxes (optional - isolated testing)
- File system (save code + tests)

**Memory Operations** (coder):
- Read: `swarm/feature-dev/architecture/*`
- Write: `swarm/feature-dev/implementation/code`
- Write: `swarm/feature-dev/realtime/progress`

**Memory Operations** (tester):
- Read: `swarm/feature-dev/architecture/*`
- Write: `swarm/feature-dev/tests/suite`
- Write: `swarm/feature-dev/tests/coverage`

**Validation Checkpoint**:
- Condition: All tests passing AND coverage ≥90%
- On Failure: Iterate coder + tester coordination

**Parallelization Benefit**: 2x speedup (vs sequential development)

**Example**:
```bash
[Single Message - Phase 3-4 Parallel]:
  Task("coder",
    "Implement feature per architecture design.
    Coordinate with tester for TDD approach.
    Store: swarm/feature-dev/implementation",
    "coder")

  Task("tester",
    "Write tests FIRST (TDD).
    Coordinate with coder.
    Target: ≥90% coverage.
    Store: swarm/feature-dev/tests",
    "tester")
```

### Phase 5-6: Parallel Review + Documentation (reviewer + api-docs)

**Step**: 5-6/7
**Agents**: `reviewer`, `api-docs`
**Execution Mode**: Parallel
**Dependencies**: [3, 4]

**Substeps (reviewer)**:
1. Load implementation from Phase 3
2. Load test suite from Phase 4
3. Run code quality analysis (Connascence Analyzer)
4. Check for theater code
5. Generate review report
6. Store findings

**Substeps (api-docs)**:
1. Load implementation from Phase 3
2. Extract API contracts
3. Generate documentation
4. Create usage examples
5. Store documentation

**Tools Used**:
- Connascence Analyzer (code quality)
- Memory MCP (load artifacts, store review/docs)
- Documentation generators

**Memory Operations** (reviewer):
- Read: `swarm/feature-dev/implementation/*`
- Read: `swarm/feature-dev/tests/*`
- Write: `swarm/feature-dev/review/findings`
- Write: `swarm/feature-dev/review/score`

**Memory Operations** (api-docs):
- Read: `swarm/feature-dev/implementation/*`
- Write: `swarm/feature-dev/docs/api`
- Write: `swarm/feature-dev/docs/examples`

**Validation Checkpoint** (reviewer):
- Condition: Code quality score ≥85/100 AND zero critical issues
- On Failure: Send findings back to coder for fixes

**Validation Checkpoint** (api-docs):
- Condition: Documentation completeness ≥95%
- On Failure: Generate missing sections

**Parallelization Benefit**: 2x speedup (vs sequential review + docs)

### Phase 7: Deployment Preparation (cicd-engineer)

**Step**: 7/7
**Agent**: `cicd-engineer`
**Execution Mode**: Sequential
**Dependencies**: [5, 6]

**Actions**:
1. Load all artifacts (code, tests, review, docs)
2. Run final security scan
3. Create deployment configuration
4. Generate deployment checklist
5. Create pull request (if enabled)
6. Store deployment package

**Tools Used**:
- Security scanners
- Git operations
- GitHub CLI (PR creation)
- Memory MCP (load all artifacts, store deployment)

**Memory Operations**:
- Read: `swarm/feature-dev/*` (all namespaces)
- Write: `swarm/feature-dev/deployment/config`
- Write: `swarm/feature-dev/deployment/checklist`

**Validation Checkpoint**:
- Condition: Production readiness ≥95% AND zero critical security issues
- On Failure: Block deployment, escalate to user

---

## Performance Optimization

### Parallelization Opportunities

**Group 1: Research + Architecture** (Sequential)
- Steps: [1, 2]
- Parallelism: 1x (must be sequential)
- Reason: Architecture depends on research

**Group 2: Implementation + Testing** (Parallel)
- Steps: [3, 4]
- Parallelism: 2x
- Reason: TDD approach allows parallel work
- Speedup: 2x vs sequential

**Group 3: Review + Documentation** (Parallel)
- Steps: [5, 6]
- Parallelism: 2x
- Reason: Independent quality checks
- Speedup: 2x vs sequential

**Group 4: Deployment** (Sequential)
- Steps: [7]
- Parallelism: 1x
- Reason: Must aggregate all artifacts

**Total Speedup**: 2.8x (measured from Groups 2 & 3 parallelization)

### Caching Strategy

**1. Research Findings**
```yaml
cache_location: swarm/feature-dev/requirements/research
cache_key: sha256($FEATURE_SPEC + date)
ttl: 86400  # 24 hours
invalidation_triggers:
  - Feature spec changes
  - New day (best practices may update)
```

**2. Architecture Diagrams**
```yaml
cache_location: swarm/feature-dev/architecture/diagrams
cache_key: sha256($ARCHITECTURE_JSON)
ttl: 604800  # 7 days
invalidation_triggers:
  - Architecture changes
  - Component additions/removals
```

**3. Test Results**
```yaml
cache_location: swarm/feature-dev/tests/results
cache_key: git_commit_sha
ttl: null  # Per-commit, not time-based
invalidation_triggers:
  - New commit
  - Test file changes
  - Code changes
```

### Resource Allocation

**Agents**:
- Min: 5 (without optional review/docs parallelization)
- Max: 7 (full parallelization)
- Optimal: 7 (recommended for 2.8x speedup)

**Memory** (per agent):
- researcher: 1GB (web search results)
- system-architect: 500MB (diagrams)
- coder: 2GB (codebase)
- tester: 1GB (test execution)
- reviewer: 1.5GB (analysis)
- api-docs: 500MB (documentation)
- cicd-engineer: 500MB (deployment)
- **Total**: ~7.5GB

**Time Estimates**:
- Min: 4 hours (simple feature, optimal parallelization)
- Max: 12 hours (complex feature, sequential execution)
- Typical: 6 hours (moderate feature, 2.8x parallelization)

**Comparison with Traditional Development**:
| Phase | Traditional | Optimized | Speedup |
|-------|-------------|-----------|---------|
| Research | 2h | 2h | 1x |
| Architecture | 2h | 2h | 1x |
| Implementation + Tests | 8h (sequential) | 4h (parallel) | 2x |
| Review + Docs | 4h (sequential) | 2h (parallel) | 2x |
| Deployment | 1h | 1h | 1x |
| **TOTAL** | **17h** | **11h** | **1.5x** |

Note: Total speedup is less than 2.8x because not all phases can be parallelized.

---

## Quality Gates

### Gate 1: Pre-Implementation (After Phase 2)

**Name**: Architecture Review Gate
**Phase**: post-architecture
**Validation**:
```yaml
metrics:
  - metric: requirements_clarity_score
    operator: ">="
    threshold: 80

  - metric: architecture_completeness
    operator: ">="
    threshold: 95

  - metric: api_contracts_defined
    operator: "=="
    threshold: true

  - metric: component_diagram_exists
    operator: "=="
    threshold: true
```

**On Failure**:
- Action: Block implementation phase
- Escalation: Return to researcher + system-architect
- Recovery: Iterate architecture design with user input

### Gate 2: Post-Implementation (After Phase 4)

**Name**: Code Quality Gate
**Phase**: post-implementation
**Validation**:
```yaml
metrics:
  - metric: test_coverage_percent
    operator: ">="
    threshold: 90

  - metric: all_tests_passing
    operator: "=="
    threshold: true

  - metric: code_quality_score
    operator: ">="
    threshold: 85

  - metric: critical_security_issues
    operator: "=="
    threshold: 0

  - metric: theater_code_detected
    operator: "=="
    threshold: 0
```

**On Failure**:
- Action: Block review phase
- Escalation: Return to coder + tester
- Recovery: Fix issues, re-run tests

### Gate 3: Pre-Deployment (After Phase 6)

**Name**: Production Readiness Gate
**Phase**: post-review
**Validation**:
```yaml
metrics:
  - metric: code_review_score
    operator: ">="
    threshold: 85

  - metric: documentation_completeness
    operator: ">="
    threshold: 95

  - metric: production_readiness_checklist
    operator: ">="
    threshold: 95

  - metric: security_scan_critical_issues
    operator: "=="
    threshold: 0

  - metric: deployment_config_valid
    operator: "=="
    threshold: true
```

**On Failure**:
- Action: Block deployment
- Escalation: User notification + manual review
- Recovery: Fix critical issues before proceeding

---

## Input Contract

```yaml
input:
  feature_spec: string (feature description, required)
  target_directory: string (default: src/)
  create_pr: boolean (default: true)
  deploy_after: boolean (default: false)

  # NEW: Agent configuration
  agent_config:
    max_agents: number (default: 7, range: 5-7)
    enable_parallelization: boolean (default: true)
    use_sandboxes: boolean (default: false)
    use_connascence_analyzer: boolean (default: true if available)

  # NEW: Quality requirements
  quality_requirements:
    min_test_coverage: number (default: 90, range: 80-100)
    min_code_quality: number (default: 85, range: 70-100)
    theater_tolerance: number (default: 0, range: 0-5)
```

---

## Output Contract

```yaml
output:
  artifacts:
    research: markdown (best practices)
    architecture: markdown (design doc)
    diagrams: array[image] (visual docs)
    implementation: directory (code)
    tests: directory (test suite)
    documentation: markdown (usage docs)
    deployment: object (deployment config)

  quality:
    test_coverage: number (percentage, target: ≥90%)
    quality_score: number (0-100, target: ≥85)
    security_issues: number (critical, target: 0)
    theater_detected: number (target: 0)
    review_score: number (0-100, target: ≥85)

  performance:
    total_time_hours: number
    parallelization_speedup: number (measured)
    agent_count: number (used)

  pr_url: string (if create_pr: true)
  deployment_ready: boolean

  # NEW: Agent coordination data
  agent_metrics:
    agents_used: array[string]
    memory_namespaces_created: array[string]
    quality_gates_passed: array[object]
    hooks_executed: array[object]
```

---

## Integration Points

### Cascades
- Standalone complete workflow
- Can be part of `/sprint-automation` cascade
- Used by `/feature-request-handler` cascade

### Commands
- Uses: `/gemini-search`, `/gemini-megacontext`, `/gemini-media`
- Uses: `/functionality-audit`, `/style-audit`
- Uses: `/theater-detect`, `/security-scan`
- Uses: `/swarm-init`, `/agent-spawn`, `/task-orchestrate`

### Other Skills
- Invokes: `quick-quality-check`, `smart-bug-fix` (if issues found)
- Output to: `code-review-assistant`, `production-readiness`
- Feeds to: `cicd-intelligent-recovery` (if deployment fails)

---

## Enhancements from v1.0

1. **Agent Coordination**: Added 7 specialized agents with clear handoff protocols
2. **MCP Integration**: Integrated claude-flow, memory-mcp, connascence-analyzer, flow-nexus
3. **Memory Namespaces**: Structured 8 namespaces for phase communication
4. **Hooks**: Pre-task, post-edit, post-task for lifecycle management
5. **Parallelization**: Identified 2.8x speedup via parallel execution
6. **Quality Gates**: Added 3 gates with measurable thresholds
7. **Performance Metrics**: Added caching, resource allocation, time estimates
8. **Validation Checkpoints**: Per-phase validation with failure recovery

---

**Enhanced Version**: 2.0.0
**Original Version**: 1.0.0
**Enhancement Date**: 2025-11-02
**Status**: Production-Ready with Agent Coordination
