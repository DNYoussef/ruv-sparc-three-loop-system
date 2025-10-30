# Skill-Agent-Command-MCP Mappings

**Version**: 2.0.0
**Last Updated**: 2025-10-30

This document provides comprehensive mappings between skills, agents, commands, and MCP tools in the Three-Loop System plugin.

---

## Table of Contents

1. [Three-Loop System Mappings](#three-loop-system-mappings)
2. [Core Skills Mappings](#core-skills-mappings)
3. [SPARC Methodology Mappings](#sparc-methodology-mappings)
4. [Agent-to-Skill Mappings](#agent-to-skill-mappings)
5. [Command-to-Skill Mappings](#command-to-skill-mappings)
6. [MCP Tool Usage by Component](#mcp-tool-usage-by-component)
7. [Complete Integration Map](#complete-integration-map)

---

## Three-Loop System Mappings

### Loop 1: research-driven-planning

**Skill**: `research-driven-planning`

**Primary Agents** (6-agent research SOP + 8-agent pre-mortem SOP):
- `web-researcher` (x3) - Web information gathering
- `github-analyst` (x2) - GitHub repository analysis
- `synthesis-coordinator` - Cross-validation and synthesis
- `pre-mortem-analyst` (x3) - Optimistic/pessimistic/realistic failure analysis
- `root-cause-detective` (x2) - Root cause investigation
- `defense-architect` - Architectural defense planning
- `cost-benefit-analyzer` - Risk-reward assessment
- `byzantine-coordinator` - 2/3 consensus on risk severity

**Commands**:
- `/sparc:spec-pseudocode` - Specification phase integration
- `/essential-commands/quick-check` - Pre-mortem validation
- Custom: `"Execute research-driven-planning skill for [project]"`

**MCP Tools**:
- `mcp__claude-flow__swarm_init` - Initialize research swarm
- `mcp__claude-flow__agent_spawn` - Spawn research agents
- `mcp__claude-flow__memory_usage` - Store planning artifacts
- `mcp__flow-nexus__github_repo_analyze` - GitHub analysis (optional)

**Evidence-Based Techniques**:
- Self-consistency (3 web researchers, 2 GitHub analysts cross-validate)
- Byzantine consensus (2/3 agreement on risk severity)
- Plan-and-solve (explicit SOP phases)
- Program-of-thought (pre-mortem reasoning)

**Output**: Planning package in `.claude/.artifacts/loop1-planning-package.json`

**Memory Namespace**: `integration/loop1-to-loop2`

---

### Loop 2: parallel-swarm-implementation (META-SKILL)

**Skill**: `parallel-swarm-implementation`

**Primary Agent**:
- `queen-coordinator` - Analyzes Loop 1 plan and compiles agent+skill execution graph

**Dynamic Agents** (selected from 86-agent registry based on task):
- Core development: `researcher`, `coder`, `tester`, `reviewer`, `planner`, `analyst`
- Specialized: `backend-dev`, `mobile-dev`, `ml-developer`, `cicd-engineer`
- Theater detection (6-agent SOP):
  - `theater-detection-audit` (x3) - Code/tests/docs theater detection
  - `sandbox-validator` (x2) - Reality validation
  - `integration-checker` - Cross-component validation
  - `byzantine-coordinator` - 4/5 consensus (no false positives)

**Commands**:
- `/sparc:code` - Implementation phase
- `/coordination/swarm-init` - Initialize implementation swarm
- Custom: `"Execute parallel-swarm-implementation using Loop 1 planning package"`

**MCP Tools**:
- `mcp__claude-flow__swarm_init` - Initialize hierarchical swarm
- `mcp__claude-flow__agent_spawn` - Spawn implementation agents dynamically
- `mcp__claude-flow__task_orchestrate` - Orchestrate parallel execution
- `mcp__ruv-swarm__swarm_monitor` - Real-time monitoring (optional)
- `mcp__flow-nexus__sandbox_execute` - Sandbox validation (optional)

**Evidence-Based Techniques**:
- Self-consistency (6-agent theater detection)
- Byzantine consensus (4/5 agreement, eliminates false positives)
- Hierarchical coordination (Queen delegates to specialists)
- Program-of-thought (explicit agent SOPs)

**Output**: Implementation package in `.claude/.artifacts/loop2-delivery-package.json`

**Memory Namespaces**: `swarm/persistent`, `swarm/realtime`, `integration/loop2-to-loop3`

---

### Loop 3: cicd-intelligent-recovery

**Skill**: `cicd-intelligent-recovery`

**Primary Agents**:

**7-Agent Analysis SOP** (Byzantine consensus 5/7):
- `failure-pattern-researcher` (x2) - External failure research + cross-validation
- `error-message-analyzer` - Error/stack trace analysis
- `code-context-investigator` - Surrounding code context
- `test-validity-auditor` (x2) - Test correctness validation + cross-validation
- `dependency-conflict-detector` - Dependency issues
- `byzantine-coordinator` - 5/7 consensus on root causes

**Graph-Based Root Cause** (Raft consensus):
- `graph-analyst` (x2) - Dependency graph construction
- `connascence-detector` (x3) - Connascence analysis (name/type/algorithm)
- `raft-manager` - Leader-based log replication coordination

**Program-of-Thought Fixes**:
- `fix-strategy-planner` - Plan fix approach
- `fix-implementation-specialist` - Execute minimal fix
- `sandbox-validator` - Validate fix in isolation
- `theater-detection-audit` - Audit fix for theater
- `fix-approval-coordinator` - Dual validation approval (Byzantine consensus)

**6-Agent Theater Detection** (Byzantine consensus 4/5):
- `theater-detection-audit` (x3) - Multi-angle theater detection
- `sandbox-validator` (x2) - Reality validation
- `byzantine-coordinator` - 4/5 consensus

**Gemini Large-Context**:
- `gemini-analyst` - 2M token window for full codebase analysis

**Commands**:
- `/sparc:debug` - Debug integration
- `/essential-commands/fix-bug` - Bug fix workflow
- `/multi-model-commands/gemini-megacontext` - Gemini analysis
- Custom: `"Execute cicd-intelligent-recovery using Loop 2 delivery package"`

**MCP Tools**:
- `mcp__claude-flow__swarm_init` - Initialize recovery swarm
- `mcp__claude-flow__agent_spawn` - Spawn analysis agents
- `mcp__claude-flow__task_orchestrate` - Orchestrate fix workflow
- `mcp__ruv-swarm__neural_train` - Train from failures (optional)
- `mcp__flow-nexus__sandbox_execute` - Sandbox validation (optional)

**Evidence-Based Techniques**:
- Gemini large-context (2M token window)
- Byzantine consensus (5/7 analysis, 4/5 theater detection, fix approval)
- Raft consensus (graph-based root cause)
- Program-of-thought (Plan → Execute → Validate → Approve)
- Self-consistency (multiple validators)

**Output**:
- Fixed code with 100% test success
- Failure patterns in `integration/loop3-feedback` namespace

**Memory Namespaces**: `integration/loop2-to-loop3`, `integration/loop3-feedback`

---

## Core Skills Mappings

### Skill: agent-creator

**Purpose**: Create specialized AI agents with optimized system prompts

**Agents**:
- `agent-creator` - Primary agent creation specialist
- `prompt-architect` - Prompt optimization
- `skill-forge` - SOP design

**Commands**:
- Custom: `"Use agent-creator skill to create [agent type]"`

**MCP Tools**:
- `mcp__claude-flow__memory_usage` - Store agent definitions

**Integration**: Used by Loop 2 for custom agent generation

---

### Skill: theater-detection-audit

**Purpose**: Eliminate fake work and validate real implementations

**Agents**:
- `theater-detection-audit` (x3) - Multi-angle detection
- `sandbox-validator` (x2) - Reality validation
- `byzantine-coordinator` - Consensus building

**Commands**:
- `/audit-commands/theater-detect` - Theater detection command
- `/essential-commands/quick-check` - Quick audit

**MCP Tools**:
- `mcp__flow-nexus__sandbox_execute` - Execute in sandbox for validation

**Integration**: Core component of Loop 2 and Loop 3 validation

---

### Skill: functionality-audit

**Purpose**: Validate code actually works through execution

**Agents**:
- `functionality-audit` - Primary audit agent
- `sandbox-validator` - Execution validation
- `test-validity-auditor` - Test correctness

**Commands**:
- `/audit-commands/functionality-audit`
- `/essential-commands/quick-check`

**MCP Tools**:
- `mcp__flow-nexus__sandbox_create` - Create test environment
- `mcp__flow-nexus__sandbox_execute` - Run tests

**Integration**: Used in Loop 2 and Loop 3 for validation

---

### Skill: code-review-assistant

**Purpose**: Comprehensive PR review with multi-agent swarm

**Agents**:
- `code-review-swarm` - Multi-agent coordinator
- `reviewer` (x multiple) - Specialized reviewers (security, performance, style)
- `theater-detection-audit` - Fake work detection
- `byzantine-coordinator` - Review consensus

**Commands**:
- `/essential-commands/review-pr`
- `/github/code-review`

**MCP Tools**:
- `mcp__claude-flow__swarm_init` - Initialize review swarm
- `mcp__claude-flow__task_orchestrate` - Parallel reviews
- `mcp__flow-nexus__github_repo_analyze` - Repository analysis

**Integration**: Can be invoked from Loop 2 or standalone

---

### Skill: smart-bug-fix

**Purpose**: Intelligent bug fixing with root cause analysis

**Agents**:
- `debugger` - Primary debugging agent
- `root-cause-detective` - Root cause analysis
- `fix-strategy-planner` - Fix planning
- `fix-implementation-specialist` - Fix execution
- `sandbox-validator` - Validation

**Commands**:
- `/essential-commands/fix-bug`
- `/sparc:debug`

**MCP Tools**:
- `mcp__claude-flow__agent_spawn` - Spawn debug agents
- `mcp__flow-nexus__sandbox_execute` - Test fixes

**Integration**: Core component of Loop 3, can be used standalone

---

### Skill: feature-dev-complete

**Purpose**: Complete feature development lifecycle

**Agents**:
- `planner` - Feature planning
- `researcher` - Best practices research
- `coder` (x multiple) - Implementation
- `tester` - Comprehensive testing
- `reviewer` - Quality review

**Commands**:
- `/essential-commands/build-feature`
- `/workflows/development`

**MCP Tools**:
- `mcp__claude-flow__swarm_init` - Initialize feature swarm
- `mcp__claude-flow__task_orchestrate` - Orchestrate development
- `mcp__flow-nexus__github_repo_analyze` - Research phase

**Integration**: Can use Loop 1 for planning, Loop 2 for implementation

---

## SPARC Methodology Mappings

### Skill: sparc-methodology

**Purpose**: Complete SPARC workflow orchestration

**Phases & Agents**:

1. **Specification** → `specification` agent
2. **Pseudocode** → `pseudocode` agent
3. **Architecture** → `architecture` agent
4. **Refinement** → `refinement` agent + `tdd-london-swarm`
5. **Completion** → `sparc-coord` orchestrator

**Commands**:
- `/sparc` - Full SPARC workflow
- `/sparc:spec-pseudocode` - Spec + Pseudocode phases
- `/sparc:architect` - Architecture phase
- `/sparc:code` - Refinement + Completion
- `/sparc:tutorial` - SPARC tutorial

**MCP Tools**:
- `mcp__claude-flow__swarm_init` - Initialize SPARC swarm
- `mcp__claude-flow__task_orchestrate` - Phase orchestration

**Integration**: Loop 1 uses Specification phase, Loop 2 uses all phases

---

## Agent-to-Skill Mappings

### Core Development Agents

| Agent | Primary Skills | Secondary Skills |
|-------|---------------|------------------|
| `researcher` | `research-driven-planning` | `feature-dev-complete`, `smart-bug-fix` |
| `coder` | `sparc-coder`, `parallel-swarm-implementation` | `feature-dev-complete` |
| `tester` | `tdd-london-swarm`, `functionality-audit` | All testing needs |
| `reviewer` | `code-review-assistant`, `theater-detection-audit` | `parallel-swarm-implementation` |
| `planner` | `research-driven-planning` | `feature-dev-complete` |
| `analyst` | `code-analyzer`, `performance-analysis` | `cicd-intelligent-recovery` |

### Specialized Development Agents

| Agent | Primary Skills | Use Cases |
|-------|---------------|-----------|
| `backend-dev` | `api-docs`, `sparc-coder` | REST/GraphQL APIs |
| `mobile-dev` | `sparc-coder` | React Native, iOS, Android |
| `ml-developer` | `ml-expert`, `ml-training-debugger`, `flow-nexus-neural` | ML models |
| `cicd-engineer` | `cicd-intelligent-recovery`, `github-workflow-automation` | CI/CD pipelines |
| `gemini-analyst` | `cicd-intelligent-recovery` | Large-context analysis (2M tokens) |

### Swarm Coordination Agents

| Agent | Primary Skills | Coordination Pattern |
|-------|---------------|---------------------|
| `queen-coordinator` | `parallel-swarm-implementation`, `hive-mind-advanced` | Hierarchical (Loop 2 META-SKILL) |
| `hierarchical-coordinator` | `hive-mind-advanced` | Queen-led hierarchy |
| `mesh-coordinator` | `swarm-orchestration` | Peer-to-peer mesh |
| `adaptive-coordinator` | `swarm-advanced` | Dynamic topology switching |
| `byzantine-coordinator` | `hive-mind-advanced` | Consensus building (2/3, 4/5, 5/7) |

### Meta-Tools Agents

| Agent | Primary Skills | Purpose |
|-------|---------------|---------|
| `skill-forge` | `skill-forge`, `prompt-architect` | Create new skills |
| `intent-analyzer` | `intent-analyzer` | Analyze user intent |
| `prompt-architect` | `prompt-architect` | Optimize prompts |
| `agent-creator` | `agent-creator` | Create new agents |
| `cascade-orchestrator` | `cascade-orchestrator` | Multi-skill workflows |

---

## Command-to-Skill Mappings

### Essential Commands

| Command | Skill | Description |
|---------|-------|-------------|
| `/essential-commands/quick-check` | `quick-quality-check` | Lightning-fast quality check |
| `/essential-commands/fix-bug` | `smart-bug-fix` | Intelligent bug fixing |
| `/essential-commands/build-feature` | `feature-dev-complete` | Complete feature development |
| `/essential-commands/review-pr` | `code-review-assistant` | Comprehensive PR review |
| `/essential-commands/deploy-check` | `production-readiness` | Deployment validation |

### SPARC Commands

| Command | Skill | Phase |
|---------|-------|-------|
| `/sparc` | `sparc-methodology` | All phases |
| `/sparc:spec-pseudocode` | `sparc-methodology` | Specification + Pseudocode |
| `/sparc:architect` | `sparc-methodology` | Architecture |
| `/sparc:code` | `sparc-coder` | Implementation |
| `/sparc:debug` | `debugger` | Debugging |
| `/sparc:tutorial` | `sparc-methodology` | Tutorial |

### Audit Commands

| Command | Skill | Audit Type |
|---------|-------|-----------|
| `/audit-commands/theater-detect` | `theater-detection-audit` | Theater detection |
| `/audit-commands/functionality-audit` | `functionality-audit` | Functionality validation |
| `/audit-commands/style-audit` | `style-audit` | Style checking |
| `/audit-commands/audit-pipeline` | All audit skills | Complete pipeline |

### Multi-Model Commands

| Command | Integration | Purpose |
|---------|-------------|---------|
| `/multi-model-commands/gemini-megacontext` | Loop 3 | 2M token analysis |
| `/multi-model-commands/gemini-search` | Loop 1 | Web research |
| `/multi-model-commands/codex-auto` | Loop 2 | Auto-implementation |

### Coordination Commands

| Command | MCP Tool | Purpose |
|---------|----------|---------|
| `/coordination/swarm-init` | `swarm_init` | Initialize swarm |
| `/coordination/agent-spawn` | `agent_spawn` | Spawn agent |
| `/coordination/task-orchestrate` | `task_orchestrate` | Orchestrate task |

### GitHub Commands

| Command | Skill | MCP Tool |
|---------|-------|----------|
| `/github/code-review` | `github-code-review` | `github_repo_analyze` |
| `/github/pr-enhance` | `github-code-review` | `pr_enhance` |
| `/github/issue-triage` | `github-project-management` | `issue_triage` |

---

## MCP Tool Usage by Component

### Claude-Flow MCP Tools (Required)

**Swarm Management**:
- `swarm_init` - Used by: All Three Loops, coordination commands
- `swarm_status` - Used by: Monitoring commands
- `swarm_monitor` - Used by: Real-time monitoring

**Agent Management**:
- `agent_spawn` - Used by: All Three Loops, dynamic agent creation
- `agent_list` - Used by: Monitoring, debugging
- `agent_metrics` - Used by: Performance analysis

**Task Orchestration**:
- `task_orchestrate` - Used by: All Three Loops, workflow coordination
- `task_status` - Used by: Progress tracking
- `task_results` - Used by: Result collection

**Memory & Neural**:
- `memory_usage` - Used by: Cross-loop integration, state persistence
- `neural_status` - Used by: Neural training monitoring
- `neural_train` - Used by: Post-task learning
- `neural_patterns` - Used by: Pattern recognition

### RUV-Swarm MCP Tools (Optional)

**Enhanced Coordination**:
- `swarm_init` - Advanced topology options
- `swarm_monitor` - Real-time dashboards
- `agent_spawn` - Enhanced agent types

**DAA (Decentralized Autonomous Agents)**:
- `daa_agent_create` - Autonomous agent creation
- `daa_workflow_execute` - Self-organizing workflows
- `daa_meta_learning` - Cross-domain learning

**Consensus**:
- Byzantine, Raft, Gossip protocols for distributed coordination

### Flow-Nexus MCP Tools (Optional)

**Sandboxes**:
- `sandbox_create` - Used by: Functionality audit, theater detection
- `sandbox_execute` - Used by: Loop 2 validation, Loop 3 fixes
- `sandbox_configure` - Used by: Custom environments

**Neural AI**:
- `neural_train` - Distributed neural training
- `neural_cluster_init` - Multi-node training clusters
- `neural_train_distributed` - Large-scale training

**Templates**:
- `template_list` - Browse templates
- `template_deploy` - Deploy from template

**GitHub Integration**:
- `github_repo_analyze` - Repository analysis (Loop 1, code review)

**Queen Seraphina**:
- `seraphina_chat` - AI assistant consultation

---

## Complete Integration Map

### Three-Loop System Data Flow

```
User Request
    ↓
┌─────────────────────────────────────────────────────┐
│ LOOP 1: research-driven-planning                    │
│                                                      │
│ Skills: research-driven-planning                    │
│ Agents: 6-agent research + 8-agent pre-mortem      │
│ Commands: /sparc:spec-pseudocode                   │
│ MCP: swarm_init, agent_spawn, memory_usage         │
│ Techniques: Self-consistency, Byzantine (2/3)      │
│                                                      │
│ Output: Planning Package                           │
└──────────────────┬──────────────────────────────────┘
                   ↓ [Memory: integration/loop1-to-loop2]
┌─────────────────────────────────────────────────────┐
│ LOOP 2: parallel-swarm-implementation (META-SKILL)  │
│                                                      │
│ Skills: parallel-swarm-implementation               │
│ Agents: Queen + Dynamic 86-agent selection         │
│        + 6-agent theater detection                 │
│ Commands: /sparc:code, /coordination/swarm-init    │
│ MCP: swarm_init, agent_spawn (dynamic),            │
│      task_orchestrate, sandbox_execute             │
│ Techniques: Byzantine (4/5), Self-consistency,     │
│             Hierarchical coordination              │
│                                                      │
│ Output: Implementation Package                     │
└──────────────────┬──────────────────────────────────┘
                   ↓ [Memory: integration/loop2-to-loop3]
┌─────────────────────────────────────────────────────┐
│ LOOP 3: cicd-intelligent-recovery                   │
│                                                      │
│ Skills: cicd-intelligent-recovery                   │
│ Agents: Gemini analyst + 7-agent analysis (5/7)    │
│        + Graph root cause (Raft)                   │
│        + Program-of-thought fixes                  │
│        + 6-agent theater (4/5)                     │
│ Commands: /sparc:debug, /gemini-megacontext        │
│ MCP: swarm_init, task_orchestrate,                 │
│      sandbox_execute, neural_train                 │
│ Techniques: Gemini 2M, Byzantine (5/7, 4/5),       │
│             Raft, Program-of-thought               │
│                                                      │
│ Output: Fixed Code + Failure Patterns             │
└──────────────────┬──────────────────────────────────┘
                   ↓ [Memory: integration/loop3-feedback]
                   ↓ (Feeds back to Loop 1 for next iteration)
                 ┌─┘
                 └─→ Continuous Improvement
```

### Skill-Agent-Command-MCP Integration Example

**Scenario**: User requests "Fix authentication bug"

1. **Command**: `/essential-commands/fix-bug` triggered
2. **Skill**: `smart-bug-fix` activated
3. **Agents Spawned**:
   - `debugger` - Primary debugging
   - `root-cause-detective` - Root cause analysis
   - `fix-strategy-planner` - Planning fix
   - `fix-implementation-specialist` - Implementing fix
   - `sandbox-validator` - Validating fix
4. **MCP Tools Used**:
   - `mcp__claude-flow__swarm_init` - Initialize debug swarm
   - `mcp__claude-flow__agent_spawn` - Spawn debug agents
   - `mcp__flow-nexus__sandbox_execute` - Test fix in isolation
5. **Hooks Triggered**:
   - `pre-task-coordination` - Initialize coordination
   - `post-edit-formatting` - Auto-format fixed code
   - `post-task-neural-training` - Learn from fix
6. **Evidence-Based Techniques**:
   - Program-of-thought (fix planning and execution)
   - Self-consistency (multiple validation checks)

---

## Usage Patterns

### Pattern 1: Complete Three-Loop Workflow

```bash
# Execute all three loops
"Execute the complete Three-Loop Integrated Development System:
1. research-driven-planning: Research + 5x pre-mortem
2. parallel-swarm-implementation: Dynamic agent selection + theater detection
3. cicd-intelligent-recovery: Gemini analysis + intelligent fixes

Project: Build user authentication system with JWT and OAuth2"
```

**Mappings**:
- Skills: `research-driven-planning` → `parallel-swarm-implementation` → `cicd-intelligent-recovery`
- Agents: 14+ (Loop 1) → Dynamic (Loop 2) → 20+ (Loop 3)
- MCP Tools: All claude-flow tools, optional flow-nexus for sandboxes
- Memory Flow: `loop1-to-loop2` → `loop2-to-loop3` → `loop3-feedback`

---

### Pattern 2: Individual Skill Usage

```bash
# Use specific skill
"Execute theater-detection-audit skill on authentication module"
```

**Mappings**:
- Skill: `theater-detection-audit`
- Agents: `theater-detection-audit` (x3), `sandbox-validator` (x2), `byzantine-coordinator`
- Command: `/audit-commands/theater-detect`
- MCP Tools: `sandbox_execute` (flow-nexus optional)
- Technique: Byzantine consensus (4/5 agreement)

---

### Pattern 3: SPARC Workflow

```bash
# Full SPARC methodology
/sparc "Implement real-time chat feature with WebSocket"
```

**Mappings**:
- Skill: `sparc-methodology`
- Agents: `specification` → `pseudocode` → `architecture` → `refinement` → `sparc-coord`
- Commands: All `/sparc:*` commands
- MCP Tools: `swarm_init`, `task_orchestrate`
- Integration: Can integrate with Loop 2 for implementation phase

---

### Pattern 4: Multi-Model Integration

```bash
# Gemini large-context analysis
/multi-model-commands/gemini-megacontext "Analyze entire codebase for authentication vulnerabilities"
```

**Mappings**:
- Skill: `cicd-intelligent-recovery` (uses Gemini internally)
- Agent: `gemini-analyst`
- MCP Tool: Gemini API integration
- Window: 2M tokens
- Use Case: Full codebase analysis in Loop 3

---

## Summary Statistics

### Component Counts

- **Skills**: 104 (comprehensive coverage)
- **Agents**: 86+ (with explicit SOPs)
- **Commands**: 138 (organized by category)
- **MCP Servers**: 3 (1 required, 2 optional)
- **Hooks**: 15+ (automated workflows)

### Evidence-Based Techniques Coverage

| Technique | Agent Count | Usage |
|-----------|-------------|-------|
| Self-consistency | 25 | Cross-validation, multiple validators |
| Byzantine consensus | 15 | Fault-tolerant agreement (2/3, 4/5, 5/7) |
| Raft consensus | 8 | Leader-based coordination |
| Program-of-thought | 18 | Explicit reasoning |
| Plan-and-solve | 22 | Phase structure |
| Gossip consensus | 3 | Eventually consistent |
| CRDT synchronization | 3 | Conflict-free replication |

### Integration Points

- **Loop 1 → Loop 2**: Memory namespace `integration/loop1-to-loop2`
- **Loop 2 → Loop 3**: Memory namespace `integration/loop2-to-loop3`
- **Loop 3 → Loop 1**: Memory namespace `integration/loop3-feedback`
- **Cross-Session**: Persistent memory via hooks
- **Real-Time**: Live monitoring via MCP tools

---

## Quick Reference

### Find the Right Skill

- **Planning/Research**: `research-driven-planning`
- **Implementation**: `parallel-swarm-implementation`
- **Bug Fixing**: `smart-bug-fix`, `cicd-intelligent-recovery`
- **Code Review**: `code-review-assistant`
- **Testing**: `functionality-audit`, `tdd-london-swarm`
- **Theater Detection**: `theater-detection-audit`
- **Feature Development**: `feature-dev-complete`
- **SPARC Workflow**: `sparc-methodology`

### Find the Right Agent

- **Research**: `researcher`, `web-researcher`, `github-analyst`
- **Implementation**: `coder`, `sparc-coder`
- **Testing**: `tester`, `sandbox-validator`
- **Analysis**: `analyst`, `gemini-analyst`, `graph-analyst`
- **Coordination**: `queen-coordinator`, `hierarchical-coordinator`, `byzantine-coordinator`
- **Debugging**: `debugger`, `root-cause-detective`

### Find the Right Command

- **Quick Actions**: `/essential-commands/*`
- **SPARC**: `/sparc:*`
- **Audit**: `/audit-commands/*`
- **Coordination**: `/coordination/*`
- **GitHub**: `/github/*`

---

**Plugin Version**: 2.0.0
**Documentation**: See `README.md` for installation and usage
**Repository**: https://github.com/yourusername/ruv-sparc-three-loop-system

