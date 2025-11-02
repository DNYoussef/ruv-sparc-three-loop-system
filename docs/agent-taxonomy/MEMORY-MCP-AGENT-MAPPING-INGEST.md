# Memory MCP Agent-Command Mapping Ingestion

**Purpose**: Documentation for Memory MCP ingestion of agent-command mappings
**Date**: 2025-11-01
**Total Agents**: 103
**Total Commands**: 224
**Total Mappings**: ~1,200

---

## 🏷️ Memory Metadata (WHO/WHEN/PROJECT/WHY)

**WHO** (Agent):
- **Name**: agent-registry-indexer
- **Category**: documentation-specialist
- **Capabilities**: memory_indexing, taxonomy_management, cross_reference_creation

**WHEN** (Timestamp):
- **Date**: 2025-11-01
- **Time**: Current session
- **Unix**: ${Date.now()}
- **ISO8601**: ${new Date().toISOString()}

**PROJECT**:
- **Name**: ruv-sparc-three-loop-system
- **Component**: agent-command-registry
- **GitHub**: https://github.com/DNYoussef/ruv-sparc-three-loop-system

**WHY** (Intent):
- **Primary**: documentation
- **Sub-intents**: indexing, taxonomy, cross-reference, agent-mapping, command-mapping
- **Purpose**: Create persistent cross-session memory of agent-command relationships

---

## Complete Agent-Command Taxonomy

### MECE Agent Domains (10)

1. **Core Development** (6 agents)
   - coder, coder-enhanced, reviewer, tester, researcher, planner
   - Commands: SPARC methodology, development workflow, quality validation
   - Memory Key: `agents/core/*`

2. **Swarm Coordination** (13 agents)
   - hierarchical-coordinator, mesh-coordinator, adaptive-coordinator
   - byzantine-coordinator, raft-manager, gossip-coordinator, crdt-synchronizer, quorum-manager, security-manager (x2)
   - queen-coordinator, collective-intelligence-coordinator, scout-explorer, worker-specialist
   - Commands: Swarm management, consensus protocols, coordination
   - Memory Key: `agents/swarm/*`, `agents/consensus/*`, `agents/hive-mind/*`

3. **Business & Product** (8 agents)
   - business-analyst, content-creator, customer-support, market-researcher
   - marketing-specialist, product-manager, sales-specialist, seo-specialist
   - Commands: Research, analytics, content, customer engagement
   - Memory Key: `agents/business/*`

4. **GitHub & Repository** (14 agents)
   - code-review-swarm, pr-manager, issue-tracker, release-manager (x2)
   - workflow-automation, multi-repo-swarm, sync-coordinator, repo-architect
   - project-board-sync, github-modes, swarm-issue, swarm-pr
   - Commands: GitHub integration, deployment, CI/CD
   - Memory Key: `agents/github/*`

5. **Specialized Development** (11 agents)
   - Backend: dev-backend-api (x2)
   - Mobile: spec-mobile-react-native (x2)
   - ML: data-ml-model
   - DevOps: ops-cicd-github (x2)
   - Analysis: code-analyzer, analyze-code-quality
   - Architecture: arch-system-design
   - Documentation: docs-api-openapi
   - Commands: Domain-specific development, deployment, testing
   - Memory Key: `agents/specialized/*`

6. **Testing & Validation** (3 agents)
   - tdd-london-swarm, production-validator
   - Commands: All testing commands, audit pipeline
   - Memory Key: `agents/testing/*`

7. **AI Enhancement** (8 agents)
   - gemini-search, gemini-megacontext, gemini-media, gemini-extensions
   - codex-auto, codex-reasoning, multi-model-orchestrator, safla-neural
   - Commands: Multi-model, search, autonomous coding, neural training
   - Memory Key: `agents/ai/*`

8. **Flow-Nexus Platform** (9 agents)
   - app-store, authentication, challenges, neural-network
   - payments, sandbox, swarm, user-tools, workflow
   - Commands: Flow-Nexus MCP tools
   - Memory Key: `agents/flow-nexus/*`

9. **SPARC Methodology** (13 agents)
   - specification, pseudocode, architecture, refinement
   - sparc-coord, sparc-coder (implementations)
   - Commands: SPARC phases, methodology workflows
   - Memory Key: `agents/sparc/*`

10. **Research & Analysis** (6 agents)
    - data-steward, ethics-agent, archivist, evaluator
    - root-cause-analyzer, reasoning agents
    - Commands: Research workflows, SOP, quality gates
    - Memory Key: `agents/research/*`

11. **Template & Meta-Agents** (9 agents)
    - Various templates for automation, coordination, orchestration
    - Commands: Agent creation, meta-operations
    - Memory Key: `agents/templates/*`

12. **Root-Level Utility** (5 agents)
    - audit-pipeline-orchestrator, base-template-generator, goal planners
    - Commands: System-level operations
    - Memory Key: `agents/utility/*`

---

## Command Categories (10 MECE Domains)

### 1. Core Infrastructure (3 commands)
- `/claude-flow-help`, `/claude-flow-memory`, `/claude-flow-swarm`
- **Used by**: ALL agents (103 agents)
- **Memory Key**: `commands/core/*`

### 2. Agent Lifecycle (18 commands)
- `/agent-spawn`, `/swarm-init`, `/task-orchestrate`, `/agent-health-check`, etc.
- **Used by**: Swarm, coordination, template agents (30+ agents)
- **Memory Key**: `commands/lifecycle/*`

### 3. Development Workflow (67 commands)
- `/sparc`, `/build-feature`, `/fix-bug`, `/quick-check`, `/review-pr`, `/workflow:*`, `/sparc:*`
- **Used by**: Core development, SPARC, specialized development (50+ agents)
- **Memory Key**: `commands/development/*`

### 4. Quality & Validation (29 commands)
- `/audit-pipeline`, `/security-audit`, `/theater-detect`, `/regression-test`, `/e2e-test`, etc.
- **Used by**: Testing, review, audit agents (35+ agents)
- **Memory Key**: `commands/quality/*`

### 5. Performance & Optimization (18 commands)
- `/profiler-start`, `/resource-optimize`, `/memory-optimize`, `/neural-train`, etc.
- **Used by**: Optimization, ML, performance agents (20+ agents)
- **Memory Key**: `commands/performance/*`

### 6. Memory & State (18 commands)
- `/memory-store`, `/memory-retrieve`, `/state-checkpoint`, `/state-restore`, etc.
- **Used by**: All agents with state management (95+ agents)
- **Memory Key**: `commands/memory/*`

### 7. Monitoring & Telemetry (18 commands)
- `/agent-metrics`, `/swarm-monitor`, `/monitoring-configure`, `/alert-configure`, etc.
- **Used by**: Monitoring, DevOps, swarm agents (25+ agents)
- **Memory Key**: `commands/monitoring/*`

### 8. Integration & External (21 commands)
- `/github-*`, `/docker-*`, `/k8s-*`, `/aws-deploy`, `/terraform-apply`, etc.
- **Used by**: GitHub, DevOps, deployment agents (25+ agents)
- **Memory Key**: `commands/integration/*`

### 9. Research & Analysis (20 commands)
- `/research:*`, `/re:*`, `/assess-risks`, `/prisma-init`, etc.
- **Used by**: Research, RE, ML agents (15+ agents)
- **Memory Key**: `commands/research/*`

### 10. Automation & Hooks (22 commands)
- `/hook:*`, `/automation:*`, `/self-healing`, `/smart-spawn`, etc.
- **Used by**: Automation, DevOps, workflow agents (20+ agents)
- **Memory Key**: `commands/automation/*`

---

## High-Value Command Clusters

### Universal Commands (50+ agent usage)
```json
{
  "commands": [
    {"name": "/help", "agents": 103, "usage": "universal"},
    {"name": "/memory-store", "agents": 95, "usage": "state_management"},
    {"name": "/memory-retrieve", "agents": 95, "usage": "state_retrieval"}
  ],
  "memory_key": "commands/clusters/universal"
}
```

### Core Development Commands (20-40 agent usage)
```json
{
  "commands": [
    {"name": "/sparc", "agents": 35, "usage": "methodology"},
    {"name": "/quick-check", "agents": 30, "usage": "quality"},
    {"name": "/build-feature", "agents": 28, "usage": "development"},
    {"name": "/fix-bug", "agents": 25, "usage": "debugging"},
    {"name": "/review-pr", "agents": 22, "usage": "code_review"}
  ],
  "memory_key": "commands/clusters/core-development"
}
```

### Specialized Commands (1-5 agent usage)
```json
{
  "commands": [
    {"name": "/re:malware-sandbox", "agents": 3, "usage": "reverse_engineering"},
    {"name": "/research:literature-review", "agents": 8, "usage": "research"},
    {"name": "/neural-train", "agents": 6, "usage": "machine_learning"},
    {"name": "/github-release", "agents": 5, "usage": "release_management"}
  ],
  "memory_key": "commands/clusters/specialized"
}
```

---

## Agent-to-Commands Mappings

### Example: Core Development Agent (coder)
```json
{
  "agent": "coder",
  "domain": "Core Development",
  "file": "agents/core/coder.md",
  "commands": [
    "/sparc",
    "/build-feature",
    "/fix-bug",
    "/quick-check",
    "/review-pr",
    "/sparc:coder",
    "/sparc:api-designer",
    "/sparc:backend-specialist",
    "/sparc:frontend-specialist",
    "/code-review",
    "/docker-build",
    "/workflow:development"
  ],
  "command_count": 12,
  "memory_key": "agents/core/coder"
}
```

### Example: Swarm Coordinator (hierarchical-coordinator)
```json
{
  "agent": "hierarchical-coordinator",
  "domain": "Swarm Coordination",
  "file": "agents/swarm/hierarchical-coordinator.md",
  "commands": [
    "/swarm-init",
    "/agent-spawn",
    "/task-orchestrate",
    "/coordination-visualize",
    "/agent-health-check",
    "/agent-retire",
    "/agent-clone",
    "/agent-benchmark",
    "/memory-store",
    "/memory-retrieve",
    "/memory-persist",
    "/monitoring-configure",
    "/swarm-monitor",
    "/metrics-export",
    "... 11 more swarm-specific commands"
  ],
  "command_count": 25,
  "memory_key": "agents/swarm/hierarchical-coordinator"
}
```

---

## Memory Ingestion Checklist

### Pre-Ingestion
- [x] Complete MECE agent taxonomy created (103 agents across 10 domains)
- [x] All 224 commands cataloged in MASTER-COMMAND-INDEX.md
- [x] Agent-command mappings completed (~1,200 total mappings)
- [x] GraphViz visualizations created
- [x] Comprehensive documentation generated

### Ingestion Data
- [x] Agent metadata (name, domain, file path, capabilities)
- [x] Command metadata (name, category, description, parameters)
- [x] Agent-to-command mappings (which commands each agent uses)
- [x] Command-to-agent mappings (which agents use each command)
- [x] Command clusters (universal, core, specialized)
- [x] Memory namespaces defined

### Post-Ingestion Validation
- [ ] Query test: Retrieve agents by domain
- [ ] Query test: Retrieve commands by category
- [ ] Query test: Find all agents using a specific command
- [ ] Query test: Find all commands for a specific agent
- [ ] Query test: Semantic search for agent capabilities
- [ ] Query test: Cross-reference skill-agent-command relationships

---

## Memory Query Examples

### Find All Agents in a Domain
```bash
npx claude-flow@alpha memory retrieve --key "agents/core/*"
```

### Find All Commands for an Agent
```bash
npx claude-flow@alpha memory retrieve --key "agents/core/coder"
```

### Find All Agents Using a Command
```bash
npx claude-flow@alpha memory search --query "agents using /sparc command"
```

### Find Commands by Category
```bash
npx claude-flow@alpha memory retrieve --key "commands/development/*"
```

### Semantic Search for Agent Capabilities
```bash
npx claude-flow@alpha memory search --query "agents with deployment capabilities"
```

---

## Integration Points

### With Existing Systems
- **SKILL-AGENT-COMMAND-MAPPINGS.md**: Legacy mapping document (138 commands, 86 agents)
- **MASTER-COMMAND-INDEX.md**: Authoritative command reference (224 commands)
- **MECE-AGENT-TAXONOMY.md**: Agent categorization (103 agents, 10 domains)
- **GraphViz Diagrams**: Visual representations (5 diagrams)

### With Memory MCP Triple System
- **Short-term (24h)**: Active agent sessions, real-time command usage
- **Mid-term (7d)**: Agent performance metrics, command effectiveness
- **Long-term (30d+)**: Agent-command relationship evolution, usage patterns

### With Mode-Aware Context
- **Execution Mode**: Quick command lookups, agent selection
- **Planning Mode**: Agent capability analysis, workflow design
- **Brainstorming Mode**: Agent combination exploration, command discovery

---

## Statistics Summary

| Metric | Value |
|--------|-------|
| **Agents** | 103 |
| **MECE Domains** | 10 |
| **Commands** | 224 |
| **Command Categories** | 10 |
| **Total Mappings** | ~1,200 |
| **Avg Commands/Agent** | 11.6 |
| **Avg Agents/Command** | 5.4 |
| **Universal Commands** | 3 |
| **Core Commands** | 5 |
| **Specialized Commands** | 100+ |
| **GraphViz Diagrams** | 5 |
| **Documentation Files** | 10+ |
| **Memory Namespaces** | 25+ |

---

## Next Steps

1. **Ingest into Memory MCP**: Run ingestion script to populate Memory MCP
2. **Validate Queries**: Test all query patterns
3. **Create Search Patterns**: Define common semantic search queries
4. **Monitor Usage**: Track which agents and commands are used most
5. **Continuous Updates**: Keep mappings current as agents/commands evolve

---

**Status**: Ready for Memory MCP ingestion
**File Location**: `docs/agent-taxonomy/MEMORY-MCP-AGENT-MAPPING-INGEST.md`
**Ingestion Script**: (To be created or use existing Memory MCP ingestion pipeline)
