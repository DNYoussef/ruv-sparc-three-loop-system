# Claude Code Configuration - SPARC Development Environment

## 🚨 CRITICAL: CONCURRENT EXECUTION & FILE MANAGEMENT

**ABSOLUTE RULES**:
1. ALL operations MUST be concurrent/parallel in a single message
2. **NEVER save working files, text/mds and tests to the root folder**
3. ALWAYS organize files in appropriate subdirectories
4. **USE CLAUDE CODE'S TASK TOOL** for spawning agents concurrently, not just MCP

### ⚡ GOLDEN RULE: "1 MESSAGE = ALL RELATED OPERATIONS"

**MANDATORY PATTERNS:**
- **TodoWrite**: ALWAYS batch ALL todos in ONE call (5-10+ todos minimum)
- **Task tool (Claude Code)**: ALWAYS spawn ALL agents in ONE message with full instructions
- **File operations**: ALWAYS batch ALL reads/writes/edits in ONE message
- **Bash commands**: ALWAYS batch ALL terminal operations in ONE message
- **Memory operations**: ALWAYS batch ALL memory store/retrieve in ONE message

### 🎯 CRITICAL: Claude Code Task Tool for Agent Execution

**Claude Code's Task tool is the PRIMARY way to spawn agents:**
```javascript
// ✅ CORRECT: Use Claude Code's Task tool for parallel agent execution
[Single Message]:
  Task("Research agent", "Analyze requirements and patterns...", "researcher")
  Task("Coder agent", "Implement core features...", "coder")
  Task("Tester agent", "Create comprehensive tests...", "tester")
  Task("Reviewer agent", "Review code quality...", "reviewer")
  Task("Architect agent", "Design system architecture...", "system-architect")
```

**MCP tools are ONLY for coordination setup:**
- `mcp__claude-flow__swarm_init` - Initialize coordination topology
- `mcp__claude-flow__agent_spawn` - Define agent types for coordination
- `mcp__claude-flow__task_orchestrate` - Orchestrate high-level workflows

### 📁 File Organization Rules

**NEVER save to root folder. Use these directories:**
- `/src` - Source code files
- `/tests` - Test files
- `/docs` - Documentation and markdown files
- `/config` - Configuration files
- `/scripts` - Utility scripts
- `/examples` - Example code

## Project Overview

This project uses SPARC (Specification, Pseudocode, Architecture, Refinement, Completion) methodology with Claude-Flow orchestration for systematic Test-Driven Development.

## SPARC Commands

### Core Commands
- `npx claude-flow sparc modes` - List available modes
- `npx claude-flow sparc run <mode> "<task>"` - Execute specific mode
- `npx claude-flow sparc tdd "<feature>"` - Run complete TDD workflow
- `npx claude-flow sparc info <mode>` - Get mode details

### Batchtools Commands
- `npx claude-flow sparc batch <modes> "<task>"` - Parallel execution
- `npx claude-flow sparc pipeline "<task>"` - Full pipeline processing
- `npx claude-flow sparc concurrent <mode> "<tasks-file>"` - Multi-task processing

### Build Commands
- `npm run build` - Build project
- `npm run test` - Run tests
- `npm run lint` - Linting
- `npm run typecheck` - Type checking

## SPARC Workflow Phases

1. **Specification** - Requirements analysis (`sparc run spec-pseudocode`)
2. **Pseudocode** - Algorithm design (`sparc run spec-pseudocode`)
3. **Architecture** - System design (`sparc run architect`)
4. **Refinement** - TDD implementation (`sparc tdd`)
5. **Completion** - Integration (`sparc run integration`)

## Code Style & Best Practices

- **Modular Design**: Files under 500 lines
- **Environment Safety**: Never hardcode secrets
- **Test-First**: Write tests before implementation
- **Clean Architecture**: Separate concerns
- **Documentation**: Keep updated

## 🚀 Available Agents (131 Total)

### Core Development (8 agents)
`coder`, `coder-enhanced`, `reviewer`, `tester`, `planner`, `researcher`, `api-designer`, `technical-debt-manager`

### Testing & Validation (9 agents) 🆕
`tdd-london-swarm`, `production-validator`, `e2e-testing-specialist`, `performance-testing-agent`, `security-testing-agent`, `visual-regression-agent`, `contract-testing-agent`, `chaos-engineering-agent`, `audit-pipeline-orchestrator`

### Frontend Development (6 agents) 🆕
`react-developer`, `vue-developer`, `ui-component-builder`, `css-styling-specialist`, `accessibility-specialist`, `frontend-performance-optimizer`

### Database & Data (7 agents) 🆕
`database-design-specialist`, `query-optimization-agent`, `database-migration-agent`, `data-pipeline-engineer`, `cache-strategy-agent`, `database-backup-recovery-agent`, `data-ml-model`

### Documentation & Knowledge (6 agents) 🆕
`api-documentation-specialist`, `developer-documentation-agent`, `knowledge-base-manager`, `technical-writing-agent`, `architecture-diagram-generator`, `docs-api-openapi`

### Swarm Coordination (15 agents)
`hierarchical-coordinator`, `mesh-coordinator`, `adaptive-coordinator`, `collective-intelligence-coordinator`, `swarm-memory-manager`, `consensus-validator`, `swarm-health-monitor`, and 8 more specialized coordinators

### Consensus & Distributed
`byzantine-coordinator`, `raft-manager`, `gossip-coordinator`, `consensus-builder`, `crdt-synchronizer`, `quorum-manager`, `security-manager`

### Performance & Optimization
`perf-analyzer`, `performance-benchmarker`, `task-orchestrator`, `memory-coordinator`, `smart-agent`

### GitHub & Repository
`github-modes`, `pr-manager`, `code-review-swarm`, `issue-tracker`, `release-manager`, `workflow-automation`, `project-board-sync`, `repo-architect`, `multi-repo-swarm`

### SPARC Methodology
`sparc-coord`, `sparc-coder`, `specification`, `pseudocode`, `architecture`, `refinement`

### Specialized Development
`backend-dev`, `mobile-dev`, `ml-developer`, `cicd-engineer`, `api-docs`, `system-architect`, `code-analyzer`, `base-template-generator`

### Testing & Validation
`tdd-london-swarm`, `production-validator`

### Migration & Planning
`migration-planner`, `swarm-init`

### Deep Research SOP (Quality Gate System)
`data-steward`, `ethics-agent`, `archivist`, `evaluator`

## 🎯 Claude Code vs MCP Tools

### Claude Code Handles ALL EXECUTION:
- **Task tool**: Spawn and run agents concurrently for actual work
- File operations (Read, Write, Edit, MultiEdit, Glob, Grep)
- Code generation and programming
- Bash commands and system operations
- Implementation work
- Project navigation and analysis
- TodoWrite and task management
- Git operations
- Package management
- Testing and debugging

### MCP Tools ONLY COORDINATE:
- Swarm initialization (topology setup)
- Agent type definitions (coordination patterns)
- Task orchestration (high-level planning)
- Memory management
- Neural features
- Performance tracking
- GitHub integration

**KEY**: MCP coordinates the strategy, Claude Code's Task tool executes with real agents.

## 🚀 Quick Setup

```bash
# Add MCP servers (Claude Flow required, others optional)
claude mcp add claude-flow npx claude-flow@alpha mcp start
claude mcp add ruv-swarm npx ruv-swarm mcp start  # Optional: Enhanced coordination
claude mcp add flow-nexus npx flow-nexus@latest mcp start  # Optional: Cloud features
```

## MCP Tool Categories

### Coordination
`swarm_init`, `agent_spawn`, `task_orchestrate`

### Monitoring
`swarm_status`, `agent_list`, `agent_metrics`, `task_status`, `task_results`

### Memory & Neural
`memory_usage`, `neural_status`, `neural_train`, `neural_patterns`

### GitHub Integration
`github_swarm`, `repo_analyze`, `pr_enhance`, `issue_triage`, `code_review`

### System
`benchmark_run`, `features_detect`, `swarm_monitor`

### Memory MCP - Persistent Cross-Session Context (PRODUCTION READY)
**Integrated**: 2025-11-01 | **Status**: GLOBAL ACCESS FOR ALL AGENTS

Memory MCP Triple System provides persistent memory with automatic tagging protocol:

**Configuration**: `C:\Users\17175\AppData\Roaming\Claude\claude_desktop_config.json`

**Available Tools**:
- `vector_search`: Semantic search with mode-aware context adaptation (5-20 results)
- `memory_store`: Store information with automatic layer assignment (24h/7d/30d+ retention)

**Key Features**:
- Triple-layer retention: Short-term (24h), Mid-term (7d), Long-term (30d+)
- Mode-aware context: 3 interaction modes (Execution/Planning/Brainstorming)
- 29 detection patterns for automatic mode classification
- 384-dimensional vector embeddings with HNSW indexing
- ChromaDB backend with semantic chunking

**Tagging Protocol** (REQUIRED for ALL writes):
ALL Memory MCP writes must include metadata tags:
1. **WHO**: Agent name, category, capabilities
2. **WHEN**: ISO timestamp, Unix timestamp, readable format
3. **PROJECT**: connascence-analyzer, memory-mcp-triple-system, claude-flow, etc.
4. **WHY**: Intent (implementation, bugfix, refactor, testing, documentation, analysis, planning, research)

**Implementation**: `C:\Users\17175\hooks\12fa\memory-mcp-tagging-protocol.js`

```javascript
const { taggedMemoryStore } = require('./hooks/12fa/memory-mcp-tagging-protocol.js');

// Auto-tagged memory write
const tagged = taggedMemoryStore('coder', 'Implemented auth feature', { task_id: 'AUTH-123' });
// Automatically includes: agent metadata, timestamps, project, intent
```

**Agent Access**:
- **ALL 37 agents** have access to Memory MCP
- Automatic metadata injection via tagging protocol
- Intent analyzer auto-detects purpose from content
- Cross-session persistence for all operations

**Documentation**: `C:\Users\17175\docs\integration-plans\MCP-INTEGRATION-GUIDE.md`

### Connascence Analyzer - Code Quality & Coupling Detection (PRODUCTION READY)
**Integrated**: 2025-11-01 | **Status**: CODE QUALITY AGENTS ONLY (14 agents)

Connascence Safety Analyzer detects 7+ violation types including NASA compliance:

**Configuration**: `C:\Users\17175\AppData\Roaming\Claude\claude_desktop_config.json`

**Available Tools**:
- `analyze_file`: Analyze single file for connascence violations and code quality
- `analyze_workspace`: Analyze entire workspace with pattern detection
- `health_check`: Verify analyzer server status

**Detection Capabilities**:
1. God Objects (26 methods vs 15 threshold)
2. Parameter Bombs/CoP (14 params vs 6 NASA limit)
3. Cyclomatic Complexity (13 vs 10 threshold)
4. Deep Nesting (8 levels vs 4 NASA limit)
5. Long Functions (72 lines vs 50 threshold)
6. Magic Literals/CoM (hardcoded ports, timeouts)
7. Configuration values, duplicate code, security violations

**Performance**: 7 violations detected in 0.018 seconds

**Agent Access** (14 Code Quality Agents ONLY):
- coder, reviewer, tester, code-analyzer
- functionality-audit, theater-detection-audit, production-validator
- sparc-coder, analyst, backend-dev, mobile-dev
- ml-developer, base-template-generator, code-review-swarm

**Planning agents do NOT have access** (prevents non-code agents from using code analysis tools)

**Documentation**: `C:\Users\17175\docs\integration-plans\MCP-INTEGRATION-GUIDE.md`

### Flow-Nexus MCP Tools (Optional Advanced Features)
Flow-Nexus extends MCP capabilities with 70+ cloud-based orchestration tools:

**Key MCP Tool Categories:**
- **Swarm & Agents**: `swarm_init`, `swarm_scale`, `agent_spawn`, `task_orchestrate`
- **Sandboxes**: `sandbox_create`, `sandbox_execute`, `sandbox_upload` (cloud execution)
- **Templates**: `template_list`, `template_deploy` (pre-built project templates)
- **Neural AI**: `neural_train`, `neural_patterns`, `seraphina_chat` (AI assistant)
- **GitHub**: `github_repo_analyze`, `github_pr_manage` (repository management)
- **Real-time**: `execution_stream_subscribe`, `realtime_subscribe` (live monitoring)
- **Storage**: `storage_upload`, `storage_list` (cloud file management)

**Authentication Required:**
- Register: `mcp__flow-nexus__user_register` or `npx flow-nexus@latest register`
- Login: `mcp__flow-nexus__user_login` or `npx flow-nexus@latest login`
- Access 70+ specialized MCP tools for advanced orchestration

## 🚀 Agent Execution Flow with Claude Code

### The Correct Pattern:

1. **Optional**: Use MCP tools to set up coordination topology
2. **REQUIRED**: Use Claude Code's Task tool to spawn agents that do actual work
3. **REQUIRED**: Each agent runs hooks for coordination
4. **REQUIRED**: Batch all operations in single messages

### Example Full-Stack Development:

```javascript
// Single message with all agent spawning via Claude Code's Task tool
[Parallel Agent Execution]:
  Task("Backend Developer", "Build REST API with Express. Use hooks for coordination.", "backend-dev")
  Task("Frontend Developer", "Create React UI. Coordinate with backend via memory.", "coder")
  Task("Database Architect", "Design PostgreSQL schema. Store schema in memory.", "code-analyzer")
  Task("Test Engineer", "Write Jest tests. Check memory for API contracts.", "tester")
  Task("DevOps Engineer", "Setup Docker and CI/CD. Document in memory.", "cicd-engineer")
  Task("Security Auditor", "Review authentication. Report findings via hooks.", "reviewer")
  
  // All todos batched together
  TodoWrite { todos: [...8-10 todos...] }
  
  // All file operations together
  Write "backend/server.js"
  Write "frontend/App.jsx"
  Write "database/schema.sql"
```

## 📋 Agent Coordination Protocol

### Every Agent Spawned via Task Tool MUST:

**1️⃣ BEFORE Work:**
```bash
npx claude-flow@alpha hooks pre-task --description "[task]"
npx claude-flow@alpha hooks session-restore --session-id "swarm-[id]"
```

**2️⃣ DURING Work:**
```bash
npx claude-flow@alpha hooks post-edit --file "[file]" --memory-key "swarm/[agent]/[step]"
npx claude-flow@alpha hooks notify --message "[what was done]"
```

**3️⃣ AFTER Work:**
```bash
npx claude-flow@alpha hooks post-task --task-id "[task]"
npx claude-flow@alpha hooks session-end --export-metrics true
```

## 🎯 Concurrent Execution Examples

### ✅ CORRECT WORKFLOW: MCP Coordinates, Claude Code Executes

```javascript
// Step 1: MCP tools set up coordination (optional, for complex tasks)
[Single Message - Coordination Setup]:
  mcp__claude-flow__swarm_init { topology: "mesh", maxAgents: 6 }
  mcp__claude-flow__agent_spawn { type: "researcher" }
  mcp__claude-flow__agent_spawn { type: "coder" }
  mcp__claude-flow__agent_spawn { type: "tester" }

// Step 2: Claude Code Task tool spawns ACTUAL agents that do the work
[Single Message - Parallel Agent Execution]:
  // Claude Code's Task tool spawns real agents concurrently
  Task("Research agent", "Analyze API requirements and best practices. Check memory for prior decisions.", "researcher")
  Task("Coder agent", "Implement REST endpoints with authentication. Coordinate via hooks.", "coder")
  Task("Database agent", "Design and implement database schema. Store decisions in memory.", "code-analyzer")
  Task("Tester agent", "Create comprehensive test suite with 90% coverage.", "tester")
  Task("Reviewer agent", "Review code quality and security. Document findings.", "reviewer")
  
  // Batch ALL todos in ONE call
  TodoWrite { todos: [
    {id: "1", content: "Research API patterns", status: "in_progress", priority: "high"},
    {id: "2", content: "Design database schema", status: "in_progress", priority: "high"},
    {id: "3", content: "Implement authentication", status: "pending", priority: "high"},
    {id: "4", content: "Build REST endpoints", status: "pending", priority: "high"},
    {id: "5", content: "Write unit tests", status: "pending", priority: "medium"},
    {id: "6", content: "Integration tests", status: "pending", priority: "medium"},
    {id: "7", content: "API documentation", status: "pending", priority: "low"},
    {id: "8", content: "Performance optimization", status: "pending", priority: "low"}
  ]}
  
  // Parallel file operations
  Bash "mkdir -p app/{src,tests,docs,config}"
  Write "app/package.json"
  Write "app/src/server.js"
  Write "app/tests/server.test.js"
  Write "app/docs/API.md"
```

### ❌ WRONG (Multiple Messages):
```javascript
Message 1: mcp__claude-flow__swarm_init
Message 2: Task("agent 1")
Message 3: TodoWrite { todos: [single todo] }
Message 4: Write "file.js"
// This breaks parallel coordination!
```

## Performance Benefits

- **84.8% SWE-Bench solve rate**
- **32.3% token reduction**
- **2.8-4.4x speed improvement**
- **27+ neural models**

## Hooks Integration

### Pre-Operation
- Auto-assign agents by file type
- Validate commands for safety
- Prepare resources automatically
- Optimize topology by complexity
- Cache searches

### Post-Operation
- Auto-format code
- Train neural patterns
- Update memory
- Analyze performance
- Track token usage

### Session Management
- Generate summaries
- Persist state
- Track metrics
- Restore context
- Export workflows

## Advanced Features (v2.0.0)

- 🚀 Automatic Topology Selection
- ⚡ Parallel Execution (2.8-4.4x speed)
- 🧠 Neural Training
- 📊 Bottleneck Analysis
- 🤖 Smart Auto-Spawning
- 🛡️ Self-Healing Workflows
- 💾 Cross-Session Memory
- 🔗 GitHub Integration

## Integration Tips

1. Start with basic swarm init
2. Scale agents gradually
3. Use memory for context
4. Monitor progress regularly
5. Train patterns from success
6. Enable hooks automation
7. Use GitHub tools first

## Support

- Documentation: https://github.com/ruvnet/claude-flow
- Issues: https://github.com/ruvnet/claude-flow/issues
- Flow-Nexus Platform: https://flow-nexus.ruv.io (registration required for cloud features)

---

Remember: **Claude Flow coordinates, Claude Code creates!**

---

## 🎯 SKILL AUTO-TRIGGER REFERENCE

**Purpose**: Auto-invoke skills based on user intent keywords without explicit requests.
**Location**: All skills in `.claude/skills/` | Full docs via `Skill` tool
**Memory**: Store trigger patterns in `claude-flow memory` for persistence

### 📋 Trigger Pattern Legend

**Format**: `Skill Name` - Trigger conditions → Action
**Usage**: When you detect trigger keywords, AUTO-INVOKE the skill via Skill tool

---

### 🏗️ Development Lifecycle Skills (Auto-trigger on project init/development)

**Planning & Architecture**
- `research-driven-planning` - "new feature", "plan", "architecture needed" → Research + 5x pre-mortem + risk mitigation
- `sparc-methodology` - "implement", "build feature", "create" → 5-phase SPARC (Spec→Code)
- `interactive-planner` - "not sure", "options?", "what approach" → Multi-select questions for requirements
- `intent-analyzer` - vague/ambiguous request → Analyze intent + clarify with Socratic questions

**Development**
- `parallel-swarm-implementation` - "build this", "implement feature" → Multi-agent parallel development
- `feature-dev-complete` - "complete feature", "end-to-end" → 12-stage workflow (research→deployment)
- `pair-programming` - "let's code together", "pair with me" → Driver/Navigator/Switch modes with real-time verification
- `cascade-orchestrator` - "complex workflow", "pipeline" → Sequential/parallel/conditional micro-skill chains

**Code Quality**
- `functionality-audit` - "does it work?", "validate", AFTER code generation → Sandbox testing + systematic debugging
- `theater-detection-audit` - "real implementation?", "not theater" → 6-agent Byzantine consensus verification
- `production-readiness` - "deploy", "production ready" → Complete audit pipeline + deployment checklist
- `quick-quality-check` - "quick check", "fast validation" → Parallel lint/security/tests (instant feedback)
- `code-review-assistant` - "review this code", "PR review" → Multi-agent swarm review (security/performance/style)

**Testing & Debugging**
- `testing-quality` - "write tests", "test coverage" → TDD framework + quality validation
- `smart-bug-fix` - "bug", "error", "failing" → Intelligent debugging + automated fixes
- `reverse-engineer-debug` - "understand code", "reverse engineer" → Code comprehension + debugging
- `ml-training-debugger` - "training failed", "model not converging" → ML-specific debugging

**Self-Improvement & Dogfooding** 🆕
- `sop-dogfooding-quality-detection` - "analyze code quality", "detect violations", "connascence check" → Phase 1: Run Connascence analysis, store in Memory-MCP with WHO/WHEN/PROJECT/WHY (30-60s)
- `sop-dogfooding-pattern-retrieval` - "find similar fixes", "pattern search", "past solutions" → Phase 2: Vector search Memory-MCP for patterns, rank & optionally apply (10-30s)
- `sop-dogfooding-continuous-improvement` - "run improvement cycle", "dogfood", "automated fixes" → Phase 3: Full cycle orchestration with sandbox testing & metrics (60-120s)

**Trigger Patterns:**
```javascript
// Quality detection → Auto-spawn code-analyzer + reviewer
"Check code quality for memory-mcp" → sop-dogfooding-quality-detection
"Run connascence analysis" → sop-dogfooding-quality-detection

// Pattern retrieval → Auto-spawn code-analyzer + coder
"Find fixes for God Object" → sop-dogfooding-pattern-retrieval
"How to fix Parameter Bomb?" → sop-dogfooding-pattern-retrieval

// Full cycle → Auto-spawn hierarchical-coordinator
"Run dogfooding cycle" → sop-dogfooding-continuous-improvement
"Improve the MCP servers" → sop-dogfooding-continuous-improvement
```

---

### 🛠️ Specialized Development (Auto-trigger by tech stack)

**Language Specialists** 🆕
- `python-specialist` - "Python", "FastAPI", "Django", "Flask", "async Python" → Python development, type hints, pytest, performance profiling
- `typescript-specialist` - "TypeScript", "Nest.js", "Node.js", "npm", "monorepo" → Advanced types, decorators, tooling

**Frontend Specialists** 🆕
- `react-specialist` - "React 18", "Next.js", "hooks", "Zustand", "React performance" → Modern React, App Router, state management, optimization

**Backend/API**
- `when-building-backend-api-orchestrate-api-development` - "API", "REST", "GraphQL", "backend" → Full API development workflow
- `sop-api-development` - "API SOP", "systematic API" → Standardized API development process

**Database Specialists** 🆕
- `sql-database-specialist` - "PostgreSQL", "MySQL", "SQL optimization", "EXPLAIN", "indexes" → Query tuning, schema design, partitioning

**Mobile**
- `mobile-dev` - "React Native", "iOS", "Android", "mobile app" → Mobile app development specialist

**Machine Learning**
- `ml-expert` - "machine learning", "neural network", "train model" → ML development + training
- `ml-developer` - "ML pipeline", "data science" → ML development specialist

**Documentation**
- `pptx-generation` - "PowerPoint", "presentation", "slides" → Enterprise PPT generation (html2pptx)
- `documentation` - "docs", "API docs", "README" → Documentation generation

---

### 🔒 Security & Compliance (Auto-trigger for security keywords)

**Security**
- `network-security-setup` - "network security", "sandbox isolation" → Configure network boundaries
- `security` - "security audit", "vulnerabilities" → Security scanning + fixes
- `sop-code-review` - "security review", "secure code" → Systematic security-focused review

**Compliance & Accessibility** 🆕
- `wcag-accessibility` - "WCAG", "accessibility", "a11y", "screen reader", "ARIA" → WCAG 2.1 AA/AAA compliance, keyboard navigation, axe-core

---

### 🎨 Code Creation & Architecture (Auto-trigger for creation tasks)

**Agents & Skills**
- `agent-creator` - "create agent", "new agent" → 4-phase SOP agent creation with evidence-based prompting
- `skill-builder` - "create skill", "new skill" → YAML frontmatter + progressive disclosure structure
- `skill-creator-agent` - "skill with agent" → Skill tied to specialist agent
- `micro-skill-creator` - "micro-skill", "atomic skill" → Focused single-purpose skills
- `slash-command-encoder` - "slash command", "/command" → Create .claude/commands/*.md

**Templates & Patterns**
- `base-template-generator` - "boilerplate", "starter template" → Clean foundational templates
- `prompt-architect` - "improve prompt", "prompt engineering" → Analyze + optimize prompts
- `skill-forge` - "complex skill", "advanced skill" → Advanced skill crafting

---

### 🔍 Analysis & Optimization (Auto-trigger for analysis requests)

**Performance**
- `performance-analysis` - "slow", "performance", "bottleneck" → Comprehensive performance analysis
- `perf-analyzer` - "workflow slow", "optimize" → Bottleneck detection + optimization

**Code Analysis**
- `style-audit` - "code style", "consistency" → Code style analysis + fixes
- `verification-quality` - "verify", "validate quality" → Quality verification

**Dependencies**
- `dependencies` - "dependencies", "imports", "requires" → Dependency analysis + mapping

---

### 🐙 GitHub Integration (Auto-trigger on GitHub keywords)

**Core GitHub**
- `github-code-review` - "review PR", "pull request review" → AI swarm PR review
- `github-project-management` - "GitHub project", "issue tracking" → Issue + project board automation
- `github-workflow-automation` - "GitHub Actions", "CI/CD", "workflow" → Intelligent CI/CD pipelines
- `github-release-management` - "release", "version", "deploy" → Automated versioning + deployment
- `github-multi-repo` - "multi-repo", "monorepo", "sync repos" → Cross-repo coordination

**Repository**
- `github-integration` - General GitHub operations → Repository management
- `sop-product-launch` - "launch product", "release product" → Complete launch workflow

---

### 🌐 Multi-Model & External Tools (Auto-trigger by tool mention)

**Gemini**
- `gemini-search` - "search web", "google", "research online" → Gemini grounded search
- `gemini-megacontext` - "huge context", "2M tokens", "large doc" → Mega context processing
- `gemini-media` - "image", "video", "audio" → Multimodal analysis
- `gemini-extensions` - "Google Workspace", "Google Maps" → Gemini extensions

**Codex**
- `codex-auto` - "Codex", "auto-execute" → Autonomous coding in sandboxes
- `codex-reasoning` - "reasoning", "chain of thought" → Advanced reasoning patterns

**Multi-Model**
- `multi-model` - "compare models", "best model" → Multi-model routing + selection

---

### 🧠 Intelligence & Learning (Auto-trigger for learning/memory)

**Memory Systems**
- `agentdb` - "vector search", "semantic search" → 150x faster AgentDB vector search
- `agentdb-memory-patterns` - "persistent memory", "session memory" → Memory patterns for stateful agents
- `agentdb-learning` - "reinforcement learning", "Q-learning", "RL" → 9 RL algorithms for agent learning
- `agentdb-optimization` - "quantization", "HNSW", "optimize vectors" → 4-32x memory reduction
- `agentdb-vector-search` - "RAG", "document retrieval" → Semantic search for knowledge bases
- `agentdb-advanced` - "QUIC sync", "multi-database", "distributed" → Advanced distributed features

**Reasoning**
- `reasoningbank-intelligence` - "learn from mistakes", "adaptive learning" → Pattern recognition + optimization
- `reasoningbank-agentdb` - "ReasoningBank", "trajectory tracking" → 46% faster learning + 88% success

---

### 🐝 Swarm & Coordination (Auto-trigger for multi-agent tasks)

**Swarm Orchestration**
- `swarm-orchestration` - "swarm", "multi-agent", "coordinate agents" → Swarm topology + orchestration
- `swarm-advanced` - "advanced swarm", "complex coordination" → Advanced swarm features
- `hive-mind-advanced` - "hive mind", "collective intelligence" → Queen-led hierarchical coordination
- `coordination` - "coordinate", "sync agents" → Agent coordination patterns

**Flow Nexus Cloud**
- `flow-nexus-platform` - "Flow Nexus", "cloud swarm" → Cloud-based orchestration
- `flow-nexus-swarm` - "cloud swarm deployment" → Swarm deployment in cloud
- `flow-nexus-neural` - "distributed neural", "cloud training" → Neural network training in E2B sandboxes

---

### 🔧 Utilities & Tools (Auto-trigger by utility need)

**Automation**
- `hooks-automation` - "automate hooks", "lifecycle events" → Hook integration + automation
- `workflow` - "workflow", "automation" → Workflow creation + execution
- `i18n-automation` - "internationalization", "i18n", "translate" → i18n workflow automation
- `stream-chain` - "streaming", "chain" → Streaming workflows

**Debugging**
- `debugging` - "debug", "troubleshoot" → Systematic debugging
- `verification-quality` - "verify", "check quality" → Quality verification

**Meta Tools**
- `meta-tools` - "meta", "tool creation" → Tool creation tools
- `specialized-tools` - "specialized tool" → Domain-specific tools
- `web-cli-teleport` - "CLI tool", "web interface" → CLI↔Web teleportation

**Platform**
- `platform` - "platform", "infrastructure" → Platform-level operations
- `sandbox-configurator` - "sandbox config", "E2B setup" → Sandbox configuration

---

### 🔬 Deep Research SOP (Auto-trigger for research workflows)

**Quality Gate System - Research Lifecycle Management (4 agents)**
- `data-steward` - "dataset", "datasheet", "bias audit", "data quality", "DVC", "Quality Gate 1" → Dataset documentation, bias auditing, data versioning, datasheet completion (Form F-C1)
- `ethics-agent` - "ethics review", "risk assessment", "safety evaluation", "fairness metrics", "privacy audit", "compliance" → Ethics & safety review across all Quality Gates, 6-domain risk assessment (ethical, safety, privacy, dual-use, reproducibility, environmental)
- `archivist` - "reproducibility", "DOI", "archive artifacts", "model card", "Quality Gate 3" → Artifact archival, version control, reproducibility packaging, DOI assignment, model card creation (Form F-G2)
- `evaluator` - "Quality Gate", "GO/NO-GO", "gate approval", "gate review" → Final authority for all Quality Gate approvals (Gates 1, 2, 3), multi-agent coordination, requirements validation

**Comprehensive Research Pipeline (9 skills)** 🆕
- `baseline-replication` - "replicate baseline", "reproduce results", "±1% tolerance", "statistical validation" → Baseline replication with ACM compliance, paired t-tests, effect size calculation
- `literature-synthesis` - "systematic review", "PRISMA 2020", "gap analysis", "research positioning" → Multi-database literature search, citation management, synthesis
- `method-development` - "novel algorithm", "ablation studies", "Bonferroni correction", "statistical rigor" → Algorithm design with statistical power analysis, hypothesis testing
- `holistic-evaluation` - "multi-metric evaluation", "performance + efficiency + robustness", "interpretability" → Comprehensive model evaluation beyond accuracy
- `deployment-readiness` - "production deployment", "A/B testing", "monitoring", "rollback" → Production ML with canary deployments, observability, incident response
- `deep-research-orchestrator` - "complete research workflow", "Pipeline F", "multi-agent coordination" → Full research pipeline orchestration from literature review to publication
- `reproducibility-audit` - "ACM Artifact Evaluation", "Docker validation", "Zenodo archival", "DOI" → Reproducibility verification, artifact badges (Available, Functional, Reproduced, Reusable)
- `research-publication` - "paper writing", "conference submission", "peer review response", "LaTeX" → Academic paper creation, citation management, submission workflows
- `gate-validation` - "Quality Gate validation", "GO/NO-GO framework", "requirement checklists" → Phase transition validation for Gates 1-3

**Trigger Patterns:**
```javascript
// Dataset onboarding/quality → Auto-spawn data-steward
"Need to document dataset" → data-steward + /init-datasheet
"Bias in training data" → data-steward + /bias-audit + ethics-agent
"Version control for data" → data-steward + /dvc-init

// Ethics/safety review → Auto-spawn ethics-agent
"Assess risks for model" → ethics-agent + /assess-risks
"Safety evaluation needed" → ethics-agent + /safety-eval
"Privacy concerns" → ethics-agent + /privacy-audit
"Compliance check" → ethics-agent + /compliance-check

// Reproducibility/archival → Auto-spawn archivist
"Package for reproducibility" → archivist + /create-reproducibility-package
"Assign DOI to dataset" → archivist + /assign-doi
"Create model card" → archivist + /init-model-card
"Test reproducibility" → archivist + /test-reproducibility

// Gate approval/validation → Auto-spawn evaluator
"Quality Gate 1 review" → evaluator + data-steward + ethics-agent
"Quality Gate 2 review" → evaluator + ethics-agent + archivist
"Quality Gate 3 review" → evaluator + archivist + ethics-agent
"GO/NO-GO decision" → evaluator + /validate-gate-{N}
```

**SOP Commands (Available to agents):**
- `/init-datasheet` - Initialize Datasheet for Datasets (Form F-C1, 7 sections, 47 questions, 80%+ completion)
- `/prisma-init` - Initialize PRISMA 2020 systematic literature review (multi-database search, 3-stage screening)
- `/assess-risks` - Comprehensive risk assessment across 6 domains (ethical, safety, privacy, dual-use, reproducibility, environmental)
- `/init-model-card` - Initialize Model Card for Model Reporting (Form F-G2, 9 sections, 90%+ completion)

**Quality Gates:**
- **Gate 1** (Data & Methods): Literature review complete, Datasheet ≥80%, Bias audit acceptable, Ethics review initiated
- **Gate 2** (Model & Evaluation): Baseline replicated, Ablations complete, HELM + CheckList passed, Fairness metrics acceptable
- **Gate 3** (Production & Artifacts): Model Card ≥90%, DOIs assigned, Reproducibility tested, ML Test Score ≥8

---

### 🛡️ Reverse Engineering & Binary Analysis (Auto-trigger for malware/security analysis) 🆕

**3 Specialized Levels for Comprehensive Binary Analysis**
- `reverse-engineering-quick` - "malware triage", "IOC extraction", "strings analysis", "static analysis" → RE Levels 1-2 (≤2 hours): String reconnaissance + disassembly with Ghidra/radare2
- `reverse-engineering-deep` - "advanced malware", "vulnerability research", "CTF", "symbolic execution" → RE Levels 3-4 (4-8 hours): GDB debugging + Angr symbolic execution for path exploration
- `reverse-engineering-firmware` - "IoT security", "firmware extraction", "router vulnerabilities", "embedded systems" → RE Level 5 (2-8 hours): binwalk + QEMU + firmadyne for firmware analysis and emulation

**Security Features**: ⚠️ VM/Docker/E2B sandboxing required for all binary execution, comprehensive malware analysis best practices

---

### ☁️ Infrastructure & Cloud (Auto-trigger for infrastructure/cloud keywords) 🆕

**Cloud Platforms**
- `aws-specialist` - "AWS", "Lambda", "ECS", "Fargate", "AWS CDK", "CloudFormation" → AWS deployment, serverless, containers, infrastructure
- `kubernetes-specialist` - "Kubernetes", "K8s", "Helm", "operators", "Istio", "service mesh" → Container orchestration, production K8s, autoscaling

**Infrastructure as Code**
- `docker-containerization` - "Docker", "container", "multi-stage build", "BuildKit", "Trivy" → Docker optimization, security scanning, Compose
- `terraform-iac` - "Terraform", "IaC", "infrastructure as code", "HCL", "state management" → Multi-cloud provisioning, modules, GitOps

**Observability**
- `opentelemetry-observability` - "OpenTelemetry", "distributed tracing", "observability", "Jaeger", "Zipkin" → Traces, metrics, W3C Trace Context, APM

---

### 📊 CI/CD & Recovery (Auto-trigger for deployment/testing)

- `cicd-intelligent-recovery` - "CI/CD", "test failed", "build failed" → Automated failure recovery + root cause analysis

---

### 🎯 USAGE PATTERN

```javascript
// When user says: "I need to build a new API with auth"
// AUTO-TRIGGER:
Skill("when-building-backend-api-orchestrate-api-development")
  → Research best practices (Gemini search)
  → Architecture design
  → Parallel swarm implementation
  → Security audit
  → Testing
  → Documentation

// When user says: "This code looks fake, validate it"
// AUTO-TRIGGER:
Skill("theater-detection-audit")
  → 6-agent Byzantine consensus
  → Sandbox execution testing
  → Real implementation verification
```

---

### 💾 MEMORY PERSISTENCE

**Store Skill Metadata in Claude-Flow Memory**:
```bash
npx claude-flow@alpha memory store \
  --key "skills/auto-triggers" \
  --value "$(cat CLAUDE.md | grep -A 200 'SKILL AUTO-TRIGGER')"

npx claude-flow@alpha memory retrieve \
  --key "skills/auto-triggers"
```

**Benefits**:
- ✅ Persistent across sessions
- ✅ Auto-load on startup
- ✅ Share across terminals
- ✅ No context window bloat (retrieve only when needed)

---

**REMEMBER**: Skills contain full details. This reference just triggers them at the right time!

---

# important-instruction-reminders
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.
Never save working files, text/mds and tests to the root folder.
