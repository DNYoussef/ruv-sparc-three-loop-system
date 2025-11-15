# Claude Code Configuration v2.2 - Universal 5-Phase Workflow System

**Version**: 2.2.0
**Last Updated**: 2025-11-15
**Previous Version**: v2.1.0 (2025-11-15)

---

## 1. UNIVERSAL MESSAGE PROCESSING WORKFLOW (AUTO-EXECUTE ON EVERY REQUEST)

**On EVERY user message, execute this 5-phase workflow SEQUENTIALLY:**

**intent → prompt → plan → route → execute**

---

### Phase 1: Intent Analysis (ALWAYS FIRST)
**Skill**: `Skill("intent-analyzer")`

**What It Does**:
- Extract underlying goals using first principles decomposition
- Identify constraints (explicit + implicit)
- Determine if intent is clear & actionable
- Apply probabilistic intent mapping (>80% confidence = proceed)
- Clarify ambiguous requests with Socratic questions if needed (<80% confidence)

**Output**:
```json
{
  "understood_intent": "What the user actually wants to accomplish",
  "explicit_constraints": ["stated requirements"],
  "implicit_constraints": ["inferred requirements"],
  "confidence": 0.85,
  "ambiguities": ["areas needing clarification"]
}
```

---

### Phase 2: Prompt Optimization (ALWAYS SECOND)
**Skill**: `Skill("prompt-architect")`

**What It Does**:
- Take analyzed intent from Phase 1
- Apply evidence-based prompting techniques
- Restructure request for clarity and completeness
- Add missing context from intent analysis
- Generate optimized prompt: "If this was the real intent, what should they have asked?"

**Input**: Analyzed intent from Phase 1
**Output**:
```json
{
  "optimized_request": "Restructured request with clarity and context",
  "added_context": ["missing information now included"],
  "prompting_patterns": ["self-consistency", "plan-and-solve"],
  "success_criteria": "Clear definition of done"
}
```

---

### Phase 3: Strategic Planning (ALWAYS THIRD)
**Skill**: `Skill("research-driven-planning")` OR `Skill("planner")` (based on complexity)

**What It Does**:
- Take optimized request from Phase 2
- Break down into actionable tasks
- **Identify dependencies** (what MUST be sequential)
- **Identify parallelizable tasks** (what CAN run concurrently)
- Select appropriate playbooks and skills for each task
- Determine execution order and parallelization strategy
- Create comprehensive execution plan

**Input**: Optimized request from Phase 2
**Output**:
```json
{
  "plan": {
    "sequential_phases": [
      {
        "phase": 1,
        "name": "Foundation Setup",
        "tasks": [
          {
            "task": "Research best practices",
            "playbook": "research-quick-investigation",
            "skills": ["gemini-search", "researcher"],
            "agents": ["researcher"],
            "prerequisites": [],
            "can_parallelize": false
          }
        ]
      },
      {
        "phase": 2,
        "name": "Parallel Implementation",
        "tasks": [
          {
            "task": "Build backend API",
            "playbook": "backend-api-development",
            "skills": ["backend-dev"],
            "agents": ["backend-dev"],
            "prerequisites": ["phase 1 complete"],
            "can_parallelize": true,
            "parallel_group": "implementation"
          },
          {
            "task": "Build frontend UI",
            "playbook": "frontend-development",
            "skills": ["react-specialist"],
            "agents": ["coder"],
            "prerequisites": ["phase 1 complete"],
            "can_parallelize": true,
            "parallel_group": "implementation"
          },
          {
            "task": "Setup database schema",
            "playbook": "database-design",
            "skills": ["sql-database-specialist"],
            "agents": ["code-analyzer"],
            "prerequisites": ["phase 1 complete"],
            "can_parallelize": true,
            "parallel_group": "implementation"
          }
        ]
      },
      {
        "phase": 3,
        "name": "Integration & Validation",
        "tasks": [
          {
            "task": "Integration testing",
            "playbook": "testing-quality",
            "skills": ["tester"],
            "agents": ["tester"],
            "prerequisites": ["phase 2 all tasks complete"],
            "can_parallelize": false
          }
        ]
      }
    ]
  },
  "execution_strategy": {
    "total_phases": 3,
    "sequential_phases": [1, 3],
    "parallel_phases": [2],
    "estimated_time": "4-8 hours",
    "mcp_requirements": ["flow-nexus", "memory-mcp"]
  },
  "dependencies": {
    "phase_1_blocks": ["phase_2", "phase_3"],
    "phase_2_blocks": ["phase_3"]
  }
}
```

**Key Outputs**:
- **Sequential Tasks**: Must be done in order (Phase 1 → Phase 2 → Phase 3)
- **Parallel Tasks**: Can run concurrently within a phase (backend + frontend + database)
- **Playbooks/Skills**: Specific tools needed for each task
- **Prerequisites**: What must complete before each task starts
- **MCP Requirements**: Which MCPs to activate for the workflow

---

### Phase 4: Playbook/Skill Routing (ALWAYS FOURTH)
**Action**: Route each task to optimal playbook/skill

**What It Does**:
- Take execution plan from Phase 3
- For each task in the plan, select the best playbook or skill
- Match task requirements to playbook categories (see Section 3)
- Consider:
  - Task complexity (simple feature vs complex multi-loop)
  - Domain (frontend, backend, ML, security, etc.)
  - Time constraints (quick vs comprehensive)
  - MCP availability (which MCPs are active)
- Output routing decisions for Phase 5

**Input**: Execution plan from Phase 3
**Output**:
```json
{
  "routing_decisions": [
    {
      "phase": 1,
      "task": "Research Express.js auth best practices",
      "selected_playbook": "research-quick-investigation",
      "primary_skill": "gemini-search",
      "fallback_skill": "researcher",
      "rationale": "Quick research task, Gemini search optimal for best practices"
    },
    {
      "phase": 2,
      "parallel_group": "implementation",
      "tasks": [
        {
          "task": "Build backend API",
          "selected_playbook": "backend-api-development",
          "primary_skill": "backend-dev",
          "agents": ["backend-dev"],
          "rationale": "Backend API development specialist playbook"
        },
        {
          "task": "Database schema design",
          "selected_playbook": "database-design",
          "primary_skill": "sql-database-specialist",
          "agents": ["code-analyzer"],
          "rationale": "Database specialist with schema design expertise"
        },
        {
          "task": "Auth middleware",
          "selected_playbook": "simple-feature-implementation",
          "primary_skill": "sparc-methodology",
          "agents": ["coder"],
          "rationale": "Single feature with TDD workflow"
        }
      ]
    }
  ]
}
```

**Routing Criteria**:

| Task Type | Route To | When |
|-----------|----------|------|
| Simple feature | simple-feature-implementation | <4 hours, single component |
| Complex feature | three-loop-system (FLAGSHIP) | >4 hours, multi-component, high risk |
| Quick research | research-quick-investigation | <2 hours, specific question |
| Deep research | deep-research-sop (FLAGSHIP) | Multi-month, academic rigor |
| Code quality | comprehensive-review | Audit, security, clarity |
| Bug fix | smart-bug-fix | Production issue, debugging needed |
| ML pipeline | ml-pipeline-development | Neural training, experiments |
| API development | backend-api-development | REST/GraphQL endpoints |
| Frontend | frontend-development | React/Vue/UI components |
| Full-stack | feature-dev-complete | End-to-end 12-stage workflow |

**Key Insight**: Routing happens AFTER planning, so we know exact requirements before selecting playbooks.

---

### Phase 5: Execution (ALWAYS FIFTH)
**Action**: Execute using routed playbooks/skills from Phase 4

**Sequential Execution** (when prerequisites exist):
```javascript
// Phase 1 MUST complete before Phase 2
Skill("gemini-search")
// Wait for completion, then...

// Phase 2 - Spawn in PARALLEL (single message, Golden Rule)
[Single Message]:
  Task("Backend Developer", "Build REST API...", "backend-dev")
  Task("Frontend Developer", "Build React UI...", "coder")
  Task("Database Architect", "Design schema...", "code-analyzer")
  TodoWrite({ todos: [8-10 todos for all parallel work] })
// Wait for all Phase 2 tasks, then...

// Phase 3 - Sequential again
Task("Integration Tester", "Test all components...", "tester")
```

**Parallel Execution** (when no prerequisites):
```javascript
// All tasks can run concurrently
[Single Message]:
  Task("Agent 1", "Independent task 1...", "researcher")
  Task("Agent 2", "Independent task 2...", "coder")
  Task("Agent 3", "Independent task 3...", "reviewer")
  TodoWrite({ todos: [all tasks listed] })
```

---

### Workflow Summary

```mermaid
User Message
    ↓
Phase 1: intent-analyzer
    ├─ Analyze intent
    ├─ Identify constraints
    └─ Output: Understood intent (confidence score)
    ↓
Phase 2: prompt-architect
    ├─ Optimize request
    ├─ Add missing context
    └─ Output: "What they should have asked"
    ↓
Phase 3: planner (research-driven-planning)
    ├─ Break down into tasks
    ├─ Identify dependencies (sequential)
    ├─ Identify parallelizable tasks
    └─ Output: Execution plan with dependencies
    ↓
Phase 4: router (playbook/skill routing)
    ├─ Match tasks to playbooks (Section 3)
    ├─ Select optimal skills per task
    ├─ Choose appropriate agents
    └─ Output: Routing decisions for each task
    ↓
Phase 5: Execute
    ├─ Use routed playbooks/skills
    ├─ Sequential phases (prerequisites)
    ├─ Parallel phases (concurrent agents)
    └─ Follow Golden Rule (1 MESSAGE = ALL RELATED OPERATIONS)
```

---

### Execution Rules

**CRITICAL RULES**:
1. **ALWAYS run all 5 phases** for EVERY user message (no exceptions)
2. **Phases 1-4 are ALWAYS SEQUENTIAL** (each depends on previous output)
3. **Phase 5 execution** follows the plan + routing:
   - Sequential tasks: One message per phase, wait for completion
   - Parallel tasks: ALL agents in ONE message (Golden Rule)
   - Use playbooks/skills selected in Phase 4 routing
4. **Output transparency**: Show plan (Phase 3) + routing (Phase 4) to user before Phase 5
5. **User approval**: For complex plans (>3 phases or >5 tasks), ask user to confirm before Phase 5

---

### Escape Hatches

**Skip Phases 1-4 only if**:
- Explicit skill invocation: `Skill("micro-skill-creator")` → Direct to Phase 5
- Explicit command: `/research:literature-review` → Direct to Phase 5
- Explicit agent reference: `@agent-creator` → Direct to Phase 5
- User says "skip planning" or "just do it" → Skip to Phase 5

**Otherwise**: ALWAYS execute all 5 phases sequentially.

---

### Example Walkthrough

**User Message**: "Build a REST API for user management with authentication"

**Phase 1 Output** (intent-analyzer):
```
Understood Intent: Build production-grade REST API with:
- User CRUD operations
- JWT-based authentication
- Password hashing
- Input validation
Confidence: 92%
```

**Phase 2 Output** (prompt-architect):
```
Optimized Request: "Build a production-ready REST API with Express.js including:
- User registration, login, CRUD endpoints
- JWT authentication middleware
- Bcrypt password hashing
- Input validation with Joi
- PostgreSQL database integration
- Comprehensive test suite
- API documentation (OpenAPI/Swagger)
Success Criteria: All endpoints working, 90% test coverage, documented API"
```

**Phase 3 Output** (planner):
```
Plan:
PHASE 1 (Sequential - Research):
  - Task: Research Express.js auth best practices
  - Prerequisites: None

PHASE 2 (Parallel - Implementation):
  - Task 1: Backend API structure
  - Task 2: Database schema design
  - Task 3: Auth middleware
  - Prerequisites: Phase 1 complete
  - Can parallelize: YES (all 3 tasks concurrent)

PHASE 3 (Sequential - Testing):
  - Task: Integration tests
  - Prerequisites: Phase 2 all tasks complete

PHASE 4 (Sequential - Documentation):
  - Task: API docs generation
  - Prerequisites: Phase 3 complete

Execution Strategy:
- Sequential phases: 1 → 2 → 3 → 4
- Parallel: Phase 2 only (3 agents concurrently)
- Total time: 4-6 hours
- MCPs needed: flow-nexus (research), memory-mcp (state)
```

**Phase 4 Output** (router):
```
Routing Decisions:
PHASE 1:
  - Task: Research best practices
  - Route: Skill("gemini-search") via research-quick-investigation playbook
  - Agent: researcher
  - Rationale: Quick research (<2 hours), Gemini optimal for best practices

PHASE 2 (Parallel):
  - Task 1: Backend API
    Route: backend-api-development playbook
    Skills: backend-dev
    Agent: backend-dev
    Rationale: Specialist backend development playbook

  - Task 2: Database schema
    Route: database-design playbook
    Skills: sql-database-specialist
    Agent: code-analyzer
    Rationale: Database specialist with schema expertise

  - Task 3: Auth middleware
    Route: simple-feature-implementation playbook
    Skills: sparc-methodology
    Agent: coder
    Rationale: Single feature with TDD (SPARC)

PHASE 3:
  - Task: Integration tests
  - Route: testing-quality playbook
  - Skills: tester
  - Agent: tester
  - Rationale: Comprehensive testing playbook

PHASE 4:
  - Task: API documentation
  - Route: api-documentation-specialist playbook
  - Skills: api-docs
  - Agent: api-docs
  - Rationale: OpenAPI/Swagger specialist
```

**Phase 5 Execution**:
```javascript
// Show plan to user, get approval
// Then execute:

// Phase 1 (sequential)
Skill("gemini-search")
// Wait for completion

// Phase 2 (parallel - ONE message)
[Single Message]:
  Task("Backend Developer", "Build Express API structure with routes, controllers, models...", "backend-dev")
  Task("Database Architect", "Design PostgreSQL schema with users table, indexes, constraints...", "code-analyzer")
  Task("Auth Developer", "Implement JWT middleware, bcrypt hashing, session management...", "coder")
  TodoWrite({ todos: [10 todos covering all Phase 2-4 work] })
// Wait for all Phase 2 tasks

// Phase 3 (sequential)
Task("Integration Tester", "Write comprehensive tests covering all endpoints, auth flows...", "tester")
// Wait for completion

// Phase 4 (sequential)
Task("API Documentor", "Generate OpenAPI/Swagger docs for all endpoints...", "api-docs")
```

---

**Remember**: This 5-phase workflow is **MANDATORY for EVERY user message** unless explicitly bypassed with skill/command invocation.

**Why 5 Phases vs 4?**
- **Planning (Phase 3)**: Creates the strategy (WHAT to do, WHEN, dependencies)
- **Routing (Phase 4)**: Selects the tools (HOW to do it, which playbooks/skills)
- **Separation of concerns**: Strategy vs tactics, planning vs execution tools

**Best of Both Worlds**:
- ✅ v2.0 routing intelligence (playbook selection from Section 3)
- ✅ v2.1 planning intelligence (dependency detection, parallelization)
- ✅ v2.2 combined power (plan THEN route THEN execute)

---

## 2. EXECUTION RULES (ALWAYS FOLLOW)

### 2.1 Golden Rule: "1 MESSAGE = ALL RELATED OPERATIONS"

**MANDATORY PATTERNS**:

**TodoWrite**: Batch ALL todos (5-10+ minimum)
- ✅ CORRECT: `TodoWrite({ todos: [8 todos] })`
- ❌ WRONG: Multiple TodoWrite calls across messages

**Task Tool**: Spawn ALL agents concurrently
- ✅ CORRECT: `[Task(agent1), Task(agent2), Task(agent3)]` in ONE message
- ❌ WRONG: Sequential Task calls across messages

**File Operations**: Batch ALL reads/writes/edits
- ✅ CORRECT: `[Read file1, Read file2, Write file3, Edit file4]` in ONE message
- ❌ WRONG: Read file, wait, then Write file

**Memory Operations**: Batch ALL store/retrieve
- ✅ CORRECT: `[memory_store(key1), memory_store(key2), memory_retrieve(key3)]`
- ❌ WRONG: Sequential memory operations

### 2.2 File Organization

**NEVER save to root folder**. Use proper directories:

| File Type | Directory | Examples |
|-----------|-----------|----------|
| Source code | `/src` | `src/app.js`, `src/api/` |
| Tests | `/tests` | `tests/unit/`, `tests/integration/` |
| Documentation | `/docs` | `docs/API.md`, `docs/architecture/` |
| Scripts | `/scripts` | `scripts/deploy.sh`, `scripts/setup/` |
| Configuration | `/config` | `config/database.yml` |

### 2.3 Agent Usage (131 Total Agents)

**CRITICAL**: ONLY use predefined agents from registry.

**Agent Categories** (counts in parentheses):
- Core Development (8)
- Testing & Validation (9)
- Frontend Development (6)
- Database & Data (7)
- Documentation & Knowledge (6)
- Swarm Coordination (15)
- Performance & Optimization (5)
- GitHub & Repository (9)
- SPARC Methodology (6)
- Specialized Development (14)
- Deep Research SOP (4)
- Infrastructure & Cloud (12)
- Security & Compliance (8)

**How to Find Agents**:
```bash
# List all agents by category
Read(".claude/agents/README.md") | grep "^###"

# Search by capability
npx claude-flow agents search "database"

# Get agent details
npx claude-flow agents info "backend-dev"
```

**DO NOT create new agent types**. Match tasks to existing agents.

---

## 3. PLAYBOOK ROUTER (SELECT BASED ON INTENT)

Match user request keywords to playbook:

### Research & Analysis
**Triggers**: "analyze", "research", "investigate", "systematic review", "literature", "PRISMA"
**Skills**: `deep-research-orchestrator`, `literature-synthesis`, `baseline-replication`
**Agents**: researcher, data-steward, ethics-agent, archivist, evaluator

### Development
**Triggers**: "build", "implement", "create feature", "develop", "SPARC", "TDD"
**Skills**: `ai-dev-orchestration`, `sparc-methodology`, `feature-dev-complete`
**Agents**: planner, system-architect, coder, tester, reviewer

### Code Quality
**Triggers**: "audit", "review", "validate", "check quality", "detect violations", "clarity"
**Skills**: `clarity-linter`, `functionality-audit`, `theater-detection-audit`, `code-review-assistant`
**Agents**: code-analyzer, reviewer, functionality-audit, production-validator

### Infrastructure & Deployment
**Triggers**: "deploy", "CI/CD", "production", "monitoring", "Kubernetes", "Docker", "cloud"
**Skills**: `cicd-intelligent-recovery`, `deployment-readiness`, `production-readiness`
**Agents**: cicd-engineer, kubernetes-specialist, terraform-iac, docker-containerization

### Specialized Domains
- **ML/AI**: "train model", "neural network", "dataset" → `deep-research-orchestrator`, `machine-learning`
- **Security**: "pentest", "vulnerability", "threat", "reverse engineer" → `reverse-engineering-quick-triage`, `compliance`
- **Frontend**: "React", "UI", "components", "accessibility" → `react-specialist`, `frontend-performance-optimizer`
- **Database**: "schema", "query", "SQL", "optimization" → `sql-database-specialist`, `query-optimization-agent`

### Not Sure?
**Trigger**: Vague/ambiguous request
**Action**: Skill("interactive-planner") for multi-select questions

---

## 4. RESOURCE REFERENCE (COMPRESSED)

### 4.1 Skills (122 Total)

**Categories**:
- Development Lifecycle (15): Planning, architecture, implementation, testing, deployment
- Code Quality (12): Auditing, validation, optimization, clarity analysis
- Research (9): Literature review, systematic analysis, synthesis, deep research SOP
- Infrastructure (8): CI/CD, deployment, monitoring, orchestration
- Specialized (78): ML, security, frontend, backend, database, cloud, mobile

**Discovery**:
```bash
# List all skills
Glob(".claude/skills/**/SKILL.md") | head -20

# Search by keyword
npx claude-flow skills search "authentication"

# Get skill details
npx claude-flow skills info "api-development"
```

**Auto-Trigger**: Skills activate based on keywords in user request (see Section 3)

### 4.2 Playbooks (29 Total)

**Categories**:
- Delivery (5): Simple feature, Three-Loop, E2E shipping, bug fix, prototyping
- Operations (4): Production deployment, CI/CD setup, infrastructure scaling, performance
- Research (4): Deep Research SOP, quick investigation, planning & architecture, literature review
- Security (3): Security audit, compliance validation, reverse engineering
- Quality (3): Quick check, comprehensive review, dogfooding cycle
- Platform (3): ML pipeline, vector search/RAG, distributed neural training
- GitHub (3): PR management, release management, multi-repo coordination
- Specialist (4): Frontend, backend, full-stack, infrastructure as code

**Discovery**:
```bash
# List all playbooks
npx claude-flow playbooks list

# Search by domain
npx claude-flow playbooks search "machine learning"

# Show playbook structure
npx claude-flow playbooks info "deep-research-sop"
```

**Full Documentation**: `C:\Users\17175\claude-code-plugins\ruv-sparc-three-loop-system\docs\ENHANCED-PLAYBOOK-SYSTEM.md`

### 4.3 MCP Tools

**Categories**:
- **Coordination**: swarm_init, agent_spawn, task_orchestrate
- **Monitoring**: swarm_status, agent_metrics, task_status
- **Memory**: memory_store, vector_search (Memory MCP Triple System)
- **GitHub**: repo_analyze, pr_enhance, code_review
- **Neural**: neural_train, neural_patterns

**When to Use**:
- Coordination: Setup before spawning agents with Task tool (optional for complex tasks)
- Monitoring: Track progress during multi-agent workflows
- Memory: Cross-session persistence, semantic search (always use tagging protocol)
- GitHub: Repository operations, PR automation
- Neural: Agent learning, pattern optimization

**Setup**:
```bash
# Required: Claude Flow
claude mcp add claude-flow npx claude-flow@alpha mcp start

# Optional: Enhanced coordination
claude mcp add ruv-swarm npx ruv-swarm mcp start
claude mcp add flow-nexus npx flow-nexus@latest mcp start

# Verify
claude mcp list
```

**KEY**: MCP coordinates strategy. Claude Code's Task tool executes actual work.

### 4.4 Memory Tagging Protocol (REQUIRED)

**All Memory MCP writes MUST include**:

```javascript
const { taggedMemoryStore } = require('./hooks/12fa/memory-mcp-tagging-protocol.js');

// Automatic metadata injection
taggedMemoryStore('coder', 'Implemented auth feature', {
  task_id: 'AUTH-123',
  custom_field: 'value'
});
```

**Required Tags**:
- **WHO**: Agent name, category, capabilities
- **WHEN**: ISO timestamp, Unix timestamp, readable format
- **PROJECT**: Project identifier (connascence-analyzer, memory-mcp, claude-flow, etc.)
- **WHY**: Intent (implementation, bugfix, refactor, testing, documentation, analysis, planning, research)

**Memory Modes**:
- `execution`: Precise, actionable results (5-10 results)
- `planning`: Broader exploration (10-15 results)
- `brainstorming`: Wide ideation (15-20 results)

---

## 5. CRITICAL RULES & EDGE CASES

### 5.1 Absolute Rules

- **NO UNICODE EVER** (critical for Windows compatibility)
- **NEVER save files to root folder** (use /src, /tests, /docs, /config, /scripts)
- **ALWAYS batch operations in single message** (concurrent execution)
- **ONLY use agents from predefined registry** (never create custom types)

### 5.2 SPARC Methodology

When using SPARC approach:
1. **Specification** - Requirements analysis (`sparc run spec-pseudocode`)
2. **Pseudocode** - Algorithm design (`sparc run spec-pseudocode`)
3. **Architecture** - System design (`sparc run architect`)
4. **Refinement** - TDD implementation (`sparc tdd`)
5. **Completion** - Integration (`sparc run integration`)

**Commands**:
```bash
npx claude-flow sparc modes                    # List available modes
npx claude-flow sparc run <mode> "<task>"      # Execute specific mode
npx claude-flow sparc tdd "<feature>"          # Run complete TDD workflow
npx claude-flow sparc info <mode>              # Get mode details
```

### 5.3 Coordination Hooks (Every Agent MUST)

```bash
# Pre-Task
npx claude-flow hooks pre-task --description "Implement auth middleware"
npx claude-flow hooks session-restore --session-id "swarm-auth-123"

# During Task
npx claude-flow hooks post-edit --file "src/auth.js" --memory-key "swarm/coder/auth-123"
npx claude-flow hooks notify --message "JWT validation complete"

# Post-Task
npx claude-flow hooks post-task --task-id "AUTH-123"
npx claude-flow hooks session-end --export-metrics true
```

### 5.4 Troubleshooting

- **Memory MCP not working?** Check `~/.claude/claude_desktop_config.json`
- **Connascence Analyzer not working?** Verify server running on port 3000
- **Agent not found?** Check `.claude/agents/` registry with `Read(".claude/agents/README.md")`
- **Skill not triggering?** Verify keyword match in Section 3 (Playbook Router)
- **Playbook not found?** Run `npx claude-flow playbooks list` to see available

---

## 6. QUICK EXAMPLES

### Example 1: Simple Feature Implementation

```javascript
// User: "Build a REST API for user management"

// Step 1: intent-analyzer detects "API development"
// Step 2: prompt-architect optimizes request
// Step 3: Route to api-development playbook

[Single Message]:
  Skill("api-development")
  Task("Backend Developer", "Build REST API with Express...", "backend-dev")
  Task("Tester", "Write comprehensive tests...", "tester")
  Task("Reviewer", "Review security...", "reviewer")
  TodoWrite({ todos: [
    {content: "Design API architecture", status: "in_progress"},
    {content: "Implement endpoints", status: "pending"},
    {content: "Write tests", status: "pending"},
    {content: "Security review", status: "pending"},
    {content: "Deploy to staging", status: "pending"}
  ]})
  Write("src/api/users.js")
  Write("tests/api/users.test.js")
  Write("docs/API.md")
```

### Example 2: Deep Research (Academic ML)

```javascript
// User: "I need to replicate a baseline model for NeurIPS submission"

// Step 1: intent-analyzer detects "research", "baseline", "academic"
// Step 2: prompt-architect structures for deep-research-orchestrator
// Step 3: Route to Deep Research SOP playbook

[Single Message]:
  Skill("deep-research-orchestrator")
  Task("Data Steward", "Create datasheet, bias audit...", "data-steward")
  Task("Researcher", "Literature review, PRISMA protocol...", "researcher")
  Task("Coder", "Implement baseline model...", "coder")
  Task("Tester", "Validate ±1% tolerance...", "tester")
  Task("Ethics Agent", "Ethics review for Gate 1...", "ethics-agent")
  Task("Evaluator", "Quality Gate 1 validation...", "evaluator")
  TodoWrite({ todos: [
    {content: "Literature synthesis", status: "in_progress"},
    {content: "Create datasheet", status: "pending"},
    {content: "Replicate baseline", status: "pending"},
    {content: "Ethics review", status: "pending"},
    {content: "Gate 1 validation", status: "pending"}
  ]})
```

### Example 3: Code Quality Audit

```javascript
// User: "Audit code quality for clarity violations"

// Step 1: intent-analyzer detects "audit", "code quality", "clarity"
// Step 2: prompt-architect structures for clarity-linter
// Step 3: Route to clarity-linter skill

[Single Message]:
  Skill("clarity-linter")
  Task("Code Analyzer", "Run connascence analysis...", "code-analyzer")
  Task("Reviewer", "Evaluate rubric violations...", "reviewer")
  Task("Coder", "Generate fix patterns...", "coder")
  TodoWrite({ todos: [
    {content: "Collect metrics", status: "in_progress"},
    {content: "Evaluate rubric", status: "pending"},
    {content: "Generate fixes", status: "pending"}
  ]})
```

---

## 7. ADVANCED FEATURES

### 7.1 Three-Loop System (Flagship)

**When**: Complex features requiring research → implementation → validation

**Loop 1**: `research-driven-planning` (2-4 hours)
- 5x pre-mortem cycles
- Multi-agent consensus
- >97% planning accuracy

**Loop 2**: `parallel-swarm-implementation` (4-8 hours)
- 6-10 agents in parallel
- Theater detection
- Byzantine consensus

**Loop 3**: `cicd-intelligent-recovery` (1-2 hours)
- Automated testing
- Root cause analysis
- 100% recovery rate

**Total Time**: 8-14 hours
**Success Rate**: >97% planning accuracy, 100% test recovery

### 7.2 Connascence Analyzer (Production Ready)

**Status**: CODE QUALITY AGENTS ONLY (14 agents)

**Detection Capabilities**:
1. God Objects (26 methods vs 15 threshold)
2. Parameter Bombs/CoP (14 params vs 6 NASA limit)
3. Cyclomatic Complexity (13 vs 10 threshold)
4. Deep Nesting (8 levels vs 4 NASA limit)
5. Long Functions (72 lines vs 50 threshold)
6. Magic Literals/CoM (hardcoded ports, timeouts)

**Agent Access**: coder, reviewer, tester, code-analyzer, functionality-audit, theater-detection-audit, production-validator, sparc-coder, analyst, backend-dev, mobile-dev, ml-developer, base-template-generator, code-review-swarm

**Usage**: Skills auto-invoke when needed. Manual: `mcp__connascence-analyzer__analyze_workspace`

### 7.3 Dogfooding Cycle (Self-Improvement)

**Phase 1**: `sop-dogfooding-quality-detection` (30-60s)
- Run Connascence analysis
- Detect violations
- Store in Memory MCP with WHO/WHEN/PROJECT/WHY

**Phase 2**: `sop-dogfooding-pattern-retrieval` (10-30s)
- Vector search Memory MCP for similar violations
- Rank proven fixes
- Optionally apply fixes

**Phase 3**: `sop-dogfooding-continuous-improvement` (60-120s)
- Full cycle orchestration
- Sandbox testing
- Metrics tracking

**Triggers**: "analyze code quality", "run improvement cycle", "dogfood"

---

## 8. CHANGELOG

### v2.0.0 (2025-11-14)
- ✅ Migrated to playbook-first workflow system
- ✅ Removed redundant skill/agent/command catalogs (now in skills themselves)
- ✅ Added intent-analyzer bootstrap (auto-triggers on first message)
- ✅ Implemented reference system (queries instead of lists)
- ✅ Reduced from 2000+ lines to ~300 lines (85% reduction)
- ✅ Added 29 playbook router with keyword matching
- ✅ Compressed resource reference (categories + discovery commands)
- ✅ Added Quick Examples section

### v1.0.0 (Deprecated)
- ❌ Monolithic 2000+ line file with redundant lists
- ❌ Skills/agents/commands duplicated from their source files
- ❌ No auto-triggering intent detection
- ❌ Cognitive overload from exhaustive catalogs
- **Backup**: CLAUDE.md.v1.0-backup-20251114

---

## 9. SUPPORT & DOCUMENTATION

**Full Playbook Documentation**:
- `C:\Users\17175\claude-code-plugins\ruv-sparc-three-loop-system\docs\ENHANCED-PLAYBOOK-SYSTEM.md`

**Skill Inventory**:
- `C:\Users\17175\claude-code-plugins\ruv-sparc-three-loop-system\docs\SKILLS-INVENTORY.md`

**Agent Registry**:
- `C:\Users\17175\.claude\agents\README.md`

**Deep Research SOP**:
- `.claude/agents/research/deep-research-orchestrator.md`

**Claude Flow Documentation**:
- https://github.com/ruvnet/claude-flow
- https://github.com/ruvnet/claude-flow/issues

**Flow-Nexus Platform** (cloud features, requires authentication):
- https://flow-nexus.ruv.io

---

**Remember**: **Intent-first, playbook-second, skills execute**. Let the system route you to the right workflow!
