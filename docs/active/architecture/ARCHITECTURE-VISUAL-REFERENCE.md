# Skill→Agent→Command Architecture - Visual Reference & Data Flows

**Document Version**: 1.0.0  
**Format**: ASCII diagrams, data flow charts, sequence diagrams  
**Purpose**: Visual representation of system architecture and integration patterns

---

## 1. Three-Layer Architecture Overview

### 1.1 System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          USER INTERFACE LAYER                               │
│                     (224+ SLASH COMMANDS)                                   │
│                                                                             │
│  /build-feature  /review-pr  /fix-bug  /quick-check  /sparc  ...          │
│  /github-release /docker-deploy /security-audit /performance-tune         │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │
                                │ "binding: skill:X"
                                │
┌───────────────────────────────▼─────────────────────────────────────────────┐
│                          SKILL LAYER                                        │
│                     (93 SPECIALIZED SKILLS)                                │
│                                                                             │
│  ┌─────────────────────┐  ┌──────────────────────┐  ┌────────────────────┐ │
│  │ feature-dev-complete│  │ code-review-assistant│  │ theater-detection  │ │
│  │ (12 stages)         │  │ (multi-agent review) │  │ (6-agent consensus)│ │
│  └──────────┬──────────┘  └──────────┬───────────┘  └────────────┬────────┘ │
│             │                        │                           │          │
│  ┌──────────▼──────────┐  ┌──────────▼───────────┐  ┌──────────▼────────┐  │
│  │ parallel-swarm-impl │  │research-driven-plan  │  │ cicd-intelligence │  │
│  │ (Loop 2)            │  │ (Loop 1)             │  │ (Loop 3)          │  │
│  └─────────────────────┘  └──────────────────────┘  └───────────────────┘  │
│                                                                             │
│  + agentdb, flow-nexus, github, reverse-engineering, ml-expert             │
│  + sparc-methodology, deep-research-orchestrator, hooks-automation         │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │
                                │ "agents: [researcher, coder, tester, ...]"
                                │
┌───────────────────────────────▼─────────────────────────────────────────────┐
│                          AGENT LAYER                                        │
│                     (131 SPECIALIZED AGENTS)                               │
│                                                                             │
│  CORE (8)         TESTING (9)          FRONTEND (6)      DATABASE (7)      │
│  ├─ coder         ├─ tdd-london        ├─ react-dev       ├─ db-design    │
│  ├─ reviewer      ├─ e2e-testing       ├─ vue-dev         ├─ query-opt    │
│  ├─ tester        ├─ perf-testing      ├─ ui-builder      ├─ migration    │
│  ├─ researcher    ├─ security-testing  ├─ css-specialist  ├─ pipeline-eng │
│  ├─ planner       ├─ visual-regression ├─ accessibility   ├─ cache-strat  │
│  ├─ api-designer  ├─ contract-testing  └─ perf-optimizer  └─ [more...]   │
│  └─ [more...]     └─ [more...]                                            │
│                                                                             │
│  COORDINATION (15)  GITHUB (9)        SPARC (6)         RESEARCH (4)       │
│  ├─ hierarchical   ├─ pr-manager      ├─ specification   ├─ data-steward │
│  ├─ mesh-coord     ├─ code-review     ├─ pseudocode      ├─ ethics-agent │
│  ├─ adaptive       ├─ release-mgr     ├─ architecture    ├─ archivist    │
│  ├─ byzantine      ├─ issue-tracker   ├─ refinement      └─ evaluator    │
│  └─ [more...]      └─ [more...]       └─ [more...]                       │
│                                                                             │
│  [All 131 agents with distinct expertise, capabilities, decision frameworks]
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │
                    ┌───────────┼───────────┐
                    │           │           │
            ┌───────▼──────┐    │    ┌──────▼──────┐
            │  PRE-HOOKS   │    │    │ POST-HOOKS  │
            ├──────────────┤    │    ├─────────────┤
            │ Auto-assign  │    │    │ Format code │
            │ Validate cmd │    │    │ Train neural│
            │ Load context │    │    │ Update mem  │
            └──────────────┘    │    └─────────────┘
                                │
                    ┌───────────▼───────────┐
                    │   MEMORY MCP LAYER    │
                    ├───────────────────────┤
                    │ • Vector search       │
                    │ • Persistent storage  │
                    │ • WHO/WHEN/WHY tags   │
                    │ • 24h/7d/30d+ tiers   │
                    │ • Semantic indexing   │
                    └───────────────────────┘
                    
                    ┌───────────────────────┐
                    │ COORDINATION LAYER    │
                    ├───────────────────────┤
                    │ • Claude Flow MCP     │
                    │ • Swarm orchestration │
                    │ • Agent metrics       │
                    │ • Task scheduling     │
                    └───────────────────────┘
```

### 1.2 Layered Dependencies

```
SKILL-AGENT-COMMAND Dependency Flow:

Command
  │
  └─ binding: skill:X (declared in YAML frontmatter)
      │
      ├─ Load SKILL.md definition
      ├─ Extract agents list: [agent1, agent2, ...]
      ├─ Extract memory patterns
      ├─ Extract success criteria
      │
      └─ Agents Array
          │
          ├─ Agent 1
          │   ├─ Type: researcher
          │   ├─ Capabilities: [analysis, synthesis, ...]
          │   ├─ MCP Servers: [memory-mcp, claude-flow]
          │   ├─ Universal Commands: 40+ available
          │   └─ Agent-Specific Commands: [custom cmd1, ...]
          │
          ├─ Agent 2
          │   ├─ Type: coder
          │   ├─ Capabilities: [code-gen, optimization, ...]
          │   ├─ MCP Servers: [memory-mcp, connascence-analyzer, ...]
          │   ├─ Universal Commands: 40+ available
          │   └─ Agent-Specific Commands: [custom cmd1, ...]
          │
          └─ Agent N
              └─ [similar structure]

All Agents Share:
  ├─ File Operations: /file-read, /file-write, /glob-search, /grep-search
  ├─ Git Operations: /git-status, /git-commit, /git-push, ...
  ├─ Memory Operations: /memory-store, /memory-retrieve, /memory-search, ...
  ├─ Communication: /communicate-notify, /agent-delegate, /agent-coordinate
  └─ Testing: /test-run, /test-coverage, /test-validate, ...
```

---

## 2. Command Execution Flow Diagrams

### 2.1 Simple Linear Command Execution

```
User Input:
  /build-feature "user authentication with JWT"
         │
         ▼
┌─────────────────────────┐
│ COMMAND LAYER           │
│ /build-feature          │
├─────────────────────────┤
│ Parse YAML frontmatter: │
│ - name: build-feature   │
│ - binding: skill:       │
│   feature-dev-complete  │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────────────┐
│ SKILL LAYER                     │
│ feature-dev-complete            │
├─────────────────────────────────┤
│ 12-Stage Workflow:              │
│ 1. Research best practices      │
│ 2. Analyze codebase patterns    │
│ 3. Design architecture          │
│ 4. Generate diagrams            │
│ 5. Rapid prototype              │
│ 6. Theater detection            │
│ 7. Comprehensive testing        │
│ 8. Style polish                 │
│ 9. Security review              │
│ 10. Documentation               │
│ 11. Production readiness check  │
│ 12. Create PR / Deploy          │
│                                 │
│ Agents: [researcher, coder,     │
│          tester, analyst,       │
│          optimizer, coordinator]│
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│ AGENT LAYER                     │
│ Spawn 6 agents concurrently:    │
├─────────────────────────────────┤
│ Task(researcher, "Research      │
│   best practices for JWT auth", │
│   "researcher")                 │
│                                 │
│ Task(coder, "Design database   │
│   schema for users + tokens",   │
│   "coder")                      │
│                                 │
│ Task(coder, "Implement REST    │
│   endpoints with JWT auth",     │
│   "coder")                      │
│                                 │
│ Task(tester, "Write             │
│   comprehensive tests",          │
│   "tester")                     │
│                                 │
│ Task(analyst, "Review security", │
│   "analyst")                    │
│                                 │
│ Task(optimizer, "Optimize API  │
│   response times",              │
│   "optimizer")                  │
│                                 │
│ Task(coordinator, "Orchestrate  │
│   implementation", "coordinator")
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│ HOOK LAYER - PRE-EXECUTION      │
├─────────────────────────────────┤
│ npx claude-flow hooks pre-task  │
│ - Auto-assign agents by file    │
│ - Validate commands             │
│ - Load context                  │
│ - Prepare resources             │
└────────────┬────────────────────┘
             │
             ▼ Agents Execute Phases 1-12
             │ In Parallel With Coordination
             │
┌─────────────────────────────────┐
│ AGENT EXECUTION - PARALLEL      │
├─────────────────────────────────┤
│ Researcher:                     │
│  - Execute Phase 1: Research    │
│  - Execute Phase 2: Analyze     │
│  - Store findings in memory     │
│                                 │
│ Coder (x2):                     │
│  - Execute Phase 3: Design      │
│  - Execute Phase 5: Implement   │
│  - Store implementation in mem  │
│                                 │
│ Tester:                         │
│  - Execute Phase 7: Testing     │
│  - Store tests & coverage       │
│                                 │
│ Analyst:                        │
│  - Execute Phase 9: Security    │
│  - Store audit results          │
│                                 │
│ Optimizer:                      │
│  - Execute optimization         │
│  - Benchmark improvements       │
│                                 │
│ Coordinator:                    │
│  - Monitor all phases           │
│  - Aggregate results            │
│  - Coordinate between teams     │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│ HOOK LAYER - POST-EXECUTION     │
├─────────────────────────────────┤
│ npx claude-flow hooks post-task │
│ - Format code                   │
│ - Train neural patterns         │
│ - Update memory with results    │
│ - Track metrics                 │
│ - Generate summary              │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│ MEMORY MCP STORAGE              │
├─────────────────────────────────┤
│ All results stored with tagging:│
│ {                               │
│   who: {agent, category},       │
│   when: {iso, unix, readable},  │
│   project: "auth-feature",      │
│   why: "implementation"         │
│ }                               │
│                                 │
│ Keys:                           │
│ - swarm/feature-dev/plan        │
│ - swarm/feature-dev/arch        │
│ - swarm/feature-dev/impl        │
│ - swarm/feature-dev/tests       │
│ - swarm/feature-dev/security    │
│ - swarm/feature-dev/complete    │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│ OUTPUT TO USER                  │
├─────────────────────────────────┤
│ ✅ Feature Development Complete │
│                                 │
│ Artifacts:                      │
│ ✅ Research (documented)        │
│ ✅ Architecture (designed)      │
│ ✅ Implementation (passing)     │
│ ✅ Tests (95% coverage)         │
│ ✅ Security (no issues)         │
│ ✅ Optimization (benchmarked)   │
│ ✅ Documentation (complete)     │
│ ✅ PR created                   │
└─────────────────────────────────┘
```

### 2.2 Multi-Agent Parallel Execution (PR Review)

```
User Input:
  /review-pr 123
         │
         ▼
┌────────────────────────────┐
│ COMMAND: /review-pr        │
│ Binding: code-review-asst  │
└────────────┬───────────────┘
             │
             ▼
┌────────────────────────────┐
│ SKILL: code-review-asst    │
│ Agents: [5 specialists]    │
│ Coordinator: adaptive      │
└────────────┬───────────────┘
             │
             ▼ SPAWN 5 PARALLEL AGENTS
             │
    ┌────────┼────────┬────────┬────────┐
    │        │        │        │        │
    ▼        ▼        ▼        ▼        ▼
┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
│Security│ │Perform│ │ Style  │ │ Test   │ │  Doc   │
│Reviewer│ │ance   │ │Reviewer│ │Special │ │Reviewer│
│(analyst)│ │Analyst│ │(analyst)│ │(analyst)│ │(analyst)│
│        │ │(optim)│ │        │ │        │ │        │
└────┬───┘ └───┬────┘ └────┬───┘ └───┬────┘ └────┬───┘
     │         │          │        │         │
     │ Review  │ Analyze  │ Check  │ Verify  │ Document
     │ Vulns  │ Bottlenecks│ Style │ Coverage│ Completeness
     │ Secrets │ Optimization│ Format│ Edge Cases│ Clarity
     │ OWASP   │ Caching    │ Names │ Mocks   │ Examples
     │         │            │       │         │
     │ Store:  │ Store:     │Store: │ Store:  │ Store:
     │ swarm/  │ swarm/     │swarm/ │ swarm/  │ swarm/
     │ review/ │ review/    │review/│ review/ │ review/
     │ security│ performance│ style │ tests   │ docs
     │         │            │       │         │
     └────┬────┴────┬───────┴───┬───┴────┬────┘
          │         │           │        │
          └─────────┼───────────┼────────┘
                    │           │
            ┌───────▼───────────▼────────┐
            │ ADAPTIVE COORDINATOR       │
            ├────────────────────────────┤
            │ Wait for all 5 agents      │
            │ Aggregate findings:        │
            │ - Security: 95/100         │
            │ - Performance: 88/100      │
            │ - Style: 90/100            │
            │ - Tests: 82/100            │
            │ - Docs: 85/100             │
            │                            │
            │ Calculate: Quality = 88/100│
            │ Decision: APPROVE          │
            │                            │
            │ Post comment on GitHub     │
            │ with all 5 reviews         │
            └────────────────────────────┘
```

### 2.3 Theater Detection - 6-Agent Byzantine Consensus

```
User Input:
  /quick-check src/auth.js
         │
         ▼
┌──────────────────────────────────────┐
│ SKILL: theater-detection-audit       │
│ Objective: Verify real implementation│
└──────────────┬──────────────────────┘
               │
        ┌──────┴──────────────┬────────────┬────────────┬───────────┐
        │                     │            │            │           │
        ▼                     ▼            ▼            ▼           ▼
┌────────────────┐  ┌───────────────┐ ┌──────────────┐ ┌────────┐ ┌──────┐
│ Agent 1:       │  │ Agent 2:      │ │ Agent 3:     │ │Agent 4:│ │Agent5:│
│ Sandbox        │  │ Coverage      │ │ Mock         │ │Integr. │ │Perf.  │
│ Execution      │  │ Validation    │ │ Detection    │ │Testing │ │Bench  │
│                │  │               │ │              │ │        │ │       │
│ Run in sandbox │  │ Check test    │ │ Find TODOs   │ │ Run    │ │ Run   │
│ Watch output   │  │ coverage %    │ │ Find stubbed │ │ integ. │ │ perf  │
│ Verify result  │  │ % vs threshold│ │ methods      │ │ tests  │ │ tests │
│                │  │ Validate      │ │ Validate     │ │ Verify │ │Measure│
│ Result:       │  │ maintained    │ │ no fakes     │ │ flow   │ │speed  │
│ ✅ Real       │  │              │ │             │ │✅Works │ │✅Fast │
│ ❌ Theater    │  │ ✅ 95% ✅   │ │ ✅ Clean   │ │       │ │       │
│               │  │ ❌ <80%❌    │ │ ❌ Fakes❌ │ │       │ │       │
└────────┬───────┘  └────────┬─────┘ └──────┬───────┘ └───┬────┘ └──┬───┘
         │                   │              │            │      │
         └───────────────────┼──────────────┼────────────┼──────┘
                             │              │            │
                    ┌────────▼──────────────▼────────────▼─────┐
                    │ BYZANTINE CONSENSUS                     │
                    ├─────────────────────────────────────────┤
                    │ Vote tallying:                          │
                    │ Agent 1: ✅ REAL                        │
                    │ Agent 2: ✅ GOOD COVERAGE               │
                    │ Agent 3: ✅ NO THEATER                  │
                    │ Agent 4: ✅ INTEGRATION WORKS           │
                    │ Agent 5: ✅ PERFORMANCE OK              │
                    │ Agent 6: ✅ SECURITY PASS               │
                    │                                         │
                    │ CONSENSUS (≥5/6):                       │
                    │ ✅ REAL IMPLEMENTATION                  │
                    │ ✅ Confidence: 100%                     │
                    │                                         │
                    │ OR                                      │
                    │                                         │
                    │ Agent 3: ❌ DETECTED MOCKS              │
                    │ Agent 4: ❌ INTEGRATION FAILS           │
                    │                                         │
                    │ CONSENSUS (≤4/6):                       │
                    │ ❌ THEATER DETECTED                     │
                    │ ❌ Confidence: 50%                      │
                    │ ⚠️  Needs investigation                 │
                    └─────────────────────────────────────────┘
```

---

## 3. Memory MCP Integration Flow

### 3.1 Memory Storage Pattern

```
Agent Operation Complete
        │
        ▼
┌───────────────────────────────────────┐
│ MEMORY WRITE WITH TAGGING              │
├───────────────────────────────────────┤
│ memory_store(                          │
│   key: "swarm/coder/implementation",   │
│   value: {                             │
│     file: "src/auth.js",              │
│     implementation: {...},             │
│     decision: {...}                   │
│   },                                   │
│   tags: {                              │
│     WHO: {                             │
│       agent: "coder",                 │
│       category: "code-quality",       │
│       capabilities: [...]             │
│     },                                 │
│     WHEN: {                            │
│       iso: "2025-11-06T10:30:00Z",   │
│       unix: 1730875800,               │
│       readable: "Nov 6, 2025 10:30"   │
│     },                                 │
│     PROJECT: "user-authentication",    │
│     WHY: "implementation"              │
│   }                                    │
│ )                                      │
└───────────────────┬───────────────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │ MEMORY MCP BACKEND    │
        ├───────────────────────┤
        │ 1. Vectorize content  │
        │    (384-dim embedding)│
        │                       │
        │ 2. Index with HNSW    │
        │    (150x faster)      │
        │                       │
        │ 3. Store metadata     │
        │    (WHO/WHEN/WHY/etc) │
        │                       │
        │ 4. Assign retention   │
        │    tier:              │
        │    - 24h (short)      │
        │    - 7d (medium)      │
        │    - 30d+ (long)      │
        │                       │
        │ 5. Tag for retrieval  │
        └───────────────────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │ RETRIEVAL PATTERNS    │
        ├───────────────────────┤
        │ memory_search(        │
        │   pattern:            │
        │   "swarm/*/impl*",    │
        │   query:              │
        │   "JWT auth strategy" │
        │ ) →                   │
        │ Returns: Top 5-20     │
        │ semantic matches      │
        │ with ranking          │
        └───────────────────────┘
```

### 3.2 Three-Layer Memory Retention

```
SHORT-TERM (24 hours)
├─ Current session context
├─ Active agent decisions
├─ In-progress implementations
└─ Temporal query results

Example Keys:
├─ swarm/feature-dev/current-phase
├─ swarm/coder/active-files
└─ swarm/review-pr/findings

MEDIUM-TERM (7 days)
├─ Completed feature artifacts
├─ Session summaries
├─ Pattern discoveries
└─ Test results & coverage

Example Keys:
├─ swarm/feature-dev/completed
├─ swarm/sessions/summary-2025-11-06
└─ swarm/patterns/javascript-best-practices

LONG-TERM (30+ days)
├─ Historical failure patterns
├─ Lessons learned
├─ Architecture decisions
├─ Re-usable solutions

Example Keys:
├─ integration/loop3-failure-patterns
├─ lessons-learned/bug-fixes
├─ architecture/design-decisions
└─ solutions/authentication-patterns
```

---

## 4. Three-Loop Integration Flows

### 4.1 Complete Three-Loop Cycle

```
ITERATION N+1:

┌─────────────────────────────────────────────────────────────────────────┐
│                            LOOP 1: PLANNING                             │
│              research-driven-planning (Skill)                           │
│                                                                         │
│  Input: User specification + Historical failures (from Loop 3)         │
│  ┌──────────┬──────────┬──────────┬──────────────────────────────┐    │
│  │          │          │          │                              │    │
│  ▼          ▼          ▼          ▼                              │    │
│ Spec    Research    Planning    Pre-Mortem (5x)                │    │
│          ↓            ↓            ↓                              │    │
│     Evidence-based   MECE      Risk mitigation                 │    │
│     findings      breakdown      + prevention                  │    │
│          ↓            ↓            ↓                              │    │
│  ┌──────────┴──────────┴──────────┴──────────────────────────┐ │    │
│  │ LOOP 1 OUTPUT:                                            │ │    │
│  │ ✅ Enhanced plan (with research)                          │ │    │
│  │ ✅ MECE task breakdown                                    │ │    │
│  │ ✅ Risk analysis (5x pre-mortem)                          │ │    │
│  │ ✅ Prevention strategies                                  │ │    │
│  │                                                           │ │    │
│  │ Memory Namespace: integration/loop1-to-loop2              │ │    │
│  │ Artifacts: .artifacts/loop1-planning-package.json         │ │    │
│  └──────────────┬──────────────────────────────────────────┘ │    │
│                 │                                             │    │
│                 └─────────────────────────────────────────────┘    │
└─────────────────────────────────┬──────────────────────────────────┘
                                  │
                                  │ Planning Package
                                  │ (enhanced plan, research, risks)
                                  │
┌─────────────────────────────────▼──────────────────────────────────┐
│                       LOOP 2: IMPLEMENTATION                        │
│            parallel-swarm-implementation (Skill)                   │
│                                                                    │
│  Input: Planning package from Loop 1                             │
│  ┌─────────┬─────────┬──────────┬──────────┬────────────────┐   │
│  │         │         │          │          │                │   │
│  ▼         ▼         ▼          ▼          ▼                ▼   │
│ Init   Discover   MECE      Deploy    Theater    Integrate  │   │
│         ↓          ↓          ↓         ↓          ↓         │   │
│      Analyze    Divide    Multi-   Validate   Combine       │   │
│      require    tasks     agent    real impl  & test        │   │
│      ments      by        swarm    vs fakes               │   │
│                 type                                        │   │
│         │         │         │         │         │         │   │
│         └────┬────┴────┬────┴────┬────┴────┬────┴─────┐   │   │
│              │         │         │         │          │   │   │
│         ┌────▼─────────▼─────────▼─────────▼──────────▼┐  │   │
│         │ LOOP 2 OUTPUT:                             │  │   │
│         │ ✅ Implementation (all files)              │  │   │
│         │ ✅ Test suite (95%+ coverage)             │  │   │
│         │ ✅ Theater audit (passed)                 │  │   │
│         │ ✅ Integration status                     │  │   │
│         │ ✅ Quality metrics                        │  │   │
│         │                                           │  │   │
│         │ Memory Namespace: integration/loop2-to-3   │  │   │
│         │ Artifacts: .artifacts/loop2-delivery-pkg   │  │   │
│         └────────────┬────────────────────────────┘  │   │
└────────────────────────┼──────────────────────────────┘   │
                         │                                  │
                         │ Delivery Package
                         │ (implementation, tests, etc)
                         │
┌────────────────────────▼───────────────────────────────────┐
│                  LOOP 3: CI/CD QUALITY                      │
│         cicd-intelligent-recovery (Skill)                  │
│                                                            │
│  Input: Delivery package from Loop 2                      │
│  ┌─────────┬──────────┬─────────┬──────────┬──────────┐  │
│  │         │          │         │          │          │  │
│  ▼         ▼          ▼         ▼          ▼          ▼  │
│ GitHub   Analysis   Root     Intelligent Validate  Learn │
│ Hooks     ↓        Cause    Fixes       ↓         ↓     │
│       Test         Detection (Codex)   Re-test  Pattern │
│       failures      ↓        ↓          ↓        Storage │
│       Coverage   Find root  Apply      Confirm          │
│       issues     cause      fixes      works            │
│                                                        │
│  ┌────────────────────────────────────────────────────┐ │
│  │ LOOP 3 OUTPUT:                                     │ │
│  │ ✅ 100% test success (or failure patterns)        │ │
│  │ ✅ Root causes identified                         │ │
│  │ ✅ Fixes validated                                │ │
│  │ ✅ Failure patterns documented                    │ │
│  │ ✅ Prevention strategies recorded                 │ │
│  │                                                   │ │
│  │ Memory Namespace: integration/loop3-feedback      │ │
│  │ Artifacts: .artifacts/loop3-failure-patterns.json │ │
│  └────────────────────┬─────────────────────────────┘ │
└───────────────────────┼────────────────────────────────┘
                        │
                        │ Failure Patterns
                        │ (what failed, why, how to prevent)
                        │
                        │ FEEDBACK
                        │ Feed historical failures back
                        │ to Loop 1 for next iteration
                        │
                        ▼
                ┌──────────────────┐
                │ NEXT ITERATION:  │
                │ Loop 1 loads     │
                │ failure patterns │
                │ Enhanced         │
                │ pre-mortem with  │
                │ real data        │
                │ Better planning  │
                │ → Better impl    │
                │ → Better quality │
                └──────────────────┘
```

### 4.2 Memory-Based Feedback Loop

```
Loop 3 Completion:
Failure Patterns Identified
        │
        ▼
memory_store(
  "integration/loop3-feedback/patterns",
  {
    failures: [
      {
        type: "race-condition",
        location: "auth/token-refresh.js",
        cause: "concurrent refresh requests",
        fix: "add mutex lock",
        prevention: "queue mechanism"
      },
      {
        type: "missing-edge-case",
        scenario: "token expiry during request",
        fix: "retry with new token",
        prevention: "test edge cases early"
      },
      ...
    ],
    patterns: {
      category: "async-bugs",
      frequency: 3,
      trend: "increasing"
    },
    lessons: [
      "Test concurrent operations early",
      "Add mutex locks for shared state",
      "Test edge cases in pre-mortem"
    ]
  },
  tags: { who: evaluator, why: "analysis" }
)
        │
        ▼
Next Project Starts:
Loop 1: research-driven-planning
        │
        ├─ Load historical failures
        │  memory_search("integration/loop3-feedback/*")
        │
        ├─ Analyze patterns
        │  "Async bugs found 3x, test concurrency early"
        │
        ├─ Incorporate into planning
        │  Add: "Concurrent operation testing to all pre-mortems"
        │
        ├─ Enhanced pre-mortem
        │  + Question: "What happens with concurrent requests?"
        │  + Question: "What happens when tokens expire mid-operation?"
        │
        └─ Better risk identification
           Better planning → Better implementation → Better quality
```

---

## 5. Agent Specialization Matrix

### 5.1 Agent Type → Capability Mapping

```
┌─────────────────────────────────────────────────────────────────────┐
│                     AGENT SPECIALIZATION MATRIX                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  RESEARCH/ANALYSIS AGENTS                                          │
│  ├─ researcher          → Analysis, Evidence, Synthesis           │
│  ├─ system-architect    → Design, Architecture, Planning          │
│  ├─ api-designer        → API design, Interface definition        │
│  └─ code-analyzer       → Code inspection, Pattern detection      │
│                                                                    │
│  IMPLEMENTATION AGENTS                                            │
│  ├─ coder               → Code generation, Refactoring           │
│  ├─ backend-dev         → Server-side, APIs, databases           │
│  ├─ frontend-developer  → UI, React, Vue, CSS                    │
│  ├─ mobile-dev          → iOS, Android, React Native             │
│  ├─ ml-developer        → ML pipelines, models, training         │
│  └─ database-architect  → Schema design, optimization            │
│                                                                    │
│  QUALITY ASSURANCE AGENTS                                         │
│  ├─ tester              → Test generation, coverage              │
│  ├─ reviewer            → Code review, best practices            │
│  ├─ e2e-testing-spec    → End-to-end testing workflows           │
│  ├─ security-tester     → Vulnerability scanning, OWASP          │
│  ├─ performance-tester  → Load testing, benchmarking             │
│  └─ accessibility-spec  → WCAG compliance, a11y                  │
│                                                                    │
│  OPTIMIZATION AGENTS                                              │
│  ├─ perf-analyzer       → Performance analysis, bottlenecks      │
│  ├─ query-optimizer     → Database query tuning                  │
│  ├─ bundle-optimizer    → Build optimization, size               │
│  └─ cache-strategist    → Caching strategies, TTLs               │
│                                                                    │
│  COORDINATION AGENTS                                              │
│  ├─ hierarchical-coord  → Tree-based coordination               │
│  ├─ mesh-coordinator    → Peer-to-peer coordination             │
│  ├─ adaptive-coordinator→ Topology-adaptive coordination        │
│  ├─ byzantine-coord     → Consensus with fault tolerance        │
│  ├─ raft-manager        → Raft consensus protocol               │
│  └─ consensus-builder   → Agreement mechanisms                   │
│                                                                    │
│  PLATFORM AGENTS                                                  │
│  ├─ github-pr-manager   → PR lifecycle, merging                 │
│  ├─ release-manager     → Versioning, deployment                │
│  ├─ cicd-engineer       → CI/CD pipelines, automation           │
│  ├─ docker-specialist   → Containerization, Compose             │
│  └─ kubernetes-expert   → Orchestration, scaling                │
│                                                                    │
│  SPECIALIZED RESEARCH AGENTS                                      │
│  ├─ data-steward        → Data quality, versioning, bias audit  │
│  ├─ ethics-agent        → Risk assessment, safety, fairness     │
│  ├─ archivist           → Reproducibility, DOI, archival        │
│  └─ evaluator           → Quality gates, GO/NO-GO decisions     │
│                                                                    │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.2 MCP Server Access Control

```
┌──────────────────────────────────────────────────────────────────┐
│                   MCP SERVER ACCESS CONTROL                       │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  CODE QUALITY AGENTS (14)                                       │
│  Can Access: ✅ memory-mcp                                      │
│              ✅ connascence-analyzer                            │
│              ✅ claude-flow                                     │
│  Cannot Access: ❌ No restrictions (all tools available)        │
│                                                                 │
│  coder, reviewer, tester, code-analyzer, analyst,              │
│  functionality-audit, theater-detection, production-validator,  │
│  sparc-coder, backend-dev, mobile-dev, ml-developer,           │
│  base-template-generator, code-review-swarm                    │
│                                                                 │
│  ─────────────────────────────────────────────────────────────│
│                                                                 │
│  PLANNING AGENTS (23)                                          │
│  Can Access: ✅ memory-mcp                                     │
│              ✅ claude-flow                                    │
│  Cannot Access: ❌ connascence-analyzer                        │
│  Reason: Cannot analyze code quality (only for code agents)   │
│                                                                 │
│  planner, researcher, system-architect, specification,          │
│  pseudocode, architecture, hierarchical-coordinator,           │
│  mesh-coordinator, adaptive-coordinator, byzantine-coordinator,│
│  raft-manager, gossip-coordinator, consensus-builder,          │
│  crdt-synchronizer, quorum-manager, security-manager,          │
│  perf-analyzer, performance-benchmarker, task-orchestrator,    │
│  memory-coordinator, [others]                                  │
│                                                                 │
│  ─────────────────────────────────────────────────────────────│
│                                                                 │
│  ALL AGENTS (131 total)                                        │
│  Can Access: ✅ claude-flow (coordination)                     │
│              ✅ memory-mcp (persistent memory)                 │
│              ✅ focused-changes (change tracking)              │
│              ✅ ToC (documentation)                            │
│              ✅ 40+ universal commands                         │
│                                                                 │
└──────────────────────────────────────────────────────────────────┘
```

---

## 6. Configuration Quick Reference

### 6.1 Key Configuration Files

```
.claude/settings.json
├─ env: CLAUDE_FLOW_HOOKS_ENABLED=true
├─ hooks: Pre/Post-task automation
├─ permissions: Allowed bash commands
└─ alwaysThinking: true

.mcp.json
├─ claude-flow (REQUIRED)
├─ ruv-swarm (optional)
└─ flow-nexus (optional)

agents/registry.json
├─ 131 agents defined
├─ Each with: type, capabilities, MCP access
└─ Prompting techniques specified

skills/MASTER-SKILLS-INDEX.md
├─ 93 skills categorized
├─ Trigger keywords documented
└─ Skill-agent bindings listed

commands/README.md
├─ 224 commands organized
├─ MECE taxonomy documented
└─ Command-skill bindings listed
```

### 6.2 Critical Rules Checklist

```
GOLDEN RULE CHECKLIST:
□ ONE MESSAGE = ALL RELATED OPERATIONS
□ Batch TodoWrite calls (5-10+ minimum)
□ Batch Task tool calls (all agents at once)
□ Batch file operations (all reads/writes together)
□ Batch Bash commands (all terminal ops together)
□ Batch memory operations (all store/retrieve together)

EXECUTION BEST PRACTICES:
□ Use specialist agent types (researcher, coder, analyst, optimizer, coordinator)
□ NOT generic types ("general-purpose", "developer", "agent")
□ Use Claude Code Task tool for execution (NOT just MCP)
□ MCP tools for coordination setup only (not execution)
□ Pre-task hooks validate and prepare
□ Post-task hooks format, train, update memory
□ All memory writes tagged with WHO/WHEN/PROJECT/WHY

FILE ORGANIZATION:
□ NEVER save working files to root folder
□ Use /src, /tests, /docs, /config, /scripts, /examples
□ Documentation goes in /docs
□ Tests go in /tests
□ Source code goes in /src
```

---

## Conclusion

This visual reference covers:
- **Architecture layers** with dependency flows
- **Command execution** with sequence diagrams
- **Parallel agent** coordination patterns
- **Byzantine consensus** for theater detection
- **Three-loop** integration with memory feedback
- **Access control** matrices
- **Configuration** management

The system enables **theater-free, production-ready software delivery** at scale through clear separation of concerns, specialized expertise, and continuous quality enforcement via hooks and memory-based feedback loops.
