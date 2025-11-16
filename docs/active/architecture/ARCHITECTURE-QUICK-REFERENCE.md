# Skill→Agent→Command Architecture - Quick Reference

**Document Version**: 1.0.0  
**Purpose**: Fast lookup guide for architecture concepts  
**Audience**: Developers, architects, AI engineers  

---

## 1. What Is This Architecture?

A **three-layer plugin system** for AI-assisted software development:

```
COMMANDS (224)           User interface (slash commands)
    ↓ bind to
SKILLS (93)              Methodology layer (how to solve problems)
    ↓ delegate to
AGENTS (131)             Expertise layer (who solves them)
    ↓ coordinated by
MEMORY MCP + HOOKS       Persistence & automation
```

**Key Innovation**: Theater detection via 6-agent Byzantine consensus validates real implementation vs fake.

---

## 2. Three Layers Explained

### Layer 1: SKILLS (93 total)

**What**: High-level frameworks for solving specific problems  
**Where**: `skills/` directory  
**Format**: Markdown with YAML frontmatter (`SKILL.md`)

**Key Skills**:
- `feature-dev-complete` - 12-stage feature development
- `research-driven-planning` - Loop 1: Planning with risk analysis
- `parallel-swarm-implementation` - Loop 2: Multi-agent execution
- `cicd-intelligent-recovery` - Loop 3: Quality & failure recovery
- `code-review-assistant` - Multi-agent PR reviews
- `theater-detection-audit` - 6-agent consensus validation
- `agent-creator` - 4-phase SOP for creating agents

**How to Find Skills**:
```bash
# Central registry
docs/MASTER-SKILLS-INDEX.md       # All 93 skills
skills/THREE-LOOP-INTEGRATION-*.md  # Three-loop architecture
skills/CREATION_SUMMARY.md          # Recent additions
```

### Layer 2: AGENTS (131 total)

**What**: Specialized AI agents with distinct expertise  
**Where**: `agents/` directory  
**Format**: Markdown with YAML frontmatter (`{agent}.md`)

**Agent Categories**:
| Category | Count | Examples |
|----------|-------|----------|
| Core Development | 8 | coder, reviewer, tester, researcher, planner |
| Testing & Validation | 9 | e2e-testing, security-testing, perf-testing |
| Frontend | 6 | react-dev, vue-dev, ui-builder |
| Database | 7 | db-design, query-optimizer, cache-strategist |
| Coordination | 15 | hierarchical, mesh, adaptive, byzantine |
| GitHub | 9 | pr-manager, release-manager, issue-tracker |
| Specialized | 12+ | backend-dev, ml-developer, cicd-engineer |

**5 Specialist Types** (use these, NOT "general-purpose"):
- `researcher` - Analyze, research, investigate
- `coder` - Build, implement, create, fix
- `analyst` - Review, test, audit, validate
- `optimizer` - Optimize, improve, tune, benchmark
- `coordinator` - Orchestrate, coordinate, manage, delegate

### Layer 3: COMMANDS (224 total)

**What**: User-facing slash commands  
**Where**: `commands/` directory  
**Format**: Markdown with YAML frontmatter  

**Command Categories** (MECE):
| Domain | Count |
|--------|-------|
| Development Workflow | 67 |
| Quality & Validation | 29 |
| Agent Lifecycle | 18 |
| Memory & State | 18 |
| Monitoring & Telemetry | 18 |
| Performance & Optimization | 18 |
| Integration & External | 21 |
| Research & Analysis | 20 |
| Automation & Hooks | 22 |
| Core Infrastructure | 3 |

**Essential Commands**:
- `/build-feature` - 12-stage feature development
- `/review-pr` - Multi-agent PR review
- `/fix-bug` - Smart bug fixing with root cause analysis
- `/quick-check` - Lightning-fast quality check
- `/sparc` - SPARC methodology execution

---

## 3. How Data Flows Through Layers

### Simple Flow: Command → Skill → Agents

```
User Input:
  /build-feature "user authentication"
         ↓
Parse YAML frontmatter:
  name: build-feature
  binding: skill:feature-dev-complete
         ↓
Load Skill: feature-dev-complete
  agents: [researcher, coder, tester, analyst, optimizer, coordinator]
  12 phases defined
         ↓
Spawn Agents (Claude Code Task tool):
  Task("researcher", "Phase 1: Research", "researcher")
  Task("coder", "Phase 3: Design", "coder")
  Task("coder", "Phase 5: Implement", "coder")
  ... (all in one message for parallelism)
         ↓
Pre-Hooks Execute:
  - Auto-assign agents by file type
  - Validate commands
  - Load context
         ↓
Agents Execute Skill Phases (parallel):
  Each agent runs its phases, stores results in memory
         ↓
Post-Hooks Execute:
  - Format code
  - Train neural patterns
  - Update memory
  - Track metrics
         ↓
User Gets Output:
  ✅ Feature complete, tests passing, PR ready
```

### Parallel Execution: Multi-Agent Review

```
/review-pr 123
         ↓
5 Agents Spawn Simultaneously:
  Task(analyst, "Security review", "analyst")
  Task(optimizer, "Performance review", "optimizer")
  Task(analyst, "Style review", "analyst")
  Task(analyst, "Test review", "analyst")
  Task(analyst, "Documentation review", "analyst")
         ↓
All 4 Execute in Parallel:
  Security → 95/100
  Performance → 88/100
  Style → 90/100
  Tests → 82/100
  Documentation → 85/100
         ↓
Adaptive Coordinator Aggregates:
  Final Score: 88/100
  Decision: APPROVE
  Post comment with all findings
```

---

## 4. Memory MCP Integration

### Pattern: WHO/WHEN/PROJECT/WHY Tagging

Every memory write includes metadata:

```javascript
memory_store(
  key: "swarm/coder/implementation",
  value: { code, decisions, ... },
  tags: {
    WHO: {
      agent: "coder",
      category: "code-quality",
      capabilities: ["code-gen", "optimization"]
    },
    WHEN: {
      iso: "2025-11-06T10:30:00Z",
      unix: 1730875800,
      readable: "Nov 6, 2025 10:30"
    },
    PROJECT: "user-authentication",
    WHY: "implementation"  // implementation|bugfix|refactor|testing|documentation|analysis|planning|research
  }
)
```

### Memory Retrieval: Semantic Search

```javascript
memory_search(
  pattern: "swarm/*/auth*",
  query: "JWT token refresh strategy"
) 
→ Returns top 5-20 semantic matches ranked by relevance
```

### Retention Tiers

- **24 hours** - Current session context, active decisions
- **7 days** - Completed features, session summaries, patterns
- **30+ days** - Historical failures, architecture decisions, lessons learned

---

## 5. Three-Loop Integration

### Loop 1: Planning (research-driven-planning)

```
Input: User specification
       ↓
Step 1: Research (Gemini Search)
        Evidence-based findings
       ↓
Step 2: Planning (MECE breakdown)
        Task division
       ↓
Step 3: Pre-Mortem (5x risk analysis)
        Identify risks & mitigation
       ↓
Output: loop1-planning-package.json
        Memory: integration/loop1-to-loop2
```

### Loop 2: Implementation (parallel-swarm-implementation)

```
Input: Planning package from Loop 1
       ↓
Step 1: Discovery & Initialization
Step 2: MECE task division
Step 3: Multi-agent swarm deployment
Step 4: Theater detection (6-agent consensus)
Step 5: Integration & validation
       ↓
Output: loop2-delivery-package.json
        (code, tests, theater audit, metrics)
        Memory: integration/loop2-to-loop3
```

### Loop 3: CI/CD Quality (cicd-intelligent-recovery)

```
Input: Delivery package from Loop 2
       ↓
Step 1: GitHub hooks trigger
Step 2: Test failures analysis
Step 3: Root cause detection
Step 4: Intelligent fixes (Codex)
Step 5: Validation re-testing
       ↓
Output: loop3-failure-patterns.json
        (what failed, why, prevention strategies)
        Memory: integration/loop3-feedback
        
        ↓ FEEDBACK
        
Next iteration's Loop 1 loads historical failures
        ↓
Better planning due to real failure data
        ↓
Better implementation
        ↓
Better quality
```

---

## 6. Golden Rules & Best Practices

### GOLDEN RULE
```
1 MESSAGE = ALL RELATED OPERATIONS

Mandatory patterns:
- TodoWrite: Batch ALL todos in ONE call (5-10+ minimum)
- Task Tool: Spawn ALL agents in ONE message
- File Ops: Batch ALL reads/writes in ONE message
- Bash Cmds: Batch ALL terminal ops in ONE message
- Memory Ops: Batch ALL store/retrieve in ONE message
```

### Agent Selection Decision Tree

```
User says "analyze"     → researcher
User says "build"       → coder
User says "review"      → analyst
User says "optimize"    → optimizer
User says "manage"      → coordinator

WRONG: "general-purpose", "developer", "agent"
RIGHT: "researcher", "coder", "analyst", "optimizer", "coordinator"
```

### Access Control Rules

```
CODE QUALITY AGENTS (14):
├─ coder, reviewer, tester, analyst, code-analyzer
├─ functionality-audit, theater-detection-audit
├─ production-validator, sparc-coder, backend-dev
├─ mobile-dev, ml-developer, base-template-generator
└─ code-review-swarm
Can Access: memory-mcp, connascence-analyzer, claude-flow

PLANNING AGENTS (23):
├─ planner, researcher, system-architect, specification
├─ pseudocode, architecture, refinement
├─ hierarchical-coordinator, mesh-coordinator, adaptive-coordinator
├─ byzantine-coordinator, raft-manager, gossip-coordinator
└─ consensus-builder, crdt-synchronizer, quorum-manager, [more]
Can Access: memory-mcp, claude-flow (NO connascence-analyzer)
```

### File Organization

```
DO:
✅ /src - Source code
✅ /tests - Test files
✅ /docs - Documentation
✅ /config - Configuration
✅ /scripts - Utility scripts

DON'T:
❌ Save working files to root folder
❌ Mix documentation and code
❌ Put tests in source directories (separate /tests folder)
```

---

## 7. Directory Quick Navigation

```
C:\Users\17175\
├── skills/                    # All 93 skills
│   ├── MASTER-SKILLS-INDEX.md    # Registry
│   ├── feature-dev-complete/
│   ├── when-*/ (trigger-based skills)
│   └── [90+ more skills]
│
├── .claude-code/skills/       # Additional skills
│   ├── advanced-coordination/
│   ├── agent-creation/
│   └── utilities/
│
├── agents/                    # All 131 agents
│   ├── registry.json             # Central registry
│   ├── core/                     # 8 core agents
│   ├── frontend/                 # 6 frontend agents
│   ├── database/                 # 7 database agents
│   ├── testing/                  # 9 testing agents
│   ├── swarm/                    # 15 coordination agents
│   └── [15+ more categories]
│
├── commands/                  # All 224 commands
│   ├── README.md                 # Registry
│   ├── essential-commands/       # 10 core commands
│   ├── sparc/                    # 31 SPARC commands
│   ├── agent-commands/           # 18 agent lifecycle
│   ├── audit-commands/           # 29 quality commands
│   ├── github/                   # 21 GitHub commands
│   └── [9+ more categories]
│
├── .claude/                   # Configuration
│   ├── settings.json             # Hooks, permissions
│   ├── sparc-modes.json
│   ├── plugins/
│   └── [other config]
│
├── .mcp.json                  # MCP server config
│   └── 3 servers: claude-flow (required), ruv-swarm, flow-nexus
│
├── hooks/12fa/                # Integration hooks
│   └── memory-mcp-tagging-protocol.js
│
├── CLAUDE.md                  # PROJECT INSTRUCTIONS
│   └── Read for core rules & patterns
│
└── docs/                      # Documentation
    ├── SKILL-AGENT-COMMAND-ARCHITECTURE.md    ← Full reference
    ├── ARCHITECTURE-VISUAL-REFERENCE.md        ← Diagrams
    ├── MASTER-SKILLS-INDEX.md
    ├── MASTER-COMMAND-INDEX.md
    └── [100+ other docs]
```

---

## 8. Common Tasks Quick Reference

### I want to create a new skill

1. Identify trigger conditions (when to use)
2. Define primary agents (who executes)
3. Design 7-12 phases with objectives
4. Document memory patterns (data storage)
5. Create YAML frontmatter in `skills/{name}/SKILL.md`
6. Add to `skills/MASTER-SKILLS-INDEX.md`
7. Register in memory namespace

**Reference**: `skills/agent-creator/SKILL.md` (4-phase methodology)

### I want to create a new agent

1. Define core identity and expertise domains
2. List 5+ capabilities specific to agent
3. Document decision frameworks (how agent decides)
4. Define quality standards (output validation)
5. Create file in `agents/{category}/{agent}.md` with YAML
6. Add to `agents/registry.json`
7. Assign MCP server access (control matrix)

**Reference**: `agents/core/coder.md` (example agent definition)

### I want to create a new command

1. Identify skill binding (which skill executes)
2. Document exact usage syntax and parameters
3. Explain multi-stage workflow
4. Provide 3-5 usage examples
5. Create file in `commands/{category}/{command}.md` with YAML
6. Add to `commands/README.md`
7. Test command → skill → agent flow

**Reference**: `commands/essential-commands/review-pr.md` (example)

### I want to integrate with MCP

1. Update `.mcp.json` to enable server
2. Add MCP tools to agent registry (capabilities)
3. Configure access control (which agents can use)
4. Update memory patterns (data storage keys)
5. Document in skill/agent/command frontmatter
6. Test in isolation first

**Reference**: `.mcp.json` (MCP configuration), `hooks/12fa/memory-mcp-tagging-protocol.js`

### I want to use the architecture

Follow the **GOLDEN RULE**:
```javascript
// Single message spawns ALL agents and operations
[Parallel Execution]:
  Task("Researcher", "Analyze API requirements", "researcher")
  Task("Backend Dev", "Implement endpoints", "coder")
  Task("Database Architect", "Design schema", "coder")
  Task("Test Engineer", "Write tests", "analyst")
  Task("Security Auditor", "Review security", "analyst")
  Task("Reviewer", "Code review", "analyst")
  Task("Coordinator", "Orchestrate workflow", "coordinator")
  
  TodoWrite { todos: [...all todos in ONE call...] }
  
  // File operations batched
  Write "file1.js"
  Write "file2.js"
  Write "file3.js"
```

---

## 9. Important Files to Know

```
CONFIGURATION:
├─ CLAUDE.md                 # Core project rules (READ THIS FIRST)
├─ .claude/settings.json     # Hooks & permissions
├─ .mcp.json                 # MCP servers
└─ agents/registry.json      # Agent definitions

REGISTRIES:
├─ skills/MASTER-SKILLS-INDEX.md       # All 93 skills
├─ commands/README.md                   # All 224 commands
└─ agents/registry.json                 # All 131 agents

ARCHITECTURE DOCS:
├─ docs/SKILL-AGENT-COMMAND-ARCHITECTURE.md      # Full reference
├─ docs/ARCHITECTURE-VISUAL-REFERENCE.md         # Diagrams
├─ docs/ARCHITECTURE-QUICK-REFERENCE.md          # This file
├─ skills/THREE-LOOP-INTEGRATION-ARCHITECTURE.md # Loop details
└─ docs/MASTER-COMMAND-INDEX.md                  # Command details

INTEGRATION:
├─ hooks/12fa/memory-mcp-tagging-protocol.js     # Memory tagging
└─ .claude-flow/                                  # Flow Nexus config
```

---

## 10. Theater Detection - Key Concept

**Problem**: How to verify code actually works (not just looks good)?

**Solution**: 6-Agent Byzantine Consensus

```
6 agents test implementation independently:
1. Sandbox execution test      → ✅ Runs without errors
2. Coverage validation         → ✅ 95% coverage
3. Mock detection             → ✅ No fake implementations
4. Integration test           → ✅ All parts work together
5. Performance benchmark      → ✅ Performance acceptable
6. Security validation        → ✅ No vulnerabilities

CONSENSUS (≥5/6 pass):
✅ REAL IMPLEMENTATION

CONSENSUS (≤4/6 pass):
❌ THEATER DETECTED - Needs investigation
```

**Skill**: `theater-detection-audit`  
**Reference**: `commands/audit-commands/theater-detect.md`

---

## 11. Memory Storage Examples

### Example 1: Store Implementation Decision

```javascript
memory_store(
  key: "swarm/feature-dev/auth-jwt-decision",
  value: {
    decision: "Use JWT with refresh tokens",
    rationale: "Stateless, scalable, industry standard",
    tradeoffs: ["need refresh token rotation", "token expiry handling"],
    alternatives: ["session-based", "OAuth 2.0"],
    rejected: ["session-based due to database dependency"]
  },
  tags: {
    WHO: { agent: "researcher", category: "planning" },
    WHEN: { iso: "2025-11-06T10:30:00Z" },
    PROJECT: "user-authentication",
    WHY: "planning"
  }
)
```

### Example 2: Retrieve Similar Decisions

```javascript
memory_search(
  pattern: "swarm/*/auth*",
  query: "JWT authentication implementation strategy"
)
→ Returns: Top 5-20 semantic matches with ranking
   1. Previous JWT decision (95% match)
   2. OAuth2 vs JWT comparison (87% match)
   3. Token refresh patterns (82% match)
   ...
```

### Example 3: Load Historical Failures

```javascript
memory_search(
  pattern: "integration/loop3-feedback/*",
  query: "async race condition bugs"
)
→ Returns: All documented async failures
   Pattern: race-condition
   Frequency: 3 occurrences
   Root causes: concurrent requests, shared state
   Prevention: mutex locks, request queueing
```

---

## 12. Execution Examples

### Example 1: Build Feature (Simple Flow)

```bash
/build-feature "user authentication with JWT"

Expected Output:
✅ Research: Best practices analyzed
✅ Architecture: Design documented
✅ Implementation: Code written
✅ Tests: 95% coverage
✅ Security: No vulnerabilities
✅ Optimization: Performance validated
✅ Documentation: Complete
✅ PR: Created and ready to merge
```

### Example 2: Review PR (Parallel Execution)

```bash
/review-pr 123

Expected Output:
🤖 Code Review Complete: PR #123
Overall Score: 87/100
  Security: 95/100 ✅
  Performance: 88/100 ✅
  Style: 90/100 ✅
  Tests: 82/100 ✅
  Docs: 85/100 ✅
Merge Ready: ✅ Yes
```

### Example 3: Theater Detection (Consensus)

```bash
/quick-check src/auth.js

Expected Output:
6-Agent Byzantine Consensus:
  Agent 1 (Sandbox):      ✅ Real
  Agent 2 (Coverage):     ✅ 95%
  Agent 3 (Mocks):        ✅ None detected
  Agent 4 (Integration):  ✅ Works
  Agent 5 (Performance):  ✅ Fast
  Agent 6 (Security):     ✅ Secure

CONSENSUS (6/6 pass):
✅ REAL IMPLEMENTATION (100% confidence)
```

---

## 13. Troubleshooting Quick Guide

| Issue | Solution |
|-------|----------|
| Command won't execute | Check skill binding in command YAML |
| Agent seems confused | Verify agent has required MCP server access |
| Memory operation fails | Ensure WHO/WHEN/PROJECT/WHY tags included |
| Agents not coordinating | Check coordination hooks in settings.json |
| Theater detection fails | May indicate real implementation issue |
| Performance problems | Run `/profiler-start` then `/profiler-stop` |
| Test coverage low | Review `functionality-audit` results |
| Slow multi-agent execution | Verify parallel Task tool usage (1 message) |

---

## 14. Key Metrics & Thresholds

| Metric | Threshold | Agent |
|--------|-----------|-------|
| Code Quality | ≥80/100 | code-analyzer |
| Test Coverage | ≥90% | tester |
| Theater Detection | ≥5/6 agents pass | analyst |
| Security Issues | 0 critical | security-tester |
| Performance | Within SLA | optimizer |
| Documentation | ≥90% complete | system-architect |
| Cyclomatic Complexity | <10 | code-analyzer |
| Function Length | <50 lines | code-analyzer |

---

## 15. Recommended Reading Order

1. **Start here**: `CLAUDE.md` (project rules)
2. **Understand architecture**: `docs/SKILL-AGENT-COMMAND-ARCHITECTURE.md`
3. **See visuals**: `docs/ARCHITECTURE-VISUAL-REFERENCE.md`
4. **Find skills**: `skills/MASTER-SKILLS-INDEX.md`
5. **Find commands**: `commands/README.md`
6. **Find agents**: `agents/registry.json`
7. **Understand loops**: `skills/THREE-LOOP-INTEGRATION-ARCHITECTURE.md`
8. **Deep dive**: Specific skill/agent/command docs as needed

---

## Summary

This is a **three-layer architecture** where:

```
COMMANDS (user interface)
    ↓ bind to
SKILLS (methodology)
    ↓ delegate to
AGENTS (expertise)
    ↓ coordinated via
MEMORY + HOOKS (persistence & automation)
```

**Key principles**:
- ✅ Clear separation of concerns
- ✅ 131 specialized agents with distinct expertise
- ✅ 93 reusable skills covering all domains
- ✅ 224 commands for granular control
- ✅ Persistent memory with semantic search
- ✅ Theater detection via Byzantine consensus
- ✅ Three-loop feedback for continuous improvement
- ✅ Parallel execution with adaptive coordination

**Central command**: 
```
GOLDEN RULE: 1 MESSAGE = ALL RELATED OPERATIONS
```

Use Claude Code's Task tool to spawn agents, batch all operations together, and leverage memory for cross-session learning.
