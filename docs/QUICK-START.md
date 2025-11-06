# Quick Start Guide - ruv-sparc-three-loop-system Plugin

**Version**: 2.0.0
**Status**: Production Ready ✅
**Test Report**: [PLUGIN-TEST-REPORT.md](./PLUGIN-TEST-REPORT.md)

---

## 🧠 How It Works: Automatic Skill Routing

**Just describe what you want in natural language. The system automatically selects the right skills and agents.**

### The Auto-Routing Flow

```
Your Request (natural language)
    ↓
🔍 intent-analyzer (Phase 0)
    ├─ Analyzes explicit & implicit goals
    ├─ Detects constraints and context
    └─ Socratic clarification if ambiguous
    ↓
🎯 orchestration-router (Auto-selection)
    ├─ Keyword extraction
    ├─ Decision tree routing (Priority 1-4)
    └─ Skill selection with rationale
    ↓
⚡ Skill Execution (e.g., parallel-swarm-implementation)
    ├─ Spawns specialized agents in parallel
    ├─ Coordinates via memory namespaces
    └─ Theater detection via Byzantine consensus
    ↓
✅ Production-Ready Output
```

**No manual skill selection required. Zero decision paralysis.**

---

## What You Get

This plugin provides **207 AI agents**, **115 skills**, and **233 commands** for comprehensive development automation:

### 🔄 Three-Loop System (Auto-Triggered)
1. **Loop 1**: Research-driven planning with 5x pre-mortem analysis
2. **Loop 2**: Parallel swarm implementation with theater detection
3. **Loop 3**: CI/CD with intelligent failure recovery

### 🎯 Key Features
- **🧠 Intelligent Auto-Routing** - orchestration-router + intent-analyzer
- **📋 Playbook Decision Tree** - Phase 0-6 systematic workflow
- **86+ Agent Registry** with explicit SOPs
- **Evidence-Based Prompting** (Byzantine consensus, Self-consistency)
- **SPARC Methodology** (Specification → Code)
- **META-SKILL Architecture** (Dynamic agent+skill compilation)
- **Theater Detection** (0% tolerance for fake implementations)
- **Multi-Agent Coordination** (Hierarchical, Mesh, Adaptive topologies)

---

## Installation

### 1. Install Required MCP Server

```bash
# Required: Claude Flow (core coordination)
claude mcp add claude-flow npx claude-flow@alpha mcp start
```

### 2. Install Optional MCP Servers

```bash
# Optional: Enhanced coordination
claude mcp add ruv-swarm npx ruv-swarm mcp start

# Optional: Cloud features (70+ tools)
claude mcp add flow-nexus npx flow-nexus@latest mcp start
```

### 3. Verify Installation

```bash
# Check MCP servers are running
claude mcp list

# Should show:
# - claude-flow (REQUIRED)
# - ruv-swarm (optional)
# - flow-nexus (optional)
```

### 4. Load Plugin

1. Clone or download this plugin repository
2. Claude Code will auto-discover:
   - 207 agents from `.claude/agents/*.md`
   - 121 skills from `skills/**/SKILL.md`
   - 233 commands from `commands/**/*.md`

---

## Quick Start Examples: Natural Language → Auto-Execution

### Example 1: Build a Feature (Complete Workflow)

**Just say what you want:**

```
User: "Build a user authentication system with JWT tokens"

🔍 intent-analyzer (auto-triggered):
   - Detects: Feature implementation intent (95% confidence)
   - Context: Authentication, security-critical, production-ready

🎯 orchestration-router (auto-triggered):
   - Routes to: parallel-swarm-implementation (Loop 2)
   - Rationale: "Build" keyword + multi-component feature

⚡ AUTO-EXECUTION:

1. Loop 1: research-driven-planning (auto-triggered before Loop 2)
   - Gemini search: Authentication best practices
   - 5x pre-mortem risk analysis
   - Validated plan: JWT vs OAuth vs Basic Auth → JWT selected

2. Loop 2: parallel-swarm-implementation (6 agents spawn in parallel)
   - researcher: Authentication patterns (OWASP, NIST guidelines)
   - coder: JWT implementation (access + refresh tokens)
   - reviewer: Security audit (XSS, CSRF, token leakage)
   - tester: Comprehensive tests (unit, integration, security)
   - documenter: API documentation (OpenAPI spec)
   - theater-detector: Byzantine consensus validation (0% fake code)

3. Loop 3: cicd-intelligent-recovery (auto-triggered after Loop 2)
   - Run tests: 45 tests
   - Test failure: "Token expiry not handled"
   - Auto-fix: Add token refresh logic
   - Re-run: All 45 tests pass ✅
   - Deploy: Staging → Production

✅ Result: Production-ready auth system in 2 hours
```

### Example 2: Interactive Planning

```
User: "I need help planning a new e-commerce checkout flow"

Claude Code will:
1. Auto-trigger interactive-planner skill
2. Ask 20-30 structured questions via AskUserQuestion tool:
   - Payment providers?
   - Security requirements?
   - User flow preferences?
   - Testing strategy?
3. Generate comprehensive specification
4. Ready for Loop 2 implementation
```

### Example 3: Code Validation

```
User: "Validate this authentication code actually works"

Claude Code will:
1. Auto-trigger functionality-audit skill
2. Create sandbox environment
3. Execute code with realistic inputs
4. Verify outputs match expectations
5. Identify bugs through systematic analysis
6. Fix issues without breaking functionality
7. Report: "✅ Code validated and working correctly"
```

### Example 4: Parallel Agent Coordination

```
User: "Implement REST API with PostgreSQL and React frontend"

Claude Code spawns agents via Task tool:
Task("Backend developer", "Build Express API with JWT auth...", "coder")
Task("Database architect", "Design PostgreSQL schema...", "code-analyzer")
Task("Frontend developer", "Create React UI with auth flows...", "coder")
Task("Test engineer", "Write comprehensive test suite...", "tester")
Task("Security auditor", "Review for OWASP vulnerabilities...", "reviewer")
Task("DevOps engineer", "Setup Docker and CI/CD...", "cicd-engineer")

All agents work in parallel, coordinating via memory namespaces.
```

---

## Agent Categories (207 Total)

### Core Development (8 agents)
- `coder` - Implementation specialist
- `reviewer` - Code quality expert
- `tester` - Testing specialist
- `planner` - Project planning
- `researcher` - Research & analysis
- `api-designer` - API architecture
- `technical-debt-manager` - Debt tracking
- `coder-enhanced` - Advanced implementation

### Swarm Coordination (15 agents)
- `hierarchical-coordinator` - Tree topology
- `mesh-coordinator` - Peer-to-peer
- `adaptive-coordinator` - Dynamic selection
- `queen-coordinator` - Hive mind leader
- And 11 more specialized coordinators...

### Testing & Quality (9 agents)
- `tdd-london-swarm` - London-style TDD
- `production-validator` - Production readiness
- `e2e-testing-specialist` - End-to-end tests
- `performance-testing-agent` - Load testing
- `security-testing-agent` - Security scans
- `chaos-engineering-agent` - Chaos testing
- And 3 more...

### Specialized Development (50+ agents)
- Frontend: React, Vue, CSS, Accessibility, Performance
- Backend: Node.js, Python, Go, Rust, API design
- Database: PostgreSQL, MongoDB, Query optimization
- Cloud: AWS, GCP, Azure, Kubernetes, Terraform
- Security: Penetration testing, Zero trust, Compliance
- ML/AI: Model training, Feature engineering, MLOps

[See full catalog](./AGENT-CATALOG.md)

---

## Skill Categories (115 Total)

### Orchestration & Coordination (9 skills) 🆕
- `orchestration-router` - **NEW**: Auto-selects optimal orchestration skill based on intent
- `swarm-orchestration` - Multi-agent swarms (basic + advanced with research/dev/test patterns)
- `coordination` - Agent coordination (basic + advanced with RAFT/Gossip/Byzantine)
- `parallel-swarm-implementation` - Loop 2 META-SKILL (critical)
- `cascade-orchestrator` - Sequential workflows with Codex
- `hive-mind-advanced` - Queen-led hierarchical coordination
- `flow-nexus-swarm` - Cloud-based orchestration
- `stream-chain` - Sequential data pipelines
- `web-cli-teleport` - Web-CLI bridge

### Development Lifecycle
- `research-driven-planning` - Loop 1 planning
- `parallel-swarm-implementation` - Loop 2 execution
- `cicd-intelligent-recovery` - Loop 3 deployment
- `feature-dev-complete` - End-to-end workflow
- `pair-programming` - Interactive coding

### Code Quality
- `functionality-audit` - Sandbox testing
- `theater-detection-audit` - Byzantine validation
- `production-readiness` - Deployment checks
- `code-review-assistant` - Multi-agent review
- `testing-quality` - TDD framework

### Agent Creation
- `agent-creator` - 4-phase SOP agent creation
- `skill-builder` - YAML frontmatter + SOPs
- `micro-skill-creator` - Atomic skills
- `cascade-orchestrator` - Workflow chains

### Specialized Tools
- `gemini-search` - Web research (real-time)
- `gemini-megacontext` - 2M token processing
- `codex-auto` - Autonomous coding
- `multi-model` - Model routing

[See full skill reference](../CLAUDE.md#skill-auto-trigger-reference)

---

## Trigger Patterns: Context-Aware Auto-Selection

**Skills auto-trigger based on intent keywords, agent count, and complexity signals:**

| Natural Language | Auto-Triggered Skill | Rationale |
|------------------|---------------------|-----------|
| "Plan feature" | `interactive-planner` | "plan" keyword → requirements gathering |
| "Validate code works" | `functionality-audit` | "validate" + "works" → sandbox testing |
| "Build auth system" | `parallel-swarm-implementation` | "build" keyword → Loop 2 implementation |
| "Review PR #123" | `code-review-assistant` | "review PR" → multi-agent code review |
| "Performance issues" | `performance-analysis` | "performance" keyword → bottleneck detection |
| "Search React 19 docs" | `gemini-search` | "search" keyword → Gemini grounded search |
| "Pair code with me" | `pair-programming` | "pair" keyword → interactive TDD |
| "Coordinate 7 agents" | `swarm-orchestration` | 7 agents (≥5) → swarm topology |
| "Byzantine consensus" | `coordination` (advanced) | "Byzantine" → advanced consensus protocols |
| "Cloud deployment" | `flow-nexus-swarm` | "cloud" keyword → Cloud swarm orchestration |

**Key Insight**: You never specify the skill name. The system detects intent and auto-routes.

[See complete trigger reference (115 skills)](../CLAUDE.md#skill-auto-trigger-reference)

---

## Memory Coordination

Agents coordinate via memory namespaces:

```javascript
// Product Manager stores requirements
mcp__claude-flow__memory_store({
  key: "product/product-manager/roadmap-2025",
  value: { themes: [...], quarters: {...} }
})

// Coder retrieves requirements
mcp__claude-flow__memory_retrieve({
  key: "product/product-manager/roadmap-2025"
})

// Namespace pattern: {category}/{agent}/{subcategory}/{item}
```

**Coordination Protocol**:
1. Store outputs in memory
2. Notify downstream agents
3. Provide context via namespace
4. Monitor handoff completion

---

## Workflow Examples

### Example: Full-Stack Feature

```
1. User: "Build user profile page with avatar upload"

2. Loop 1: research-driven-planning
   - Research: Best practices for file uploads
   - Analysis: Security considerations (file types, size limits)
   - Planning: 5x pre-mortem risk analysis
   - Output: Validated implementation plan

3. Loop 2: parallel-swarm-implementation
   Spawns 7 agents concurrently:
   - researcher: Avatar storage patterns (S3 vs local)
   - backend-dev: Upload API endpoint + validation
   - frontend-dev: React component with preview
   - security-auditor: File validation + sanitization
   - tester: Upload tests (size, type, malicious files)
   - documenter: API documentation
   - reviewer: Code quality audit

4. Loop 3: cicd-intelligent-recovery
   - Run tests: 45 tests
   - Test failure: "File type validation missing"
   - Auto-fix: Add MIME type validation
   - Re-run tests: All pass ✅
   - Deploy to staging

5. Result: Production-ready feature in 2 hours
```

### Example: Bug Fix with Root Cause

```
1. User: "Users report authentication randomly fails"

2. Trigger: smart-bug-fix skill
   - Analyze error logs
   - Reproduce bug in sandbox
   - Root cause: Race condition in token refresh
   - Generate fix: Add mutex lock
   - Test fix: 1000 concurrent requests
   - Result: ✅ Bug fixed, no regressions

3. Trigger: production-validator skill
   - Security audit: Pass
   - Performance test: Pass
   - Integration test: Pass
   - Deploy approved: ✅
```

---

## Best Practices

### 1. Use Skills for Complex Tasks
```
❌ Don't: "Write code for authentication"
✅ Do: "Use parallel-swarm-implementation to build authentication"
```

### 2. Leverage Agent Coordination
```
❌ Don't: Work sequentially
✅ Do: Spawn 5-10 agents in parallel via Task tool
```

### 3. Validate Everything
```
❌ Don't: Assume code works
✅ Do: Use functionality-audit for sandbox testing
```

### 4. Follow Three-Loop Pattern
```
Loop 1: Plan → Loop 2: Implement → Loop 3: Deploy
(Research)    (Theater-free)      (Auto-fix failures)
```

### 5. Use Memory for Coordination
```
Store outputs: mcp__claude-flow__memory_store
Retrieve inputs: mcp__claude-flow__memory_retrieve
```

---

## Troubleshooting

### Issue: Skills not auto-triggering

**Solution**: Use skill names explicitly:
```
User: "Use parallel-swarm-implementation skill to build this feature"
```

### Issue: Agent spawning fails

**Solution**: Verify MCP servers running:
```bash
claude mcp list
# Should show claude-flow as REQUIRED
```

### Issue: Memory coordination not working

**Solution**: Check namespace pattern:
```javascript
// ✅ Correct
key: "product/product-manager/roadmap-2025"

// ❌ Wrong
key: "roadmap-2025"  // Missing category/agent
```

### Issue: Tests failing in Loop 3

**Solution**: Let cicd-intelligent-recovery auto-fix:
```
Loop 3 will:
1. Detect failure
2. Run root cause analysis
3. Auto-fix code
4. Re-run tests
5. Repeat until 100% pass
```

---

## Next Steps

1. ✅ Install MCP servers (claude-flow required)
2. ✅ Load plugin in Claude Code
3. 🎯 Try Example 1: "Build user authentication"
4. 📚 Read [Complete Test Report](./PLUGIN-TEST-REPORT.md)
5. 📖 Explore [Agent Catalog](./AGENT-CATALOG.md)
6. 🔧 Review [Workflow Examples](../CLAUDE.md)

---

## Support

- **Documentation**: [../CLAUDE.md](../CLAUDE.md)
- **Test Report**: [PLUGIN-TEST-REPORT.md](./PLUGIN-TEST-REPORT.md)
- **Agent Catalog**: [AGENT-CATALOG.md](./AGENT-CATALOG.md)
- **Repository**: https://github.com/DNYoussef/ruv-sparc-three-loop-system
- **Issues**: Create GitHub issue with detailed description

---

## License

MIT License - See [LICENSE](../LICENSE) for details

---

**Status**: 🚀 **Production Ready** | **Version**: 2.0.0 | **Test Coverage**: 100%
