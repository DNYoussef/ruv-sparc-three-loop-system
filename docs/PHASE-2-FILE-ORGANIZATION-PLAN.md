# Phase 2: File Organization Plan

**Generated**: 2025-11-02
**Status**: ✅ COMPLETE

---

## File Movement Plan

### A. Agent Files (29 loose files → organized directories)

#### 1. Agent Definitions (16 .md files + backups)

**Move to agents/core/** (base functionality):
```bash
mv agents/base-template-generator.md agents/core/
mv agents/base-template-generator.md.backup agents/core/
```

**Move to agents/analysis/** (analytical agents):
```bash
mv agents/root-cause-analyzer.md agents/analysis/
mv agents/root-cause-analyzer.md.backup agents/analysis/
mv agents/root-cause-analyzer-config.json agents/analysis/
mv agents/audit-pipeline-orchestrator.md agents/analysis/
mv agents/audit-pipeline-orchestrator.md.backup agents/analysis/
```

**Move to agents/flow-nexus/** (Gemini/Codex integration):
```bash
mv agents/gemini-search-agent.md agents/flow-nexus/
mv agents/gemini-search-agent.md.backup agents/flow-nexus/
mv agents/gemini-media-agent.md agents/flow-nexus/
mv agents/gemini-media-agent.md.backup agents/flow-nexus/
mv agents/gemini-megacontext-agent.md agents/flow-nexus/
mv agents/gemini-megacontext-agent.md.backup agents/flow-nexus/
mv agents/gemini-extensions-agent.md agents/flow-nexus/
mv agents/gemini-extensions-agent.md.backup agents/flow-nexus/
mv agents/codex-auto-agent.md agents/flow-nexus/
mv agents/codex-auto-agent.md.backup agents/flow-nexus/
mv agents/codex-reasoning-agent.md agents/flow-nexus/
mv agents/codex-reasoning-agent.md.backup agents/flow-nexus/
mv agents/multi-model-orchestrator.md agents/flow-nexus/
mv agents/multi-model-orchestrator.md.backup agents/flow-nexus/
```

#### 2. Registry & Scripts

**Create agents/registry/ directory**:
```bash
mkdir -p agents/registry
```

**Move to agents/registry/**:
```bash
mv agents/registry.json agents/registry/
mv agents/registry.json.backup agents/registry/
mv agents/add-mcp-to-registry.js agents/registry/
mv agents/remove-firebase.js agents/registry/
mv agents/update-mcp-free-only.js agents/registry/
mv agents/update-to-installed-only.js agents/registry/
mv agents/MIGRATION_SUMMARY.md agents/registry/
```

**Keep agents/README.md in root** (explains structure)

#### 3. Summary: Agent File Movements

| Source | Destination | Files |
|--------|-------------|-------|
| agents/ root | agents/core/ | 2 (base-template-generator) |
| agents/ root | agents/analysis/ | 5 (root-cause, audit-pipeline) |
| agents/ root | agents/flow-nexus/ | 14 (Gemini, Codex, multi-model) |
| agents/ root | agents/registry/ | 7 (registry, scripts, migration) |
| **Total** | **4 directories** | **28 files moved, 1 kept** |

### B. Skills Files (10 audit JSON files → _pipeline-automation/audits/)

```bash
mkdir -p skills/_pipeline-automation/audits

mv skills/advanced-coordination-audit.json skills/_pipeline-automation/audits/
mv skills/agent-creation-audit.json skills/_pipeline-automation/audits/
mv skills/agent-creator-audit.json skills/_pipeline-automation/audits/
mv skills/agentdb-advanced-audit.json skills/_pipeline-automation/audits/
mv skills/agentdb-audit.json skills/_pipeline-automation/audits/
mv skills/agentdb-learning-audit.json skills/_pipeline-automation/audits/
mv skills/agentdb-memory-patterns-audit.json skills/_pipeline-automation/audits/
mv skills/agentdb-optimization-audit.json skills/_pipeline-automation/audits/
mv skills/agentdb-vector-search-audit.json skills/_pipeline-automation/audits/
mv skills/baseline-replication-audit.json skills/_pipeline-automation/audits/
```

**Summary**: 10 audit JSONs moved to centralized audit directory

---

## Specialist Agent Mapping Table (for CLAUDE.md)

### 1. Specialist Agent Type Reference

| Type | Purpose | Best For | Examples |
|------|---------|----------|----------|
| **researcher** | Analysis & Investigation | Requirements gathering, SOTA research, pattern analysis | "Analyze API requirements", "Research best practices", "Investigate root cause" |
| **coder** | Implementation & Development | Feature development, bug fixes, refactoring | "Implement REST API", "Fix authentication bug", "Refactor database layer" |
| **analyst** | Testing & Quality Assurance | Code review, testing, quality audits | "Review security vulnerabilities", "Analyze test coverage", "Audit code quality" |
| **optimizer** | Performance & Optimization | Performance tuning, resource optimization | "Optimize database queries", "Reduce memory usage", "Improve API latency" |
| **coordinator** | Multi-Agent Orchestration | Workflow coordination, task delegation | "Orchestrate 5-agent feature development", "Coordinate testing pipeline" |

### 2. Task-to-Agent Mapping

| Task Category | Specialist Agent | Typical Workflow |
|---------------|------------------|------------------|
| **Requirements Analysis** | researcher | Gather requirements → Analyze patterns → Document findings |
| **API Development** | coder | Design endpoints → Implement handlers → Write tests |
| **Database Design** | coder + analyst | Design schema → Implement migrations → Review indexes |
| **Security Audit** | analyst | Scan vulnerabilities → Review code → Generate report |
| **Performance Tuning** | optimizer | Profile bottlenecks → Optimize code → Benchmark improvements |
| **Full Feature** | coordinator | Delegate to researcher/coder/analyst/optimizer → Integrate |

### 3. Agent Selection Decision Tree

```
User Request Analysis
│
├─ "Analyze", "Research", "Investigate" → researcher
├─ "Build", "Implement", "Create", "Fix" → coder
├─ "Review", "Test", "Audit", "Validate" → analyst
├─ "Optimize", "Improve", "Tune", "Benchmark" → optimizer
└─ "Orchestrate", "Coordinate", "Manage", "Delegate" → coordinator
```

### 4. CLAUDE.md Update Template

```markdown
## 🎯 CRITICAL: Specialist Agent Selection

**ALWAYS use specialist agent types, NOT "general-purpose":**

### Specialist Agent Types

| Type | Use When | Example Task |
|------|----------|--------------|
| `researcher` | Analyzing requirements, researching patterns, investigating issues | "Analyze API requirements and best practices for authentication" |
| `coder` | Building features, fixing bugs, refactoring code | "Implement REST endpoints with JWT authentication" |
| `analyst` | Reviewing code, testing, quality audits | "Review security vulnerabilities and test coverage" |
| `optimizer` | Tuning performance, optimizing resources, benchmarking | "Optimize database queries and reduce API latency" |
| `coordinator` | Coordinating workflows, delegating tasks, managing swarms | "Orchestrate 5-agent feature development pipeline" |

### ✅ CORRECT Examples

```javascript
// Research phase
Task("Requirements analyst", "Analyze API requirements and authentication best practices. Research OAuth 2.0 vs JWT.", "researcher")

// Implementation phase
Task("Backend developer", "Implement REST endpoints with JWT authentication using Express.js.", "coder")
Task("Database architect", "Design PostgreSQL schema for user authentication and sessions.", "coder")

// Testing phase
Task("Security auditor", "Review authentication implementation for vulnerabilities.", "analyst")
Task("Test engineer", "Create comprehensive test suite with 90% coverage.", "analyst")

// Optimization phase
Task("Performance engineer", "Optimize API response times and database queries.", "optimizer")

// Coordination phase
Task("Feature coordinator", "Orchestrate full authentication feature development across 5 agents.", "coordinator")
```

### ❌ WRONG Examples

```javascript
Task("Research agent", "...", "general-purpose")  // ❌ Use "researcher"
Task("Coder agent", "...", "general")             // ❌ Use "coder"
Task("Tester agent", "...", "developer")          // ❌ Use "analyst"
```

### Agent Selection Guide

**Question to ask**: "What is the PRIMARY purpose of this task?"

- **Understand/Investigate** → `researcher`
- **Build/Create/Fix** → `coder`
- **Test/Review/Audit** → `analyst`
- **Optimize/Improve** → `optimizer`
- **Coordinate/Orchestrate** → `coordinator`

**Rule**: Every agent spawned via Task tool MUST use one of these 5 specialist types.
```

---

## Script Path Updates

### Scripts Affected by File Movements

**agents/registry/add-mcp-to-registry.js**:
- Change: `./registry.json` → `./agents/registry/registry.json` (if called from root)
- OR keep paths relative to script location (no change needed)

**agents/registry/update-mcp-free-only.js**:
- Same as above

**agents/registry/update-to-installed-only.js**:
- Same as above

**skills/_pipeline-automation/audit-skill.py**:
- Add ACCEPTABLE_PROJECT_DIRS whitelist:
```python
ACCEPTABLE_PROJECT_DIRS = {
    'docs', 'scripts', 'resources', 'templates',
    'integrations', 'patterns', 'wcag-accessibility',
    'aws-specialist', 'kubernetes-specialist', 'gcp-specialist',
    'python-specialist', 'typescript-specialist', 'react-specialist',
    'sql-database-specialist', 'docker-containerization',
    'terraform-iac', 'opentelemetry-observability'
}
```

### Testing After Reorganization

**Test 1: Agent Registry Scripts**
```bash
cd agents/registry
node add-mcp-to-registry.js --test
node update-mcp-free-only.js --test
node update-to-installed-only.js --test
```

**Test 2: Skill Enhancement Pipeline**
```bash
cd skills/_pipeline-automation
python enhance-skill.py ../test-skill --tier Bronze
python audit-skill.py ../test-skill
python cleanup-skill.py ../test-skill
```

---

## Phase 2 Deliverables

✅ **File Movement Plan**: 29 agent files + 10 audit JSONs mapped to destinations
✅ **Specialist Agent Mapping Table**: 5 agent types with decision tree
✅ **CLAUDE.md Update Template**: Ready to integrate
✅ **Script Update Plan**: Registry scripts + audit whitelist
✅ **Testing Plan**: Verification steps after reorganization

---

## Next Step: Phase 3 Cleanup

Execute file movements:
1. Create agents/registry/ directory
2. Move 28 agent files to 4 destinations
3. Move 10 audit JSONs to _pipeline-automation/audits/
4. Update audit-skill.py whitelist
5. Test all scripts still work
