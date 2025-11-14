# Claude Artifacts Documentation Audit - Quick Fixes Checklist
**Date:** 2025-11-09
**Based on:** CLAUDE-ARTIFACTS-DOCUMENTATION-AUDIT-REPORT.md

---

## 🚨 CRITICAL ISSUES (Fix Immediately)

### 1. Memory MCP Documentation Mismatch
**Issue:** CLAUDE.md claims Memory MCP is integrated, but it's NOT in `.mcp.json`

**Location:** CLAUDE.md lines ~680-710

**Options:**
- [ ] **Option A:** Remove Memory MCP references from CLAUDE.md
- [ ] **Option B:** Add Memory MCP server to `.mcp.json`
- [ ] **Option C:** Add clarification that Memory MCP is planned/in-progress

**Affected sections:**
```markdown
### Memory MCP - Persistent Cross-Session Context (PRODUCTION READY)
**Integrated**: 2025-11-01 | **Status**: GLOBAL ACCESS FOR ALL AGENTS
```

**Action:** Choose option and update CLAUDE.md accordingly.

---

### 2. Connascence Analyzer Documentation Mismatch
**Issue:** CLAUDE.md claims Connascence Analyzer is integrated, but it's NOT in `.mcp.json`

**Location:** CLAUDE.md lines ~712-750

**Options:**
- [ ] **Option A:** Remove Connascence Analyzer references from CLAUDE.md
- [ ] **Option B:** Add Connascence Analyzer server to `.mcp.json`
- [ ] **Option C:** Clarify status as "planned" or "local-only"

**Affected sections:**
```markdown
### Connascence Analyzer - Code Quality & Coupling Detection (PRODUCTION READY)
**Integrated**: 2025-11-01 | **Status**: CODE QUALITY AGENTS ONLY (14 agents)
```

---

### 3. Hardcoded Windows Paths
**Issue:** Absolute Windows paths will break on other systems

**Locations:**
- Line 683: `C:\Users\17175\docs\integration-plans\MCP-INTEGRATION-GUIDE.md`
- Line 702: `C:\Users\17175\hooks\12fa\memory-mcp-tagging-protocol.js`

**Fix:**
```diff
- **Documentation**: `C:\Users\17175\docs\integration-plans\MCP-INTEGRATION-GUIDE.md`
+ **Documentation**: `./docs/integration-plans/MCP-INTEGRATION-GUIDE.md`

- **Implementation**: `C:\Users\17175\hooks\12fa\memory-mcp-tagging-protocol.js`
+ **Implementation**: `./hooks/12fa/memory-mcp-tagging-protocol.js`
```

- [ ] Replace hardcoded paths with relative paths
- [ ] Verify referenced files exist

---

### 4. Agent Count Mismatch
**Issue:** CLAUDE.md claims "131 Total Agents" but filesystem has 203+

**Location:** CLAUDE.md line 446

**Current:**
```markdown
## 🚀 Available Agents (131 Total)
```

**Fix:**
```markdown
## 🚀 Available Agents (203+ Total)
```

- [ ] Update agent count
- [ ] Add note about 10 functional categories
- [ ] Consider linking to separate AGENT-REGISTRY.md

---

## ⚠️ HIGH PRIORITY FIXES

### 5. Document 32+ Undocumented Skills

**Missing skill directories:**
1. `advanced-coordination`
2. `agent-creation`
3. `compliance`
4. `infrastructure`
5. `machine-learning`
6. `ml`
7. `observability`
8. `platform-integration`
9. `specialized-workflow`
10. `testing`
11. `utilities`
12. ...and 21+ more

**Action Items:**
- [ ] Audit each skill directory for functionality
- [ ] Create auto-trigger patterns for documented skills
- [ ] Add to CLAUDE.md skill auto-trigger section
- [ ] OR create separate SKILL-REGISTRY.md

---

### 6. Skill Naming Inconsistency

**Current situation:**
| CLAUDE.md | Filesystem |
|-----------|------------|
| `hooks-automation` | `when-automating-workflows-use-hooks-automation` |
| `pair-programming` | `when-collaborative-coding-use-pair-programming` |
| `smart-bug-fix` | `when-fixing-complex-bug-use-smart-bug-fix` |

**Options:**
- [ ] **Option A:** Rename skill directories to short names (recommended)
- [ ] **Option B:** Update CLAUDE.md to use full "when-X-use-Y" names
- [ ] **Option C:** Support both names (symlinks or aliasing)

**Recommendation:** Option A for consistency and ease of use.

---

### 7. MCP Setup Instructions Platform Mismatch

**Issue:** CLAUDE.md gives Unix commands, but config uses Windows batch

**CLAUDE.md (line ~565):**
```bash
claude mcp add claude-flow npx claude-flow@alpha mcp start
```

**Actual .mcp.json:**
```json
"command": "cmd",
"args": ["/c", "npx", "claude-flow@alpha", "mcp", "start"]
```

**Fix:**
- [ ] Add platform-specific installation instructions
- [ ] Provide Windows and Unix/Mac examples
- [ ] OR use generic instructions that work cross-platform

---

### 8. Document Missing MCP Tools

**Configured but undocumented in CLAUDE.md:**
1. `mcp__flow-nexus__seraphina_chat` - Queen Seraphina AI assistant
2. `mcp__ruv-swarm__daa_*` - 10 DAA (Decentralized Autonomous Agent) tools:
   - `daa_init`
   - `daa_agent_create`
   - `daa_agent_adapt`
   - `daa_workflow_create`
   - `daa_workflow_execute`
   - `daa_knowledge_share`
   - `daa_learning_status`
   - `daa_cognitive_pattern`
   - `daa_meta_learning`
   - `daa_performance_metrics`

**Action:**
- [ ] Add Queen Seraphina to Multi-Model & External Tools section
- [ ] Create DAA Tools section or add to Swarm & Coordination
- [ ] Document use cases and examples

---

## 📋 MEDIUM PRIORITY FIXES

### 9. Move Support Files from Agents Directory

**Files to move from `.claude/agents/` to `docs/`:**
- `foundry-registry-MIGRATION_SUMMARY.md`
- `quality-audit-BATCH-5-AUDIT-VALIDATION-AGENTS-SUMMARY.md`
- `operations-infrastructure-INFRASTRUCTURE-AGENTS-SUMMARY.md`
- Any other non-agent files

**Commands:**
```bash
mv .claude/agents/*SUMMARY.md docs/agents/
mv .claude/agents/*README.md docs/agents/
```

- [ ] Identify all support/documentation files
- [ ] Move to appropriate docs/ subdirectory
- [ ] Update any references

---

### 10. Create Agent Registry Document

**Action:** Create `docs/AGENT-REGISTRY.md`

**Contents:**
- Full list of 203+ agents
- Organized by 10 categories
- Agent type, phase, capabilities
- Links to agent files
- Auto-generated from .claude/agents/

**Template:**
```markdown
# Agent Registry

**Total Agents:** 203
**Categories:** 10
**Last Updated:** [Auto-generated date]

## Category: Delivery (18 agents)

### delivery-development-backend-dev-backend-api
- **Type:** coder
- **Phase:** development
- **Capabilities:** REST API, Express, authentication
- **File:** `.claude/agents/delivery-development-backend-dev-backend-api.md`

...
```

- [ ] Create AGENT-REGISTRY.md
- [ ] Auto-generate or maintain manually
- [ ] Link from CLAUDE.md

---

### 11. Create Skill Registry Document

**Action:** Create `docs/SKILL-REGISTRY.md`

**Contents:**
- Full list of 103+ skills
- Categories and subcategories
- Auto-trigger patterns
- Dependencies (which skills call other skills)
- Agent usage (which agents use which skills)

- [ ] Create SKILL-REGISTRY.md
- [ ] List all 103 skill directories
- [ ] Document functionality for each
- [ ] Add cross-references to agents

---

### 12. Fix Skill Auto-Trigger Patterns

**Current issue:** Some skills documented but pattern may not work

**Example:**
```markdown
- `reasoningbank-intelligence` - "learn from mistakes", "adaptive learning"
```

**Filesystem check:**
- ❌ `reasoningbank-intelligence` directory NOT found
- ✅ `reasoningbank-agentdb` directory EXISTS

**Action:**
- [ ] Verify each documented skill exists
- [ ] Test auto-trigger patterns
- [ ] Remove or fix broken patterns

---

## 🔧 AUTOMATION TASKS

### 13. Create Validation Script

**File:** `scripts/validate-documentation.sh`

**Function:**
```bash
#!/bin/bash
# Validate documentation consistency

# Check all documented skills exist
# Check all documented agents exist
# Verify MCP configurations match docs
# Report orphaned files
```

- [ ] Create validation script
- [ ] Run before commits
- [ ] Add to CI/CD pipeline

---

### 14. Create Sync Script

**File:** `scripts/sync-documentation.sh`

**Function:**
```bash
#!/bin/bash
# Auto-sync documentation from artifacts

# Generate SKILL-REGISTRY.md from .claude/skills/
# Generate AGENT-REGISTRY.md from .claude/agents/
# Update MCP tool lists from .mcp.json
# Update counts in CLAUDE.md
```

- [ ] Create sync script
- [ ] Run periodically (weekly?)
- [ ] Document manual review process

---

## 📊 TRACKING PROGRESS

### Completion Checklist

**Critical Issues (4):**
- [ ] Fix Memory MCP documentation
- [ ] Fix Connascence Analyzer documentation
- [ ] Replace hardcoded Windows paths
- [ ] Update agent count (131 → 203+)

**High Priority (8):**
- [ ] Document 32+ undocumented skills
- [ ] Fix skill naming inconsistency
- [ ] Add platform-specific MCP setup instructions
- [ ] Document Queen Seraphina tool
- [ ] Document 10 DAA tools
- [ ] Move support files from agents/ to docs/
- [ ] Create AGENT-REGISTRY.md
- [ ] Create SKILL-REGISTRY.md

**Medium Priority (2):**
- [ ] Fix broken skill auto-trigger patterns
- [ ] Create cross-reference documentation

**Automation (2):**
- [ ] Create validation script
- [ ] Create sync script

**Total Tasks:** 16
**Completed:** 0 / 16 (0%)

---

## 🎯 RECOMMENDED ORDER

1. **Critical Issues First** (30 minutes)
   - Update agent count
   - Fix hardcoded paths
   - Add clarification notes for Memory MCP and Connascence Analyzer

2. **Quick Wins** (1 hour)
   - Move support files
   - Document Queen Seraphina
   - Add platform-specific MCP instructions

3. **Documentation Creation** (2-3 hours)
   - Create AGENT-REGISTRY.md
   - Create SKILL-REGISTRY.md
   - Document DAA tools

4. **Skill Reconciliation** (4-6 hours)
   - Audit 32+ undocumented skills
   - Add to SKILL-REGISTRY.md
   - Create auto-trigger patterns

5. **Automation** (2-3 hours)
   - Create validation script
   - Create sync script
   - Test and integrate

**Total estimated time:** 10-14 hours for complete fixes

---

## 📝 NOTES

- This checklist generated from comprehensive audit report
- See `CLAUDE-ARTIFACTS-DOCUMENTATION-AUDIT-REPORT.md` for detailed findings
- Update this checklist as items are completed
- Consider creating GitHub issues for tracking

**Next Review:** After completing Critical + High Priority fixes
