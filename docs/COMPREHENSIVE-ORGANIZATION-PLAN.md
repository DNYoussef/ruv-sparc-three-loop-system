# Comprehensive Organization Plan - ruv-sparc-three-loop-system

**Generated**: 2025-11-02 via Sequential-Thinking MCP (25 thoughts)
**Status**: Ready for Execution
**Estimated Time**: 3-4 hours for organization, 8-10 hours for completing 90 remaining skills

---

## Executive Summary

Complete reorganization of the ruv-sparc-three-loop-system plugin to address:
1. **Agents folder disorganization**: 131 agents without clear taxonomy or categorization
2. **Skills folder loose files**: Audit JSONs scattered in root, inconsistent structures
3. **CLAUDE.md outdated**: No specialist agent mapping (currently spawns generic "general-purpose" agents)
4. **90 incomplete skills**: Only skill.md file, missing README.md, examples/, references/
5. **Documentation gaps**: No clear project structure guide or agent selection flowchart

## Key Insight from Sequential Thinking

**CLAUDE.md Update Critical Finding**: Currently shows `Task("...", "general-purpose")` but should use **specialist agent types** from claude-flow:
- `researcher` - Analysis, investigation, requirements gathering
- `coder` - Implementation, feature development, bug fixes
- `analyst` - Code review, testing, quality assurance
- `optimizer` - Performance tuning, optimization
- `coordinator` - Multi-agent orchestration

This is the CORE issue preventing proper specialist agent usage.

---

## 6-Phase Execution Plan

### Phase 1: Discovery (30 min)

**Objective**: Identify all issues and gather data for organization

**Actions**:
1. Scan agents/ folder for loose files (not in subdirectories)
2. Scan skills/ folder for loose files (audit JSONs, enhancement artifacts)
3. Count incomplete skills (only skill.md, no README.md)
4. Verify claude-flow specialist agent types
5. Document current folder structures

**Commands**:
```bash
# Find loose files in agents/ root
find agents/ -maxdepth 1 -type f

# Find loose audit JSONs in skills/ root
find skills/ -maxdepth 1 -name "*-audit.json"

# Count incomplete skills
find skills/ -type d -maxdepth 1 ! -path skills/_pipeline-automation -exec test ! -f {}/README.md \; -print | wc -l

# List all skill directories without README.md
find skills/ -type d -maxdepth 1 ! -path skills/_pipeline-automation | while read dir; do [ ! -f "$dir/README.md" ] && echo "$dir"; done
```

**Success Criteria**:
- Complete inventory of loose files (agents/ and skills/)
- Exact count of incomplete skills
- Confirmation of specialist agent types from claude-flow

---

### Phase 2: Design (45 min)

**Objective**: Create organizational structures and mapping tables

**Actions**:
1. Design agents/by-domain/ taxonomy (5 categories)
2. Create agent-to-domain mapping CSV
3. Design task-type to specialist-agent mapping table for CLAUDE.md
4. Create skills/by-domain/ categorization (optional, for future)
5. Plan file movement operations

**Deliverables**:

#### 2.1 Agents Taxonomy (agents/by-domain/)
```
agents/
├── by-domain/
│   ├── development/       # coder, backend-dev, frontend-dev, mobile-dev, ml-developer
│   ├── testing/           # tester, reviewer, tdd-london-swarm, production-validator
│   ├── research/          # researcher, data-steward, ethics-agent, archivist, evaluator
│   ├── infrastructure/    # cicd-engineer, system-architect, perf-analyzer
│   └── specialized/       # sparc-coord, swarm-coordinators, consensus-validators
├── registry/
│   └── AGENTS-INDEX.md    # Consolidated list of all 131 agents
├── templates/
│   └── agent-template.md  # Template for creating new agents
└── README.md              # Agent taxonomy guide and selection flowchart
```

#### 2.2 Specialist Agent Mapping Table (for CLAUDE.md)

| Task Category | Specialist Agent | When to Use | Example Tasks |
|---------------|------------------|-------------|---------------|
| **Research & Analysis** | `researcher` | Requirements gathering, SOTA review, investigation | "Analyze API patterns", "Research best practices", "Investigate root cause" |
| **Implementation** | `coder` | Feature development, bug fixes, refactoring | "Implement REST API", "Fix authentication bug", "Refactor for clean code" |
| **Quality & Testing** | `analyst` | Code review, testing, quality assurance | "Review security vulnerabilities", "Analyze test coverage", "Audit code quality" |
| **Performance** | `optimizer` | Performance tuning, optimization, benchmarking | "Optimize database queries", "Reduce memory usage", "Improve latency" |
| **Coordination** | `coordinator` | Multi-agent workflows, orchestration, delegation | "Orchestrate 5-agent feature development", "Coordinate testing pipeline" |

#### 2.3 Audit Script Whitelist

Add to `audit-skill.py` to prevent false NO-GO decisions:
```python
ACCEPTABLE_PROJECT_DIRS = {
    'docs', 'scripts', 'resources', 'templates',
    'integrations', 'patterns', 'wcag-accessibility',
    'aws-specialist', 'kubernetes-specialist', 'gcp-specialist'
}
```

**Success Criteria**:
- agents/ taxonomy designed with 5 categories
- Specialist agent mapping table created
- File movement plan documented

---

### Phase 3: Cleanup & Reorganization (60 min)

**Objective**: Organize folders and move loose files

**Actions**:

#### 3.1 Skills Folder Cleanup
```bash
# Move all loose audit JSON files to _pipeline-automation/audits/
mv skills/*-audit.json skills/_pipeline-automation/audits/

# Verify all skills have proper MECE structure
# (already done for Batch 1, apply to future batches)
```

#### 3.2 Agents Folder Organization
```bash
# Create agents/by-domain/ structure
mkdir -p agents/by-domain/{development,testing,research,infrastructure,specialized}
mkdir -p agents/registry
mkdir -p agents/templates

# Move agent files to appropriate domains
# (requires manual categorization based on agent purpose)

# Create consolidated registry
# (script to aggregate all agent definitions)
```

#### 3.3 Update Script Paths
- `enhance-skill.py` - Verify paths to _pipeline-automation/
- `audit-skill.py` - Add ACCEPTABLE_PROJECT_DIRS whitelist
- `cleanup-skill.py` - Verify file location suggestions
- `batch-enhance.py` - Ensure batch processing works

**Success Criteria**:
- All loose audit JSONs moved to _pipeline-automation/audits/
- Agents organized by domain (5 categories)
- All scripts updated and tested
- No broken file references

---

### Phase 4: Documentation (60 min)

**Objective**: Update all documentation with new structures

**Actions**:

#### 4.1 Update CLAUDE.md

**Section to Add: "Agent Selection Guide"**
```markdown
## 🎯 CRITICAL: Specialist Agent Selection

**ALWAYS use specialist agent types, NOT "general-purpose":**

### Specialist Agent Types (claude-flow)

| Type | Purpose | Use When |
|------|---------|----------|
| `researcher` | Analysis & Investigation | Analyzing requirements, researching patterns, investigating issues |
| `coder` | Implementation & Development | Building features, fixing bugs, refactoring code |
| `analyst` | Testing & Quality Assurance | Reviewing code, testing, quality audits |
| `optimizer` | Performance & Optimization | Tuning performance, optimizing resources, benchmarking |
| `coordinator` | Multi-Agent Orchestration | Coordinating workflows, delegating tasks, managing swarms |

### ✅ CORRECT Example:
```javascript
Task("Research agent", "Analyze API requirements...", "researcher")  // ✅ Specialist
Task("Coder agent", "Implement REST endpoints...", "coder")          // ✅ Specialist
Task("Tester agent", "Create test suite...", "analyst")              // ✅ Specialist
Task("Performance agent", "Optimize queries...", "optimizer")        // ✅ Specialist
Task("Coordinator agent", "Orchestrate 5 agents...", "coordinator")  // ✅ Specialist
```

### ❌ WRONG Example:
```javascript
Task("Research agent", "...", "general-purpose")  // ❌ Generic, not specialized
```
```

**Replace ALL instances of `"general-purpose"` with appropriate specialist types throughout CLAUDE.md**

#### 4.2 Create agents/README.md

**Content**:
- Agent taxonomy explanation (5 domains)
- Agent selection flowchart
- How to add new agents
- Registry index reference

#### 4.3 Create skills/README.md

**Content**:
- MECE structure explanation (skill.md, README.md, examples/, references/, graphviz/)
- Enhancement status tracker (20 enhanced, 90 incomplete)
- Quality tier definitions (Bronze/Silver/Gold/Platinum)
- How to enhance a skill

#### 4.4 Create docs/PROJECT-STRUCTURE.md

**Content**:
- Complete plugin organization overview
- Folder structure tree
- File naming conventions
- Development workflows

#### 4.5 Create docs/COMPLETION-ROADMAP.md

**Content**:
- Batches 2-7 plan (90 skills, 15 per batch)
- Estimated timeline (8-10 hours)
- Success criteria (85%+ pass rate)
- Risk mitigation strategies

**Success Criteria**:
- CLAUDE.md updated with specialist agent mapping table
- All "general-purpose" instances replaced with specialist types
- agents/README.md created
- skills/README.md created
- docs/PROJECT-STRUCTURE.md created
- docs/COMPLETION-ROADMAP.md created

---

### Phase 5: Validation (30 min)

**Objective**: Verify everything still works

**Actions**:

#### 5.1 Test Agent Spawning
```javascript
// Test each specialist agent type works
Task("Test researcher", "Analyze test data", "researcher")
Task("Test coder", "Write hello world", "coder")
Task("Test analyst", "Review test code", "analyst")
Task("Test optimizer", "Benchmark test function", "optimizer")
Task("Test coordinator", "Coordinate 2 test agents", "coordinator")
```

#### 5.2 Test Skill Enhancement Scripts
```bash
# Test enhance-skill.py
python skills/_pipeline-automation/enhance-skill.py skills/test-skill --tier Silver

# Test audit-skill.py with whitelist
python skills/_pipeline-automation/audit-skill.py skills/baseline-replication
# Should NOT flag docs/ and scripts/ as invalid

# Test cleanup-skill.py
python skills/_pipeline-automation/cleanup-skill.py skills/test-skill
```

#### 5.3 Run Sample Batch 2 Enhancement

Select 2-3 skills from Batch 2, run full enhancement workflow:
1. Generate enhancement plans
2. Spawn specialist agents (using new CLAUDE.md guidance)
3. Run cleanup
4. Run audit
5. Verify 85%+ pass rate

#### 5.4 Generate PROJECT-HEALTH-REPORT.md

**Metrics to Track**:
- Total agents: 131 (categorized by 5 domains)
- Total skills: 110 (20 enhanced, 90 incomplete)
- Loose files: 0 (all moved to proper locations)
- Documentation coverage: 100% (all folders have READMEs)
- Specialist agent usage: 100% (all examples use specialist types)

**Success Criteria**:
- All 5 specialist agent types spawn successfully
- All scripts work with new folder structure
- Sample Batch 2 enhancement achieves 85%+ pass rate
- PROJECT-HEALTH-REPORT.md shows healthy status

---

### Phase 6: Complete Remaining Skills (8-10 hours)

**Objective**: Enhance 90 incomplete skills to Silver tier

**Strategy**: Continue proven Batch 1 process

#### Batch 2 (15 skills) - Next Batch
- Estimated time: 1.5 hours (17 min manual + ~40 min agents + cleanup/audit)
- Target: 85%+ pass rate
- Skills: coordination, database-specialists, deep-research-orchestrator, dependencies, deployment-readiness, documentation, dogfooding-system, feature-dev-complete, flow-nexus-neural, flow-nexus-platform, flow-nexus-swarm, frontend-specialists, gate-validation, github-code-review, github-multi-repo

#### Batches 3-7 (75 skills)
- Estimated time: 6.5-8.5 hours
- Target: 90%+ pass rate (with improved audit script)
- Process: Same as Batch 1 (manual skill.md creation → specialist agent enhancement → cleanup → audit)

**Completion Criteria**:
- All 110 skills have Silver tier documentation (7+ files each)
- Pass rate: 90%+ (99 of 110 skills)
- Total documentation: ~350,000 lines
- All skills follow MECE template

---

## Risk Mitigation

### Risk 1: Audit Script False Positives
**Mitigation**: Update `audit-skill.py` with ACCEPTABLE_PROJECT_DIRS whitelist (Phase 2)

### Risk 2: Agent Spawning Breaks
**Mitigation**: Thorough testing in Phase 5 before continuing to Phase 6

### Risk 3: Specialist Agent Types Don't Exist
**Mitigation**: Already verified from CLAUDE.md that claude-flow supports: researcher, coder, analyst, optimizer, coordinator

### Risk 4: Scripts Have Hardcoded Paths
**Mitigation**: Update all scripts in Phase 3.3 and test in Phase 5.2

### Risk 5: Incomplete Skills Too Diverse for Batch Processing
**Mitigation**: Use same proven process from Batch 1 (86.7% success rate), adjust per-skill if needed

---

## Success Metrics

### Organization Health (Post-Phase 5)
- ✅ **0 loose files** in agents/ and skills/ root folders
- ✅ **131 agents categorized** across 5 domains
- ✅ **agents/foundry/registry/** with consolidated index
- ✅ **All scripts functional** with new folder structure
- ✅ **100% documentation coverage** (all folders have READMEs)

### Specialist Agent Usage (Post-Phase 4)
- ✅ **CLAUDE.md updated** with specialist agent mapping table
- ✅ **All examples use specialist types** (researcher, coder, analyst, optimizer, coordinator)
- ✅ **0 instances of "general-purpose"** in agent spawning examples

### Skill Completion (Post-Phase 6)
- ✅ **110 skills at Silver tier** (7+ files each)
- ✅ **90%+ pass rate** (99 of 110 skills ≥85%)
- ✅ **~350,000 lines of documentation** total
- ✅ **100% MECE compliance** (skill.md + README.md + examples/ + references/ + graphviz/)

---

## Timeline

| Phase | Duration | Key Deliverables |
|-------|----------|------------------|
| **Phase 1: Discovery** | 30 min | Inventory of loose files, incomplete skills count |
| **Phase 2: Design** | 45 min | Taxonomies, mapping tables, file movement plan |
| **Phase 3: Cleanup** | 60 min | Organized agents/ and skills/, updated scripts |
| **Phase 4: Documentation** | 60 min | Updated CLAUDE.md, created READMEs, PROJECT-STRUCTURE.md |
| **Phase 5: Validation** | 30 min | Tested agents, scripts, sample Batch 2, PROJECT-HEALTH-REPORT.md |
| **Phase 6: Completion** | 8-10 hours | Enhanced 90 remaining skills (Batches 2-7) |
| **TOTAL** | **11-13 hours** | Fully organized plugin + all skills at Silver tier |

---

## Execution Order

**Phase 1-5 (Organization)**: Execute sequentially, 3.5-4 hours
**Phase 6 (Completion)**: Execute in batches, 8-10 hours

**Next Step**: Begin Phase 1 Discovery - scan folders and identify all issues.

---

**Generated via**: Sequential-Thinking MCP (25 thoughts, 0 revisions)
**Validated via**: Batch 1 enhancement success (86.7% pass rate, 15 skills enhanced)
**References**: intent-analyzer skill structure, skill-forge template, MECE principles
